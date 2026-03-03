# Databricks notebook source
# MAGIC %md
# MAGIC # Operator Cost Coefficient — Empirical Experiment Harness (Personal Compute)
# MAGIC
# MAGIC **Purpose:** Run the controlled query series from the coefficient-transparency-session-brief
# MAGIC on a Databricks Personal Compute cluster, measure wall-clock time and Spark task metrics,
# MAGIC and derive empirical operator cost ratios.
# MAGIC
# MAGIC **Why classic compute (not serverless)?** Serverless bills per-DBU per-query, which
# MAGIC is ideal, but it locks out most Spark configs (including disabling AQE and setting
# MAGIC broadcast thresholds). Classic compute gives full config control and a fixed resource
# MAGIC envelope where wall-clock time is proportional to resource consumption.
# MAGIC
# MAGIC **Cluster-agnostic:** This harness runs on any cluster size -- single-node personal
# MAGIC compute, multi-node standard clusters, etc. Run it on multiple cluster shapes to see
# MAGIC how ratios shift with node count, network topology, and memory per executor. The
# MAGIC harness captures full cluster metadata so results from different runs are comparable.
# MAGIC
# MAGIC **Measurement basis:** Wall-clock time + Spark task-level metrics (executor CPU time,
# MAGIC bytes read, shuffle bytes, etc.) captured via the Spark status tracker.
# MAGIC
# MAGIC **Table sizes:** Default ~10 GB. Increase ROW_COUNT for larger clusters.
# MAGIC produce meaningful ratios while completing in reasonable time.
# MAGIC
# MAGIC **Usage:**
# MAGIC 1. Run §0 to configure (edit catalog/schema/table sizes)
# MAGIC 2. Run §1 to create test tables
# MAGIC 3. Run §2 to execute the full query series with plan validation
# MAGIC 4. Run §3 to compute ratios and compare to current coefficients
# MAGIC 5. Run §4 to export results

# COMMAND ----------

# MAGIC %md
# MAGIC ## §0 — Configuration

# COMMAND ----------

import json
import time
import statistics
import io
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from contextlib import redirect_stdout

# ---------------------------------------------------------------------------
# Experiment configuration — edit these before running
# ---------------------------------------------------------------------------
CATALOG        = "main"               # Unity Catalog catalog (or hive_metastore)
SCHEMA         = "test_weights"       # Schema for experiment tables
RUNS_PER_QUERY = 10                   # More runs than serverless — wall-clock has more variance
WARMUP_RUNS    = 2                    # Discard first N runs (JIT warmup, cache priming)
SETTLE_SECONDS = 3                    # Pause between runs for GC / disk flush
ROW_COUNT      = 50_000_000          # ~50M rows → ~10 GB in Delta with 50 double columns
NUM_DATES      = 30                   # Generate 30 days of partitions (not 365, to keep setup fast)

# Fully qualified names
FQ = lambda table: f"{CATALOG}.{SCHEMA}.{table}"

LARGE_PARQUET = FQ("large_parquet")
SMALL_DELTA   = FQ("small_delta")
LARGE_CSV     = FQ("large_csv")
SORTED_OUTPUT = FQ("sorted_output")

# ---------------------------------------------------------------------------
# Current coefficients from CostCoefficients.scala
# Each operator has TWO weights: Coefficient(standard, photon).
# The harness auto-selects the correct column based on cluster metadata.
# ---------------------------------------------------------------------------
COEFFICIENTS_STANDARD = {
    "ScanColumnar":       {"weight": 0.02, "ratio": 1.0},
    "ScanRowBased":       {"weight": 0.06, "ratio": 3.0},
    "Shuffle":            {"weight": 0.15, "ratio": 7.5},
    "Sort":               {"weight": 0.08, "ratio": 4.0},
    "BroadcastHashJoin":  {"weight": 0.02, "ratio": 1.0},
    "SortMergeJoin":      {"weight": 0.15, "ratio": 7.5},
    "ShuffledHashJoin":   {"weight": 0.10, "ratio": 5.0},
    "CartesianProduct":   {"weight": 1.00, "ratio": 50.0},
    "PySparkUDF":         {"weight": 0.10, "ratio": 5.0},
}

COEFFICIENTS_PHOTON = {
    "ScanColumnar":       {"weight": 0.01, "ratio": 1.0},
    "ScanRowBased":       {"weight": 0.04, "ratio": 4.0},
    "Shuffle":            {"weight": 0.08, "ratio": 8.0},
    "Sort":               {"weight": 0.04, "ratio": 4.0},
    "BroadcastHashJoin":  {"weight": 0.01, "ratio": 1.0},
    "SortMergeJoin":      {"weight": 0.08, "ratio": 8.0},
    "ShuffledHashJoin":   {"weight": 0.06, "ratio": 6.0},
    "CartesianProduct":   {"weight": 0.60, "ratio": 60.0},
    "PySparkUDF":         {"weight": 0.10, "ratio": 10.0},
}

# Auto-selected after cluster metadata capture (S1b). Default: standard.
CURRENT_COEFFICIENTS = COEFFICIENTS_STANDARD

def select_coefficients(metadata):
    global CURRENT_COEFFICIENTS
    photon = str(metadata.get("photon_enabled", "false")).lower()
    if photon in ("true", "1", "yes"):
        CURRENT_COEFFICIENTS = COEFFICIENTS_PHOTON
        label = "PHOTON"
    else:
        CURRENT_COEFFICIENTS = COEFFICIENTS_STANDARD
        label = "STANDARD (non-Photon)"
    print(f"\nCoefficient set: {label}")
    print(f"  (comparing empirical ratios against {label} weights from CostCoefficients.scala)")
    return CURRENT_COEFFICIENTS

print(f"Config: {CATALOG}.{SCHEMA}")
print(f"  ROW_COUNT      = {ROW_COUNT:,}")
print(f"  NUM_DATES      = {NUM_DATES}")
print(f"  RUNS_PER_QUERY = {RUNS_PER_QUERY} (+ {WARMUP_RUNS} warmup)")
print(f"  Target table   ≈ {ROW_COUNT * 50 * 8 / 1e9:.0f} GB raw (Delta compression → ~10 GB)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## §1 — Spark Config Lock-In
# MAGIC
# MAGIC Personal compute gives us full control. Set configs ONCE at the top
# MAGIC so every query runs under identical conditions.

# COMMAND ----------

def lock_spark_config():
    """
    Set Spark configs for deterministic, reproducible experiment runs.
    These are the configs we COULDN'T set on serverless.
    """
    configs = {
        # ── AQE: OFF ──
        # Critical: prevents runtime plan rewrites (SMJ→BHJ, partition coalescing)
        "spark.sql.adaptive.enabled": "false",

        # ── Broadcast threshold: explicit control ──
        # Default is 10MB. We'll override per-query via SET, but start with default.
        "spark.sql.autoBroadcastJoinThreshold": "10485760",  # 10 MB

        # ── Shuffle partitions: fixed ──
        # Default 200 is fine for 10 GB. Fixing it prevents variance from auto-tuning.
        "spark.sql.shuffle.partitions": "200",

        # ── Disable caching / codegen variance ──
        # Keep codegen ON (it's part of real-world cost), but disable query result cache
        "spark.databricks.io.cache.enabled": "false",

        # ── Photon ──
        # Personal compute may or may not have Photon. Log which one we're running.
        # Don't try to set this — it's determined by the cluster config.
    }

    print("Locking Spark configuration:")
    for k, v in configs.items():
        try:
            spark.conf.set(k, v)
            actual = spark.conf.get(k)
            status = "✅" if actual == v else f"⚠️  wanted {v}, got {actual}"
            print(f"  {k} = {actual}  {status}")
        except Exception as e:
            print(f"  ❌ {k}: {e}")

    # Report Photon status
    try:
        photon = spark.conf.get("spark.databricks.photon.enabled", "unknown")
        print(f"\n  Photon enabled: {photon}")
    except Exception:
        print(f"\n  Photon status: unknown (config not readable)")

    # Report cluster resources
    sc = spark.sparkContext
    print(f"\n  Executor count: {sc._jsc.sc().getExecutorMemoryStatus().size()}")
    print(f"  Default parallelism: {sc.defaultParallelism}")

    # Verify AQE is actually off
    aqe = spark.conf.get("spark.sql.adaptive.enabled")
    if aqe.lower() != "false":
        print("\n  🚨 WARNING: AQE is still enabled. Plan shapes may be unreliable.")
        print("     Check cluster Spark config for overrides.")
        return False

    print("\n✅ All configs locked. AQE is OFF.")
    return True


# Run immediately:
aqe_off = lock_spark_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ## S1b - Cluster Metadata Capture
# MAGIC
# MAGIC Records what this harness is running on so results from different
# MAGIC cluster shapes can be compared.

# COMMAND ----------

def capture_cluster_metadata():
    sc = spark.sparkContext
    meta = {
        "capture_timestamp": datetime.now().isoformat(),
        "spark_version": sc.version,
        "default_parallelism": sc.defaultParallelism,
    }
    try:
        status = sc._jsc.sc().getExecutorMemoryStatus()
        meta["total_executors"] = status.size()
        meta["worker_executors"] = max(status.size() - 1, 1)
        meta["executor_memory"] = spark.conf.get("spark.executor.memory", "unknown")
        meta["executor_cores"] = spark.conf.get("spark.executor.cores", "unknown")
        meta["driver_memory"] = spark.conf.get("spark.driver.memory", "unknown")
    except Exception as e:
        meta["executor_topology_error"] = str(e)
    try:
        meta["databricks_runtime"] = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion", "unknown")
    except Exception:
        meta["databricks_runtime"] = "unknown"
    try:
        meta["photon_enabled"] = spark.conf.get("spark.databricks.photon.enabled", "unknown")
    except Exception:
        meta["photon_enabled"] = "unknown"
    for tag in ["clusterNodeType","driverNodeType","clusterWorkers","clusterName","clusterId"]:
        try:
            meta[tag] = spark.conf.get(f"spark.databricks.clusterUsageTags.{tag}", "unknown")
        except Exception:
            meta[tag] = "unknown"
    try:
        meta["external_shuffle_service"] = spark.conf.get("spark.shuffle.service.enabled", "false")
    except Exception:
        pass
    workers = meta.get("worker_executors", 1)
    is_multi = workers > 1
    mode = "MULTI-NODE (real network shuffle)" if is_multi else "SINGLE-NODE (loopback shuffle)"
    print(f"Cluster Metadata:")
    print(f"  Spark:        {meta.get('spark_version')}")
    print(f"  DBR:          {meta.get('databricks_runtime')}")
    print(f"  Photon:       {meta.get('photon_enabled')}")
    print(f"  Executors:    {workers} workers + driver")
    print(f"  Exec memory:  {meta.get('executor_memory')}")
    print(f"  Exec cores:   {meta.get('executor_cores')}")
    print(f"  Parallelism:  {meta.get('default_parallelism')}")
    print(f"  Node type:    {meta.get('clusterNodeType', 'unknown')}")
    print(f"  Cluster:      {meta.get('clusterName', 'unknown')}")
    print(f"  Mode:         {mode}")
    if not is_multi:
        print(f"  Note: shuffle/join ratios are a LOWER BOUND vs multi-node clusters")
    return meta


cluster_metadata = capture_cluster_metadata()
select_coefficients(cluster_metadata)

# COMMAND ----------

# MAGIC %md
# MAGIC ## §2 — Test Data Setup

# COMMAND ----------

def setup_schema():
    """Create the experiment schema if it doesn't exist."""
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {CATALOG}.{SCHEMA}")
    print(f"Schema {CATALOG}.{SCHEMA} ready.")


def setup_large_parquet():
    """Create the ~10 GB Delta table with 50 columns, partitioned by date."""
    extra_cols = ",\n    ".join(
        [f"CAST(rand() * 1000 AS DOUBLE) AS val{i}" for i in range(1, 49)]
    )

    spark.sql(f"DROP TABLE IF EXISTS {LARGE_PARQUET}")

    sql = f"""
    CREATE TABLE {LARGE_PARQUET}
    USING DELTA
    PARTITIONED BY (date)
    AS
    SELECT
        id,
        uuid() AS key,
        {extra_cols},
        dt AS date
    FROM RANGE(0, {ROW_COUNT // NUM_DATES}) AS t(id)
    CROSS JOIN (
        SELECT explode(sequence(
            DATE '2025-01-01',
            DATE '2025-01-{NUM_DATES:02d}',
            INTERVAL 1 DAY
        )) AS dt
    )
    """

    print(f"Creating {LARGE_PARQUET} ...")
    t0 = time.time()
    spark.sql(sql)
    elapsed = time.time() - t0

    # Report actual size
    try:
        desc = spark.sql(f"DESCRIBE DETAIL {LARGE_PARQUET}").collect()[0]
        size_gb = desc["sizeInBytes"] / (1024**3) if desc["sizeInBytes"] else "unknown"
        num_files = desc["numFiles"]
        row_count = spark.sql(f"SELECT COUNT(*) FROM {LARGE_PARQUET}").collect()[0][0]
        print(f"  Created: {size_gb:.1f} GB, {num_files} files, {row_count:,} rows ({elapsed:.0f}s)")
    except Exception as e:
        print(f"  Created (couldn't read details: {e})")

    return size_gb


def setup_small_delta():
    """Create the ~100 MB small-side join table."""
    spark.sql(f"DROP TABLE IF EXISTS {SMALL_DELTA}")

    sql = f"""
    CREATE TABLE {SMALL_DELTA}
    USING DELTA
    AS
    SELECT DISTINCT key, val1 AS lookup_val
    FROM {LARGE_PARQUET}
    LIMIT 1000000
    """
    print(f"Creating {SMALL_DELTA} ...")
    t0 = time.time()
    spark.sql(sql)
    elapsed = time.time() - t0

    try:
        desc = spark.sql(f"DESCRIBE DETAIL {SMALL_DELTA}").collect()[0]
        size_mb = desc["sizeInBytes"] / (1024**2) if desc["sizeInBytes"] else "unknown"
        row_count = spark.sql(f"SELECT COUNT(*) FROM {SMALL_DELTA}").collect()[0][0]
        print(f"  Created: {size_mb:.0f} MB, {row_count:,} rows ({elapsed:.0f}s)")
    except Exception as e:
        print(f"  Created ({e})")


def setup_large_csv():
    """Create a CSV copy of the large table for row-based scan experiments.

    Unity Catalog only allows Delta for managed tables, so we write CSV
    to a DBFS path and register it as an external table (or temp view as fallback).
    """
    csv_path = "dbfs:/tmp/cluster_yield_experiment/large_csv"

    # Clean up previous data
    try:
        dbutils.fs.rm(csv_path, recurse=True)
    except Exception:
        pass

    print(f"Writing CSV to {csv_path} (subset of columns to manage disk) ...")
    t0 = time.time()

    (spark.table(LARGE_PARQUET)
        .select("id", "key", "val1", "val2", "val3", "val4", "val5", "date")
        .write.mode("overwrite")
        .option("header", "true")
        .csv(csv_path))

    elapsed = time.time() - t0
    print(f"  Written ({elapsed:.0f}s)")

    # Register so queries can reference it by name.
    # Try external table first; fall back to temp view if UC blocks it.
    global LARGE_CSV
    try:
        spark.sql(f"DROP TABLE IF EXISTS {LARGE_CSV}")
        spark.sql(f"""
            CREATE TABLE {LARGE_CSV}
            USING CSV
            OPTIONS (header 'true', inferSchema 'true')
            LOCATION '{csv_path}'
        """)
        print(f"  Registered as external table: {LARGE_CSV}")
    except Exception as e:
        print(f"  External table failed ({e})")
        print(f"  Falling back to temp view...")
        spark.read.option("header", "true").option("inferSchema", "true") \
            .csv(csv_path).createOrReplaceTempView("large_csv_view")
        LARGE_CSV = "large_csv_view"
        print(f"  Registered as temp view: {LARGE_CSV}")


def run_full_setup():
    """Run all setup steps."""
    setup_schema()
    setup_large_parquet()
    setup_small_delta()
    setup_large_csv()
    print("\n✅ All test tables created.")

# COMMAND ----------

# Uncomment to run setup:
# run_full_setup()

# COMMAND ----------

# MAGIC %md
# MAGIC ## §3 — Experiment Harness
# MAGIC
# MAGIC ### Measurement Strategy
# MAGIC
# MAGIC On personal compute, we measure:
# MAGIC 1. **Wall-clock time** (primary) — proportional to resource cost on fixed hardware
# MAGIC 2. **Spark task metrics** — bytesRead, shuffleBytesWritten, executorCpuTime, etc.
# MAGIC    captured from the Spark listener after each job completes
# MAGIC 3. **Plan shape validation** — parsed from EXPLAIN FORMATTED (same as serverless harness)
# MAGIC
# MAGIC We use `spark.sparkContext.statusTracker` and the last job's stage metrics
# MAGIC to capture fine-grained task-level data.

# COMMAND ----------

# ---------------------------------------------------------------------------
# Spark metrics capture
# ---------------------------------------------------------------------------

def capture_spark_metrics() -> dict:
    """
    Capture aggregate task metrics from the most recently completed Spark job.

    Strategy 1: Spark UI REST API (most reliable on Databricks).
    Strategy 2: Internal statusStore JVM API (fallback).
    """
    import json
    from urllib.request import urlopen

    empty = {
        "bytes_read": 0, "bytes_written": 0,
        "shuffle_read_bytes": 0, "shuffle_write_bytes": 0,
        "executor_cpu_time_ns": 0, "executor_run_time_ms": 0,
        "result_size": 0,
        "records_read": 0, "records_written": 0,
        "shuffle_records_read": 0, "shuffle_records_written": 0,
    }

    e1 = None

    # ---- Strategy 1: Spark UI REST API ----
    try:
        sc = spark.sparkContext
        ui_url = sc.uiWebUrl
        if not ui_url:
            raise ValueError("uiWebUrl not available")
        app_id = sc.applicationId

        jobs_url = f"{ui_url}/api/v1/applications/{app_id}/jobs"
        with urlopen(jobs_url) as resp:
            jobs = json.loads(resp.read())

        last_job = next((j for j in jobs if j.get("status") == "SUCCEEDED"), None)
        if last_job is None:
            return empty

        stage_ids = last_job.get("stageIds", [])
        metrics = dict(empty)

        for sid in stage_ids:
            try:
                stage_url = f"{ui_url}/api/v1/applications/{app_id}/stages/{sid}"
                with urlopen(stage_url) as resp:
                    attempts = json.loads(resp.read())
                if not attempts:
                    continue
                s = attempts[-1]
                metrics["bytes_read"]              += s.get("inputBytes", 0)
                metrics["records_read"]            += s.get("inputRecords", 0)
                metrics["bytes_written"]           += s.get("outputBytes", 0)
                metrics["records_written"]         += s.get("outputRecords", 0)
                metrics["shuffle_read_bytes"]      += s.get("shuffleReadBytes", 0)
                metrics["shuffle_write_bytes"]     += s.get("shuffleWriteBytes", 0)
                metrics["shuffle_records_read"]    += s.get("shuffleReadRecords", 0)
                metrics["shuffle_records_written"] += s.get("shuffleWriteRecords", 0)
                metrics["executor_cpu_time_ns"]    += s.get("executorCpuTime", 0)
                metrics["executor_run_time_ms"]    += s.get("executorRunTime", 0)
                metrics["result_size"]             += s.get("resultSize", 0)
            except Exception:
                continue

        metrics["_source"] = "spark_rest_api"
        return metrics

    except Exception as ex1:
        e1 = ex1

    # ---- Strategy 2: statusStore JVM API ----
    try:
        sc = spark.sparkContext
        jsc = sc._jsc
        status_store = jsc.sc().statusStore()
        completed_jobs = status_store.jobsList(None).toArray()
        if len(completed_jobs) == 0:
            return empty

        last_job = completed_jobs[-1]
        stage_ids = last_job.stageIds()
        metrics = dict(empty)

        for sid in stage_ids:
            try:
                stage_data = status_store.stageData(sid, False)
                if stage_data.isEmpty():
                    continue
                stage = stage_data.get()
                m = stage.metrics()
                if m.isEmpty():
                    continue
                sm = m.get()
                metrics["bytes_read"]              += sm.inputMetrics().bytesRead()
                metrics["records_read"]            += sm.inputMetrics().recordsRead()
                metrics["bytes_written"]           += sm.outputMetrics().bytesWritten()
                metrics["records_written"]         += sm.outputMetrics().recordsWritten()
                metrics["shuffle_read_bytes"]      += sm.shuffleReadMetrics().totalBytesRead()
                metrics["shuffle_write_bytes"]     += sm.shuffleWriteMetrics().bytesWritten()
                metrics["shuffle_records_read"]    += sm.shuffleReadMetrics().recordsRead()
                metrics["shuffle_records_written"] += sm.shuffleWriteMetrics().recordsWritten()
                metrics["executor_cpu_time_ns"]    += sm.executorCpuTime()
                metrics["executor_run_time_ms"]    += sm.executorRunTime()
                metrics["result_size"]             += sm.resultSize()
            except Exception:
                continue

        metrics["_source"] = "status_store_jvm"
        return metrics

    except Exception as ex2:
        empty["_capture_error"] = f"REST API: {e1} | statusStore: {ex2}"
        return empty

# COMMAND ----------

# ---------------------------------------------------------------------------
# Plan shape validators (same as serverless harness)
# ---------------------------------------------------------------------------

def get_explain_plan(sql: str) -> str:
    """Get the physical plan for a SQL query via EXPLAIN FORMATTED."""
    explain_sql = sql.strip()

    # For CTAS, explain the SELECT portion
    if explain_sql.upper().startswith("CREATE"):
        as_idx = explain_sql.upper().find(" AS ")
        if as_idx >= 0:
            explain_sql = explain_sql[as_idx + 4:]

    try:
        rows = spark.sql(f"EXPLAIN FORMATTED {explain_sql}").collect()
        return "\n".join([r[0] for r in rows])
    except Exception as e:
        return f"EXPLAIN failed: {e}"


def validate_plan(plan_text: str, required_ops: list, forbidden_ops: list = None) -> tuple:
    """
    Check that the physical plan contains all required operators
    and none of the forbidden ones.
    Returns (is_valid, reason).
    """
    plan_upper = plan_text.upper()
    forbidden_ops = forbidden_ops or []

    for op in required_ops:
        if op.upper() not in plan_upper:
            return False, f"Missing required operator: {op}"

    for op in forbidden_ops:
        if op.upper() in plan_upper:
            return False, f"Found forbidden operator: {op} (AQE rewrite?)"

    return True, "OK"

# COMMAND ----------

# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRun:
    """One execution of one experiment query."""
    query_label: str            # e.g., "Q1", "Q4"
    run_number: int
    is_warmup: bool             # True for warmup runs (excluded from analysis)
    wall_clock_ms: float
    plan_valid: bool
    plan_validation_msg: str
    # Spark metrics
    bytes_read: int = 0
    bytes_written: int = 0
    shuffle_read_bytes: int = 0
    shuffle_write_bytes: int = 0
    executor_cpu_time_ns: int = 0
    executor_run_time_ms: int = 0
    records_read: int = 0
    records_written: int = 0
    shuffle_records_written: int = 0
    # Derived
    read_gb: float = 0.0
    shuffle_write_gb: float = 0.0
    cpu_time_ms: float = 0.0
    # Raw plan (stored but not printed)
    plan_text: str = ""
    timestamp: str = ""

    def __post_init__(self):
        self.read_gb = self.bytes_read / (1024**3) if self.bytes_read else 0.0
        self.shuffle_write_gb = self.shuffle_write_bytes / (1024**3) if self.shuffle_write_bytes else 0.0
        self.cpu_time_ms = self.executor_cpu_time_ns / 1e6 if self.executor_cpu_time_ns else 0.0

    def to_dict(self):
        d = asdict(self)
        d.pop("plan_text", None)  # Exclude bulky plan from exports
        return d


# Accumulator for all runs
experiment_results: list[ExperimentRun] = []

# COMMAND ----------

# ---------------------------------------------------------------------------
# Query definitions
# ---------------------------------------------------------------------------
# Each query uses direct SET configs (not hints) since personal compute
# gives us full control. We still validate plan shapes after each run.
# ---------------------------------------------------------------------------

def build_query_series() -> list:
    """
    Build the query series. Defined as a function so table names
    use the current config values.
    """
    return [
        # ── Baseline: Columnar Scan ──
        {
            "label": "Q1",
            "description": "Pure columnar scan (narrow) — BASELINE",
            "pre_configs": {},
            "sql": f"""
                SELECT COUNT(*), SUM(val1)
                FROM {LARGE_PARQUET}
                WHERE date BETWEEN '2025-01-01' AND '2025-01-{NUM_DATES:02d}'
            """,
            "required_ops": ["Scan"],
            "forbidden_ops": ["Sort"],
        },
        {
            "label": "Q2",
            "description": "Wide columnar scan (all 48 value columns)",
            "pre_configs": {},
            "sql": f"""
                SELECT COUNT(*),
                       {", ".join([f"SUM(val{i})" for i in range(1, 49)])}
                FROM {LARGE_PARQUET}
                WHERE date BETWEEN '2025-01-01' AND '2025-01-{NUM_DATES:02d}'
            """,
            "required_ops": ["Scan"],
            "forbidden_ops": [],
        },
        {
            "label": "Q3",
            "description": "Row-based scan (CSV) — deserialization overhead",
            "pre_configs": {},
            "sql": f"""
                SELECT COUNT(*), SUM(val1)
                FROM {LARGE_CSV}
            """,
            "required_ops": ["Scan"],
            "forbidden_ops": [],
        },

        # ── Shuffle Isolation ──
        {
            "label": "Q4",
            "description": "Scan + shuffle (GROUP BY key)",
            "pre_configs": {},
            "sql": f"""
                SELECT key, COUNT(*), SUM(val1)
                FROM {LARGE_PARQUET}
                WHERE date BETWEEN '2025-01-01' AND '2025-01-{NUM_DATES:02d}'
                GROUP BY key
            """,
            "required_ops": ["Exchange", "HashAggregate"],
            "forbidden_ops": [],
        },

        # ── Sort Isolation ──
        {
            "label": "Q5b",
            "description": "Scan + shuffle + sort (ORDER BY, materialized)",
            "pre_configs": {},
            "pre_sql": f"DROP TABLE IF EXISTS {SORTED_OUTPUT}",
            "sql": f"""
                CREATE TABLE {SORTED_OUTPUT}
                AS SELECT key, val1
                FROM {LARGE_PARQUET}
                WHERE date BETWEEN '2025-01-01' AND '2025-01-{NUM_DATES:02d}'
                ORDER BY key
            """,
            "required_ops": ["Sort", "Exchange"],
            "forbidden_ops": [],
        },

        # ── SortMergeJoin ──
        # Force SMJ by disabling broadcast entirely
        {
            "label": "Q6",
            "description": "SortMergeJoin (broadcast disabled via SET)",
            "pre_configs": {
                "spark.sql.autoBroadcastJoinThreshold": "-1",
                "spark.sql.join.preferSortMergeJoin": "true",
            },
            "sql": f"""
                SELECT COUNT(*), SUM(a.val1)
                FROM {LARGE_PARQUET} a
                JOIN {SMALL_DELTA} b ON a.key = b.key
                WHERE a.date BETWEEN '2025-01-01' AND '2025-01-{NUM_DATES:02d}'
            """,
            "required_ops": ["SortMergeJoin"],
            "forbidden_ops": ["BroadcastHashJoin", "BroadcastExchange"],
            "post_configs": {
                "spark.sql.autoBroadcastJoinThreshold": "10485760",
            },
        },

        # ── BroadcastHashJoin ──
        # Force BHJ by raising threshold well above small table size
        {
            "label": "Q7",
            "description": "BroadcastHashJoin (threshold raised via SET)",
            "pre_configs": {
                "spark.sql.autoBroadcastJoinThreshold": "2147483647",
            },
            "sql": f"""
                SELECT /*+ BROADCAST(b) */ COUNT(*), SUM(a.val1)
                FROM {LARGE_PARQUET} a
                JOIN {SMALL_DELTA} b ON a.key = b.key
                WHERE a.date BETWEEN '2025-01-01' AND '2025-01-{NUM_DATES:02d}'
            """,
            "required_ops": ["BroadcastHashJoin"],
            "forbidden_ops": ["SortMergeJoin"],
            "post_configs": {
                "spark.sql.autoBroadcastJoinThreshold": "10485760",
            },
        },

        # ── CartesianProduct ──
        # Use a TINY subset — cartesian on 1M rows is already 10^12 output
        # We create a micro table inline to keep it sane
        {
            "label": "Q8",
            "description": "CartesianProduct (micro × micro, ~10K rows each side)",
            "pre_configs": {},
            "pre_sql": f"""
                CREATE OR REPLACE TEMPORARY VIEW micro_table AS
                SELECT key, lookup_val FROM {SMALL_DELTA} LIMIT 10000
            """,
            "sql": """
                SELECT COUNT(*)
                FROM micro_table a
                CROSS JOIN micro_table b
            """,
            "required_ops": ["CartesianProduct"],
            "forbidden_ops": [],
        },
    ]

# Q9 (PySpark UDF) handled separately — see below

# COMMAND ----------

# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------

def apply_configs(configs: dict):
    """Apply a set of Spark configs via SET."""
    for k, v in configs.items():
        spark.conf.set(k, v)


def run_single(label: str, sql: str, run_number: int, is_warmup: bool,
               required_ops: list, forbidden_ops: list,
               pre_configs: dict = None, post_configs: dict = None,
               pre_sql: str = None, description: str = "") -> ExperimentRun:
    """Execute one query, capture wall-clock time + Spark metrics, validate plan."""

    # Apply pre-configs
    if pre_configs:
        apply_configs(pre_configs)

    # Run pre-SQL (e.g., DROP TABLE)
    if pre_sql:
        spark.sql(pre_sql)

    # Validate plan shape BEFORE running
    plan_text = get_explain_plan(sql)
    plan_valid, plan_msg = validate_plan(plan_text, required_ops, forbidden_ops)

    if not plan_valid:
        tag = "🔥" if not is_warmup else "  "
        print(f"  {tag} ❌ {label} {'(warmup) ' if is_warmup else ''}#{run_number}: {plan_msg}")
        if post_configs:
            apply_configs(post_configs)
        return ExperimentRun(
            query_label=label, run_number=run_number, is_warmup=is_warmup,
            wall_clock_ms=0, plan_valid=False, plan_validation_msg=plan_msg,
            plan_text=plan_text, timestamp=datetime.now().isoformat(),
        )

    # Force GC before measurement
    spark.sparkContext._jvm.System.gc()
    time.sleep(0.5)

    # ── Execute and measure ──
    t0 = time.time()
    is_ctas = sql.strip().upper().startswith("CREATE")
    if is_ctas:
        spark.sql(sql)
    else:
        spark.sql(sql).collect()
    t1 = time.time()
    wall_ms = (t1 - t0) * 1000.0

    # Capture Spark metrics for this job
    metrics = capture_spark_metrics()
    # One-time diagnostic: which metrics source worked?
    if not hasattr(capture_spark_metrics, "_diag_logged"):
        _src = metrics.get("_source", "NONE")
        _err = metrics.get("_capture_error", "")
        if _err:
            print(f"  [metrics] CAPTURE FAILED: {_err}")
        else:
            br = metrics.get("bytes_read", 0)
            print(f"  [metrics] source={_src} | first bytes_read={br:,}")
        capture_spark_metrics._diag_logged = True

    # Restore post-configs
    if post_configs:
        apply_configs(post_configs)

    run = ExperimentRun(
        query_label=label,
        run_number=run_number,
        is_warmup=is_warmup,
        wall_clock_ms=wall_ms,
        plan_valid=True,
        plan_validation_msg=plan_msg,
        bytes_read=metrics.get("bytes_read", 0),
        bytes_written=metrics.get("bytes_written", 0),
        shuffle_read_bytes=metrics.get("shuffle_read_bytes", 0),
        shuffle_write_bytes=metrics.get("shuffle_write_bytes", 0),
        executor_cpu_time_ns=metrics.get("executor_cpu_time_ns", 0),
        executor_run_time_ms=metrics.get("executor_run_time_ms", 0),
        records_read=metrics.get("records_read", 0),
        records_written=metrics.get("records_written", 0),
        shuffle_records_written=metrics.get("shuffle_records_written", 0),
        plan_text=plan_text,
        timestamp=datetime.now().isoformat(),
    )

    tag = "  " if is_warmup else "🔥"
    warmup_str = "(warmup) " if is_warmup else ""
    print(
        f"  {tag} ✅ {label} {warmup_str}#{run_number}: "
        f"{wall_ms:>8.0f} ms | "
        f"read {run.read_gb:.2f} GB | "
        f"shuffle {run.shuffle_write_gb:.2f} GB | "
        f"cpu {run.cpu_time_ms:.0f} ms"
    )

    return run


def run_query_series(queries: list = None, runs: int = None, warmup: int = None):
    """Execute the full query series with warmup + measured runs."""
    queries = queries or build_query_series()
    runs = runs or RUNS_PER_QUERY
    warmup = warmup or WARMUP_RUNS
    total_runs = warmup + runs

    print(f"\n{'='*80}")
    print(f"  EXPERIMENT: {len(queries)} queries × ({warmup} warmup + {runs} measured) runs")
    print(f"{'='*80}")

    for qdef in queries:
        label = qdef["label"]
        desc = qdef.get("description", "")
        print(f"\n  ── {label}: {desc} ──")

        valid_count = 0
        for i in range(1, total_runs + 1):
            is_warmup = (i <= warmup)
            run = run_single(
                label=label,
                sql=qdef["sql"],
                run_number=i,
                is_warmup=is_warmup,
                required_ops=qdef.get("required_ops", []),
                forbidden_ops=qdef.get("forbidden_ops", []),
                pre_configs=qdef.get("pre_configs", {}),
                post_configs=qdef.get("post_configs", {}),
                pre_sql=qdef.get("pre_sql"),
                description=desc,
            )
            experiment_results.append(run)
            if run.plan_valid and not is_warmup:
                valid_count += 1
            time.sleep(SETTLE_SECONDS)

        if valid_count < runs:
            print(f"  ⚠️  {label}: Only {valid_count}/{runs} valid measured runs.")

    print(f"\n{'='*80}")
    total = len(experiment_results)
    valid_measured = sum(1 for r in experiment_results if r.plan_valid and not r.is_warmup)
    print(f"  Complete. {total} total runs, {valid_measured} valid measured runs.")
    print(f"{'='*80}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q9 — PySpark UDF (separate cell)

# COMMAND ----------

def run_q9_pyspark_udf(runs: int = None, warmup: int = None):
    """
    Q9: Scan with identity Python UDF.
    Measures serialization overhead of shipping data to Python and back.
    """
    from pyspark.sql.functions import udf, col, sum as spark_sum
    from pyspark.sql.types import DoubleType

    runs = runs or RUNS_PER_QUERY
    warmup = warmup or WARMUP_RUNS
    total = warmup + runs

    @udf(DoubleType())
    def identity_udf(x):
        return x

    print(f"\n  ── Q9: PySpark UDF — identity function overhead ──")

    for i in range(1, total + 1):
        is_warmup = (i <= warmup)

        # Force GC
        spark.sparkContext._jvm.System.gc()
        time.sleep(0.5)

        t0 = time.time()
        df = (
            spark.table(LARGE_PARQUET)
            .filter(f"date BETWEEN '2025-01-01' AND '2025-01-{NUM_DATES:02d}'")
            .withColumn("val1_processed", identity_udf(col("val1")))
        )
        result = df.agg(spark_sum("val1_processed")).collect()
        t1 = time.time()
        wall_ms = (t1 - t0) * 1000.0

        metrics = capture_spark_metrics()

        # Validate plan — should contain Python / BatchEvalPython / ArrowEvalPython
        f_buf = io.StringIO()
        with redirect_stdout(f_buf):
            df.explain(mode="formatted")
        plan_text = f_buf.getvalue()
        plan_valid, plan_msg = validate_plan(plan_text, ["Python"], [])

        run = ExperimentRun(
            query_label="Q9",
            run_number=i,
            is_warmup=is_warmup,
            wall_clock_ms=wall_ms,
            plan_valid=plan_valid,
            plan_validation_msg=plan_msg,
            bytes_read=metrics.get("bytes_read", 0),
            shuffle_write_bytes=metrics.get("shuffle_write_bytes", 0),
            executor_cpu_time_ns=metrics.get("executor_cpu_time_ns", 0),
            executor_run_time_ms=metrics.get("executor_run_time_ms", 0),
            records_read=metrics.get("records_read", 0),
            plan_text=plan_text,
            timestamp=datetime.now().isoformat(),
        )
        experiment_results.append(run)

        tag = "  " if is_warmup else "🔥"
        warmup_str = "(warmup) " if is_warmup else ""
        status = "✅" if plan_valid else "❌"
        print(
            f"  {tag} {status} Q9 {warmup_str}#{i}: "
            f"{wall_ms:>8.0f} ms | read {run.read_gb:.2f} GB | cpu {run.cpu_time_ms:.0f} ms"
        )
        time.sleep(SETTLE_SECONDS)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute Everything

# COMMAND ----------

# Uncomment to run the full experiment:
# run_query_series()
# run_q9_pyspark_udf()

# COMMAND ----------

# MAGIC %md
# MAGIC ## §4 — Analysis: Derive Operator Cost Ratios
# MAGIC
# MAGIC Uses wall-clock time as the primary measurement. CPU time is reported
# MAGIC as a secondary signal. Ratios are computed the same way as the session
# MAGIC brief specifies — by differencing queries that add one operator at a time.

# COMMAND ----------

def get_valid_runs(label: str, results: list = None) -> list:
    """Get valid, non-warmup runs for a given query label."""
    results = results or experiment_results
    return [r for r in results
            if r.query_label == label and r.plan_valid and not r.is_warmup]


def safe_mean(values: list) -> float:
    return statistics.mean(values) if values else 0.0


def safe_stdev(values: list) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def cv_pct(values: list) -> float:
    """Coefficient of variation as a percentage."""
    m = safe_mean(values)
    return (safe_stdev(values) / m * 100) if m > 0 else 0.0


def analyze_results(results: list = None):
    """
    Compute per-operator cost ratios from wall-clock time measurements.
    """
    results = results or experiment_results

    # ── Helper: get wall-clock times for a label ──
    def wall_times(label):
        return [r.wall_clock_ms for r in get_valid_runs(label, results)]

    def cpu_times(label):
        return [r.cpu_time_ms for r in get_valid_runs(label, results)]

    def read_gbs(label):
        return [r.read_gb for r in get_valid_runs(label, results) if r.read_gb > 0]

    def shuffle_write_gbs(label):
        return [r.shuffle_write_gb for r in get_valid_runs(label, results) if r.shuffle_write_gb > 0]

    # ── Print raw measurements ──
    print(f"\n{'='*90}")
    print(f"  RAW MEASUREMENTS (wall-clock ms, excluding warmup)")
    print(f"{'='*90}")
    print(f"  {'Label':<8} {'N':>3} {'Mean ms':>10} {'StdDev':>10} {'CV%':>6} "
          f"{'CPU ms':>10} {'Read GB':>9} {'ShufW GB':>9}")
    print(f"  {'-'*8} {'-'*3} {'-'*10} {'-'*10} {'-'*6} {'-'*10} {'-'*9} {'-'*9}")

    for label in ["Q1", "Q2", "Q3", "Q4", "Q5b", "Q6", "Q7", "Q8", "Q9"]:
        wt = wall_times(label)
        ct = cpu_times(label)
        rg = read_gbs(label)
        sg = shuffle_write_gbs(label)
        n = len(wt)
        if n == 0:
            print(f"  {label:<8} {'—':>3} {'NO DATA':>10}")
            continue

        cv = cv_pct(wt)
        cv_flag = " ⚠️" if cv > 20 else ""
        print(
            f"  {label:<8} {n:>3} {safe_mean(wt):>10.0f} {safe_stdev(wt):>10.0f} "
            f"{cv:>5.1f}%{cv_flag}"
            f" {safe_mean(ct):>9.0f} {safe_mean(rg):>9.2f} {safe_mean(sg):>9.2f}"
        )

    # ── Compute ratios ──
    q1_mean = safe_mean(wall_times("Q1"))
    if q1_mean == 0:
        print("\n❌ Q1 (baseline) has no data. Cannot compute ratios.")
        return None

    # Wall-clock per GB for baseline
    q1_read_gb = safe_mean(read_gbs("Q1"))
    q1_ms_per_gb = q1_mean / q1_read_gb if q1_read_gb > 0 else q1_mean

    ratios = {}

    # ScanColumnar — reference
    ratios["ScanColumnar"] = {"ratio": 1.0, "source": "Q1 (reference)"}

    # ScanRowBased — Q3 / Q1 (normalized by bytes read)
    q3_mean = safe_mean(wall_times("Q3"))
    q3_read_gb = safe_mean(read_gbs("Q3"))
    if q3_mean > 0 and q3_read_gb > 0:
        q3_ms_per_gb = q3_mean / q3_read_gb
        ratios["ScanRowBased"] = {
            "ratio": q3_ms_per_gb / q1_ms_per_gb if q1_ms_per_gb > 0 else 0,
            "source": "(Q3 ms/GB) / (Q1 ms/GB)",
        }

    # Shuffle — (Q4 - Q1) / shuffle_gb, normalized to Q1 ms/GB
    q4_mean = safe_mean(wall_times("Q4"))
    q4_shuffle_gb = safe_mean(shuffle_write_gbs("Q4"))
    if q4_mean > q1_mean and q4_shuffle_gb > 0:
        shuffle_ms_per_gb = (q4_mean - q1_mean) / q4_shuffle_gb
        ratios["Shuffle"] = {
            "ratio": shuffle_ms_per_gb / q1_ms_per_gb if q1_ms_per_gb > 0 else 0,
            "source": "(Q4 - Q1) ms / shuffle_GB / (Q1 ms/GB)",
        }

    # Sort — (Q5b - Q4) / sort_volume, normalized
    q5b_mean = safe_mean(wall_times("Q5b"))
    q5b_shuffle_gb = safe_mean(shuffle_write_gbs("Q5b"))
    if q5b_mean > q4_mean:
        # Sort volume ≈ data being sorted (shuffle output)
        sort_gb = q5b_shuffle_gb if q5b_shuffle_gb > 0 else q4_shuffle_gb
        if sort_gb > 0:
            sort_ms_per_gb = (q5b_mean - q4_mean) / sort_gb
            ratios["Sort"] = {
                "ratio": sort_ms_per_gb / q1_ms_per_gb if q1_ms_per_gb > 0 else 0,
                "source": "(Q5b - Q4) ms / sort_GB / (Q1 ms/GB)",
            }

    # SortMergeJoin — (Q6 - Q1) / join_shuffle_gb
    q6_mean = safe_mean(wall_times("Q6"))
    q6_shuffle_gb = safe_mean(shuffle_write_gbs("Q6"))
    if q6_mean > q1_mean and q6_shuffle_gb > 0:
        smj_ms_per_gb = (q6_mean - q1_mean) / q6_shuffle_gb
        ratios["SortMergeJoin"] = {
            "ratio": smj_ms_per_gb / q1_ms_per_gb if q1_ms_per_gb > 0 else 0,
            "source": "(Q6 - Q1) ms / shuffle_GB / (Q1 ms/GB)",
        }

    # BroadcastHashJoin — (Q7 - Q1) / read_gb
    q7_mean = safe_mean(wall_times("Q7"))
    q7_read_gb = safe_mean(read_gbs("Q7"))
    if q7_mean > 0 and q7_read_gb > 0:
        bhj_overhead_ms = q7_mean - q1_mean
        # BHJ overhead is small; normalize against read volume
        bhj_ms_per_gb = bhj_overhead_ms / q7_read_gb if q7_read_gb > 0 else 0
        ratios["BroadcastHashJoin"] = {
            "ratio": max(bhj_ms_per_gb / q1_ms_per_gb, 0) if q1_ms_per_gb > 0 else 0,
            "source": "(Q7 - Q1) ms / read_GB / (Q1 ms/GB)",
            "note": "May be <1× if BHJ is cheaper than an extra scan"
        }

    # CartesianProduct — Q8 total / estimated output GB
    q8_mean = safe_mean(wall_times("Q8"))
    q8_runs = get_valid_runs("Q8", results)
    if q8_mean > 0 and q8_runs:
        # 10K × 10K = 100M rows. Estimate ~100 bytes/row → ~9.3 GB output
        output_rows = 10000 * 10000  # from our micro_table LIMIT
        output_gb_est = (output_rows * 100) / (1024**3)
        if output_gb_est > 0:
            cart_ms_per_gb = q8_mean / output_gb_est
            ratios["CartesianProduct"] = {
                "ratio": cart_ms_per_gb / q1_ms_per_gb if q1_ms_per_gb > 0 else 0,
                "source": "Q8 ms / output_GB_est / (Q1 ms/GB)",
                "note": f"Output est: {output_rows:,} rows ≈ {output_gb_est:.1f} GB",
            }

    # PySpark UDF — (Q9 - Q1) / read_gb
    q9_mean = safe_mean(wall_times("Q9"))
    q9_read_gb = safe_mean(read_gbs("Q9"))
    if q9_mean > q1_mean and q9_read_gb > 0:
        udf_ms_per_gb = (q9_mean - q1_mean) / q9_read_gb
        ratios["PySparkUDF"] = {
            "ratio": udf_ms_per_gb / q1_ms_per_gb if q1_ms_per_gb > 0 else 0,
            "source": "(Q9 - Q1) ms / read_GB / (Q1 ms/GB)",
        }

    # ── Print comparison table ──
    print(f"\n{'='*90}")
    coeff_label = "PHOTON" if CURRENT_COEFFICIENTS is COEFFICIENTS_PHOTON else "STANDARD"
    print(f"  OPERATOR COST RATIOS -- Empirical (wall-clock) vs {coeff_label} Coefficients")
    print(f"{'='*90}")
    print(f"  {'Operator':<22} {'Empirical':>10} {'Current':>10} {'Emp/Cur':>10} {'Status':>12}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    all_ops = ["ScanColumnar", "ScanRowBased", "Shuffle", "Sort",
               "BroadcastHashJoin", "SortMergeJoin", "CartesianProduct", "PySparkUDF"]

    for op in all_ops:
        emp = ratios.get(op, {}).get("ratio")
        cur = CURRENT_COEFFICIENTS.get(op, {}).get("ratio")

        if emp is not None and cur is not None:
            delta = emp / cur if cur > 0 else float('inf')
            if 0.5 <= delta <= 2.0:
                status = "✓ close"
            else:
                status = "⚠️  REVIEW"
            print(f"  {op:<22} {emp:>9.1f}× {cur:>9.1f}× {delta:>9.2f}× {status:>12}")
        elif emp is not None:
            print(f"  {op:<22} {emp:>9.1f}× {'N/A':>10} {'—':>10} {'—':>12}")
        else:
            cur_str = f"{cur:.1f}×" if cur else "N/A"
            print(f"  {op:<22} {'NO DATA':>10} {cur_str:>10} {'—':>10} {'—':>12}")

    print(f"\n  Baseline: Q1 = {q1_mean:.0f} ms for {q1_read_gb:.2f} GB "
          f"= {q1_ms_per_gb:.0f} ms/GB")
    print(f"  Ratios flagged ⚠️  REVIEW differ by >2× from current coefficients.\n")

    # ── Secondary: CPU-time-based ratios ──
    q1_cpu = safe_mean(cpu_times("Q1"))
    if q1_cpu > 0 and q1_read_gb > 0:
        q1_cpu_per_gb = q1_cpu / q1_read_gb
        print(f"  SECONDARY SIGNAL — CPU time ratios (for cross-check):")
        for label, op in [("Q3","ScanRowBased"), ("Q4","Shuffle"), ("Q6","SortMergeJoin"),
                          ("Q7","BroadcastHashJoin"), ("Q9","PySparkUDF")]:
            ct = safe_mean(cpu_times(label))
            rg = safe_mean(read_gbs(label))
            if ct > 0 and rg > 0:
                cpu_ratio = (ct / rg) / q1_cpu_per_gb
                print(f"    {op:<22} CPU ratio: {cpu_ratio:.1f}×")

    return ratios

# COMMAND ----------

# MAGIC %md
# MAGIC ## §5 — Export Results

# COMMAND ----------

def export_results_json(path: str = "/tmp/coefficient_experiment_results.json"):
    """Export all experiment results to JSON."""
    data = {
        "experiment_config": {
            "catalog": CATALOG,
            "schema": SCHEMA,
            "runs_per_query": RUNS_PER_QUERY,
            "warmup_runs": WARMUP_RUNS,
            "row_count": ROW_COUNT,
            "measurement_basis": "wall_clock_ms",
            "current_coefficients": CURRENT_COEFFICIENTS,
            "all_coefficients": {"standard": COEFFICIENTS_STANDARD, "photon": COEFFICIENTS_PHOTON},
            "selected_coefficient_set": "photon" if CURRENT_COEFFICIENTS is COEFFICIENTS_PHOTON else "standard",
        },
        "cluster_metadata": cluster_metadata,
        "runs": [r.to_dict() for r in experiment_results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Exported {len(experiment_results)} runs to {path}")


def export_results_delta(table_name: str = None):
    """Export results to a Delta table."""
    table_name = table_name or FQ("experiment_results")
    rows = [r.to_dict() for r in experiment_results]
    df = spark.createDataFrame(rows)
    df.write.mode("overwrite").saveAsTable(table_name)
    print(f"Exported {len(rows)} runs to {table_name}")


def print_summary_table():
    """Print a compact summary suitable for pasting into a doc or Slack."""
    all_labels = ["Q1", "Q2", "Q3", "Q4", "Q5b", "Q6", "Q7", "Q8", "Q9"]

    print("\n┌─────────────────────────────────────────────────────┐")
    print("│  Coefficient Experiment — Summary                   │")
    print("├────────┬──────┬──────────┬──────────┬───────────────┤")
    print("│ Query  │  N   │ Mean(ms) │  CV(%)   │ Read(GB)      │")
    print("├────────┼──────┼──────────┼──────────┼───────────────┤")

    for label in all_labels:
        runs = get_valid_runs(label)
        wt = [r.wall_clock_ms for r in runs]
        rg = [r.read_gb for r in runs if r.read_gb > 0]
        n = len(wt)
        if n == 0:
            print(f"│ {label:<6} │  —   │    —     │    —     │       —       │")
        else:
            cv = cv_pct(wt)
            print(f"│ {label:<6} │ {n:>4} │ {safe_mean(wt):>8.0f} │ {cv:>6.1f}%  │ {safe_mean(rg):>8.2f}      │")

    print("└────────┴──────┴──────────┴──────────┴───────────────┘")

# COMMAND ----------

# MAGIC %md
# MAGIC ## §6 — Quick-Start Checklist
# MAGIC
# MAGIC ```
# MAGIC [ ] 1. Create a compute cluster (any size)
# MAGIC        - Single node for baseline, multi-node to validate shuffle/join ratios
# MAGIC        - Single node: Standard_DS4_v2 or i3.xlarge (≥32 GB RAM)
# MAGIC        - DBR 14.3+ LTS recommended
# MAGIC        - Photon ON or OFF (just note which — it affects ratios)
# MAGIC
# MAGIC [ ] 2. Import this notebook and attach to the cluster
# MAGIC
# MAGIC [ ] 3. Run §0 (config) — verify settings look right
# MAGIC
# MAGIC [ ] 4. Run §1 (lock Spark config) — confirm AQE is OFF
# MAGIC
# MAGIC [ ] 5. Run §2 (setup tables) — takes ~10-20 min for 10 GB
# MAGIC
# MAGIC [ ] 6. Run §3 (experiment) — ~30-60 min total
# MAGIC        - Watch for ⚠️ plan validation failures
# MAGIC        - Watch for ⚠️ CV > 20% (may need more runs)
# MAGIC
# MAGIC [ ] 7. Run §4 (analysis) — instant, prints ratio table
# MAGIC
# MAGIC [ ] 8. Run §5 (export) — save results to Delta / JSON
# MAGIC
# MAGIC [ ] 9. Compare ratios to CostCoefficients.scala
# MAGIC        - ✓ close (within 2×): current weight is fine
# MAGIC        - ⚠️ REVIEW (>2× off): investigate and update
# MAGIC ```