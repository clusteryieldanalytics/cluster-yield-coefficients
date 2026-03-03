# Databricks notebook source
# MAGIC %md
# MAGIC # Operator Cost Coefficient — Empirical Experiment Harness
# MAGIC
# MAGIC **Purpose:** Run the controlled query series from the coefficient-transparency-session-brief
# MAGIC on Databricks Serverless, collect DBU consumption, validate plan shapes, and derive
# MAGIC empirical operator cost ratios.
# MAGIC
# MAGIC **Environment:** Databricks Serverless SQL Warehouse or Serverless Compute.
# MAGIC On serverless, `SET spark.sql.*` configs may be read-only. This harness uses
# MAGIC query hints (`/*+ BROADCAST */`, `/*+ MERGE */`, `/*+ SHUFFLE_HASH */`) and
# MAGIC CTAS materialization patterns to force plan shapes without requiring config changes.
# MAGIC
# MAGIC **Usage:**
# MAGIC 1. Run §1 (Setup) once to create test tables.
# MAGIC 2. Run §2 (Experiment) to execute the full query series with plan validation.
# MAGIC 3. Run §3 (Collection) to pull DBU data from `system.billing.usage`.
# MAGIC 4. Run §4 (Analysis) to compute ratios and compare to current coefficients.
# MAGIC
# MAGIC Alternatively, run experiments manually and use §3 to collect results afterward.

# COMMAND ----------

# MAGIC %md
# MAGIC ## §0 — Configuration

# COMMAND ----------

import json
import time
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Experiment configuration — edit these before running
# ---------------------------------------------------------------------------
CATALOG = "main"                       # Unity Catalog catalog
SCHEMA  = "test_weights"               # Schema for experiment tables
RUNS_PER_QUERY = 5                     # Number of repetitions per query
SETTLE_SECONDS = 5                     # Pause between runs to let billing flush
BILLING_LAG_MINUTES = 10               # system.billing.usage ingestion delay
TARGET_TABLE_SIZE_GB = 100             # Approximate target for large table
ROW_COUNT = 500_000_000               # Rows in large table (tune to hit target GB)

# Fully qualified names
FQ = lambda table: f"{CATALOG}.{SCHEMA}.{table}"

LARGE_PARQUET = FQ("large_parquet")
SMALL_DELTA   = FQ("small_delta")
LARGE_CSV     = FQ("large_csv")
SORTED_OUTPUT = FQ("sorted_output")

# Current coefficients from CostCoefficients.scala (for comparison)
CURRENT_COEFFICIENTS = {
    "ScanColumnar":       {"weight": 0.02, "ratio": 1.0},
    "ScanRowBased":       {"weight": 0.06, "ratio": 3.0},
    "Shuffle":            {"weight": 0.15, "ratio": 7.5},
    "Sort":               {"weight": 0.08, "ratio": 4.0},
    "BroadcastHashJoin":  {"weight": 0.02, "ratio": 1.0},
    "SortMergeJoin":      {"weight": 0.15, "ratio": 7.5},
    "CartesianProduct":   {"weight": 1.00, "ratio": 50.0},
    "PySparkUDF":         {"weight": 0.10, "ratio": 5.0},
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## §1 — Test Data Setup
# MAGIC
# MAGIC Creates three tables:
# MAGIC - **large_parquet**: ~100 GB Delta, 50 columns, partitioned by date
# MAGIC - **small_delta**: ~1 GB Delta, same key space (join small-side)
# MAGIC - **large_csv**: Same data as large_parquet but CSV (row-based scan)

# COMMAND ----------

def setup_schema():
    """Create the experiment schema if it doesn't exist."""
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    print(f"Schema {CATALOG}.{SCHEMA} ready.")


def setup_large_parquet():
    """
    Create the ~100 GB Delta table with 50 columns, partitioned by date.
    Uses explode(sequence(...)) to generate one row per (id, date) pair.
    """
    # Generate column expressions for 48 additional numeric columns
    extra_cols = ",\n    ".join(
        [f"CAST(rand() * 1000 AS DOUBLE) AS val{i}" for i in range(1, 49)]
    )

    sql = f"""
    CREATE OR REPLACE TABLE {LARGE_PARQUET}
    USING DELTA
    PARTITIONED BY (date)
    AS
    WITH base AS (
        SELECT
            id,
            uuid() AS key,
            {extra_cols},
            dt AS date
        FROM RANGE(0, {ROW_COUNT}) AS t(id)
        CROSS JOIN (
            SELECT explode(sequence(
                DATE '2025-01-01',
                DATE '2025-12-31',
                INTERVAL 1 DAY
            )) AS dt
        )
    )
    SELECT * FROM base
    """

    print(f"Creating {LARGE_PARQUET} (~{TARGET_TABLE_SIZE_GB} GB target)...")
    print("This will take a while on serverless. Go get coffee.")
    spark.sql(sql)

    # Report actual size
    desc = spark.sql(f"DESCRIBE DETAIL {LARGE_PARQUET}").collect()[0]
    size_gb = desc["sizeInBytes"] / (1024**3) if desc["sizeInBytes"] else "unknown"
    num_files = desc["numFiles"]
    print(f"Created {LARGE_PARQUET}: {size_gb:.1f} GB, {num_files} files")
    return size_gb


def setup_small_delta():
    """Create the ~1 GB small-side join table."""
    sql = f"""
    CREATE OR REPLACE TABLE {SMALL_DELTA}
    USING DELTA
    AS
    SELECT DISTINCT key, val1 AS lookup_val
    FROM {LARGE_PARQUET}
    LIMIT 10000000
    """
    print(f"Creating {SMALL_DELTA} (~1 GB target)...")
    spark.sql(sql)

    desc = spark.sql(f"DESCRIBE DETAIL {SMALL_DELTA}").collect()[0]
    size_gb = desc["sizeInBytes"] / (1024**3) if desc["sizeInBytes"] else "unknown"
    print(f"Created {SMALL_DELTA}: {size_gb:.2f} GB")
    return size_gb


def setup_large_csv():
    """Create a CSV copy of the large table for row-based scan experiments."""
    sql = f"""
    CREATE OR REPLACE TABLE {LARGE_CSV}
    USING CSV
    AS SELECT * FROM {LARGE_PARQUET}
    """
    print(f"Creating {LARGE_CSV} (CSV copy of large_parquet)...")
    spark.sql(sql)
    print(f"Created {LARGE_CSV}")


def run_full_setup():
    """Run all setup steps."""
    setup_schema()
    setup_large_parquet()
    setup_small_delta()
    setup_large_csv()
    print("\n✅ All test tables created.")

# COMMAND ----------

# Uncomment to run setup (takes significant time on first run):
# run_full_setup()

# COMMAND ----------

# MAGIC %md
# MAGIC ## §2 — Experiment Harness
# MAGIC
# MAGIC ### Plan Shape Validation
# MAGIC
# MAGIC On serverless, `SET spark.sql.adaptive.enabled = false` may be rejected.
# MAGIC Instead we:
# MAGIC 1. Use hint-based plan forcing (`/*+ MERGE(t) */`, `/*+ BROADCAST(t) */`)
# MAGIC 2. After each query, pull the executed physical plan from Spark UI / query history
# MAGIC 3. Validate that the expected operator appears in the plan
# MAGIC 4. Discard the run if the plan was rewritten unexpectedly

# COMMAND ----------

# ---------------------------------------------------------------------------
# Plan shape validators
# ---------------------------------------------------------------------------

def get_executed_plan(query_or_df) -> str:
    """
    Get the physical plan string for a query.
    For SQL queries, runs EXPLAIN FORMATTED first.
    For DataFrames, uses df.explain(mode="formatted").
    """
    if isinstance(query_or_df, str):
        rows = spark.sql(f"EXPLAIN FORMATTED {query_or_df}").collect()
        return "\n".join([r[0] for r in rows])
    else:
        # DataFrame — capture explain output
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            query_or_df.explain(mode="formatted")
        return f.getvalue()


def validate_plan_contains(plan_text: str, required_operators: list[str],
                           forbidden_operators: list[str] = None) -> tuple[bool, str]:
    """
    Check that the executed plan contains all required operators
    and none of the forbidden ones.

    Returns (is_valid, reason).
    """
    plan_upper = plan_text.upper()
    forbidden_operators = forbidden_operators or []

    for op in required_operators:
        if op.upper() not in plan_upper:
            return False, f"Missing required operator: {op}"

    for op in forbidden_operators:
        if op.upper() in plan_upper:
            return False, f"Found forbidden operator: {op} (AQE rewrite?)"

    return True, "Plan shape OK"


def try_disable_aqe():
    """
    Attempt to disable AQE. On serverless this may silently fail.
    Returns True if the SET succeeded, False otherwise.
    """
    try:
        spark.sql("SET spark.sql.adaptive.enabled = false")
        val = spark.sql("SET spark.sql.adaptive.enabled").collect()[0][1]
        if val.strip().lower() == "false":
            print("✅ AQE disabled via SET.")
            return True
        else:
            print("⚠️  SET accepted but AQE still reports enabled. Config may be read-only.")
            return False
    except Exception as e:
        print(f"⚠️  Cannot SET AQE config on this runtime: {e}")
        print("    Will rely on hints and post-hoc plan validation.")
        return False

# COMMAND ----------

# ---------------------------------------------------------------------------
# Run metadata collection
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRun:
    """One execution of one experiment query."""
    query_id: str
    query_label: str           # e.g., "Q1", "Q4"
    run_number: int
    started_at: str            # ISO timestamp
    finished_at: str
    wall_clock_ms: float
    plan_valid: bool
    plan_validation_msg: str
    plan_text: str             # full physical plan
    # These get filled in later from billing / Spark metrics
    dbu_consumed: Optional[float] = None
    bytes_read: Optional[int] = None
    shuffle_bytes_written: Optional[int] = None
    output_rows: Optional[int] = None

    def to_dict(self):
        return asdict(self)


# Accumulator for all runs
experiment_results: list[ExperimentRun] = []

# COMMAND ----------

# ---------------------------------------------------------------------------
# Query definitions
# ---------------------------------------------------------------------------

# Each query is a dict with:
#   label:              short name (Q1, Q2, ...)
#   sql:                the query text (or None for PySpark)
#   pyspark_fn:         a callable returning a DataFrame for PySpark queries
#   required_ops:       operators that MUST appear in physical plan
#   forbidden_ops:      operators that MUST NOT appear (to catch AQE rewrites)
#   description:        human-readable intent
#   pre_sql:            optional SQL to run before (e.g., drop temp tables)

QUERY_SERIES = [
    # ---- Baseline: Columnar Scan ----
    {
        "label": "Q1",
        "description": "Pure columnar scan (narrow read) — BASELINE",
        "sql": f"""
            SELECT COUNT(*), SUM(val1)
            FROM {LARGE_PARQUET}
            WHERE date BETWEEN '2025-01-01' AND '2025-12-31'
        """,
        "required_ops": ["Scan", "parquet"],  # FileScan parquet / ColumnarBatch
        "forbidden_ops": ["Exchange", "Sort"],
    },
    {
        "label": "Q2",
        "description": "Wide columnar scan (all 48 value columns)",
        "sql": f"""
            SELECT COUNT(*),
                   {", ".join([f"SUM(val{i})" for i in range(1, 49)])}
            FROM {LARGE_PARQUET}
            WHERE date BETWEEN '2025-01-01' AND '2025-12-31'
        """,
        "required_ops": ["Scan"],
        "forbidden_ops": ["Exchange"],
    },
    {
        "label": "Q3",
        "description": "Row-based scan (CSV) — isolates deserialization overhead",
        "sql": f"""
            SELECT COUNT(*), SUM(val1)
            FROM {LARGE_CSV}
        """,
        "required_ops": ["Scan"],
        "forbidden_ops": [],
    },

    # ---- Shuffle Isolation ----
    {
        "label": "Q4",
        "description": "Scan + shuffle (GROUP BY key)",
        "sql": f"""
            SELECT key, COUNT(*), SUM(val1)
            FROM {LARGE_PARQUET}
            WHERE date BETWEEN '2025-01-01' AND '2025-12-31'
            GROUP BY key
        """,
        "required_ops": ["Exchange", "HashAggregate"],
        "forbidden_ops": ["Sort"],
    },

    # ---- Sort Isolation ----
    {
        "label": "Q5b",
        "description": "Scan + shuffle + sort (ORDER BY, materialized to table)",
        "pre_sql": f"DROP TABLE IF EXISTS {SORTED_OUTPUT}",
        "sql": f"""
            CREATE OR REPLACE TABLE {SORTED_OUTPUT}
            AS SELECT key, val1
            FROM {LARGE_PARQUET}
            WHERE date BETWEEN '2025-01-01' AND '2025-12-31'
            ORDER BY key
        """,
        "required_ops": ["Sort", "Exchange"],
        "forbidden_ops": [],
    },

    # ---- Join: SortMergeJoin (forced via MERGE hint) ----
    {
        "label": "Q6",
        "description": "SortMergeJoin — forced via /*+ MERGE */ hint",
        "sql": f"""
            SELECT /*+ MERGE(b) */ COUNT(*), SUM(a.val1)
            FROM {LARGE_PARQUET} a
            JOIN {SMALL_DELTA} b ON a.key = b.key
            WHERE a.date BETWEEN '2025-01-01' AND '2025-12-31'
        """,
        "required_ops": ["SortMergeJoin"],
        "forbidden_ops": ["BroadcastHashJoin", "BroadcastExchange"],
    },

    # ---- Join: BroadcastHashJoin (forced via BROADCAST hint) ----
    {
        "label": "Q7",
        "description": "BroadcastHashJoin — forced via /*+ BROADCAST */ hint",
        "sql": f"""
            SELECT /*+ BROADCAST(b) */ COUNT(*), SUM(a.val1)
            FROM {LARGE_PARQUET} a
            JOIN {SMALL_DELTA} b ON a.key = b.key
            WHERE a.date BETWEEN '2025-01-01' AND '2025-12-31'
        """,
        "required_ops": ["BroadcastHashJoin"],
        "forbidden_ops": ["SortMergeJoin"],
    },

    # ---- CartesianProduct (small × small) ----
    {
        "label": "Q8",
        "description": "CartesianProduct (small × small only!)",
        "sql": f"""
            SELECT COUNT(*)
            FROM {SMALL_DELTA} a
            CROSS JOIN {SMALL_DELTA} b
        """,
        "required_ops": ["CartesianProduct"],
        "forbidden_ops": [],
    },
]

# Q9 (PySpark UDF) is defined as a function, not SQL — see below

# COMMAND ----------

# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------

def extract_query_id_from_plan(plan_text: str) -> Optional[str]:
    """Try to extract a query/statement ID from the plan text or recent history."""
    # On Databricks, query_id is in query_history. We'll tag it via statement.
    return None  # Filled in from billing join later


def run_single_sql(label: str, sql: str, run_number: int,
                   required_ops: list, forbidden_ops: list,
                   pre_sql: str = None, description: str = "") -> ExperimentRun:
    """Execute a single SQL query, validate plan, record metadata."""

    if pre_sql:
        spark.sql(pre_sql)

    # Tag the query so we can find it in billing later
    tag = f"cy_coeff_{label}_run{run_number}_{int(time.time())}"
    spark.sql(f"SET `spark.databricks.queryTag` = '{tag}'")

    started_at = datetime.utcnow()

    # First: EXPLAIN to get the plan BEFORE running
    explain_sql = sql.strip()
    # For CTAS, explain the SELECT portion
    if explain_sql.upper().startswith("CREATE"):
        # Extract the AS SELECT ... portion
        as_idx = explain_sql.upper().find(" AS ")
        if as_idx >= 0:
            explain_sql = explain_sql[as_idx + 4:]

    try:
        plan_rows = spark.sql(f"EXPLAIN FORMATTED {explain_sql}").collect()
        plan_text = "\n".join([r[0] for r in plan_rows])
    except Exception as e:
        plan_text = f"EXPLAIN failed: {e}"

    # Validate plan shape
    plan_valid, plan_msg = validate_plan_contains(plan_text, required_ops, forbidden_ops)

    if not plan_valid:
        print(f"  ❌ {label} run {run_number}: {plan_msg}")
        print(f"     SKIPPING — plan shape invalid. Review EXPLAIN output.")
        return ExperimentRun(
            query_id=tag,
            query_label=label,
            run_number=run_number,
            started_at=started_at.isoformat(),
            finished_at=datetime.utcnow().isoformat(),
            wall_clock_ms=0,
            plan_valid=False,
            plan_validation_msg=plan_msg,
            plan_text=plan_text,
        )

    # Run the actual query
    t0 = time.time()
    spark.sql(sql).collect() if not sql.strip().upper().startswith("CREATE") else spark.sql(sql)
    t1 = time.time()
    wall_ms = (t1 - t0) * 1000.0
    finished_at = datetime.utcnow()

    print(f"  ✅ {label} run {run_number}: {wall_ms:.0f} ms — {plan_msg}")

    return ExperimentRun(
        query_id=tag,
        query_label=label,
        run_number=run_number,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        wall_clock_ms=wall_ms,
        plan_valid=True,
        plan_validation_msg=plan_msg,
        plan_text=plan_text,
    )


def run_query_series(queries: list[dict] = None, runs: int = None):
    """
    Execute the full query series with N runs each.
    Appends to the global experiment_results list.
    """
    queries = queries or QUERY_SERIES
    runs = runs or RUNS_PER_QUERY

    aqe_disabled = try_disable_aqe()
    if not aqe_disabled:
        print("Proceeding with hint-based plan forcing + post-hoc validation.\n")

    for qdef in queries:
        label = qdef["label"]
        desc = qdef.get("description", "")
        print(f"\n{'='*60}")
        print(f"  {label}: {desc}")
        print(f"{'='*60}")

        valid_runs = 0
        attempt = 0
        max_attempts = runs * 2  # Allow some retries for plan validation failures

        while valid_runs < runs and attempt < max_attempts:
            attempt += 1
            run = run_single_sql(
                label=label,
                sql=qdef["sql"],
                run_number=valid_runs + 1,
                required_ops=qdef.get("required_ops", []),
                forbidden_ops=qdef.get("forbidden_ops", []),
                pre_sql=qdef.get("pre_sql"),
                description=desc,
            )
            experiment_results.append(run)
            if run.plan_valid:
                valid_runs += 1

            time.sleep(SETTLE_SECONDS)

        if valid_runs < runs:
            print(f"  ⚠️  {label}: Only {valid_runs}/{runs} valid runs after {attempt} attempts.")
            print(f"       Consider reviewing plan hints or running manually.")

    print(f"\n{'='*60}")
    print(f"  Query series complete. {len(experiment_results)} total runs recorded.")
    print(f"{'='*60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q9 — PySpark UDF (separate cell because it uses Python API)

# COMMAND ----------

def run_q9_pyspark_udf(runs: int = None):
    """
    Q9: Scan with identity Python UDF.
    Must be run separately from SQL series because it uses the DataFrame API.
    """
    from pyspark.sql.functions import udf, col, sum as spark_sum
    from pyspark.sql.types import DoubleType

    runs = runs or RUNS_PER_QUERY

    @udf(DoubleType())
    def identity_udf(x):
        return x

    print(f"\n{'='*60}")
    print(f"  Q9: PySpark UDF — identity function overhead")
    print(f"{'='*60}")

    for run_num in range(1, runs + 1):
        tag = f"cy_coeff_Q9_run{run_num}_{int(time.time())}"
        spark.conf.set("spark.databricks.queryTag", tag)

        started_at = datetime.utcnow()
        t0 = time.time()

        df = (
            spark.table(LARGE_PARQUET)
            .filter("date BETWEEN '2025-01-01' AND '2025-12-31'")
            .withColumn("val1_processed", identity_udf(col("val1")))
        )
        result = df.agg(spark_sum("val1_processed")).collect()

        t1 = time.time()
        wall_ms = (t1 - t0) * 1000.0
        finished_at = datetime.utcnow()

        # Get plan
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            df.explain(mode="formatted")
        plan_text = f.getvalue()

        # Validate: should contain PythonUDF or BatchEvalPython
        plan_valid, plan_msg = validate_plan_contains(
            plan_text,
            required_ops=["Python"],  # Matches PythonUDF, BatchEvalPython, ArrowEvalPython
            forbidden_ops=[]
        )

        status = "✅" if plan_valid else "❌"
        print(f"  {status} Q9 run {run_num}: {wall_ms:.0f} ms — {plan_msg}")

        experiment_results.append(ExperimentRun(
            query_id=tag,
            query_label="Q9",
            run_number=run_num,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            wall_clock_ms=wall_ms,
            plan_valid=plan_valid,
            plan_validation_msg=plan_msg,
            plan_text=plan_text,
        ))
        time.sleep(SETTLE_SECONDS)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute Full Experiment

# COMMAND ----------

# Uncomment to run the full experiment:
# run_query_series()
# run_q9_pyspark_udf()

# COMMAND ----------

# MAGIC %md
# MAGIC ## §3 — DBU Collection from `system.billing.usage`
# MAGIC
# MAGIC Serverless queries appear in `system.billing.usage` with a short lag.
# MAGIC These queries join billing data back to our experiment runs by timestamp
# MAGIC and query tag.
# MAGIC
# MAGIC **If running experiments manually**, use the queries below directly in
# MAGIC a SQL editor, substituting your own time window.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a — Raw billing query (run this in SQL editor or below)
# MAGIC
# MAGIC ```sql
# MAGIC -- Find all experiment queries by tag prefix.
# MAGIC -- Run this ≥10 min after the experiment completes (billing ingestion lag).
# MAGIC
# MAGIC SELECT
# MAGIC     u.usage_date,
# MAGIC     u.usage_start_time,
# MAGIC     u.usage_end_time,
# MAGIC     u.usage_quantity           AS dbu_consumed,
# MAGIC     u.usage_metadata.job_id,
# MAGIC     u.usage_metadata.notebook_id,
# MAGIC     q.statement_id             AS query_id,
# MAGIC     q.statement_text,
# MAGIC     q.total_duration_ms,
# MAGIC     q.rows_produced,
# MAGIC     q.read_bytes,
# MAGIC     q.shuffle_write_bytes,
# MAGIC     q.status
# MAGIC FROM system.billing.usage u
# MAGIC LEFT JOIN system.query.history q
# MAGIC   ON u.usage_metadata.statement_id = q.statement_id
# MAGIC WHERE u.sku_name LIKE '%SERVERLESS%'
# MAGIC   AND u.usage_date >= CURRENT_DATE - INTERVAL 3 DAYS
# MAGIC   -- Narrow to experiment window:
# MAGIC   AND u.usage_start_time >= '2025-XX-XX 00:00:00'   -- ← FILL IN
# MAGIC   AND u.usage_start_time <= '2025-XX-XX 23:59:59'   -- ← FILL IN
# MAGIC ORDER BY u.usage_start_time
# MAGIC ```

# COMMAND ----------

def generate_billing_query(experiment_start: str = None, experiment_end: str = None) -> str:
    """
    Generate a billing SQL query scoped to the experiment time window.
    If start/end not provided, uses timestamps from experiment_results.
    """
    if experiment_results:
        starts = [r.started_at for r in experiment_results if r.plan_valid]
        ends = [r.finished_at for r in experiment_results if r.plan_valid]
        if starts and ends:
            experiment_start = experiment_start or min(starts)
            experiment_end = experiment_end or max(ends)

    start_str = experiment_start or "2025-01-01T00:00:00"
    end_str = experiment_end or "2025-12-31T23:59:59"

    # Add buffer for billing lag
    from datetime import datetime, timedelta
    try:
        end_dt = datetime.fromisoformat(end_str) + timedelta(minutes=30)
        end_str = end_dt.isoformat()
    except Exception:
        pass

    return f"""
    SELECT
        q.statement_id                              AS query_id,
        q.statement_text,
        q.status,
        q.total_duration_ms,
        q.rows_produced,
        q.read_bytes,
        q.read_bytes / (1024*1024*1024)             AS read_gb,
        q.shuffle_write_bytes,
        q.shuffle_write_bytes / (1024*1024*1024)    AS shuffle_write_gb,
        u.usage_quantity                            AS dbu_consumed,
        u.usage_start_time,
        u.sku_name
    FROM system.query.history q
    LEFT JOIN system.billing.usage u
        ON u.usage_metadata.statement_id = q.statement_id
    WHERE q.execution_status = 'FINISHED'
      AND q.start_time >= '{start_str}'
      AND q.start_time <= '{end_str}'
      AND (
          q.statement_text LIKE '%test_weights%'
          OR q.statement_text LIKE '%cy_coeff%'
      )
    ORDER BY q.start_time
    """


def collect_billing_data():
    """
    Run the billing query and attach DBU/bytes data to experiment_results.
    Call this ≥10 min after the experiment finishes.
    """
    sql = generate_billing_query()
    print("Running billing collection query...")
    print(f"(Ensure ≥{BILLING_LAG_MINUTES} min have passed since last experiment run)\n")

    billing_df = spark.sql(sql)
    billing_rows = billing_df.collect()

    print(f"Found {len(billing_rows)} billing records.\n")

    # Build a lookup by statement text fragment → billing row
    # We match on the query tag we embedded
    for run in experiment_results:
        if not run.plan_valid:
            continue
        for brow in billing_rows:
            stmt = brow["statement_text"] or ""
            if run.query_id in stmt or (run.query_label in stmt and f"run{run.run_number}" in stmt):
                run.dbu_consumed = brow["dbu_consumed"]
                run.bytes_read = brow["read_bytes"]
                run.shuffle_bytes_written = brow["shuffle_write_bytes"]
                run.output_rows = brow["rows_produced"]
                break

    matched = sum(1 for r in experiment_results if r.plan_valid and r.dbu_consumed is not None)
    total_valid = sum(1 for r in experiment_results if r.plan_valid)
    print(f"Matched {matched}/{total_valid} valid runs to billing data.")

    if matched < total_valid:
        print("\n⚠️  Unmatched runs may need manual billing lookup.")
        print("   Try the raw billing query in §3a with a wider time window.")

    return billing_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b — Manual experiment recording template
# MAGIC
# MAGIC If automated collection is unreliable, record results manually here.

# COMMAND ----------

def create_manual_results_template():
    """
    Print a template for manually recording experiment results.
    Fill this in after running queries manually in the SQL editor.
    """
    template = """
# ── Manual Experiment Results ──
# Copy this cell, fill in the values, and run it.

manual_results = [
    # (label, run, dbu_consumed, read_gb, shuffle_write_gb, wall_ms, plan_valid)
    ("Q1", 1, None, None, None, None, True),  # Pure columnar scan
    ("Q1", 2, None, None, None, None, True),
    ("Q1", 3, None, None, None, None, True),
    ("Q1", 4, None, None, None, None, True),
    ("Q1", 5, None, None, None, None, True),

    ("Q2", 1, None, None, None, None, True),  # Wide columnar scan
    ("Q2", 2, None, None, None, None, True),
    ("Q2", 3, None, None, None, None, True),
    ("Q2", 4, None, None, None, None, True),
    ("Q2", 5, None, None, None, None, True),

    ("Q3", 1, None, None, None, None, True),  # Row-based scan (CSV)
    ("Q3", 2, None, None, None, None, True),
    ("Q3", 3, None, None, None, None, True),
    ("Q3", 4, None, None, None, None, True),
    ("Q3", 5, None, None, None, None, True),

    ("Q4", 1, None, None, None, None, True),  # Scan + shuffle (GROUP BY)
    ("Q4", 2, None, None, None, None, True),
    ("Q4", 3, None, None, None, None, True),
    ("Q4", 4, None, None, None, None, True),
    ("Q4", 5, None, None, None, None, True),

    ("Q5b", 1, None, None, None, None, True), # Scan + shuffle + sort (CTAS ORDER BY)
    ("Q5b", 2, None, None, None, None, True),
    ("Q5b", 3, None, None, None, None, True),
    ("Q5b", 4, None, None, None, None, True),
    ("Q5b", 5, None, None, None, None, True),

    ("Q6", 1, None, None, None, None, True),  # SortMergeJoin (MERGE hint)
    ("Q6", 2, None, None, None, None, True),
    ("Q6", 3, None, None, None, None, True),
    ("Q6", 4, None, None, None, None, True),
    ("Q6", 5, None, None, None, None, True),

    ("Q7", 1, None, None, None, None, True),  # BroadcastHashJoin (BROADCAST hint)
    ("Q7", 2, None, None, None, None, True),
    ("Q7", 3, None, None, None, None, True),
    ("Q7", 4, None, None, None, None, True),
    ("Q7", 5, None, None, None, None, True),

    ("Q8", 1, None, None, None, None, True),  # CartesianProduct (small × small)
    ("Q8", 2, None, None, None, None, True),
    ("Q8", 3, None, None, None, None, True),
    ("Q8", 4, None, None, None, None, True),
    ("Q8", 5, None, None, None, None, True),

    ("Q9", 1, None, None, None, None, True),  # PySpark UDF
    ("Q9", 2, None, None, None, None, True),
    ("Q9", 3, None, None, None, None, True),
    ("Q9", 4, None, None, None, None, True),
    ("Q9", 5, None, None, None, None, True),
]

# Convert to ExperimentRun objects
for label, run_num, dbu, read_gb, shuffle_gb, wall_ms, valid in manual_results:
    if dbu is not None:
        experiment_results.append(ExperimentRun(
            query_id=f"manual_{label}_run{run_num}",
            query_label=label,
            run_number=run_num,
            started_at="manual",
            finished_at="manual",
            wall_clock_ms=wall_ms or 0,
            plan_valid=valid,
            plan_validation_msg="manual entry",
            plan_text="",
            dbu_consumed=dbu,
            bytes_read=int(read_gb * 1024**3) if read_gb else None,
            shuffle_bytes_written=int(shuffle_gb * 1024**3) if shuffle_gb else None,
        ))
"""
    print(template)
    return template


# Uncomment to print the template:
# create_manual_results_template()

# COMMAND ----------

# MAGIC %md
# MAGIC ## §4 — Analysis: Derive Operator Cost Ratios

# COMMAND ----------

import statistics

def analyze_results(results: list[ExperimentRun] = None):
    """
    Compute per-operator cost ratios from experiment results.
    Returns the ratio table and comparison to current coefficients.
    """
    results = results or experiment_results
    valid = [r for r in results if r.plan_valid and r.dbu_consumed is not None]

    if not valid:
        print("❌ No valid results with billing data. Run collect_billing_data() first.")
        return None

    # Group by query label
    by_label: dict[str, list[ExperimentRun]] = {}
    for r in valid:
        by_label.setdefault(r.query_label, []).append(r)

    # Compute per-query DBU/GB
    def dbu_per_gb(runs: list[ExperimentRun]) -> tuple[float, float, list[float]]:
        """Returns (mean, stdev, raw_values) of DBU per GB read."""
        vals = []
        for r in runs:
            if r.bytes_read and r.bytes_read > 0 and r.dbu_consumed:
                gb = r.bytes_read / (1024**3)
                vals.append(r.dbu_consumed / gb)
        if not vals:
            return (0.0, 0.0, [])
        return (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0, vals)

    def mean_dbu(runs: list[ExperimentRun]) -> tuple[float, float]:
        """Returns (mean, stdev) of raw DBU consumed."""
        vals = [r.dbu_consumed for r in runs if r.dbu_consumed]
        if not vals:
            return (0.0, 0.0)
        return (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)

    def mean_shuffle_gb(runs: list[ExperimentRun]) -> float:
        vals = [r.shuffle_bytes_written / (1024**3)
                for r in runs if r.shuffle_bytes_written and r.shuffle_bytes_written > 0]
        return statistics.mean(vals) if vals else 0.0

    # ---- Step 1: Baseline ----
    q1_dpg_mean, q1_dpg_std, _ = dbu_per_gb(by_label.get("Q1", []))
    q1_dbu_mean, _ = mean_dbu(by_label.get("Q1", []))

    if q1_dpg_mean == 0:
        print("❌ Q1 (baseline) has no valid data. Cannot compute ratios.")
        return None

    print(f"{'='*70}")
    print(f"  BASELINE: ScanColumnar (Q1)")
    print(f"    DBU/GB  = {q1_dpg_mean:.6f} ± {q1_dpg_std:.6f}")
    print(f"    Avg DBU = {q1_dbu_mean:.4f}")
    print(f"{'='*70}\n")

    # ---- Step 2: Derive per-operator ratios ----
    ratios = {}

    # ScanColumnar — reference
    ratios["ScanColumnar"] = {
        "ratio": 1.0, "stdev": 0.0, "source": "Q1 (reference)",
        "dbu_per_gb": q1_dpg_mean
    }

    # ScanRowBased
    q3_dpg_mean, q3_dpg_std, _ = dbu_per_gb(by_label.get("Q3", []))
    if q3_dpg_mean > 0:
        ratios["ScanRowBased"] = {
            "ratio": q3_dpg_mean / q1_dpg_mean,
            "stdev": q3_dpg_std / q1_dpg_mean if q1_dpg_mean else 0,
            "source": "Q3_dbu_per_gb / Q1_dbu_per_gb",
            "dbu_per_gb": q3_dpg_mean
        }

    # Shuffle = (Q4 - Q1) / shuffle_gb
    q4_dbu_mean, q4_dbu_std = mean_dbu(by_label.get("Q4", []))
    q4_shuffle_gb = mean_shuffle_gb(by_label.get("Q4", []))
    if q4_dbu_mean > 0 and q4_shuffle_gb > 0:
        shuffle_cost_per_gb = (q4_dbu_mean - q1_dbu_mean) / q4_shuffle_gb
        ratios["Shuffle"] = {
            "ratio": shuffle_cost_per_gb / q1_dpg_mean if q1_dpg_mean else 0,
            "stdev": 0.0,  # Would need proper error propagation
            "source": "(Q4_dbu - Q1_dbu) / shuffle_gb / Q1_dbu_per_gb",
            "dbu_per_gb": shuffle_cost_per_gb
        }

    # Sort = (Q5b - Q4) / sorted_volume_gb
    q5b_dbu_mean, _ = mean_dbu(by_label.get("Q5b", []))
    q5b_shuffle_gb = mean_shuffle_gb(by_label.get("Q5b", []))
    if q5b_dbu_mean > 0 and q4_dbu_mean > 0:
        # Sort volume ≈ shuffle output (the data being sorted)
        sort_gb = q5b_shuffle_gb if q5b_shuffle_gb > 0 else q4_shuffle_gb
        if sort_gb > 0:
            sort_cost_per_gb = (q5b_dbu_mean - q4_dbu_mean) / sort_gb
            ratios["Sort"] = {
                "ratio": sort_cost_per_gb / q1_dpg_mean if q1_dpg_mean else 0,
                "stdev": 0.0,
                "source": "(Q5b_dbu - Q4_dbu) / sort_gb / Q1_dbu_per_gb",
                "dbu_per_gb": sort_cost_per_gb
            }

    # SortMergeJoin = (Q6 - scan costs) / join volume
    q6_dbu_mean, _ = mean_dbu(by_label.get("Q6", []))
    if q6_dbu_mean > 0:
        # Approximate: subtract Q1 baseline (large scan) + small table scan
        # Small table scan is negligible at ~1 GB, but we include it
        smj_overhead = q6_dbu_mean - q1_dbu_mean
        # Join volume ≈ shuffle bytes from the join
        q6_shuffle_gb = mean_shuffle_gb(by_label.get("Q6", []))
        join_gb = q6_shuffle_gb if q6_shuffle_gb > 0 else 1.0
        smj_cost_per_gb = smj_overhead / join_gb
        ratios["SortMergeJoin"] = {
            "ratio": smj_cost_per_gb / q1_dpg_mean if q1_dpg_mean else 0,
            "stdev": 0.0,
            "source": "(Q6_dbu - Q1_dbu) / join_shuffle_gb / Q1_dbu_per_gb",
            "dbu_per_gb": smj_cost_per_gb
        }

    # BroadcastHashJoin = (Q7 - scan costs) / join volume
    q7_dbu_mean, _ = mean_dbu(by_label.get("Q7", []))
    if q7_dbu_mean > 0:
        bhj_overhead = q7_dbu_mean - q1_dbu_mean
        q7_read_gb = statistics.mean([
            r.bytes_read / (1024**3) for r in by_label.get("Q7", [])
            if r.bytes_read and r.bytes_read > 0
        ]) if by_label.get("Q7") else 0
        join_gb = q7_read_gb if q7_read_gb > 0 else 1.0
        bhj_cost_per_gb = bhj_overhead / join_gb if join_gb > 0 else 0
        ratios["BroadcastHashJoin"] = {
            "ratio": bhj_cost_per_gb / q1_dpg_mean if q1_dpg_mean else 0,
            "stdev": 0.0,
            "source": "(Q7_dbu - Q1_dbu) / read_gb / Q1_dbu_per_gb",
            "dbu_per_gb": bhj_cost_per_gb
        }

    # CartesianProduct — use raw DBU / output volume (different basis)
    q8_dbu_mean, _ = mean_dbu(by_label.get("Q8", []))
    q8_output_rows = statistics.mean([
        r.output_rows for r in by_label.get("Q8", [])
        if r.output_rows and r.output_rows > 0
    ]) if by_label.get("Q8") else 0
    if q8_dbu_mean > 0 and q8_output_rows > 0:
        # Estimate output GB (rough: ~100 bytes per output row for small_delta schema)
        output_gb_est = (q8_output_rows * 100) / (1024**3)
        cartesian_cost_per_gb = q8_dbu_mean / output_gb_est if output_gb_est > 0 else 0
        ratios["CartesianProduct"] = {
            "ratio": cartesian_cost_per_gb / q1_dpg_mean if q1_dpg_mean else 0,
            "stdev": 0.0,
            "source": "Q8_dbu / output_gb_est / Q1_dbu_per_gb",
            "dbu_per_gb": cartesian_cost_per_gb,
            "note": "output_gb estimated from row count × ~100 bytes/row"
        }

    # PySpark UDF = (Q9 - Q1) / bytes_read_gb
    q9_dbu_mean, _ = mean_dbu(by_label.get("Q9", []))
    q9_read_gb = statistics.mean([
        r.bytes_read / (1024**3) for r in by_label.get("Q9", [])
        if r.bytes_read and r.bytes_read > 0
    ]) if by_label.get("Q9") else 0
    if q9_dbu_mean > 0 and q9_read_gb > 0:
        udf_overhead_per_gb = (q9_dbu_mean - q1_dbu_mean) / q9_read_gb
        ratios["PySparkUDF"] = {
            "ratio": udf_overhead_per_gb / q1_dpg_mean if q1_dpg_mean else 0,
            "stdev": 0.0,
            "source": "(Q9_dbu - Q1_dbu) / read_gb / Q1_dbu_per_gb",
            "dbu_per_gb": udf_overhead_per_gb
        }

    # ---- Step 3: Print comparison table ----
    print(f"\n{'='*78}")
    print(f"  OPERATOR COST RATIOS — Empirical vs Current Coefficients")
    print(f"{'='*78}")
    print(f"  {'Operator':<22} {'Empirical':>10} {'Current':>10} {'Delta':>10} {'Status':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for op in ["ScanColumnar", "ScanRowBased", "Shuffle", "Sort",
               "BroadcastHashJoin", "SortMergeJoin", "CartesianProduct", "PySparkUDF"]:
        empirical = ratios.get(op, {}).get("ratio")
        current = CURRENT_COEFFICIENTS.get(op, {}).get("ratio")

        if empirical is not None and current is not None:
            delta = empirical / current if current > 0 else float('inf')
            if 0.5 <= delta <= 2.0:
                status = "✓ close"
            else:
                status = "⚠️ REVIEW"
            print(f"  {op:<22} {empirical:>9.1f}× {current:>9.1f}× {delta:>9.2f}× {status:>10}")
        elif empirical is not None:
            print(f"  {op:<22} {empirical:>9.1f}× {'N/A':>10} {'':>10} {'':>10}")
        else:
            print(f"  {op:<22} {'NO DATA':>10} {current:>9.1f}× {'':>10} {'':>10}")

    print(f"\n  Ratios flagged ⚠️ REVIEW differ by >2× from current coefficients.")
    print(f"  These should be investigated before updating CostCoefficients.scala.\n")

    return ratios

# COMMAND ----------

# MAGIC %md
# MAGIC ## §5 — Export Results

# COMMAND ----------

def export_results_json(path: str = "/tmp/coefficient_experiment_results.json"):
    """Export all experiment results to JSON for offline analysis."""
    data = {
        "experiment_config": {
            "catalog": CATALOG,
            "schema": SCHEMA,
            "runs_per_query": RUNS_PER_QUERY,
            "target_table_size_gb": TARGET_TABLE_SIZE_GB,
            "current_coefficients": CURRENT_COEFFICIENTS,
        },
        "runs": [r.to_dict() for r in experiment_results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Exported {len(experiment_results)} runs to {path}")


def export_results_delta(table_name: str = None):
    """Export results to a Delta table for SQL analysis."""
    table_name = table_name or FQ("experiment_results")
    rows = [r.to_dict() for r in experiment_results]
    # Remove plan_text to keep table size manageable
    for row in rows:
        row.pop("plan_text", None)
    df = spark.createDataFrame(rows)
    df.write.mode("overwrite").saveAsTable(table_name)
    print(f"Exported {len(rows)} runs to {table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## §6 — Quick-Reference: Manual SQL Queries
# MAGIC
# MAGIC If you prefer to run experiments manually in the SQL editor, here are
# MAGIC all queries ready to copy-paste. After running, use the billing query
# MAGIC in §3a to collect results.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Billing lookup for a specific query (by time window)
# MAGIC
# MAGIC ```sql
# MAGIC -- After running a specific experiment query, find its DBU cost.
# MAGIC -- Replace the timestamps with your actual execution window.
# MAGIC SELECT
# MAGIC     q.statement_id,
# MAGIC     q.statement_text,
# MAGIC     q.total_duration_ms,
# MAGIC     q.read_bytes / (1024*1024*1024)          AS read_gb,
# MAGIC     q.shuffle_write_bytes / (1024*1024*1024)  AS shuffle_write_gb,
# MAGIC     q.rows_produced,
# MAGIC     u.usage_quantity                          AS dbu_consumed
# MAGIC FROM system.query.history q
# MAGIC LEFT JOIN system.billing.usage u
# MAGIC     ON u.usage_metadata.statement_id = q.statement_id
# MAGIC WHERE q.start_time >= '2025-XX-XXTXX:XX:00'
# MAGIC   AND q.start_time <= '2025-XX-XXTXX:XX:00'
# MAGIC   AND q.execution_status = 'FINISHED'
# MAGIC ORDER BY q.start_time DESC
# MAGIC LIMIT 20
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plan validation query (run after each experiment query)
# MAGIC
# MAGIC ```sql
# MAGIC -- Check the physical plan for the most recent query.
# MAGIC -- Use this to validate plan shape if not using the automated harness.
# MAGIC
# MAGIC -- For Q6 (should show SortMergeJoin, NOT BroadcastHashJoin):
# MAGIC EXPLAIN FORMATTED
# MAGIC SELECT /*+ MERGE(b) */ COUNT(*), SUM(a.val1)
# MAGIC FROM main.test_weights.large_parquet a
# MAGIC JOIN main.test_weights.small_delta b ON a.key = b.key
# MAGIC WHERE a.date BETWEEN '2025-01-01' AND '2025-12-31';
# MAGIC
# MAGIC -- For Q7 (should show BroadcastHashJoin, NOT SortMergeJoin):
# MAGIC EXPLAIN FORMATTED
# MAGIC SELECT /*+ BROADCAST(b) */ COUNT(*), SUM(a.val1)
# MAGIC FROM main.test_weights.large_parquet a
# MAGIC JOIN main.test_weights.small_delta b ON a.key = b.key
# MAGIC WHERE a.date BETWEEN '2025-01-01' AND '2025-12-31';
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Serverless DBU rate check
# MAGIC
# MAGIC ```sql
# MAGIC -- Verify the serverless DBU rate on this account before running experiments.
# MAGIC SELECT
# MAGIC     sku_name,
# MAGIC     pricing.default AS list_price_per_dbu,
# MAGIC     pricing.effective AS effective_price_per_dbu
# MAGIC FROM system.billing.list_prices
# MAGIC WHERE sku_name LIKE '%SERVERLESS%'
# MAGIC   AND price_start_time = (
# MAGIC       SELECT MAX(price_start_time)
# MAGIC       FROM system.billing.list_prices
# MAGIC       WHERE sku_name LIKE '%SERVERLESS%'
# MAGIC   )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Table size verification
# MAGIC
# MAGIC ```sql
# MAGIC -- Verify table sizes before running experiments
# MAGIC DESCRIBE DETAIL main.test_weights.large_parquet;
# MAGIC DESCRIBE DETAIL main.test_weights.small_delta;
# MAGIC DESCRIBE DETAIL main.test_weights.large_csv;
# MAGIC ```