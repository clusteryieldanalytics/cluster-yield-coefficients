# Databricks notebook source
# MAGIC %md
# MAGIC # 3. Run Query — Parameterized
# MAGIC
# MAGIC **Do not run this notebook interactively.** It is designed to be called as a
# MAGIC Databricks Workflow task with a `query_id` parameter.
# MAGIC
# MAGIC Each workflow task runs this notebook with a different `query_id` (Q1, Q2, ...Q9, Q7b).
# MAGIC Each task is a separate job run → separate `job_run_id` → clean per-query billing
# MAGIC attribution in `system.billing.usage`.
# MAGIC
# MAGIC Results are appended to `test_weights.experiment.experiment_runs`.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import time
import json
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text("query_id", "", "Query ID (Q1, Q2, ... Q9, Q7b)")
dbutils.widgets.text("use_smj_hint", "false", "Force SHUFFLE_MERGE hint for Q6")

query_id = dbutils.widgets.get("query_id").strip().upper()
use_smj_hint = dbutils.widgets.get("use_smj_hint").strip().lower() == "true"

assert query_id, "query_id parameter is required"
print(f"Query ID: {query_id}")
print(f"Use SMJ hint: {use_smj_hint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Capture job context

# COMMAND ----------

# Get the current job_run_id for billing attribution.
# Multiple fallback paths because the context structure varies across
# Databricks versions and compute types.
job_run_id = ""
job_id = ""

try:
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()

    # Path 1: tags dict (most reliable across versions)
    try:
        tags = ctx.tags()
        job_run_id = str(tags.get("multitaskParentRunId") or tags.get("jobRunId") or "")
        job_id = str(tags.get("jobId") or "")
    except Exception:
        pass

    # Path 2: currentRunId (fallback)
    if not job_run_id:
        try:
            job_run_id = str(ctx.currentRunId().get())
        except Exception:
            pass

    # Path 3: JSON parsing (last resort)
    if not job_run_id:
        try:
            ctx_json = json.loads(ctx.toJson())
            job_run_id = str(ctx_json.get("currentRunId", {}).get("id", ""))
            job_id = job_id or str(ctx_json.get("tags", {}).get("jobId", ""))
        except Exception:
            pass

except Exception as e:
    print(f"Context extraction failed: {e}")

if job_run_id:
    print(f"Job Run ID: {job_run_id}  (for system.billing.usage lookup)")
    print(f"Job ID:     {job_id}")
else:
    print("⚠️  No job_run_id — running interactively? Billing attribution won't work.")
    print("    Run this notebook as a Workflow task for clean billing records.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query definitions

# COMMAND ----------

val_sums = ", ".join([f"SUM(val{i})" for i in range(1, 49)])

# Q6 SQL depends on the hint strategy decision from 02_validate_plans
if use_smj_hint:
    Q6_SQL = f"""
        SELECT /*+ SHUFFLE_MERGE(b) */ COUNT(*), SUM(a.val1)
        FROM {FQ}.large_delta a
        JOIN {FQ}.medium_delta b ON a.key = b.key
        WHERE a.date BETWEEN '{DATE_START}' AND '{DATE_END}'
    """
else:
    Q6_SQL = f"""
        SELECT COUNT(*), SUM(a.val1)
        FROM {FQ}.large_delta a
        JOIN {FQ}.medium_delta b ON a.key = b.key
        WHERE a.date BETWEEN '{DATE_START}' AND '{DATE_END}'
    """

QUERIES = {
    "Q1": {
        "label": "Narrow columnar scan (baseline)",
        "sql": f"""
            SELECT COUNT(*), SUM(val1)
            FROM {FQ}.large_delta
            WHERE date BETWEEN '{DATE_START}' AND '{DATE_END}'
        """,
    },
    "Q2": {
        "label": "Wide columnar scan (all columns)",
        "sql": f"""
            SELECT COUNT(*), {val_sums}
            FROM {FQ}.large_delta
            WHERE date BETWEEN '{DATE_START}' AND '{DATE_END}'
        """,
    },
    "Q3": {
        "label": "Row-based scan (CSV)",
        "sql": f"""
            SELECT COUNT(*), SUM(val1)
            FROM {FQ}.large_csv
        """,
    },
    "Q4": {
        "label": "Scan + shuffle (GROUP BY)",
        "sql": f"""
            SELECT key, COUNT(*), SUM(val1)
            FROM {FQ}.large_delta
            WHERE date BETWEEN '{DATE_START}' AND '{DATE_END}'
            GROUP BY key
        """,
    },
    "Q5": {
        "label": "Scan + shuffle + sort (ORDER BY, CTAS)",
        "type": "ctas",
        "sql": f"""
            CREATE OR REPLACE TABLE {FQ}.sorted_output AS
            SELECT key, val1
            FROM {FQ}.large_delta
            WHERE date BETWEEN '{DATE_START}' AND '{DATE_END}'
            ORDER BY key
        """,
    },
    "Q6": {
        "label": "SortMergeJoin" + (" (SHUFFLE_MERGE hint)" if use_smj_hint else " (natural)"),
        "sql": Q6_SQL,
    },
    "Q7": {
        "label": "BroadcastHashJoin (BROADCAST hint, medium table)",
        "sql": f"""
            SELECT /*+ BROADCAST(b) */ COUNT(*), SUM(a.val1)
            FROM {FQ}.large_delta a
            JOIN {FQ}.medium_delta b ON a.key = b.key
            WHERE a.date BETWEEN '{DATE_START}' AND '{DATE_END}'
        """,
    },
    "Q7B": {
        "label": "BroadcastHashJoin (natural, small table cross-check)",
        "sql": f"""
            SELECT COUNT(*), SUM(a.val1)
            FROM {FQ}.large_delta a
            JOIN {FQ}.small_delta b ON a.key = b.key
            WHERE a.date BETWEEN '{DATE_START}' AND '{DATE_END}'
        """,
    },
    "Q8": {
        "label": "CartesianProduct (CROSS JOIN, small × small)",
        "sql": f"""
            SELECT COUNT(*)
            FROM {FQ}.small_delta a
            CROSS JOIN {FQ}.small_delta b
        """,
    },
    "Q9": {
        "label": "PySpark UDF (identity)",
        "type": "pyspark",
    },
}

assert query_id in QUERIES, f"Unknown query_id: {query_id}. Valid: {list(QUERIES.keys())}"
query_def = QUERIES[query_id]
print(f"Running: {query_id} — {query_def['label']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helpers

# COMMAND ----------

def get_plan_text(query_sql):
    """Capture EXPLAIN FORMATTED output. For CTAS, explain the SELECT portion."""
    try:
        explain_sql = query_sql.strip()
        if explain_sql.upper().startswith("CREATE"):
            as_idx = explain_sql.upper().find(" AS ")
            if as_idx >= 0:
                explain_sql = explain_sql[as_idx + 4:]
        rows = spark.sql(f"EXPLAIN FORMATTED {explain_sql}").collect()
        return "\n".join([r[0] for r in rows])
    except Exception as e:
        return f"EXPLAIN failed: {e}"


def now_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute query N times

# COMMAND ----------

results = []
query_type = query_def.get("type", "sql")

# Only register UDF for Q9 — avoid polluting other queries' measurements
# with UDF registration overhead on Spark Connect
identity_udf = None
if query_type == "pyspark":
    @F.udf(DoubleType())
    def identity_udf(x):
        return x
    print("  Registered identity_udf for Q9.")

for run_num in range(1, NUM_RUNS + 1):
    print(f"  Run {run_num}/{NUM_RUNS}...")

    start_ts = now_iso()
    start_time = time.time()
    plan_text = ""

    if query_type == "pyspark":
        # Q9: PySpark UDF query
        df = (
            spark.table(f"{FQ}.large_delta")
            .filter(f"date BETWEEN '{DATE_START}' AND '{DATE_END}'")
            .withColumn("val1_processed", identity_udf("val1"))
            .agg(F.sum("val1_processed").alias("total"))
        )
        df.collect()
        # Capture plan after execution
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            df.explain(mode="formatted")
        plan_text = f.getvalue()
    elif query_type == "ctas":
        # Q5: CTAS — explain the SELECT portion, then execute the full CTAS
        plan_text = get_plan_text(query_def["sql"])
        spark.sql(query_def["sql"])
    else:
        # Standard SQL
        plan_text = get_plan_text(query_def["sql"])
        spark.sql(query_def["sql"]).collect()

    elapsed_s = time.time() - start_time
    end_ts = now_iso()

    results.append({
        "query_id": query_id,
        "run_number": run_num,
        "job_run_id": job_run_id,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "elapsed_seconds": round(elapsed_s, 2),
        "plan_text": plan_text if run_num == 1 else "",  # Save plan only for first run
    })
    print(f"    Elapsed: {elapsed_s:.2f}s")

print(f"\n✅ {query_id} complete. {NUM_RUNS} runs.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Append results to experiment_runs table

# COMMAND ----------

import pandas as pd

results_df = spark.createDataFrame(pd.DataFrame(results))
results_df.write.mode("append").saveAsTable(f"{FQ}.experiment_runs")

print(f"✅ {len(results)} rows appended to {FQ}.experiment_runs")
print(f"   job_run_id: {job_run_id}")
results_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify plan shape (first run only)

# COMMAND ----------

plan = results[0].get("plan_text", "")
if plan:
    # Quick sanity checks
    checks = {
        "Q1": ("FileScan", True),
        "Q6": ("SortMergeJoin", True),
        "Q7": ("BroadcastHashJoin", True),
        "Q7B": ("BroadcastHashJoin", True),
        "Q8": ("CartesianProduct|BroadcastNestedLoopJoin", True),
    }
    if query_id in checks:
        pattern, expected = checks[query_id]
        import re
        found = bool(re.search(pattern, plan))
        status = "✅" if found == expected else "⚠️  UNEXPECTED"
        print(f"{status} Plan shape check for {query_id}: looking for '{pattern}' → {'found' if found else 'NOT found'}")
    print(f"\nPlan (first 500 chars):\n{plan[:500]}")
else:
    print("No plan text captured.")
