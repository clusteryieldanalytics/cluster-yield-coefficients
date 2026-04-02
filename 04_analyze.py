# Databricks notebook source
# MAGIC %md
# MAGIC # 4. Analyze — Produce Ratio Table
# MAGIC
# MAGIC **Run after the workflow completes and billing data has propagated.**
# MAGIC
# MAGIC This notebook joins `system.billing.usage` to `experiment_runs` via `job_run_id`
# MAGIC for direct per-query DBU attribution — no apportionment or estimation needed.
# MAGIC
# MAGIC Output: empirical operator cost ratios with 95% confidence intervals,
# MAGIC cross-checks, comparison to current heuristic weights, and a saved Delta table.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json
import math
import statistics
import pandas as pd
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4a. Load experiment runs

# COMMAND ----------

runs_df = spark.table(f"{FQ}.experiment_runs")
print(f"Total experiment runs: {runs_df.count()}")
runs_df.groupBy("query_id").count().orderBy("query_id").display()

# COMMAND ----------

# Verify all expected queries are present
expected = {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q7B", "Q8", "Q9"}
found = set(runs_df.select("query_id").distinct().toPandas()["query_id"])
missing = expected - found
if missing:
    print(f"⚠️  Missing queries: {missing}")
    print("   Re-run the workflow for missing tasks before proceeding.")
else:
    print(f"✅ All {len(expected)} queries present.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4b. Join to system.billing.usage via job_run_id
# MAGIC
# MAGIC Each workflow task has a unique `job_run_id`. Billing records are keyed on
# MAGIC `usage_metadata.job_run_id`. This gives us direct DBU per query — no
# MAGIC apportionment needed.

# COMMAND ----------

# Get distinct job_run_ids from the experiment
job_run_ids = (
    runs_df.select("query_id", "job_run_id")
    .distinct()
    .toPandas()
)
print("Job run IDs per query:")
for _, row in job_run_ids.iterrows():
    print(f"  {row['query_id']}: {row['job_run_id']}")

# Check for retries: if a query has multiple job_run_ids, billing will be wrong
dupes = job_run_ids.groupby("query_id").size()
dupes_multi = dupes[dupes > 1]
if len(dupes_multi) > 0:
    print(f"\n⚠️  Multiple job_run_ids found (workflow retry?):")
    for qid, count in dupes_multi.items():
        print(f"    {qid}: {count} job_run_ids")
    print("   Fix: DELETE duplicate rows from experiment_runs, keeping only the latest run.")
    print("   Example: DELETE FROM experiment_runs WHERE job_run_id = '<old_run_id>'")
    print("   Then re-run this notebook.")

# COMMAND ----------

# Join billing data — use DISTINCT job_run_id to avoid fan-out.
# experiment_runs has N rows per query (one per run), all sharing the same
# job_run_id. Joining directly would multiply billing by N.
billing_df = spark.sql(f"""
    WITH distinct_runs AS (
        SELECT DISTINCT query_id, job_run_id
        FROM {FQ}.experiment_runs
    )
    SELECT
        r.query_id,
        r.job_run_id,
        SUM(b.usage_quantity) AS dbu_consumed
    FROM distinct_runs r
    JOIN system.billing.usage b
        ON CAST(r.job_run_id AS STRING) = CAST(b.usage_metadata.job_run_id AS STRING)
    WHERE b.sku_name LIKE '%SERVERLESS%'
    GROUP BY r.query_id, r.job_run_id
""")

print("DBU per query (from billing):")
billing_df.orderBy("query_id").display()

# COMMAND ----------

# Check for missing billing records
billing_pd = billing_df.toPandas()
billed_queries = set(billing_pd["query_id"])
missing_billing = expected - billed_queries

if missing_billing:
    print(f"⚠️  No billing records for: {missing_billing}")
    print(f"   Billing lag may be up to 24 hours. Current wait: {BILLING_LAG_MINUTES} min.")
    print(f"   Re-run this notebook after more time has passed.")
else:
    print(f"✅ Billing records found for all {len(expected)} queries.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4c. Load table sizes

# COMMAND ----------

sizes_json = spark.table(f"{FQ}.table_sizes").collect()[0]["sizes_json"]
table_sizes = json.loads(sizes_json)

LARGE_DELTA_GB = table_sizes["large_delta"]["gb"]
SMALL_DELTA_GB = table_sizes["small_delta"]["gb"]
MEDIUM_DELTA_GB = table_sizes["medium_delta"]["gb"]
LARGE_CSV_GB = table_sizes["large_csv"]["gb"]

print(f"large_delta:  {LARGE_DELTA_GB} GB")
print(f"small_delta:  {SMALL_DELTA_GB} GB")
print(f"medium_delta: {MEDIUM_DELTA_GB} GB")
print(f"large_csv:    {LARGE_CSV_GB} GB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4d. Load per-statement Spark metrics from query history
# MAGIC
# MAGIC These metrics are available for manual inspection and can be used to refine
# MAGIC the shuffle ratio (normalize by `shuffle_read_bytes` instead of `read_gb`).
# MAGIC The primary analysis uses billing DBU with read-volume normalization, but
# MAGIC shuffle bytes from query history provide a more precise denominator if needed.

# COMMAND ----------

# Get experiment time window from runs
window = runs_df.agg(
    F.min("start_ts").alias("exp_start"),
    F.max("end_ts").alias("exp_end"),
).collect()[0]

query_metrics = spark.sql(f"""
    SELECT
        statement_id,
        statement_text,
        start_time,
        end_time,
        total_duration_ms,
        total_task_duration_ms,
        read_bytes,
        shuffle_read_bytes,
        result_rows,
        execution_status
    FROM system.query.history
    WHERE start_time >= '{window["exp_start"]}'
      AND start_time <= TIMESTAMPADD(HOUR, 1, TIMESTAMP('{window["exp_end"]}'))
      AND execution_status = 'FINISHED'
    ORDER BY start_time
""")

print(f"Query history records in experiment window: {query_metrics.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4e. Build analysis table
# MAGIC
# MAGIC For each query: per-run wall clock from `experiment_runs`, total DBU from
# MAGIC billing, and Spark metrics (shuffle bytes, read bytes) from query history.

# COMMAND ----------

# Merge billing DBU with per-run timings
runs_pd = runs_df.toPandas()

# Map read_gb for per-GB normalization
READ_GB_MAP = {
    "Q1": LARGE_DELTA_GB, "Q2": LARGE_DELTA_GB, "Q3": LARGE_CSV_GB,
    "Q4": LARGE_DELTA_GB, "Q5": LARGE_DELTA_GB, "Q6": LARGE_DELTA_GB,
    "Q7": LARGE_DELTA_GB, "Q7B": LARGE_DELTA_GB, "Q8": SMALL_DELTA_GB,
    "Q9": LARGE_DELTA_GB,
}

analysis = {}
for qid in expected:
    qid_runs = runs_pd[runs_pd["query_id"] == qid]
    if len(qid_runs) == 0:
        print(f"⚠️  {qid}: no runs found. Skipping.")
        continue

    # Per-run wall clock from experiment_runs
    wall_clocks = qid_runs["elapsed_seconds"].tolist()

    # DBU from billing (one record per job_run_id = per query)
    billing_row = billing_pd[billing_pd["query_id"] == qid]
    dbu = billing_row["dbu_consumed"].sum() if len(billing_row) > 0 else None

    read_gb = READ_GB_MAP.get(qid, 1.0)

    analysis[qid] = {
        "n_runs": len(wall_clocks),
        "dbu": dbu,
        "dbu_per_gb": round(dbu / read_gb, 6) if dbu and read_gb > 0 else None,
        "wall_clocks": wall_clocks,
        "wall_mean": statistics.mean(wall_clocks),
        "wall_std": statistics.stdev(wall_clocks) if len(wall_clocks) > 1 else 0,
        "read_gb": read_gb,
    }

    dbu_str = f"{dbu:.4f}" if dbu else "N/A"
    dbu_gb_str = f"{analysis[qid]['dbu_per_gb']:.6f}" if analysis[qid]['dbu_per_gb'] else "N/A"
    print(f"{qid}: {len(wall_clocks)} runs, "
          f"DBU={dbu_str}, DBU/GB={dbu_gb_str}, "
          f"wall={analysis[qid]['wall_mean']:.1f}s ±{analysis[qid]['wall_std']:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4f. Compute operator cost ratios via differencing
# MAGIC
# MAGIC All ratios use direct DBU from `system.billing.usage`. Since each query ran
# MAGIC as a separate workflow task, DBU is directly attributable — no apportionment.

# COMMAND ----------

# Baseline: Q1 DBU per GB = 1.0×
q1 = analysis.get("Q1", {})
q1_dbu_per_gb = q1.get("dbu_per_gb")
q1_dbu = q1.get("dbu")

if not q1_dbu_per_gb:
    print("❌ Q1 billing data not available. Cannot compute ratios.")
    dbutils.notebook.exit("Q1 billing missing — re-run after billing propagation")

print(f"Q1 baseline: {q1_dbu:.4f} DBU total, {q1_dbu_per_gb:.6f} DBU/GB\n")

ratios = {}

# --- ScanColumnar (baseline) ---
ratios["ScanColumnar"] = {
    "ratio": 1.0,
    "dbu": q1_dbu,
    "dbu_per_gb": q1_dbu_per_gb,
    "method": "Q1 (baseline)",
}

# --- ScanRowBased = Q3 / Q1 ---
q3 = analysis.get("Q3", {})
if q3.get("dbu_per_gb"):
    ratios["ScanRowBased"] = {
        "ratio": round(q3["dbu_per_gb"] / q1_dbu_per_gb, 2),
        "dbu": q3["dbu"],
        "dbu_per_gb": q3["dbu_per_gb"],
        "method": "Q3 / Q1",
    }

# --- Shuffle = (Q4 - Q1) / shuffle_volume ---
# For the shuffle ratio, we difference total DBU and normalize by shuffle bytes.
# Since we don't have per-run shuffle bytes from billing, we use Q4 DBU - Q1 DBU
# and normalize by read_gb as an approximation. The blog narrative can refine this
# with shuffle_bytes from query history if available.
q4 = analysis.get("Q4", {})
if q4.get("dbu") and q1_dbu:
    shuffle_dbu = q4["dbu"] - q1_dbu
    # Normalize by read volume (shuffle volume is typically a fraction of read volume)
    shuffle_dbu_per_gb = shuffle_dbu / q4["read_gb"]
    ratios["Shuffle"] = {
        "ratio": round(shuffle_dbu_per_gb / q1_dbu_per_gb, 2),
        "dbu": shuffle_dbu,
        "dbu_per_gb": round(shuffle_dbu_per_gb, 6),
        "method": "(Q4_dbu - Q1_dbu) / read_GB",
    }

# --- Sort = (Q5 - Q4) / sort_volume ---
q5 = analysis.get("Q5", {})
if q5.get("dbu") and q4.get("dbu"):
    sort_dbu = q5["dbu"] - q4["dbu"]
    sort_dbu_per_gb = sort_dbu / q5["read_gb"]
    ratios["Sort"] = {
        "ratio": round(sort_dbu_per_gb / q1_dbu_per_gb, 2),
        "dbu": sort_dbu,
        "dbu_per_gb": round(sort_dbu_per_gb, 6),
        "method": "(Q5_dbu - Q4_dbu) / read_GB",
    }

# --- SortMergeJoin = Q6 / Q1 ---
q6 = analysis.get("Q6", {})
if q6.get("dbu") and q1_dbu:
    ratios["SortMergeJoin"] = {
        "ratio": round(q6["dbu"] / q1_dbu, 2),
        "dbu": q6["dbu"],
        "dbu_per_gb": q6.get("dbu_per_gb"),
        "method": "Q6_dbu / Q1_dbu",
    }

# --- BroadcastHashJoin = Q7 / Q1 ---
q7 = analysis.get("Q7", {})
if q7.get("dbu") and q1_dbu:
    ratios["BroadcastHashJoin"] = {
        "ratio": round(q7["dbu"] / q1_dbu, 2),
        "dbu": q7["dbu"],
        "dbu_per_gb": q7.get("dbu_per_gb"),
        "method": "Q7_dbu / Q1_dbu",
    }

# --- BHJ cross-check = Q7b / Q1 ---
q7b = analysis.get("Q7B", {})
if q7b.get("dbu") and q1_dbu:
    ratios["BroadcastHashJoin_crosscheck"] = {
        "ratio": round(q7b["dbu"] / q1_dbu, 2),
        "dbu": q7b["dbu"],
        "dbu_per_gb": q7b.get("dbu_per_gb"),
        "method": "Q7b_dbu / Q1_dbu (cross-check)",
    }

# --- SMJ vs BHJ multiplier (the money shot) ---
if "SortMergeJoin" in ratios and "BroadcastHashJoin" in ratios:
    smj_vs_bhj = ratios["SortMergeJoin"]["ratio"] / ratios["BroadcastHashJoin"]["ratio"]
    ratios["SMJ_vs_BHJ_multiplier"] = {
        "ratio": round(smj_vs_bhj, 2),
        "method": "SMJ_ratio / BHJ_ratio",
    }

# --- CartesianProduct ---
q8 = analysis.get("Q8", {})
if q8.get("dbu"):
    cart_dbu_per_input_gb = q8["dbu"] / SMALL_DELTA_GB
    ratios["CartesianProduct"] = {
        "ratio": round(cart_dbu_per_input_gb / q1_dbu_per_gb, 2),
        "dbu": q8["dbu"],
        "dbu_per_gb": round(cart_dbu_per_input_gb, 6),
        "method": "Q8_dbu / (Q1_dbu_per_gb × small_delta_gb)",
        "note": "Extrapolated from small tables; O(n×m) scaling",
    }

# --- PySpark UDF ---
q9 = analysis.get("Q9", {})
if q9.get("dbu") and q1_dbu:
    ratios["PySpark UDF"] = {
        "ratio": round(q9["dbu"] / q1_dbu, 2),
        "dbu": q9["dbu"],
        "dbu_per_gb": q9.get("dbu_per_gb"),
        "method": "Q9_dbu / Q1_dbu",
    }

# --- Per-column marginal scan cost ---
q2 = analysis.get("Q2", {})
if q2.get("dbu_per_gb") and q1_dbu_per_gb:
    ratios["MarginalColumnScan"] = {
        "ratio": round(q2["dbu_per_gb"] / q1_dbu_per_gb, 2),
        "method": "Q2 / Q1 (all cols vs 1 col)",
        "per_column_marginal_pct": round(
            ((q2["dbu_per_gb"] / q1_dbu_per_gb) - 1) / 47 * 100, 2
        ),
    }

print("=" * 85)
print("OPERATOR COST RATIOS (relative to ScanColumnar = 1.0×)  [metric: billing DBU]")
print("=" * 85)
for op, data in ratios.items():
    current = CURRENT_WEIGHTS.get(op, {}).get("ratio", "—")
    dbu_str = f"  DBU={data.get('dbu', '—'):.4f}" if isinstance(data.get("dbu"), (int, float)) else ""
    print(f"  {op:30s}  {data['ratio']:8.2f}×    (heuristic: {current}×){dbu_str}    [{data['method']}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4g. 95% Confidence Intervals
# MAGIC
# MAGIC CIs are computed from per-run wall-clock variance (all runs within a single
# MAGIC job share the same billing record, so we can't compute per-run DBU variance).
# MAGIC The CI on the ratio uses the coefficient of variation from wall-clock times
# MAGIC applied to the DBU-based ratio.

# COMMAND ----------

def ci_95(values):
    """95% CI using t-distribution for small samples."""
    n = len(values)
    if n < 2:
        return (None, None)
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    t_values = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    t = t_values.get(n, 1.96)
    return (round(mean - t * se, 3), round(mean + t * se, 3))


ci_results = {}
for qid, data in analysis.items():
    wall = data.get("wall_clocks", [])
    if len(wall) >= 2:
        ci = ci_95(wall)
        mean_wall = statistics.mean(wall)
        cv = round(statistics.stdev(wall) / mean_wall * 100, 1)

        # Derive ratio CI: ratio × (1 ± CV) scaled by t-interval
        ci_ratio_low = round(ci[0] / mean_wall, 3) if ci[0] else None
        ci_ratio_high = round(ci[1] / mean_wall, 3) if ci[1] else None

        ci_results[qid] = {
            "wall_mean": mean_wall,
            "wall_ci": ci,
            "cv": cv,
            "ci_scale_low": ci_ratio_low,   # multiplier to apply to DBU ratio
            "ci_scale_high": ci_ratio_high,
        }
        print(f"{qid}: wall={mean_wall:.1f}s, 95% CI=[{ci[0]}, {ci[1]}]s, CV={cv}%")

# Flag high-variance queries
for qid, data in ci_results.items():
    if data["cv"] > 20:
        print(f"\n⚠️  {qid} CV={data['cv']}% — consider increasing to 10 runs.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4h. Final Ratio Table (blog-ready)

# COMMAND ----------

print()
print("=" * 95)
print("FINAL RATIO TABLE — Blog Post 2  [direct billing DBU]")
print("=" * 95)
print(f"{'Operator':<25} {'Ratio':>8} {'95% CI':>20} {'DBU':>10} {'Current':>10} {'Status':>10}")
print("-" * 95)

for op in BLOG_OPERATORS:
    r = ratios.get(op, {})
    ratio_val = r.get("ratio", "—")
    current = CURRENT_WEIGHTS.get(op, {}).get("ratio", "—")
    dbu_val = r.get("dbu")
    dbu_str = f"{dbu_val:.4f}" if isinstance(dbu_val, (int, float)) else "—"

    # CI: apply wall-clock variance scaling to the DBU ratio
    ci_qid = OPERATOR_TO_QID.get(op)
    ci = ci_results.get(ci_qid, {})
    if ci.get("ci_scale_low") is not None and isinstance(ratio_val, (int, float)):
        ci_low = round(ratio_val * ci["ci_scale_low"], 1)
        ci_high = round(ratio_val * ci["ci_scale_high"], 1)
        ci_str = f"[{ci_low}, {ci_high}]"
    else:
        ci_str = "[—, —]"

    # Status vs heuristic
    if isinstance(ratio_val, (int, float)) and isinstance(current, (int, float)):
        diff = abs(ratio_val - current) / current
        status = "✓ close" if diff < 0.25 else ("~ differs" if diff < 1.0 else "⚠ >2×")
    else:
        status = "—"

    print(f"  {op:<23} {str(ratio_val) + '×':>8} {ci_str:>20} {dbu_str:>10} "
          f"{str(current) + '×':>10} {status:>10}")

print("-" * 95)

# Headline numbers
if "SMJ_vs_BHJ_multiplier" in ratios:
    print(f"\n  ★ SortMergeJoin costs {ratios['SMJ_vs_BHJ_multiplier']['ratio']}× a BroadcastHashJoin")

if "BroadcastHashJoin" in ratios and "BroadcastHashJoin_crosscheck" in ratios:
    hint = ratios["BroadcastHashJoin"]["ratio"]
    natural = ratios["BroadcastHashJoin_crosscheck"]["ratio"]
    delta_pct = abs(hint - natural) / natural * 100
    print(f"  ★ BHJ cross-check: hint={hint}× vs natural={natural}× (Δ={delta_pct:.1f}%)")
    if delta_pct > 20:
        print(f"    ⚠️  >20% gap — broadcasting 500 MB may cause memory pressure.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4i. BHJ Cross-Check Detail

# COMMAND ----------

print("BHJ Cross-Check: Q7 (500MB hint) vs Q7B (5MB natural)")
print("-" * 60)

q7_data = analysis.get("Q7", {})
q7b_data = analysis.get("Q7B", {})

if q7_data.get("dbu") and q7b_data.get("dbu"):
    q7_dbu_per_gb = q7_data["dbu"] / LARGE_DELTA_GB
    q7b_dbu_per_gb = q7b_data["dbu"] / LARGE_DELTA_GB
    ratio = q7_dbu_per_gb / q7b_dbu_per_gb

    print(f"  Q7  (500MB broadcast): {q7_dbu_per_gb:.6f} DBU/GB")
    print(f"  Q7B (5MB broadcast):   {q7b_dbu_per_gb:.6f} DBU/GB")
    print(f"  Ratio Q7/Q7B:          {ratio:.2f}×")

    if ratio > 1.3:
        print("  ⚠️  Broadcasting 500 MB is measurably more expensive.")
        print("     Memory pressure likely inflating Q7. Use Q7B ratio for BHJ baseline.")
    else:
        print("  ✅ Consistent — broadcast size doesn't significantly affect per-GB cost.")
else:
    print("  ❌ Missing billing data for Q7 or Q7B.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4j. Compare to CostCoefficients.scala

# COMMAND ----------

print("\nComparison: Empirical (billing DBU) vs Current Heuristic Weights")
print("=" * 80)
print(f"{'Operator':<25} {'Empirical':>10} {'Current':>10} {'Delta':>10} {'Action':>15}")
print("-" * 80)

for op in BLOG_OPERATORS:
    emp = ratios.get(op, {}).get("ratio")
    cur = CURRENT_WEIGHTS.get(op, {}).get("ratio")
    if emp and cur:
        delta = emp - cur
        delta_pct = abs(delta) / cur * 100
        action = "UPDATE ⚠️" if delta_pct > 100 else ("Review" if delta_pct > 25 else "OK")
        print(f"  {op:<23} {emp:>8.1f}× {cur:>8.1f}× {delta:>+8.1f}× {action:>15}")
    else:
        print(f"  {op:<23} {'—':>10} {str(cur)+'×' if cur else '—':>10} {'—':>10} {'N/A':>15}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4k. Save final results

# COMMAND ----------

ratio_rows = []
for op in BLOG_OPERATORS:
    r = ratios.get(op, {})
    cur = CURRENT_WEIGHTS.get(op, {})
    ci_qid = OPERATOR_TO_QID.get(op)
    ci = ci_results.get(ci_qid, {})

    ci_low = None
    ci_high = None
    ratio_val = r.get("ratio")
    if ci.get("ci_scale_low") is not None and ratio_val:
        ci_low = round(ratio_val * ci["ci_scale_low"], 2)
        ci_high = round(ratio_val * ci["ci_scale_high"], 2)

    ratio_rows.append({
        "operator": op,
        "empirical_ratio": ratio_val,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "cv_pct": ci.get("cv"),
        "dbu": r.get("dbu"),
        "dbu_per_gb": r.get("dbu_per_gb"),
        "current_heuristic_ratio": cur.get("ratio"),
        "method": r.get("method", ""),
        "n_runs": analysis.get(ci_qid, {}).get("n_runs"),
    })

ratio_df = spark.createDataFrame(pd.DataFrame(ratio_rows))
ratio_df.write.mode("overwrite").saveAsTable(f"{FQ}.ratio_table_final")
print(f"✅ Ratio table saved to {FQ}.ratio_table_final")
ratio_df.display()

# COMMAND ----------

# Save raw analysis for reproducibility
analysis_rows = []
for qid, data in analysis.items():
    analysis_rows.append({
        "query_id": qid,
        "n_runs": data["n_runs"],
        "dbu": data.get("dbu"),
        "dbu_per_gb": data.get("dbu_per_gb"),
        "wall_mean_s": data.get("wall_mean"),
        "wall_std_s": data.get("wall_std"),
        "read_gb": data.get("read_gb"),
    })

raw_df = spark.createDataFrame(pd.DataFrame(analysis_rows))
raw_df.write.mode("overwrite").saveAsTable(f"{FQ}.analysis_raw")
print(f"✅ Raw analysis saved to {FQ}.analysis_raw")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary & Next Steps
# MAGIC
# MAGIC ### What this notebook produced:
# MAGIC 1. **Ratio table** (`ratio_table_final`) — empirical per-operator cost ratios from billing DBU
# MAGIC 2. **Raw analysis** (`analysis_raw`) — per-query DBU, wall-clock stats, read volumes
# MAGIC 3. **Cross-check validation** — Q7 vs Q7B BHJ consistency
# MAGIC 4. **Heuristic comparison** — empirical vs CostCoefficients.scala weights
# MAGIC
# MAGIC ### Key blog numbers:
# MAGIC ```
# MAGIC Shuffle / Scan:    ratios["Shuffle"]["ratio"]
# MAGIC SMJ / BHJ:         ratios["SMJ_vs_BHJ_multiplier"]["ratio"]
# MAGIC UDF overhead:      ratios["PySpark UDF"]["ratio"]
# MAGIC CSV vs Delta:      ratios["ScanRowBased"]["ratio"]
# MAGIC ```
# MAGIC
# MAGIC ### Next steps:
# MAGIC - [ ] Update `CostCoefficients.scala` with empirical ratios (if >25% delta)
# MAGIC - [ ] Extract methodology to `clusteryield.app/methodology`
# MAGIC - [ ] Write the blog post using the ratio table and worked examples
# MAGIC - [ ] Export raw data CSV for reproducibility (publish with the post)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Cleanup (run when experiment is complete and data is exported)
# MAGIC ```sql
# MAGIC DROP TABLE IF EXISTS test_weights.experiment.large_delta;
# MAGIC DROP TABLE IF EXISTS test_weights.experiment.small_delta;
# MAGIC DROP TABLE IF EXISTS test_weights.experiment.medium_delta;
# MAGIC DROP TABLE IF EXISTS test_weights.experiment.large_csv;
# MAGIC DROP TABLE IF EXISTS test_weights.experiment.sorted_output;
# MAGIC DROP TABLE IF EXISTS test_weights.experiment.experiment_runs;
# MAGIC DROP TABLE IF EXISTS test_weights.experiment.table_sizes;
# MAGIC DROP TABLE IF EXISTS test_weights.experiment.ratio_table_final;
# MAGIC DROP TABLE IF EXISTS test_weights.experiment.analysis_raw;
# MAGIC DROP SCHEMA IF EXISTS test_weights.experiment;
# MAGIC DROP CATALOG IF EXISTS test_weights;
# MAGIC ```
