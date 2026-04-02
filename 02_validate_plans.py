# Databricks notebook source
# MAGIC %md
# MAGIC # 2. Validate Plan Shapes
# MAGIC
# MAGIC **Run interactively before the experiment.** This notebook follows the SMJ/BHJ
# MAGIC sizing decision tree from `blog-post-2-strategy-final.md` and confirms that
# MAGIC each query produces the expected physical plan.
# MAGIC
# MAGIC **Record your decisions at the bottom before proceeding to the workflow.**

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import time

def get_plan_shape(query_sql):
    """Return EXPLAIN FORMATTED output."""
    rows = spark.sql(f"EXPLAIN FORMATTED {query_sql}").collect()
    return rows[0][0]

def check_plan_for_operator(plan_text, operator_name):
    return operator_name in plan_text

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2a. Q6 — Should be SortMergeJoin (no hint)

# COMMAND ----------

q6_plan = get_plan_shape(f"""
    SELECT COUNT(*), SUM(a.val1)
    FROM {FQ}.large_delta a
    JOIN {FQ}.medium_delta b ON a.key = b.key
    WHERE a.date BETWEEN '{DATE_START}' AND '{DATE_END}'
""")

print("=" * 80)
print("Q6 PLAN (no hint — expect SortMergeJoin):")
print("=" * 80)
print(q6_plan)
print()

has_smj = check_plan_for_operator(q6_plan, "SortMergeJoin")
has_bhj = check_plan_for_operator(q6_plan, "BroadcastHashJoin")

if has_smj:
    print("✅ Q6 → SortMergeJoin. Proceed with natural plan.")
elif has_bhj:
    print("⚠️  Q6 → BroadcastHashJoin (AQE promoted).")
    print("    Options:")
    print("    1. Increase medium_delta to 1 GB (re-run 01_setup)")
    print("    2. Use /*+ SHUFFLE_MERGE(b) */ hint — set USE_SMJ_HINT = True below")
else:
    print("❓ Unexpected plan. Review manually.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2b. Q7 — Should be BroadcastHashJoin (BROADCAST hint)

# COMMAND ----------

q7_plan = get_plan_shape(f"""
    SELECT /*+ BROADCAST(b) */ COUNT(*), SUM(a.val1)
    FROM {FQ}.large_delta a
    JOIN {FQ}.medium_delta b ON a.key = b.key
    WHERE a.date BETWEEN '{DATE_START}' AND '{DATE_END}'
""")

print("=" * 80)
print("Q7 PLAN (BROADCAST hint — expect BroadcastHashJoin):")
print("=" * 80)
print(q7_plan)
print()

if check_plan_for_operator(q7_plan, "BroadcastHashJoin"):
    print("✅ Q7 → BroadcastHashJoin. Hint respected.")
else:
    print("⚠️  Q7 did not produce BroadcastHashJoin.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2c. Q7 — Test execution (check for OOM)

# COMMAND ----------

print("Test-executing Q7 with BROADCAST hint...")
try:
    t0 = time.time()
    spark.sql(f"""
        SELECT /*+ BROADCAST(b) */ COUNT(*), SUM(a.val1)
        FROM {FQ}.large_delta a
        JOIN {FQ}.medium_delta b ON a.key = b.key
        WHERE a.date BETWEEN '{DATE_START}' AND '{DATE_END}'
    """).collect()
    elapsed = time.time() - t0
    print(f"✅ Q7 executed successfully in {elapsed:.1f}s. No OOM.")
except Exception as e:
    print(f"❌ Q7 FAILED: {e}")
    print("   → Reduce medium_delta to 200 MB, or use hint-controlled fallback.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2d. Q7b — Should be BroadcastHashJoin (natural, small table)

# COMMAND ----------

q7b_plan = get_plan_shape(f"""
    SELECT COUNT(*), SUM(a.val1)
    FROM {FQ}.large_delta a
    JOIN {FQ}.small_delta b ON a.key = b.key
    WHERE a.date BETWEEN '{DATE_START}' AND '{DATE_END}'
""")

print("=" * 80)
print("Q7b PLAN (no hint, small table — expect BroadcastHashJoin):")
print("=" * 80)
print(q7b_plan)
print()

if check_plan_for_operator(q7b_plan, "BroadcastHashJoin"):
    print("✅ Q7b → BroadcastHashJoin (natural). Good for cross-check.")
else:
    print("⚠️  Q7b did not produce BHJ. small_delta may be too large.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2e. Q8 — Should be CartesianProduct or BroadcastNestedLoopJoin

# COMMAND ----------

q8_plan = get_plan_shape(f"""
    SELECT COUNT(*)
    FROM {FQ}.small_delta a
    CROSS JOIN {FQ}.small_delta b
""")

print("=" * 80)
print("Q8 PLAN (CROSS JOIN — expect CartesianProduct/BNLJ):")
print("=" * 80)
print(q8_plan)
print()

has_cartesian = check_plan_for_operator(q8_plan, "CartesianProduct")
has_bnlj = check_plan_for_operator(q8_plan, "BroadcastNestedLoopJoin")
if has_cartesian or has_bnlj:
    op = "CartesianProduct" if has_cartesian else "BroadcastNestedLoopJoin"
    print(f"✅ Q8 → {op}.")
else:
    print("❓ Unexpected plan for CROSS JOIN.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Record
# MAGIC
# MAGIC **Fill in before proceeding to the workflow:**
# MAGIC
# MAGIC | Check | Result | Action |
# MAGIC |-------|--------|--------|
# MAGIC | Q6 plan shape | _________ | natural / SHUFFLE_MERGE hint |
# MAGIC | Q7 execution | _________ | success / OOM |
# MAGIC | Q7b plan shape | _________ | natural BHJ confirmed |
# MAGIC | Q8 plan shape | _________ | CartesianProduct / BNLJ |
# MAGIC | Final medium_delta size | _________ MB | |
# MAGIC | Hint strategy | _________ | natural / hint-controlled / fallback |
# MAGIC
# MAGIC **If Q6 was BHJ:** Set `USE_SMJ_HINT` to `"true"` in the workflow task parameters for Q6.
# MAGIC
# MAGIC **If Q7 OOM'd:** Reduce medium_delta size in `00_config`, re-run `01_setup`, re-validate.
# MAGIC
# MAGIC **Next:** Create the Databricks Workflow using `03_run_query` as the task notebook.
