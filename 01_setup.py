# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Setup — Create Test Tables
# MAGIC
# MAGIC Creates the catalog, schema, and four test tables for the operator cost experiment.
# MAGIC Run once manually before the experiment. Re-run only if you need to resize tables.
# MAGIC
# MAGIC | Table | Target Size | Format | Purpose |
# MAGIC |-------|-------------|--------|---------|
# MAGIC | `large_delta` | ~100 GB | Delta | Primary scan target, partitioned by date |
# MAGIC | `small_delta` | ~5 MB | Delta | BHJ cross-check, CROSS JOIN |
# MAGIC | `medium_delta` | ~500 MB | Delta | SMJ/BHJ comparison |
# MAGIC | `large_csv` | ~100 GB | CSV | Row-based scan baseline |

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create catalog & schema

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
print(f"Using: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0 — Validate measurement path (canary query)
# MAGIC
# MAGIC Run a cheap query and confirm you can see it in both `system.billing.usage`
# MAGIC and `system.query.history`. Record the billing lag.

# COMMAND ----------

import time
from datetime import datetime

canary_start = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
print(f"Canary start: {canary_start}")

spark.sql("SELECT COUNT(*) AS cnt FROM range(1000000)").show()

canary_end = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
print(f"Canary end: {canary_end}")
print(f"\nWait ~{BILLING_LAG_MINUTES} min, then run the next two cells.")

# COMMAND ----------

# Check system.query.history (should appear quickly)
spark.sql(f"""
    SELECT statement_id, statement_text, total_duration_ms,
           total_task_duration_ms, read_bytes, execution_status
    FROM system.query.history
    WHERE start_time >= '{canary_start}'
      AND execution_status = 'FINISHED'
    ORDER BY start_time DESC
    LIMIT 10
""").display()

# COMMAND ----------

# Check system.billing.usage (may take 15-30+ min to appear)
spark.sql(f"""
    SELECT usage_date, sku_name, usage_quantity AS dbu,
           usage_metadata.job_id, usage_metadata.job_run_id,
           usage_start_time, usage_end_time
    FROM system.billing.usage
    WHERE usage_start_time >= '{canary_start}'
      AND sku_name LIKE '%SERVERLESS%'
    ORDER BY usage_start_time DESC
    LIMIT 10
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1a. Large Delta Table (~100 GB)

# COMMAND ----------

print(f"Creating large_delta with {ROW_COUNT_LARGE:,} rows...")
print("This will take 15-30 minutes on Serverless.")

spark.sql(f"""
CREATE OR REPLACE TABLE {FQ}.large_delta
USING DELTA
PARTITIONED BY (date)
AS
WITH base AS (
    SELECT
        id,
        CONCAT('key_', CAST(id % 10000000 AS STRING)) AS key,
        RAND(id)       AS val1,  RAND(id + 1)  AS val2,
        RAND(id + 2)   AS val3,  RAND(id + 3)  AS val4,
        RAND(id + 4)   AS val5,  RAND(id + 5)  AS val6,
        RAND(id + 6)   AS val7,  RAND(id + 7)  AS val8,
        RAND(id + 8)   AS val9,  RAND(id + 9)  AS val10,
        RAND(id + 10)  AS val11, RAND(id + 11) AS val12,
        RAND(id + 12)  AS val13, RAND(id + 13) AS val14,
        RAND(id + 14)  AS val15, RAND(id + 15) AS val16,
        RAND(id + 16)  AS val17, RAND(id + 17) AS val18,
        RAND(id + 18)  AS val19, RAND(id + 19) AS val20,
        RAND(id + 20)  AS val21, RAND(id + 21) AS val22,
        RAND(id + 22)  AS val23, RAND(id + 23) AS val24,
        RAND(id + 24)  AS val25, RAND(id + 25) AS val26,
        RAND(id + 26)  AS val27, RAND(id + 27) AS val28,
        RAND(id + 28)  AS val29, RAND(id + 29) AS val30,
        RAND(id + 30)  AS val31, RAND(id + 31) AS val32,
        RAND(id + 32)  AS val33, RAND(id + 33) AS val34,
        RAND(id + 34)  AS val35, RAND(id + 35) AS val36,
        RAND(id + 36)  AS val37, RAND(id + 37) AS val38,
        RAND(id + 38)  AS val39, RAND(id + 39) AS val40,
        RAND(id + 40)  AS val41, RAND(id + 41) AS val42,
        RAND(id + 42)  AS val43, RAND(id + 43) AS val44,
        RAND(id + 44)  AS val45, RAND(id + 45) AS val46,
        RAND(id + 46)  AS val47, RAND(id + 47) AS val48,
        CONCAT('str_', CAST(RAND(id + 100) * 10000 AS INT)) AS str1,
        CONCAT('str_', CAST(RAND(id + 101) * 10000 AS INT)) AS str2,
        CONCAT('str_', CAST(RAND(id + 102) * 10000 AS INT)) AS str3,
        CONCAT('str_', CAST(RAND(id + 103) * 10000 AS INT)) AS str4
    FROM RANGE(0, {ROW_COUNT_LARGE})
),
dates AS (
    SELECT explode(sequence(
        TO_DATE('{DATE_START}'),
        TO_DATE('{DATE_END}'),
        INTERVAL 1 DAY
    )) AS date
)
SELECT
    b.id, b.key,
    b.val1, b.val2, b.val3, b.val4, b.val5, b.val6, b.val7, b.val8,
    b.val9, b.val10, b.val11, b.val12, b.val13, b.val14, b.val15, b.val16,
    b.val17, b.val18, b.val19, b.val20, b.val21, b.val22, b.val23, b.val24,
    b.val25, b.val26, b.val27, b.val28, b.val29, b.val30, b.val31, b.val32,
    b.val33, b.val34, b.val35, b.val36, b.val37, b.val38, b.val39, b.val40,
    b.val41, b.val42, b.val43, b.val44, b.val45, b.val46, b.val47, b.val48,
    b.str1, b.str2, b.str3, b.str4,
    d.date
FROM base b
JOIN dates d ON (b.id % 365) = DATEDIFF(d.date, '{DATE_START}')
""")

print("✅ large_delta created.")

# COMMAND ----------

# Check size
actual_gb = spark.sql(f"""
    SELECT ROUND(sizeInBytes / (1024*1024*1024), 2) AS size_gb
    FROM (DESCRIBE DETAIL {FQ}.large_delta)
""").collect()[0]["size_gb"]

print(f"large_delta: {actual_gb} GB (target: ~{LARGE_TABLE_TARGET_GB} GB)")
if actual_gb < LARGE_TABLE_TARGET_GB * 0.5:
    print("⚠️  Too small. Increase ROW_COUNT_LARGE in 00_config and re-run.")
elif actual_gb > LARGE_TABLE_TARGET_GB * 2:
    print("⚠️  Too large. Decrease ROW_COUNT_LARGE in 00_config and re-run.")
else:
    print("✅ Size is in acceptable range.")

spark.sql(f"DESCRIBE DETAIL {FQ}.large_delta").select(
    "name", "format", "numFiles", "sizeInBytes"
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1b. Small Delta Table (~5 MB)

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {FQ}.small_delta
USING DELTA
AS
SELECT DISTINCT key, val1 AS lookup_val
FROM {FQ}.large_delta
LIMIT 50000
""")

print("✅ small_delta created.")
spark.sql(f"DESCRIBE DETAIL {FQ}.small_delta").select(
    "name", "format", "numFiles", "sizeInBytes"
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1c. Medium Delta Table (~500 MB)

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {FQ}.medium_delta
USING DELTA
AS
SELECT key, val1 AS lookup_val, val2, val3, val4, val5
FROM {FQ}.large_delta
LIMIT {MEDIUM_ROW_LIMIT}
""")

medium_mb = spark.sql(f"""
    SELECT ROUND(sizeInBytes / (1024*1024), 2) AS size_mb
    FROM (DESCRIBE DETAIL {FQ}.medium_delta)
""").collect()[0]["size_mb"]

print(f"medium_delta: {medium_mb} MB (target: ~{MEDIUM_TABLE_TARGET_MB} MB)")
spark.sql(f"DESCRIBE DETAIL {FQ}.medium_delta").select(
    "name", "format", "numFiles", "sizeInBytes"
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1d. Large CSV Table (~100 GB, row-based)

# COMMAND ----------

print("Creating large_csv — this will be slow...")

spark.sql(f"""
CREATE OR REPLACE TABLE {FQ}.large_csv
USING CSV
AS
SELECT id, key, val1, val2, val3, val4, val5, val6, val7, val8,
       val9, val10, val11, val12, val13, val14, val15, val16,
       val17, val18, val19, val20, val21, val22, val23, val24,
       val25, val26, val27, val28, val29, val30, val31, val32,
       val33, val34, val35, val36, val37, val38, val39, val40,
       val41, val42, val43, val44, val45, val46, val47, val48,
       str1, str2, str3, str4, date
FROM {FQ}.large_delta
""")

print("✅ large_csv created.")
spark.sql(f"DESCRIBE DETAIL {FQ}.large_csv").select(
    "name", "format", "numFiles", "sizeInBytes"
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1e. ANALYZE all Delta tables

# COMMAND ----------

for tbl in ["large_delta", "small_delta", "medium_delta"]:
    print(f"Analyzing {tbl}...")
    spark.sql(f"ANALYZE TABLE {FQ}.{tbl} COMPUTE STATISTICS FOR ALL COLUMNS")
    print(f"  ✅ Done.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1f. Create experiment_runs log table

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {FQ}.experiment_runs (
    query_id STRING,
    run_number INT,
    job_run_id STRING,
    start_ts STRING,
    end_ts STRING,
    elapsed_seconds DOUBLE,
    plan_text STRING
)
USING DELTA
""")
print(f"✅ {FQ}.experiment_runs table ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1g. Save table sizes for the analysis notebook

# COMMAND ----------

import json

table_sizes = {}
for tbl in ["large_delta", "small_delta", "medium_delta", "large_csv"]:
    detail = spark.sql(f"DESCRIBE DETAIL {FQ}.{tbl}").collect()[0]
    size_bytes = detail["sizeInBytes"]
    size_gb = round(size_bytes / (1024**3), 4)
    table_sizes[tbl] = {"bytes": size_bytes, "gb": size_gb}
    print(f"  {tbl}: {size_gb} GB")

# Store as a single-row Delta table for the analysis notebook
spark.sql(f"""
CREATE OR REPLACE TABLE {FQ}.table_sizes
AS SELECT
    '{json.dumps(table_sizes)}' AS sizes_json
""")
print(f"\n✅ Table sizes saved to {FQ}.table_sizes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC All tables created. Verify sizes above, then proceed to **02_validate_plans**.
