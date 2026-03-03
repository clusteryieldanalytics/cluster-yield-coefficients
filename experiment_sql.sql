-- =============================================================================
-- Operator Cost Coefficient Experiment — Manual SQL Queries
-- =============================================================================
-- Use this file if running experiments manually in a Databricks SQL editor
-- rather than through the Python notebook harness.
--
-- INSTRUCTIONS:
-- 1. Run §0 (Setup) once.
-- 2. Run each experiment query 5× in sequence.
-- 3. After each query, run the EXPLAIN FORMATTED variant to confirm plan shape.
-- 4. Wait ≥10 min after the last query, then run §3 (Billing Collection).
-- 5. Record results in the manual template or export from billing query.
--
-- SERVERLESS NOTES:
-- - SET spark.sql.adaptive.enabled = false may not work on serverless.
--   Queries use /*+ MERGE */ and /*+ BROADCAST */ hints instead.
-- - Always verify plan shape via EXPLAIN FORMATTED after running.
-- =============================================================================


-- =============================================================================
-- §0 — SETUP
-- =============================================================================

CREATE CATALOG IF NOT EXISTS main;
CREATE SCHEMA IF NOT EXISTS main.test_weights;

-- Table A: Large columnar (~100 GB Delta)
CREATE OR REPLACE TABLE main.test_weights.large_parquet
USING DELTA
PARTITIONED BY (date)
AS
SELECT
    id,
    uuid() AS key,
    CAST(rand() * 1000 AS DOUBLE) AS val1,
    CAST(rand() * 1000 AS DOUBLE) AS val2,
    CAST(rand() * 1000 AS DOUBLE) AS val3,
    CAST(rand() * 1000 AS DOUBLE) AS val4,
    CAST(rand() * 1000 AS DOUBLE) AS val5,
    CAST(rand() * 1000 AS DOUBLE) AS val6,
    CAST(rand() * 1000 AS DOUBLE) AS val7,
    CAST(rand() * 1000 AS DOUBLE) AS val8,
    CAST(rand() * 1000 AS DOUBLE) AS val9,
    CAST(rand() * 1000 AS DOUBLE) AS val10,
    CAST(rand() * 1000 AS DOUBLE) AS val11,
    CAST(rand() * 1000 AS DOUBLE) AS val12,
    CAST(rand() * 1000 AS DOUBLE) AS val13,
    CAST(rand() * 1000 AS DOUBLE) AS val14,
    CAST(rand() * 1000 AS DOUBLE) AS val15,
    CAST(rand() * 1000 AS DOUBLE) AS val16,
    CAST(rand() * 1000 AS DOUBLE) AS val17,
    CAST(rand() * 1000 AS DOUBLE) AS val18,
    CAST(rand() * 1000 AS DOUBLE) AS val19,
    CAST(rand() * 1000 AS DOUBLE) AS val20,
    CAST(rand() * 1000 AS DOUBLE) AS val21,
    CAST(rand() * 1000 AS DOUBLE) AS val22,
    CAST(rand() * 1000 AS DOUBLE) AS val23,
    CAST(rand() * 1000 AS DOUBLE) AS val24,
    CAST(rand() * 1000 AS DOUBLE) AS val25,
    CAST(rand() * 1000 AS DOUBLE) AS val26,
    CAST(rand() * 1000 AS DOUBLE) AS val27,
    CAST(rand() * 1000 AS DOUBLE) AS val28,
    CAST(rand() * 1000 AS DOUBLE) AS val29,
    CAST(rand() * 1000 AS DOUBLE) AS val30,
    CAST(rand() * 1000 AS DOUBLE) AS val31,
    CAST(rand() * 1000 AS DOUBLE) AS val32,
    CAST(rand() * 1000 AS DOUBLE) AS val33,
    CAST(rand() * 1000 AS DOUBLE) AS val34,
    CAST(rand() * 1000 AS DOUBLE) AS val35,
    CAST(rand() * 1000 AS DOUBLE) AS val36,
    CAST(rand() * 1000 AS DOUBLE) AS val37,
    CAST(rand() * 1000 AS DOUBLE) AS val38,
    CAST(rand() * 1000 AS DOUBLE) AS val39,
    CAST(rand() * 1000 AS DOUBLE) AS val40,
    CAST(rand() * 1000 AS DOUBLE) AS val41,
    CAST(rand() * 1000 AS DOUBLE) AS val42,
    CAST(rand() * 1000 AS DOUBLE) AS val43,
    CAST(rand() * 1000 AS DOUBLE) AS val44,
    CAST(rand() * 1000 AS DOUBLE) AS val45,
    CAST(rand() * 1000 AS DOUBLE) AS val46,
    CAST(rand() * 1000 AS DOUBLE) AS val47,
    CAST(rand() * 1000 AS DOUBLE) AS val48,
    dt AS date
FROM RANGE(0, 500000000) AS t(id)
CROSS JOIN (
    SELECT explode(sequence(
        DATE '2025-01-01',
        DATE '2025-12-31',
        INTERVAL 1 DAY
    )) AS dt
);

-- Verify size
DESCRIBE DETAIL main.test_weights.large_parquet;

-- Table B: Small Delta (~1 GB, join small-side)
CREATE OR REPLACE TABLE main.test_weights.small_delta
USING DELTA
AS SELECT DISTINCT key, val1 AS lookup_val
FROM main.test_weights.large_parquet
LIMIT 10000000;

DESCRIBE DETAIL main.test_weights.small_delta;

-- Table C: Large CSV (row-based scan comparison)
CREATE OR REPLACE TABLE main.test_weights.large_csv
USING CSV
AS SELECT * FROM main.test_weights.large_parquet;


-- =============================================================================
-- §1 — EXPERIMENT QUERIES
-- =============================================================================
-- Run each query 5 times. Note the start/end time for billing lookup.
-- After each, run the corresponding EXPLAIN to verify plan shape.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- Q1 — Pure columnar scan (narrow read) — BASELINE
-- Expected plan: FileScan parquet → HashAggregate (NO Exchange, NO Sort)
-- ---------------------------------------------------------------------------
SELECT COUNT(*), SUM(val1)
FROM main.test_weights.large_parquet
WHERE date BETWEEN '2025-01-01' AND '2025-12-31';

-- Plan check:
EXPLAIN FORMATTED
SELECT COUNT(*), SUM(val1)
FROM main.test_weights.large_parquet
WHERE date BETWEEN '2025-01-01' AND '2025-12-31';


-- ---------------------------------------------------------------------------
-- Q2 — Wide columnar scan (all 48 value columns)
-- Expected plan: same as Q1 but reading all columns
-- ---------------------------------------------------------------------------
SELECT COUNT(*),
       SUM(val1),  SUM(val2),  SUM(val3),  SUM(val4),  SUM(val5),
       SUM(val6),  SUM(val7),  SUM(val8),  SUM(val9),  SUM(val10),
       SUM(val11), SUM(val12), SUM(val13), SUM(val14), SUM(val15),
       SUM(val16), SUM(val17), SUM(val18), SUM(val19), SUM(val20),
       SUM(val21), SUM(val22), SUM(val23), SUM(val24), SUM(val25),
       SUM(val26), SUM(val27), SUM(val28), SUM(val29), SUM(val30),
       SUM(val31), SUM(val32), SUM(val33), SUM(val34), SUM(val35),
       SUM(val36), SUM(val37), SUM(val38), SUM(val39), SUM(val40),
       SUM(val41), SUM(val42), SUM(val43), SUM(val44), SUM(val45),
       SUM(val46), SUM(val47), SUM(val48)
FROM main.test_weights.large_parquet
WHERE date BETWEEN '2025-01-01' AND '2025-12-31';


-- ---------------------------------------------------------------------------
-- Q3 — Row-based scan (CSV)
-- Expected plan: FileScan csv → HashAggregate
-- ---------------------------------------------------------------------------
SELECT COUNT(*), SUM(val1)
FROM main.test_weights.large_csv;


-- ---------------------------------------------------------------------------
-- Q4 — Scan + shuffle (GROUP BY)
-- Expected plan: FileScan → partial HashAggregate → Exchange(hashpartitioning) → final HashAggregate
-- Should contain Exchange but NOT Sort
-- ---------------------------------------------------------------------------
SELECT key, COUNT(*), SUM(val1)
FROM main.test_weights.large_parquet
WHERE date BETWEEN '2025-01-01' AND '2025-12-31'
GROUP BY key;

-- Plan check (look for Exchange, no Sort):
EXPLAIN FORMATTED
SELECT key, COUNT(*), SUM(val1)
FROM main.test_weights.large_parquet
WHERE date BETWEEN '2025-01-01' AND '2025-12-31'
GROUP BY key;


-- ---------------------------------------------------------------------------
-- Q5b — Scan + shuffle + sort (ORDER BY, materialized via CTAS)
-- Expected plan: FileScan → Exchange → Sort → write
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS main.test_weights.sorted_output;
CREATE TABLE main.test_weights.sorted_output
AS SELECT key, val1
FROM main.test_weights.large_parquet
WHERE date BETWEEN '2025-01-01' AND '2025-12-31'
ORDER BY key;

-- Plan check:
EXPLAIN FORMATTED
SELECT key, val1
FROM main.test_weights.large_parquet
WHERE date BETWEEN '2025-01-01' AND '2025-12-31'
ORDER BY key;


-- ---------------------------------------------------------------------------
-- Q6 — SortMergeJoin (forced via MERGE hint)
-- Expected plan: must contain SortMergeJoin, must NOT contain BroadcastHashJoin
-- NOTE: If /*+ MERGE */ is ignored on your serverless runtime, try also
--       setting spark.sql.autoBroadcastJoinThreshold = -1 (may or may not work).
-- ---------------------------------------------------------------------------
SELECT /*+ MERGE(b) */ COUNT(*), SUM(a.val1)
FROM main.test_weights.large_parquet a
JOIN main.test_weights.small_delta b ON a.key = b.key
WHERE a.date BETWEEN '2025-01-01' AND '2025-12-31';

-- Plan check (MUST show SortMergeJoin):
EXPLAIN FORMATTED
SELECT /*+ MERGE(b) */ COUNT(*), SUM(a.val1)
FROM main.test_weights.large_parquet a
JOIN main.test_weights.small_delta b ON a.key = b.key
WHERE a.date BETWEEN '2025-01-01' AND '2025-12-31';

-- FALLBACK if MERGE hint is ignored: try SET + hint combo
-- SET spark.sql.autoBroadcastJoinThreshold = -1;
-- SET spark.sql.adaptive.enabled = false;
-- Then re-run Q6 and re-check EXPLAIN.


-- ---------------------------------------------------------------------------
-- Q7 — BroadcastHashJoin (forced via BROADCAST hint)
-- Expected plan: must contain BroadcastHashJoin, must NOT contain SortMergeJoin
-- ---------------------------------------------------------------------------
SELECT /*+ BROADCAST(b) */ COUNT(*), SUM(a.val1)
FROM main.test_weights.large_parquet a
JOIN main.test_weights.small_delta b ON a.key = b.key
WHERE a.date BETWEEN '2025-01-01' AND '2025-12-31';

-- Plan check (MUST show BroadcastHashJoin):
EXPLAIN FORMATTED
SELECT /*+ BROADCAST(b) */ COUNT(*), SUM(a.val1)
FROM main.test_weights.large_parquet a
JOIN main.test_weights.small_delta b ON a.key = b.key
WHERE a.date BETWEEN '2025-01-01' AND '2025-12-31';


-- ---------------------------------------------------------------------------
-- Q8 — CartesianProduct (small × small ONLY)
-- Expected plan: CartesianProduct or BroadcastNestedLoopJoin
-- WARNING: Do NOT run this on large × large.
-- ---------------------------------------------------------------------------
SELECT COUNT(*)
FROM main.test_weights.small_delta a
CROSS JOIN main.test_weights.small_delta b;

-- Plan check:
EXPLAIN FORMATTED
SELECT COUNT(*)
FROM main.test_weights.small_delta a
CROSS JOIN main.test_weights.small_delta b;


-- =============================================================================
-- §2 — Q6 ALTERNATIVE: SHUFFLE_HASH hint if MERGE is not producing SMJ
-- =============================================================================
-- Some serverless runtimes may prefer ShuffledHashJoin over SortMergeJoin.
-- If Q6's EXPLAIN shows ShuffledHashJoin instead of SortMergeJoin, that's
-- useful data too — record it as ShuffledHashJoin and note the difference.
-- ---------------------------------------------------------------------------

SELECT /*+ SHUFFLE_HASH(b) */ COUNT(*), SUM(a.val1)
FROM main.test_weights.large_parquet a
JOIN main.test_weights.small_delta b ON a.key = b.key
WHERE a.date BETWEEN '2025-01-01' AND '2025-12-31';

EXPLAIN FORMATTED
SELECT /*+ SHUFFLE_HASH(b) */ COUNT(*), SUM(a.val1)
FROM main.test_weights.large_parquet a
JOIN main.test_weights.small_delta b ON a.key = b.key
WHERE a.date BETWEEN '2025-01-01' AND '2025-12-31';


-- =============================================================================
-- §3 — BILLING COLLECTION QUERIES
-- =============================================================================
-- Run these ≥10 min after the last experiment query.
-- Replace the timestamps with your actual experiment window.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- 3a — All experiment queries with billing data
-- ---------------------------------------------------------------------------
SELECT
    q.statement_id                              AS query_id,
    SUBSTRING(q.statement_text, 1, 120)         AS query_preview,
    q.execution_status,
    q.total_duration_ms,
    q.rows_produced,
    q.read_bytes,
    ROUND(q.read_bytes / (1024*1024*1024), 2)   AS read_gb,
    q.shuffle_write_bytes,
    ROUND(q.shuffle_write_bytes / (1024*1024*1024), 2) AS shuffle_write_gb,
    u.usage_quantity                             AS dbu_consumed,
    q.start_time
FROM system.query.history q
LEFT JOIN system.billing.usage u
    ON u.usage_metadata.statement_id = q.statement_id
WHERE q.execution_status = 'FINISHED'
  AND q.start_time >= '2025-XX-XX 00:00:00'     -- ← REPLACE with experiment start
  AND q.start_time <= '2025-XX-XX 23:59:59'     -- ← REPLACE with experiment end
  AND q.statement_text LIKE '%test_weights%'
ORDER BY q.start_time;


-- ---------------------------------------------------------------------------
-- 3b — Aggregate: mean DBU per query label
-- ---------------------------------------------------------------------------
-- This requires you to tag which rows are Q1, Q2, etc.
-- Easiest approach: run each query type in sequence and use time ordering.
--
-- Alternatively, if you add a comment tag to each query:
--   SELECT /* Q1 */ COUNT(*), SUM(val1) FROM ...
-- Then you can filter on the comment:

SELECT
    CASE
        WHEN q.statement_text LIKE '%/* Q1 */%'  THEN 'Q1'
        WHEN q.statement_text LIKE '%/* Q2 */%'  THEN 'Q2'
        WHEN q.statement_text LIKE '%/* Q3 */%'  THEN 'Q3'
        WHEN q.statement_text LIKE '%/* Q4 */%'  THEN 'Q4'
        WHEN q.statement_text LIKE '%/* Q5b */%' THEN 'Q5b'
        WHEN q.statement_text LIKE '%/* Q6 */%'  THEN 'Q6'
        WHEN q.statement_text LIKE '%/* Q7 */%'  THEN 'Q7'
        WHEN q.statement_text LIKE '%/* Q8 */%'  THEN 'Q8'
        WHEN q.statement_text LIKE '%/* Q9 */%'  THEN 'Q9'
        ELSE 'UNKNOWN'
    END AS query_label,
    COUNT(*)                                             AS run_count,
    ROUND(AVG(u.usage_quantity), 6)                      AS mean_dbu,
    ROUND(STDDEV(u.usage_quantity), 6)                   AS stddev_dbu,
    ROUND(AVG(q.read_bytes / (1024*1024*1024)), 2)       AS mean_read_gb,
    ROUND(AVG(q.shuffle_write_bytes / (1024*1024*1024)), 2) AS mean_shuffle_write_gb,
    ROUND(AVG(q.total_duration_ms), 0)                   AS mean_duration_ms
FROM system.query.history q
LEFT JOIN system.billing.usage u
    ON u.usage_metadata.statement_id = q.statement_id
WHERE q.execution_status = 'FINISHED'
  AND q.start_time >= '2025-XX-XX 00:00:00'     -- ← REPLACE
  AND q.start_time <= '2025-XX-XX 23:59:59'     -- ← REPLACE
  AND q.statement_text LIKE '%test_weights%'
GROUP BY 1
ORDER BY 1;


-- ---------------------------------------------------------------------------
-- 3c — Serverless DBU price check
-- ---------------------------------------------------------------------------
SELECT
    sku_name,
    pricing.default     AS list_price_per_dbu,
    pricing.effective   AS effective_price_per_dbu
FROM system.billing.list_prices
WHERE sku_name LIKE '%SERVERLESS%'
  AND price_start_time = (
      SELECT MAX(price_start_time)
      FROM system.billing.list_prices
      WHERE sku_name LIKE '%SERVERLESS%'
  );


-- ---------------------------------------------------------------------------
-- 3d — Coefficient of variation check (flag queries needing more runs)
-- ---------------------------------------------------------------------------
SELECT
    query_label,
    run_count,
    mean_dbu,
    stddev_dbu,
    ROUND(stddev_dbu / mean_dbu * 100, 1) AS cv_pct,
    CASE
        WHEN stddev_dbu / mean_dbu > 0.20 THEN '⚠️  CV > 20% — increase to 10 runs'
        ELSE '✓ stable'
    END AS recommendation
FROM (
    -- Paste the §3b query as a subquery here, or use a CTE
    SELECT
        CASE
            WHEN q.statement_text LIKE '%/* Q1 */%'  THEN 'Q1'
            WHEN q.statement_text LIKE '%/* Q2 */%'  THEN 'Q2'
            WHEN q.statement_text LIKE '%/* Q3 */%'  THEN 'Q3'
            WHEN q.statement_text LIKE '%/* Q4 */%'  THEN 'Q4'
            WHEN q.statement_text LIKE '%/* Q5b */%' THEN 'Q5b'
            WHEN q.statement_text LIKE '%/* Q6 */%'  THEN 'Q6'
            WHEN q.statement_text LIKE '%/* Q7 */%'  THEN 'Q7'
            WHEN q.statement_text LIKE '%/* Q8 */%'  THEN 'Q8'
            WHEN q.statement_text LIKE '%/* Q9 */%'  THEN 'Q9'
            ELSE 'UNKNOWN'
        END AS query_label,
        COUNT(*)                          AS run_count,
        AVG(u.usage_quantity)             AS mean_dbu,
        STDDEV(u.usage_quantity)          AS stddev_dbu
    FROM system.query.history q
    LEFT JOIN system.billing.usage u
        ON u.usage_metadata.statement_id = q.statement_id
    WHERE q.execution_status = 'FINISHED'
      AND q.start_time >= '2025-XX-XX 00:00:00'
      AND q.start_time <= '2025-XX-XX 23:59:59'
      AND q.statement_text LIKE '%test_weights%'
    GROUP BY 1
)
WHERE mean_dbu > 0
ORDER BY cv_pct DESC;


-- =============================================================================
-- §4 — CLEANUP (run when done)
-- =============================================================================

-- DROP TABLE IF EXISTS main.test_weights.large_parquet;
-- DROP TABLE IF EXISTS main.test_weights.small_delta;
-- DROP TABLE IF EXISTS main.test_weights.large_csv;
-- DROP TABLE IF EXISTS main.test_weights.sorted_output;
-- DROP SCHEMA IF EXISTS main.test_weights;