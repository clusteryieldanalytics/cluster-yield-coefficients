# Coefficient Transparency Experiment — Harness Files

## Files

| File | Purpose |
|------|---------|
| `coefficient_experiment_harness.py` | Full Python notebook for Databricks. Automated execution with plan validation, billing collection, and ratio analysis. Import as a Databricks notebook. |
| `coefficient_experiment_manual.sql` | Standalone SQL queries for manual execution in a SQL editor. Copy-paste each query, run 5×, then use the billing queries to collect results. |

## Two Workflows

### Option A: Automated (Python notebook)

1. Import `coefficient_experiment_harness.py` as a Databricks notebook
2. Edit §0 config (catalog, schema, row count)
3. Run §1 to create test tables (~100 GB, takes a while)
4. Run §2 to execute the full query series with plan validation
5. Wait ≥10 min, then run §3 to collect billing data
6. Run §4 to compute ratios and compare to current coefficients

### Option B: Manual (SQL editor)

1. Run the setup queries from `coefficient_experiment_manual.sql` §0
2. Run each experiment query 5× with a `/* Q1 */` style comment tag
3. After each, `EXPLAIN FORMATTED` to verify plan shape
4. Wait ≥10 min, then run the §3 billing queries to pull results
5. Use the §3d CV check to see if any queries need more runs

## Serverless Considerations

**Config immutability:** On Databricks Serverless, `SET spark.sql.*` may be read-only. The harness uses these strategies instead:

| Need | Classic approach | Serverless approach |
|------|-----------------|---------------------|
| Force SortMergeJoin | `SET autoBroadcastJoinThreshold = -1` | `/*+ MERGE(b) */` hint |
| Force BroadcastHashJoin | `SET autoBroadcastJoinThreshold = MAX` | `/*+ BROADCAST(b) */` hint |
| Force ShuffledHashJoin | `SET preferSortMergeJoin = false` | `/*+ SHUFFLE_HASH(b) */` hint |
| Disable AQE | `SET adaptive.enabled = false` | Post-hoc plan validation; discard runs where AQE rewrote the plan |
| Force sort materialization | N/A | CTAS (`CREATE TABLE ... AS ... ORDER BY`) |

**Plan validation:** Every run checks the physical plan for required operators (e.g., Q6 must contain `SortMergeJoin`) and forbidden operators (e.g., Q6 must NOT contain `BroadcastHashJoin`). Invalid runs are logged but excluded from analysis.

**If hints are ignored:** Some Databricks runtimes may override hints. If `/*+ MERGE */` still produces a BroadcastHashJoin, the automated harness will flag this and skip the run. You may need to:
- Try `SET spark.sql.autoBroadcastJoinThreshold = -1` (works on some serverless SKUs)
- Increase the small table size beyond the broadcast threshold
- Use a classic compute cluster for the join experiments only

## Billing Data

All queries reference `system.billing.usage` joined to `system.query.history`. Key fields:

- `system.query.history.statement_id` — unique query ID
- `system.billing.usage.usage_quantity` — DBUs consumed
- `system.query.history.read_bytes` — total bytes scanned
- `system.query.history.shuffle_write_bytes` — shuffle output bytes

There's typically a 5–10 minute lag before billing data appears. The harness accounts for this.

## Expected Output

The analysis produces a ratio table like:

```
Operator           Empirical  Current  Delta   Status
ScanColumnar           1.0×     1.0×   1.00×   ✓ baseline
ScanRowBased           ?.?×     3.0×   ?.??×   ?
Shuffle                ?.?×     7.5×   ?.??×   ?
Sort                   ?.?×     4.0×   ?.??×   ?
BroadcastHashJoin      ?.?×     1.0×   ?.??×   ?
SortMergeJoin          ?.?×     7.5×   ?.??×   ?
CartesianProduct       ?.?×    50.0×   ?.??×   ?
PySparkUDF             ?.?×     5.0×   ?.??×   ?
```

Any ratio where empirical differs by >2× from the current coefficient is flagged for review. The output feeds into CostCoefficients.scala and the methodology page.
