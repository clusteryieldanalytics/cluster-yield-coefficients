# Spark Operator Cost Coefficient Experiment

**How much does a shuffle actually cost compared to a scan? We measured it.**

This repository contains the experiment harness used to empirically derive the operator cost coefficients that power [Cluster Yield](https://clusteryield.app)'s cost attribution engine. The results and methodology are described in the accompanying blog post: [TODO: blog post link].

## Background

When Cluster Yield decomposes your Spark job's cost across operators -- scans, shuffles, sorts, joins -- it uses relative weights to distribute the bill. A shuffle processes data differently than a scan: it serializes, spills to disk, transfers across the network, and deserializes. That costs more per GB. But *how much* more?

The weights need to be grounded in measurement, not intuition. This experiment produces those measurements.

## What's Here

```
.
├── coefficient_experiment_personal_compute.py   # Main experiment notebook (Databricks)
├── coefficient_experiment_manual.sql            # Standalone SQL for manual runs
└── README.md                                    # You are here
```

### `coefficient_experiment_personal_compute.py`

A Databricks Python notebook that runs the full experiment end-to-end:

1. **Setup** -- Creates test tables (~10 GB Delta, ~100 MB join table, CSV copy)
2. **Config lock** -- Disables AQE, fixes shuffle partitions, controls broadcast threshold
3. **Execution** -- Runs 9 query variants × (2 warmup + 10 measured) runs each
4. **Validation** -- Programmatically checks physical plan shapes before counting each run
5. **Metrics** -- Captures wall-clock time, executor CPU time, bytes read, shuffle bytes written
6. **Analysis** -- Derives per-operator cost ratios by differencing queries, compares to published coefficients

Import it into any Databricks workspace and attach to a Personal Compute cluster.

### `coefficient_experiment_manual.sql`

If you prefer to run experiments manually in a SQL editor:

- All 9 experiment queries with `EXPLAIN FORMATTED` validation variants
- `/* Q1 */` comment tags for automatic grouping in billing queries
- Billing collection queries against `system.query.history` (for paid workspaces)
- Coefficient of variation checks to flag queries needing more runs

## Quick Start

### Prerequisites

- Databricks workspace (any tier with Personal Compute)
- A Personal Compute cluster: single node, 32+ GB RAM, DBR 14.3+ LTS
- ~30 GB free storage for test tables

### Run the Experiment

```
1. Import coefficient_experiment_personal_compute.py as a Databricks notebook
2. Attach to your Personal Compute cluster
3. Run cells in order: Config → Setup → Experiment → Analysis
4. Total time: ~60-90 minutes (mostly table creation + query execution)
```

### What You Get

A ratio table like this (with your actual numbers):

```
Operator           Empirical   Current   Emp/Cur   Status
ScanColumnar           1.0x      1.0x     1.00x   baseline
ScanRowBased           ?.?x      3.0x     ?.??x   ?
Shuffle                ?.?x      7.5x     ?.??x   ?
Sort                   ?.?x      4.0x     ?.??x   ?
BroadcastHashJoin      ?.?x      1.0x     ?.??x   ?
SortMergeJoin          ?.?x      7.5x     ?.??x   ?
CartesianProduct       ?.?x     50.0x     ?.??x   ?
PySparkUDF             ?.?x      5.0x     ?.??x   ?
```

Plus raw metrics for every run: wall-clock ms, CPU time, bytes read, shuffle bytes written, records processed.

## The Query Series

Each query isolates one operator type by differing from the baseline (Q1) by exactly one physical operation.

| Query | Operator Isolated | Method |
|-------|-------------------|--------|
| Q1 | ScanColumnar (baseline) | `SELECT COUNT(*), SUM(val1) FROM delta_table` |
| Q2 | Column count effect | Same as Q1 but reads all 48 value columns |
| Q3 | ScanRowBased | Same logic as Q1 but reads from CSV table |
| Q4 | Shuffle | Adds `GROUP BY key` (introduces Exchange operator) |
| Q5b | Sort | Adds `ORDER BY key` via CTAS (introduces Sort + Exchange) |
| Q6 | SortMergeJoin | Join with broadcast disabled (`autoBroadcastJoinThreshold = -1`) |
| Q7 | BroadcastHashJoin | Same join with broadcast forced (`BROADCAST` hint + high threshold) |
| Q8 | CartesianProduct | `CROSS JOIN` on micro tables (10K x 10K rows) |
| Q9 | PySpark UDF | Identity UDF on scan (pure serialization overhead) |

**Ratio derivation:** Each operator's cost is computed by differencing the compound query from the simpler query that lacks that operator. For example, Shuffle cost = (Q4 time - Q1 time) / shuffle_bytes_written. All ratios are normalized to Q1 (columnar scan = 1.0x).

## Methodology Notes

### Why Wall-Clock Time?

On a fixed cluster (whether single-node or multi-node), execution time is proportional to resource consumption. A query that takes 3x longer consumed ~3x the compute resources. This makes wall-clock time a valid proxy for cost.

The original experiment design targeted Databricks Serverless, where per-query DBU consumption would be the cost signal. Serverless was not available for this run due to config restrictions (cannot disable AQE).

### Run It On Multiple Cluster Shapes

The harness is cluster-agnostic and captures full cluster metadata (node type, executor count,
memory, Photon status) so results from different runs are directly comparable. Run it on a
single node for your baseline, then on a multi-node cluster to see how shuffle and join ratios
shift when real network I/O is involved.

Expected differences on multi-node clusters:
- **Shuffle ratios increase** -- network transfer adds cost that loopback I/O does not
- **SortMergeJoin ratios increase** -- SMJ shuffles both sides over the network
- **Sort ratios may increase** -- more likely to spill with less memory per executor
- **UDF ratios stay stable** -- serialization overhead is per-batch, CPU-bound, topology-independent

### Why Not Serverless?

Serverless provides cleaner per-query billing (DBU), but it locks out most Spark configs. You cannot disable AQE or set `autoBroadcastJoinThreshold` on serverless, which makes it impossible to force specific plan shapes. Personal Compute gives full experimental control at the cost of a noisier measurement signal (hence 10 runs instead of 5).

### Plan Validation

Every run validates the physical plan before execution. The harness checks for required operators (e.g., Q6 must contain `SortMergeJoin`) and forbidden operators (e.g., Q6 must NOT contain `BroadcastHashJoin`). Runs where Spark's optimizer silently rewrote the plan are logged but excluded from analysis.

### Warmup Runs

The first 2 runs of each query are discarded. These absorb JIT compilation, OS page cache priming, Delta file listing caches, and other first-run artifacts that would skew measurements.

### Photon

The experiment records whether Photon is enabled but does not attempt to toggle it (Photon is a cluster-level setting). Ratios may differ between Photon and non-Photon runtimes -- Photon accelerates some operators (scans, shuffles) more than others (UDFs). If you run this on both, please share your results.

## How These Ratios Are Used

Cluster Yield's cost attribution engine uses these ratios as weights to distribute your job's actual cost across operators:

```
operator_cost = total_job_cost * (operator_weight * operator_volume_gb) / sum(all operator weighted volumes)
```

The weights determine the *relative* distribution. The absolute dollar amounts come from your cost and your data volumes -- the weights just determine how the bill is split between scan, shuffle, sort, and join.

The full methodology is published at [clusteryield.app](https://clusteryield.app).

## Contributing

If you run this experiment and get different ratios, we want to know. File an issue with:

- Your Databricks runtime version and Photon status
- Cluster instance type and size
- The raw measurements table (printed by the analysis step)
- Any plan validation failures you encountered

Different hardware, runtime versions, and Photon settings are all expected to produce somewhat different ratios. We are building a matrix of results across configurations.

## License

Apache 2.0

## Links

- **Product:** [Cluster Yield](https://clusteryield.app)
- **Blog Post:** [How Much Does a Spark Shuffle Actually Cost?](TODO)
- **Methodology:** [clusteryield.app/methodology](https://clusteryield.app) (TODO: deep link once page exists)