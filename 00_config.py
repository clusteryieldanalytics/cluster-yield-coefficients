# Databricks notebook source
# MAGIC %md
# MAGIC # Shared Experiment Config
# MAGIC
# MAGIC Shared constants for the operator cost ratio experiment.
# MAGIC Import via `%run ./00_config` from any sibling notebook.

# COMMAND ----------

CATALOG = "test_weights"
SCHEMA = "experiment"
FQ = f"{CATALOG}.{SCHEMA}"

NUM_RUNS = 5
DATE_START = "2025-01-01"
DATE_END = "2025-12-31"

# Target table sizes (adjust row counts during setup if needed)
LARGE_TABLE_TARGET_GB = 100
MEDIUM_TABLE_TARGET_MB = 500
SMALL_TABLE_TARGET_MB = 5
ROW_COUNT_LARGE = 500_000_000
MEDIUM_ROW_LIMIT = 5_000_000

# Current heuristic weights from CostCoefficients.scala
CURRENT_WEIGHTS = {
    "ScanColumnar":      {"weight": 0.02, "ratio": 1.0},
    "ScanRowBased":      {"weight": 0.06, "ratio": 3.0},
    "Shuffle":           {"weight": 0.15, "ratio": 7.5},
    "Sort":              {"weight": 0.08, "ratio": 4.0},
    "BroadcastHashJoin": {"weight": 0.02, "ratio": 1.0},
    "SortMergeJoin":     {"weight": 0.15, "ratio": 7.5},
    "CartesianProduct":  {"weight": 1.00, "ratio": 50.0},
    "PySpark UDF":       {"weight": 0.10, "ratio": 5.0},
}

# Blog-ready operator list (display order)
BLOG_OPERATORS = [
    "ScanColumnar", "ScanRowBased", "Shuffle", "Sort",
    "BroadcastHashJoin", "SortMergeJoin", "CartesianProduct", "PySpark UDF"
]

# Maps operators to the query ID used for measurement
OPERATOR_TO_QID = {
    "ScanColumnar": "Q1", "ScanRowBased": "Q3", "Shuffle": "Q4",
    "Sort": "Q5", "BroadcastHashJoin": "Q7", "SortMergeJoin": "Q6",
    "CartesianProduct": "Q8", "PySpark UDF": "Q9",
}

# Billing lag observed in Step 0 — update after canary test
BILLING_LAG_MINUTES = 30
