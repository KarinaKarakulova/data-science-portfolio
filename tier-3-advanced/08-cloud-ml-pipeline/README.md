# 08 · Cloud-Style ML Pipeline — dbt Feature Store + Model Registry-Lite

An end-to-end analytics-engineering pipeline for pharma demand forecasting: raw Rx panel → **dbt** staging/marts/feature models with **7 data tests as a build gate** → time-split model training → model metadata that makes every trained artifact traceable to its exact data and transformations.

**Why dbt-duckdb instead of BigQuery here:** so the repo runs end-to-end on a laptop in seconds. The models are dialect-portable SQL and the profile swap to BigQuery is a config change (documented in `rx_analytics/profiles.yml`) — the *patterns* (sources → staging → marts → features, tests as contracts, manifest-hashed lineage) are exactly what runs in production warehouses. This mirrors real workflow: develop locally, deploy to cloud.

## Pipeline
```
data/rx_monthly_raw.csv ──> dbt sources
        └─> stg_rx_monthly, stg_drugs        (views, typing only)
              └─> fct_rx_state_month          (+launch age; grain-uniqueness test)
                    └─> ft_demand_training    (leakage-safe lags/rollings; lead() target)
                          └─> src/train.py    (time split, baseline gate, registry)
```

## What it demonstrates
- **Tests as contracts:** during development, adding 7 new states made the `accepted_values` state test fail the build — the gate caught schema drift exactly as designed (kept in the git history deliberately)
- **Leakage-safe features by construction:** every feature is a `lag`/rolling over past rows; the target is a `lead()` — plus a **time-based split** (train ≤ 2025-05), because random splits on panel data flatter models
- **Baseline gate:** the GBM must beat lag-1 naive; final result **MAE 62.5 vs 71.7 (−12.8%)** on the held-out 6 months
- **Planted-signal recovery:** the generator injects a lagged detailing (rep promotion) effect and seasonality; permutation importance recovers both — the pipeline finds what is actually there
- **Registry-lite:** `reports/model_metadata.json` stores the SHA-256 of the exact training frame, the dbt manifest hash, params, split, and metrics — "which data produced this model?" has a one-file answer

## Run
```bash
python src/generate_data.py
cd rx_analytics && DBT_PROFILES_DIR=. dbt build && cd ..   # 4 models + 7 tests
python src/train.py
```
