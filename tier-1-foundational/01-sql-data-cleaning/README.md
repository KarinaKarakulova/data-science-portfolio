# 01 · SQL Data Cleaning Capstone — Pharmacy Claims

A production-style SQL cleaning pipeline for a deliberately messy pharmacy claims feed: profile → validate → quarantine/repair → acceptance-test. Built on DuckDB; every pattern (staging schemas, violation ledgers, quality gates) transfers directly to BigQuery/Snowflake + dbt.

## Business problem

Pharmacy claims feeds (IQVIA-style) arrive with duplicated submissions, mixed date formats, invalid NDC codes, orphan pharmacy references, and fat-finger prices. Analytics built on the raw feed produces wrong spend totals and distorted territory views. This project turns a 50,500-row raw feed into a trustworthy 48,238-row fact table with a full audit trail.

## Results at a glance

- **11 defect classes** identified through profiling, each with measured prevalence
- **95.5%** of rows recovered as analysis-ready; **3.5%** quarantined *with reason codes*, never silently dropped
- **9 acceptance tests** enforced as a hard gate — the pipeline exits non-zero on any failure
- Full row **reconciliation**: raw = clean + quarantine + duplicates removed, exactly

Read the narrative: [`reports/data_quality_report.md`](reports/data_quality_report.md)
Raw metrics (auto-generated): [`reports/audit_tables.md`](reports/audit_tables.md)

## Why synthetic data

Real claims data is proprietary and PHI-restricted. `src/generate_raw_data.py` synthesizes a feed that reproduces the *defect classes* of real feeds (documented Q1–Q11 in the script) with a fixed seed, so the repo is fully reproducible and the "answer key" is auditable — you can verify the pipeline catches exactly what was injected, plus judge how it handles edge cases (e.g., rows failing multiple rules at once).

## How to run

```bash
pip install -r requirements.txt
python src/generate_raw_data.py   # writes data/raw/*.csv  (seeded, deterministic)
python src/run_pipeline.py        # runs sql/00→03, exports clean data + audit report
```

Outputs:
- `data/clean/fact_claims.parquet` — analysis-ready fact table with provenance flags
- `data/clean/claims_quarantine.csv` — rejected rows + rule codes
- `reports/audit_tables.md` — regenerated audit metrics

## Repo structure

```
01-sql-data-cleaning/
├── data/raw/          # generated messy feed (CSV)
├── data/clean/        # pipeline outputs (parquet + quarantine)
├── sql/
│   ├── 00_load_raw.sql       # staging load — all VARCHAR, on purpose
│   ├── 01_profiling.sql      # completeness, cardinality, format census, distributions
│   ├── 02_validation.sql     # 11 rules -> row-level violations ledger (REJECT/REPAIR/FLAG)
│   └── 03_cleaning.sql       # quarantine, dedupe, normalize, repair, acceptance tests
├── src/
│   ├── generate_raw_data.py  # seeded synthetic feed with documented defects
│   └── run_pipeline.py       # orchestrator + quality gate (non-zero exit on failure)
└── reports/
    ├── data_quality_report.md
    └── audit_tables.md
```

## SQL techniques demonstrated

Staging/audit/clean schema separation · `TRY_STRPTIME` cascades for multi-format date parsing · regex constraint validation · `ANTI JOIN` quarantining · `QUALIFY ROW_NUMBER()` dedup over business keys · reference-anchored outlier repair · `STRING_AGG` reason-code rollups · quantile profiling (`quantile_cont`) · acceptance tests as data
