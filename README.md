# Data Science Portfolio — Karina Karakulova

Data Analyst / BI Developer (5 yrs, incl. 4 at EPAM Systems) preparing for graduate study in Data Science. Ten runnable projects progressing from data engineering fundamentals through applied ML to cloud-native, domain-specific analytics — with emphasis on statistical rigor, honest evaluation, reproducibility, and business translation.

**Stack:** SQL (BigQuery, DuckDB), Python, dbt, Power BI/Tableau · **Domains:** pharma/CPG, financial analytics · **Languages:** EN / RU / ZH

Every project runs end-to-end from a fresh clone (seeded data or committed public data, no hard-coded paths), reports uncertainty and limitations alongside results, and ends findings with the decision they inform.

## Projects

### Tier 1 · Foundational
| # | Project | One-line result |
|---|---------|-----------------|
| [01](tier-1-foundational/01-sql-data-cleaning/) | **SQL Data Cleaning Capstone** — pharmacy claims | 50,500-row messy feed → 11 defect classes profiled, REJECT/REPAIR/FLAG rule ledger, quarantine with reason codes, 9 acceptance tests as a hard gate, exact row reconciliation |
| [02](tier-1-foundational/02-eda-financial/) | **EDA — S&P 500, 155 years** (real Shiller data) | 16 figures + formal tests: fat asymmetric tails (kurtosis 11.2), −84.8% max drawdown, volatility clustering, CAPE↔forward-return with overlapping-window caveat, and a caught data-construction artifact (Working effect) |
| [03](tier-1-foundational/03-python-pipeline/) | **Python ETL — macro-financial mart** (3 real sources) | Explicit task DAG with failure propagation, cached lineage-hashed extracts, SQLite star schema with constraints, validation gate, cross-source analytical check |

### Tier 2 · Intermediate
| # | Project | One-line result |
|---|---------|-----------------|
| [04](tier-2-intermediate/04-ml-classification/) | **ML Classification — churn** (real IBM Telco) | Leakage-safe pipelines, 3 model families under stratified CV, test ROC-AUC 0.843, calibration verified, profit-optimal threshold ≈0.43 from stated retention economics |
| [05](tier-2-intermediate/05-timeseries-forecasting/) | **Time-Series Forecasting** (real daily data) | Rolling-origin backtest vs hard baselines; harmonic-Fourier + ARMA champion (MAE 2.03 vs 2.19 climatology), 80% intervals with 83% empirical coverage, Ljung–Box diagnostics |
| [06](tier-2-intermediate/06-ab-testing/) | **A/B Testing & Statistical Rigor** | Power analysis → SRM guardrail → frequentist + Bayesian agreement; peeking simulation shows 4.8% → 22.4% false-positive inflation from daily looks |
| [07](tier-2-intermediate/07-unsupervised-learning/) | **Unsupervised — CPG segmentation** | Gradable clustering: hidden archetypes recovered at ARI 0.97, K-means×Ward agreement 0.98, anomalies screened *before* clustering at 100% recall |

### Tier 3 · Advanced
| # | Project | One-line result |
|---|---------|-----------------|
| [08](tier-3-advanced/08-cloud-ml-pipeline/) | **dbt Feature Pipeline + Model Registry-lite** | Sources→staging→marts→features with 7 dbt tests as a build gate (one caught schema drift mid-development, kept in history), time-split training beats naive by 12.8%, SHA-hashed data/manifest lineage per model |
| [09](tier-3-advanced/09-nlp/) | **NLP — spam/abuse filtering** (5,572 real SMS) | Char n-grams beat word models for a stated linguistic reason; precision-first operating point 99.3%P/94.6%R; error analysis on actual texts identifies where transformers would pay |
| [10](tier-3-advanced/10-domain-capstone/) | **Pharma Capstone — HCP segmentation + call-plan optimization** | SQL KPI layer → value×momentum segments validated twice (ARI 0.82 vs hidden truth) → provably-optimal greedy reallocation: priority coverage 51.5%→100%, +64% expected uplift under stated assumptions, with a randomized-pilot validation design |

## Setup
```bash
pip install -r shared/requirements.txt
# then follow each project's README (every project regenerates its own outputs)
```

## Data policy
Real public data is used where licensing permits (Shiller S&P, World Bank, IBM Telco, UCI SMS). Pharma/CPG projects use **seeded synthetic data with documented, planted structure** — real claims/prescriber data is PHI-restricted or strictly licensed, and planting hidden ground truth turns otherwise-ungradable analyses (segmentation, cleaning) into measurable ones. Each README states which and why.
