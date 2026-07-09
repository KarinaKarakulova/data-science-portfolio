# 02 · EDA — S&P 500, 155 Years of Monthly Data

Exploratory analysis of the Shiller S&P 500 dataset (1871–2026, real public data): 16 publication-quality figures, formal statistical testing, and an insight narrative where every claim carries a number and a "so what."

## Questions asked
1. What did the index actually return — nominal vs real?
2. How non-normal are returns, and in which direction?
3. Is anything about returns predictable (level, volatility, valuation)?
4. How does the macro regime (inflation) condition the return distribution?

## Headline findings (details in [`reports/insights.md`](reports/insights.md))
- $1 (1871) → **$1,017 nominal but ~$41 real** — deflators change the answer by 25×
- Skew −0.52, excess kurtosis **11.2**; normality rejected (Jarque–Bera p ≈ 0) — fat, asymmetric tails
- Volatility clusters strongly (ACF of |r|) even though direction barely does → GARCH-style structure
- Max drawdown **−84.8%** (1932) — the risk that a 14% annualized vol figure hides
- CAPE vs next-decade real returns: corr **−0.33**, honestly caveated for overlapping windows
- A data-construction artifact caught during EDA: Shiller's monthly *averaged* prices mechanically inflate lag-1 autocorrelation (Working effect) — analyzed, not naively reported as momentum

## Statistical methods
Jarque–Bera normality test · ADF stationarity tests (level vs returns) · autocorrelation analysis · quantile/drawdown analysis · regime conditioning · overlapping-window bias awareness

## How to run
```bash
pip install -r ../../shared/requirements.txt
python src/eda_sp500.py    # regenerates all 16 figures + statistical_summary.md + metrics.json
```

Data: [`datasets/s-and-p-500`](https://github.com/datasets/s-and-p-500) (Robert Shiller's long-run series, ODC-PDDL). A copy is committed to `data/` for reproducibility.

## Structure
```
02-eda-financial/
├── data/sp500_monthly.csv
├── src/eda_sp500.py          # single reproducible script: figures + tests + metrics
├── figures/                  # 16 PNGs (01_price_log ... 16_extremes)
└── reports/
    ├── statistical_summary.md    # auto-generated: tests, tables
    ├── insights.md               # human narrative: findings + limitations
    └── metrics.json              # machine-readable results
```
