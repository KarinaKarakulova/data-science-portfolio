# 05 · Time-Series Forecasting — Baseline-Disciplined

Forecasting 10 years of real daily data (3,650 obs, Melbourne minimum temperatures; methodology framed for energy-demand planning, and transfers unchanged to sales/demand series).

## The point of this project
Most forecasting work fails not on model choice but on evaluation discipline. Here every model must beat two hard baselines under a **rolling-origin backtest (6 origins × 30-day horizon)** — never a single lucky split.

| model | backtest MAE (°C) | role |
|---|---:|---|
| **harmonic Fourier + ARMA(2,1)** | **2.03** | champion |
| climatology (day-of-year mean) | 2.19 | hard baseline |
| Holt, trend-only | 2.79 | deliberate failure exhibit |
| seasonal naive (t−365) | 3.41 | hard baseline |

- Held-out final year (1990): **MAE 1.96°C**; 80% prediction intervals achieve **83% empirical coverage** — uncertainty is quantified and verified, not decorative
- Residual diagnostics: Ljung–Box, ACF, QQ (fig 05) — autocorrelation absorbed
- Modeling judgment on display: seasonal ARIMA is intractable at period 365, so seasonality enters as **Fourier exogenous terms with ARMA errors** (SARIMAX) — the standard long-period remedy, explained in the report
- The honest business finding: model skill over climatology concentrates in **days 1–7** and vanishes by ~2 weeks — so the recommendation is where to invest (short-horizon procurement), not a fake promise of 30-day skill

Full narrative: [`reports/forecast_report.md`](reports/forecast_report.md)

## Run
```bash
python src/forecast.py    # ~2-3 min (multiple SARIMAX fits across backtest origins)
```
