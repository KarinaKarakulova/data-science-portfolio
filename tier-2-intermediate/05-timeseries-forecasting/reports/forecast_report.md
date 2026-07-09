# Forecasting Report — Daily Temperatures (Energy-Demand Framing)

## Why baselines first
Every model must beat **climatology** (day-of-year average) and **seasonal naive**
(same day last year) to justify its complexity. Most forecasting failures in industry
are models that never faced this test.

## Rolling-origin backtest (6 origins × 30-day horizon)

| model           |   MAE |   RMSE |
|:----------------|------:|-------:|
| harmonic_arma   | 2.025 |  2.525 |
| climatology     | 2.193 |  2.755 |
| holt_trend_only | 2.791 |  3.662 |
| seasonal_naive  | 3.411 |  4.287 |

- **Holt (trend-only) is the deliberate failure exhibit**: without seasonality it
  extrapolates the recent level and degrades fast with horizon (fig 03).
- Seasonal naive is noisy because any single "same day last year" inherits that
  day's weather noise; climatology averages that noise away.
- **Harmonic regression + ARMA(2,1) errors wins** (harmonic_arma MAE
  2.03°C): Fourier terms capture the smooth annual cycle,
  the ARMA component captures multi-day weather persistence that climatology
  ignores. This structure was chosen over seasonal ARIMA because a 365-lag
  seasonal difference is statistically and computationally ill-behaved —
  harmonics are the standard remedy for long seasonal periods.

## Held-out final year (1990)
- MAE **1.96°C** vs seasonal naive 2.87°C
- 80% prediction interval empirical coverage: **83%** — intervals are
  honest, slightly conservative
- Ljung–Box p-values at lags 7/14/30: 0.135, 0.366, 0.353 —
  residual autocorrelation is largely absorbed (values > 0.05 mean no significant
  structure remains at that lag; see fig 05)

## Business translation
For an energy retailer, the gap between climatology (2.19°C) and the champion
(2.03°C) concentrates in the **first ~7 days** of the horizon (fig 03) —
exactly the window where day-ahead and week-ahead purchasing decisions are made.
Beyond ~2 weeks the champion converges to climatology, which is the correct,
honest behavior: weather persistence is a short-memory signal. **So what:** invest
in short-horizon accuracy and procurement automation; do not promise 30-day skill.

## Limitations
- One location, one variable; a load-forecasting system would add calendars,
  humidity, and holiday effects as exogenous regressors (same SARIMAX interface).
- Fourier K=3 chosen by backtest MAE among K ∈ {2,3,4} offline; a fuller study
  would show that selection curve.
- Prediction intervals assume Gaussian errors; QQ plot shows mild tail deviation.
