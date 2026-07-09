# Statistical Summary — S&P 500 Monthly (1871–2026)

- Sample: **1,866 months** (1871-01-01 → 2026-06-01)
- Annualized return **4.9%** nominal, volatility **14.0%**
- Nominal growth of $1 since 1871: **$1,017**; real: **$41**

## Distribution of monthly log returns

|         |   count |   mean |    std |     min |    25% |    50% |    75% |    max |    skew |   kurtosis |
|:--------|--------:|-------:|-------:|--------:|-------:|-------:|-------:|-------:|--------:|-----------:|
| log_ret |    1865 |  0.004 | 0.0405 | -0.3075 | -0.015 | 0.0069 | 0.0274 | 0.4075 | -0.5229 |    11.2409 |

- Skew -0.52, excess kurtosis 11.24 — fat left tail
- Jarque–Bera p = 0.00e+00 → **normality decisively rejected**

## Stationarity (ADF)

- log price: p = 0.998 → non-stationary (as expected for a price level)
- returns: p = 1.82e-18 → stationary

## Dependence structure

- lag-1 autocorrelation of returns: 0.274 (weak)
- lag-1 autocorrelation of |returns|: 0.199 (strong) → volatility clustering: magnitude is predictable even where direction is not

## Valuation signal

- Correlation(CAPE, subsequent 10-y real return) = **-0.33**
- Current CAPE 30.8 vs long-run mean 17.4

_Caveat: overlapping 10-year windows inflate the apparent strength of the CAPE relationship; the effective sample is ~15 independent decades, so treat the correlation as indicative, not tradable._