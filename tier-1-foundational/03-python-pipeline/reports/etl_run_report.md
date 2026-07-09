# ETL Run Report

## Warehouse row counts

- `dim_country`: **262** rows
- `fact_gdp_annual`: **13,979** rows
- `fact_inflation_annual`: **13,778** rows
- `fact_sp500_monthly`: **1,866** rows

## Validation gate

| check | failing rows |
|---|---:|
| gdp rows orphaned from dim_country | 0 |
| inflation rows orphaned from dim_country | 0 |
| gdp years outside 1960-2026 | 0 |
| sp500 months with non-positive price | 0 |
| sp500 duplicate months | 0 |
| monthly return magnitude > 60% (sanity) | 0 |

## Source lineage (SHA-256, first 12 hex)

- `gdp`: `8d84ef6bcacc`
- `inflation`: `b47dbd6a5d44`
- `sp500`: `28d16941c581`

## Cross-source analytical check

Correlation between US annual inflation and same-year S&P 500 nominal return (1960–2024, N=63): **-0.17** — consistent with the regime finding in Project 02 (high inflation associates with weaker nominal equity returns, though the same-year relationship is noisy).

Figures: `figures/01_g7_gdp.png`, `figures/02_inflation_vs_returns.png`