# Insights — 155 Years of the S&P 500

*All numbers computed by `src/eda_sp500.py` from the Shiller monthly dataset (1,866 months, Jan 1871 – Jun 2026). Figures referenced are in `figures/`.*

## 1. Inflation ate 96% of the headline story
A dollar in the index in 1871 grew to roughly **$1,017 nominal — but only ~$41 in real terms** (fig 02). Any long-horizon financial analysis that ignores deflators is answering a different question than the one being asked. **So what:** always state whether results are nominal or real; the gap here is a factor of ~25.

## 2. Returns are decisively non-normal — in a specific, asymmetric way
Monthly log returns show skew of **−0.52** and excess kurtosis of **11.2**; Jarque–Bera rejects normality at any conventional level (figs 03–04). The tails are fat *and* the left tail is fatter: the worst months (1931–32, 1929, 2008) are far more extreme than the best (fig 16). **So what:** risk models assuming Gaussian returns will systematically understate crash risk; quantile-based or heavy-tailed methods are the appropriate default.

## 3. Direction is hard to predict; magnitude is not
Returns themselves carry limited memory, but **absolute** returns are strongly autocorrelated (fig 14) — calm months follow calm months, turbulent follow turbulent (figs 05–06). This is the volatility-clustering stylized fact that motivates GARCH-family models. **So what:** even without a view on direction, volatility is forecastable enough to matter for position sizing and risk budgeting.

*Measurement caveat found during EDA:* the raw lag-1 return autocorrelation (0.27) is suspiciously high for an efficient market. Shiller's prices are **monthly averages of daily values**, and averaging mechanically induces positive autocorrelation (a Working-effect artifact). This is a data-construction property, not a tradable signal — an example of why knowing how a dataset was built matters as much as analyzing it.

## 4. Drawdowns are the risk that summary statistics hide
The maximum peak-to-trough decline was **−84.8% (June 1932)**; the index has spent substantial fractions of its history more than 20% below a prior peak (fig 07). Annualized volatility of ~14% sounds tame; an 85% drawdown does not. **So what:** report drawdown alongside volatility — they answer different investor questions ("how bumpy?" vs "how bad can it get and how long to recover?").

## 5. Starting valuation says something about the next decade — carefully stated
Higher CAPE at purchase associates with lower subsequent 10-year real returns (corr ≈ **−0.33**, fig 12). Current CAPE (~31) sits well above its long-run mean (~17). Two honesty checks: (a) overlapping 10-year windows leave only ~15 independent observations, so the confidence interval on that correlation is wide; (b) the relationship has been weaker post-1990. **So what:** valuation is a tilt for long-horizon expectations, not a timing tool — and the statistical limitation belongs in the headline, not a footnote.

## 6. Inflation regimes reshape the return distribution
Nominal returns cluster tighter and higher in the 0–5% inflation band; both deflationary and >5% inflation regimes widen the distribution and drag the median (fig 15). The 1970s decade box (fig 08) shows this persistently. **So what:** macro regime is a legitimate conditioning variable for scenario analysis — a lesson directly transferable to budget-planning under inflation uncertainty.

## Limitations
- Survivorship/index-construction effects: the "S&P 500" before 1957 is a reconstructed series.
- Monthly averaging (see §3) affects all autocorrelation-based findings.
- Dividend and CPI fields are missing for the most recent months (recorded as `0.0` in the source and treated as missing here) — recent real-return and yield figures end earlier than price figures.
- No transaction costs, taxes, or dividend-reinvestment timing are modeled in growth-of-$1 calculations (dividends are excluded entirely, understating total return).
