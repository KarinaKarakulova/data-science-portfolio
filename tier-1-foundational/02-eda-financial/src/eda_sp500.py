"""
EDA — S&P 500, 150+ years of monthly data (Shiller dataset).

Produces 16 publication-quality figures (figures/), a statistical summary
(reports/statistical_summary.md), and the quantitative inputs cited in
reports/insights.md.

Run:  python src/eda_sp500.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
REP = ROOT / "reports"
FIG.mkdir(exist_ok=True); REP.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})

# ---------------------------------------------------------------- load/clean
df = pd.read_csv(ROOT / "data" / "sp500_monthly.csv", parse_dates=["Date"])
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
df = df.set_index("date")

# Recent months ship with 0.0 placeholders for fundamentals — treat as missing
for c in ["dividend", "earnings", "consumer_price_index", "real_price",
          "real_dividend", "real_earnings", "pe10", "long_interest_rate"]:
    df[c] = df[c].replace(0.0, np.nan)

df = df[df["sp500"].notna() & (df["sp500"] > 0)]

df["log_price"] = np.log(df["sp500"])
df["ret"] = df["sp500"].pct_change()                 # monthly nominal return
df["log_ret"] = np.log1p(df["ret"])
df["real_ret"] = df["real_price"].pct_change()
df["div_yield"] = df["dividend"] / df["sp500"] * 100
df["roll_vol_ann"] = df["log_ret"].rolling(36).std() * np.sqrt(12) * 100
df["drawdown"] = df["sp500"] / df["sp500"].cummax() - 1
df["decade"] = (df.index.year // 10) * 10

metrics = {}

def save(fig_name):
    plt.savefig(FIG / f"{fig_name}.png"); plt.close()

# 01 nominal price, log scale --------------------------------------------------
plt.figure(figsize=(10, 4.5))
plt.semilogy(df.index, df["sp500"], lw=0.9)
plt.title("S&P 500 nominal price, 1871–2026 (log scale)")
plt.ylabel("Index level (log)"); save("01_price_log")

# 02 real vs nominal growth of $1 ---------------------------------------------
base = df.dropna(subset=["real_price"])
plt.figure(figsize=(10, 4.5))
plt.semilogy(base.index, base["sp500"] / base["sp500"].iloc[0], lw=0.9, label="Nominal")
plt.semilogy(base.index, base["real_price"] / base["real_price"].iloc[0], lw=0.9, label="Real (CPI-deflated)")
plt.legend(); plt.title("Growth of $1: nominal vs inflation-adjusted (log scale)")
save("02_real_vs_nominal")
metrics["nominal_growth_x"] = float(base["sp500"].iloc[-1] / base["sp500"].iloc[0])
metrics["real_growth_x"] = float(base["real_price"].dropna().iloc[-1] / base["real_price"].iloc[0])

# 03 return distribution vs normal --------------------------------------------
r = df["log_ret"].dropna()
plt.figure(figsize=(8, 4.5))
sns.histplot(r, bins=120, stat="density")
x = np.linspace(r.min(), r.max(), 400)
plt.plot(x, stats.norm.pdf(x, r.mean(), r.std()), "r--", lw=1.2, label="Normal fit")
plt.legend(); plt.title("Monthly log returns vs fitted normal"); plt.xlabel("log return")
save("03_return_hist")

# 04 QQ plot -------------------------------------------------------------------
plt.figure(figsize=(5.5, 5.5))
stats.probplot(r, dist="norm", plot=plt)
plt.title("QQ plot of monthly log returns"); save("04_qq")

# 05 rolling volatility ---------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(df.index, df["roll_vol_ann"], lw=0.9)
plt.title("Rolling 36-month annualized volatility (%)"); save("05_rolling_vol")

# 06 volatility clustering: |returns| ------------------------------------------
plt.figure(figsize=(10, 3.5))
plt.plot(r.index, r.abs() * 100, lw=0.5)
plt.title("Absolute monthly returns (%) — volatility clusters"); save("06_vol_cluster")

# 07 drawdowns ------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.fill_between(df.index, df["drawdown"] * 100, 0, alpha=0.6)
plt.title("Drawdown from prior peak (%)"); save("07_drawdowns")
metrics["max_drawdown_pct"] = float(df["drawdown"].min() * 100)
metrics["max_dd_date"] = str(df["drawdown"].idxmin().date())

# 08 decade return boxplots -----------------------------------------------------
plt.figure(figsize=(11, 4.5))
sub = df.dropna(subset=["ret"])
sns.boxplot(x="decade", y=sub["ret"] * 100, data=sub, fliersize=1.5)
plt.xticks(rotation=45); plt.ylabel("monthly return %")
plt.title("Monthly return distribution by decade"); save("08_decade_box")

# 09 monthly seasonality ----------------------------------------------------------
plt.figure(figsize=(9, 4))
monthly = sub.groupby(sub.index.month)["ret"].mean() * 100
se = sub.groupby(sub.index.month)["ret"].sem() * 100
plt.bar(monthly.index, monthly.values, yerr=1.96 * se.values, capsize=3)
plt.xticks(range(1, 13)); plt.ylabel("mean return % (±95% CI)")
plt.title("Average return by calendar month, 1871–2026"); save("09_seasonality")

# 10 dividend yield ----------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(df.index, df["div_yield"], lw=0.9)
plt.title("Dividend yield (%)"); save("10_div_yield")

# 11 PE10 (CAPE) ---------------------------------------------------------------------
plt.figure(figsize=(10, 4))
pe = df["pe10"].replace(0, np.nan).dropna()
plt.plot(pe.index, pe, lw=0.9)
plt.axhline(pe.mean(), color="r", ls="--", lw=1, label=f"mean {pe.mean():.1f}")
plt.legend(); plt.title("CAPE (PE10) — cyclically adjusted P/E"); save("11_cape")

# 12 CAPE vs subsequent 10y real return ----------------------------------------------
fwd = df["real_price"].pct_change(120).shift(-120)
fwd_ann = (1 + fwd) ** (1 / 10) - 1
valid = pd.concat([df["pe10"], fwd_ann], axis=1).dropna()
valid.columns = ["pe10", "fwd10"]
plt.figure(figsize=(6.5, 5))
plt.scatter(valid["pe10"], valid["fwd10"] * 100, s=4, alpha=0.4)
b, a = np.polyfit(valid["pe10"], valid["fwd10"] * 100, 1)
# Math 257 check (Wk 11, Module 38): np.polyfit(deg=1) is least squares on the
# design matrix [x 1] — solving the normal equations by hand must reproduce it.
Xd = np.column_stack([valid["pe10"].values, np.ones(len(valid))])
beta = np.linalg.solve(Xd.T @ Xd, Xd.T @ (valid["fwd10"].values * 100))
assert np.allclose(beta, [b, a]), "polyfit != normal-equations solution"
print(f"[Math 257] polyfit reproduced via A^T A b = A^T y: slope {beta[0]:+.3f}")
xs = np.linspace(valid["pe10"].min(), valid["pe10"].max(), 50)
plt.plot(xs, a + b * xs, "r--", lw=1.2)
plt.xlabel("CAPE"); plt.ylabel("subsequent 10y real return %/yr")
plt.title("Starting valuation vs forward 10-year real return"); save("12_cape_vs_fwd")
metrics["cape_fwd_corr"] = float(valid["pe10"].corr(valid["fwd10"]))

# 13 correlation heatmap ---------------------------------------------------------------
corr_cols = ["ret", "div_yield", "pe10", "long_interest_rate", "roll_vol_ann"]
plt.figure(figsize=(6, 5))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Correlation matrix"); save("13_corr_heatmap")

# 14 ACF of returns vs |returns| --------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
plot_acf(r, lags=36, ax=axes[0], title="ACF: returns")
plot_acf(r.abs(), lags=36, ax=axes[1], title="ACF: |returns|")
save("14_acf")

# 15 inflation vs nominal returns by regime ----------------------------------------------
cpi_yoy = df["consumer_price_index"].pct_change(12) * 100
reg = pd.cut(cpi_yoy, [-np.inf, 0, 2, 5, np.inf],
             labels=["deflation", "0–2%", "2–5%", ">5%"])
plt.figure(figsize=(8, 4.2))
sns.boxplot(x=reg, y=df["ret"] * 100, fliersize=1.5)
plt.ylabel("monthly nominal return %"); plt.xlabel("trailing 12m inflation regime")
plt.title("Returns across inflation regimes"); save("15_inflation_regimes")

# 16 worst/best months table as figure ------------------------------------------------------
ext = pd.concat([r.nsmallest(8), r.nlargest(8)]).sort_values()
plt.figure(figsize=(8, 4.5))
colors = ["#d62728" if v < 0 else "#2ca02c" for v in ext]
plt.barh([d.strftime("%Y-%m") for d in ext.index], ext * 100, color=colors)
plt.xlabel("monthly log return %"); plt.title("Extreme months, 1871–2026"); save("16_extremes")

# ------------------------------------------------------------------ statistics
jb_stat, jb_p = stats.jarque_bera(r)
adf_price = adfuller(df["log_price"].dropna())
adf_ret = adfuller(r)
ljung = pd.Series(r).autocorr(1)

desc = (r.describe().to_frame("log_ret").T
        .assign(skew=r.skew(), kurtosis=r.kurt()))

metrics.update({
    "n_months": int(len(df)),
    "span": f"{df.index[0].date()} → {df.index[-1].date()}",
    "mean_monthly_ret_pct": float(r.mean() * 100),
    "ann_ret_pct": float((np.exp(r.mean() * 12) - 1) * 100),
    "ann_vol_pct": float(r.std() * np.sqrt(12) * 100),
    "skew": float(r.skew()), "excess_kurtosis": float(r.kurt()),
    "jarque_bera_p": float(jb_p),
    "adf_logprice_p": float(adf_price[1]), "adf_ret_p": float(adf_ret[1]),
    "acf1_ret": float(ljung), "acf1_absret": float(r.abs().autocorr(1)),
    "mean_div_yield": float(df["div_yield"].mean()),
    "cape_now_vs_mean": [float(pe.iloc[-1]), float(pe.mean())],
})

lines = [
    "# Statistical Summary — S&P 500 Monthly (1871–2026)", "",
    f"- Sample: **{metrics['n_months']:,} months** ({metrics['span']})",
    f"- Annualized return **{metrics['ann_ret_pct']:.1f}%** nominal, volatility **{metrics['ann_vol_pct']:.1f}%**",
    f"- Nominal growth of $1 since 1871: **${metrics['nominal_growth_x']:,.0f}**; real: **${metrics['real_growth_x']:,.0f}**", "",
    "## Distribution of monthly log returns", "",
    desc.round(4).to_markdown(),
    "",
    f"- Skew {metrics['skew']:.2f}, excess kurtosis {metrics['excess_kurtosis']:.2f} — fat left tail",
    f"- Jarque–Bera p = {jb_p:.2e} → **normality decisively rejected**", "",
    "## Stationarity (ADF)", "",
    f"- log price: p = {metrics['adf_logprice_p']:.3f} → non-stationary (as expected for a price level)",
    f"- returns: p = {metrics['adf_ret_p']:.2e} → stationary", "",
    "## Dependence structure", "",
    f"- lag-1 autocorrelation of returns: {metrics['acf1_ret']:.3f} (weak)",
    f"- lag-1 autocorrelation of |returns|: {metrics['acf1_absret']:.3f} (strong) → volatility clustering: "
    "magnitude is predictable even where direction is not",
    "",
    "## Valuation signal",
    "",
    f"- Correlation(CAPE, subsequent 10-y real return) = **{metrics['cape_fwd_corr']:.2f}**",
    f"- Current CAPE {metrics['cape_now_vs_mean'][0]:.1f} vs long-run mean {metrics['cape_now_vs_mean'][1]:.1f}",
    "",
    "_Caveat: overlapping 10-year windows inflate the apparent strength of the CAPE relationship; "
    "the effective sample is ~15 independent decades, so treat the correlation as indicative, not tradable._",
]
(REP / "statistical_summary.md").write_text("\n".join(lines))
(REP / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
print(f"figures: {len(list(FIG.glob('*.png')))}")
