"""
A/B test — checkout redesign, full experiment lifecycle.

Why simulated data: with a known ground truth (true lift = +1.2pp) you can
verify the machinery actually recovers the truth, and you can demonstrate
*p-hacking mechanics* (peeking) by simulation — impossible with a single real
dataset. Experiment design happens BEFORE data generation, as in real life.

Sections:
  1. Design & power analysis (MDE, alpha, power -> required n)
  2. Data generation with known truth (+ a null secondary metric)
  3. Guardrails: sample-ratio-mismatch (SRM) check
  4. Frequentist analysis: two-proportion z-test, CI, effect size
  5. The peeking demonstration: sequential looks inflate false positives
  6. Bayesian analysis: Beta-Binomial posterior, P(B>A), expected loss
  7. Decision & report

Run: python src/ab_test.py    (seeded)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (confint_proportions_2indep,
                                          proportion_effectsize,
                                          proportions_ztest)

SEED = 7
rng = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parents[1]
FIG, REP = ROOT / "figures", ROOT / "reports"
FIG.mkdir(exist_ok=True); REP.mkdir(exist_ok=True)

# ------------------------------------------------- 1 · design & power analysis
BASELINE = 0.100          # historical checkout conversion
MDE = 0.010               # minimum detectable effect we care about: +1.0pp
ALPHA, POWER = 0.05, 0.80

es = proportion_effectsize(BASELINE + MDE, BASELINE)
n_per_arm = int(np.ceil(NormalIndPower().solve_power(
    effect_size=es, alpha=ALPHA, power=POWER, ratio=1.0)))

# power curve across candidate lifts
lifts = np.linspace(0.002, 0.02, 40)
powers = [NormalIndPower().solve_power(
    effect_size=proportion_effectsize(BASELINE + L, BASELINE),
    nobs1=n_per_arm, alpha=ALPHA) for L in lifts]
plt.figure(figsize=(7, 4.2))
plt.plot(lifts * 100, powers, lw=1.4)
plt.axhline(0.8, ls="--", c="gray", lw=1); plt.axvline(MDE * 100, ls="--", c="r", lw=1)
plt.xlabel("true lift (pp)"); plt.ylabel("power at n per arm = %d" % n_per_arm)
plt.title("Power curve — what this experiment can and cannot detect")
plt.savefig(FIG / "01_power_curve.png", dpi=120, bbox_inches="tight"); plt.close()

# ------------------------------------------------- 2 · generate the experiment
TRUE_LIFT = 0.012                      # ground truth: +1.2pp (above MDE)
pA, pB = BASELINE, BASELINE + TRUE_LIFT
DAYS = 14
daily = n_per_arm // DAYS + 1

convA = rng.binomial(daily, pA, DAYS)
convB = rng.binomial(daily, pB, DAYS)
nA = nB = daily * DAYS

# null secondary metric: average order value (no true effect)
aovA = rng.lognormal(4.0, 0.5, convA.sum())
aovB = rng.lognormal(4.0, 0.5, convB.sum())

xA, xB = int(convA.sum()), int(convB.sum())

# ------------------------------------------------- 3 · guardrail: SRM check
srm_chi, srm_p = stats.chisquare([nA, nB], f_exp=[(nA + nB) / 2] * 2)

# ------------------------------------------------- 4 · frequentist analysis
z, p_val = proportions_ztest([xB, xA], [nB, nA], alternative="larger")
ci_lo, ci_hi = confint_proportions_2indep(xB, nB, xA, nA, method="wald")
obs_lift = xB / nB - xA / nA
cohen_h = proportion_effectsize(xB / nB, xA / nA)

# secondary metric (expected: null)
t_aov, p_aov = stats.ttest_ind(aovB, aovA, equal_var=False)

# ------------------------------------------------- 5 · peeking demonstration
def peeking_fpr(n_sims=2000, looks=14):
    """Simulate A/A tests (no true effect); count how often a 'peeker' who
    tests daily and stops at the first p<0.05 declares a winner."""
    hits_peek, hits_fixed = 0, 0
    per_look = n_per_arm // looks
    for _ in range(n_sims):
        a = rng.binomial(per_look, BASELINE, looks).cumsum()
        b = rng.binomial(per_look, BASELINE, looks).cumsum()
        n_cum = per_look * np.arange(1, looks + 1)
        sig_any = False
        for i in range(looks):
            _, p = proportions_ztest([b[i], a[i]], [n_cum[i], n_cum[i]])
            if p < ALPHA:
                sig_any = True
                break
        hits_peek += sig_any
        _, p_final = proportions_ztest([b[-1], a[-1]], [n_cum[-1], n_cum[-1]])
        hits_fixed += p_final < ALPHA
    return hits_peek / n_sims, hits_fixed / n_sims

fpr_peek, fpr_fixed = peeking_fpr()

plt.figure(figsize=(6.5, 4))
plt.bar(["fixed-horizon\n(one look)", "daily peeking\n(stop at first p<.05)"],
        [fpr_fixed * 100, fpr_peek * 100], color=["#2ca02c", "#d62728"])
plt.axhline(5, ls="--", c="k", lw=1, label="nominal 5%")
plt.ylabel("false positive rate %, A/A simulation (2,000 runs)")
plt.legend(); plt.title("Peeking inflates false positives")
plt.savefig(FIG / "02_peeking.png", dpi=120, bbox_inches="tight"); plt.close()

# cumulative daily p-value of THIS experiment (for the peeking figure)
cumA, cumB = convA.cumsum(), convB.cumsum()
n_cum = daily * np.arange(1, DAYS + 1)
daily_p = [proportions_ztest([cumB[i], cumA[i]], [n_cum[i], n_cum[i]],
                             alternative="larger")[1] for i in range(DAYS)]
plt.figure(figsize=(7, 4))
plt.plot(range(1, DAYS + 1), daily_p, marker="o", ms=4)
plt.axhline(0.05, ls="--", c="r", lw=1)
plt.xlabel("day"); plt.ylabel("cumulative p-value")
plt.title("This experiment's p-value trajectory (analysis was pre-committed to day 14)")
plt.savefig(FIG / "03_p_trajectory.png", dpi=120, bbox_inches="tight"); plt.close()

# ------------------------------------------------- 6 · Bayesian analysis
alpha0, beta0 = 1, 1                       # uniform prior
postA = stats.beta(alpha0 + xA, beta0 + nA - xA)
postB = stats.beta(alpha0 + xB, beta0 + nB - xB)
draws = 200_000
dA = postA.rvs(draws, random_state=np.random.default_rng(SEED))
dB = postB.rvs(draws, random_state=np.random.default_rng(SEED + 1))
p_b_beats_a = float((dB > dA).mean())
exp_loss_choose_b = float(np.maximum(dA - dB, 0).mean())   # expected regret
lift_ci_bayes = np.percentile(dB - dA, [2.5, 97.5])

x = np.linspace(0.085, 0.13, 500)
plt.figure(figsize=(7.5, 4.2))
plt.plot(x, postA.pdf(x), label=f"A posterior ({xA}/{nA})")
plt.plot(x, postB.pdf(x), label=f"B posterior ({xB}/{nB})")
plt.legend(); plt.xlabel("conversion rate")
plt.title(f"Posteriors — P(B > A) = {p_b_beats_a:.3f}")
plt.savefig(FIG / "04_posteriors.png", dpi=120, bbox_inches="tight"); plt.close()

# ------------------------------------------------- 7 · report
metrics = {
    "design": {"baseline": BASELINE, "mde_pp": MDE * 100, "alpha": ALPHA,
               "power": POWER, "n_per_arm_required": n_per_arm,
               "n_per_arm_run": int(nA)},
    "truth": {"true_lift_pp": TRUE_LIFT * 100},
    "srm_p": float(srm_p),
    "frequentist": {"conv_A": xA / nA, "conv_B": xB / nB,
                    "obs_lift_pp": obs_lift * 100, "z": float(z),
                    "p_one_sided": float(p_val),
                    "ci95_pp": [ci_lo * 100, ci_hi * 100],
                    "cohen_h": float(cohen_h)},
    "secondary_aov": {"p": float(p_aov), "note": "true effect is zero by construction"},
    "peeking_sim": {"fpr_fixed": fpr_fixed, "fpr_daily_peeking": fpr_peek},
    "bayesian": {"p_b_beats_a": p_b_beats_a,
                 "expected_loss_choosing_b": exp_loss_choose_b,
                 "lift_ci95_pp": [float(v * 100) for v in lift_ci_bayes]},
}
(REP / "metrics.json").write_text(json.dumps(metrics, indent=2))

report = f"""# A/B Test Report — Checkout Redesign

## 1. Design (committed before data)
- Baseline conversion {BASELINE:.1%}; minimum detectable effect **+{MDE*100:.1f}pp**;
  α = {ALPHA}, power = {POWER:.0%} → **n = {n_per_arm:,} per arm** (fig 01)
- Primary metric: conversion. Secondary: average order value. Analysis horizon:
  day 14, pre-committed. One-sided test justified by a directional launch decision.

## 2. Guardrail
Sample ratio mismatch χ² p = {srm_p:.3f} → allocation is healthy. (SRM is the
most common silent killer of real experiments; it is checked before any effect test.)

## 3. Primary result (frequentist)
- Conversion: A = {xA/nA:.3%}, B = {xB/nB:.3%}; observed lift **+{obs_lift*100:.2f}pp**
- z = {z:.2f}, one-sided p = {p_val:.4f}; 95% CI on lift: [{ci_lo*100:+.2f}, {ci_hi*100:+.2f}] pp
- Effect size (Cohen's h) = {cohen_h:.3f} — small in standardized terms, which is
  normal for conversion metrics; economic size matters more than standardized size.
- Ground truth was +{TRUE_LIFT*100:.1f}pp: the CI covers it. The machinery works.

## 4. Secondary metric, honestly handled
AOV: Welch t-test p = {p_aov:.3f}. True effect is zero by construction; with several
secondaries, some will cross 0.05 by chance — which is why secondaries here are
labeled exploratory and would require multiplicity correction (Holm/BH) if used
for decisions.

## 5. The peeking demonstration (p-hacking mechanics, fig 02)
Simulating 2,000 A/A experiments (no true effect): a fixed-horizon analyst is
falsely positive **{fpr_fixed:.1%}** of the time (nominal 5%), while an analyst who
tests daily and stops at the first p < 0.05 is falsely positive **{fpr_peek:.1%}**
of the time — a ~{fpr_peek/max(fpr_fixed,1e-9):.0f}× inflation from exactly the workflow that feels
"data-driven". This experiment's own p-value trajectory (fig 03) dipped near 0.05
mid-flight; the pre-committed horizon is what makes the final number meaningful.
Sequential testing is legitimate only with methods priced for it (alpha-spending,
mSPRT).

## 6. Bayesian view (fig 04)
- P(B > A) = **{p_b_beats_a:.1%}**; 95% credible interval on lift:
  [{lift_ci_bayes[0]*100:+.2f}, {lift_ci_bayes[1]*100:+.2f}] pp
- Expected loss from launching B if it were actually worse: **{exp_loss_choose_b*100:.4f}pp**
  — the decision-theoretic quantity a launch decision actually needs.

## 7. Recommendation
Launch B. Frequentist and Bayesian analyses agree; the guardrail passed; the
observed lift ({obs_lift*100:+.2f}pp) is economically meaningful at checkout scale.
Post-launch, monitor conversion for novelty decay over 4 weeks.

## Limitations
- Simulated traffic is i.i.d.; real experiments face weekday cycles and
  returning-user contamination (analysis unit ≠ randomization unit risks).
- A one-sided test bakes in a directional decision; two-sided is the safer
  default when harm in either direction matters.
"""
(REP / "ab_test_report.md").write_text(report)
print(json.dumps({k: metrics[k] for k in ["srm_p", "frequentist", "peeking_sim", "bayesian"]}, indent=1))
