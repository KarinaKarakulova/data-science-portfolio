# A/B Test Report — Checkout Redesign

## 1. Design (committed before data)
- Baseline conversion 10.0%; minimum detectable effect **+1.0pp**;
  α = 0.05, power = 80% → **n = 14,745 per arm** (fig 01)
- Primary metric: conversion. Secondary: average order value. Analysis horizon:
  day 14, pre-committed. One-sided test justified by a directional launch decision.

## 2. Guardrail
Sample ratio mismatch χ² p = 1.000 → allocation is healthy. (SRM is the
most common silent killer of real experiments; it is checked before any effect test.)

## 3. Primary result (frequentist)
- Conversion: A = 9.901%, B = 11.589%; observed lift **+1.69pp**
- z = 4.68, one-sided p = 0.0000; 95% CI on lift: [+0.98, +2.39] pp
- Effect size (Cohen's h) = 0.055 — small in standardized terms, which is
  normal for conversion metrics; economic size matters more than standardized size.
- Ground truth was +1.2pp: the CI covers it. The machinery works.

## 4. Secondary metric, honestly handled
AOV: Welch t-test p = 0.175. True effect is zero by construction; with several
secondaries, some will cross 0.05 by chance — which is why secondaries here are
labeled exploratory and would require multiplicity correction (Holm/BH) if used
for decisions.

## 5. The peeking demonstration (p-hacking mechanics, fig 02)
Simulating 2,000 A/A experiments (no true effect): a fixed-horizon analyst is
falsely positive **4.8%** of the time (nominal 5%), while an analyst who
tests daily and stops at the first p < 0.05 is falsely positive **22.4%**
of the time — a ~5× inflation from exactly the workflow that feels
"data-driven". This experiment's own p-value trajectory (fig 03) dipped near 0.05
mid-flight; the pre-committed horizon is what makes the final number meaningful.
Sequential testing is legitimate only with methods priced for it (alpha-spending,
mSPRT).

## 6. Bayesian view (fig 04)
- P(B > A) = **100.0%**; 95% credible interval on lift:
  [+0.98, +2.39] pp
- Expected loss from launching B if it were actually worse: **0.0000pp**
  — the decision-theoretic quantity a launch decision actually needs.

## 7. Recommendation
Launch B. Frequentist and Bayesian analyses agree; the guardrail passed; the
observed lift (+1.69pp) is economically meaningful at checkout scale.
Post-launch, monitor conversion for novelty decay over 4 weeks.

## Limitations
- Simulated traffic is i.i.d.; real experiments face weekday cycles and
  returning-user contamination (analysis unit ≠ randomization unit risks).
- A one-sided test bakes in a directional decision; two-sided is the safer
  default when harm in either direction matters.
