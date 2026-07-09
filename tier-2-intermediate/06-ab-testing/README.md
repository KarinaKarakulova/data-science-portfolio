# 06 · Statistical Analysis & A/B Testing — Full Experiment Lifecycle

A checkout-redesign experiment executed the way experiments should be run: design and power analysis committed **before** data exists, guardrails checked before effect tests, frequentist and Bayesian analyses side by side, and a simulation that demonstrates exactly how p-hacking happens.

**Why simulated data (and why that's the right call here):** the ground truth lift is known (+1.2pp), so the report can *verify* that every method recovers it — and the peeking demonstration requires running 2,000 counterfactual A/A experiments, which no single real dataset permits.

## What's inside
1. **Power analysis first:** baseline 10%, MDE +1.0pp, α=0.05, power 80% → n per arm derived, with a power curve showing what the design can and cannot detect
2. **SRM guardrail** (χ² on allocation) before any effect test — the most common silent killer of real experiments
3. **Frequentist:** two-proportion z-test, 95% CI on the lift **[+0.98, +2.39]pp** (covers the +1.2pp truth), Cohen's h, and a null secondary metric handled honestly (labeled exploratory, multiplicity discussed)
4. **The peeking demonstration:** 2,000 simulated A/A tests show a fixed-horizon analyst is falsely positive **4.8%** of the time vs **22.4%** for one who checks daily and stops at the first p<0.05 — a ~5× inflation from a workflow that feels rigorous (`figures/02_peeking.png`)
5. **Bayesian:** Beta-Binomial posteriors, **P(B>A) > 99.99%**, expected loss of a wrong launch ≈ 0 — the decision-theoretic quantity a launch call actually needs; credible interval matches the frequentist CI under a flat prior, as theory says it must
6. **Recommendation with limitations:** launch B; monitor novelty decay; i.i.d.-traffic and randomization-unit caveats stated

Report: [`reports/ab_test_report.md`](reports/ab_test_report.md) · Run: `python src/ab_test.py`
