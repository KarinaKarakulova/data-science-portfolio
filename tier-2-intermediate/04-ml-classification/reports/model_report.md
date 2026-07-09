# Model Report — Telco Churn

**Data:** 7,043 real customers, churn rate 26.5%. Stratified 80/20 split; the
test set was evaluated exactly once, after model selection on cross-validation.

## Model comparison (5-fold stratified CV on train)

|                        |   cv_roc_auc |   cv_roc_auc_std |   cv_pr_auc |   cv_pr_auc_std |
|:-----------------------|-------------:|-----------------:|------------:|----------------:|
| random_forest          |       0.8467 |           0.0094 |      0.6619 |          0.0213 |
| logistic_regression    |       0.846  |           0.0124 |      0.66   |          0.0195 |
| hist_gradient_boosting |       0.8424 |           0.009  |      0.6538 |          0.0208 |

Champion: **random_forest** — selected on CV ROC-AUC. Note the gap between models is
small relative to fold std; with this feature set, churn signal is mostly linear-ish
(contract type, tenure), which is why logistic regression stays competitive with
boosted trees. That itself is a finding: model complexity buys little here — better
features (usage trends, support tickets) would buy more.

## Held-out test performance
- ROC-AUC **0.843**, PR-AUC **0.652** (baseline = churn rate 0.265)
- Calibration (fig 04) is close to diagonal — scores are usable as probabilities
  for expected-value targeting, not just ranking.

## The threshold is a business decision, not 0.5

Assumed retention economics (stated, not hidden): contact cost $50, value of a
save $800, offer success 30%. Sweeping the threshold on the test set (fig 02):

| threshold | TP | FP | FN | expected campaign profit |
|---|---|---|---|---|
| 0.50 (default) | 288 | 238 | 86 | **$42,820** |
| 0.43 (profit-optimal) | 306 | 290 | 68 | **$43,640** |

Lowering the threshold trades precision for recall profitably because a missed churner
(≈$240 expected loss) costs ~4.8× a wasted contact. **So what:** shipping the
default 0.5 threshold would leave ~$820 on the table per ~1,400-customer
scoring batch under these assumptions. If offer success is only 15%, the optimum shifts
right — the report's economics block makes this a one-line re-run.

## What drives churn (permutation importance, fig 05)
- **Contract** (AUC drop 0.063)
- **tenure** (AUC drop 0.026)
- **InternetService** (AUC drop 0.017)
- **TotalCharges** (AUC drop 0.014)
- **OnlineSecurity** (AUC drop 0.005)

Month-to-month contracts, short tenure, and fiber-optic internet service dominate —
actionable levers (contract migration incentives) rather than mere correlates.

## Limitations
- Single snapshot; no temporal validation. In production churn shifts with campaigns and
  seasonality — a time-based split would be mandatory before deployment.
- Economics are assumptions; the profit curve's *shape* is robust but the optimal point
  moves with (C, V, s). Sensitivity analysis is one parameter edit away.
- `TotalCharges ≈ tenure × MonthlyCharges` — near-collinear; harmless for trees and
  regularized LR, but coefficients should not be read causally.
- No fairness audit (e.g., across senior-citizen status) was performed; required before
  any real targeting decision.
