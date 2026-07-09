# 04 · ML Classification — Customer Churn (Real Data)

Binary churn prediction on the IBM Telco dataset (7,043 real customers, 26.5% churn). The emphasis is on the parts of applied ML that separate practitioners from tutorials: leakage-safe pipelines, honest model selection, calibrated probabilities, and a decision threshold derived from retention economics instead of the 0.5 default.

## Results
- Three model families (regularized logistic regression, random forest, gradient boosting) compared under **5-fold stratified CV with all preprocessing inside the fold** (ColumnTransformer + Pipeline — no leakage)
- Champion: random forest, **test ROC-AUC 0.843 / PR-AUC 0.66** vs 0.265 baseline — but the honest headline is that all three models land within one fold-std of each other, meaning the churn signal in this feature set is largely linear (contract, tenure) and better *features* would beat better *models*
- **Profit-optimal threshold ≈ 0.43** under stated economics ($50 contact, $800 save value, 30% offer success); the profit curve (`figures/02_threshold_profit.png`) makes the precision/recall trade-off a dollar-denominated decision
- Calibration verified (fig 04) so scores work as probabilities for expected-value targeting
- Permutation importance on the held-out set identifies actionable levers: contract type, tenure, internet service

Full narrative with limitations (temporal validity, collinearity, fairness audit gap): [`reports/model_report.md`](reports/model_report.md)

## Run
```bash
python src/churn_model.py   # seeded; regenerates figures + reports + metrics.json
```

## Techniques
Stratified CV · sklearn Pipeline/ColumnTransformer · class weighting · ROC/PR analysis · cost-sensitive threshold optimization · calibration curves · permutation importance · one-shot test-set discipline
