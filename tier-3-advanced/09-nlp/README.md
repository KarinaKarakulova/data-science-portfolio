# 09 · NLP — SMS Spam Detection (Real Data, Precision-First)

Text classification on 5,572 real SMS messages (13.4% spam), framed as an abuse filter where **blocking a legitimate message is the costly error** — so evaluation and thresholding are precision-first, not accuracy-first.

## Results
| model | CV PR-AUC |
|---|---|
| **char 3–5-gram TF-IDF + LinearSVC** | **0.987 ± 0.004** |
| word 1–2-gram TF-IDF + LogReg | 0.982 ± 0.005 |
| MultinomialNB (baseline) | 0.958 ± 0.008 |

- Char n-grams win for a stated linguistic reason: spam obfuscates tokens ("FR33", "W1N £1000") and subword patterns survive obfuscation
- Champion calibrated (Platt) so a probability threshold is settable; deployed operating point: **precision 99.3% / recall 94.6% / 1 false positive** on the held-out set
- **Error analysis with the actual texts** — the report prints every would-be-blocked legitimate message and the ten hardest missed spams, and identifies the pattern: conversational-style fraud without prize vocabulary is exactly where n-grams fail and transformer embeddings would pay
- Vectorizers live inside the CV pipeline (no test-vocabulary leakage); PR-AUC used because ROC flatters at 13% positives

## The transformer question, answered honestly
This repo's evaluation harness is model-agnostic; the documented next step is DistilBERT fine-tuning on the identical split (expected gains concentrated in the conversational-fraud FN class shown in the error analysis). It isn't run here because the build environment cannot reach pretrained-weight hosts — the report says so plainly rather than pretending n-grams are the ceiling.

Report: [`reports/nlp_report.md`](reports/nlp_report.md) · Run: `python src/spam_model.py`
