# NLP Report — SMS Spam Detection

**Data:** 5,572 real SMS (13.4% spam). Stratified 80/20; vectorizers live
inside the CV pipeline so folds never see test vocabulary.

## Model comparison (5-fold CV, PR-AUC — the right metric at 13% positives)

| model | CV PR-AUC |
|---|---|
| nb_word | 0.9577 ± 0.0083 |
| logreg_word12 | 0.9815 ± 0.0054 |
| svc_char35 | 0.9867 ± 0.0035 |

**Champion: svc_char35.** Character n-grams win because spam obfuscates words
("FR33", "W1N £1000") — subword patterns survive obfuscation; whole-word models
don't. NB is the classic strong-baseline sanity check.

## Held-out test
- PR-AUC **0.988**, ROC-AUC **0.996**
- Deployed operating point chosen precision-first (blocking real messages is the
  costly error): threshold 0.48 → precision **99.3%**,
  recall **94.6%**, false positives on test: **1**

## Error analysis (actual test-set texts)

False positives (legitimate messages the filter would block):
- p=0.48 : 'K..u also dont msg or reply to his msg..'

Hardest false negatives (spam that slipped through, lowest scores):
- p=0.00 : "Do you realize that in about 40 years, we'll have thousands of old ladies running around with tattoos?"
- p=0.01 : "Sorry I missed your call let's talk when you have the time. I'm on 07090201529"
- p=0.03 : 'ROMCAPspam Everyone around should be responding well to your presence since you are so warm and outgoing. You '
- p=0.08 : "RCT' THNQ Adrian for U text. Rgds Vatian"
- p=0.08 : "Hi ya babe x u 4goten bout me?' scammers getting smart..Though this is a regular vodafone no, if you respond y"
- p=0.10 : 'For sale - arsenal dartboard. Good condition but no doubles or trebles!'
- p=0.16 : 'Email AlertFrom: Jeri StewartSize: 2KBSubject: Low-cost prescripiton drvgsTo listen to email call 123'
- p=0.27 : 'Latest News! Police station toilet stolen, cops have nothing to go on!'

Pattern: missed spam is mostly *conversational-style* fraud without telltale
tokens (no CAPS, no prize vocabulary) — exactly the class where contextual
embeddings (transformer fine-tuning) buy accuracy that n-grams cannot.

## Strongest learned signals
- spam: txt, call, text, reply, uk, free, www, claim, stop, 150p
- ham: my, me, ok, that, gt, lt, can, ll, but, da

## Limitations & the transformer path
- This corpus is 2000s-era UK SMS; vocabulary drift is severe in production —
  a deployed filter needs continuous retraining and drift monitoring.
- The documented next step is fine-tuning a small transformer (e.g.
  DistilBERT/DeBERTa-v3-small via HuggingFace `Trainer`) with the identical
  split and PR-first evaluation; expected gains concentrate exactly in the
  conversational-fraud FN class shown above. Not run in this repo because the
  build environment has no access to pretrained-weight hosts; the evaluation
  harness here is model-agnostic and ready for it.
- Single language; multilingual abuse needs char-level or multilingual models.
