# 07 · Unsupervised Learning — CPG Customer Segmentation (Gradable)

Segmentation's classic weakness is that it can't be graded — "the clusters look reasonable" is not evidence. This project fixes that: 237k loyalty-card transactions are generated from **four hidden behavioral archetypes plus a planted reseller-anomaly group**, the pipeline never sees the labels, and recovery is scored at the end.

## Results
- **Recovery ARI vs hidden truth: 0.97**; planted anomalies recovered at **100% recall**
- K-means and Ward hierarchical agree at ARI **0.98** — structure is real, not an algorithm artifact
- Silhouette selected **K=5** where truth had 4 archetypes — the confusion table in the report shows exactly which archetype the extra cluster split, and why that's a finding rather than a failure
- Anomaly detection runs **before** clustering (IsolationForest) because resellers drag centroids — an ordering decision most tutorials get wrong
- Heavy-tailed monetary features are log-transformed before scaling, with the distance-geometry reasoning stated

## Pipeline
transactions → RFM + behavioral features (promo affinity, category breadth) → log1p + standardize → anomaly screen → PCA (2 PCs = 74% variance) → K by silhouette/elbow → K-means × Ward cross-check → personas with revenue share → validation vs truth

Report with personas table, confusion matrix, and business actions per segment: [`reports/segmentation_report.md`](reports/segmentation_report.md)

## Run
```bash
python src/segmentation.py    # regenerates data, figures, reports (seeded)
```
