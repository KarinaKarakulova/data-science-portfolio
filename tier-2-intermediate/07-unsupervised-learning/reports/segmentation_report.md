# Segmentation Report — CPG Loyalty Customers

**Data:** 237,431 transactions / 3,500 customers over 52 weeks
(synthetic with hidden ground truth — see README for why that is the point).

## Method
RFM + behavioral features (promo share, category breadth) → log-transform heavy-tailed
monetary features → standardize → **anomaly detection first** (IsolationForest set
aside 70 customers; resellers would otherwise drag centroids) →
PCA (2 PCs = 74% of variance) → K by silhouette → K-means, cross-checked
against Ward hierarchical.

## Results
- Chosen **K = 5**; silhouette 0.67 (K-means) vs 0.67 (Ward);
  the two methods agree with ARI **0.98** — structure is not an
  algorithm artifact.
- **Recovery of hidden truth: ARI = 0.97** — the pipeline genuinely found the
  planted archetypes, and the confusion table shows where boundaries blur:

| true_segment   |    0 |   1 |   2 |   3 |   4 |
|:---------------|-----:|----:|----:|----:|----:|
| convenience    | 1371 |   0 |   0 |   0 |  24 |
| loyal_family   |    0 |   0 | 895 |   1 |   3 |
| premium_light  |    0 |   0 |   0 | 348 |  89 |
| promo_hunter   |    0 | 696 |   0 |   0 |   3 |

- Anomaly screen recovered **100%** of planted resellers.

## Segment personas (medians)

|   cluster |   recency |   frequency |   monetary |   avg_basket |   promo_share |   category_breadth |   size |   revenue_share_% |
|----------:|----------:|------------:|-----------:|-------------:|--------------:|-------------------:|-------:|------------------:|
|         0 |         6 |          46 |    1121.42 |        24.26 |          0.15 |                  4 |   1371 |              11.2 |
|         1 |         5 |          57 |    2951.12 |        52.15 |          0.81 |                  6 |    696 |              15.2 |
|         2 |         3 |         113 |    9996.1  |        88.78 |          0.25 |                  9 |    895 |              64.9 |
|         3 |         8 |          22 |    2754.67 |       126.29 |          0.05 |                  5 |    349 |               7   |
|         4 |        43 |          20 |    1854.94 |       123.63 |          0.07 |                  5 |    119 |               1.7 |

## Business translation
- The highest-frequency, broad-basket segment concentrates a disproportionate
  revenue share — retention economics justify a loyalty investment there.
- The promo-affine segment (high promo_share) is where discount depth should be
  tested, not sprayed across all segments — cross-reference Project 06's
  experimentation framework.
- Anomalous accounts merit a policy review (reseller terms), not marketing spend.

## Limitations
- Real loyalty data adds seasonality, churned customers, and household sharing —
  all absent here by design.
- Silhouette rewards convex, similar-size clusters; density-based methods (HDBSCAN)
  would be the cross-check if real data showed irregular shapes.
- K-means on standardized features implies Euclidean trade-offs between, e.g.,
  1 unit of recency and 1 unit of promo share — a modeling choice, stated openly.
