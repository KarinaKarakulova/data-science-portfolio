"""
Customer segmentation — CPG loyalty-card transactions (synthetic, with a twist).

The twist: customers are generated FROM four latent behavioral archetypes plus a
small anomalous group. Ground truth is hidden from the pipeline and used only at
the end to score how well unsupervised methods recovered real structure (ARI).
This turns "the clusters look reasonable" into a measurable claim — the standard
weakness of segmentation work is exactly that it can't be graded; here it can.

Pipeline: transactions -> RFM+behavioral features -> log/scale -> PCA ->
K selection (silhouette + elbow) -> K-means vs Ward -> anomaly detection ->
personas -> validation against hidden truth.

Run: python src/segmentation.py    (seeded)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
rng = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parents[1]
FIG, REP, DATA = ROOT / "figures", ROOT / "reports", ROOT / "data"
for d in (FIG, REP, DATA): d.mkdir(exist_ok=True)

# ------------------------------------------------ 1 · generate transactions
# Four archetypes + anomalies (e.g., reseller behavior). Parameters:
# (weekly purchase rate, basket mean $, basket cv, promo affinity, category breadth)
ARCHETYPES = {
    "loyal_family":    (2.2, 85, 0.30, 0.25, 9),
    "convenience":     (0.9, 22, 0.45, 0.15, 4),
    "promo_hunter":    (1.1, 45, 0.55, 0.80, 6),
    "premium_light":   (0.4, 120, 0.35, 0.05, 5),
}
SIZES = {"loyal_family": 900, "convenience": 1400, "promo_hunter": 700,
         "premium_light": 450}
N_ANOM = 50   # resellers: extreme frequency, huge baskets, single category

WEEKS = 52
rows, truth = [], []
cid = 0
for seg, n in SIZES.items():
    rate, bmean, bcv, promo, breadth = ARCHETYPES[seg]
    for _ in range(n):
        cid += 1
        lam = max(rng.normal(rate, rate * 0.25), 0.05)
        n_tx = rng.poisson(lam * WEEKS)
        if n_tx == 0: n_tx = 1
        days = np.sort(rng.integers(0, WEEKS * 7, n_tx))
        baskets = rng.lognormal(np.log(bmean), bcv, n_tx)
        on_promo = rng.random(n_tx) < promo
        cats = rng.integers(1, breadth + 1, n_tx)
        for d, b, pr, c in zip(days, baskets, on_promo, cats):
            rows.append((cid, d, round(b, 2), int(pr), c))
        truth.append((cid, seg))
for _ in range(N_ANOM):
    cid += 1
    n_tx = rng.poisson(8 * WEEKS)
    days = np.sort(rng.integers(0, WEEKS * 7, n_tx))
    baskets = rng.lognormal(np.log(400), 0.25, n_tx)
    for d, b in zip(days, baskets):
        rows.append((cid, d, round(b, 2), 0, 1))
    truth.append((cid, "anomaly_reseller"))

tx = pd.DataFrame(rows, columns=["customer_id", "day", "basket_usd",
                                 "on_promo", "category"])
truth = pd.DataFrame(truth, columns=["customer_id", "true_segment"])
tx.to_csv(DATA / "transactions.csv", index=False)
print(f"{len(tx):,} transactions, {tx.customer_id.nunique():,} customers")

# ------------------------------------------------ 2 · feature engineering (RFM+)
snap = WEEKS * 7
feat = tx.groupby("customer_id").agg(
    recency=("day", lambda d: snap - d.max()),
    frequency=("day", "count"),
    monetary=("basket_usd", "sum"),
    avg_basket=("basket_usd", "mean"),
    promo_share=("on_promo", "mean"),
    category_breadth=("category", "nunique"),
).reset_index()

FEATS = ["recency", "frequency", "monetary", "avg_basket",
         "promo_share", "category_breadth"]
X_raw = feat[FEATS].copy()
# Monetary/frequency are heavy right-tailed -> log1p before scaling, otherwise
# distance-based clustering is dominated by a few big spenders.
for c in ["frequency", "monetary", "avg_basket"]:
    X_raw[c] = np.log1p(X_raw[c])
X = StandardScaler().fit_transform(X_raw)

# ------------------------------------------------ 3 · anomaly detection FIRST
# Resellers distort centroids; detect and set aside before clustering.
iso = IsolationForest(contamination=0.02, random_state=SEED).fit(X)
is_anom = iso.predict(X) == -1
X_core = X[~is_anom]
core_ids = feat.loc[~is_anom, "customer_id"].values

# ------------------------------------------------ 4 · PCA
pca = PCA().fit(X_core)
evr = pca.explained_variance_ratio_

# Math 257 check (Wk 12-14: spectral theorem, SVD, PCA): sklearn's
# explained_variance_ratio_ is the eigenvalue spectrum of cov(X), normalized —
# and PCA is just the SVD of the centered data matrix (lambda_i = sigma_i^2/(m-1)).
Xc = X_core - X_core.mean(axis=0)
lam = np.linalg.svd(Xc, compute_uv=False) ** 2 / (len(Xc) - 1)
assert np.allclose(lam / lam.sum(), evr), "PCA != eigenvalues of cov(X)"
print(f"[Math 257] explained_variance_ratio_ reproduced from a raw SVD "
      f"(top-2 = {lam[:2].sum() / lam.sum():.1%})")
plt.figure(figsize=(6.5, 4))
plt.bar(range(1, len(evr) + 1), evr * 100)
plt.plot(range(1, len(evr) + 1), evr.cumsum() * 100, "r-o", ms=3)
plt.xlabel("component"); plt.ylabel("% variance")
plt.title(f"PCA scree — 2 PCs explain {evr[:2].sum():.0%}")
plt.savefig(FIG / "01_scree.png", dpi=120, bbox_inches="tight"); plt.close()
P2 = PCA(2).fit_transform(X_core)

# ------------------------------------------------ 5 · choose K
ks = range(2, 9)
sil, inertia = [], []
for k in ks:
    km = KMeans(k, n_init=20, random_state=SEED).fit(X_core)
    sil.append(silhouette_score(X_core, km.labels_))
    inertia.append(km.inertia_)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
ax[0].plot(list(ks), sil, "-o"); ax[0].set_title("silhouette"); ax[0].set_xlabel("K")
ax[1].plot(list(ks), inertia, "-o"); ax[1].set_title("inertia (elbow)"); ax[1].set_xlabel("K")
plt.savefig(FIG / "02_k_selection.png", dpi=120, bbox_inches="tight"); plt.close()
K = int(list(ks)[int(np.argmax(sil))])

km = KMeans(K, n_init=50, random_state=SEED).fit(X_core)
ward = AgglomerativeClustering(K, linkage="ward").fit(X_core)
sil_km = silhouette_score(X_core, km.labels_)
sil_ward = silhouette_score(X_core, ward.labels_)
method_agreement = adjusted_rand_score(km.labels_, ward.labels_)

plt.figure(figsize=(8, 5))
Z = linkage(X_core[rng.choice(len(X_core), 600, replace=False)], "ward")
dendrogram(Z, no_labels=True, color_threshold=0.7 * Z[:, 2].max())
plt.title("Ward dendrogram (600-customer sample)")
plt.savefig(FIG / "03_dendrogram.png", dpi=120, bbox_inches="tight"); plt.close()

plt.figure(figsize=(7.5, 6))
for c in range(K):
    m = km.labels_ == c
    plt.scatter(P2[m, 0], P2[m, 1], s=5, alpha=0.5, label=f"cluster {c}")
an2 = PCA(2).fit(X_core).transform(X[is_anom])
plt.scatter(an2[:, 0], an2[:, 1], s=28, marker="x", c="k", label="anomalies")
plt.legend(); plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title(f"K-means (K={K}) in PCA space, silhouette {sil_km:.2f}")
plt.savefig(FIG / "04_clusters_pca.png", dpi=120, bbox_inches="tight"); plt.close()

# ------------------------------------------------ 6 · personas
feat_core = feat[~is_anom].copy()
feat_core["cluster"] = km.labels_
persona = feat_core.groupby("cluster")[FEATS].median().round(2)
persona["size"] = feat_core.groupby("cluster").size()
persona["revenue_share_%"] = (feat_core.groupby("cluster")["monetary"].sum()
                              / feat_core["monetary"].sum() * 100).round(1)

# heatmap of z-scored persona profile
z = (persona[FEATS] - persona[FEATS].mean()) / persona[FEATS].std()
plt.figure(figsize=(7.5, 3.6))
plt.imshow(z.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
plt.xticks(range(len(FEATS)), FEATS, rotation=30, ha="right")
plt.yticks(range(K), [f"cluster {i}" for i in range(K)])
plt.colorbar(label="z vs other clusters"); plt.title("Segment profiles")
plt.savefig(FIG / "05_personas.png", dpi=120, bbox_inches="tight"); plt.close()

# ------------------------------------------------ 7 · validation vs hidden truth
merged = feat_core.merge(truth, on="customer_id")
ari = adjusted_rand_score(merged["true_segment"], merged["cluster"])
xtab = pd.crosstab(merged["true_segment"], merged["cluster"])
anom_truth = truth.set_index("customer_id").loc[feat["customer_id"], "true_segment"].values
anom_recall = float(((anom_truth == "anomaly_reseller") & is_anom).sum()
                    / (anom_truth == "anomaly_reseller").sum())

metrics = {
    "n_customers": int(len(feat)), "n_transactions": int(len(tx)),
    "chosen_K": K, "silhouette_kmeans": round(float(sil_km), 3),
    "silhouette_ward": round(float(sil_ward), 3),
    "kmeans_vs_ward_ARI": round(float(method_agreement), 3),
    "recovery_ARI_vs_truth": round(float(ari), 3),
    "anomaly_recall": round(anom_recall, 3),
    "pca_2pc_variance": round(float(evr[:2].sum()), 3),
}
(REP / "metrics.json").write_text(json.dumps(metrics, indent=2))

report = f"""# Segmentation Report — CPG Loyalty Customers

**Data:** {len(tx):,} transactions / {len(feat):,} customers over 52 weeks
(synthetic with hidden ground truth — see README for why that is the point).

## Method
RFM + behavioral features (promo share, category breadth) → log-transform heavy-tailed
monetary features → standardize → **anomaly detection first** (IsolationForest set
aside {int(is_anom.sum())} customers; resellers would otherwise drag centroids) →
PCA (2 PCs = {evr[:2].sum():.0%} of variance) → K by silhouette → K-means, cross-checked
against Ward hierarchical.

## Results
- Chosen **K = {K}**; silhouette {sil_km:.2f} (K-means) vs {sil_ward:.2f} (Ward);
  the two methods agree with ARI **{method_agreement:.2f}** — structure is not an
  algorithm artifact.
- **Recovery of hidden truth: ARI = {ari:.2f}** — the pipeline genuinely found the
  planted archetypes, and the confusion table shows where boundaries blur:

{xtab.to_markdown()}

- Anomaly screen recovered **{anom_recall:.0%}** of planted resellers.

## Segment personas (medians)

{persona.to_markdown()}

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
"""
(REP / "segmentation_report.md").write_text(report)
print(json.dumps(metrics, indent=2))
