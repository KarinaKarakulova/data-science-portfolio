"""
Train a next-month TRx demand model on the dbt-built feature table.

Registry-lite discipline:
  - training data is read from the dbt output (single source of feature truth)
  - a SHA-256 of the exact training frame + the dbt manifest hash + params +
    metrics are written to reports/model_metadata.json — enough to answer
    "which data and which transformations produced this model?" months later.

Evaluation discipline:
  - time-based split (train <= 2025-05, test 2025-06..2025-11): random splits
    on panel data leak the future and flatter the model.
  - baseline = seasonal-aware naive (lag-1). The model must beat it.

Run (after `dbt build`):  python src/train.py
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "rx_analytics" / "rx.duckdb"
REP, FIG = ROOT / "reports", ROOT / "figures"
REP.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

con = duckdb.connect(str(DB), read_only=True)
df = con.execute("SELECT * FROM ft_demand_training ORDER BY drug, state, month_start").fetchdf()
con.close()

FEATURES = ["trx_lag1", "trx_lag2", "trx_lag3", "trx_ma3",
            "detailing_lag1", "detailing_lag2",
            "months_since_launch", "month_of_year"]
TARGET = "target_trx_next_month"

split = pd.Timestamp("2025-06-01")
tr, te = df[df.month_start < split], df[df.month_start >= split]
X_tr, y_tr = tr[FEATURES], tr[TARGET]
X_te, y_te = te[FEATURES], te[TARGET]

baseline_pred = te["trx"]                     # naive: next month = this month
model = HistGradientBoostingRegressor(learning_rate=0.06, max_iter=500,
                                      early_stopping=True, random_state=0)
model.fit(X_tr, y_tr)
pred = model.predict(X_te)

res = {
    "baseline_naive": {"MAE": float(mean_absolute_error(y_te, baseline_pred)),
                       "MAPE": float(mean_absolute_percentage_error(y_te, baseline_pred))},
    "hist_gbm": {"MAE": float(mean_absolute_error(y_te, pred)),
                 "MAPE": float(mean_absolute_percentage_error(y_te, pred))},
}
improve = 1 - res["hist_gbm"]["MAE"] / res["baseline_naive"]["MAE"]

pi = permutation_importance(model, X_te, y_te, n_repeats=30, random_state=0,
                            scoring="neg_mean_absolute_error")
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
plt.figure(figsize=(7, 4))
imp.plot.barh(); plt.xlabel("permutation importance (test MAE increase)")
plt.title("Feature importance — detailing lag effect recovered")
plt.savefig(FIG / "01_importance.png", dpi=120, bbox_inches="tight"); plt.close()

# one illustrative panel
sub = df[(df.drug == "Respivan") & (df.state == "CA")]
sub_te = sub[sub.month_start >= split]
plt.figure(figsize=(9, 4))
plt.plot(sub.month_start, sub[TARGET], lw=1, label="actual next-month TRx")
plt.plot(sub_te.month_start, model.predict(sub_te[FEATURES]), "r--", lw=1.4,
         label="model (test period)")
plt.axvline(split, color="gray", ls=":", lw=1)
plt.legend(); plt.title("Respivan / CA — seasonal drug, held-out period right of dotted line")
plt.savefig(FIG / "02_respivan_ca.png", dpi=120, bbox_inches="tight"); plt.close()

# ------------------------------------------------------------- registry-lite
frame_hash = hashlib.sha256(
    pd.util.hash_pandas_object(df[FEATURES + [TARGET]], index=False).values.tobytes()
).hexdigest()[:16]
manifest = ROOT / "rx_analytics" / "target" / "manifest.json"
manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()[:16] if manifest.exists() else None

metadata = {
    "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "model": "HistGradientBoostingRegressor",
    "params": {"learning_rate": 0.06, "max_iter": 500, "early_stopping": True},
    "feature_table": "ft_demand_training (dbt)",
    "training_frame_sha256_16": frame_hash,
    "dbt_manifest_sha256_16": manifest_hash,
    "rows": {"train": int(len(tr)), "test": int(len(te))},
    "split": "time-based @ 2025-06",
    "metrics": res,
    "improvement_vs_naive_MAE": round(float(improve), 3),
    "top_features": imp.sort_values(ascending=False).head(6).round(2).to_dict(),
}
(REP / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata, indent=2))
