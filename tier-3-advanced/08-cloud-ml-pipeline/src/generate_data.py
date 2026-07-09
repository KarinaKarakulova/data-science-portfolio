"""
Generate a monthly pharmaceutical demand panel: drug x state x month TRx counts
with realistic structure the feature pipeline must recover:
  - drug-level launch curves (adoption ramps)
  - annual seasonality (respiratory drugs peak in winter)
  - state size effects
  - detailing (rep promotion) with a LAGGED effect on demand — the causal-ish
    signal the model should find via lag features
  - noise

Seeded and deterministic. Output: data/rx_monthly_raw.csv (long format).
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 11
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

MONTHS = pd.period_range("2022-01", "2025-12", freq="M")
STATES = {"CA": 1.9, "TX": 1.5, "FL": 1.2, "NY": 1.1, "PA": 0.8,
          "IL": 0.7, "NJ": 0.55, "MA": 0.45, "GA": 0.65, "NC": 0.6,
          "OH": 0.7, "MI": 0.6, "WA": 0.5, "AZ": 0.45, "VA": 0.5}
DRUGS = {
    # name: (class, base_demand, launch_month_idx, seasonal_amp)
    "Cardexa":  ("Cardiovascular", 900, 0,  0.05),
    "Glucora":  ("Diabetes",      1300, 0,  0.03),
    "Respivan": ("Respiratory",    700, 0,  0.35),
    "Neurilex": ("CNS",            500, 10, 0.04),
    "Immunara": ("Immunology",     300, 22, 0.05),
}

rows = []
for drug, (tclass, base, launch, amp) in DRUGS.items():
    # detailing intensity by state-month (rep calls, in hundreds)
    for state, size in STATES.items():
        # Detailing arrives in campaign waves (quarterly pushes with random
        # phase + occasional blitzes), not as a flat level — this is what makes
        # promo lags a learnable signal distinct from the demand level itself.
        detail_level = rng.uniform(0.5, 1.2)
        phase = rng.uniform(0, 2 * np.pi)
        for i, m in enumerate(MONTHS):
            t = i - launch
            if t < 0:
                continue
            ramp = 1 - np.exp(-t / 8)                       # adoption curve
            season = 1 + amp * np.cos(2 * np.pi * (m.month - 1) / 12)
            wave = 0.6 * np.sin(2 * np.pi * i / 3 + phase)   # quarterly cadence
            blitz = 1.2 if rng.random() < 0.08 else 0.0
            detailing = max(rng.normal(detail_level + wave + blitz, 0.15), 0.02)
            rows.append({"month": str(m), "drug": drug, "state": state,
                         "detailing_calls_100s": round(detailing, 3),
                         "_ramp": ramp, "_season": season, "_size": size,
                         "_base": base})

panel = pd.DataFrame(rows).sort_values(["drug", "state", "month"]).reset_index(drop=True)
# lagged promo effect: demand responds to detailing 1-2 months ago
panel["_d1"] = panel.groupby(["drug", "state"])["detailing_calls_100s"].shift(1)
panel["_d2"] = panel.groupby(["drug", "state"])["detailing_calls_100s"].shift(2)
promo_mult = 1 + 0.16 * panel["_d1"].fillna(0.0) + 0.08 * panel["_d2"].fillna(0.0)

mu = panel["_base"] * panel["_size"] * panel["_ramp"] * panel["_season"] * promo_mult
panel["trx"] = rng.poisson(np.maximum(mu, 1))

panel[["month", "drug", "state", "detailing_calls_100s", "trx"]].to_csv(
    OUT / "rx_monthly_raw.csv", index=False)

drug_dim = pd.DataFrame(
    [(d, v[0]) for d, v in DRUGS.items()], columns=["drug", "therapeutic_class"])
drug_dim.to_csv(OUT / "drug_dim.csv", index=False)
print(f"panel rows: {len(panel):,}  months: {len(MONTHS)}  drugs: {len(DRUGS)}  states: {len(STATES)}")
