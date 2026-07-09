"""
Generate an IQVIA-flavored commercial pharma dataset:

  dim_hcp           2,500 prescribers: specialty, region, coordinates, decile
  fact_rx_monthly   18 months x HCP: market TRx and our-brand TRx, driven by
                    latent dynamics (loyalists, growers, at-risk switchers,
                    low-potential) that the analysis must recover from data
  fact_calls        rep call activity that is deliberately MISALLOCATED
                    (historic habit: calls follow raw volume, not opportunity)
                    — the business problem the capstone quantifies and fixes.

Synthetic because HCP-level claims/prescriber data (IQVIA Xponent-style) is
strictly licensed; the generator reproduces its structure and pathologies.
Seeded, deterministic. Run: python src/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 21
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

N_HCP, MONTHS, N_REPS = 2500, 18, 20
SPECIALTIES = {"Cardiology": 0.28, "Endocrinology": 0.22, "Internal Medicine": 0.30,
               "Nephrology": 0.12, "Geriatrics": 0.08}
REGIONS = ["Northeast", "South", "Midwest", "West"]

# latent behavioral archetypes (hidden from analysis)
#            share0, share_trend/mo, market_base, label
ARCH = {
    "loyal_high":   (0.42,  0.000, 60, 0.20),
    "grower":       (0.12, +0.012, 45, 0.25),
    "at_risk":      (0.38, -0.015, 55, 0.15),
    "low_potential":(0.15,  0.000, 10, 0.40),
}

hcps, rx, arch_truth = [], [], []
for i in range(1, N_HCP + 1):
    arch = rng.choice(list(ARCH), p=[v[3] for v in ARCH.values()])
    share0, trend, mbase, _ = ARCH[arch]
    spec = rng.choice(list(SPECIALTIES), p=list(SPECIALTIES.values()))
    region = rng.choice(REGIONS)
    lon = {"Northeast": -73, "South": -90, "Midwest": -93, "West": -115}[region] + rng.normal(0, 4)
    lat = {"Northeast": 42, "South": 32, "Midwest": 42, "West": 38}[region] + rng.normal(0, 3)
    mkt_level = max(rng.normal(mbase, mbase * 0.35), 2)
    hcps.append((f"HCP{i:05d}", spec, region, round(lon, 3), round(lat, 3)))
    arch_truth.append((f"HCP{i:05d}", arch))
    share = np.clip(rng.normal(share0, 0.06), 0.02, 0.9)
    for m in range(MONTHS):
        mkt = max(int(rng.normal(mkt_level, mkt_level * 0.15)), 0)
        s = np.clip(share + trend * m + rng.normal(0, 0.02), 0.0, 0.95)
        ours = rng.binomial(mkt, s)
        rx.append((f"HCP{i:05d}", f"2024-{1+m:02d}" if m < 12 else f"2025-{m-11:02d}",
                   mkt, ours))

dim = pd.DataFrame(hcps, columns=["hcp_id", "specialty", "region", "lon", "lat"])
fact = pd.DataFrame(rx, columns=["hcp_id", "month", "market_trx", "our_trx"])

# calls: habit-driven — reps visit where relationships already exist
# (long-standing loyal accounts), with only weak sensitivity to volume and
# none to momentum. This is the documented real-world failure mode the
# analysis must detect: at-risk and headroom accounts are under-called.
vol = fact.groupby("hcp_id").market_trx.mean()
arch_map = dict(arch_truth)
affinity = np.array([1.9 if arch_map[h] == "loyal_high" else
                     0.55 if arch_map[h] in ("at_risk", "grower") else 1.0
                     for h in vol.index])
affinity = affinity * rng.lognormal(0, 0.5, len(vol))
p = affinity * (vol ** 0.35); p = p / p.sum()
total_calls_pm = N_REPS * 120        # capacity: 120 calls/rep/month
calls = []
for m in range(MONTHS):
    alloc = rng.multinomial(total_calls_pm, p.values)
    for hcp, c in zip(p.index, alloc):
        if c:
            calls.append((hcp, f"2024-{1+m:02d}" if m < 12 else f"2025-{m-11:02d}", int(c)))
callsf = pd.DataFrame(calls, columns=["hcp_id", "month", "calls"])

# territory (rep) assignment: geographic k-means-ish by rounding coordinates
dim["territory"] = ("T" + (pd.qcut(dim.lon, 5, labels=False).astype(int) * 4
                    + pd.qcut(dim.lat, 4, labels=False).astype(int)).astype(str))

dim.to_csv(OUT / "dim_hcp.csv", index=False)
fact.to_csv(OUT / "fact_rx_monthly.csv", index=False)
callsf.to_csv(OUT / "fact_calls.csv", index=False)
pd.DataFrame(arch_truth, columns=["hcp_id", "true_archetype"]).to_csv(
    OUT / "_hidden_archetypes.csv", index=False)   # used only for validation
print(f"hcps={len(dim):,} rx_rows={len(fact):,} call_rows={len(callsf):,} "
      f"territories={dim.territory.nunique()}")
