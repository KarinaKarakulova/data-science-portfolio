"""
Generate a realistic, deliberately messy pharmacy claims dataset.

Why synthetic? Real claims data (IQVIA, Symphony, payer feeds) is proprietary and
PHI-restricted. This generator reproduces the *quality issues* those feeds exhibit
in practice, so the cleaning pipeline solves a realistic problem while remaining
fully reproducible and shareable.

Injected issue classes (each tagged in comments below):
  Q1  Exact duplicate rows (double-submitted claims)
  Q2  Near-duplicates: same claim resubmitted under a new claim_id
  Q3  Inconsistent date formats within one column (ISO, US, text month)
  Q4  Fill date earlier than written date / dates in the future
  Q5  Invalid NDC codes (wrong length, alpha characters, nulls)
  Q6  Negative or zero quantity_dispensed
  Q7  Fat-finger price outliers (unit price x100)
  Q8  Missing values (days_supply, patient_gender, prescriber_npi)
  Q9  Inconsistent categorical encodings (state as 'NY'/'ny'/'New York';
      gender as 'M'/'male'/'MALE'/'F'/'Female'/'U')
  Q10 Whitespace / casing noise in drug names
  Q11 Referential integrity: claims pointing to pharmacy_ids that do not
      exist in the pharmacy dimension

Usage:
    python src/generate_raw_data.py
Outputs CSVs to data/raw/. Fixed seed => byte-identical output on every run.
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

SEED = 42
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

OUT = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

N_CLAIMS = 50_000
N_PHARMACIES = 500
N_DRUGS = 150

STATE_MESS = {  # Q9: three encodings of the same state
    "NY": ["NY", "ny", "New York"],
    "CA": ["CA", "ca", "California"],
    "TX": ["TX", "tx", "Texas"],
    "FL": ["FL", "fl", "Florida"],
    "IL": ["IL", "il", "Illinois"],
    "PA": ["PA", "pa", "Pennsylvania"],
    "NJ": ["NJ", "nj", "New Jersey"],
    "MA": ["MA", "ma", "Massachusetts"],
}
GENDER_MESS = ["M", "male", "MALE", "F", "female", "Female", "U", ""]  # Q8/Q9

THERAPEUTIC_CLASSES = [
    "Cardiovascular", "Diabetes", "Oncology", "CNS",
    "Respiratory", "Immunology", "Anti-Infective",
]


def make_ndc(valid: bool = True) -> str:
    """NDC-11 as 5-4-2. Invalid variants mimic real feed corruption (Q5)."""
    if valid:
        return f"{random.randint(10000, 99999):05d}-{random.randint(0, 9999):04d}-{random.randint(0, 99):02d}"
    kind = random.choice(["short", "alpha", "nodash"])
    if kind == "short":
        return f"{random.randint(100, 9999)}-{random.randint(0, 999)}"
    if kind == "alpha":
        return f"{random.randint(10000, 99999)}-{fake.lexify('????').upper()}-{random.randint(0, 99):02d}"
    return f"{random.randint(10**9, 10**11 - 1)}"


def messy_date(d: date) -> str:
    """Q3: one logical date, three storage formats."""
    fmt = random.choices(["iso", "us", "text"], weights=[0.70, 0.22, 0.08])[0]
    if fmt == "iso":
        return d.isoformat()
    if fmt == "us":
        return d.strftime("%m/%d/%Y")
    return d.strftime("%d-%b-%Y")


# ---------------------------------------------------------------- dimensions
drugs = []
for i in range(N_DRUGS):
    name = fake.unique.lexify("????????").capitalize() + random.choice(["ol", "ine", "mab", "pril", "statin"])
    drugs.append({
        "ndc": make_ndc(valid=True),
        "drug_name": name,
        "therapeutic_class": random.choice(THERAPEUTIC_CLASSES),
        "wac_unit_price": round(random.uniform(0.5, 850.0), 2),  # wholesale acquisition cost
    })

with open(OUT / "drug_master.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=drugs[0].keys())
    w.writeheader()
    w.writerows(drugs)

pharmacies = []
for i in range(1, N_PHARMACIES + 1):
    st = random.choice(list(STATE_MESS))
    pharmacies.append({
        "pharmacy_id": f"PH{i:05d}",
        "pharmacy_name": f"{fake.last_name()} Pharmacy",
        "city": fake.city(),
        "state": st,  # dimension kept clean; mess lives in the fact table
        "channel": random.choices(["Retail", "Mail Order", "Specialty"], weights=[0.75, 0.15, 0.10])[0],
    })

with open(OUT / "pharmacy_master.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=pharmacies[0].keys())
    w.writeheader()
    w.writerows(pharmacies)

# ---------------------------------------------------------------- fact table
rows = []
start = date(2025, 1, 1)

for i in range(1, N_CLAIMS + 1):
    drug = random.choice(drugs)
    pharm = random.choice(pharmacies)

    written = start + timedelta(days=random.randint(0, 360))
    fill = written + timedelta(days=random.randint(0, 14))

    qty = random.choice([30, 30, 30, 60, 90])
    unit_price = round(drug["wac_unit_price"] * random.uniform(0.85, 1.15), 2)
    days_supply = qty  # simple 1/day regimen keeps the check auditable

    row = {
        "claim_id": f"CLM{i:08d}",
        "ndc": drug["ndc"],
        "drug_name_raw": drug["drug_name"],
        "pharmacy_id": pharm["pharmacy_id"],
        "state_raw": random.choice(STATE_MESS[pharm["state"]]),          # Q9
        "patient_gender": random.choice(GENDER_MESS),                    # Q8/Q9
        "patient_age": random.randint(18, 90),
        "prescriber_npi": f"{random.randint(10**9, 10**10 - 1)}",
        "date_written": messy_date(written),                             # Q3
        "date_filled": messy_date(fill),                                 # Q3
        "quantity_dispensed": qty,
        "days_supply": days_supply,
        "unit_price": unit_price,
        "total_paid": round(qty * unit_price, 2),
    }

    # --- targeted corruption ------------------------------------------------
    r = random.random()
    if r < 0.010:                                                         # Q5
        row["ndc"] = make_ndc(valid=False)
    elif r < 0.013:
        row["ndc"] = ""
    if random.random() < 0.008:                                          # Q6
        row["quantity_dispensed"] = random.choice([0, -30, -1])
    if random.random() < 0.004:                                          # Q7
        row["unit_price"] = round(row["unit_price"] * 100, 2)
        row["total_paid"] = round(row["quantity_dispensed"] * row["unit_price"], 2)
    if random.random() < 0.020:                                          # Q8
        row["days_supply"] = ""
    if random.random() < 0.015:                                          # Q8
        row["prescriber_npi"] = ""
    if random.random() < 0.006:                                          # Q4
        row["date_filled"] = messy_date(written - timedelta(days=random.randint(1, 30)))
    if random.random() < 0.003:                                          # Q4
        row["date_filled"] = messy_date(date(2027, random.randint(1, 12), random.randint(1, 28)))
    if random.random() < 0.005:                                          # Q11
        row["pharmacy_id"] = f"PH{random.randint(90000, 99999)}"
    if random.random() < 0.030:                                          # Q10
        name = row["drug_name_raw"]
        row["drug_name_raw"] = random.choice([name.upper(), name.lower(), f"  {name}", f"{name}  "])

    rows.append(row)

# Q1: exact duplicates (~0.6%)
dupes = random.sample(rows, int(N_CLAIMS * 0.006))
rows.extend(dict(d) for d in dupes)

# Q2: resubmissions — same claim, new claim_id (~0.4%)
resubs = random.sample(rows, int(N_CLAIMS * 0.004))
for j, d in enumerate(resubs):
    d2 = dict(d)
    d2["claim_id"] = f"CLM9{j:07d}"
    rows.append(d2)

random.shuffle(rows)

with open(OUT / "pharmacy_claims_raw.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows):,} claim rows, {N_PHARMACIES} pharmacies, {N_DRUGS} drugs -> {OUT}")
