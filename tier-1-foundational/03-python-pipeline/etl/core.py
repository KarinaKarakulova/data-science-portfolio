"""
Macro-Financial Data Mart — ETL package.

Tasks form an explicit DAG (see run_etl.py). Each layer is a pure function:
    extract_*   : source -> cached raw file          (idempotent, cached)
    transform_* : raw file -> typed DataFrame        (no I/O side effects)
    load        : DataFrames -> SQLite star schema   (transactional, replace)
    validate    : warehouse -> pass/fail checks      (pipeline gate)
    report      : warehouse -> figures + report.md

Design choices documented inline; assumptions in ASSUMPTIONS.md.
"""

import hashlib
import sqlite3
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
DB = ROOT / "warehouse" / "macro_mart.db"

SOURCES = {
    "gdp": "https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv",
    "inflation": "https://raw.githubusercontent.com/datasets/inflation/master/data/inflation-consumer.csv",
    "sp500": "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/data.csv",
}

# ------------------------------------------------------------------ EXTRACT
def extract(name: str) -> Path:
    """Download source to cache unless a cached copy exists.

    Caching makes the pipeline reproducible offline and protects against
    upstream changes mid-project; delete data/cache/* to force refresh.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{name}.csv"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    urllib.request.urlretrieve(SOURCES[name], dest)
    return dest


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------- TRANSFORM
def transform_gdp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = ["country", "country_code", "year", "gdp_usd"]
    df = df.dropna(subset=["gdp_usd"])
    df = df[df["gdp_usd"] > 0]
    df["year"] = df["year"].astype(int)
    # World Bank files mix countries with aggregates (e.g. 'World', 'Euro area').
    # Aggregates have non-ISO3166 codes or known aggregate names; keep a
    # explicit blocklist rather than guessing.
    aggregates = {"WLD", "EUU", "OED", "HIC", "LIC", "LMC", "UMC", "MIC",
                  "EAS", "ECS", "LCN", "MEA", "NAC", "SAS", "SSF", "ARB",
                  "CEB", "EAP", "ECA", "EMU", "FCS", "HPC", "IBD", "IBT",
                  "IDA", "IDB", "IDX", "LAC", "LDC", "LMY", "LTE", "MNA",
                  "OSS", "PRE", "PSS", "PST", "SSA", "SST", "TEA", "TEC",
                  "TLA", "TMN", "TSA", "TSS", "EAR", "AFE", "AFW"}
    df["is_aggregate"] = df["country_code"].isin(aggregates)
    return df


def transform_inflation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = ["country", "country_code", "year", "inflation_pct"]
    df = df.dropna(subset=["inflation_pct"])
    df["year"] = df["year"].astype(int)
    # Hyperinflation episodes are real data, not errors (Venezuela, Zimbabwe);
    # they are KEPT, and the report uses medians/log scales accordingly.
    return df


def transform_sp500(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "month", "SP500": "sp500",
                            "Consumer Price Index": "cpi",
                            "Real Price": "real_price", "PE10": "pe10"})
    df = df[["month", "sp500", "cpi", "real_price", "pe10"]]
    for c in ["cpi", "real_price", "pe10"]:
        df[c] = df[c].replace(0.0, np.nan)   # recent-month placeholders
    df["monthly_return"] = df["sp500"].pct_change()
    df["year"] = df["month"].dt.year
    return df.dropna(subset=["sp500"])


def build_dim_country(gdp: pd.DataFrame, infl: pd.DataFrame) -> pd.DataFrame:
    d = (pd.concat([gdp[["country", "country_code", "is_aggregate"]],
                    infl[["country", "country_code"]].assign(is_aggregate=False)])
           .drop_duplicates("country_code")
           .reset_index(drop=True))
    d.insert(0, "country_key", d.index + 1)
    return d


# -------------------------------------------------------------------- LOAD
DDL = """
DROP TABLE IF EXISTS dim_country;
DROP TABLE IF EXISTS fact_gdp_annual;
DROP TABLE IF EXISTS fact_inflation_annual;
DROP TABLE IF EXISTS fact_sp500_monthly;

CREATE TABLE dim_country (
    country_key   INTEGER PRIMARY KEY,
    country       TEXT NOT NULL,
    country_code  TEXT NOT NULL UNIQUE,
    is_aggregate  INTEGER NOT NULL
);
CREATE TABLE fact_gdp_annual (
    country_key INTEGER NOT NULL REFERENCES dim_country(country_key),
    year        INTEGER NOT NULL,
    gdp_usd     REAL NOT NULL CHECK (gdp_usd > 0),
    PRIMARY KEY (country_key, year)
);
CREATE TABLE fact_inflation_annual (
    country_key   INTEGER NOT NULL REFERENCES dim_country(country_key),
    year          INTEGER NOT NULL,
    inflation_pct REAL NOT NULL,
    PRIMARY KEY (country_key, year)
);
CREATE TABLE fact_sp500_monthly (
    month          TEXT PRIMARY KEY,
    sp500          REAL NOT NULL,
    cpi            REAL,
    real_price     REAL,
    pe10           REAL,
    monthly_return REAL,
    year           INTEGER NOT NULL
);
CREATE INDEX idx_gdp_year  ON fact_gdp_annual(year);
CREATE INDEX idx_infl_year ON fact_inflation_annual(year);
"""


def load(dim_country, gdp, infl, sp500) -> dict:
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(DDL)

    key = dim_country.set_index("country_code")["country_key"]
    gdp_f = gdp.assign(country_key=gdp["country_code"].map(key))[
        ["country_key", "year", "gdp_usd"]].drop_duplicates(["country_key", "year"])
    infl_f = infl.assign(country_key=infl["country_code"].map(key))[
        ["country_key", "year", "inflation_pct"]].drop_duplicates(["country_key", "year"])

    dim_country.to_sql("dim_country", con, if_exists="append", index=False)
    gdp_f.to_sql("fact_gdp_annual", con, if_exists="append", index=False)
    infl_f.to_sql("fact_inflation_annual", con, if_exists="append", index=False)
    sp = sp500.copy(); sp["month"] = sp["month"].dt.strftime("%Y-%m-%d")
    sp.to_sql("fact_sp500_monthly", con, if_exists="append", index=False)
    con.commit()

    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ["dim_country", "fact_gdp_annual",
                        "fact_inflation_annual", "fact_sp500_monthly"]}
    con.close()
    return counts


# ---------------------------------------------------------------- VALIDATE
def validate() -> list[tuple[str, int]]:
    """Post-load checks. Returns list of (check, failing_rows); gate fails if any > 0."""
    con = sqlite3.connect(DB)
    checks = {
        "gdp rows orphaned from dim_country":
            "SELECT COUNT(*) FROM fact_gdp_annual f LEFT JOIN dim_country d USING(country_key) WHERE d.country_key IS NULL",
        "inflation rows orphaned from dim_country":
            "SELECT COUNT(*) FROM fact_inflation_annual f LEFT JOIN dim_country d USING(country_key) WHERE d.country_key IS NULL",
        "gdp years outside 1960-2026":
            "SELECT COUNT(*) FROM fact_gdp_annual WHERE year < 1960 OR year > 2026",
        "sp500 months with non-positive price":
            "SELECT COUNT(*) FROM fact_sp500_monthly WHERE sp500 <= 0",
        "sp500 duplicate months":
            "SELECT COUNT(*) - COUNT(DISTINCT month) FROM fact_sp500_monthly",
        "monthly return magnitude > 60% (sanity)":
            "SELECT COUNT(*) FROM fact_sp500_monthly WHERE ABS(monthly_return) > 0.6",
    }
    results = [(name, con.execute(q).fetchone()[0]) for name, q in checks.items()]
    con.close()
    return results
