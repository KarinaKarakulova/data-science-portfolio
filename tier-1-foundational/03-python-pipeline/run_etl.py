"""
run_etl.py — orchestrates the ETL DAG, renders the DAG figure, builds the report.

DAG:
    extract_gdp ─┐
    extract_infl ├─> transforms ─> build_dim ─> load ─> validate ─> report
    extract_sp  ─┘

A ~40-line topological runner is used deliberately instead of Airflow: the
point is to show the *concepts* (explicit dependencies, task status, timing,
failure propagation) without infrastructure overhead. The same graph maps
1:1 onto Airflow/Dagster tasks or dbt models.
"""

import sqlite3
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from etl import core

ROOT = core.ROOT
FIG = ROOT / "figures"
REP = ROOT / "reports"


# ------------------------------------------------------------ tiny DAG runner
class Task:
    def __init__(self, name, fn, deps=()):
        self.name, self.fn, self.deps = name, fn, list(deps)
        self.status, self.seconds, self.result = "pending", None, None


def run_dag(tasks: dict[str, Task]):
    done, order = set(), []
    while len(done) < len(tasks):
        ready = [t for t in tasks.values()
                 if t.name not in done and all(d in done for d in t.deps)]
        if not ready:
            raise RuntimeError("cyclic or unsatisfiable DAG")
        for t in ready:
            t0 = time.perf_counter()
            try:
                t.result = t.fn(tasks)
                t.status = "success"
            except Exception as e:                      # fail fast, mark chain
                t.status = f"failed: {e}"
                _mark_downstream_skipped(tasks, t.name)
                _print_status(tasks)
                raise
            t.seconds = time.perf_counter() - t0
            done.add(t.name); order.append(t.name)
    _print_status(tasks)
    return order


def _mark_downstream_skipped(tasks, failed):
    for t in tasks.values():
        if failed in t.deps and t.status == "pending":
            t.status = "skipped"
            _mark_downstream_skipped(tasks, t.name)


def _print_status(tasks):
    print(f"\n{'task':<22}{'status':<12}{'seconds':>8}")
    for t in tasks.values():
        s = f"{t.seconds:.2f}" if t.seconds else "-"
        print(f"{t.name:<22}{t.status:<12}{s:>8}")


# ------------------------------------------------------------------ reporting
def build_report(tasks):
    FIG.mkdir(exist_ok=True); REP.mkdir(exist_ok=True)
    con = sqlite3.connect(core.DB)

    # Fig 1: G7 GDP trajectories
    g7 = con.execute("""
        SELECT d.country, f.year, f.gdp_usd/1e12 AS gdp_tn
        FROM fact_gdp_annual f JOIN dim_country d USING(country_key)
        WHERE d.country_code IN ('USA','JPN','DEU','GBR','FRA','ITA','CAN')
        ORDER BY d.country, f.year""").fetchall()
    df = pd.DataFrame(g7, columns=["country", "year", "gdp_tn"])
    plt.figure(figsize=(9, 4.5))
    for c, g in df.groupby("country"):
        plt.plot(g["year"], g["gdp_tn"], lw=1.2, label=c)
    plt.legend(ncol=2, fontsize=8); plt.ylabel("GDP, $ trillion (nominal)")
    plt.title("G7 nominal GDP, World Bank"); plt.savefig(FIG / "01_g7_gdp.png",
                                                         dpi=120, bbox_inches="tight")
    plt.close()

    # Fig 2: US inflation vs S&P real return by year
    joined = pd.read_sql("""
        WITH us_infl AS (
            SELECT year, inflation_pct FROM fact_inflation_annual f
            JOIN dim_country d USING(country_key) WHERE d.country_code='USA'),
        annual_ret AS (
            SELECT year, EXP(SUM(LN(1+monthly_return)))-1 AS nominal_ret
            FROM fact_sp500_monthly WHERE monthly_return IS NOT NULL GROUP BY year)
        SELECT u.year, u.inflation_pct, a.nominal_ret*100 AS nominal_ret_pct
        FROM us_infl u JOIN annual_ret a USING(year)""", con)
    plt.figure(figsize=(6.5, 5))
    plt.scatter(joined["inflation_pct"], joined["nominal_ret_pct"], s=18, alpha=0.7)
    plt.axvline(0, lw=0.5, color="gray"); plt.axhline(0, lw=0.5, color="gray")
    plt.xlabel("US CPI inflation % (World Bank)"); plt.ylabel("S&P 500 annual return %")
    plt.title("Annual equity returns vs inflation, 1960–2024")
    plt.savefig(FIG / "02_inflation_vs_returns.png", dpi=120, bbox_inches="tight")
    corr = joined["inflation_pct"].corr(joined["nominal_ret_pct"])
    plt.close()

    counts = tasks["load"].result
    checks = tasks["validate"].result
    src_hashes = {n: core.file_sha256(core.CACHE / f"{n}.csv") for n in core.SOURCES}

    lines = [
        "# ETL Run Report", "",
        "## Warehouse row counts", "",
        *[f"- `{t}`: **{n:,}** rows" for t, n in counts.items()], "",
        "## Validation gate", "",
        "| check | failing rows |", "|---|---:|",
        *[f"| {name} | {n} |" for name, n in checks], "",
        "## Source lineage (SHA-256, first 12 hex)", "",
        *[f"- `{n}`: `{h}`" for n, h in src_hashes.items()], "",
        "## Cross-source analytical check", "",
        f"Correlation between US annual inflation and same-year S&P 500 nominal "
        f"return (1960–2024, N={len(joined)}): **{corr:.2f}** — consistent with the "
        "regime finding in Project 02 (high inflation associates with weaker "
        "nominal equity returns, though the same-year relationship is noisy).",
        "", "Figures: `figures/01_g7_gdp.png`, `figures/02_inflation_vs_returns.png`",
    ]
    (REP / "etl_run_report.md").write_text("\n".join(lines))
    con.close()
    if any(n > 0 for _, n in checks):
        raise RuntimeError("validation gate failed")
    return "report written"


def draw_dag(tasks):
    pos = {"extract_gdp": (0, 2), "extract_inflation": (0, 1), "extract_sp500": (0, 0),
           "transform": (1, 1), "build_dim": (2, 1), "load": (3, 1),
           "validate": (4, 1), "report": (5, 1)}
    plt.figure(figsize=(10, 3))
    for t in tasks.values():
        x, y = pos[t.name]
        plt.scatter(x, y, s=1600, c="#2ca02c" if t.status == "success" else "#d62728",
                    zorder=3, edgecolors="k")
        plt.text(x, y, t.name.replace("_", "\n"), ha="center", va="center",
                 fontsize=6.5, zorder=4, color="white", weight="bold")
        for d in t.deps:
            dx, dy = pos[d]
            plt.annotate("", xy=(x - 0.13, y), xytext=(dx + 0.13, dy),
                         arrowprops=dict(arrowstyle="->", lw=1))
    plt.axis("off"); plt.title("ETL task DAG (green = success)")
    plt.savefig(FIG / "00_dag.png", dpi=130, bbox_inches="tight"); plt.close()


# ------------------------------------------------------------------ pipeline
def main():
    T = {}
    T["extract_gdp"] = Task("extract_gdp", lambda t: core.extract("gdp"))
    T["extract_inflation"] = Task("extract_inflation", lambda t: core.extract("inflation"))
    T["extract_sp500"] = Task("extract_sp500", lambda t: core.extract("sp500"))
    T["transform"] = Task("transform", lambda t: {
        "gdp": core.transform_gdp(t["extract_gdp"].result),
        "infl": core.transform_inflation(t["extract_inflation"].result),
        "sp500": core.transform_sp500(t["extract_sp500"].result)},
        deps=["extract_gdp", "extract_inflation", "extract_sp500"])
    T["build_dim"] = Task("build_dim", lambda t: core.build_dim_country(
        t["transform"].result["gdp"], t["transform"].result["infl"]), deps=["transform"])
    T["load"] = Task("load", lambda t: core.load(
        t["build_dim"].result, t["transform"].result["gdp"],
        t["transform"].result["infl"], t["transform"].result["sp500"]),
        deps=["build_dim"])
    T["validate"] = Task("validate", lambda t: core.validate(), deps=["load"])
    T["report"] = Task("report", build_report, deps=["validate"])

    # Math 257 check (Wk 6: graphs & adjacency matrices): (A^k)_ij counts walks
    # of length k, and a DAG has no walk longer than n-1 nodes — so A^n must be
    # the zero matrix (nilpotent). A cycle anywhere would make this fail.
    names = list(T)
    Adj = np.zeros((len(T), len(T)), dtype=int)
    for t in T.values():
        for d in t.deps:
            Adj[names.index(t.name), names.index(d)] = 1
    assert not np.linalg.matrix_power(Adj, len(T)).any(), "cycle in task DAG!"
    print(f"[Math 257] task-graph adjacency matrix is nilpotent: A^{len(T)} = 0")

    run_dag(T)
    draw_dag(T)


if __name__ == "__main__":
    main()
