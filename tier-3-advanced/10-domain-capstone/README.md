# 10 · Domain Capstone — HCP Segmentation & Call-Plan Optimization (Pharma Commercial Analytics)

The end-to-end project: SQL KPI engineering → business segmentation (validated two independent ways) → constrained optimization of field-force effort → executive summary with a pilot design. Built on IQVIA-style prescriber data (synthetic — real Xponent-class data is strictly licensed — with hidden behavioral archetypes planted for validation).

## The business problem
20 reps, ~23k calls per 6 months, allocated by habit: reps visit where relationships already exist. Share **momentum** is ignored — an eroding top prescriber gets less attention than a comfortable loyal one.

## What the analysis found and did
- **SQL KPI layer** (`sql/kpi_build.sql`, DuckDB): one row per HCP — L6M volume, brand share, momentum as a *fitted monthly share slope* (chosen over a period-difference after showing the difference estimator is noisier than the signal at typical script counts), call intensity; grain/range checks gate the run
- **Segmentation** on value × momentum → DEFEND / GROW / MAINTAIN / MONITOR. Rule-based 2×2 chosen deliberately over clustering (reps must be able to explain why an account is where it is), then validated twice: K-means agreement **ARI 0.81**, hidden-archetype recovery **ARI 0.82**
- **The quantified gap:** DEFEND + GROW hold the majority of addressable volume but receive a minority of calls; only **51.5%** of them get ≥4 calls per 6 months
- **Optimization:** greedy marginal-value allocation under capacity with diminishing returns and a per-HCP cap — provably optimal here (submodular, independent accounts), so an ILP would add nothing; result: priority coverage **51.5% → 100%**, expected incremental TRx **+64%** at equal headcount, *explicitly labeled a prioritization estimate under stated response assumptions, not a causal forecast*
- **The recommendation includes its own validation plan:** a territory-randomized pilot (10 vs 10 territories, 2 quarters) — Project 06's experimentation framework applied to this decision

## Deliverables
[`reports/executive_summary.md`](reports/executive_summary.md) (one page, business language) · [`reports/analysis_report.md`](reports/analysis_report.md) (methods + limitations) · figures (segment matrix, call reallocation, geographic footprint)

## Run
```bash
python src/generate_data.py && python src/capstone.py
```
