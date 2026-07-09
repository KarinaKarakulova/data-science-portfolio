# Analysis Report — HCP Segmentation & Call-Plan Optimization

## Data
2,500 HCPs · 18 months of Rx (45,000 rows) · 22,471 call records · 20 territories.
Synthetic IQVIA-style data (licensing makes real prescriber data unshareable);
latent archetypes planted for validation, hidden from the pipeline.

## KPI layer (SQL)
`sql/kpi_build.sql` builds one row per HCP in DuckDB: L6M market and brand TRx,
our share, 3-month share momentum, per-month TRx slope (regr_slope), call counts,
call intensity — with grain and range checks that gate the run.

## Segmentation

| segment                 |   hcps |   market_trx |   our_share |   calls_now |   calls_opt |
|:------------------------|-------:|-------------:|------------:|------------:|------------:|
| DEFEND (at risk)        |    392 |       131278 |        0.19 |        1946 |        4704 |
| GROW (headroom)         |    599 |       174081 |        0.31 |        2863 |        7111 |
| MAINTAIN (loyal)        |    384 |       145315 |        0.42 |        5318 |        2585 |
| MONITOR (low potential) |   1125 |        69606 |        0.16 |        4273 |           0 |

Thresholds: value cut at the low-potential separation point (45th pct of market
TRx); momentum = fitted share slope beyond ±0.4pp/month. A rule-based 2×2 was chosen over clustering for the primary
scheme because reps must be able to explain *why* an account is in a segment;
K-means cross-check agrees (ARI 0.81) and hidden-archetype recovery is
ARI 0.82.

## Optimization
Greedy marginal-value allocation of 14,400 calls: each next call goes to the
HCP with the highest expected incremental TRx, with diminishing returns
(decay 0.85) and a 12-call per-HCP cap. Greedy is optimal here because
marginal values are independent across HCPs and strictly decreasing — this is a
submodular assignment where greedy achieves the exact optimum, so an ILP would
add complexity without adding calls' worth of value.

Result: priority-account coverage 51% → 100%; expected
incremental field-driven TRx **+64%** at equal capacity
(figures 01–03).

## Limitations
- Response coefficients are assumptions; measuring true call elasticity needs
  either experimentation (preferred; see exec summary pilot design) or
  panel methods with careful confounding control (calls target growers —
  naive regression of TRx on calls is biased upward by construction).
- Pure marginal-value allocation drives MONITOR-segment calls to zero; a
  production plan adds a minimum-coverage floor (relationship maintenance,
  new-writer detection) as a hard constraint — one line in the greedy loop.
- Travel time and rep-HCP relationships are ignored; a routing layer
  (VRP-style) would sit downstream of this prioritization.
- Segment thresholds are business-tuned; the sensitivity of headline numbers
  to VALUE_CUT/MOM_CUT is one parameter edit away.
