# Data Quality Report — Pharmacy Claims Feed

**Dataset:** `pharmacy_claims_raw.csv` (50,500 rows) + pharmacy and drug dimensions
**Pipeline:** DuckDB SQL, orchestrated by `src/run_pipeline.py`
**Outcome:** 48,238 analysis-ready claims (95.5%), 1,774 quarantined (3.5%), 488 duplicate rows removed (1.0%)

---

## 1. Executive summary

The raw claims feed is unusable for analysis as delivered. Profiling identified eleven distinct quality problems spanning four categories: structural (duplicates, resubmissions), format (three date encodings in one column, six spellings of gender), integrity (invalid NDC codes, orphan pharmacy references), and value-level defects (negative quantities, decimal-shifted prices, missing fields).

Rather than dropping bad rows silently, the pipeline classifies every defect by severity — **REJECT** (quarantined with reason codes), **REPAIR** (deterministically fixed with a provenance flag), or **FLAG** (retained, marked for review) — and enforces nine post-clean acceptance tests as a hard quality gate. Every raw row reconciles to exactly one of clean / quarantine / duplicate-removed.

**Business impact of not cleaning:** the 187 decimal-shifted prices alone inflate apparent total spend materially — a single 100x fat-finger on a specialty drug can exceed the true value of hundreds of legitimate claims. Territory-level analysis would also be distorted: 24 raw state encodings collapse to 8 actual states.

## 2. What profiling found

| # | Issue | Evidence (measured, not assumed) | Severity |
|---|-------|----------------------------------|----------|
| 1 | Exact duplicate rows | 300 surplus rows share a `claim_id` with an identical twin | REPAIR |
| 2 | Resubmitted claims under new IDs | 200 rows identical on business key but with fresh `claim_id` | REPAIR |
| 3 | Mixed date formats | `date_filled`: 70.0% ISO, 21.9% US, 8.1% `DD-Mon-YYYY` | REPAIR |
| 4 | Impossible date logic | 302 fills precede the written date; 160 fills are in the future | REJECT |
| 5 | Invalid NDC codes | 663 rows fail the 5-4-2 NDC-11 pattern (short, alphanumeric, undashed, or blank) | REJECT |
| 6 | Non-positive quantities | 407 rows with quantity ≤ 0 | REJECT |
| 7 | Decimal-shifted prices | 187 rows priced > 20× the drug's WAC reference | REPAIR (÷100) |
| 8 | Missing values | `days_supply` 2.0% blank; `prescriber_npi` 1.5% blank; `patient_gender` 12.7% blank | REPAIR / FLAG |
| 9 | Inconsistent categoricals | 24 state encodings for 8 states; 8 gender encodings | REPAIR |
| 10 | Free-text drug name noise | ~3% of names carry casing/whitespace variants | REPAIR (mastered) |
| 11 | Orphan pharmacy references | 258 claims point to pharmacy IDs absent from the dimension | REJECT |

Full column-level profile, value censuses, and price distributions are in `audit_tables.md` (auto-regenerated on every run).

## 3. Key design decisions

**Quarantine, don't delete.** Rejected rows are written to `clean.claims_quarantine` with comma-separated rule codes. This matters because rejection patterns are themselves signal: if one pharmacy or one feed batch dominates the quarantine, that's an upstream conversation, not a cleaning problem. Fourteen rows failed multiple rules simultaneously.

**Repair only when deterministic, and always with provenance.** Prices are repaired only when > 20× the drug's reference WAC — an unambiguous decimal-shift signature — and the row carries `price_was_repaired = TRUE`. Missing `days_supply` is imputed from quantity (profiling showed a 1-unit/day regimen dominates) and flagged `days_supply_was_imputed = TRUE`. Downstream analysts can exclude repaired rows in one WHERE clause if their use case demands untouched data.

**Robust outlier statistics.** The price threshold anchors to each drug's reference price rather than a global z-score. Claim prices are heavy-tailed across therapeutic classes (Anti-Infective median $105 vs. Respiratory median $434), so a global threshold would either miss oncology fat-fingers or reject legitimate specialty claims.

**Recompute, don't trust, derived fields.** `total_paid` is recalculated as `quantity × unit_price` in the clean table; the feed's own value is discarded. An acceptance test enforces the identity to the cent.

**Missing gender is kept as explicit `'U'`, not NULL.** 12.7% missingness is too high to drop and too structured to ignore; an explicit category keeps those claims in segment analyses instead of silently vanishing from GROUP BYs.

**Type at transform time, not load time.** All raw columns land as VARCHAR. Casting during load would convert defects into NULLs before they could be measured.

## 4. Reconciliation & quality gate

| raw rows | → clean | → quarantined | → removed as duplicates |
|---:|---:|---:|---:|
| 50,500 | 48,238 | 1,774 | 488 |

All nine acceptance tests pass on the clean table: unique claim IDs, valid NDC format, positive quantities, coherent date ordering, no future fills, 2-letter states, gender ∈ {M, F, U}, full referential integrity to the pharmacy dimension, and exact `total_paid` arithmetic. The pipeline exits non-zero if any test fails, so a regression can never ship a "clean" table quietly.

## 5. Limitations & next steps

- The 20× WAC price threshold was tuned against this feed's profile; a production version should be validated against known-good claims and monitored for drift.
- `days_supply` imputation assumes a 1-unit/day regimen; drugs with weekly dosing would need regimen-aware logic from a drug master enrichment.
- Duplicate detection uses a full business-key match; fuzzy near-duplicates (e.g., same claim with a 1-cent price difference) are out of scope here and would suit a probabilistic matching pass.
- A production deployment would move these rules into dbt tests / great_expectations with scheduled runs and alerting — that pattern is demonstrated in Project 08 (cloud ML pipeline).
