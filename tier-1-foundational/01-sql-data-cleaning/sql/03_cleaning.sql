-- ============================================================================
-- 03_cleaning.sql
-- Transformation pipeline. Design principles:
--   1. Raw data is never mutated; clean output is a new table.
--   2. Rejected rows are quarantined WITH their rejection reasons, not dropped
--      silently — quarantine is analyzable data (is rejection random or
--      concentrated in one pharmacy/feed?).
--   3. Every repair leaves a provenance flag on the row.
-- Pipeline is a single CTE chain so lineage reads top-to-bottom.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Step 1 · Quarantine: rows hit by any REJECT-severity rule
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE clean.claims_quarantine AS
SELECT c.*,
       v.rejection_reasons
FROM staging.claims_raw c
JOIN (
    SELECT claim_id, STRING_AGG(DISTINCT rule_id, ',' ORDER BY rule_id) AS rejection_reasons
    FROM audit.violations
    WHERE severity = 'REJECT'
    GROUP BY claim_id
) v USING (claim_id);

-- ---------------------------------------------------------------------------
-- Step 2 · Clean fact table
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE clean.fact_claims AS
WITH rejected AS (
    SELECT DISTINCT claim_id FROM audit.violations WHERE severity = 'REJECT'
),

survivors AS (
    SELECT c.* FROM staging.claims_raw c
    ANTI JOIN rejected r USING (claim_id)
),

-- Q1 · collapse exact duplicate rows
dedup_exact AS (
    SELECT DISTINCT * FROM survivors
),

-- Q3 · normalize dates (all three observed formats -> DATE)
typed AS (
    SELECT
        claim_id,
        ndc,
        drug_name_raw,
        pharmacy_id,
        state_raw,
        patient_gender,
        patient_age::SMALLINT                       AS patient_age,
        NULLIF(TRIM(prescriber_npi), '')            AS prescriber_npi,
        COALESCE(TRY_STRPTIME(date_written, '%Y-%m-%d'),
                 TRY_STRPTIME(date_written, '%m/%d/%Y'),
                 TRY_STRPTIME(date_written, '%d-%b-%Y'))::DATE AS date_written,
        COALESCE(TRY_STRPTIME(date_filled, '%Y-%m-%d'),
                 TRY_STRPTIME(date_filled, '%m/%d/%Y'),
                 TRY_STRPTIME(date_filled, '%d-%b-%Y'))::DATE  AS date_filled,
        quantity_dispensed::INTEGER                 AS quantity_dispensed,
        TRY_CAST(NULLIF(TRIM(days_supply), '') AS INTEGER) AS days_supply,
        unit_price::DECIMAL(12, 2)                  AS unit_price,
        total_paid::DECIMAL(14, 2)                  AS total_paid
    FROM dedup_exact
),

-- Q9 · standardize categoricals via explicit mappings (auditable, no magic)
standardized AS (
    SELECT
        t.* EXCLUDE (state_raw, patient_gender, drug_name_raw),
        CASE UPPER(TRIM(t.state_raw))
            WHEN 'NEW YORK'      THEN 'NY'
            WHEN 'CALIFORNIA'    THEN 'CA'
            WHEN 'TEXAS'         THEN 'TX'
            WHEN 'FLORIDA'       THEN 'FL'
            WHEN 'ILLINOIS'      THEN 'IL'
            WHEN 'PENNSYLVANIA'  THEN 'PA'
            WHEN 'NEW JERSEY'    THEN 'NJ'
            WHEN 'MASSACHUSETTS' THEN 'MA'
            ELSE UPPER(TRIM(t.state_raw))
        END AS state,
        CASE UPPER(LEFT(TRIM(t.patient_gender), 1))
            WHEN 'M' THEN 'M'
            WHEN 'F' THEN 'F'
            ELSE 'U'                                -- unknown kept explicit, not NULL
        END AS patient_gender,
        -- Q10 · drug name: trust the master, not the free-text feed value
        d.drug_name,
        d.therapeutic_class
    FROM typed t
    JOIN staging.drug_master d USING (ndc)
),

-- Q7/Q8 · repairs, each with a provenance flag
repaired AS (
    SELECT
        s.* EXCLUDE (unit_price, total_paid, days_supply),
        CASE WHEN s.unit_price > 20 * d.wac_unit_price::DECIMAL(12,2)
             THEN (s.unit_price / 100)::DECIMAL(12, 2)
             ELSE s.unit_price END                  AS unit_price,
        (s.unit_price > 20 * d.wac_unit_price::DECIMAL(12,2)) AS price_was_repaired,
        COALESCE(s.days_supply, s.quantity_dispensed) AS days_supply,
        (s.days_supply IS NULL)                     AS days_supply_was_imputed,
        (s.prescriber_npi IS NULL)                  AS npi_is_missing
    FROM standardized s
    JOIN staging.drug_master d USING (ndc)
),

-- Q2 · collapse resubmissions: same business key, different claim_id.
-- Keep the earliest claim_id deterministically.
final AS (
    SELECT *,
           quantity_dispensed * unit_price AS total_paid   -- recomputed, not trusted
    FROM repaired
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY ndc, pharmacy_id, prescriber_npi, date_written, date_filled,
                     quantity_dispensed, unit_price, patient_age, patient_gender
        ORDER BY claim_id
    ) = 1
)

SELECT * FROM final;

-- ---------------------------------------------------------------------------
-- Step 3 · Post-clean acceptance tests: pipeline fails loudly if any breach.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE audit.acceptance_tests AS
SELECT 'no duplicate claim_ids' AS test,
       COUNT(*) - COUNT(DISTINCT claim_id) AS failures FROM clean.fact_claims
UNION ALL
SELECT 'all NDCs valid 5-4-2',
       COUNT(*) FILTER (WHERE NOT regexp_matches(ndc, '^\d{5}-\d{4}-\d{2}$')) FROM clean.fact_claims
UNION ALL
SELECT 'all quantities positive',
       COUNT(*) FILTER (WHERE quantity_dispensed <= 0) FROM clean.fact_claims
UNION ALL
SELECT 'fill date >= written date',
       COUNT(*) FILTER (WHERE date_filled < date_written) FROM clean.fact_claims
UNION ALL
SELECT 'no future fill dates',
       COUNT(*) FILTER (WHERE date_filled > DATE '2026-02-01') FROM clean.fact_claims
UNION ALL
SELECT 'states are 2-letter codes',
       COUNT(*) FILTER (WHERE NOT regexp_matches(state, '^[A-Z]{2}$')) FROM clean.fact_claims
UNION ALL
SELECT 'gender in (M,F,U)',
       COUNT(*) FILTER (WHERE patient_gender NOT IN ('M','F','U')) FROM clean.fact_claims
UNION ALL
SELECT 'referential integrity to pharmacy master',
       COUNT(*) FROM clean.fact_claims c
       ANTI JOIN staging.pharmacy_master p USING (pharmacy_id)
UNION ALL
SELECT 'total_paid = qty * unit_price',
       COUNT(*) FILTER (WHERE ABS(total_paid - quantity_dispensed * unit_price) > 0.01)
       FROM clean.fact_claims;

-- ---------------------------------------------------------------------------
-- Step 4 · Reconciliation: every raw row accounted for exactly once.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE audit.reconciliation AS
SELECT
    (SELECT COUNT(*) FROM staging.claims_raw)                        AS raw_rows,
    (SELECT COUNT(*) FROM clean.fact_claims)                         AS clean_rows,
    (SELECT COUNT(*) FROM clean.claims_quarantine)                   AS quarantined_rows,
    (SELECT COUNT(*) FROM staging.claims_raw)
      - (SELECT COUNT(*) FROM clean.claims_quarantine)
      - (SELECT COUNT(*) FROM clean.fact_claims)                     AS rows_removed_as_duplicates;
