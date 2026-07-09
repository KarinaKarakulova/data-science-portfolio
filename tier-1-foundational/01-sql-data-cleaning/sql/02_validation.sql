-- ============================================================================
-- 02_validation.sql
-- Declarative constraint validation. Each rule writes row-level evidence into
-- audit.violations (one row per rule hit), so every rejected or repaired
-- record can be traced back to the exact rule that flagged it.
--
-- Rule severities:
--   REJECT : row cannot be trusted for analysis -> quarantined
--   REPAIR : deterministic fix exists -> corrected in 03_cleaning.sql
--   FLAG   : suspicious but usable -> kept, marked for review
-- ============================================================================

CREATE OR REPLACE TABLE audit.violations (
    claim_id   VARCHAR,
    rule_id    VARCHAR,
    rule_desc  VARCHAR,
    severity   VARCHAR,
    bad_value  VARCHAR
);

-- Parsed view used by several rules (single source of date-parsing truth)
CREATE OR REPLACE VIEW staging.claims_parsed AS
SELECT
    *,
    COALESCE(
        TRY_STRPTIME(date_written, '%Y-%m-%d'),
        TRY_STRPTIME(date_written, '%m/%d/%Y'),
        TRY_STRPTIME(date_written, '%d-%b-%Y')
    )::DATE AS date_written_parsed,
    COALESCE(
        TRY_STRPTIME(date_filled, '%Y-%m-%d'),
        TRY_STRPTIME(date_filled, '%m/%d/%Y'),
        TRY_STRPTIME(date_filled, '%d-%b-%Y')
    )::DATE AS date_filled_parsed
FROM staging.claims_raw;

-- R01 · NDC must match the 5-4-2 NDC-11 pattern ------------------------ REJECT
INSERT INTO audit.violations
SELECT claim_id, 'R01', 'NDC missing or not in 5-4-2 format', 'REJECT', ndc
FROM staging.claims_raw
WHERE ndc IS NULL OR TRIM(ndc) = ''
   OR NOT regexp_matches(ndc, '^\d{5}-\d{4}-\d{2}$');

-- R02 · NDC must exist in drug master ---------------------------------- REJECT
INSERT INTO audit.violations
SELECT c.claim_id, 'R02', 'NDC not found in drug master', 'REJECT', c.ndc
FROM staging.claims_raw c
LEFT JOIN staging.drug_master d USING (ndc)
WHERE d.ndc IS NULL
  AND c.ndc IS NOT NULL AND TRIM(c.ndc) <> ''
  AND regexp_matches(c.ndc, '^\d{5}-\d{4}-\d{2}$');

-- R03 · pharmacy_id must exist in pharmacy master ----------------------- REJECT
INSERT INTO audit.violations
SELECT c.claim_id, 'R03', 'pharmacy_id has no match in pharmacy master', 'REJECT', c.pharmacy_id
FROM staging.claims_raw c
LEFT JOIN staging.pharmacy_master p USING (pharmacy_id)
WHERE p.pharmacy_id IS NULL;

-- R04 · quantity_dispensed must be a positive number -------------------- REJECT
INSERT INTO audit.violations
SELECT claim_id, 'R04', 'quantity_dispensed <= 0 or non-numeric', 'REJECT', quantity_dispensed
FROM staging.claims_raw
WHERE TRY_CAST(quantity_dispensed AS INTEGER) IS NULL
   OR TRY_CAST(quantity_dispensed AS INTEGER) <= 0;

-- R05 · both dates must parse ------------------------------------------- REJECT
INSERT INTO audit.violations
SELECT claim_id, 'R05', 'unparseable date_written or date_filled', 'REJECT',
       date_written || ' | ' || date_filled
FROM staging.claims_parsed
WHERE date_written_parsed IS NULL OR date_filled_parsed IS NULL;

-- R06 · fill date cannot precede written date --------------------------- REJECT
INSERT INTO audit.violations
SELECT claim_id, 'R06', 'date_filled earlier than date_written', 'REJECT',
       date_written_parsed::VARCHAR || ' -> ' || date_filled_parsed::VARCHAR
FROM staging.claims_parsed
WHERE date_filled_parsed < date_written_parsed;

-- R07 · fill date cannot be in the future ------------------------------- REJECT
INSERT INTO audit.violations
SELECT claim_id, 'R07', 'date_filled in the future', 'REJECT', date_filled_parsed::VARCHAR
FROM staging.claims_parsed
WHERE date_filled_parsed > DATE '2026-02-01';   -- feed snapshot date (2025 scripts fill through mid-Jan)

-- R08 · unit price outlier vs drug reference price ---------------------- REPAIR
-- A unit price more than 20x the drug's wholesale acquisition cost is a
-- decimal-shift / fat-finger signature; repaired by /100 in cleaning.
INSERT INTO audit.violations
SELECT c.claim_id, 'R08', 'unit_price > 20x WAC reference (decimal shift)', 'REPAIR',
       c.unit_price
FROM staging.claims_raw c
JOIN staging.drug_master d USING (ndc)
WHERE TRY_CAST(c.unit_price AS DOUBLE) > 20 * TRY_CAST(d.wac_unit_price AS DOUBLE);

-- R09 · missing days_supply --------------------------------------------- REPAIR
-- Imputable from quantity under the dominant 1-unit/day regimen observed in
-- profiling; imputation is marked with a provenance flag, never silent.
INSERT INTO audit.violations
SELECT claim_id, 'R09', 'days_supply missing (imputed from quantity)', 'REPAIR', days_supply
FROM staging.claims_raw
WHERE days_supply IS NULL OR TRIM(days_supply) = '';

-- R10 · missing prescriber NPI ------------------------------------------ FLAG
INSERT INTO audit.violations
SELECT claim_id, 'R10', 'prescriber_npi missing', 'FLAG', prescriber_npi
FROM staging.claims_raw
WHERE prescriber_npi IS NULL OR TRIM(prescriber_npi) = '';

-- R11 · exact duplicate rows -------------------------------------------- REPAIR
INSERT INTO audit.violations
SELECT claim_id, 'R11', 'exact duplicate row (kept first occurrence)', 'REPAIR',
       CAST(cnt AS VARCHAR) || ' copies'
FROM (
    SELECT claim_id, COUNT(*) AS cnt
    FROM staging.claims_raw
    GROUP BY ALL
    HAVING COUNT(*) > 1
);

-- Rule-level summary -----------------------------------------------------
CREATE OR REPLACE TABLE audit.violation_summary AS
SELECT rule_id,
       ANY_VALUE(rule_desc) AS rule_desc,
       ANY_VALUE(severity)  AS severity,
       COUNT(*)             AS violations,
       COUNT(DISTINCT claim_id) AS distinct_claims
FROM audit.violations
GROUP BY rule_id
ORDER BY rule_id;
