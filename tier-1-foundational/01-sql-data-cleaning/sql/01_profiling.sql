-- ============================================================================
-- 01_profiling.sql
-- Profile the raw claims feed BEFORE writing any cleaning logic.
-- Principle: never clean what you haven't measured. Every rule in
-- 03_cleaning.sql should be traceable to a finding produced here.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. Column-level profile: completeness and cardinality
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE audit.column_profile AS
WITH base AS (SELECT COUNT(*) AS n FROM staging.claims_raw)
SELECT
    col.column_name,
    base.n                                            AS total_rows,
    col.null_or_blank                                 AS null_or_blank_rows,
    ROUND(100.0 * col.null_or_blank / base.n, 2)      AS pct_missing,
    col.distinct_values
FROM base,
LATERAL (
    SELECT 'claim_id' AS column_name,
           COUNT(*) FILTER (WHERE claim_id IS NULL OR TRIM(claim_id) = '') AS null_or_blank,
           COUNT(DISTINCT claim_id) AS distinct_values
    FROM staging.claims_raw
    UNION ALL
    SELECT 'ndc',
           COUNT(*) FILTER (WHERE ndc IS NULL OR TRIM(ndc) = ''),
           COUNT(DISTINCT ndc)
    FROM staging.claims_raw
    UNION ALL
    SELECT 'pharmacy_id',
           COUNT(*) FILTER (WHERE pharmacy_id IS NULL OR TRIM(pharmacy_id) = ''),
           COUNT(DISTINCT pharmacy_id)
    FROM staging.claims_raw
    UNION ALL
    SELECT 'state_raw',
           COUNT(*) FILTER (WHERE state_raw IS NULL OR TRIM(state_raw) = ''),
           COUNT(DISTINCT state_raw)
    FROM staging.claims_raw
    UNION ALL
    SELECT 'patient_gender',
           COUNT(*) FILTER (WHERE patient_gender IS NULL OR TRIM(patient_gender) = ''),
           COUNT(DISTINCT patient_gender)
    FROM staging.claims_raw
    UNION ALL
    SELECT 'prescriber_npi',
           COUNT(*) FILTER (WHERE prescriber_npi IS NULL OR TRIM(prescriber_npi) = ''),
           COUNT(DISTINCT prescriber_npi)
    FROM staging.claims_raw
    UNION ALL
    SELECT 'days_supply',
           COUNT(*) FILTER (WHERE days_supply IS NULL OR TRIM(days_supply) = ''),
           COUNT(DISTINCT days_supply)
    FROM staging.claims_raw
    UNION ALL
    SELECT 'date_written',
           COUNT(*) FILTER (WHERE date_written IS NULL OR TRIM(date_written) = ''),
           COUNT(DISTINCT date_written)
    FROM staging.claims_raw
    UNION ALL
    SELECT 'date_filled',
           COUNT(*) FILTER (WHERE date_filled IS NULL OR TRIM(date_filled) = ''),
           COUNT(DISTINCT date_filled)
    FROM staging.claims_raw
) AS col;

-- ---------------------------------------------------------------------------
-- 2. Date format census: how many storage formats live in one column?
--    (Drives the COALESCE(try_strptime...) cascade in the cleaning step.)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE audit.date_format_census AS
SELECT
    CASE
        WHEN regexp_matches(date_filled, '^\d{4}-\d{2}-\d{2}$') THEN 'ISO (YYYY-MM-DD)'
        WHEN regexp_matches(date_filled, '^\d{2}/\d{2}/\d{4}$') THEN 'US (MM/DD/YYYY)'
        WHEN regexp_matches(date_filled, '^\d{2}-[A-Za-z]{3}-\d{4}$') THEN 'Text (DD-Mon-YYYY)'
        ELSE 'Unrecognized'
    END AS date_format,
    COUNT(*) AS rows,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM staging.claims_raw
GROUP BY 1
ORDER BY rows DESC;

-- ---------------------------------------------------------------------------
-- 3. Categorical value census: expose inconsistent encodings
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE audit.state_value_census AS
SELECT state_raw, COUNT(*) AS rows
FROM staging.claims_raw
GROUP BY state_raw
ORDER BY rows DESC;

CREATE OR REPLACE TABLE audit.gender_value_census AS
SELECT COALESCE(NULLIF(TRIM(patient_gender), ''), '<blank>') AS gender_raw,
       COUNT(*) AS rows
FROM staging.claims_raw
GROUP BY 1
ORDER BY rows DESC;

-- ---------------------------------------------------------------------------
-- 4. Numeric distribution profile for price outlier detection.
--    Median/IQR chosen over mean/stddev: claim prices are heavy-tailed and a
--    handful of fat-finger values would drag a z-score threshold badly.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE audit.price_profile AS
SELECT
    d.therapeutic_class,
    COUNT(*)                                             AS claims,
    ROUND(MIN(c.unit_price::DOUBLE), 2)                  AS min_price,
    ROUND(quantile_cont(c.unit_price::DOUBLE, 0.25), 2)  AS p25,
    ROUND(quantile_cont(c.unit_price::DOUBLE, 0.50), 2)  AS median,
    ROUND(quantile_cont(c.unit_price::DOUBLE, 0.75), 2)  AS p75,
    ROUND(MAX(c.unit_price::DOUBLE), 2)                  AS max_price
FROM staging.claims_raw c
JOIN staging.drug_master d USING (ndc)
GROUP BY 1
ORDER BY claims DESC;

-- ---------------------------------------------------------------------------
-- 5. Duplicate profile
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE audit.duplicate_profile AS
WITH exact_dupes AS (
    SELECT COUNT(*) - COUNT(DISTINCT claim_id) AS surplus_rows_same_claim_id
    FROM staging.claims_raw
),
resubmissions AS (
    -- same business content, different claim_id
    SELECT SUM(cnt - 1) AS surplus_rows_new_claim_id
    FROM (
        SELECT COUNT(*) AS cnt
        FROM (SELECT DISTINCT * FROM staging.claims_raw)   -- collapse exact dupes first
        GROUP BY ndc, pharmacy_id, prescriber_npi, date_written, date_filled,
                 quantity_dispensed, unit_price, total_paid,
                 patient_age, patient_gender
        HAVING COUNT(*) > 1
    )
)
SELECT * FROM exact_dupes, resubmissions;
