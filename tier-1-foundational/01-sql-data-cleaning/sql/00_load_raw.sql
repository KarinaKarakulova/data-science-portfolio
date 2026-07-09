-- ============================================================================
-- 00_load_raw.sql
-- Load raw CSV extracts into a staging schema, preserving them exactly as
-- received. All columns land as VARCHAR on purpose: typing raw data at load
-- time silently destroys evidence of quality problems (e.g. '30 days' in a
-- numeric column becomes NULL and the defect disappears from view).
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS clean;

CREATE OR REPLACE TABLE staging.claims_raw AS
SELECT * FROM read_csv('data/raw/pharmacy_claims_raw.csv',
                       header = true, all_varchar = true);

CREATE OR REPLACE TABLE staging.pharmacy_master AS
SELECT * FROM read_csv('data/raw/pharmacy_master.csv',
                       header = true, all_varchar = true);

CREATE OR REPLACE TABLE staging.drug_master AS
SELECT * FROM read_csv('data/raw/drug_master.csv',
                       header = true, all_varchar = true);
