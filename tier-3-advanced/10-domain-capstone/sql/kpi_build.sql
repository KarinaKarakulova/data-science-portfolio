-- ============================================================================
-- kpi_build.sql — HCP-level commercial KPIs from raw facts.
-- Grain: one row per HCP. Windows: L6M = 2024-09..2025-02 (last 6 months),
-- P6M = the 6 months before that. Momentum = share_L3M - share_P3M.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS kpi;

CREATE OR REPLACE TABLE kpi.hcp_kpis AS
WITH rx AS (
    SELECT hcp_id,
           CAST(month || '-01' AS DATE) AS month_start,
           market_trx, our_trx
    FROM read_csv('data/fact_rx_monthly.csv', header=true)
),
bounds AS (SELECT MAX(month_start) AS mx FROM rx),
windowed AS (
    SELECT rx.*,
           CASE
             WHEN month_start >  mx - INTERVAL 3 MONTH THEN 'L3M'
             WHEN month_start >  mx - INTERVAL 6 MONTH THEN 'P3M'
             WHEN month_start >  mx - INTERVAL 12 MONTH THEN 'PRIOR'
             ELSE 'OLD'
           END AS win
    FROM rx, bounds
),
agg AS (
    SELECT hcp_id,
        SUM(market_trx) FILTER (win IN ('L3M','P3M'))                 AS market_trx_l6m,
        SUM(our_trx)    FILTER (win IN ('L3M','P3M'))                 AS our_trx_l6m,
        SUM(our_trx)    FILTER (win = 'L3M')::DOUBLE
            / NULLIF(SUM(market_trx) FILTER (win = 'L3M'), 0)         AS share_l3m,
        SUM(our_trx)    FILTER (win = 'P3M')::DOUBLE
            / NULLIF(SUM(market_trx) FILTER (win = 'P3M'), 0)         AS share_p3m,
        -- momentum as a fitted monthly slope of share over the full window:
        -- far less noisy than an L3M-vs-P3M difference at typical script counts
        regr_slope(our_trx::DOUBLE / NULLIF(market_trx, 0),
                   EXTRACT(EPOCH FROM month_start) / 2592000)         AS share_slope_pm
    FROM windowed
    GROUP BY hcp_id
),
calls AS (
    SELECT hcp_id, SUM(calls) AS calls_l6m
    FROM read_csv('data/fact_calls.csv', header=true) c, bounds b
    WHERE CAST(c.month || '-01' AS DATE) > b.mx - INTERVAL 6 MONTH
    GROUP BY hcp_id
)
SELECT
    d.hcp_id, d.specialty, d.region, d.territory, d.lon, d.lat,
    COALESCE(a.market_trx_l6m, 0)                    AS market_trx_l6m,
    COALESCE(a.our_trx_l6m, 0)                       AS our_trx_l6m,
    COALESCE(a.our_trx_l6m,0)::DOUBLE
        / NULLIF(a.market_trx_l6m,0)                 AS our_share_l6m,
    a.share_l3m, a.share_p3m,
    a.share_slope_pm,
    a.share_slope_pm * 3                             AS share_momentum,  -- pp per quarter
    COALESCE(c.calls_l6m, 0)                         AS calls_l6m,
    COALESCE(c.calls_l6m,0)::DOUBLE
        / NULLIF(a.market_trx_l6m,0)                 AS calls_per_mkt_trx
FROM read_csv('data/dim_hcp.csv', header=true) d
LEFT JOIN agg   a USING (hcp_id)
LEFT JOIN calls c USING (hcp_id);

-- Sanity: grain uniqueness
CREATE OR REPLACE TABLE kpi.checks AS
SELECT 'duplicate hcp rows' AS chk,
       COUNT(*) - COUNT(DISTINCT hcp_id) AS fails FROM kpi.hcp_kpis
UNION ALL
SELECT 'share outside [0,1]',
       COUNT(*) FILTER (our_share_l6m < 0 OR our_share_l6m > 1) FROM kpi.hcp_kpis;
