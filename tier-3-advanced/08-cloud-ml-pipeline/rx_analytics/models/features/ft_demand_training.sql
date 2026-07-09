-- Feature table for next-month demand forecasting.
-- Grain: drug x state x month. Target: trx one month ahead (lead), so every
-- feature is strictly past-or-present relative to the target — leakage-safe
-- by construction, enforced further by the time-based split in training.
with f as (
    select
        *,
        lag(trx, 1) over w  as trx_lag1,
        lag(trx, 2) over w  as trx_lag2,
        lag(trx, 3) over w  as trx_lag3,
        avg(trx) over (w rows between 2 preceding and current row) as trx_ma3,
        lag(detailing_calls_100s, 1) over w as detailing_lag1,
        lag(detailing_calls_100s, 2) over w as detailing_lag2,
        lead(trx, 1) over w as target_trx_next_month
    from {{ ref('fct_rx_state_month') }}
    window w as (partition by drug, state order by month_start)
)
select * from f
where trx_lag3 is not null            -- need full history
  and target_trx_next_month is not null
