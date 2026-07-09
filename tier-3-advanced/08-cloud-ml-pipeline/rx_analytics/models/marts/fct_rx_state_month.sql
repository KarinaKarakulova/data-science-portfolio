-- Conformed fact: one row per drug x state x month, enriched with class and
-- launch-relative age (window over the drug's own history, not a hardcode).
with base as (
    select r.*, d.therapeutic_class
    from {{ ref('stg_rx_monthly') }} r
    join {{ ref('stg_drugs') }} d using (drug)
),
launch as (
    select drug, min(month_start) as launch_month from base group by 1
)
select
    b.*,
    datediff('month', l.launch_month, b.month_start) as months_since_launch,
    month(b.month_start)                             as month_of_year
from base b
join launch l using (drug)
