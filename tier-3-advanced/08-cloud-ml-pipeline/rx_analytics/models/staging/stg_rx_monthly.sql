-- Typing + light cleaning only; business logic lives in marts/features.
select
    cast(month || '-01' as date)          as month_start,
    drug,
    state,
    cast(detailing_calls_100s as double)  as detailing_calls_100s,
    cast(trx as integer)                  as trx
from {{ source('raw', 'rx_monthly_raw') }}
