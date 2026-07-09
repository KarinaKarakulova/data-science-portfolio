-- Composite-grain uniqueness: fails if any (drug, state, month) duplicates.
select drug, state, month_start, count(*) as n
from {{ ref('fct_rx_state_month') }}
group by 1,2,3
having count(*) > 1
