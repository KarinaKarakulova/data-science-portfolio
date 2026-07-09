select * from {{ ref('fct_rx_state_month') }} where trx < 0
