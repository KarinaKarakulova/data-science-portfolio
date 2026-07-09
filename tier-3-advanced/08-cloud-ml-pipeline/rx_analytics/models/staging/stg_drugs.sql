select drug, therapeutic_class
from {{ source('raw', 'drug_dim') }}
