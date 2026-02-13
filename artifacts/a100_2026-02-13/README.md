# A100 Export Package (No Weights)

This bundle contains reproducible artifacts synced from A100 (`117.50.192.217`) under `/opt/dfrope/results`.

## Included
- `data/`: experiment JSON/log/meta outputs
- `scripts/`: experiment runner scripts used on server
- `figures/`: summary plots for advisor report
- `A100_RESULTS_SUMMARY_FOR_ADVISOR_2026-02-13.md`: concise report

## Excluded
- model checkpoints and large binary caches (`*.pt`, `*.bin`, `*.safetensors`, `*.tokens.u16`)

## Key files
- `data/50m_theta_factorial/results.json`
- `data/unified_search_3cfg_3seed/results.json`
- `data/unified_search_2cfg_10seed/results.json`
- `data/100m_scaling/results.json`
- `data/350m_final/results.json`
