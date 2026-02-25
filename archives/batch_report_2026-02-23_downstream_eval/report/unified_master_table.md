# Unified Master Table

Ranking rule: `longbench_avg` desc, then `passkey_margin_16k`, then `niah_mean`.

| rank | method | longbench_avg | ΔLongBench vs baseline | niah_mean | passkey_tf@16k | passkey_margin@16k | Δmargin vs baseline | source_profile | coverage |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | anchored_sigmoid | 0.0717 | +14.50% | 1.0000 | 1.0000 | 6.5242 | +39.27% | downstream_eval_autorun | 3 |
| 2 | sigmoid | 0.0687 | +9.62% | 1.0000 | 1.0000 | 5.4137 | +15.56% | downstream_eval_autorun | 3 |
| 3 | pi | 0.0665 | +6.19% | 1.0000 | 1.0000 | 6.5850 | +40.57% | downstream_eval_autorun | 3 |
| 4 | yarn | 0.0656 | +4.78% | 1.0000 | 1.0000 | 5.2433 | +11.93% | downstream_eval_parallel_seed42_m2 | 3 |
| 5 | baseline | 0.0626 | +0.00% | 0.9545 | 1.0000 | 4.6846 | +0.00% | downstream_eval_autorun | 3 |
