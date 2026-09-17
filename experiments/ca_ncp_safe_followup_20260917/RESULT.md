# CA-NCP safety follow-up result

Status: complete on the frozen OLMo-2-0425-1B-Instruct Native-4K
Full-13x10 development panel. This is a same-panel mechanism diagnostic, not an
independent confirmation and not a new deployable method.

## Fixed-panel result

| Arm | Full-13 score | Relevant contrast |
|---|---:|---:|
| Native `N0` | 73.53% | reference |
| NCP `C0` | 76.78% | best frozen reference |
| carrier-NCP `P0` | 76.01% | `P0-C0 -0.77pp` |
| `N_operator_cap` | 73.76% | `+0.23pp` vs N0; `-3.03pp` vs C0 |
| `P_operator_cap` | 75.09% | `-0.92pp` vs P0; `-1.69pp` vs C0 |
| `P_axis_consensus` | 73.59% | `-2.42pp` vs P0; `-3.19pp` vs C0 |

The capped operator recovers much of the full-strength CA-NCP failure, but none
of the three task-blind safety constructions exceeds the original NCP table.
`N_operator_cap` is effectively Native-level; the two carrier-NCP variants are
worse than their own `P0` control. Under the preregistered contract, this closes
the activation-driven coordinate-reassignment route rather than promoting a new
candidate.

The paired bootstrap is retained as stability analysis, not as the authority for
the fixed-panel scores. Its 95% intervals are `[-1.77,+1.89]pp` for
`N_operator_cap-N0`, `[-3.42,+1.54]pp` for `P_operator_cap-P0`, and
`[-5.35,+0.46]pp` for `P_axis_consensus-P0`.

## Evidence

- [Formal paired report](reports/server_20260917/paired_report.json)
- [N operator-cap raw rows](reports/server_20260917/raw/N_operator_cap.jsonl)
- [P operator-cap raw rows](reports/server_20260917/raw/P_operator_cap.jsonl)
- [P axis-consensus raw rows](reports/server_20260917/raw/P_axis_consensus.jsonl)

The report records the frozen method receipt, all baseline and candidate hashes,
per-task scores, output health, and the exact aggregation contract.
