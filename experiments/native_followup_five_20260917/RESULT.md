# Native attention follow-up result

Status: complete on frozen OLMo-2-0425-1B-Instruct inside the Native 4K
window. These are development diagnostics on two fixed Full-13x10 blocks plus
the previously frozen 99-row Natural-QA census and matched PPL46 documents.

## Main result

| Method | Full-13 block 1 | Full-13 block 2 | Two-block mean |
|---|---:|---:|---:|
| NCP | 76.78% | 69.92% | 73.35% |
| B: Native mass projection | 76.47% | 70.54% | 73.51% |
| B_raw | 75.21% | not run | — |
| C: Native confidence + NCP rank | 76.01% | not run | — |
| A: even-only | 0.08% | not run | — |
| A_odd control | 0.08% | not run | — |

B is the only surviving construction. It trails NCP by `0.31pp` on the first
block, leads by `0.62pp` on the disjoint second block, and leads by `0.15pp`
when the two equal-sized blocks are combined. This is a near-tie, not a large
new Native improvement. The B versus B_raw gap on block 1 supports the value of
preserving Native local probabilities and total far-key mass rather than merely
splicing Native-local and NCP-far logits.

A and its odd control collapse almost completely and are closed. C reproduces
carrier-NCP-level Full-13 behavior but does not improve it.

## Natural-QA and language-model health

| Method | Natural-QA task-macro F1 | PPL46 at 4K |
|---|---:|---:|
| Native baseline | 39.41% | different formal LM128 contract; do not mix |
| NCP baseline | 38.46% | different formal LM128 contract; do not mix |
| B: mass projection | 39.02% | 9.92377 |
| B_raw | 38.82% | 9.92592 |
| C: rank assignment | 38.07% | 9.94471 |

B recovers most of Native Natural-QA and is the best of the three new
operators on matched QA and PPL46, but it does not exceed Native QA. Its PPL
advantage over B_raw is only `0.00022` NLL (`9.92377` versus `9.92592` PPL), so
the evidence is structural consistency rather than a practically large LM win.

## Evidence

- [B block 1 summary](reports/server_20260917/runs/mass_projection/summary.json)
  and [raw rows](reports/server_20260917/runs/mass_projection/generations.jsonl)
- [NCP block 2 summary](reports/server_20260917/validation/ncp_block2/summary.json)
  and [raw rows](reports/server_20260917/validation/ncp_block2/generations.jsonl)
- [B block 2 summary](reports/server_20260917/validation/mass_projection_block2/summary.json)
  and [raw rows](reports/server_20260917/validation/mass_projection_block2/generations.jsonl)
- [B QA/PPL summary](reports/server_20260917/validation/mass_projection_qa99_ppl46/summary.json)
- [B_raw QA/PPL summary](reports/server_20260917/validation/mass_raw_qa99_ppl46/summary.json)
- [C QA/PPL summary](reports/server_20260917/validation/rank_assignment_qa99_ppl46/summary.json)
- [Original Native/NCP result owner](../native_enhancement_oral_20260915/index.md)

The complete server artifact mirror under `reports/server_20260917/` includes
contracts, runtime receipts, summaries, LM rows, and generation rows for every
reported new arm.
