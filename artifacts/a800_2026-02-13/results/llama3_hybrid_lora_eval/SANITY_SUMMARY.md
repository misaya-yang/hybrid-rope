# SANITY CHECKS SUMMARY

- Timestamp: 2026-02-13_072657
- Device: NVIDIA A800-SXM4-80GB
- Lengths: [2048, 8192, 12288, 14336, 16384]
- Chunks per length: 5

## Overall

- Check A (numeric stability): PASS
- Check B (position indexing): PASS
- Check D (slice robustness @16K): PASS
- Overall: PASS

## A. Numeric Stability

| Model | Strategy | Length | n | all finite | PPL |
|---|---|---:|---:|---|---:|
| base_unfinetuned | sequential | 2048 | 5 | True | 15.889 |
| base_unfinetuned | sequential | 8192 | 5 | True | 13.633 |
| base_unfinetuned | sequential | 12288 | 5 | True | 53.25 |
| base_unfinetuned | sequential | 14336 | 5 | True | 127.74 |
| base_unfinetuned | sequential | 16384 | 5 | True | 190.566 |
| base_unfinetuned | random_start | 2048 | 5 | True | 14.467 |
| base_unfinetuned | random_start | 8192 | 5 | True | 16.291 |
| base_unfinetuned | random_start | 12288 | 5 | True | 55.806 |
| base_unfinetuned | random_start | 14336 | 5 | True | 113.437 |
| base_unfinetuned | random_start | 16384 | 5 | True | 192.963 |
| hybrid_lora | sequential | 2048 | 5 | True | 15.352 |
| hybrid_lora | sequential | 8192 | 5 | True | 13.506 |
| hybrid_lora | sequential | 12288 | 5 | True | 14.198 |
| hybrid_lora | sequential | 14336 | 5 | True | 14.973 |
| hybrid_lora | sequential | 16384 | 5 | True | 15.4 |
| hybrid_lora | random_start | 2048 | 5 | True | 14.198 |
| hybrid_lora | random_start | 8192 | 5 | True | 17.451 |
| hybrid_lora | random_start | 12288 | 5 | True | 14.603 |
| hybrid_lora | random_start | 14336 | 5 | True | 16.089 |
| hybrid_lora | random_start | 16384 | 5 | True | 14.467 |

## B. Position Indexing Consistency

| Model | Length | max_position_id | max abs diff (default vs explicit) | consistent | out_of_bound_vs_config |
|---|---:|---:|---:|---|---|
| base_unfinetuned | 2048 | 2046 | 0.000e+00 | True | False |
| base_unfinetuned | 8192 | 8190 | 0.000e+00 | True | False |
| base_unfinetuned | 12288 | 12286 | 0.000e+00 | True | True |
| base_unfinetuned | 14336 | 14334 | 0.000e+00 | True | True |
| base_unfinetuned | 16384 | 16382 | 0.000e+00 | True | True |
| hybrid_lora | 2048 | 2046 | 0.000e+00 | True | False |
| hybrid_lora | 8192 | 8190 | 0.000e+00 | True | False |
| hybrid_lora | 12288 | 12286 | 0.000e+00 | True | True |
| hybrid_lora | 14336 | 14334 | 0.000e+00 | True | True |
| hybrid_lora | 16384 | 16382 | 0.000e+00 | True | True |

## C. Mid-Length Trend (12K/14K/16K)

| Model | Strategy | PPL@12K | PPL@14K | PPL@16K | Trend |
|---|---|---:|---:|---:|---|
| base_unfinetuned | sequential | 53.25 | 127.74 | 190.566 | gradual_degradation |
| base_unfinetuned | random_start | 55.806 | 113.437 | 192.963 | gradual_degradation |
| hybrid_lora | sequential | 14.198 | 14.973 | 15.4 | gradual_degradation |
| hybrid_lora | random_start | 14.603 | 16.089 | 14.467 | stable_or_non_monotonic |

## D. Slice Robustness @16K

| Strategy | base PPL@16K | hybrid PPL@16K | base/hybrid | hybrid better |
|---|---:|---:|---:|---|
| sequential | 190.566 | 15.4 | 12.374 | True |
| random_start | 192.963 | 14.467 | 13.338 | True |

## Risks

- 16K position ids exceed config max_position_embeddings=8192; no mismatch vs explicit ids was observed, but this remains an extrapolation regime.
