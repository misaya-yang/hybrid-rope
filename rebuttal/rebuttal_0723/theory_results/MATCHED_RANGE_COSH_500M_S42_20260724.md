# Exact-range Cosh allocation diagnostic — 500M tokens, seed 42

Date: 2026-07-24

Status: **complete / single-seed supporting diagnostic**

Reviewer-facing aggregate note: the completed three-training-seed result is
summarized in `MATCHED_RANGE_COSH_500M_3SEED_20260724.md`. This file remains the
raw-backed seed-42 provenance record and should not be used as the aggregate
headline.

## Question

When the sampled frequency extrema and log-span are exactly matched to the
local FMRoPE grid, does changing only the interior allocation to Cosh still
change language-model NLL?

This is the narrow range-versus-allocation test requested by AC.1/AC.3. It is
not a new paper-primary result.

## Protocol

- Model: 151,898,880 parameters.
- Seed: 42.
- Training length: 256.
- Training budget: 499,974,144 tokens, 7,629 optimizer steps.
- Data, initialization, row order, optimizer, LR schedule, global/micro batch,
  and 32 evaluation anchors match the retained 500M FMRoPE baseline.
- Baseline: uniform FMRoPE exponent grid, training base 256.
- New arm: endpoint-normalized Cosh interior spacing at \(\tau=4\), with the
  exact same highest sampled frequency, lowest sampled frequency, and log-span
  as the FMRoPE baseline.
- Evaluation:
  - `fixed`: retain each checkpoint's training range;
  - `target-matched`: set the range to the declared evaluation length while
    preserving each arm's normalized interior spacing.
- Metric: paired final-128-token teacher-forced NLL over 32 frozen anchors.
  Reported PPL is \(\exp(\text{mean NLL})\).

Execution used FP32 master weights, BF16 autocast, Flash-only SDPA, fused AdamW,
and `torch.compile`. Training completed in 2,960.6 seconds.

## Results

### Absolute PPL

| condition | 256 | 512 | 1,024 | 2,048 |
| --- | ---: | ---: | ---: | ---: |
| FMRoPE, fixed training range | **28.386** | 142.851 | 342.056 | 564.353 |
| Cosh interior, fixed matched range | 29.331 | **88.615** | **278.659** | **504.131** |
| FMRoPE, target-matched range | **28.386** | **28.038** | **35.209** | **100.111** |
| Cosh interior, target-matched range | 29.331 | 29.804 | 42.230 | 132.271 |

### Paired NLL attribution

Negative means the Cosh-interior arm is better.

| Cosh minus uniform FMRoPE | 256 | 512 | 1,024 | 2,048 |
| --- | ---: | ---: | ---: | ---: |
| Fixed matched range | +0.03276 | **-0.47750** | **-0.20499** | **-0.11284** |
| Target-matched range | +0.03276 | +0.06109 | +0.18182 | +0.27857 |

Fixed-range Cosh wins 32/32, 27/32, and 22/32 anchors at
512/1,024/2,048. Under target-matched range it wins only 5/32, 1/32, and
2/32 anchors.

## Interpretation

The positive result is clean but narrow:

> With sampled extrema and span held exactly fixed, changing only the
> finite-\(K\) interior allocation materially changes extrapolation NLL.

At fixed training range, Cosh improves all three OOD lengths. This isolates an
allocation-shape effect that a scalar range match cannot remove.

The combination result is negative:

> The tested Cosh shape does not add to target-aware FMRoPE retargeting.

When both arms receive the correct target-matched range, the uniform FMRoPE
grid is better at every evaluated length. The diagnostic therefore supports
allocation as a distinct design variable, but not orthogonality, additive
gains, or universal optimality of Cosh.

## Rebuttal use

Safe:

- “At identical sampled extrema and span, a Cosh interior allocation improves
  fixed-range OOD NLL by 0.478/0.205/0.113 at 512/1K/2K.”
- “This shows that finite-channel allocation is not reducible to a scalar
  base/range change.”
- “Target-aware FMRoPE remains stronger when both schedules are retargeted.”

Do not claim:

- Cosh improves FMRoPE under target-aware deployment;
- empirical orthogonality or synergy;
- multi-seed or large-model confirmation.

## Provenance

- Comparison JSON:
  `results/fmrope_evq2_500m_s42_20260724/comparison/comparison.json`,
  SHA256
  `801792f062fe680afe1570fbf23ed710d77be0659c03a3924361b79ccbf334b1`.
- New-arm raw evaluation:
  `results/fmrope_evq2_500m_s42_20260724/evaluation/results.json`,
  SHA256
  `a4f381dd1f9d98811f8ee6ca396eae7361e06848389105e8d54e6a9281ee9f3f`.
- Training metadata SHA256:
  `a8a366c46b96fa804959f6a43c40b2daa8225b7ebf1bbc1567269b0637e71744`.
- Immutable training frequency SHA256:
  `e0b201711857c01b85933b4f8618929f5124eee1827b014c4d9b03a47e48a0eb`.
- Trainable initialization SHA256:
  `fb17648236fc6b976795f6f4985dc055421b0122fae594382c7cb99937872452`.
- Row-order SHA256:
  `c6fc5b4d7dea51134ad609ca87864e54b8cf46c169838cd08e1a20b9bcf47452`.
