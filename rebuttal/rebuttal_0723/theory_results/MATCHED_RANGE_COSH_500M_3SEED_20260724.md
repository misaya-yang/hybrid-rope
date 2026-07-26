# Exact-range Cosh allocation — three-training-seed aggregate

Date: 2026-07-24

Status: **runs complete / author-confirmed aggregate / local raw promotion pending**

## Reviewer or AC concern addressed

- `AC.1`: whether EVQ is merely another base/range choice.
- `AC.3`: whether the Cosh intervention has direct controlled attribution.

This is the method-identification experiment. It tests whether changing only
the interior locations of a finite RoPE grid changes trained-model behavior
after every sampled scalar range quantity is held fixed.

## Existing evidence

The seed-42 arm and its raw receipts are retained in
`MATCHED_RANGE_COSH_500M_S42_20260724.md`. The current three-seed aggregate was
reported by the authors after seeds 137 and 256 completed. A local sanitized
aggregate containing the per-seed contrasts, source hashes, and confidence
intervals has not yet been promoted into this checkout.

## Protocol

- Model: 151,898,880 parameters.
- Training seeds: 42, 137, and 256.
- Training length: 256.
- Training budget: 499,974,144 tokens per arm.
- Evaluation: final-128-token teacher-forced NLL on 32 frozen anchors.
- Within each seed pair, the two arms use the same:
  - sampled highest frequency;
  - sampled lowest frequency;
  - log-frequency span;
  - model architecture and trainable initialization;
  - token order;
  - optimizer and training budget;
  - frozen evaluation anchors.
- Baseline: FMRoPE's geometric grid, uniform in log-frequency coordinates.
- Intervention: endpoint-normalized Cosh spacing at \(\tau=4\).
- The only changed variable is the location of the \(K-2=30\) interior
  frequencies.

“Matched initialization” is a within-seed-pair statement; the three training
seeds are independently initialized.

## Author-confirmed aggregate

Negative means that the Cosh interior has lower NLL.

| Cosh minus uniform FMRoPE | 512 | 1,024 | 2,048 |
| --- | ---: | ---: | ---: |
| Fixed training range, three-seed mean | **-0.3159** | **-0.1949** | **-0.1674** |
| Training seeds favoring Cosh | **3/3** | **3/3** | **3/3** |

The authors also report that, after both grids are retargeted to the evaluation
length, the three-seed mean favors uniform FMRoPE at every tested OOD length.
The local record does not yet contain the retargeted numeric aggregate or
per-seed direction counts, so neither is reconstructed here.

## Interpretation

The supported identification claim is:

> In this controlled protocol, finite training-time interior allocation is a
> separately identifiable design variable and the observed fixed-range effect
> cannot be reduced to scalar base or sampled-range selection.

The adjacent deployment boundary is:

> When both grids receive target-aware range transport, uniform FMRoPE is
> stronger at every tested OOD length in the reported three-seed mean.

These two findings are compatible. The first identifies the optimization
variable; the second says that the author-reported three-seed mean favors
target-aware range transport in the tested deployment. They do not establish
empirical orthogonality,
additive gains, universal Cosh optimality, or superiority over FMRoPE.

The Cosh arm here is an endpoint-normalized shape diagnostic. It is not
identical to the submitted raw midpoint EVQ grid, whose \(\tau\) also changes
sampled extrema and span.

## Safe reviewer-facing paragraph

> We added a post-submission three-training-seed exact-range control. Within
> each seed pair, the two arms used the same 151.9M architecture, trainable
> initialization, token order, optimizer, 499,974,144-token budget, and 32
> frozen evaluation anchors. Their sampled highest and lowest frequencies and
> log-frequency span were identical; only the 30 interior frequency positions
> differed. At the fixed training range, endpoint-normalized Cosh minus
> FMRoPE's geometric, uniform-in-log grid was
> \(-0.3159/-0.1949/-0.1674\) final-128-token NLL at 512/1K/2K, with all three
> training-seed contrasts favoring Cosh. This controlled result rules out
> scalar base/range selection as a complete explanation in this protocol:
> finite training-time interior allocation is separately identifiable. When
> both grids were retargeted, uniform FMRoPE was stronger at every tested OOD
> length in the three-seed mean. This is a deployment boundary, not evidence
> that allocation and range transport are equivalent or additive.

## Send gate

Before these aggregate numbers are sent externally, obtain and promote a raw
three-seed aggregate and verify:

1. per-seed NLL contrasts;
2. data-manifest, code, initialization, row-order, frequency, and evaluation
   receipts;
3. the three-training-seed confidence intervals using training seed, not
   evaluation anchor, as the statistical unit;
4. retargeted numeric means and per-seed directions.

Until that promotion is complete, do not add confidence intervals, statistical
significance language, retargeted numeric values, or a `RAW_BACKED` label.

## Current tracked provenance

- Structured aggregate:
  `matched_range_cosh_500m_3seed_result_20260724.json`.
- Seed-42 raw-backed report:
  `MATCHED_RANGE_COSH_500M_S42_20260724.md`.
- Aggregation implementation:
  `experiments/fmrope_125m_l256_500m/run_experiment.py`.
