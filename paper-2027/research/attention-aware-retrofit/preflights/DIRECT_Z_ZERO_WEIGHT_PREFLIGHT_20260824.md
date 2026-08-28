# Direct fixed-support `z` zero-weight preflight (2026-08-24)

Status: executed; stopped by
[`../results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md`](../results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md)
before downstream evaluation.

## Decision this experiment can change

The missing mature-checkpoint result is whether the paper's third axis --
interior spectral allocation at fixed sampled support -- is actionable as a
simple retrofit rather than only a from-scratch causal explanation. A positive
result would justify methodizing `z`; a negative result would stop direct table
calibration before LoRA.

## Frozen intervention

- Checkpoint: released OLMo-2-0425-1B-Instruct, bound by the existing READY
  receipt.
- Model weights: frozen, zero updates.
- Support: exact released Native sampled endpoints and log span.
- Variable: 63 positive-gap logits, with 62 effective simplex degrees of
  freedom, whose cumulative normalized gaps define interior coordinates `z`;
  endpoints remain exactly 0 and 1.
- Runtime: one static table at both 1x and 2x; attention scaling exactly 1.0;
  no target-aware or observed-request routing.
- Optimization data: four independently prepared 8K FineWeb-Edu rows; rows
  0--1 design, rows 2--3 held out.
- Objective: minimize design 2x tail NLL with a per-row 1x Native no-harm
  penalty. No attention prior, collision surrogate, or Cosh prior is used.

## Frozen pilot gates

The full ten-step calibration passes only when all are true:

1. the realized table differs from Native while retaining exact Native support;
2. every held-out row has final-64-token teacher-forced 1x tail delta NLL at
   most `+0.05`;
3. mean held-out final-64-token 2x tail delta NLL is strictly below zero;
4. no held-out row has 2x tail delta NLL above `+0.05`.

A one-step smoke checks runtime and identity only and cannot satisfy the
scientific gate. The four-row pilot is candidate-selection evidence, not a
paper result.

## Promotion and stop conditions

If the pilot passes, evaluate the identical hash-bound table on a small PG-19
1x/2x screen and an 8K RULER screen. Human inspection is required before the
20-row-per-cell natural-task screen; full Qasper and 2Wiki are last. Stop at the
first material 1x regression, absent 2x improvement, identity drift, non-finite
loss, OOM, or intervention mismatch.

Only a surviving result would motivate the LoRA follow-up belonging to this
direct-calibration branch. This gate does not govern the already completed
zero-training Native/s4 policy or the separate mature-adaptation owners. A
branch-local LoRA would test whether limited co-adaptation closes its residual
gap; it is not part of the causal claim that `z` itself matters.

## Claim boundary

This uses an explicit ratio-2 workload rather than an absolute `L_target`.
It therefore tests a model-relative 2x retrofit, not a horizon-free optimum.
"Zero-weight" must never be rewritten as zero-search or zero learned
parameters: the table is calibrated through 63 gap logits (62 effective
allocation degrees of freedom).
