# Attention-aware phase-chord allocation experiment

- **Date:** 2026-08-21
- **Status:** `COMPLETE_TWO_SEED_INTERNAL`
- **Paper role:** internal method discovery; no manuscript claim or figure
- **Question:** can measured attention demand be transformed through the RoPE
  operator to obtain a fixed-support table that improves both the training
  window and extrapolation?
- **Evidence receipt:** [`../evidence/RESULTS_20260821.json`](../../evidence/RESULTS_20260821.json)

## 1. Protocol

The scientific training contract is the existing 151.9M exact-range pipeline:

- 151,898,880 parameters, train length 256;
- 499,974,144 stored training tokens, 7,629 optimizer steps;
- global batch 256, BF16, fused AdamW;
- within each seed, the candidate and controls share the stored-token prefix,
  seed-specific row order, validation anchors, and final-128-token NLL
  evaluator;
- fixed sampled endpoints equal to the paper-faithful FMRoPE `base=256`
  support;
- evaluation lengths 256/512/1024/2048, 32 anchors per length.

Seed 137 is the method-selection pilot. Its demand profile was measured from
the completed seed-137 FMRoPE checkpoint. Seed 42 is an out-of-profile
confirmation run and uses its canonical `64 x 4` micro-batch/accumulation
geometry, versus `128 x 2` for seed 137. The seeds intentionally retain
different row-order and execution-protocol hashes; comparisons are matched
within seed, not treated as bitwise-identical executions. Seed 256 was not
launched because the author requested shutdown after seed 42.

## 2. R0 measurement

R0 streamed attention-distance mass from causally masked attention
probabilities without storing attention matrices. It was run on:

1. the mature Native OLMo-2 1.485B Instruct checkpoint over 1,024 separate
   4K FineWeb-Edu sequences, 32 query positions, all 16 layers and 16 heads;
2. seed-137 151.9M FMRoPE and anchored EVQ-Cosh checkpoints over 1,024 length-256
   validation windows, all 12 layers and 12 heads.

The direct endpoint-log occupancy profile was non-uniform but did not pass the
pre-registered multi-peak gate. Anchored EVQ-Cosh increased the 151.9M
long-half attention mass from `0.09105` to `0.10178`, consistent with learned
table/attention co-adaptation. This is descriptive and not a causal
decomposition.

During audit, the bootstrap storage was corrected from one aggregate row per
execution batch to one row per independent sequence. The preserved R0 owners
contain the corrected 1,024-sequence analyses. The later Q/K pair-band profile
implementation passed its code tests but was not re-collected because it was
not used to choose or judge the phase-chord schedule.

## 3. Candidate definitions

### Direct distance map — falsified

The initial candidate mapped attention distance to frequency coordinate in the
same direction and used

\[
\rho_\lambda(\phi)\propto[(1-\lambda)m(\phi)+\lambda]^{1/3}.
\]

At `lambda=0.1`, its median normalized coordinate is `0.568`, slower-heavy
than FMRoPE's uniform log allocation (`0.500`).

### Phase-chord map

RoPE consumes distance through phase, so the operator-aware demand is

\[
m_{\rm chord}(\phi)=
\mathbb E_{\Delta\sim D_{\rm att}}
[1-\cos(\omega(\phi)\Delta)],
\]

followed by the same cube-root high-rate allocation with `lambda=0.1`. Its
median coordinate is `0.435`, between FMRoPE (`0.500`) and anchored EVQ-Cosh
(`0.173`). It uses the same finite endpoints as both controls.

This kernel is the squared chord energy of one rotary pair up to a constant.
It is a bounded phase-discrimination surrogate, not an identified derivative
of LM loss.

## 4. Seed-137 result

### Direct map

| Length | direct minus FMRoPE tail NLL | share of anchored EVQ-Cosh OOD gain |
| ---: | ---: | ---: |
| 256 | -0.0071 | n/a |
| 512 | +0.0735 | negative |
| 1024 | +0.0900 | negative |
| 2048 | -0.0198 | 11% |

Decision: `STOP_DIRECT_DISTANCE_MAP`. Attention occupancy is not a valid
frequency-demand oracle without the rotary phase kernel.

### Phase chord

| Length | FMRoPE tail NLL | anchored EVQ-Cosh tail NLL | phase-chord tail NLL | phase minus FMRoPE | retained anchored EVQ-Cosh gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 3.33992 | 3.36895 | 3.33169 | **-0.00824** | n/a |
| 512 | 4.94114 | 4.67063 | 4.70416 | **-0.23698** | 87.6% |
| 1024 | 5.76801 | 5.57712 | 5.61603 | **-0.15198** | 79.6% |
| 2048 | 6.36928 | 6.18845 | 6.15409 | **-0.21519** | 119.0% |

Seed 137 therefore improves every endpoint, including the training window,
while retaining most or all of the anchored EVQ-Cosh extrapolation gain. This is
a method-selection result because the seed-137 FMRoPE checkpoint supplied the
demand profile.

## 5. Seed-42 schedule-frozen confirmation

The schedule derived from seed 137 was frozen before seed 42 training. Seed 42
completed all 7,629 steps and 499,974,144 tokens in 2,813 seconds on an RTX
5090, using its canonical `64 x 4` micro-batch/accumulation geometry. Evaluation
used the same 32 anchors and final-128-token tail-NLL contract as the controls.

| Length | FMRoPE tail NLL | anchored EVQ-Cosh tail NLL | phase-chord tail NLL | phase minus FMRoPE | retained anchored EVQ-Cosh gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 3.34589 | 3.37865 | 3.35554 | **+0.00965** | n/a |
| 512 | 4.96180 | 4.48430 | 4.87775 | **-0.08405** | 17.6% |
| 1024 | 5.83497 | 5.62999 | 5.67542 | **-0.15956** | 77.8% |
| 2048 | 6.33568 | 6.22284 | 6.14043 | **-0.19525** | 173.0% |

The schedule-frozen seed confirms the important direction: phase-chord improves
all three extrapolation lengths over FMRoPE while holding the 256 cost below the
pre-registered `+0.01` gate. It does **not** pass the stricter requirement to
retain at least 80% of anchored EVQ-Cosh's gain at every OOD length; the 512
gain contracts substantially, and 1024 falls just below that threshold.

Across the two completed seeds, the unweighted training-seed means are:

| Length | phase minus FMRoPE tail NLL | anchored EVQ-Cosh minus FMRoPE tail NLL | phase PPL change vs FMRoPE |
| ---: | ---: | ---: | ---: |
| 256 | **+0.00070** | +0.03089 | +0.07% |
| 512 | **-0.16051** | -0.37400 | -14.83% |
| 1024 | **-0.15577** | -0.19793 | -14.42% |
| 2048 | **-0.20522** | -0.14684 | -18.55% |

This is a promising Pareto shift: the average in-window cost is nearly
removed, both seeds improve every OOD endpoint over FMRoPE, and the 2K gain is
larger than anchored EVQ-Cosh in both seeds. With only two training seeds, one of
which selected the method, these means are descriptive and do not support a
significance claim or manuscript promotion.

## 6. Numerical and implementation audit

- Fixed an outer quartic bisection bug: after hitting the normalization root,
  the solver took a second interval midpoint instead of returning the hit.
- Moved the small-tau quartic check out of the float64 cancellation regime.
- Corrected finite RoPE support documentation/tests to
  `omega_min=base^(-(K-1)/K)`.
- Required strictly positive demand for the zero-uniform-mixture endpoint.
- Changed R0 bootstrap storage from batch aggregates to per-sequence rows.
- Added streamed per-layer/head Q/K 2D-pair L2 profiles using the actual
  frequency-band definition; this is not a positional/symbolic score.

Final validation completed locally in Conda `aidemo`:

- 22 attention-demand, numerical, and schedule-contract tests passed;
- 11 mature-retrofit preparation-contract tests passed;
- the standalone phase-demand self-test and report-metric assertions passed.

The compact receipt preserves source hashes and exact metrics without copying
machine-specific paths into the durable research directory.

## 7. Decision boundary

The completed evidence supports:

- direct distance mapping is falsified under the tested protocol;
- the phase-chord kernel moves the FMRoPE/anchored EVQ-Cosh Pareto frontier
  under two completed 151.9M training seeds;
- the frozen seed-42 schedule preserves the OOD direction while reducing the
  in-window cost relative to anchored EVQ-Cosh.

It does not support:

- a paper-facing replicated method claim or generic significance statement;
- the stronger gate of retaining at least 80% of anchored EVQ-Cosh gain at
  every OOD length;
- mature-model retention or LoRA success;
- replacing EVQ-Cosh in the current manuscript;
- universal optimality of the chord kernel or `lambda=0.1`.

Decision: `PROMISING_PARETO_SHIFT_NOT_PAPER_READY`. Do not launch a larger
method run until the mature-model preservation question has a clean operator
and diagnostic. If this table is later promoted, seed 256 is the smallest
missing replication.

The next mature-model research route is specified separately in
[`../README.md`](../../README.md).
