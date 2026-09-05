# Native-isotonic profile exposes an endpoint-dependent tradeoff

- **Date:** 2026-09-03
- **Status:** `COMPLETE / VALID EXECUTION / REUSED 1X DOUBLE GATE PASSED /
  NO UNIFORM DOMINANCE / FRESH SEED-202609037 CORE-4 NEGATIVE AT 4K AND 8K`
- **Evidence labels:** numerical construction identities are **Derived results**;
  executed endpoint values are **Observations**; the fresh 4K/8K contrast is a
  **Negative result** for replacing current-p2 with this candidate at those
  endpoints.
- **Question:** Does the declared squared Native-geometry surrogate
  `m=pinned-Iso(1-u)`, installed as one global log-s4 table with the inherited
  `c=.074` gain, improve the joint Native/long operating point?
- **Preflight:**
  [`NATIVE_ISOTONIC_PROFILE_PREFLIGHT_20260903.md`](../../preflights/zero-training-deployment/NATIVE_ISOTONIC_PROFILE_PREFLIGHT_20260903.md)
- **Theory boundary:**
  [`NATIVE_ONLY_MOVEMENT_PROFILE_IDENTIFIABILITY_20260903.md`](../../theory/NATIVE_ONLY_MOVEMENT_PROFILE_IDENTIFIABILITY_20260903.md)
- **Compact receipt:**
  [`NATIVE_ISOTONIC_PROFILE_RECEIPT_20260903.json`](../../evidence/NATIVE_ISOTONIC_PROFILE_RECEIPT_20260903.json)

## 1. Decision

The candidate produces a reproducible endpoint-dependent tradeoff, but it is
**not** a replacement for the current `log_s4` profile.

It passes the reused OLMo 1x double gate; its point estimates have more gate
margin than current-p2:

- PG-19 PPL retention: `0.885660` versus `0.875302`;
- five-task retention: `0.969075` versus `0.915103`.

Its PG-19 tail-NLL point estimates at 2x/4x are lower by
`-0.026968/-0.035009`, with paired 95% intervals
`[-0.039705,-0.015417]` and `[-0.051515,-0.019929]`. On the matched
development RULER-13 panel it is essentially neutral: 4K/8K/16K deltas are
`-0.008205/+0.012628/+0.004744`.

The fresh seed rejects a stronger conclusion. On core-4, primary minus
current-p2 is:

- 4K: `-0.1450`, paired 95% interval `[-0.2350,-0.0550]`;
- 8K: `-0.1750`, interval `[-0.2600,-0.0925]`;
- 16K: `+0.0075`, interval `[-0.0650,+0.0750]`.

Thus the squared-loss geometry surrogate finds a table with lower measured
natural NLL and higher short natural-retention point estimates relative to
current-p2 under the inherited gain, while sacrificing fresh structured
retrieval/tracking at 4K/8K. The tested surrogate does not identify one profile
that dominates current-p2 across the measured endpoints. Current-p2 remains the
safer tracked engineering profile.

No exponent, gain, rank cutoff, normalization, boundary, head selector, or
rescue arm follows from this result.

## 2. Frozen construction and numerical entrance

The authoritative Native float32 `inv_freq` hash is
`dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34`.
The builder uses all integer lags `0..4095`, causal pair-count weights, a
high-precision closed-form `128 x 128` phase Gram, exact-rank block Schur
residuals, endpoint-pinned interior PAVA, and

\[
\omega_k'=\omega_k4^{-m_k}.
\]

The initial `200/240/280` decimal-digit ladder did not converge and produced no
LM result. The amended `360/440/520` ladder converged: 440 and 520 digits gave
byte-identical float64 movement. The primary float32 tensor hash is
`3266663596a113b0bf8edbf5e89f57254fd96d980eb353614b9c66ea65f25bfa`.

This is point-MP convergence, not an interval-arithmetic certificate of exact
rounding. Moving every Native frequency simultaneously down or up by one
float32 ULP changed movement
by at most `1.634e-7` at pair 14. Primary versus current-p2 differs by `0.9971`
at pair 16, so the tested contrast is not numerical-rank noise. Formal
preflight and a Flash-only 4K/16K smoke passed; the smoke scored `1/1` on its
single-key rows and owns no selection claim.

## 3. Factorial attribution at 1x

All non-Native arms use the frozen attention scaling
`1.102585782722872`. Native values are PG-19 NLL `2.971047461` and five-task
macro `0.345134307`.

| Phase diagnostic | Link | PG-19 NLL | PPL retention | Five-task macro | Task retention |
| --- | --- | ---: | ---: | ---: | ---: |
| legacy-u | p2 | `3.104234` | `0.875302` | `0.315833` | `0.915103` |
| legacy-u | Iso | `3.093035` | `0.885159` | `0.321023` | `0.930139` |
| exact-u | p2 | `3.108764` | `0.871345` | `0.329253` | `0.953986` |
| **exact-u** | **Iso** | **`3.092470`** | **`0.885660`** | **`0.334461`** | **`0.969075`** |
| Native table | inherited gain only | `3.085932` | `0.891469` | `0.378216` | `1.095851` |

The exact-u by Iso interaction is `-0.005096` NLL but only `+0.000018` task
macro. In words:

- replacing p2 with the declared linear link improves likelihood and task
  retention under legacy-u;
- exact-u under p2 improves task retention but worsens the likelihood gate;
- exact-u/Iso recovers the likelihood gate and has the best task retention of
  the four movement profiles;
- gain-only is best at these 1x downstream tasks, confirming that the 1x result
  is a table-by-gain operating point rather than a pure-m claim.

Primary minus current-p2 paired estimates are `-0.011764` PG-19 NLL, 95%
bootstrap `[-0.025070,+0.000937]`, and `+0.018628` five-task macro,
`[-0.036407,+0.073927]`. Point estimates improve, but these paired intervals
do not separate at this row count.

## 4. Matched natural long-context matrix

The current evaluator reproduced the historical current-p2 values exactly.

| Endpoint | Current-p2 | Native-isotonic | Candidate minus current |
| --- | ---: | ---: | ---: |
| PG-19 NLL, 2x | `3.083278` | **`3.056309`** | `-0.026968` |
| PG-19 NLL, 4x | `3.081946` | **`3.046936`** | `-0.035009` |
| Six-task macro, 2x | **`0.307614`** | `0.281901` | `-0.025712` |
| Six-task macro, 4x | `0.260055` | **`0.263761`** | `+0.003705` |

The task-macro paired intervals are `[-0.065973,+0.012378]` at 2x and
`[-0.037345,+0.046615]` at 4x. PG-19 and generated-task endpoints therefore
must remain separate: better likelihood did not imply uniformly better task
generation.

## 5. Development RULER-13

Both arms use the same current evaluator, data manifest, decoder, scorer, rows,
and gain.

| Length | Current-p2 | Native-isotonic | Delta |
| --- | ---: | ---: | ---: |
| 4K | **`0.713974`** | `0.705769` | `-0.008205` |
| 8K | `0.667051` | **`0.679679`** | `+0.012628` |
| 16K | `0.498590` | **`0.503333`** | `+0.004744` |

These small task-equal macro changes contain opposing task-level shifts. They
support retained development-panel RULER capability, not uniform improvement.

## 6. Fresh core-4 confirmation

The candidate was frozen before official RULER core-4 seed `202609037` was
generated. Each arm has 20 rows per task at 4K/8K/16K; the tasks are
single-key, numeric multikey, UUID multikey, and variable tracking.

Intervals use paired row differences, resampled within each of the four tasks
and then task-equal macro averaged: 10,000 bootstrap replicates, RNG seed
`202609038`, two-sided marginal 95% percentile intervals. They are not
simultaneous or familywise-adjusted intervals. This confirmation is one fresh
seed, four tasks, 20 rows per task/length, and the same frozen evaluator,
decoder, scorer, table, and inherited gain.

| Length | Current-p2 | Native-isotonic | Delta | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| 4K | **`0.8375`** | `0.6925` | `-0.1450` | `[-0.2350,-0.0550]` |
| 8K | **`0.7225`** | `0.5475` | `-0.1750` | `[-0.2600,-0.0925]` |
| 16K | `0.4025` | `0.4100` | `+0.0075` | `[-0.0650,+0.0750]` |

At 4K the task deltas are `0/-0.10/-0.25/-0.23`; at 8K they are
`0/-0.25/-0.35/-0.10`; at 16K they are `0/+0.10/-0.10/+0.03`, ordered as
single-key / numeric multikey / UUID multikey / VT. The adverse 4K/8K effects
are not one-task artifacts.

This is the decisive limit on promotion. The candidate may be useful when
natural likelihood is the deployment priority, but it does not dominate the
current profile's structured short/mid-range capability.

## 7. Supported and unsupported claims

### Supported

- The high-precision exact-u construction and final table are deterministic
  under the recorded point-MP contract and insensitive to a global one-ULP
  perturbation relative to the tested table difference.
- The exact-u/Iso composite passes the reused 1x natural double gate and
  improves PG-19 NLL at 2x/4x.
- It retains development RULER-13 approximately, with task redistribution.
- On a fresh core-4 seed it is materially worse than current-p2 at 4K/8K and
  unresolved at 16K.
- The measured endpoints exhibit an endpoint-dependent tradeoff; no formal
  Pareto frontier or deployment utility was identified.

### Unsupported

- a unique checkpoint-derived behavioural law, universal optimum, or static
  replacement for current-p2;
- pure-m attribution independent of the inherited gain;
- cross-checkpoint/K transfer, natural-QA superiority, SOTA, or universal
  context extension;
- treating the `0.875` operating threshold as a scientific discontinuity;
- any parameter rescue, selector sweep, or manuscript promotion.

## 8. Stop decision

The prospective tree stops for the exact
`full-lag-u + pinned-Iso + log-s4 + c=.074` candidate as a universal or
current-p2 replacement. The fresh confirmation supplied the missing
counterevidence for that decision, so more seeds or curves are not needed to
rescue this exact claim. This does not close the isotonic-link class, unit-gain
variants, other tasks/checkpoints, or a likelihood-priority deployment use.
Raw rows, predictions, logs, commands, exit codes, numerical arrays, and hashes
remain on the work machine; the repository retains only this bounded owner and
a machine-path-free compact receipt.
