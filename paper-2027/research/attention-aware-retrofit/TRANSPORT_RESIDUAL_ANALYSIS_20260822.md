# Transport residual: what a Q/K adapter can and cannot repair

- **Date:** 2026-08-22
- **Status:** CPU-only analysis complete; no checkpoint loaded, no GPU, no training
- **Evidence role:** internal mechanism analysis and candidate-table screen.
  It is a frequency-table result under an isotropic content model, never a task,
  capability, or checkpoint result.
- **Implementation:** `scripts/analysis/rope_transport/`
- **Receipt:** `receipt_20260822.json`, SHA-256
  `29516f39f80a81a413b376d5d8de5ca16c3880cabb1cf701af29e9ab8a27bfb3`
- **Executed on:** remote CPU-only process, `CUDA_VISIBLE_DEVICES=-1`,
  NumPy 2.3.2, 561.9 s wall, 14/14 unit tests passed locally and on the host

## 1. The three objects have disjoint scopes

For one head the logit at relative distance `D` is
`l(q,k,D) = q^T R_Omega(D) k = sum_k A_k cos(w_k D + psi_k)`.
The table is a **basis**; content is only **coordinates**.

- Pretrained attention fits coordinates on `[0, L]`. Behaviour beyond `L` is the
  forced analytic continuation of that fit, not a separate learnable thing.
- A non-geometric allocation changes the **basis**: it moves both in-window
  representability and out-of-window continuation.
- A Q/K LoRA is a fixed, position-independent content map `q -> Mq`, `k -> Nk`.
  It changes coordinates and can never leave `span(Omega')`.

So "can LoRA repair a table change" is exactly "is the identity map between two
restricted bases realisable by a fixed content map". The transplant obstruction
answers this for all `D` and all content: no. It says nothing about the
restriction to `[0, L]`, which is the only regime retention depends on. That
restriction is what this analysis measures, and it is measurable in closed form.

## 2. Measured quantities

Under isotropic content `E[(q^T A k - q^T B k)^2] = ||A - B||_F^2` exactly, so
these are logit-space errors, not proxies. Values are divided by
`d = ||R(D)||_F^2 = 128`, so **1.0 is the error of emitting no positional
signal at all**.

- `D0` — hard swap, no repair.
- `D*` — best fixed `(M, N)`. This is the obstruction theorem's own operator
  class and a **strict superset of any Q/K LoRA**: any rank, plus cross-pair
  mixing no practical adapter has. `D*` is therefore a hard ceiling on adapter
  repair, and the multi-start solver returns an upper bound on it, never an
  underestimate.
- Phase-safe fraction at a target length — the share of channels whose deployed
  phase arc stays inside the arc seen during training.

In-window weight is the exact causal pair count at `L = 4096`, 2048 support
points; `base = 500000`, `d_head = 128`, `K = 64`. Frozen tables are loaded from
`phase_adarope_20260822/assets/target_manifest.json` and never reconstructed.

## 3. Result

| table | `D0` | `D*` | safe@8K | safe@16K |
| --- | ---: | ---: | ---: | ---: |
| Native | 0.000 | 0.000 | 0.500 | 0.500 |
| anchored EVQ-Cosh tau 2 | 1.412 | **0.691** | 0.500 | 0.500 |
| phase-chord OLMo R0 | 1.437 | **0.698** | 0.500 | 0.500 |
| matched exponential control | 1.087 | 0.525 | 0.875 | 0.750 |
| PI s=4 | 1.098 | 0.530 | 1.000 | 1.000 |
| official YaRN s=4 | 0.625 | **0.290** | 1.000 | 1.000 |
| uniqueness-budgeted s=4, p=2 | 0.496 | **0.224** | 1.000 | 1.000 |

Three facts fall out.

**(a) The ordering reproduces every completed retrofit outcome.** YaRN has the
lowest unrepairable in-window damage of the published operators and is the only
one with a zero-training positive result in this repository (RULER-13 macro
`52.19%` at 8K, `OLMO2_1B_FRESH_GENERAL_QK_EVQ_YARN_20260727.md`). EVQ-Cosh has
2.4x YaRN's `D*` and is the one whose retrofit collapsed. PI sits between them
and is known to be worse than YaRN. The metric was not tuned to reproduce this.

**(b) Two of the three prepared ADaRoPE positional arms are strictly
dominated.** Endpoint preservation is enforced at
`olmo2_phase_adarope_5090/phase_adarope.py:143`, so anchored EVQ-Cosh and
phase-chord keep the Native slow endpoint: their phase-safe fraction at 8K and
16K is `0.500`, identical to Native. They pay `D* = 0.69/0.70` — 2.4x to 2.8x
YaRN's unrepairable damage, and more than two thirds of the way to having no
positional signal at all — and buy nothing on the extrapolation axis. The 16K
Pareto frontier contains only `native` and the budgeted table. `matched
exponential control` is a partial stretch and is dominated by YaRN s=4.

**(c) The unrepairable residual sits where the theory says it must.** Native
in-window uniqueness has a sharp cliff: **pairs 0-21 are resolvable, pairs 22-63
(42 of 64) have uniqueness below 0.01** — the per-channel form of low-frequency
collapse, and the free displacement budget. Residual attribution:

| table | pairs carrying most residual | position relative to the cliff |
| --- | --- | --- |
| anchored EVQ-Cosh | 4-9 | deep inside the resolvable block |
| phase-chord | 4-9 | deep inside the resolvable block |
| official YaRN s=4 | 17-22 | at the cliff, transitioning slightly early |
| budgeted s=4 p=2 | 21-26 | at the cliff, on the redundant side |

## 4. What this says about LoRA capacity

A rank-`r` LoRA on `q_proj` induces `rank(M - I) <= r`, and that budget is
**shared by all 16 heads in the layer**. The repository's standard arm is
rank-64 on Q/K (`8,388,608` parameters), which is **4 ranks per head**.

Fraction of the achievable full-rank repair captured at per-head rank `r`:

| table | r=4 | r=8 | r=16 | r=32 | r=64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| anchored EVQ-Cosh | **0.092** | 0.178 | 0.352 | 0.698 | 1.000 |
| phase-chord | 0.091 | 0.177 | 0.346 | 0.684 | 1.000 |
| official YaRN s=4 | 0.203 | 0.393 | 0.763 | 0.999 | 1.000 |
| budgeted s=4 p=2 | **0.251** | 0.482 | **0.890** | 0.999 | 1.000 |

The completed EVQ retrofit arms therefore ran at roughly **9% of the repair
their own operator class permits**, and even at unlimited rank their ceiling
leaves `D* = 0.691`. Two independent things were wrong at once:

1. **the table** — no rank closes a 0.691 floor;
2. **the adapter structure** — a global low-rank update cannot give 16 heads
   independent re-bases; per-head rank 16 (a block-diagonal per-head map, the
   *same* 8.39M parameter budget rearranged) reaches 0.89 of the ceiling on a
   transport-shaped table.

This is the quantitative form of the user's intuition that a LoRA should be able
to carry the repair. It can — on a table whose ceiling is low and with rank
allocated per head. It cannot on an endpoint-preserving interior reallocation.

## 5. Derived candidate

`budgeted_s4_p2` interpolates each pair toward `w/4` by
`(1 - normalised uniqueness)^2`, so resolvable pairs stay put and redundant
pairs absorb the range extension. Official YaRN's wavelength ramp is the crude
binary special case. Against YaRN s=4 at identical phase safety it gives
`D* = 0.224` versus `0.290` (**-23%**), needs rank 33 versus 39 for 90% of the
map energy, and captures 0.890 versus 0.763 of its ceiling at per-head rank 16.
Emitted as float32 with hashes under `out/derived_tables/`.

**Identity caveat:** the budgeted table depends on the uniqueness measurement,
which depends on the support-point count. `budgeted_s2_p1` moved from
`D0 = 0.731` at 512 points to `0.598` at 2048. Any table promoted to a GPU run
must carry its support count, weight family, and float32 hash as part of its
identity.

## 6. Claim boundary

- Isotropic content is worst-case-symmetric. A trained checkpoint's `q, k` lie
  on a lower-dimensional manifold, on which the true repair can only be
  **better** than reported. The refinement is one forward pass collecting
  post-projection `q, k` and reweighting the same expectations. Not implemented,
  not authorised, no GPU used.
- The causal distance weight is content-free. A measured attention-distance
  profile is supported through the `empirical` family and has not been measured.
- `D*` is not a task metric. It bounds what an adapter can restore. It does not
  predict RULER, 2Wiki, or NLL, and the agreement with four completed outcomes
  in section 3(a) is ordinal agreement on four points, not a validated predictor.
- No claim here supersedes any canonical owner, and nothing in this note is
  manuscript evidence.

## 7. Consequence for the next experiment

The registered ADaRoPE tournament (`PHASE_ADAROPE_LORA_PREFLIGHT_20260822.md`)
should not spend GPU on `phase_chord` as a mature-model retrofit direction, and
has no arm at all for the operator family that owns this repository's only
zero-training positive result. The smallest change that makes it decisive is to
lift the endpoint-preservation constraint for the retrofit setting, add an
official-YaRN arm as the practical baseline, and put the budgeted table in as
the candidate — leaving the per-head `alpha`, the budget-aware temperature, the
Stage-0 capability gate, and every receipt contract exactly as built.
