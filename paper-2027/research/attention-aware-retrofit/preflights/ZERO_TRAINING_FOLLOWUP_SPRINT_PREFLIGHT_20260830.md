# Zero-training success-first portfolio and confirmation protocol

- **Date:** 2026-08-30
- **State:** `SUCCESS_FIRST_PORTFOLIO_DESIGN_ONLY`; four candidate families are
  specified below, but no realised candidate tensor, runner certification, model
  result, or GPU authorization exists
- **Agenda owner:** repository `INDEX.md` §6
- **Purpose:** maximise the probability of finding a frozen-checkpoint table that
  improves Native-window and long-range behaviour within three to five
  scientifically distinct attempts, without repeatedly tuning on one confirmation
  set

## 1. Direct decision

The next research cycle is not a minimum-compute contest. Its objective is to
find the strongest credible zero-weight-update allocation, or to close the
declared shared-static-table programme after a serious four-family search.

The programme is therefore a **candidate tournament**, not one tiny candidate
followed by improvised retries:

1. four high-prior candidate families are developed on a common development
   split;
2. each surviving family freezes at most one representative before a separate
   selection split is opened; a family that fails its development gate
   nominates none;
3. the selection split chooses one global winner from the surviving families by
   a predeclared lexicographic rule;
4. the final confirmation split is opened once for that winner;
5. failure of the four-family portfolio triggers a pivot to heterogeneous
   allocation or matched LoRA, not a fifth renamed static-score sweep.

Compute cost is secondary to scientific success and clean confirmation. It is
still part of ROI: extra compute is justified when it increases the probability
of a decisive result, not when it repeats an answered causal question.

This document does not authorize inference, training, paid compute, server
access, or a manuscript claim. Every GPU stage still requires a live readiness
receipt and separate explicit author authorization.

## 2. What “zero training” means in this programme

All candidate families keep every model weight frozen. They do not train a new
model, continue pretraining, or update a LoRA adapter. Their exact construction
identity must nevertheless remain visible:

| Label | Allowed construction | Maximum outward description |
| --- | --- | --- |
| `ZERO_SEARCH_REFERENCE` | one table fixed before any candidate likelihood/capability outcome is read; a frozen checkpoint measurement may define its direction | zero-search, zero-weight-update table; not data-free unless stated |
| `FORWARD_CALIBRATED` | choose among a finite frozen table grid using disjoint forward-only calibration | forward-calibrated, zero-weight-update table; not zero-search |
| `GRADIENT_CALIBRATED_Z` | backpropagate through allocation coordinates while all model weights remain frozen | calibrated allocation with zero model-weight updates; not zero-learned-parameter |
| `JOINT_SUPPORT_ALLOCATION` | calibrate one request-independent support and allocation with gain/routing disabled | fixed support-allocation system; not a pure-`z` intervention |

The serving artefact is always one static table installed for the complete
request and KV-cache lifetime. It never reads the request's target length,
changes after prefill, routes between tables, or changes attention gain. A table
selected on model-relative `1x/2x/4x` development horizons is
**multiscale-calibrated**, not universally target-free. Only a construction that
never reads those outcomes may claim target-length-free selection.

The target verdicts are deliberately stronger than “something moved”:

- `JOINT_IMPROVEMENT`: Native-prefix and far-tail likelihood both improve on
  final confirmation, while long dense NLL passes its no-material-regression
  guard;
- `DEPLOYABLE_PARETO`: Native-prefix and long dense NLL are non-inferior and the
  far tail improves, but Native-prefix improvement is not established;
- `MECHANISM_ONLY`: a valid table moves one endpoint but fails the joint
  deployment contract;
- `FAIL` or `UNRESOLVED`: the registered direction fails or the assay does not
  resolve it.

No downstream score may convert `MECHANISM_ONLY`, `FAIL`, or an invalid run into
`JOINT_IMPROVEMENT`.

## 3. Why these four families

The repository already answers whether allocation exists and whether it can
matter. The new question is which checkpoint-aware low-dimensional construction
can satisfy the joint objective.

The portfolio incorporates every material completed constraint:

- the 151.9M exact-range and mature same-support owners already identify pure
  interior allocation;
- the 50M and 151.9M crossings show that table/weight compatibility dominates a
  static geometry score;
- two analytic Native-support Cosh tables and the 62-effective-degree,
  two-document direct-`z` calibration failed their Native/robustness gates;
- the 128-document dose study shows graded far-tail improvement can coexist with
  Native and long-middle costs, so two endpoint means are not enough;
- the mature learned direction is nearly neutral at `1x` and improves the far
  tail, but worsens long dense NLL and is an oracle rather than a zero-training
  construction;
- the phase-chord construction uses measured checkpoint attention mass and has
  two-seed from-training Pareto evidence, while its mature frozen morph family
  was prepared but never evaluated.

Consequently the four shots progress from the cleanest model-conditioned prior
to the strongest behavioural oracle, then relax support only once.

## 4. Four candidate families

### F1 — phase-chord morph family (`PC-MORPH`)

**Hypothesis.** The mature OLMo attention-mass phase-chord direction is useful,
but the complete target is too large a coordinate shock; an intermediate morph
contains a joint point.

Construction:

1. recover the canonical mature OLMo R0 collection: 1,024 separate Native-window
   FineWeb-Edu sequences, all layers and heads, with the existing owner hash;
2. form
   \[
   m_{\rm chord}(\phi)=\mathbb E_{\Delta\sim D_{\rm att}}
   [1-\cos(\omega(\phi)\Delta)]
   \]
   and the frozen `lambda=0.1` cube-root companding target;
3. preserve the Native endpoints and log-span exactly;
4. morph in log-frequency space from Native to the phase-chord target at
   \[
   t\in\{0,0.025,0.05,0.10,0.20,0.35,0.50\};
   \]
5. retain `t=0.05` as the historical zero-search reference; selecting another
   `t` on development data makes the winner `FORWARD_CALIBRATED`.

This family escapes the closed static-selector class because its direction is
conditioned on measured checkpoint attention behaviour. It still aggregates
layers, heads, and content into one demand profile, so it is a prior rather than
an LM-loss derivative or success theorem.

**Family stop:** no nonzero morph passes the development Native-prefix and long
dense guards while improving far-tail mean NLL.

### F2 — Native-retention projected phase chord (`PC-RETENTION-PROJECT`)

**Hypothesis.** The useful phase-chord direction has a component aligned with
Native loss sensitivity; removing that component yields a target-length-free
checkpoint-calibrated table with better retention.

Construction:

1. represent interior log-frequency displacements in a five-dimensional
   piecewise-linear hat basis with fixed zero displacement at both endpoints;
2. on development documents, use only Native-prefix next-token loss to measure
   the gradient and an empirical Gauss--Newton approximation with respect to
   those five coefficients;
3. project the F1 phase-chord direction into the registered Native-loss trust
   region, enforcing positive ordered gaps and exact Native support;
4. choose the largest projected step that passes the Native-prefix development
   guard, without reading middle or far-tail outcomes during construction;
5. freeze the realised float32 table and hash before evaluating its long-range
   endpoints.

All model weights remain frozen, but gradients through `z` make this
`GRADIENT_CALIBRATED_Z`, not a zero-learned-parameter construction. Because its
construction sees only Native-prefix behaviour, the long-range test remains a
genuine out-of-construction outcome. The local metric constrains one declared
direction; it is not revived as a standalone finite-table selector.

**Family stop:** unstable local metric, failure of order/support checks, no
nonzero feasible projected step, or wrong far-tail direction on the development
readout.

### F3 — five-degree behavioural allocation (`Z5-BEHAVIOUR`)

**Hypothesis.** A low-dimensional allocation selected on actual position-resolved
checkpoint loss can find the joint table that the 62-degree/two-document pilot
could not estimate.

Construction:

1. use seven fixed normalized pair-index knots, pin the two endpoints, and learn
   five ordered interior knot values; piecewise-linear interpolation yields all
   `K` coordinates;
2. parameterise positive knot gaps and normalise their cumulative sum, so every
   forward pass is an ordered same-support table;
3. minimize far-tail NLL on the development split subject to hard development
   constraints on Native-prefix and long dense NLL;
4. use a frozen optimizer/budget and four predeclared initialisations: Native,
   the F1 family winner, the completed learned-direction shape as a discovery
   teacher, and the completed coarse budgeted direction;
5. select one family representative on development data, then discard optimizer
   state and freeze only its float32 table and receipt.

The learned-direction initialisation is an internal discovery aid. It cannot
make the final table a zero-search construction or transfer its prior owner's
claims. This family is promotable only as a reproducible multiscale calibration
algorithm with an untouched selection and confirmation result.

**Family stop:** every initialization violates a hard guard, different
initializations produce incompatible tables with no selection stability, or the
selection candidate fails outside development.

### F4 — fixed support-allocation joint family (`SR-Z5`)

**Hypothesis.** If no Native-support allocation passes, fixed Native support is
the binding constraint; one request-independent support and a low-dimensional
allocation can satisfy the joint objective without gain or routing.

Construction:

1. extend F3 with one model-relative support factor
   \[
   s\in\{1,1.25,1.5,2,3,4\};
   \]
2. keep the fast endpoint, pair count, operator, gain `1`, and all weights fixed;
3. optimise the five-degree allocation inside each support under the same
   development constraints;
4. nominate one fixed `(s,z)` pair before the selection split;
5. if F4 wins, final confirmation must add a same-support geometric control and
   the selected normalized `z` embedded back into Native support. Those controls
   separate the complete method result from support and allocation attribution.

This is deliberately not pure `z`. It is the final shared-static-table shot
because support policy is already a strong lever and reverses the tested
allocation ordering. The serving table remains fixed and request-independent.

**Family stop:** no `(s,z)` passes the development joint guard, or the nominated
pair fails selection. Do not add gain, routing, or another support grid after
seeing the result.

## 5. Data firewall and tournament stages

Use one physical `4x` causal sequence per document. With a static table and
gain `1`, one forward pass yields Native-prefix, intermediate, long-dense, and
far-tail losses. A standalone `1x` versus `4x`-prefix parity smoke must pass a
predeclared numerical tolerance before this reuse is accepted.

The work-machine preflight must materialise four disjoint sources:

| Split | Rows | Permitted use |
| --- | ---: | --- |
| construction owner | existing R0 1,024 | F1/F2 direction only; never outcome selection |
| development `D` | 64 | within-family strengths, gradients, optimisation, and representative selection |
| family selection `S` | 64 | evaluate exactly one frozen representative per family and choose one global winner |
| final confirmation `T` | 128 | evaluate the global winner once; no tuning or replacement after opening |

`D`, `S`, and `T` must be source-hash disjoint from one another, the R0
collection, the direct-`z` pilot, the co-adaptive/learned-direction training and
evaluation documents, and the completed dose-response rows. If the local corpus
cannot satisfy this firewall, stop and acquire a new owner-backed shard rather
than silently reusing outcome-seen rows.

All arms within a split use identical token IDs, row order, precision, attention
backend, decoder, loss masks, and code. Document/source is the paired sampling
unit; tokens and position bins are not independent samples.

## 6. Endpoints, selection, and final verdict

For every document/table, write per-row numerators and denominators for:

1. `native_prefix_nll`: target positions `1..L_native-1`;
2. fixed 1,024-token position-bin NLL across the complete `4x` sequence;
3. `long_dense_nll`: every valid next-token target in the `4x` sequence;
4. `far_tail_nll`: the final 1,024 target tokens.

The development and selection rule is lexicographic, not a pooled score:

1. reject a candidate if Native-prefix mean NLL exceeds Native by more than
   `+0.01` or long-dense mean NLL exceeds Native by more than `+0.01`;
2. among feasible candidates, choose the lowest far-tail mean NLL;
3. if candidates differ by less than `0.01` far-tail NLL, prefer in order:
   fewer calibrated degrees of freedom, no long-range outcomes in construction,
   smaller exact operator-chord displacement, then smaller support movement.

The Native-prefix `+0.01` margin reuses the completed joint gate's practical
scale; applying the same margin to long-dense NLL is a new decision rule frozen
here before the new rows exist. Neither is an uncertainty interval or universal
constant.

On final confirmation, use paired-document bootstrap intervals and issue exactly
one verdict:

- `JOINT_IMPROVEMENT`: the upper 95% interval endpoint is below zero for both
  Native-prefix and far-tail NLL, and at most `+0.01` for long-dense NLL;
- `DEPLOYABLE_PARETO`: the upper endpoint is at most `+0.01` for Native-prefix
  and long-dense NLL and below zero for far-tail NLL, but Native-prefix does not
  meet the strict improvement rule;
- `MECHANISM_ONLY`: far-tail improves but either no-harm guard fails;
- `FAIL`: a primary direction is wrong;
- `UNRESOLVED`: directions are favourable but the frozen confirmation does not
  resolve one required interval, or a registered positive control fails.

Position bins are always emitted in the same forward pass. They are inspected
after the primary verdict to localise a crossover; they do not create another
GPU experiment, replace the long-dense guard, or identify a causal frequency
band.

## 7. Capability and checkpoint transfer

Only a `JOINT_IMPROVEMENT` or `DEPLOYABLE_PARETO` table enters capability work.
Freeze the candidate and run, without retuning:

1. the unsaturated RULER multikey-2, multikey-3, and variable-tracking tasks;
   single-key remains a descriptive positive control rather than a primary
   endpoint;
2. one natural full-200 2WikiMultiHopQA endpoint with token F1 and normalized
   exact reported separately from answer-token NLL;
3. source ablation when the natural endpoint improves, so generation success is
   connected to use of the remote evidence rather than answer priors.

Likelihood and capability remain separate. Capability may confirm a likelihood
winner but cannot rescue a failed confirmation gate.

After OLMo likelihood and at least one preregistered capability endpoint pass,
apply the **construction algorithm**, not the OLMo table, to Qwen2.5-1.5B.
Recreate model-relative R0/calibration inputs and freeze a new table before Qwen
test outcomes. A sign reversal is checkpoint-regime evidence; do not add a 7B
model to rescue it.

Matched LoRA follows only after the zero-weight-update result is stable. Its
first clean comparison is Native versus the selected same-support candidate
under identical Q/K-LoRA data, order, parameter count, optimiser, budget, gain,
and evaluation. Further from-scratch scaling remains closed.

## 8. Conditional diagnostics, not candidate shots

The matched-content virtual-gap bridge enters only when the global winner has a
valid but mechanism-ambiguous result and the answer changes the next method:

- rewrite the old four-cell preflight around the actual winning candidate;
- start with table-by-gap only; gain is a later third factor;
- use an exact duplicate determinism control;
- treat global-offset invariance with a preregistered floating-point tolerance,
  not a BF16/Flash bitwise requirement;
- remember that a virtual gap changes cross-block phase without adding real
  distractor tokens or a long softmax denominator.

The current leave-one-band-out design in
`PROTECTED_RAMP_RIGOROUS_COMPOSITE_20260828.md` §9 is **invalid as written**.
Restoring B0/B0°/B1/B2 abruptly to Native breaks strict frequency ordering by
creating crossings at the restoration boundary; B0 also changes sampled support, and B4
is a small nonzero intervention rather than an exact sham. It must not be
executed. Any future attribution requires a new monotonicity-checked cumulative
or smoothly projected intervention and a separate support arm.

Classic PI/NTK rows, a support-by-allocation factorial, grouped allocation, and
per-head attribution are follow-ups only when they change the interpretation or
deployment of a confirmed winner.

## 9. Theory lane

The current paper's support-allocation identification, spectral-budget identity,
slow-collapse theorem, Cosh surrogate optimum, and transplant obstruction stand.
New theory does not select a portfolio winner and does not block the tournament.

The useful local order is:

1. formalise the exact quadratic regret identity and finite-`K` theorem for the
   equal-mass quantile **histogram**; the existing CPU audit is numerical support,
   not proof or an atomic-table LM bound;
2. if pursued, state nested collapse only as a slow-tail cumulative bridge with
   a uniform remainder and declared measure; it does not derive the complete
   `min` kernel from exact geometry;
3. use the exact chord identity/local quadratic bound only to report candidate
   shock and define a trust region, never to rank finite tables;
4. keep exact-retention claims at the single-table linear-operator level and do
   not infer universal routing necessity;
5. the historical O6 linearised rank note now uses the square root of retained
   singular-value energy for first-order descent amplitude; Q/K joint-budget
   claims remain open and must not prioritise a LoRA run.

## 10. Work-machine readiness and code boundary

Before requesting any GPU stage, the work machine must:

1. recover and hash-verify the R0 collection and historical function-morph
   target manifest;
2. locate the completed dose-run per-row position bins by their owner hash before
   deciding whether any historical analysis needs recomputation;
3. materialise every family grid/table, validate strict order, endpoints,
   support, pair count, float32 hash, and construction-data firewall;
4. adapt `scripts/eval/eval_allocation_dose_grid.py` to accept a generic candidate
   manifest, emit Native-prefix/long-dense/far-tail/bin denominators, record
   checkpoint/config/code/data hashes, and support a no-GPU contract mode;
5. run canonical `aidemo` no-GPU/contract tests;
6. freeze new output paths, disk/shutdown plan, and the exact development,
   selection, confirmation, and capability manifests.

After the author separately authorizes the stage, run one real row as a
parity/backend smoke before the remaining stage rows. A smoke authorization does
not authorize development, selection, confirmation, or capability as a bundle.

Stop before GPU on missing raw owners, data overlap, non-monotone tables,
support/gain/routing drift, missing hashes, backend fallback, or an unfrozen
selection rule.

## 11. Programme stop and success rules

- Do not open `T` until all four family development verdicts, every surviving
  representative, the selection result, and the selection rule are frozen.
- Do not replace the global winner after any `T` output is visible.
- Do not run a fifth shared-table shape on `T` after failure.
- If the four-family portfolio fails, record the bounded conclusion: under the
  declared checkpoint, data, static-operator, and calibration families, no
  confirmed joint table was found. Then pivot to grouped/per-layer allocation
  or matched LoRA with a new estimand.
- If it succeeds, write the result owner before expanding baselines, mechanisms,
  checkpoints, or adaptation.
- No plan, code path, candidate tensor, smoke, or partial output is a result.

This protocol is the sole active owner for the zero-training candidate
programme. The supplied desktop sprint, old function-morph runner, protected
ramp plans, and matched-content design remain inputs or conditional protocols;
none independently owns the execution order.
