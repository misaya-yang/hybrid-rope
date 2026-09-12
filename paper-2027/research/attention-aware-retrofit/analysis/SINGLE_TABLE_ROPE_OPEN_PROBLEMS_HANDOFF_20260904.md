# Single-table RoPE: current evidence, open problems, and executable handoff

- **Date:** 2026-09-04
- **Status:** research handoff; repository evidence audited; new code locally
  checked but not run on the work machine
- **Scope:** one global request-static RoPE table and one fixed gain; no routing,
  dual table, boundary switch, cache handoff, or head-specific clock
- **Author target:** (Z) zero model-weight training, about 0.12 maximum Native
  damage on both NLL and downstream tasks, then maximize extrapolation; (F)
  small-data/few-step LoRA exposed only to physical 2x/4x, with useful unseen
  8x/16x/32x behavior and the same Native-retention requirement
- **Machine state:** the author shut the GPU server down. No run from the new
  program was started.

This document is an index and research brief, not a new result owner. Existing
patterns and external-model arguments are treated as hypotheses until they
match raw/hash-backed owners.

> **2026-09-04 execution correction.** This brief is historical context for the
> earlier prototype, not tomorrow's execution order. Use the
> [independent first-principles owner](../theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md)
> and [redesigned protocol](../preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md).
> The author explicitly rejected assuming the old training/evaluation chain was
> correct. The replacement independently tests source twins, local oracle,
> deleted-source control, complete output and EOS before costly experiments.

## 1. Bottom line

The original problem was previously misstated as exact Native preservation plus
universal long-context dominance. That stronger statement is irrelevant here.
The requested problem is a constrained Pareto problem and is not known to be
impossible.

Two genuine gaps remain:

1. **Zero-training ceiling:** the repository establishes useful one-table 2x/4x
   operation and a failed direct s8 candidate, but it does not explain or prove
   the maximum feasible factor under the Native-damage budget. The failure of
   one s8 table/gain is not a global upper bound.
2. **LoRA capability conversion:** this was already unresolved under EVQ-Cosh.
   Long NLL, routing, or source dependence can improve while top-1 generation
   and downstream tasks remain weak. The recent log-p2 Q/K-only run repeats the
   same separation. More rank or more steps on the same loss is not a solution.

The most defensible new theoretical statement found in this pass is a
**conditional phase-cost/dilution ceiling**, not a universal 8x theorem. The
most useful new experiment is physical 2x/4x QKVO LoRA with an identifiable
same-target source counterfactual, not another generic CE continuation.

## 2. Exact requirement and metric convention

The deployment object is fixed for the entire request and for every tested
length:

```text
one frequency tensor + one attention gain + one model weight state
```

There is no requirement of bitwise Native equivalence. Native damage must be
measured separately for:

- PG-19 tail NLL/PPL retention;
- downstream/generated-task retention.

The historical repository gate `retention >= 0.875` permits damage up to 0.125.
Because the author described the budget as “0.12 左右,” every future table must
also report the literal `retention >= 0.88` verdict. Values in `[0.875,0.88)`
are **marginal**. Do not average NLL and downstream retention to pass a joint
gate.

For NLL,

\[
R_{\rm PPL}=\frac{{\rm PPL}_{N}}{{\rm PPL}_{C}}
=\exp[-(\operatorname{NLL}_{C}-\operatorname{NLL}_{N})].
\]

Thus retention `0.88/0.875` corresponds to NLL increases at most
`0.127833/0.133531` nats per token.

## 3. What is already supported

### 3.1 Zero-training one-table evidence

The current strongest tracked OLMo arm is full legacy-u p2 installed by the
log-frequency law at s4 with fixed gain coefficient `.074`:

\[
\omega'_k=\omega_k4^{-m_k},\qquad
g=1+0.074\log4.
\]

It uses one table/gain at every length, zero routing, and zero model updates.
The owner reports:

| Endpoint | Candidate | Native | Retention or score |
| --- | ---: | ---: | ---: |
| 1x PG-19 tail NLL | 3.104234 | 2.971047 | retention 0.875302 |
| 1x five-task macro | 0.315833 | 0.345134 | retention 0.915103 |
| 2x PG-19 NLL | 3.083278 | 7.100855 | candidate lower |
| 4x PG-19 NLL | 3.081946 | 7.205538 | candidate lower |
| 2x six-task macro | 0.307614 | 0.058883 | candidate higher |
| 4x six-task macro | 0.260055 | 0.021517 | candidate higher |

The same owner reports RULER-13 `0.71397/0.66705/0.49859` at
4K/8K/16K, compared with Native `0.71308/0/0.00385` and repository YaRN-4
`0.43141/0.24308/0.10564`.

This supports a strong one-table 4x operating point. Its PG-19 retention is
marginal under the literal 0.88 threshold and passes the historical 0.875 gate.

The scale-consistent log law is formulaically valid, but direct s8 did not
become a robust method. The old s8 log arm used gain coefficient `.10`, scored
core-4 `0.5925/0.4575/0.4625/0.3025` at 1x/2x/4x/8x, and its full-task run
stopped after single-key-3 collapse. This closes only that candidate/protocol.
It does not establish that every static table must fail at 8x.

Primary owner:
[`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831`](../results/coupling-transfer/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md).

### 3.2 LoRA evidence

There are three distinct observations; do not pool them:

1. On Llama-3-8B-Instruct, 300-step rank-64 QKVO EVQ-LoRA improves temporal
   NLL at 16K/32K and materially increases causal remote-source use, while true
   16K exact generation remains zero and aggregate QA is worse at/below the
   adaptation length. This proves signal/routing effects, not capability
   conversion.
2. On OLMo-2 1.485B, a matched historical phase continuation recovers strong
   4K task behavior and clear EVQ transfer at 8K, with nonzero but weak 16K
   capability. It uses a Stage-A QKVO parent plus 300-step Q/K continuation and
   task-family phase exposure. It is not a fresh small LoRA on the current
   log-p2 substrate.
3. The recent exact log-p2/c=.074 rank-8 Q/K-only 96-step run improves paired
   PG-19 at 1x/4x, but five natural tasks and fresh core-4 do not improve. It is
   a valid negative for that objective/scope, not for all LoRA.

Owners:

- [`EVQ_8B_ADAPTATION_EVIDENCE_20260724`](../../../../rebuttal/rebuttal_0723/theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md)
- [`OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729`](../../../../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md)
- [`LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904`](../results/adaptation-coadaptation/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md)

The stable conclusion is that the missing step is **conversion of remote
positional signal into reliable generated output under a retention constraint**.
Likelihood alone is not the endpoint.

## 4. Why the zero-training ceiling is still unresolved

### 4.1 Superseded mixed-construction fit — not a same-path prediction

**Correction 2026-09-04:** the cited s2 receipt uses compressed G(x)/C2,
whereas the retained s4 tensor is full legacy-u p2. Their movement identity is
not established. The fitted 3.91/4.01 values below must NOT select a factor,
stop a run, or enter theory as a p2 ceiling. They are preserved to make the
stale inference searchable. See the replacement first-principles owner.


Using the already measured Native, s2, and s4 PG-19 points for the p2/c=.074
path, fit

\[
\Delta\operatorname{NLL}(t)=at+bt^2,\qquad t=\log s,
\]

through `t=0,log2,log4`. The fitted coefficients are
`a=-0.0494925`, `b=0.105004`. It predicts the NLL retention boundary near
factor `3.91` for 0.88 and `4.01` for 0.875. This explains why blindly pushing
the same p2/c=.074 path beyond 4 is unlikely to work.

It is post-outcome, one-dimensional, and based on two nonzero points. It is not
a global RoPE or checkpoint upper bound. Another direction or gain may have a
different curve.

### 4.2 Conditional Native-sensitivity identity

Let

\[
u=(\Delta x_1,\ldots,\Delta x_K,\Delta\log g),
\qquad x_k=-\log\omega_k,
\]

and let `t=log S`. If all of the following are fixed independently:

- an exact linear extension functional `a^T u=t`;
- a Native risk with zero gradient at `u=0`;
- a positive-definite local curvature `H`;
- no order, endpoint, or box constraint active;

then the quadratic surrogate problem

\[
\min_u \frac12u^THu\quad\text{s.t.}\quad a^Tu=t
\]

has the exact solution

\[
u_t^*=\frac{tH^{-1}a}{a^TH^{-1}a},\qquad
Q_{\min}(t)=\frac{t^2}{2a^TH^{-1}a}.
\]

This identity is mathematically correct but is only a surrogate result. It is
not yet a lower bound on actual finite-factor NLL, downstream damage, or usable
context. It becomes tautological if `a`, `H`, the damping, or coordinates are
chosen after seeing the preferred table.

To promote it to a radius-limited lower bound one needs a verified remainder,

\[
D_N(u)-D_N(0)\ge \frac12u^THu-C\lVert u\rVert^3,
\]

through the full candidate displacement. A Hessian at Native alone is
insufficient at `log 8 = 2.079`. Native is also not guaranteed stationary with
respect to a post-hoc RoPE/gain coordinate, so the linear term must first be
measured. A singular Native Fisher can contain directions invisible at Native
length but catastrophic at long length; arbitrary ridge damping would then
manufacture the claimed capacity.

NLL curvature cannot certify downstream retention without an additional link
between risks. Therefore the safe name is **Native sensitivity metric**, not
“context bound.”

### 4.3 Unresolved conditional two-ceiling construction

**Correction 2026-09-04:** the phase-cost lower-bound step below is not certified:
actual displacements were not shown to belong to the box/isotonic surrogate,
and coordinatewise clipping need not preserve a quadratic-form lower bound.
Native stationarity and finite-region lower curvature also lack validation.
Only the explicitly assumed log-sum-exp dilution identity is retained; this
section does not establish a behavioural ceiling. See the replacement owner.


A more informative working derivation separates phase/readout cost from
softmax dilution:

\[
S_{\max}\le \min(S_{\rm phase},S_{\rm dilution}).
\]

For a frozen query/key pair, one table/gain produces

\[
\ell(\Delta)=\sum_k\gamma_k
\left[a_k\cos(\omega_ke^{-d_k}\Delta)
+b_k\sin(\omega_ke^{-d_k}\Delta)\right],
\]

where `d_k` is log-frequency movement, `gamma_k` is gain, and `a_k,b_k`
encode the checkpoint's Q/K coupling. Gain changes amplitude but cannot repair
phase.

Assume each Native metric `j` has a verified lower curvature through the full
trust region. After optimistically eliminating gain by a Schur complement,
let `Hbar_j` be the remaining frequency-movement curvature. Let `w_k` be the
pre-registered frozen readout mass of pair `k`, and assume supporting log scale
`T` requires an `eta` fraction of that mass to move at least `T-b`, where `b`
is a declared phase tolerance. Define

\[
\kappa_{\eta,j}=\min_{z}
z^T\bar H_jz
\quad\text{s.t.}\quad
0\le z_1\le\cdots\le z_K\le1,\;w^Tz\ge\eta.
\]

Then, under those assumptions,

\[
D_j\ge\frac12(T-b)^2\kappa_{\eta,j},
\qquad
S_{\rm phase}\le
\exp\left[b+\min_j\sqrt{\frac{2\varepsilon_j}{\kappa_{\eta,j}}}\right].
\]

This is nontrivial only if `w,eta,b,Hbar` are frozen independently and
`kappa>0`. It is conditional, not evidence that the value is 8x.

Separately, if every retention-feasible table/gain bounds the relevant logit by
`B`, distractor exponential moment is at least `mu`, and effective competitors
grow at least as `n0*S`, then target attention mass obeys

\[
A_S\le[1+n_0S\mu e^{-B}]^{-1}.
\]

Requiring `A_S>=rho` gives

\[
S_{\rm dilution}\le
\frac{1-\rho}{\rho n_0\mu}e^B.
\]

This explains why a bounded static gain cannot compensate an indefinitely
growing denominator. The bound may be loose, and its calibration assumptions
must be checked beyond 4x. It supplies a mechanism and falsifiable prediction,
not a free theorem.

### 4.4 What would make the ceiling claim publishable

A reviewer-safe claim needs:

1. independent definitions of the extension constraint and Native measure;
2. measured gradient, spectrum/nullspace, damping sensitivity, and finite-
   displacement Taylor residual;
3. separate NLL and downstream budgets;
4. separation-only, dilution-only, and full-physical assays;
5. a positive control that solves the long task and a position-destroyed
   negative control;
6. a prospective prediction of the first failing factor, tested only after the
   prediction is frozen;
7. at least one second architecture/checkpoint for any claim beyond OLMo.

Without these, report only “largest factor tested that met the named gates.”

## 5. Why the LoRA problem remains hard

Training at 2x/4x cannot mathematically determine behavior at 8x/16x/32x.
Infinitely many response functions interpolate two scales. Transfer needs the
static table to already expose useful long carriers and the LoRA-induced
readout to vary smoothly across log distance.

The chain that can fail has four distinct links:

1. **Source selection:** the query must distinguish the correct remote span.
2. **Value transport:** V/O and the residual stream must carry that information.
3. **Top-1 readout:** the correct token must move from improved rank/NLL to the
   generated argmax, with terminal EOS behavior.
4. **Dilution reserve:** the relevant logit margin at 4x must exceed the extra
   denominator cost at unseen lengths. A rough stationary-statistics diagnostic
   subtracts `log(S/4)` from the 4x margin reserve: `log2`, `log4`, and `log8`
   are needed for 8x, 16x, and 32x respectively.

The exact per-query/head identity is

\[
G=\operatorname{LSE}_{R}(z)-\operatorname{LSE}_{D}(z)
=\operatorname{logit}(A_R).
\]

If the relevant numerator stays fixed and the distractor exponential sum is
literally replicated by factor `q`, then `G` falls by exactly `log q`. Use the
actual unmasked effective competitor ratio, not nominal sequence length. With
only 2x and 4x observations, a linear or power-law continuation is unidentified:
an added term proportional to `(log_2 S-1)(log_2 S-2)` preserves both observed
points and changes every longer prediction. Therefore the `log q` continuation
is a prospective conditional prediction, not a fitted extrapolation law.

The old EVQ results show that links 1 or probability-level source dependence can
improve without link 3. Q/K-only adaptation changes routing but may lack enough
V/O/readout capacity. Generic CE can reduce continuation NLL while sending weak
credit to the actual source span.

The new candidate therefore uses:

- standard QKVO PEFT LoRA, rank 64 / alpha 128;
- only the already identifiable natural correct/deranged pair views;
- physical 8K and 16K training, plus physical 4K replay;
- a `2x,4x,2x,4x,1x` 300-step schedule;
- answer-plus-EOS CE;
- an output-level same-target source counterfactual margin.

For each correct variant, the counterfactual keeps its answer teacher-forcing
tokens fixed but swaps only to the other variant's remote source content. The
loss requires the gold answer log-probability to be higher under the correct
source than under the corrupted source. This directly trains source dependence
while retaining Flash attention; it does not materialize a quadratic attention
matrix.

This is still only a candidate. It may fail because 300 steps are insufficient,
because task acquisition should precede long exposure, because the s8 table
destroys useful local carriers, or because output-level credit remains too
weak. A negative result closes this exact protocol only.

For a mechanism claim, compare frozen, Q/K, and Q/K/V/O arms with identical
data/order/token/optimizer budgets and actual trainable parameter counts matched
within 1%; same nominal LoRA rank is not parameter matching under GQA. If QKVO
beats same-budget QK only under oracle-routed answer/EOS evaluation, that is
evidence for downstream transport/readout capacity. A raw QKVO task win alone
does not identify V/O, because V/O changes later residual states and can alter
routing indirectly.

A stronger future data contract should make routing externally observable by
requiring a context-random evidence/route nonce followed by a context-random
answer nonce and immediate EOS. Counterfactual twins, target-deleted controls,
and position-moved twins must follow the current context rather than emit a
memorized value. Split documents, latent facts, all length renderings, bindings,
distractors, and nonces together; reject tokenized cross-split overlap after
allowlisted prompt boilerplate is removed.

## 6. Prepared experiment and code status

The full prospective protocol is
[`CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904`](../preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md).

Prepared files:

| File | Purpose | Current verification |
| --- | --- | --- |
| [`export_log_p2_factor_frontier.py`](../../../../scripts/analysis/export_log_p2_factor_frontier.py) | recover exact p2 movement from Native/s4 tensors and freeze factor/gain arms | local syntax + unit test only |
| [`train_log_p2_phase_transfer_lora.py`](../../../../scripts/train/train_log_p2_phase_transfer_lora.py) | physical 1x/2x/4x QKVO source-counterfactual LoRA | local syntax + schedule test only; no torch/runtime smoke |
| [`run_log_p2_frontier_and_transfer_4080.sh`](../../../../scripts/eval/run_log_p2_frontier_and_transfer_4080.sh) | separate build, retention, smoke, train, and far-eval stages | Bash syntax only |
| [`target_free_ruler_smoke.py`](../../../../scripts/eval/target_free_ruler_smoke.py) | adapter-aware physical RULER; now accepts 16x/32x | earlier adapter path has work-machine evidence; new lengths untested |
| `tests/test_export_log_p2_factor_frontier.py` | scale composition/export check | passed |
| `tests/test_log_p2_phase_transfer_lora.py` | frozen training schedule check | passed |

Local verification completed:

```text
Python syntax: pass
Bash syntax: pass
focused tests: 2/2 pass
git diff --check: pass
```

Not verified:

- work-machine paths and asset identities;
- table exporter against the exact remote Native/s4 tensors;
- trainer PEFT/backbone call path;
- route-explicit retrieval scoring and parameter-matched QK/QKVO controls; the
  current trainer is a prototype, not a scientifically complete LoRA screen;
- one-step BF16 Flash gradients and memory headroom;
- 32K/64K/128K RULER data generation and evaluation;
- any new NLL or task outcome.

## 7. Superseded prototype execution order (historical only)

Do not launch the entire matrix at once.

### Stage Z: zero-training boundary

1. Re-read `AGENTS.md`, `README.md`, `paper-2027/HANDOFF.md`, this handoff, and
   the two current result owners.
2. Inspect the dirty worktree and copy only the named new code/docs to the work
   machine. Do not overwrite unrelated remote changes.
3. Bind exact checkpoint, Native table, retained log-s4 table, token manifest,
   RULER root, and 4K/8K/16K identifiable-pair views. Run no-GPU preflight.
4. Freeze factor `4,5,6,7,8` and gain coefficient `.05,.074,.10` arms.
5. Use a small 1x PG-19 calibration subset to select one gain per factor.
6. Evaluate only those five arms on held-out 1x PG-19, five natural tasks, and
   core-4. Select the largest factor passing every separate 0.875 gate; mark
   strict-0.88 status separately.
7. Open 2x/4x/8x only for that selected factor.

Interpretation:

- no gain rescues 1x PG-19: p2 table movement sets the path-specific bound;
- PG-19 passes but generation fails: frozen readout/capability compatibility is
  the boundary;
- both Native gates pass but 8x fails: long geometry/dilution is the boundary.

This is an empirical p2-family diagnosis, not the global upper-bound proof.

### Stage F: low-data adaptation

1. Add and freeze a route-explicit counterfactual dataset/evaluator, or narrow
   the claim to joint answer+EOS without separately claiming retrieval. Add a
   QK-first arm and exact parameter-count receipt; do not interpret same-rank
   QK versus QKVO.
2. Generate/check held-out core-4 at 32K/64K/128K before training, but do not
   read outcomes until the adapter and all selection rules are frozen.
3. Run the exact-shape one-step s4/c=.074 QK smoke. Require finite loss,
   finite nonzero gradient, verified source-counterfactual differences, Flash-
   only execution, and at least 1 GiB headroom.
4. Run the 300-step s4 QK positive-control substrate. Stop early only on a declared
   systems/validity failure, not because an intermediate loss is disappointing.
5. Evaluate 1x PG-19, five natural tasks, and core-4 first. If either retention
   gate fails, stop that adapter before far lengths.
6. If retention passes, evaluate untouched 8x/16x/32x core-4 and the registered
   routing/dilution/transport/readout diagnostics together.
7. Open parameter-matched QKVO only if QK routing succeeds but oracle-routed
   answer/EOS fails. Otherwise QKVO cannot identify the bottleneck.
8. Run the s8/selected-gain adapter only if the s4 protocol is resolving. Run a
   matched Native-table adapter only if a non-Native adapter succeeds and causal
   attribution is needed.
9. Expand to RULER-13 and held-out natural QA only after non-floor core-4 at 8x
   and a nontrivial signal at 16x. A 32x floor does not erase a valid lower-
   length result.

Required controls for a paper-grade result are Native, uniform PI or official
YaRN under a clearly matched adaptation contract, candidate table with unit
gain, Native table with candidate gain, frozen candidate, and candidate LoRA.
The minimum projection screen is frozen/QK/QKVO; match actual QK and QKVO
parameter budgets before interpreting placement. Keep native-final and
8x/16x/32x blind sets sealed until the 2x/4x checkpoint is selected, then reveal
all three long lengths together. Do not pretend the smallest pilot supplies
this full factorial.

## 8. Historical prototype success and stop rules (use current preflight)

### Zero-training success

- one table/gain fixed across every request and length;
- 1x PG-19 and downstream retention each at least 0.875, with strict-0.88 status
  shown;
- useful held-out generated capability through 8x;
- matched Native and YaRN/PI comparators;
- raw/hash-backed per-row evidence.

### LoRA success

- training rows are exclusively physical 1x replay plus physical 2x/4x;
- 8x/16x/32x rows and outcomes never enter model or hyperparameter selection;
- 1x NLL and downstream gates pass separately;
- generated capability, not only NLL/rank/attention, improves at unseen length;
- the fully long-trained positive control resolves and the position-destroyed
  negative control fails;
- multi-seed and second-model replication precede a general method claim.

### Immediate stop

Stop on checkpoint/table/data/code hash drift, train/eval overlap, source-
counterfactual identity failure, attention fallback, OOM, non-finite loss or
gradient, adapter reload drift, scorer/prompt mismatch, or author stop. Preserve
partial raw outputs and record exclusions.

## 9. Important file index

Start here:

- [`AGENTS.md`](../../../../AGENTS.md) — repository/evidence rules
- [`README.md`](../../../../README.md) — paper identity and current direction
- [`paper-2027/HANDOFF.md`](../../../HANDOFF.md) — live Git/machine state
- [`INDEX.md`](../../../../index.md) — canonical claim routing

Current zero-training owners:

- [`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831`](../results/coupling-transfer/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md)
- [`SINGLE_STATIC_LOG_P2_SELECTION_AND_LORA_20260903`](../theory/SINGLE_STATIC_LOG_P2_SELECTION_AND_LORA_20260903.md)
- [`NATIVE_ISOTONIC_PROFILE_RESULT_20260903`](../results/zero-training-deployment/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md)
- [`K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901`](../results/coupling-transfer/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md)

Current LoRA/adaptation owners:

- [`LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904`](../results/adaptation-coadaptation/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md)
- [`OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729`](../../../../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md)
- [`EVQ_8B_ADAPTATION_EVIDENCE_20260724`](../../../../rebuttal/rebuttal_0723/theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md)
- [`FAR_PASS_CHORD_EXPERIMENT_REPORT_20260821`](../results/adaptation-coadaptation/FAR_PASS_CHORD_EXPERIMENT_REPORT_20260821.md)
- [`COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825`](../results/adaptation-coadaptation/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md)

Do not revive as selectors:

- scale-orbit boundary count, Gram/Ky-Fan, and best-`D_j` operator tightness;
- unrestricted slot permutations;
- another rank/alpha/gain/step sweep on the failed 96-step Q/K objective;
- routing, dual tables, or cache handoff as substitutes for the requested
  single-table problem.

## 10. Statements safe to hand to another researcher

**Observation:** one OLMo static log-p2 s4/c=.074 table approximately meets the
author's Native budget and strongly improves 2x/4x endpoints.

**Observation:** the tested direct s8 form fails robust all-family capability;
this is not a universal upper bound.

**Observation:** EVQ-Cosh and log-p2 adaptation can improve long likelihood or
source use without reliable generated-task conversion.

**Derived result under assumptions:** a Native-sensitivity quadratic has an
exact constrained optimum scaling as `(log S)^2`; with verified finite-radius
curvature and independently specified extension demand it yields a conditional
phase-cost bound.

**Working hypothesis:** the practical ceiling is the minimum of a phase/readout
compatibility limit and a softmax-dilution limit.

**Working hypothesis:** physical 2x/4x QKVO training with same-target source
counterfactual credit and 1x replay has a better chance of producing unseen-
length generation than the completed Q/K likelihood objective.

**Unresolved:** the global zero-training maximum factor, a non-tautological
closed-form bound tied to actual downstream damage, and a LoRA protocol that
retains Native behavior while transferring useful capability to 8x/16x/32x.
