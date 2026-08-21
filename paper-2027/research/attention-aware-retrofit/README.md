# Attention-aware phase allocation and mature-model retrofit agenda

- **Date:** 2026-08-21
- **Status:** internal research memo; not manuscript evidence
- **Purpose:** define the shortest route from the current spectral-budget result
  to (i) a fixed table that improves both in-window and extrapolation behavior,
  and (ii) a mature-model adaptation that preserves Native capability.

Executed results are summarized in
[`EXPERIMENT_REPORT_20260821.md`](EXPERIMENT_REPORT_20260821.md); the compact
machine-path-free receipt is indexed under [`evidence/`](evidence/). The
prepared but unexecuted mature-model finite-path audit is owned by
[`FUNCTION_MORPH_PREFLIGHT_20260821.md`](FUNCTION_MORPH_PREFLIGHT_20260821.md).
The current single-arm retrofit protocol is owned by
[`FAR_PASS_CHORD_RESIDUAL_PREFLIGHT_20260821.md`](FAR_PASS_CHORD_RESIDUAL_PREFLIGHT_20260821.md).
Its completed ablation chain and stop decision are now owned by
[`FAR_PASS_CHORD_EXPERIMENT_REPORT_20260821.md`](FAR_PASS_CHORD_EXPERIMENT_REPORT_20260821.md).

## 1. The two problems are different

The current evidence separates two costs that must not be conflated.

1. **Training-time allocation cost.** At fixed support, anchored EVQ-Cosh
   shifts the effective-context frontier but pays a small in-window cost. The
   151.9M three-seed owner reports `+0.026` NLL at 256 and gains at longer
   lengths. This is a table-design problem.
2. **Mature-model coordinate shock.** Replacing a pretrained model's table
   changes the coordinate system under weights already adapted to Native
   RoPE. The exact transplant theorem shows that a changed frequency multiset
   cannot be recovered exactly by static, position-independent Q/K maps. The
   seven-arm OLMo search and the partial Q/K-only recovery are manifestations
   of this second problem, not evidence that the first problem is impossible.

The method target is therefore also two-part: derive a better training-time
table, then retrofit it without deleting the Native attention path.

## 2. What today's result changes

Directly relabeling attention distance as frequency demand failed. With a
151.9M FMRoPE checkpoint, the same-direction map

\[
\Delta\longmapsto\phi,\qquad \rho(\phi)\propto
[(1-\lambda)m(\phi)+\lambda]^{1/3}
\]

improved the 256 endpoint slightly but degraded 512 and 1K. This falsifies the
idea that an attention-distance histogram alone is a frequency-demand oracle.

RoPE does not consume a distance bin directly. A pair at frequency \(\omega\)
responds to distance \(\Delta\) through its phase. The squared chord distance
between the two phases is

\[
\|R_\omega(\Delta)-I\|_F^2
=4\,[1-\cos(\omega\Delta)].
\]

This gives the first operator-aware demand proxy:

\[
m_{\rm chord}(\phi)
=\mathbb E_{\Delta\sim D_{\rm att}}
  [1-\cos(\omega(\phi)\Delta)],
\qquad
\rho_\lambda(\phi)
\propto[(1-\lambda)m_{\rm chord}(\phi)+\lambda]^{1/3}.
\]

It is bounded, uses the model's measured attention-distance distribution, and
passes that distribution through the actual rotary operator before allocating
channels. It is still a surrogate, not an identified LM-risk derivative.

For the seed-137 151.9M pilot, `lambda=0.1` has median normalized coordinate
`0.435`, between FMRoPE (`0.500`) and anchored EVQ-Cosh (`0.173`). The complete
500M-token run changes tail NLL relative to the matched FMRoPE checkpoint by
`-0.008/-0.237/-0.152/-0.215` at 256/512/1K/2K. Thus the pilot improves the
training window while retaining approximately `88%/80%/119%` of anchored
EVQ-Cosh's OOD gain. A schedule-frozen seed-42 confirmation subsequently gave
`+0.010/-0.084/-0.160/-0.195`: the OOD direction replicated and the in-window
cost remained small, but the 512 effect contracted. Across the two seeds, the
mean phase-minus-FMRoPE result is approximately `+0.001/-0.161/-0.156/-0.205`.
This is a promising Pareto shift, not yet a paper-facing method result.

The key scientific update is not merely a better curve. The full anchored
EVQ-Cosh in-window cost is not an unavoidable consequence of a finite
frequency budget. The result is consistent with demand-measure mismatch:
attention mass must first be transformed through rotary phase sensitivity.

## 3. Candidate retrofit: preserve a measured Native subspace

For one head, Native attention logits have the form

\[
\ell^N_{ij}=q_i^\top R_{\Omega_N}(i-j)k_j.
\]

A hard swap replaces \(R_{\Omega_N}\) everywhere. Q/K LoRA then tries to make
position-independent weight updates compensate for a position-dependent
change. The transplant theorem rules out exact global compensation when the
frequency multisets differ. Smooth table morphing changes optimization
dynamics but does not remove this obstruction. The seven-arm OLMo search and
partial Q/K-only recovery are consistent with this coordinate-shock account,
although the theorem alone does not explain finite-step retraining dynamics.

The repository already contains the correct first test in
`OLMO2_NATIVE_IMPORTANCE_PROTECTED_EVQ_HYPOTHESIS_20260728.md` and its
implemented Stage-D diagnostic. Reuse it instead of inventing a 50/50 split.
For every Native rotary pair it computes the exact leave-one-pair-out attention
KL and asks whether at most 16 of 64 pairs capture 80% of the aggregate score,
with split-half stability and per-layer coverage gates. No optimizer is needed.

Only if that diagnostic passes should the protected-table route continue:

1. retain the selected Native frequencies exactly;
2. conditionally re-quantize the complement using phase-chord demand rather
   than taking same-index anchored EVQ-Cosh values;
3. reject minimum-spacing violations and frequency-order reversals;
4. mask Q/K LoRA output rows so protected pair coordinates receive zero direct
   update;
5. freeze V/O, the backbone, and the protected frequency buffers;
6. train a matched full phase-chord restoration control before attributing a
   gain to protection.

The protected-set size is therefore measured, not fixed at one half. If Stage
D fails, pair-level protection is dead for this model; only then consider a
coarser layer/head frozen-swap diagnostic or move directly to the residual
route. A literal Native/new index splice is forbidden: the existing route audit
already found near-collisions and even frequency-order reversal in related
configurations.

For one head, decompose its Native logit into pair contributions

\[
\ell_{ij}=\sum_k s_{ij,k},\qquad
s_{ij,k}=q_{i,k}^{\top}R_{\omega_k}(i-j)k_{j,k}.
\]

The pair-level short-function score is not the Q/K norm by itself. Use the
interventional attention change

\[
I_{\ell h k}^{\rm short}
=\mathbb E_i\,\mathrm{KL}\!\left[
\operatorname{softmax}(\ell_i)\;\middle\|\;
\operatorname{softmax}(\ell_i-s_{i,k})\right].
\]

Q/K pair norm is retained only as a cheap descriptive cross-check. The Stage-D
implementation already owns the pair score. For a protected set
\(S\), add a frozen immediate-hybrid gate

\[
C(S)=\Delta\mathrm{NLL}_{4K}(S),\qquad
B(S)=-\Delta\mathrm{NLL}_{8K}(S),
\]

before creating an adapter. If the exact pair diagnostic fails, a separate
whole-head or layer swap may test coarser locality, but it is not part of the
first matrix.

## 4. Escalation route: an additive positional residual

Implementation update: the full-table rotation-difference sketch below is
historical motivation. The registered experiment uses the narrower
`I-R(Delta)` far-pass chord operator and fixed 8-pair no-wrap band in
`FAR_PASS_CHORD_RESIDUAL_PREFLIGHT_20260821.md`; that owner supersedes the
implementation details in this section.

If the measured protected-subspace route still loses too much Native
capability, keep the
entire Native term and learn an additive positional residual:

\[
\ell_{ij}=\ell^N_{ij}
+g_{\ell h}(i,j)\,
\big\langle A^q_{\ell h}h_i,
[R_{\Omega_*}(i-j)-R_{\Omega_N}(i-j)]
A^k_{\ell h}h_j\big\rangle.
\]

Here \(\Omega_*\) is the fixed phase-chord table; \(A^q,A^k\) are low-rank
trainable projections; and the residual gate is initialized to zero. This
operator has four important properties.

1. **Exact Native initialization.** At a zero gate the model is bitwise the
   original model, so retention is a construction property rather than an
   optimization hope.
2. **The theorem is respected.** The method does not claim that a static Q/K
   map restores a changed table. It changes the attention operator by adding a
   second phase route.
3. **Near-distance perturbations are naturally small.** The residual rotation
   difference is zero at \(\Delta=0\) and locally scales with
   \(\Delta(\omega_* - \omega_N)\). The phase-chord target is also much closer
  to Native than the previous anchored EVQ-Cosh target.
4. **Inference cost is controllable.** The residual can use 8--16 selected
   pairs and can remain disabled for queries within the Native context.

This is preferable to a full extra EVQ attention head. It preserves the
Native content/value route and adds only the positional correction that the
old coordinate system cannot express.

## 5. Selecting protected or residual pairs without another heuristic

The previous slow-only residual proposal guarantees weak variation in-window,
but channels that barely move also have weak positional resolution. The new
selection rule should price short-range damage and far-range benefit directly.

For each candidate Native-to-target pair replacement, compute

\[
C_k=\mathbb E_{\Delta\sim D_{\rm short}}
  \|R_{\omega_k^*}(\Delta)-R_{\omega_k^N}(\Delta)\|_F^2,
\]

\[
B_k=\mathbb E_{\Delta\sim D_{\rm far}}
  \|R_{\omega_k^*}(\Delta)-R_{\omega_k^N}(\Delta)\|_F^2.
\]

Select a fixed residual budget by the ratio `B_k / (C_k + epsilon)`, subject
to minimum log-frequency spacing. This replaces “protect the fastest 16” or
“use the slowest 16” with a measurable operator criterion. It also aligns with
the conditional-table audit: protected Native pairs and new residual pairs
must be optimized jointly enough to avoid near-duplicate frequencies.

The first numerical gate is free: compare the short/far Pareto curves of
Native-to-anchored-EVQ-Cosh and Native-to-phase-chord targets at residual
budgets 8, 12, and 16. If phase-chord does not reduce short cost at matched far
benefit, do not start the mature adaptation.

## 6. LoRA and training objective

**Executed update.** The Q/K-only, learned V/O transport, first-token-weighted,
and physical continuous-8K variants have now run. All improved at least one
teacher-forced probability or rank endpoint, but the final physical-8K arm
still obtained only `1/64` first-token top-1 and `0/64` strict answer-plus-EOS.
The CE-only route is stopped. The objective below remains historical design
context; any resumption must first add direct source-position alignment on the
independent natural rows, not another frequency/rank/gain sweep.

The minimum trainable scope is Q/K residual projections plus scalar or
per-layer gates. Native Q/K/V/O and the entire Native attention path remain
frozen. V/O freezing is important because the objective is to repair routing,
not relearn content transport.

Use two data regimes and three losses:

\[
\mathcal L=
\mathcal L_{\rm LM/task}
+\beta\,\mathrm{KL}(A_N\|A_S)_{\rm short}
+\gamma\,\|(A_S-A_N)V_N\|_2^2{}_{\rm short}.
\]

- **Short/retention batches:** the Native teacher supplies attention and
  attention-context targets. These terms are applied only to the added
  residual route's total output; the Native path itself is frozen.
- **Long/routing batches:** task or natural-LM loss teaches the residual when
  distant evidence matters. Include the existing remote-block intervention
  examples or an independently frozen natural-span routing set, but do not use
  RULER examples as the sole training source.
- **Gate curriculum:** keep the residual gate at zero for a short calibration
  phase, then increase only its allowed maximum. Do not morph the Native table.

Urrutia et al.'s positional/symbolic separation suggests a useful restriction:
enable the residual only in layers or heads with strong positional diagnostics,
while symbolic/content heads retain Native RoPE. Gu et al.'s deposit result
suggests monitoring whether one shallow head captures the entire residual, but
it does not justify adding a deposit regularizer before that phenomenon is
reproduced in this model.

Selective RoPE motivates an input- or context-dependent gate, but the first
implementation should use the simplest Flash-compatible query-length/per-layer
gate. A relative-distance gate is scientifically cleaner but should not force a
custom quadratic attention path.

## 7. Experimental ladder and stop rules

### R4-M0: mature finite function-morph audit

- Candidates: the OLMo R0 phase-chord table, a non-attention-aware
  endpoint/RMS-log-displacement-matched control, and anchored EVQ-Cosh.
- Path: finite log-frequency morphs at `t={0,.05,.25,.5,.75,1}`.
- Report: per-example Native-teacher forward KL and tail-token NLL delta at
  4K, 8K, and 16K; teacher-forced only.
- Stop if phase-chord does not dominate the matched control on the measured
  4K-cost/far-NLL tradeoff. This diagnostic does not gate the residual route.
- Current state: exact code, assets, targets, and no-GPU receipt are frozen,
  but the audit is deprioritized and is not a prerequisite for the practical
  LoRA route. GPU evaluation has not been authorized or run.

### R4-0: zero-training operator audit

- Targets: anchored EVQ-Cosh versus phase-chord.
- Budgets: 8/12/16 residual pairs.
- Report: short cost, far benefit, minimum frequency spacing, and selected
  pairs per layer.
- Stop if phase-chord does not dominate anchored EVQ-Cosh on short cost at matched far
  benefit.

### R4-1: 151.9M retrofit sanity check

- Start from a completed FMRoPE checkpoint; do not retrain the base.
- Compare hard Q/K LoRA table swap, the best measured protected/adaptable
  granularity, and Native-plus-phase residual.
- Use one seed and a short registered budget.
- Stop if the residual does not preserve the 256 endpoint better than the hard
  swap while improving at least one OOD endpoint.

### R4-2: mature OLMo pilot

- Model: `OLMo-2-0425-1B-Instruct`.
- One seed screen within a frozen Q/K-residual protocol; use an early stop
  rather than redefining the registered final budget.
- Apply the existing complete 4K gate against a freshly matched Native
  control: 2Wiki token-F1 drop at most 5 points, RULER macro drop at most 10
  points, natural-NLL increase at most 0.10, no Native-positive family collapse,
  strict autoregressive evaluation, and an independent retention pass.
- Do not run the 8K gate unless every 4K condition passes; at 8K require the
  candidate to beat Native on the registered strict autoregressive endpoint.

### R4-3: matched multi-seed result

Only after R4-2 passes, run three matched seeds for Native control and the
residual candidate. Report 4K natural NLL, 2Wiki exact/F1, all-family RULER,
remote-block deletion, and an independent instruction/capability retention
slice. Do not make 8B multi-seed the default.

## 8. Relationship to adjacent work

- **Gu et al.:** RoPE alters the full Q/K interaction multiplicatively. This
  supports designing the retrofit at the attention-logit layer, not claiming
  that a frequency table is an independent bias. The deposit-head mechanism
  remains a testable diagnostic, not an explanation already established here.
- **Urrutia et al.:** positional and symbolic behavior compete within a head.
  This motivates retaining Native symbolic routes and adding positional
  capacity selectively; it is not proof that their exclusivity bound equals
  this paper's waterbed surrogate.
- **Selective RoPE:** attention is input-dependent and rotation alone is not a
  complete memory mechanism. This motivates a gate/decay variable while
  keeping the phase-chord frequencies fixed and interpretable.
- **MrRoPE:** inference-time range transport is orthogonal to training-time
  allocation. A better substrate may help MrRoPE, but the interaction is
  untested and must not be claimed from the YaRN-style result.
- **LeRoPE:** learned tables reveal in-window preferences under joint
  optimization. The Native-plus-residual construction preserves that existing
  in-window coordinate system rather than forcing one static table to serve
  both roles.

## 9. Decision

The first objective is practical mature-model LoRA, not another whole-table
swap or another table-ranking experiment. Freeze the released Native model as
the main attention path, preserve its short-request route exactly, and add the
registered 8-pair `I-R(Delta)` far-pass chord Q/K residual through one softmax
with zero-padded residual values. Train the one residual arm on an independent
FineWeb-Edu natural-span retrieval view; RULER remains held-out evaluation.
Because the complete 8K+16K RULER evaluation costs only minutes under the
completed runtime receipt, it runs the full 20 rows per family rather than
adding a noisy subsample gate. It does not sweep rank, gain, frequency band, pair count, or
training length.

The finite whole-table morph audit remains available but is not a prerequisite.
Stage-D protection and conditional table construction are fallback diagnostics
only if the residual route fails for a localized, measurable reason. The
current phase-chord result is useful training-time evidence, not the blocker
for solving LoRA practicality.
