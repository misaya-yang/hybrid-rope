# Local functional compatibility and exact Q/K--frequency gauge audit

- **Date:** 2026-09-03
- **Status:** `COMPLETE THEORY / POST-RUN VALIDITY AUDIT / NO NEW EXPERIMENT`
- **Evidence labels:** the calculus and gauge identities below are **Derived
  results**; the reading of the frozen calibration code is an **Implementation
  audit**; implications for the executed Selective-31 result are an
  **Interpretation** at that owner's exact scope.
- **Questions:** What did the Selective-31 calibration score measure? When can a
  compatibility derivative predict endpoint damage? Does exact joint
  frequency--Q/K relabeling invalidate the permutation or head-selective
  results?
- **Inputs:** the hash-bound Selective-31 runner and calibration identities,
  the current OLMo2 split-half implementation, and the two existing result
  owners linked below.
- **Compute:** source inspection and algebra only. No training, GPU inference,
  new result row, or behavioural selection was performed.
- **Owners affected:** this note narrows mechanism language in
  [`HEAD_SELECTIVE_ZERO_TRAINING_SIX_ARM_RESULT_20260903.md`](../results/zero-training-deployment/HEAD_SELECTIVE_ZERO_TRAINING_SIX_ARM_RESULT_20260903.md).
  It does not supersede that negative result or
  [`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`](../results/coupling-transfer/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md).

## 1. Decision

Three corrections are required.

1. The frozen calibration score was an exact relative Frobenius displacement
   between Native and endpoint-perturbed attention maps, computed from Native
   hidden states on twelve 128-token HotpotQA prefixes. It was not a loss
   derivative, not `chi_func`, and not a task-free compatibility metric.
2. A functional sensitivity is local and conditional on a checkpoint,
   distribution, loss, and intervention path. Its signed mean predicts the
   initial average direction; its root-mean-square magnitude does not predict
   whether the endpoint improves or degrades.
3. Jointly permuting logical Q/K rotary coordinates and the corresponding
   frequency entries is a gauge identity. The existing same-multiset
   permutation experiments deliberately permuted frequencies while holding
   Q/K fixed, so their fixed-readout non-exchangeability estimand is different
   and remains valid. The executed head-selective intervention also held Q/K
   fixed; the gauge identity is an engineering control, not its comparator.

The audit therefore does **not** turn either result owner invalid. It removes a
mechanistic interpretation that those owners do not need: Selective-31 tests a
fixed attention-change-derived mask, not the causal value of a universally
defined sensitivity.

## 2. Local functional objects

For an intervention path `I_alpha` with `I_0` Native and endpoint `I_1`, let

\[
F_z(\alpha)=\ell(M_{I_\alpha};z),\qquad
d_z=F_z'(0).
\]

The two basic summaries are

\[
\mu_{\rm func}=\mathbb E_z[d_z],\qquad
\chi_{\rm func}=\sqrt{\mathbb E_z[d_z^2]}.
\]

`mu_func` is the signed first-order change in the declared mean loss.
`chi_func` is a magnitude/heterogeneity summary. Squaring removes direction, so
`chi_func` alone cannot predict damage rather than benefit. Both are local:

\[
F_z(1)-F_z(0)=F_z'(0)+\int_0^1(1-\alpha)F_z''(\alpha)\,d\alpha.
\]

Endpoint prediction consequently requires a separately checked Taylor
remainder. Adjacency of a slot swap does not establish locality; realized
score change and endpoint curvature do.

The clean implementation is one differentiable scalar `alpha` through the
complete model. For a simultaneous multi-layer path, the baseline differential
obeys

\[
\left.\frac{d\ell}{d\alpha}\right|_0
=\sum_\ell
\left\langle
\left.\frac{\partial\ell}{\partial s_\ell}\right|_0,
\left.\frac{d s_\ell}{d\alpha}\right|_0
\right\rangle.
\]

Thus summing correctly evaluated baseline partial derivatives is valid by
linearity. What is invalid is summing finite-endpoint full-run differences, or
mixing layer deltas evaluated on already intervened upstream states. Detaching
each layer input yields a useful direct local diagnostic, but it is not the
full-model functional derivative.

For one attention row with baseline scores `s`, endpoint perturbation `delta`,
and `a=softmax(s)`, the exact attention divergence is

\[
\operatorname{KL}(a\|\operatorname{softmax}(s+\delta))
=\log\mathbb E_a e^\delta-\mathbb E_a\delta
\le \frac{(\max\delta-\min\delta)^2}{8}.
\]

This gives a valid scale diagnostic, not a universal numerical threshold. Any
cutoff on row oscillation, KL, or Taylor remainder is an operational
preregistration choice.

Natural-text LM loss and compact-task answer/margin losses therefore define
different objects. A `chi_native` or `chi_task` can be useful under its frozen
distribution and objective; neither is a checkpoint-only solution for the
movement profile `m`.

## 3. What Selective-31 actually measured

The recovered calibration source computes, for each layer/head, Native and
mid-band-divided-by-four attention matrices while holding that layer's Q/K
inputs at their Native values, then records

\[
d_{\rm attn}
=\frac{\|A_{s4}-A_{\rm Native}\|_F}{\|A_{\rm Native}\|_F}.
\]

The median over twelve fixed prefixes ranks the 256 layer/head identities;
Selective-31 takes the lowest 31. This is an exact endpoint attention-map
displacement for one frozen perturbation and calibration pack. It is not a
gradient of LM loss, a downstream utility score, or an identified mediator.

Layer matching removes a large architectural confound, but Reverse-31 is not
matched on baseline evidence utility, output coupling, or head-output norm.
The six-arm experiment can therefore decide whether this fixed mask beats its
registered controls. It cannot decide that low attention displacement causes
protection. The observed adverse KL, Top-1, and answer-NLL directions already
reject the fixed candidate's proposed joint objective; a rematched reverse arm
would answer a new mechanism question, not rescue this candidate.

## 4. Exact joint relabeling

OLMo2 uses split-half rotary pairs `(j,j+64)`. Define `P_pi` by

\[
(P_\pi x)_{j}=x_{\pi(j)},\qquad
(P_\pi x)_{j+64}=x_{\pi(j)+64},
\]

and `omega'_j=omega_{pi(j)}`. Pairwise rotation then satisfies

\[
R_{\omega'}(t)P_\pi=P_\pi R_\omega(t).
\]

Applying the same `P_pi` to Q and K preserves every attention logit because
`P_pi` is orthogonal. A permanent weight-level implementation must permute the
Q/K projection output rows, their output biases when present, and the learned
Q/K RMSNorm scales. V and the output projection remain unchanged. A runtime
test may instead permute normalized Q/K tensors immediately before RoPE.

For GQA, every Q head sharing one KV head must use the KV head's permutation;
MQA therefore requires one common permutation. OLMo2's stock rotary buffer is
shared across heads, so distinct per-head permutations additionally require a
custom per-head frequency path even though MHA permits them algebraically.

Cached keys are stored after RoPE. Rebuilding the cache or using no cache is
the simplest valid test protocol. Algebraically, a pure relabel can instead
transform every cached rotated key by `P_pi`, so cache rebuild is sufficient,
not logically necessary.

## 5. Effect on current owners

The same-multiset result changes only the final frequency-to-slot assignment
while the learned Q/K coordinate system stays fixed. Its collapse therefore
supports the bounded claim that the learned rotary subspaces and frequencies
form a non-exchangeable ordered pairing. A joint Q/K--frequency relabel changes
the coordinate names together and should be null. The two statements are
consistent.

The head-selective runner constructs per-layer/per-head phases as duplicated
split-half frequencies and calls the same OLMo2 `rotate_half` operation. Its
custom all-Native path was bitwise identical to stock execution at short and
16,309-token checks. Source inspection therefore supports the realized pair
mapping. A nontrivial joint-relabel test was not executed, so exact gauge
invariance must not be reported as an observation.

If a future permutation or per-head mechanism study is authorized, the first
gate should be a nontrivial joint-relabel identity test in FP64/FP32 on the
exact custom path, with cache rebuilt. That gate is not a reason to reopen the
completed Selective-31 sweep or to spend more GPU on the present candidate.

## 6. Claim boundary

### Supported

- Functional compatibility derivatives are local and objective-conditioned.
- The calibration score is exact attention displacement under one endpoint,
  not `chi_func` or behavioural damage.
- Exact joint Q/K--frequency relabeling is a gauge identity under the stated
  pair layout and synchronized parameter permutation.
- Fixed-Q/K frequency-only permutation and joint gauge relabeling are distinct
  estimands; the former's observed collapse is not contradicted by the latter.
- The Selective-31 negative remains valid at its exact arm/protocol scope, with
  no sensitivity-mediation interpretation.

### Unsupported or unresolved

- one task-agnostic functional metric for arbitrary positional interventions;
- endpoint damage from an unchecked first derivative or squared magnitude;
- causal sensitivity protection without utility/output-coupling matching;
- an executed nontrivial OLMo joint-relabel identity receipt;
- using activation/loss-conditioned `chi` to solve the checkpoint-only `m`
  identification problem;
- a new head selector, arm, sweep, GPU run, or manuscript claim.

The practical conclusion is to keep both completed results, narrow their
language, and stop. The audit improves identification discipline; it does not
provide a new checkpoint-only movement law.
