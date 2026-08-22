# Phase-direction headwise RoPE/LoRA retrofit preflight

- **Date:** 2026-08-22
- **Status:** no-GPU data and implementation prepared; **no GPU result**
- **Evidence role:** prospective mature-model retrofit protocol, not paper
  evidence
- **Base:** released OLMo-2-0425-1B-Instruct

## Decision

The previously prepared global-static-table experiment is revoked before GPU
use. Its target continuation was assigned to one of seven natural blocks, but
the input contained no query or other observable variable identifying that
block. More compute cannot recover missing information.

The replacement protocol tests a low-dimensional, headwise allocation family:

\[
\log \omega_{\ell h k}
=
\log \omega^{N}_{k}
+
\alpha_{\ell h}
\left(
\log \omega^{T}_{k}-\log \omega^{N}_{k}
\right),
\qquad
0\leq\alpha_{\ell h}\leq1.
\]

Each layer/head learns only how far to move along one registered whole-table
direction. It cannot independently rewrite the 64 pairs. Convex interpolation
in log frequency preserves the two endpoints and strict ordering for every
realized head table.

The model also learns a per-layer/head, request-budget-aware inverse
temperature.  The budget is fixed before prefill and is held constant for
every cached decode step:

\[
\lambda_{\ell h}(B)=
\begin{cases}
1,&B\leq L_{\rm ref},\\
\exp\!\left(\operatorname{clip}_{[-\log4,\log4]}
\left\{\beta_{\ell h}
[\log(B/L_{\rm ref})]^{1+\operatorname{softplus}(\gamma_{\ell h})}
\right\}\right),&B>L_{\rm ref},
\end{cases}
\qquad L_{\rm ref}=4096.
\]

It is applied to the normalized, rotated query immediately before the standard
attention call.  The braced term is the product of the learned log-gain
coefficient and the powered log budget ratio.  The scale is exactly one
throughout the Native window, including after training.  Q/K/V remain D128; there is one
Flash-compatible attention operation and an ordinary-width KV cache.  A
missing or changing request budget is a hard error because it would invalidate
prefill/decode cache equivalence.

## Why all three factors are required

1. **Observable source credit.** The successful historical query-gap chain
   used explicit source/value counterfactuals, target-range phase exposure,
   answer supervision, and terminal EOS. Generic CE and the completed
   far-pass QK/QKVO/physical-8K chain improved NLL or rank without producing
   strict generation.
2. **Headwise positional spectrum.** A global mature-model table replacement
   changes 63/64 rotary pairs and cannot be exactly repaired by fixed Q/K
   conjugacy. A scalar per head lets irrelevant heads remain at Native while
   useful heads move along a pre-registered non-geometric direction.
3. **Headwise attention concentration.** If a correct key has logit margin
   \(\delta>0\) over \(n\) distractors, maintaining target probability
   \(\eta\) requires approximately
   \(\lambda\delta\ge\log(n\eta/(1-\eta))\). Frequency can change ordering;
   temperature controls dilution after ordering is correct. Temperature alone
   cannot repair a negative margin.

AdaRoPE independently supports the need for head-specific frequency and
length-aware temperature, and reports its strongest mature strategy as a
positional warm-up followed by joint LoRA/positional optimization. This
protocol does not relabel AdaRoPE as our contribution. The paper-specific
intervention is the constrained fixed-support phase direction and its matched
direction control. See [AdaRoPE](https://arxiv.org/abs/2607.19363).

## Candidate and control identities

All completed arms must share the Stage-0 parent, training rows/order, token
budget, optimizer schedule, QKVO rank-64/alpha-128 LoRA, short replay, compile
mode, and evaluation rows.

| Arm | Headwise frequency | Headwise temperature | Scientific role |
| --- | --- | --- | --- |
| `lora_only_null` | Native (`alpha=0`, frozen) | fixed one | matched long-LoRA null |
| `native_scale` | Native (`alpha=0`) | learned | tests attention dilution without changing coordinates |
| `context_stretch_exp_negative` | learned alpha along the frozen negative-mean exponential warp | learned | practical context-stretch prior |
| `phase_chord` | learned alpha along OLMo-R0 phase-chord direction | learned | proposed attention-derived direction |
| `moment_matched_same_sign_control` | learned alpha along a fixed two-basis analytic warp matching phase signed mean and RMS log displacement | learned | conditional phase-shape attribution control |

The three practical candidates are `native_scale`,
`context_stretch_exp_negative`, and `phase_chord`.  `lora_only_null` is the
required practical null.  The moment-matched control is not eligible to win
the practical tournament and is run only if `phase_chord` wins.  Its two fixed
endpoint-zero basis functions are `u(1-u)` and
`u(1-u)(2u-1)`; a pre-registered positive square-root branch matches both the
signed mean and RMS of phase-chord's log-frequency displacement without using
the attention profile.  Only a final phase-versus-moment-control comparison,
with all retention gates passed, can support a phase-shape attribution.

## Identifiable natural-data contract

Every semantic group is derived only from frozen FineWeb-Edu token rows. A
compact physical-4K passage contains 16 unique eight-token anchors, each
followed by an eight-token natural answer. The counterfactual variant applies
a fixed-point-free permutation only to those answer spans; its query and
geometry remain unchanged. Therefore each query-visible anchor identifies one
recoverable source answer in both variants.

Physical 8K and 16K views preserve the compact semantic group and insert
exactly 4K or 12K natural distractor tokens between passage and query. This
changes real key count, softmax denominator, cache exposure, and cross-layer
states; no virtual position IDs substitute for the physical evaluation.

The splits are document/source-row disjoint and use different templates:

- `train`: 128 semantic groups, 16 queries per sequence, dense teacher-forced
  answer and alternate-token supervision;
- `train_eos`: 32 of the training semantic groups under a distinct single-
  query prompt, used only for the fixed 32-step answer-plus-immediate-EOS
  continuation;
- `component_gate`: 32 disjoint groups, one strict query whose anchor bin
  rotates across rows;
- `final_validation`: 16 validation-only groups, a second unseen strict-query
  template, never available to method selection;
- `warmup4k_clm` and `retention4k_raw`: independent source rows.

The root V3 data-manifest SHA-256 is
`142f34ebd7d05f04c16dc53288d734c12335888e844c84098b1f64165bd3caf4`.
It owns 12 matched views (`train`, `train_eos`, `component_gate`, and
`final_validation` at 4K/8K/16K). Strict views stop the prompt at the unique
answer start and define the complete target as eight natural tokens followed
immediately by terminal EOS.

No RULER, NIAH, 2Wiki, or benchmark generator is imported by the data module.
Official LongBench 2Wiki is evaluation-only; the frozen official archive has
SHA-256 `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`
and contains exactly 200 `2wikimqa` rows.

## Adaptive GPU sequence

### Stage 0: acquire the computation in-window

- Native RoPE; standard QKVO LoRA, rank 64 / alpha 128.
- `train4k`: 300 steps, physical 4K, dense 16-query answer CE plus registered
  correct-over-alternate margin.
- `train_eos4k`: fixed 32-step strict answer-plus-EOS continuation.
- Independent raw-4K natural replay.
- Advancement requires, on `component_gate4k`, at least 0.90 complete
  answer-plus-terminal-EOS exact, 1.00 terminal-EOS rate, and 0.90 positive
  correct-source effect.

If Stage 0 fails, no positional method is run: the source/readout computation
was not acquired, so a long-position experiment would be uninterpretable.

### Stage 1: matched 50-step practical tournament

All four practical arms start from the same frozen Stage-0 adapter and consume
the same serialized `train16k` rows, raw-4K replay, optimizer schedule, and 50
physical-16K steps:

- `lora_only_null`: QKVO LoRA updates for all 50 steps;
- `native_scale`: 20 temperature-only steps, then 30 temperature-plus-LoRA
  steps;
- `context_stretch_exp_negative`: 20 alpha-plus-temperature steps, then 30
  joint LoRA steps;
- `phase_chord`: the same 20-plus-30 schedule along the phase direction.

The first 20 steps are a positional warm-up, not a separate experiment.  LoRA
gradients remain in one compiled graph but receive no optimizer update during
that warm-up.  Selection uses only `component_gate16k`.  Each candidate is
paired against the 50-step null on exactly 32 documents and must satisfy all
of the following: positive source-effect delta in at least 90% of documents;
positive mean delta in both fixed 16-document halves; lower mean first-token
gold rank; mean 16K NLL delta at most `+0.10`; and raw-4K candidate-minus-Native
NLL at most `+0.10`.  The deterministic winner is the qualifying arm with the
largest mean source-effect delta, with registered rank and arm-order
tie-breakers.  If none qualifies, the null is selected.  At most one practical
candidate can advance.

### Stage 2: continue the winner and matched null

The selected practical arm and `lora_only_null` continue from their own
50-step bundles for 250 further physical-16K steps, then receive the fixed
32-step `train_eos16k` continuation.  They do not restart from Stage 0 and do
not merge separately trained components.  Global row order and learning-rate
position continue from step 50.

If and only if `phase_chord` wins, the moment-matched control first receives
the same 50-step tournament schedule.  Failure to beat phase on the registered
paired mechanism gate kills the phase-specific attribution.  A passing
control may continue to 300 steps for the matched causal comparison; it is not
retroactively eligible to change the practical winner.

The initial shape-specific smoke must establish Flash-only execution, finite
loss, nonzero finite gradients in every authorized parameter group, ordinary
D128 cache width, fixed-budget full-forward versus prefill/decode cache parity,
at least 1 GiB allocated-memory headroom, throughput, and a fresh
PEFT-plus-sidecar roundtrip before any full run.  A second one-step smoke owns
the Stage-1-to-Stage-2 continuation path.

## Final endpoints and claim boundary

The first capability endpoint is complete eight-token answer equality plus a
single terminal EOS on never-selected natural final rows. Teacher-forced NLL,
attention mass, source effect, or first-token rank cannot substitute for it.

The full practical winner and full `lora_only_null` are always evaluated on
held-out RULER and 2Wiki after their emitted PEFT adapters and phase sidecars
pass fresh-load identity checks.  Final natural metrics do not decide whether
negative downstream evidence is recorded; using them as a second selection
gate would create an avoidable post-selection path.

RULER and 2Wiki remain evaluation-only.  Final natural evaluation is run at
4K, 8K, and 16K, but none of those final rows is available to tournament
selection.  A positive natural result without
positive held-out downstream transfer supports only natural source-use
stretching. A positive `phase_chord - moment_matched_same_sign_control`
downstream result, with 4K retention, supports a protocol-specific
phase-shape retrofit within this registered direction family. One
training seed cannot support stability, universality, significance, or a
claim that the method has a known greater-than-90-percent success probability.

## Compute boundary

Historical RTX-5090 receipts put standard 16K QKVO LoRA near 24K input
tokens/s and roughly 21--22 GiB before the differentiable headwise rotary
activations.  The four 50-step tournament arms plus one 250-step winner and
one 250-step null total 700 registered 16K steps.  The phase-only attribution
path raises the worst case to 1,000 steps.  At the historical throughput this
fits the 2.5-hour window with substantial evaluation headroom, but these are
planning estimates, not runtime evidence.  If a measured 300-step equivalent
exceeds 25 minutes, the priority is winner plus null; the conditional control
is dropped before either matched arm is shortened.  The exact-shape smoke owns
the final batch, memory, and throughput decision; quadratic math-attention
fallback is forbidden.
