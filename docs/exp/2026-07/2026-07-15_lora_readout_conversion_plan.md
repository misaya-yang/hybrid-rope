# 16K readout-conversion: next experiments (plan, 2026-07-15)

## Status

This is a forward-looking experiment plan, not a results report. It does not
modify or replace any paper result and touches the frozen matched step-300
Geo/EVQ adapters only as read-only inputs. It builds on
`2026-07-14_lora_retrieval_conversion_probe.md` and treats every number in
that probe as single-seed, ten-case supporting evidence: those numbers
license which experiment to run next, never a paper claim.

Four categories are kept strictly separate. Every experiment below is tagged
with exactly one, and results must be reported under that tag:

- `ZT-0`: no parameter update; no dev-set selection of any scalar or layer.
- `ZT-cal`: no parameter update, but one global scalar or layer chosen on an
  independent dev set. This is calibration, not zero-shot.
- `oracle-diagnostic`: uses the true gold block or a per-sample-best layer.
  It establishes whether information exists; it can never be reported as a
  deployable result.
- `supervised`: updates model parameters on 16K gold-labeled data.

Training a tuned-lens translator is `oracle-diagnostic`, not zero-training.

## Bottom line: one measurement settles the fork

The whole question of whether training-free conversion is still possible
reduces to the behaviour of the first-token causal logit-delta

    d_j(t) = z_full(t) - z_ablate(t)

evaluated at the first gold answer token `g_1`, where `z_full` is the answer-
position logit with the full 16K context and `z_ablate` is the same logit
with the gold block removed on every head (the probe's `gold_drop_all`).

Two binary tests decide everything:

1. Does the causal contribution concentrate at `g_1` (not only in the
   suffix / continuation tokens)?
2. Does `d_{g_1}(.)` follow a query->key association swap (below)?

- Both yes -> the precise answer identity is present in the residual but
  suppressed by the language-model prior. A non-monotonic, task-aware readout
  (causal-delta amplification or extractive decode) can convert it without
  weights. Report as `ZT-cal` / wrapper, never as base-model zero-shot.
- Either no -> the checkpoint encodes "an answer is here / keep copying", not
  the identity. No decode rule can manufacture identity; the failure is value
  content or copy strength and requires 16K `supervised` training. EVQ's
  established contribution stays "better selection and lower NLL", not "a
  convertible readout".

Prior from first principles plus the probe's own nulls (below): conversion is
a narrow path. Rough odds that a deployable `ZT-cal`/wrapper reaches robust
16K exact match: 15-25%. Odds it at least lifts decoy-matched candidate MRR
materially (a wrapper-only claim): ~50%. The expected terminal branch is
`supervised`. But the gating measurement is near-free and decisive, so it is
run before any GPU is spent on training.

## First-principles framing (attention and RoPE)

Standard Llama RoPE rotates only Q and K, not V. Three consequences drive the
design.

1. "The value coming back is insufficient" is two physically distinct
   hypotheses that no single intervention separates:
   - `1a` content degradation of `h_gold` at an out-of-distribution absolute
     position. V itself is unrotated, but the hidden state that produces
     `V_gold` was built by rotated attention in earlier layers. The low-
     frequency RoPE channels have period longer than the 8K training length,
     so absolute phase `theta_i * n` at `n > 8K` is a phase never seen in
     training and `h_gold`'s content is corrupted. This is a positional
     extrapolation effect and is partially reachable by the frequency base.
   - `1b` weak OV/copy readout: the path that writes `h_gold`'s token-identity
     subspace into the residual so the unembedding reads `g_1` is too weak, or
     is overwritten by later layers. This is orthogonal to RoPE, is a weight
     property, and cannot be moved without training.

2. EVQ improves selection, not copy. Redistributing `theta_i` keeps more
   channels non-aliased and the relative phase `theta_i * (m - n)` resolvable
   at 16K, which explains hit@16 rising to 64% and first-token rank moving
   33,775 -> 2,043 (addressing improved). It does not touch the OV / MLP /
   unembedding copy circuit, so if the residual gap is mostly `1b`, EVQ is
   structurally unable to close it, consistent with rank stuck near 2,000 and
   0% exact match.

3. absolute-position, relative-distance, and total-length are confounded in
   the current results and must be separated. At 8K success both the gold
   absolute position `a_g` and the length `L` are in-distribution; at 16K all
   three can be OOD at once. This is the cheapest lever to disentangle `1a`,
   `1b`, and query-side OOD, and needle depth is already logged per case.

Induction-head reading: the probe's forced-gold result (`+0.034` NLL when
attention is maximised on gold) says the match / addressing step is not the
bottleneck; the copy step is. A broken copy step has only two fates -
identity present but prior-suppressed (rescuable without weights) or identity
absent / degraded (training only). The association swap decides which.

## Already settled: do not revisit

Reported as likelihood ratios under the probe's per-token geometric-mean NLL
convention:

- 15.51 -> 9.06 answer NLL is `exp(6.45) ~ 633x` per-token likelihood, so EVQ
  genuinely moved the target.
- gold-block deletion on every head is `+1.5055` NLL, `~4.51x` likelihood
  loss: the source is causally used.
- forced-gold inclusion is `+0.034` NLL, `~3.5%`: maximising attention on gold
  does almost nothing.

The asymmetry (huge target movement, negligible forced-gold gain) plus RoPE
principle 2 kills three families of training-free levers:

- monotonic decoding: temperature does not reorder; top-k / top-p only delete
  the rank-2043 gold earlier; beam must retain >2043 first-step branches and
  the sequence score still need not flip. rank data alone closes this.
- input-side attention re-weighting: score-sparse, fixed, forced-gold, and
  matched-budget oracle inclusion are all null or near-null on EVQ.
- runtime-frequency re-scaling: the source-canary cross-swap shows the EVQ
  adapter at native runtime frequency is 9.246 NLL versus 9.073 at EVQ runtime
  frequency, only `0.17` of headroom; the gain is training-time co-adaptation.
  A zero-training YaRN-style re-scale therefore has almost no room on this
  adapter.

Do not scan sparse budgets, block counts, sparse thresholds, temperatures, or
beam widths. What remains is only output-side non-monotonic readout, which is
where this plan concentrates.

## Reconciled root-cause ranking

| Rank | Root cause | Judgement under current evidence |
| --- | --- | --- |
| 1 | first-token answer-specific value/readout, split into `1a` content-OOD (RoPE positional, partly frequency-reachable) and `1b` copy strength (orthogonal to RoPE, training-only) | Highest. Extra attention mass is nearly useless; deleting gold hurts. Gate on Z0: is the NLL gain at `g_1` or in the suffix. |
| 2 | 8K -> true-16K length distribution and target-length credit assignment | High, but a `supervised` question. The 8K 50-step tune never updated parameters at 16K positions. |
| 3 | cross-layer routing / late-layer overwrite | Medium-high. Discriminated by the per-layer causal-delta trajectory. |
| 4 | training objective / loss allocation | Conditional. answer-only CE already gives near-maximal `g_1`-logit gradient when `p_g` is tiny (`dL/dz_g = p_g - 1 ~ -1`); only promote if Z0 shows `g_1` is under-trained relative to the suffix. |
| 5 | ordinary decode calibration | Low / closed. Monotonic operators cannot move rank 2043 to top-1. |
| - | query-side position OOD (Q at absolute position 16K has OOD low-frequency phase) | Sibling of `1a`. Discriminated by whether gold-late (short relative distance, high absolute position) recovers readout. |

LoRA target modules are a sub-question of ranks 1 and 3, not a separate cause:
if adaptation mostly changes Q/K while V/O, late MLP, and final norm stay
effectively frozen, it can improve addressing without building copy readout.
Treat this as an experimental prior (branch into S3), not a conclusion.

## The linchpin figure

The single most informative next artifact is not a new exact-match bar chart.
It is

    rank_t[ z_full(t) - z_ablate(t) ]

plotted as a function of (answer position `j`, layer `l`, needle depth `d`),
overlaid with the association swap. One figure separates four mechanisms at
once: the `j` axis separates continuation from identity, the `l` axis
separates mid-layer readability from late-layer loss, the `d` axis exposes
absolute-position OOD, and the swap separates identity from generic signal.

## Track Z: zero-training conversion and diagnostics

All Track Z experiments reuse the unchanged matched step-300 adapters, dense
16K prefill, the same block geometry, and matched Geo/EVQ pairing. None update
weights.

### Z0. First-token vs suffix causal decomposition (`ZT-0` / `oracle-diagnostic`)

- manipulated variable: for each answer position `j`, compute NLL, rank, and
  logit margin under full and gold-block-ablated context; separately force
  only `g_1` and then greedy-decode the suffix normally.
- prediction: if the remote signal is mainly continuation, the `g_1` block-
  ablation effect is small and later-position effects are larger, and suffix
  exact match after forcing `g_1` approaches saturation. If every position is
  weak, this is not a single first-token gate.
- metric: per-position `dNLL_j`, `dz_j(g_j)`, per-position rank; suffix-EM
  given forced `g_1`; share of total NLL gain attributable to `g_1` versus
  suffix.
- kill condition: if suffix EM after forcing `g_1` stays below 80%, or several
  later positions keep median rank above 10, kill the "only the first token is
  stuck" story.
- budget: 256 fixed 16K samples; if teacher-forced logits are reused, added
  cost is 256 short decodes, at most 256 16K prefills.

### Z1. Association-swap causal-delta rank (`ZT-0` diagnostic; oracle-gated `ZT-cal`)

Build two contexts with the same token multiset: a queried key maps to answer
A and a decoy key maps to B; the matched version swaps the association so the
queried key maps to B and the decoy key to A. Keep physical layout, positions,
first-token frequency bucket, and template identical; change only which key
the query selects (position-symmetric swap, no depth confound).

- manipulated variable: the query->key association only, run under full and
  gold-block ablation.
- prediction: if the value path carries precise content, `d(A) - d(B)` flips
  with the swap. If it carries only position, format, or an "answer needed"
  signal, the delta does not follow A/B.
- metric: causal rank `r_d(g)`; swap-follow score
  `[d_xA(A) - d_xA(B)] - [d_xB(A) - d_xB(B)]`; gold-vs-matched-decoy causal
  margin.
- kill condition: swap-follow paired CI does not exceed 0, or gold stays near
  random absolute rank in `d`, kill "this checkpoint already has directly
  readable precise identity" and go to Track S.
- budget: 128 dev + 128 test matched pairs, two forwards each, ~1,024 16K-
  prefill equivalents; sharply less if ablation logits are reused.

Scalar amplifiability without a sweep. For `s_alpha(t) = z(t) + alpha * d(t)`,
solve analytically per sample: gold is top-1 iff for every competitor `t`,
`alpha * [d(g) - d(t)] >= z(t) - z(g)`. Each competitor is one half-line in
`alpha`; the sample is amplifiable iff the intersection is non-empty. Order:
(1) oracle gold block for the upper bound; (2) only if oracle-feasible, the
frozen block selector for the deployable delta; (3) `alpha = 1` is `ZT-0`;
(4) a single global `alpha` chosen on dev is `ZT-cal`.

- kill condition: if under oracle gold block at least 90% of samples have no
  feasible `alpha`, stop all causal-delta amplification. If oracle-feasible
  but no global `alpha` gives material test top-1 / EM, stop scalar contrast.
  Only if oracle succeeds and the selector version fails, revisit the selector
  (no sparse threshold or block-count scan).

### Z2. Cross-layer causal trajectory (`oracle-diagnostic`)

Per layer `l`, read `d_l = W_U Norm(h_full_l) - W_U Norm(h_ablate_l)`. Prefer
the full-minus-ablated delta over a raw logit lens because the subtraction
cancels most per-layer baseline misalignment.

- manipulated variable: readout layer `l`; aggregate to layer, no per-head
  search; no weight change.
- prediction: a mid layer with low `r_{d_l}(g)` while the final layer degrades
  supports late-layer overwrite / routing failure; no answer-specific layer
  anywhere supports missing value content.
- metric: per-layer causal rank, causal margin, scalar-feasible fraction;
  retention ratio `= final causal margin / peak causal margin`.
- causal confirmation: patch only the peak layer and one layer either side via
  residual-difference amplification, continue through the remaining layers,
  and confirm the effect is not a logit-lens artifact.
- kill condition: no layer improves causal rank/margin on most samples and the
  peak-layer patch gives near-zero paired improvement in final gold margin ->
  kill the cross-layer-overwrite hypothesis.
- budget: reuse Z1 full/ablated forwards, storing only the final query-position
  residual per layer; extra causal patching limited to 3 layers x 64 samples.

### Z3. RoPE position disentangling and extractive decode (`oracle-diagnostic`, then `ZT-0`/`ZT-cal`)

Two parts share the same 16K prefills.

Part A, position stratification (near-free, uses logged depth). At fixed
`L = 16K`, report EVQ first-token gold rank stratified by needle depth, plus
two matched cells: gold-early (`a_g ~ 2K`, long relative distance, in-
distribution gold value) and gold-late (`a_g ~ 15K`, short relative distance,
OOD absolute gold position).

- manipulated variable: gold absolute position at fixed length; query position
  fixed at the end.
- prediction: early works and late fails -> `1a` value absolute-position
  extrapolation. early fails and late works -> relative distance / query
  reach. both fail -> query-side absolute-position OOD dominates. rank flat
  across depth -> length / query OOD, not gold-specific.
- metric: first-token gold rank and NLL by depth; early-vs-late paired
  difference.
- kill condition: if the early-vs-late paired difference CI covers 0 and depth
  slope is flat, drop the value-position-OOD branch and concentrate later
  supervision on copy strength (`1b`).
- budget: reuse existing 16K runs plus at most 2 x 128 re-placed-needle
  prefills.

Part B, extractive / trie-constrained decode. Constrain decode support from
the full vocabulary to context spans matching the answer length / character
class, via trie-constrained AR decode or frozen-model sequence rescoring; the
candidate set must not use gold location, and each sample must contain several
same-format decoy passkeys or an external regex solves it and the test is
meaningless.

- manipulated variable: decode support (full vocab vs oracle-complete context
  candidates).
- prediction: if this is mainly an output-prior problem, gold candidate rank
  drops from thousands to 1; if association/value is not established, candidate
  rank stays poor.
- metric: candidate top-1 EM, MRR, sequence-score margin, restricted first-
  token rank, and unrestricted EM for contrast.
- kill condition: in the oracle-complete, decoy-matched candidate set, gold MRR
  is indistinguishable from the random baseline, or 128 test samples still hit
  0, stop the extractive wrapper.
- budget: 256 prefills; up to 32 short candidate continuations per sample with
  shared KV, far below re-running 16K prefills.
- reporting: any success is "frozen LM + extractive decoding wrapper", never
  "the base model obtained unconstrained 16K passkey EM"; report full-sequence
  EM, not only first-token, because causal amplification can cost suffix
  fluency.

## Track S: 16K supervised learning

Every Track S experiment updates parameters on 16K gold-labeled examples,
strictly separate from any `ZT-cal` scalar/layer calibration. Shared setup:
fix the same EVQ checkpoint, data generator, answer / tokenization
distribution, and gold relative-position buckets; primary endpoint is the
first-token logit margin `m = z(g_1) - max_{t != g_1} z(t)` and full EM;
secondary endpoints are first-token rank, per-token NLL, and gold-block causal
margin; a fixed 512-example held-out set spans at least 8 position buckets;
seed-42 is discovery/kill only, and any positive recipe is frozen and re-run on
3 new training seeds; all budgets are reported in input tokens and answer
instances, never "N steps". Do not repeat the failed 8K 50-step recipe.

### S1. Length-distribution three-arm matched experiment (`supervised`)

`N = 1024` semantically matched training samples.

| Arm | Data | Control purpose |
| --- | --- | --- |
| E16-N | N true 16K samples | target-length handling |
| E8-N | 8K matched version of the same N | answer-instance matched |
| E8-2N | 2N 8K samples | input-token / approximate compute matched |

Objective, optimizer, LoRA modules, answer-position distribution, and data-
order strategy are held identical; 8K/16K pairs share query, answer, and noise
semantics and differ only in filler length.

- manipulated variable: training length distribution and sample count across
  the three arms.
- prediction: if target-length OOD is primary, E16-N beats both 8K arms on 16K
  first-token margin, top-1, and far-position buckets. If it is only more
  data / more answers, E8-2N matches E16-N.
- metric: margin, top-1, EM at 8K/12K/16K, bucketed by gold-query distance and
  relative position; causal-delta rank; 8K regression check.
- kill condition: after training on 2N 16K-equivalent examples, E16 beats
  neither 8K control on first-token margin and EM and its learning-curve slope
  is near 0 -> kill "length exposure alone suffices".
- budget: first pass E16-N ~16.8M tokens, E8-N ~8.4M, E8-2N ~16.8M; evaluate
  every 256 answer instances; only the arm with a rising margin extends to 2N.

### S2. 16K training-objective experiment (`supervised`)

Pick the single necessary contrast from the current loss; no three-or-four
loss sweep.

If the current loss covers many context tokens: original loss vs answer-only
CE with context tokens masked. If it is already answer-only CE: answer-only CE
vs answer-only CE plus a first-token hard-competitor term
`L_pair = log(1 + exp[z(d*) - z(g_1)])`, where `d*` is the strongest non-gold
first-token logit, including the strongest matched-decoy passkey first token.
Set the weight once by first-batch gradient-norm matching, no sweep.

- manipulated variable: loss only; 16K data, parameters, adapter modules, and
  token budget identical.
- prediction: if objective / teacher-forcing is primary, the pairwise arm
  raises margin-crossing rate and EM even when total NLL barely moves.
- metric: fraction of samples with `m > 0`, first-token top-1, full EM,
  suffix-EM, answer NLL.
- kill condition: at equal tokens the pairwise / answer-only arm beats the CE
  baseline on neither first-token margin nor EM -> drop objective as primary;
  do not proceed to RL or sequence-level policy optimization.
- budget: 2 arms x 1024 16K samples, ~33.6M input tokens; extend to 2048 only
  on a positive margin slope. Only promote to S2 if Z0 shows `g_1` is under-
  trained relative to the suffix.

### S3. Adapter locus chosen by Z1/Z2 (`supervised`)

No full-module enumeration; one experiment arm chosen by the diagnostics plus
a parameter-matched Q/K addressing control, trainable parameter counts within
about +/-5%.

- branch A (Z1: no layer has answer-specific delta): experiment arm V/O LoRA
  on all layers; control Q/K LoRA. Prediction: only V/O improves swap-follow
  and causal gold rank.
- branch B (Z2: mid-layer answer-specific, late-layer lost): experiment arm
  last-8-layer O projection plus MLP/readout adapter, opening final norm if
  needed; control a parameter-matched Q/K adapter. Prediction: the late-layer
  arm restores retention ratio, first-token margin, and EM while gold-block
  attention mass is roughly unchanged.
- branch C (causal delta already answer-specific, only under-scaled): late
  readout / final-norm / lm-head-side adaptation; do not change Q/K or sparse
  pattern.
- metric: first-token margin/EM; swap-follow; per-layer causal rank; gold-block
  attention mass; causal logit gain per unit gold-block attention.
- kill condition: at equal parameters and equal tokens the mechanism-predicted
  arm beats the Q/K / current-LoRA control on neither margin nor EM -> kill the
  adapter-locus explanation; only then consider opening norm/embedding or a
  small full-layer fine-tune.
- budget: 2 arms x 1024 16K samples, ~33.6M tokens; only a positive result
  goes to multi-seed.

### S4. EVQ-specific confirmation, not more seed-42 (`supervised`)

Only after a fixed 16K recipe produces non-zero EM: run
`{Geo, EVQ} x {8K supervision, 16K supervision}` with 3 new seeds not used in
recipe selection.

- manipulated variable: substrate x supervision length, replicated across new
  seeds.
- prediction: EVQ x 16K learns faster and ends higher only if EVQ's remote
  signal is a target-length-usable substrate.
- metric: interaction `I = (M_EVQ,16K - M_EVQ,8K) - (M_Geo,16K - M_Geo,8K)`,
  with `M` first-token margin primary and EM secondary; per-seed CIs.
- kill condition: the interaction direction is inconsistent across new seeds or
  its CI covers 0 -> do not claim "EVQ is more convertible to top-1"; keep any
  seed-42-only positive as a supporting observation, not a main result.
- budget: 4 arms x 3 seeds x 16.8-33.6M tokens, ~200-400M input tokens. This is
  confirmatory budget and must not be paid before Z1/Z2 locate the mechanism.

## Execution order and decision tree

1. Run Z0 and Z1 first; both can largely reuse existing ablation outputs. They
   answer the core question: is the remote contribution at the first token and
   does it carry precise identity. Produce the linchpin figure here.
2. Only if Z1 is oracle-feasible, run Z2/Z3: `d(g)` already near top-1 ->
   causal-contrast readout; mid-layer good, final bad -> cross-layer readout;
   candidate-set top-1 -> an engineering wrapper.
3. If Z1 is answer-specific at no layer/position, stop zero-training conversion
   immediately and enter S1. Do not retry temperature, beam, top-p, sparse
   block counts, or forced attention.
4. After S1, branch on the learning curve: E16-only gain -> length; NLL down
   but margin not crossing zero -> S2; margin unmoved -> S3.
5. Only a fixed, already-successful recipe proceeds to the S4 Geo x EVQ x
   length confirmation on new seeds.

## Discipline and labeling

Single-seed results license the next experiment, never a paper claim. Any
"EVQ is more convertible" statement must come from S4's four-arm, three-new-
seed interaction with a CI excluding 0. `ZT-0`, `ZT-cal`, `oracle-diagnostic`,
and `supervised` are never mixed in reporting; a dev-selected scalar or layer
is calibration, and an oracle gold block or per-sample-best layer is a
diagnostic upper bound, not a deployable number.

## Feasibility notes

- Z0/Z1/Z2 reuse `_first_step_logits`, `gold_drop_all`, `_target_rank`, and the
  dense/full logit-parity path already in
  `experiments/lora_evq_v2/eval_sparse_conversion.py`; the causal-delta rank is
  a small addition over saved dense and gold-ablated first-step logits.
- Needle depth is already recorded per case (the probe reports effects "at
  every registered needle depth"), so Z3 Part A depth stratification is a
  re-analysis of existing 16K runs plus a small re-placed-needle set.
- The rank metric is teacher-forced against the gold string, so rank ~2,000 is
  a clean readout measurement, not a harness artifact. Separately, the 0% EM
  absorbs a generation-format pathology (no EOS, fixed first token, full 32-
  token budget) that Z0's forced-`g_1` / suffix-EM split isolates from the
  readout-rank gap.

## References as priors, not conclusions

Context-aware decoding (amplify the with-evidence vs without-evidence logit
gap) motivates Z1's causal-delta amplification; tuned lens and DoLa-style
layer contrast motivate Z2 but require per-layer translators or are task-
specific, so they are diagnostic only and do not by themselves imply passkey
gains; long-context PEFT work suggests standard LoRA adaptation sites may be
insufficient and that opening normalization/embedding can matter, used here
only as the S3 branch prior. None of these is imported as a result.
