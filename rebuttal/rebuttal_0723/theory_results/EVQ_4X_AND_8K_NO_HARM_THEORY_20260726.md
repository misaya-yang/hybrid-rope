# EVQ 4× capability and 8K no-harm: theory and experiment design

Date: 2026-07-26

Status: **internal theory synthesis / design only / not a reviewer-facing result**

Concern mapping: `R27bE.2`, `R27bE.5`, `AC.2`

## 1. Five-line contract

1. **Concern addressed.** Can EVQ provide useful behavior at four times the
   adaptation length on a mature 8B model without sacrificing its native 8K
   behavior?
2. **Existing evidence.** Across mature Llama-8B adaptation, mature OLMo-2 1B
   adaptation, and a matched OLMo-2 1B from-scratch trajectory, EVQ repeatedly
   pays an in-window NLL cost and improves longer-window NLL. It converts one
   2× NIAH task after matched task-family adaptation, but current 4× RULER
   behavior remains at or near zero.
3. **Smallest missing evidence.** One matched arm must retain the final
   Native-LoRA 8K endpoint while producing nonzero autoregressive 32K behavior
   on held-out rows. Lower 32K NLL alone is insufficient.
4. **Smallest executable plan.** First finish the already-running matched
   Native-LoRA control. If the expected crossover is confirmed, test one fixed
   EVQ arm with physical-8K, phase-covered 1×/2×/4× routing supervision and a
   pre-registered 8K no-harm constraint. Do not begin with a schedule sweep.
5. **Stop condition.** Stop if the candidate cannot match Native-LoRA at 8K,
   does not improve the existing EVQ 16K result, or remains zero on all
   pre-registered 32K autoregressive canaries. Do not interpret a PPL-only
   improvement as 4× capability.

Here “4×” always means **8K physical adaptation to 32K evaluation**, not a
fourfold score improvement. “8K better” means a matched no-harm condition
relative to Native-LoRA, not merely better than an earlier EVQ checkpoint.

## 2. The empirical pattern that the theory must explain

### 2.1 Mature Llama-8B, original matched LongAlpaca adaptation

The retained seed-42 evidence reports EVQ-minus-Native NLL of

\[
+0.390,\quad -1.510,\quad -2.048
\]

at 8K, 16K, and 32K. The long-window direction holds on every one of 24
temporal packs, while the 8K direction is worse on all 24. At true 16K, EVQ
uses the remote source more strongly and improves the correct-token rank, but
does not reach top-1 generation.

### 2.2 Mature OLMo-2 1B, physical-4K task adaptation

Matched Native/EVQ Q/K/V/O LoRA gives:

| Arm | 4K NLL | 8K NLL | 16K NLL |
| --- | ---: | ---: | ---: |
| Native | 2.2354 | 3.7353 | 4.8511 |
| EVQ, seed 1 | 2.5480 | 2.7033 | 2.9245 |
| EVQ, seed 2 | 2.5418 | 2.6891 | 2.8989 |

EVQ is worse at the adaptation length and much better beyond it. On a fresh
all-long-gap 8K NIAH set, Native is 0/100 and the two EVQ training seeds are
49/100 and 48/100. At 16K the same narrow task is close to zero, and held-out
4K RULER tasks reveal severe function loss after full frequency replacement.

### 2.3 OLMo-2 1B from-scratch trajectory

At the same 2.097B-token checkpoint, EVQ-minus-Geo NLL is

\[
+0.0381,\quad -0.0437,\quad -0.1351
\]

at 4K, 8K, and 16K. This is weaker than the mature-model contrast but has the
same crossover direction without a post-hoc LoRA transplant.

### 2.4 Current Llama-8B physical-8K RULER experiment

The live, not-yet-promoted EVQ arm repeats the probability crossover:

| Prefix | Bare Native NLL | EVQ plus RULER-mix LoRA NLL | Delta |
| --- | ---: | ---: | ---: |
| 8K | 2.0729 | 2.5068 | +0.4339 |
| 16K | 5.0144 | 3.4689 | -1.5455 |
| 32K | 7.3089 | 5.0994 | -2.2094 |

Its complete 13-task RULER official macro is 77.60%/14.03%/0% at
8K/16K/32K. The untouched Native model is 91.82%/0%/0%. This comparison is
not matched adaptation; the matched Native-LoRA continuation is currently
running and owns the attribution test. These live numbers must not enter a
reviewer response until raw outputs and a standalone report are promoted.

At step 375 of 516, the matched Native continuation has a last-25-step mean
training loss of 0.6695 versus 0.8733 for EVQ at the identical step. Native is
lower at every logged checkpoint from step 25 through 375. This is live
optimization evidence, not a final evaluation result, but it supports the
prediction that the Native substrate is easier to fit inside the training
window.

### 2.5 Stable inference from all four observations

The most economical explanation is not “EVQ is generally better”:

1. EVQ improves the conditioning or coverage of positional phase features
   outside the training window.
2. A single global full-EVQ table pays an in-window cost.
3. In a mature model, abrupt replacement adds a second cost: the pretrained
   Q/K content channels were co-adapted to Native frequencies.
4. Better long-window token probability does not teach an autoregressive
   retrieval, counting, tracking, or QA algorithm.

The target is therefore a constrained multi-objective problem, not another
single-metric \(\tau\) search.

## 3. A minimal mathematical model

For one attention head, after grouping each rotary pair, the positional part
of a score can be written as a trigonometric feature expansion

\[
s_h(x_i,x_j,\Delta)
=
\sum_{k=1}^{K}
\left[
a_{hk}(x_i,x_j)\cos(\omega_k\Delta)
+
b_{hk}(x_i,x_j)\sin(\omega_k\Delta)
\right].
\]

The frequency table \(\Omega=\{\omega_k\}\) chooses the phase-feature basis;
training chooses the content-dependent coefficients \(a_{hk},b_{hk}\).

### 3.1 What EVQ can improve

Over a target lag distribution \(D\), define the phase-feature matrix

\[
\Phi_\Omega(D)
=
\left[
\cos(\omega_k\Delta),\sin(\omega_k\Delta)
\right]_{\Delta\in D,k\le K}.
\]

A non-geometric allocation can reduce feature coherence or improve effective
rank over a wider lag set. This is consistent with the submitted exact-kernel
diagnostics and the repeated long-window NLL crossover.

### 3.2 What EVQ alone cannot identify

Training only constrains the learned trigonometric polynomial on observed
lags. Two coefficient/frequency systems can behave similarly on
\(|\Delta|\le L\) and diverge at \(4L\). Consequently,

\[
\text{better }\Phi_\Omega(D_{4L})
\not\Rightarrow
\text{the model learned the required routing algorithm at }4L.
\]

This is the probability-to-capability gap seen at 32K. The missing information
is not merely a better basis; it is supervision that makes the correct remote
source causally determine the generated answer at target phases.

### 3.3 Four times more context also increases competition

Even if the correct-source logit is phase-stable, increasing the effective
number of distractors by approximately four introduces a softmax competition
penalty on the order of

\[
\log 4.
\]

This is only a calibration heuristic, not a theorem for transformer attention,
but it explains why stable NLL and a visible remote-source signal may still
leave the correct answer far from top-1. A 4× curriculum must cover both
target phases and hard-negative density.

## 4. Why full EVQ hurts 8K

There are two distinct mechanisms and they must not be conflated.

### 4.1 Allocation trade-off from initialization

With finite \(K\), moving nodes toward long-period behavior reduces resources
elsewhere. The from-scratch OLMo result shows a smaller in-window cost even
when all weights co-adapt from step 1. No static schedule has been empirically
best at every length: Cosh, exponential, and an attention-derived two-band
shape exchange rank as deployment length changes.

Thus a fixed global schedule naturally defines a Pareto frontier:

\[
\min_\Omega
\left(
R_{8K}(\Omega),R_{16K}(\Omega),R_{32K}(\Omega)
\right),
\]

not one universal optimum.

### 4.2 Mature-model coordinate transplant

For a pretrained model, replacing \(\omega_k\) changes the bilinear form
\(R(\omega_k\Delta)\). Except for permutations, sign aliases, and integer
\(2\pi\) aliases, no position-independent Q/K maps can exactly preserve that
form at every relative position after changing the frequency multiset.

The current Llama parent artifacts make the size of the intervention explicit.
The complete serialized artifact hashes are
`09fab0f4da1c96bf31cf5d54dd45a935dc63f1ef8d7b1669dd5361087e4bb794`
for Native and
`49b205926ae989cdf973ceaa612e0dab937bf21fd63f095f72485ba3d4a4bacf`
for EVQ. All 64 Native frequencies differ from the EVQ table. At 8K, the
maximum unwrapped phase displacement is about 1,030 radians; 49/64 pairs
exceed 0.5 radians. The mean geometry-only phase-alignment cosine is 0.236 at
8K. This is not a measured attention similarity, but it proves that the full
transplant is not a small in-window perturbation.

LoRA must therefore spend capacity both on coordinate repair and on the new
task. This explains why natural-text NLL can recover while pre-existing task
circuits are not reliably preserved.

## 5. A phase-budget window exists for 8K no-harm and 32K change

Let \(\delta\omega_k=\omega_k'-\omega_k^{N}\). A useful protected pair should
satisfy

\[
|\delta\omega_k|L\le\epsilon
\quad\text{and}\quad
|\delta\omega_k|(4L)\ge\eta.
\]

This interval is nonempty whenever \(\eta\le4\epsilon\). For example,
\(\epsilon=0.5\) and \(\eta=1\) permit a pair to be a small perturbation at
8K and an order-one phase intervention at 32K.

The frozen Llama frequency artifacts contain exactly such a tail. Preserve
the first \(c\) high/medium-frequency pairs as Native and replace only the
lowest-frequency tail with the corresponding EVQ values:

\[
\omega_k^{H(c)}
=
\begin{cases}
\omega_k^N,&k<c,\\
\omega_k^E,&k\ge c.
\end{cases}
\]

Pure geometry gives:

| Cutoff \(c\) | EVQ tail pairs | 8K max phase shift | 8K pairs above 0.5 rad | 32K max phase shift | 32K pairs above 0.5 rad | alignment cosine 8K / 32K |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 46 | 18 | 1.154 | 3 | 4.616 | 8 | 0.9777 / 0.8700 |
| 48 | 16 | 0.660 | 1 | 2.641 | 6 | 0.9926 / 0.9174 |
| 50 | 14 | 0.366 | 0 | 1.466 | 4 | 0.9978 / 0.9685 |

The \(c=48\) row is the clearest initial bracket: 75% of pairs are exactly
Native, only one pair moves by more than 0.5 rad at 8K, and six pairs undergo
order-one displacement at 32K.

This calculation proves **geometric feasibility only**. It does not prove that
the changed tail carries useful content, that the model will allocate routing
to it, or that it will beat a Native/YaRN control. Historical hybrid receipts
cannot answer this question because an aliasing bug caused the active tensor
to differ from the declared hybrid. A future hybrid must clone both endpoint
tables and verify its active hash before GPU execution.

## 6. Three solution levels

### 6.1 Level A: unchanged submitted EVQ, better supervision

Keep the full EVQ table and change only the continuation curriculum:

1. all physical sequences remain at 8K;
2. the same content appears at 1×, 2×, and 4× virtual source-query gaps;
3. local positions inside source, distractor, query, and answer blocks remain
   consecutive;
4. only the gap between blocks is enlarged in `position_ids`;
5. answer-only and source-counterfactual objectives prevent ordinary local LM
   loss from hiding failure to use the remote source;
6. natural and 8K task replay enforce a no-harm objective.

This is still combined EVQ-plus-adaptation evidence, not pure Cosh
attribution. It is the smallest way to test whether missing target-phase
supervision explains the current 32K zero.

It does **not** guarantee 8K parity because the full table remains a large
coordinate change.

### 6.2 Level B: Native-protected tail EVQ

Start from the Native parent, retain the first 46/48/50 pairs, and expose only
the EVQ tail to the same paired phase curriculum. This directly targets the
phase-budget window in Section 5.

The scientific question is:

> Can a small low-frequency EVQ subspace learn 32K routing while the unchanged
> Native majority preserves 8K behavior?

This is the most plausible single-table route to simultaneous 8K retention and
4× improvement. It is a new hybrid method, not the submitted raw EVQ, and
must be reported as such.

Only one cutoff should receive a complete run. A cheap no-training check may
compare 46/48/50 geometry, but the initial trained candidate should be
\(c=48\), selected before seeing task outcomes.

### 6.3 Level C: exact function-preserving Native plus EVQ residual

If 8K behavior must be guaranteed at initialization, a single replaced
frequency table is the wrong model class. Use a length-conditioned score
interpolation

\[
S_h
=(1-\gamma_h(\ell))S_h^N+\gamma_h(\ell)S_h^E
=S_h^N+\gamma_h(\ell)(S_h^E-S_h^N),
\]

or equivalently a zero-initialized EVQ residual. At
\(\gamma_h(8K)=0\), the model is exactly Native at 8K; the EVQ branch can be
activated for longer contexts. A length-conditioned table/profile switch is
the cheaper deployment variant.

That equality is literal only if the complete Native branch—including its
Q/K/V/O projections—is frozen and trainable updates live in the gated residual
or in a separate long-context profile. A shared LoRA update applied to the
Native branch would break exact function preservation even when
\(\gamma_h(8K)=0\).

This is the strongest route to a hard no-harm property, but it changes the
attention parameterization and may complicate FlashAttention. It is EVQ-v2
research, not rebuttal evidence for the submitted fixed table.

A two-profile deployment—Native-LoRA at no more than 8K and EVQ-LoRA above
8K—is an even lower-risk envelope once the matched Native run is complete.
It is not one universal model and must not be described as raw EVQ dominance.

## 7. Training objective for 4× capability

For each semantic example, construct positive and source-swapped negative
versions at gaps \(g\in\{1,2,4\}\). Let \(y\) be the full answer token sequence.
Use

\[
\mathcal L
=
\lambda_1\mathcal L_{\mathrm{ans}}^{1\times}
+\lambda_2\mathcal L_{\mathrm{ans}}^{2\times}
+\lambda_4\mathcal L_{\mathrm{ans}}^{4\times}
+\lambda_{\mathrm{cf}}\mathcal L_{\mathrm{cf}}
+\lambda_{\mathrm{inv}}\mathcal L_{\mathrm{gap}}
+\lambda_{\mathrm{ret}}\mathcal L_{\mathrm{retain}}.
\]

The terms are:

\[
\mathcal L_{\mathrm{ans}}^{g}
=-\log p_\theta(y\mid x,g),
\]

\[
\mathcal L_{\mathrm{cf}}
=
\operatorname{softplus}
\left[
m-
\left(
\log p_\theta(y\mid x^+,g)
-
\log p_\theta(y\mid x^-,g)
\right)
\right],
\]

and

\[
\mathcal L_{\mathrm{gap}}
=
\operatorname{Var}_{g\in\{1,2,4\}}
\left[
\frac{1}{|y|}\log p_\theta(y\mid x,g)
\right].
\]

The margin should be pre-registered. A reasonable initial bracket is
\(m=m_0+\log 4\), reflecting the 4× distractor increase, but it must be
validated as a training hyperparameter rather than presented as an attention
theorem.

`retain` should use two sources:

- 8K rows on which the frozen Native parent is already correct;
- natural 4K–8K instruction rows.

For mature-model retention, response-token KL to the frozen Native model may
be added at 1× only. Prior sparse-KL experiments failed during full frequency
morphing, so KL is a guardrail, not the proposed mechanism. The mechanism is
target-phase and source-counterfactual supervision.

## 8. Virtual gaps without 32K backward passes

Uniformly multiplying every token position by four destroys local language
geometry. Instead, keep positions consecutive within blocks and insert a
virtual offset between the source region and the query:

\[
p_t=
\begin{cases}
t,&t<b,\\
t+\Delta_{\mathrm{virtual}},&t\ge b.
\end{cases}
\]

The sequence still contains at most 8K physical tokens and uses the ordinary
causal order, while source-query RoPE phases reach 16K or 32K. The training
mixture must also densify hard distractors, because virtual distance alone
does not reproduce the number of competitors in a physical 32K sequence.

The same key/value/query instance must appear at multiple gaps within the same
optimization window. Otherwise the model may learn independent template or
position-bin shortcuts rather than a distance-stable binding.

This technique trains target phases; final evidence must still use true
physical 16K/32K sequences.

## 9. Pre-registered decision path

### Gate 0: finish the matched Native-LoRA arm

Required outputs:

- 8K/16K/32K natural-text NLL;
- complete 13-task official and strict RULER scores;
- the same frozen evaluation rows as EVQ;
- final adapter, frequency, data, and output hashes.

Interpretation:

- Native wins 8K and EVQ wins 16K while both fail 32K: confirmed
  range/capability trade-off; proceed only if 4× evidence is essential.
- Native also wins 16K: the current EVQ curriculum has no capability advantage;
  stop method extension.
- EVQ matches Native at 8K and wins 16K: no hybrid is needed; test only the
  phase-covered continuation.

### Gate 1: one unchanged-EVQ phase-covered screen

Use the current EVQ adapter, one fixed seed, physical 8K only, and a small
pre-registered set of NIAH, FWE, VT, and CWE rows. Evaluate all at true
8K/16K/32K.

Continue only if:

- 8K official macro is no more than two percentage points below matched
  Native-LoRA and temporal NLL is no more than 0.05 worse;
- 16K is no worse than the existing EVQ arm;
- at 32K, at least two task families are nonzero or the four-task macro is at
  least 5%.

These are proposed design thresholds, not observed significance statements.

### Gate 2: Native-protected tail EVQ only if Gate 1 misses 8K no-harm

Run only cutoff \(c=48\), using the same data, row order, seed, token budget,
optimizer, and evaluation. The scientific contrast is full EVQ versus
Native-protected tail EVQ; a matched Native arm already exists.

Continue to \(n=100\) only if:

- 8K reaches or exceeds matched Native-LoRA within the declared tolerance;
- 16K retains a directional advantage;
- 32K has nonzero strict autoregressive exact on held-out rows.

Do not sweep cutoffs after seeing task results.

### Gate 3: exact residual architecture only as future work

If both fixed-table arms fail the no-harm/capability conjunction, the evidence
supports a model-class limitation: a single shared frequency table cannot
simultaneously preserve the mature Native circuit and supply the needed long
phase substrate under the tested LoRA budget. The next step is a
function-preserving residual or length-conditioned profile, not more rank,
more \(\tau\), or more examples from one task.

## 10. Falsifiable predictions

The analysis predicts:

1. Matched Native-LoRA will learn the 8K RULER mixture at least as fast as full
   EVQ and will likely retain a higher 8K official score.
2. Full EVQ will retain a relative 16K/32K natural-text NLL advantage.
3. Ordinary 8K task-family training without target-phase coverage will remain
   weak or zero at 32K.
4. Phase-covered counterfactual training can improve 32K source dependence
   before it improves broad 32K task accuracy.
5. Tail EVQ will preserve 8K better than full EVQ but may have weaker 32K NLL;
   its success depends on whether LoRA routes semantic binding through the
   changed low-frequency pairs.
6. If tail EVQ succeeds only on trained task families, the bottleneck is still
   task computation, not frequency geometry.
7. If full EVQ and tail EVQ both improve 32K NLL but remain zero on generation,
   the 4× limit is readout/algorithmic and frequency changes should stop.

## 11. Reviewer and paper boundary

The submitted EVQ claim remains training-time frequency allocation. Neither
virtual-gap training, a Native-protected tail, nor a residual dual branch can
be retroactively presented as the submitted method.

For rebuttal:

- the current matched Native/EVQ Llama experiment may become supporting
  mature-model task-adaptation evidence after raw promotion;
- a phase-covered full-EVQ continuation would be combined
  EVQ-plus-adaptation evidence;
- a tail or residual hybrid belongs to future method development;
- PPL, causal source use, correct-token rank, strict generation, and task
  accuracy must remain separate endpoints.

The correct theoretical conclusion today is:

> EVQ already supplies a useful long-window positional substrate, but a full
> fixed-table transplant spends in-window capacity and does not identify 4×
> task behavior from 8K supervision. Four-times capability requires explicit
> target-phase and source-dependent training; simultaneous 8K no-harm most
> plausibly requires protecting the Native frequency subspace or retaining an
> exact Native branch.

## 12. Evidence owners

- Mature Llama probability/routing ladder:
  `EVQ_8B_ADAPTATION_EVIDENCE_20260724.md`.
- Mature OLMo matched 4K-only conversion:
  `OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md`.
- Full-transplant compensation obstruction:
  `OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md`.
- Complete RULER positive and negative ledger:
  `OLMO2_1B_OVERNIGHT_EXPERIMENT_SUMMARY_20260726.md`.
- Training-time schedule objective boundary:
  `EVQ_TRUE_OBJECTIVE_ULTRA_AUDIT.md`.
- Reviewer-facing selection and claim limits:
  `../01_REBUTTAL_PLAYBOOK.md`.
