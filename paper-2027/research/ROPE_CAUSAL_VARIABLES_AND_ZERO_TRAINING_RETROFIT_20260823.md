# RoPE causal variables and the zero-training retrofit

- **Date:** 2026-08-23
- **Status:** authoritative internal theory/narrative foundation; not manuscript
  prose and not a numerical evidence owner
- **Purpose:** give future authors and agents one global vocabulary for the
  existing paper, the new mature-checkpoint experiments, and the practical
  zero-training replacement
- **Supersedes only:** narratives that treat scalar base, exponent allocation,
  support extension, attention gain, and serving policy as either one
  indistinguishable knob or five unrelated methods
- **Does not supersede:** theorem proofs, experiment owners, receipts, or the
  active manuscript

## 0. Decision first

The theory should be unified at the level of the **physical RoPE object**, but
not at the level of the **causal estimand**.

The physical object used by ordinary RoPE attention is an ordered frequency
tensor

\[
\Omega=(\omega_0,\ldots,\omega_{K-1}),
\qquad \omega_0>\cdots>\omega_{K-1}>0.
\]

Once a checkpoint's Native base \(\theta_0>1\) is fixed as a coordinate
convention, the same tensor can be written uniquely as an exponent curve

\[
e_k=-\frac{\log\omega_k}{\log\theta_0},
\qquad \omega_k=\theta_0^{-e_k}.
\]

This does **not** make every intervention scientifically identical. A scalar
base rule, a nonlinear exponent rule, a post-training vector transport, a
learned per-band table, an attention-temperature change, and a serving router
act at different stages and impose different restrictions. Those restrictions
define the causal question.

The global story is therefore:

> A finite head receives only \(K\) rotary pairs, so the exponent curve decides
> how that finite positional basis spends its spectral budget. The paper
> identifies interior allocation during training with support fixed, explains
> its full sin/cos redundancy and weight co-adaptation, and gives EVQ-Cosh as
> one closed-form zero-learned-parameter construction. The new frozen-checkpoint
> study asks a different question: whether a deterministic, zero-training
> replacement can exploit the same allocation degree of freedom after
> pretraining while preserving the exact Native short path. Same-support and
> amplitude controls decompose that complete replacement without pretending
> that the estimands are the same.

This is one research programme, not one forced mechanism.

## 1. Numerical representation is not causal identity

### 1.1 The table is the physical object

For one head and relative displacement \(D\), standard RoPE contributes

\[
\ell(D)=\mathbf q^\top R_\Omega(D)\mathbf k
       =\sum_{j=0}^{K-1} A_j\cos(\omega_jD+\psi_j).
\]

The frequency tensor \(\Omega\) selects the positional basis. Trained Q/K
weights determine the content-dependent amplitudes and phases
\((A_j,\psi_j)\). If rotary cos/sin outputs are multiplied by amplitude \(c\)
on both Q and K, the corresponding logit multiplier is \(\alpha=c^2\).
These objects interact in the final output, but they are not the same
intervention.

### 1.2 Base and exponent have a representation symmetry

If both \(\theta>1\) and \(e_k\) are allowed to vary freely, then for any
\(c>0\),

\[
\theta^{-e_k}=(\theta^c)^{-e_k/c}.
\]

Thus a realised frequency tensor alone cannot reveal which textual pair
"base/exponent" generated it. This algebraic symmetry is only a warning about
parameterisation. It must **not** be used to erase the scientific distinction
between the following restricted interventions:

- a scalar-base family changes one scalar while retaining a prescribed
  geometric exponent rule;
- an exponent-allocation family changes the index-to-exponent map while the
  nominal base and sampled support are controlled;
- a whole-vector retrofit directly transforms the realised Native tensor after
  the weights have already co-adapted to it.

The exact-range experiment distinguishes the first two by intervention, not by
claiming that \(\theta\) and \(e\) are metaphysically separate. With sampled
endpoints and log-span fixed, no scalar base inside the geometric family can
move only the interior \(K-2\) frequencies.

### 1.3 Native-base coordinates keep the engineering description honest

For a mature checkpoint, use its verified Native base \(\theta_0\) only as a
coordinate gauge and derive \(e_k\) from the actual Native frequency buffer.
Do not hard-code the OLMo or Qwen base, pair count, or Native length.

A scalar base change is then a restricted affine/geometric exponent curve. For
standard \(e_k^{\rm nat}=k/K\), changing \(\theta_0\) to \(\theta'\) gives

\[
e'_k=\frac{\log\theta'}{\log\theta_0}e_k^{\rm nat},
\]

which keeps the normalised interior order geometric. A nonlinear \(e'_k\)
cannot be produced by that one-parameter family.

The rule is:

> Use \(\Omega\) or \(e\) to describe what code installs; use the permitted
> intervention family and stage to say what caused what.

## 2. The causal ledger

Any paper claim or experiment must name all variables below. Omitting the
stage or the held-fixed set turns distinct evidence into an invalid pooled
story.

| Variable | Mathematical object | What it controls | Typical intervention |
| --- | --- | --- | --- |
| realised RoPE table | \(\Omega\) or \(e\) | positional basis | initialise, learn, or replace the frequency tensor |
| sampled support | \(a=x_0\), \(R=x_{K-1}-x_0\), \(x_k=-\log\omega_k\) | fastest anchor and log-frequency span | PI translation, scalar-base/range dilation, endpoint pinning |
| interior allocation | \(z_k=(x_k-a)/R\) | where the finite \(K\) samples lie inside support | geometric, Cosh, learned, ramp, or derived profile |
| attention amplitude | rotary amplitude \(c\), logit gain \(\alpha=c^2\) | logit scale and softmax concentration | YaRN-style `mscale` or another fixed gain |
| content map | trained weights \(W\), optionally an adapter | coefficients used in the installed basis | from-scratch training, continued training, LoRA, frozen crossing |
| position/operator map | \(p\mapsto f(p)\) and attention routing | which relative positions are compared under which view | PI, grouped positions, bifocal or custom attention |
| session policy | \(\pi(\text{request})\) | which already-frozen operator is selected | exact Native or one frozen long profile before prefill |
| model profile | \((L_{\rm native},K,\Omega_{\rm native})\) | model-relative construction inputs | observed covariate, not a task label or tunable target |

The identity

\[
x_k=a+Rz_k,\qquad z_0=0,\quad z_{K-1}=1
\]

is a **causal accounting system**, not an assertion that an implementation
must execute "support extension" and then "allocation" as two modules. Any
single table has all three coordinates simultaneously. Controls hold two
coordinates fixed to identify the third.

## 3. Where the main method families sit

The following map is deliberately nonexclusive. It classifies what is
parameterised, when it acts, and what evidence it can own; it does not claim
that different methods can never emit numerically similar tensors.

| Family | Primary action and stage | Coordinates touched | What it establishes or owns |
| --- | --- | --- | --- |
| Native / Geo | fixed geometric table before training | fixed support and geometric \(z\) | reference substrate; `Native` and `Geo` remain distinct locked terms |
| scalar-base rules, including the evaluated FMRoPE rule | choose or retarget a scalar base while keeping the exponent order geometric | primarily support, geometric \(z\) retained | base/range selection; not pure interior-allocation evidence |
| position interpolation | scale positions, equivalently translate all log frequencies for the simple linear case | support translation; \(R,z\) retained | phase-range transport to a declared scale |
| YaRN | per-band NTK-by-parts frequency transform plus attention temperature | support, interior \(z\), and \(c\)/\(\alpha\) | an inference/extension operator; its ramp may emit a non-geometric table |
| LongRoPE / LongRoPE2 | search per-dimension rescaling and other extension choices | whole realised spectrum, often support and \(z\) together | searched mature-model extension, not a pure allocation control |
| LeRoPE | learn one scale per band jointly with model training | \(e\) and \(W\) jointly | learned-table value; Fixed LeRoPE partially separates final table from joint dynamics |
| AdaRoPE | learn finer head/dimension frequency and scale structure | headwise \(e\), gain, and \(W\) | learned heterogeneous operator; broader than one shared fixed table |
| EVQ-Cosh | set a closed-form nonlinear exponent map before training | training-time \(z\), with support controlled in exact-range | zero-learned-parameter construction and reproducible intervention on the allocation axis |
| current zero-training replacement | deterministically transform a released Native tensor, add a fixed long-profile gain, and route once per session | full long-table intervention plus \(c\)/\(\alpha\) and \(\pi\); controls separately identify \(z\) | mature frozen-checkpoint retrofit and deployment case study |
| Jet-Long | dynamically remap remote positions with bifocal attention and cache correction | position/operator map and routing, not merely one static table | zero-training dynamic extension with a custom multi-view/kernel path |

Two boundaries follow.

First, "the output table is non-geometric" is too broad to carry novelty:
YaRN, LongRoPE, LeRoPE, AdaRoPE, Resonance RoPE, CoPE, MrRoPE, and other methods
can alter per-band behaviour. The paper's defensible novelty is the explicit
fixed-support identification of finite interior allocation, the accompanying
full-basis theory, and a closed-form zero-learned-parameter point on that axis.

Second, the new replacement should not be sold as a new YaRN family. Its value
is that it turns the paper's allocation analysis into a deterministic mature
checkpoint intervention, and that same-support controls show the outcome is
not explained by endpoint extension alone. Its nearest label-free ramp matches
the detailed derived profile in the completed gates, so profile-detail novelty
is not supported.

## 4. The evidence chain, without splicing estimands

| Scientific question | Held fixed | Changed | Canonical evidence | Maximum conclusion |
| --- | --- | --- | --- | --- |
| Does interior allocation matter during training? | sampled endpoints, log-span, recipe, seed pairing | \(K-2\) interior frequencies and the weights learned with them | three-seed 151.9M exact-range; M4 direction check | \(z\) is a separately identifiable training-time variable |
| Does static basis geometry equal task quality? | frozen weights and evaluation | runtime table | 50M rank/PPL counterexample | no; static redundancy is not an LM-quality predictor |
| Do weights co-adapt to the table? | factorial protocol | trained weights \(\times\) runtime table | 50M 2x2 and two-seed 151.9M crossing | table/weight compatibility is strong and exact arbitrary transplantation is obstructed |
| Does attention gain act independently of the table? | checkpoint and tasks | table \(\times\) gain | mature 2x2 owner | no in the tested cells; softmax couples them and the interaction is large |
| Does interior allocation still matter with mature weights frozen? | support, amplitude, checkpoint, rows, decoding | geometric versus non-geometric \(z\) | OLMo/Qwen same-support owner | yes in the stated frozen RULER protocols |
| Is the detailed derived uniqueness curve necessary? | same frozen controls | derived curve versus nearest label-free ramp | same-support owner | no evidence of necessity; the ramp matches under the registered gate |
| Can a no-training deployment preserve the Native path? | checkpoint and request | session route | session-policy owner | exact OLMo Native short-route parity and one frozen long route are demonstrated |
| Is the replacement universally better on natural tasks? | — | — | Qasper, 2Wiki, formal LongBench, PG-19 | no; natural-task effects are heterogeneous and model/task generality remains bounded |

The first and fifth rows use the same coordinate language but are not the same
estimand. One retrains weights under different \(z\); the other freezes a
mature checkpoint and changes its runtime table. The crossing experiments are
the bridge: they show why training history must remain explicit.

## 5. The zero-training replacement as a complete intervention

### 5.1 What the code actually does

Let a model profile provide the verified Native tensor
\(\omega_k^{\rm nat}\), Native window \(L_{\rm native}\), and pair count \(K\).
A task-label-free analysis produces movement coefficients \(m_k\in[0,1]\).
For a frozen long factor \(s>1\), the completed long table is

\[
\omega_k^{\rm long}
=\omega_k^{\rm nat}\left[(1-m_k)+\frac{m_k}{s}\right],
\]

with the fast and slow endpoints explicitly pinned. In the Native-base
coordinate this same operation is exactly

\[
e_k^{\rm long}
=e_k^{\rm nat}
-\frac{\log[(1-m_k)+m_k/s]}{\log\theta_0}.
\]

The implementation installs the realised frequency tensor directly. It does
not replace `rope_theta` with one new scalar. The code option named
`exponent=2` shapes \(m_k=(1-\tilde u_k)^2\); it is not the RoPE exponent curve
\(e_k\).

The long branch also uses one frozen attention amplitude. The serving rule is

```text
required_tokens = prefill_tokens + max_new_tokens
required_tokens <= L_native  -> call the exact Native rotary module
required_tokens >  L_native  -> install the one frozen long profile
```

The route is selected before prefill and remains fixed for the request's
entire KV-cache lifetime.

### 5.2 What is unified and what is decomposed

The long table is one deterministic transformation, not an implementation that
first changes a base and then separately fills an interior. Nevertheless, its
causal effect contains several possible sources:

1. the slow endpoint is extended;
2. interior channels are placed non-geometrically;
3. attention amplitude is changed;
4. the serving policy preserves an exact Native branch.

The completed controls separate them:

- **same-support geometric:** identical fast/slow endpoints and amplitude, but
  a log-linear interior; algebraically the corresponding scalar-base/NTK-aware
  control for the standard geometric Native tables used here;
- **nearest label-free ramp:** identical endpoints and amplitude, with the
  closest discrete linear-ramp profile;
- **derived table:** identical endpoints and amplitude, with the full
  model-relative profile;
- **frequency \(\times\) gain 2x2:** separates the table and attention-amplitude
  main effects and their softmax interaction;
- **Native-route parity:** isolates the serving policy's short-context
  preservation from long-table quality.

These owners are not one pooled factorial. The same-support study evaluates
the final factor-four table controls, the frequency-by-gain 2x2 owns its own
earlier matched cells, and route parity owns the serving branch. Together they
answer the component questions; they do not justify an additive numerical
attribution of every point in the final policy's score.

This is the correct use of causal decomposition: analyse a complete practical
method without pretending that the method itself is a bag of independently
deployed modules.

### 5.3 Why this result is valuable

The zero-training replacement adds a qualitatively different asset to the
paper's existing foundation:

- **no checkpoint retraining:** zero optimisation steps and zero training
  tokens;
- **no learned positional parameters:** the table and gain are deterministic;
- **constructive short retention:** the Native branch directly calls the
  checkpoint's original rotary module;
- **standard long attention path:** the long branch changes a static
  inverse-frequency tensor and amplitude without changing tensor shapes;
- **ordinary FlashAttention compatibility:** completed runs used the standard
  rotary-before-attention path rather than a quadratic fallback;
- **ordinary KV-cache semantics:** keys remain valid because the selected table
  never changes after prefill;
- **model-relative inputs:** the construction reads each checkpoint's own
  Native tensor/window rather than assuming every model is OLMo-4K;
- **two mature checkpoints:** the fixed-support direction appears in OLMo and
  Qwen, which have different Native windows and bases.

It also has explicit limits:

- it is not a dynamic within-session operator; changing the table after keys
  are cached would mix coordinate systems;
- it still uses a request's known prefill and maximum-generation budget to
  choose the route;
- only one frozen long profile is currently retained, not a proof that \(s=4\)
  is universally optimal;
- natural-document evidence is strong on full Qasper, null/mixed on other
  endpoints, and absent for the Qwen checkpoint;
- the detailed uniqueness curve is not identified beyond its much simpler
  nearest ramp;
- RULER remains task-family adaptation, not unseen natural-task transfer.

### 5.4 Relationship to Jet-Long and other zero-shot work

Exact short-path preservation and zero-training deployment are not unique
systems claims. Jet-Long also recovers the base model inside its Native window,
but does so through a dynamic bifocal position map, inclusion--exclusion
attention, and on-the-fly cache correction. The current replacement instead
chooses one standard static RoPE table before prefill and therefore keeps the
ordinary FlashAttention/KV-cache contract, at the cost of not adapting the
profile while generation grows.

That distinction is useful rather than threatening:

- Jet-Long owns dynamic position/operator remapping and a specialised efficient
  kernel;
- this work owns fixed-support allocation identification and tests whether a
  deterministic table replacement can realise that variable in a frozen
  checkpoint;
- the session router is an engineering vehicle, not the central novelty claim.

## 6. How EVQ and the mature replacement are related

They share a scientific question but not a protocol identity.

**EVQ-Cosh** asks before training: if a finite geometric exponent grid is
redundant, can a stated variational surrogate produce a closed-form nonlinear
allocation? The model then trains with that table, allowing full co-adaptation.
In the exact-range arm, endpoint normalisation fixes support and isolates the
interior shape.

**The mature replacement** asks after training: can a released Native model be
given a deterministic long profile without optimisation, and which part of
that replacement matters when weights are frozen? It operates on the realised
Native tensor and combines a long table, gain, and session-static routing.

The valid bridge is:

> The from-training experiments establish that finite interior allocation is
> a real design variable; the crossings establish that weights learn the
> installed coordinate system; the frozen same-support controls establish that
> the allocation coordinate remains consequential after pretraining; and the
> session policy shows one zero-training way to deploy that observation.

The invalid bridges are:

- calling the mature profile "EVQ" when it is not the EVQ-Cosh quantile table;
- presenting the frozen result as another training seed for exact-range;
- using zero-training deployment success to claim EVQ-Cosh is universally
  optimal;
- describing a nearest-ramp match as proof of a uniquely necessary uniqueness
  curve.

## 7. Reviewer-facing story

### 7.1 The one thing to remember

> RoPE gives each head a finite spectral budget. Range tells us where the
> spectrum begins and ends; interior allocation tells us how the finite pairs
> are spent. We causally identify allocation with range fixed, explain why
> redundant slow subspaces and trained weights make that choice consequential,
> and show that the same variable can be used in a released checkpoint through
> a deterministic zero-training replacement while retaining the exact Native
> short path.

### 7.2 The evidence order

1. **Problem:** a scalar base silently couples spectral range to geometric
   placement of a finite number of pairs.
2. **Theory:** each frequency is a two-dimensional sin/cos subspace; slow bands
   can collapse toward a redundant positional subspace.
3. **Identification:** exact-range changes only interior placement and changes
   trained behaviour across three seeds.
4. **Construction:** EVQ-Cosh is one closed-form, zero-learned-parameter point,
   not a universal optimum.
5. **Co-adaptation:** weights learn coefficients in the installed basis, and
   exact arbitrary frozen transplantation is obstructed.
6. **Mature corollary:** same-support controls show that interior placement also
   matters with OLMo/Qwen weights frozen.
7. **Practical corollary:** a binary Native/frozen-long session policy uses this
   without retraining and retains standard cache semantics.

Do not open with "we beat YaRN." YaRN is simultaneously a related method, an
amplitude/frequency control, and a useful published reference point. The
scientific result is the fixed-support causal contrast; the practical result is
the zero-training replacement; the ranking is supporting evidence.

### 7.3 The causal figure/table the story needs if promoted

A single compact panel should expose, rather than hide, the decomposition:

```text
verified Native profile
   (Omega_native, L_native, K)
             |
             v
 label-free long-table construction ------> fixed rotary amplitude c
             |                                  |
             +---------- standard RoPE + softmax --------> outcome
             |
      decompose only for controls
        support (a,R) / interior z

request budget -- session policy pi --> exact Native OR frozen long table
trained weights W remain fixed throughout the mature comparison
```

The matching result table should answer four questions, in order: geometric
same support, nearest ramp, full derived profile, and Native-route/natural-data
retention. A long baseline leaderboard without the held-fixed columns would
weaken the result.

## 8. Claims, inferences, and open questions

### Established mathematical facts

- \(\Omega\leftrightarrow e\) is one-to-one after fixing \(\theta_0\).
- the `base/exponent` pair has a representation symmetry if both are free;
- \(x=a+Rz\) is an exact decomposition of any ordered nondegenerate table;
- scalar-base change with the standard exponent order is a restricted
  geometric/affine exponent family;
- the full-RoPE stable-rank identity, low-frequency collapse, and exact static
  transplant obstruction retain their canonical owners.

### Established empirical facts

- training-time \(z\) matters at fixed support across three 151.9M seeds;
- static rank alone does not predict frozen-model loss;
- weights and runtime tables show strong crossings at 50M and 151.9M;
- table and gain interact in the tested mature-model cells;
- same-support geometric versus non-geometric tables separate strongly on
  OLMo and directionally on Qwen under the completed protocols;
- the nearest label-free ramp is not separated from the detailed derived
  profile;
- the OLMo binary policy has exact Native short-route parity, strong RULER and
  full-Qasper results, and heterogeneous natural-task outcomes.

### Inferences, not proofs

- the derived model-relative split is a useful operating prior;
- moving extension budget away from redundant slow bands may explain part of
  the mature-checkpoint gain;
- the zero-training case study can strengthen the ICLR story if presented as a
  mature corollary rather than a new operator family.

### Open

- whether the split transfers to additional architectures/pair counts;
- whether a natural Qwen task reproduces the fixed-support direction;
- whether another label-free allocation beats the nearest ramp;
- whether the zero-training replacement improves a broad natural-task
  population;
- how to derive attention gain from held-out model statistics rather than an
  inherited coefficient;
- whether a simple adapter can improve the deterministic replacement without
  losing Native retention.

## 9. Terminology contract

Use:

- **realised frequency tensor** or **exponent curve** for what is installed;
- **scalar-base/geometric family** for the restricted one-parameter control;
- **sampled support** \((a,R)\) and **normalised interior allocation** \(z\) for
  causal decomposition;
- **training-time allocation** for exact-range/EVQ evidence;
- **frozen same-support allocation intervention** for the mature controls;
- **zero-training Native/frozen-long session policy** for the complete
  practical method;
- **attention amplitude** for `mscale`, kept separate from the frequency table;
- **zero learned positional parameters** for EVQ-Cosh, and **zero training** for
  the mature replacement.

Avoid:

- "base and exponent are physically independent";
- "base and exponent cannot be distinguished" without naming the intervention
  family;
- "first extend support, then redistribute" as an implementation description;
- "new YaRN" or "better YaRN" as the contribution;
- "the uniqueness curve is necessary";
- "target-free" when the serving route actually reads the observed request
  budget;
- "zero parameter" without saying whether it means no learned positional
  parameters or no checkpoint training.

## 10. Authority and reading order

This file owns notation and narrative grammar only. Read numerical and theorem
claims from their canonical owners:

1. [`EXACT_RANGE_151M_3SEED_RESULT_20260820.md`](EXACT_RANGE_151M_3SEED_RESULT_20260820.md)
   for training-time fixed-support identification;
2. [`FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`](FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md)
   for subspace theory, co-adaptation, base controls, and theorem routing;
3. [`attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`](attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md)
   for mature same-support controls and the 151.9M frozen crossing;
4. [`attention-aware-retrofit/results/JOINT_MECHANISM_REPORT_20260822.md`](attention-aware-retrofit/results/JOINT_MECHANISM_REPORT_20260822.md)
   for the frequency-by-gain 2x2 and its limitations;
5. [`attention-aware-retrofit/results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md`](attention-aware-retrofit/results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md)
   for the practical zero-training policy and natural-context evidence;
6. primary papers for related-work facts; external-model reviews remain
   non-canonical analysis inputs.

Primary anchors for the distinctions used here:

- [YaRN](https://arxiv.org/abs/2309.00071) for the NTK-by-parts per-band
  transform, attention temperature, and dynamic-scaling distinction;
- [FMRoPE](https://openreview.net/forum?id=PR1PPxvG9Q) for scalar-base/frequency-
  band selection tied to context length;
- [LeRoPE](https://arxiv.org/abs/2607.10134) for learned per-band frequencies
  and the Fixed-LeRoPE separation between final-table value and joint dynamics;
- [Jet-Long](https://arxiv.org/abs/2607.07740) for a zero-training dynamic
  bifocal position/operator method with cache correction and a specialised
  fused path.

No result in this document is a new experiment, and no sentence here promotes
the frozen-checkpoint case study into the active manuscript.
