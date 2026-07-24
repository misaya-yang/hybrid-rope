# EVQ-Cosh Rebuttal Playbook

Last updated: 2026-07-25. Status: **internal authoring guide; not a response to paste verbatim**.
Task mode: `revise`. Package readiness: `needs_author_input` pending
three-seed raw promotion and promotion of the two missing full review sources.

This is the reviewer-facing decision ledger for Submission 11628. Formal
concerns are in `00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`; standalone reports
own protocol, numbers, uncertainty, and provenance. This file owns evidence
selection, ordering, safe wording, and the send gate.

Source boundary: the current checkout contains a payload-hashed full review
only for `R27bE` and an author-supplied, independently unverified AC
metareview. The additional review texts discussed in external author guidance
have not been promoted here. Until they are, route the FMRoPE/dead-frequency
issue under `AC.1`; do not attribute exact wording to “Reviewer 2” or invent a
stable reviewer ID.

## 1. The reviewer should understand this in 30 seconds

### AC-ready opening

> **We agree that FMRoPE is directly relevant and should have been cited and
> compared explicitly. This related-work omission does not, however, establish
> technical equivalence. The submitted paper already defines EVQ as a
> different optimization object: it asks where a fixed number of RoPE
> frequencies should be placed before training and derives a fixed,
> closed-form inverse-CDF grid with zero learned frequency parameters, rather
> than selecting or transporting a target spectral range. The paper does not
> claim novelty for the prior observation of dead or under-utilized channels.
> The new three-seed exact-range control turns this submitted distinction into
> a direct experiment: sampled extrema and log-span are identical, and only the
> 30 interior nodes differ. Cosh lowers fixed-range OOD NLL at 2x/4x/8x in all
> three training seeds, while target-retargeted FMRoPE remains stronger. The
> supported claim is a non-reducible training-time allocation variable, not
> replacement of or additive superiority over target-aware range transport.**

The one-sentence method hook is:

> **To our knowledge, EVQ-Cosh is the first explicit variational formulation
> of finite training-time frequency-grid allocation for standard RoPE, with a
> closed-form, zero-learned-parameter inverse-CDF construction.**

### Problem definition

A RoPE head has only \(K\) rotary frequency pairs, and the conventional
geometric schedule allocates them uniformly in log-frequency coordinates. The
general EVQ question is:

> Before model optimization begins, how should a finite number of frequency
> nodes be allocated inside the training-time grid?

This is more precise than “frequencies matter.” EVQ makes finite training-time
grid allocation an explicit optimization object. For the paper's stated convex
surrogate, the unique minimizer is a Cosh density; a closed-form inverse CDF
constructs the fixed table once at initialization. EVQ adds no learned
frequency parameters and performs no per-channel search.

To identify allocation separately from range, the exact-range control
additionally fixes the sampled highest frequency, sampled lowest frequency,
and log-frequency span, then changes only the \(K-2\) interior nodes. This
identification control is not a claim that submitted raw midpoint EVQ itself
keeps the sampled extrema and span fixed.

### What was already explicit in the submitted paper

The rebuttal must distinguish submitted claims from post-submission evidence:

| Submitted-paper anchor | What it already establishes | What it does not establish |
| --- | --- | --- |
| Introduction and contribution list | Training-time frequency allocation is a third design axis; EVQ is a closed-form, zero-learned-parameter inverse-CDF family; the operating rule is a basin default | A direct FMRoPE comparison |
| Related Work §2 | Base/range selection and post-hoc transport are separated from a pre-optimization training grid; EVQ holds the nominal base fixed and reallocates finite channels | That every range method changes only one scalar |
| Table 1 and Theory §3 | Conditional surrogate theorem, pure-tether branch choice, exact-family implementation, and empirical operating rule have different epistemic status | That the approximation chain is fully attributed in trained models |
| Appendix A.6 | The surrogate-derived EVQ schedule reduces exact-kernel collision in all 12 tested configurations by 24–92% and increases effective rank by 24–570% | Pure interior-shape identification at fixed sampled extrema/span |
| Appendix A.14 | A language-model-free collision-only diagnostic shows that non-geometric placement changes exact-kernel off-diagonal structure and that EVQ moves in the same collision-reducing direction as numerical search | That EVQ itself achieves every “−65%/−66%” annotation; those table annotations describe the oracle column |
| Evidence-tier table and supporting appendices | The empirical scope already includes 454M three-seed MHA, 432M three-seed MLA, 750M continued pretraining, 129M/382M video DiT, progressive training, QuALITY gold-answer NLL, and LLaMA-3-8B-Instruct LoRA | A clean controlled trajectory at or above 1B, or stable downstream capability conversion |
| Scope and limitations | EVQ is not a replacement for inference-time scaling and is not claimed to deliver universal downstream superiority | Production-scale or downstream closure |

Dead/under-utilized channels appear in the submitted supporting video mechanism
discussion, but discovering them is not one of the abstract/Introduction
contributions. The novelty defense must therefore say “not our claimed
novelty,” not “the submitted paper never discusses dead channels.”

The valid omission is narrow and should be conceded without qualification:
FMRoPE was not cited or directly compared. The new three-seed exact-range
experiment supplies that missing comparison; it does not introduce a post-hoc
identity for EVQ.

### Score-changing submitted-evidence correction matrix

This is the compact internal map for concerns that undercount evidence already
present in the submission. Use it to clarify the record, not to accuse a
reviewer:

| Negative reading to correct | Assessment | Response action | Submitted anchor | Remaining valid gap |
| --- | --- | --- | --- | --- |
| Surrogate proof, exact-kernel validation, and trained-model evidence were not separated | Overstates the absence | `CLARIFY_EXISTING` | Table 1, Theory §3, A.6, A.14, and §§4–5 | The map could be more prominent; trained-model attribution among Cosh, \(\tau\), and alternative schedules remained incomplete |
| Dead or under-utilized channels are the claimed novelty | Targets the wrong claim | `CLARIFY_EXISTING` | Introduction, contribution list, and Related Work already credit prior frequency specialization/allocation observations | Dead channels do appear as supporting video mechanism context and must not be described as absent from the paper |
| Allocation-versus-range is a rebuttal-stage redefinition | Not supported by the submitted text | `CLARIFY_EXISTING` + `ACCEPT_EXPERIMENT` | Introduction and §2 define pre-training allocation separately from base/range selection and post-hoc transport | FMRoPE was omitted and no direct matched comparison was submitted; the new exact-range experiment addresses this gap |
| The paper contains no larger, pretrained, cross-architecture, or cross-modal evidence | Too broad, but points to a real evidence-tier weakness | `PARTIAL` | Submitted 750M continuation, 432M MLA, 129M/382M video DiT, progressive training, QuALITY, and 8B LoRA | The strongest controlled multi-seed tier remains below 1B; 750M/8B/382M rows are supporting and capability conversion remains open |

The response posture is:

> The submitted evidence was broader and more explicitly stratified than the
> negative summary suggests. We will surface those anchors more clearly while
> accepting the remaining attribution, controlled-scale, and downstream
> capability gaps.

Do not turn this matrix into “the reviewers were wrong.” The score-changing
point is narrower: several heavy negative conclusions should be reassessed
after the already-submitted evidence is counted at its correct tier and the
genuinely missing FMRoPE control is added.

### Submitted wording that must be narrowed

The submission uses “orthogonal” for the shape/range distinction and an
appendix caption says the corrections are “additive.” The new
target-retargeted control does not support empirical additivity. Do not deny
the submitted wording. Use:

> In the submission, “orthogonal” described different intervention objects and
> stages. The new target-retargeted control shows that this structural
> distinction does not imply additive empirical gains. We therefore narrow the
> empirical statement to distinct optimization objects and stages.

### Rebuttal delta and decision summary

The three-training-seed exact-range control identifies this object directly.
Within each seed pair, Cosh and uniform-in-log FMRoPE have identical sampled
extrema, identical log-frequency span, and matched architecture, trainable
initialization, token order, optimizer, training budget, and frozen evaluation
anchors. Only the \(30\) interior frequencies differ. At fixed training range,
Cosh minus uniform FMRoPE is
\(-0.3159/-0.1949/-0.1674\) mean final-128-token NLL at 512/1K/2K, and all
three training-seed contrasts favor Cosh.

This result makes scalar base/range selection an incomplete explanation in the
tested protocol: interior allocation is separately identifiable. When both
grids are retargeted to the evaluation length, the three-seed mean favors
uniform FMRoPE at every tested OOD length. The correct claim is therefore:

- allocation is a real training-time design variable;
- the author-reported three-seed mean favors target-aware range transport in
  this tested deployment;
- no additive synergy or empirical orthogonality has been established;
- Cosh is not claimed to be the empirical optimum over all schedules.

The score-changing logic is therefore:

\[
\boxed{
\text{submitted method identity}
\rightarrow \text{valid missing comparison}
\rightarrow \text{new exact-range identification}
\rightarrow \text{narrow deployment boundary}
}
\]

## 2. Position by optimization object and stage

Do not draw the distinction as “EVQ changes shape while every other method
changes one scalar.” YaRN, LongRoPE, and other methods can also move channels
non-uniformly. The defensible distinction is the decision being optimized, the
stage at which it is made, and how the table is constructed.

| Method family | Optimization object and stage | Interior can change? | Boundary relative to EVQ-Cosh |
| --- | --- | ---: | --- |
| FMRoPE / geometric base selection | Select or retarget the usable spectral range while retaining a geometric, uniform-in-log exponent order | Through the selected range, not its normalized geometric order | Range selection/transport, not an explicit variational allocation of the training grid |
| NTK-aware / YaRN | Map an existing spectrum to a target context; YaRN uses frequency-dependent interpolation and attention scaling | Yes | Target-context transport of an existing model/spectrum, not the same training-time optimization problem |
| LongRoPE / LongRoPE2 | Search target-specific per-dimension rescaling factors for a pretrained checkpoint, with adaptation in the full method | Yes | Post-hoc target search/adaptation rather than a fixed closed-form pretraining allocation |
| Resonance-style analytic deformation | Modify individual frequencies to satisfy a specified interpolation/periodicity criterion | Yes | A different analytic objective; prevents a broad “first analytic non-geometric grid” claim |
| Learned-frequency methods / LeRoPE | Optimize \(O(K)\) frequency degrees of freedom during training | Yes | Learns the grid jointly with model weights; EVQ fixes it before training with zero learned frequency parameters |
| EVQ-Cosh | Solve a stated variational surrogate for finite training-time allocation and instantiate its Cosh density by inverse CDF | Yes | Closed-form, fixed before training, zero learned frequency parameters, no per-channel search |

Use:

> These families may ultimately produce non-uniform numerical frequency
> tables, but they optimize different decisions at different stages. EVQ
> chooses a fixed training substrate before learning; range methods select or
> transport a spectrum for a target context; learned-frequency methods
> optimize the table through training. The distinction is the optimization
> object, stage, and construction—not a prediction of additive empirical gain.

The narrow novelty statement is:

> To our knowledge, EVQ-Cosh is the first explicit variational formulation of
> finite training-time RoPE-grid allocation for standard RoPE with a
> closed-form, zero-learned-parameter inverse-CDF construction.

Do not broaden this to “first to change interior frequencies,” “first analytic
non-geometric RoPE,” or “first to show frequencies matter.”

## 3. Linchpin: three-seed exact-range identification

This is not one more performance ablation. It is the experimental identity test
for `AC.1` and a direct attribution test for `AC.3`.

\[
\boxed{
\text{same sampled support and training protocol}
+\text{different interior nodes}
\Longrightarrow
\text{different trained OOD behavior}
}
\]

Within each seed pair:

- sampled highest frequency: identical;
- sampled lowest frequency: identical;
- sampled log-frequency span: identical;
- model architecture and trainable initialization: identical;
- token order, optimizer, and 499,974,144-token budget: identical;
- 32 frozen evaluation anchors: identical;
- only the \(K-2=30\) interior locations differ.

Negative means Cosh is better:

| Fixed training range: Cosh minus uniform FMRoPE | 512 | 1K | 2K |
| --- | ---: | ---: | ---: |
| Three-training-seed mean NLL | **-0.3159** | **-0.1949** | **-0.1674** |
| Training seeds favoring Cosh | **3/3** | **3/3** | **3/3** |

Safe conclusion:

> Because every sampled scalar range degree of freedom is fixed, this control
> establishes finite-channel interior allocation as a genuine training-time
> variable rather than a reparameterization of base or range.

Adjacent boundary:

> When both tables are target-retargeted, the author-reported three-seed mean
> favors uniform FMRoPE at every evaluated OOD length. We therefore claim
> identification of a non-reducible allocation variable, not additivity or
> superiority over target-aware range transport.

The endpoint-normalized Cosh arm is a pure-shape diagnostic. It is not identical
to submitted raw midpoint EVQ, where changing \(\tau\) also changes sampled
extrema and span.

Evidence owner:
`theory_results/MATCHED_RANGE_COSH_500M_3SEED_20260724.md`.

### Concurrent confirmation, not method evidence

LeRoPE (arXiv:2607.10134) appeared after this submission and learns one scalar
per RoPE frequency, with experiments up to approximately 2.5B parameters. Its
scalable language-modeling gains, and the fact that frozen learned frequencies
preserve part of the benefit, support frequency-grid optimization as a serious
design axis. It does not establish the Cosh family, raw EVQ extrapolation, or
zero-parameter allocation, and it cannot substitute for EVQ's own controlled
scale-transfer experiment.

## 4. Contribution hierarchy and evidence ladder

### Contribution 1: an explicit finite-grid optimization object

EVQ's central contribution is not the observation that frequency bands exist.
It formalizes where a fixed number of training-time RoPE nodes should be placed.
The exact-range control establishes that this object is experimentally
identifiable after range is removed.

### Contribution 2: a closed-form variational realization

For the stated convex surrogate, Cosh is the unique density minimizer. The
inverse-CDF construction produces a fixed, one-parameter grid before training,
with zero learned frequency parameters and no per-channel search.

### Contribution 3: controlled attribution and robustness

Use the evidence in this order:

1. three-seed exact-range identification;
2. submitted exact-kernel corroboration in Appendices A.6 and A.14;
3. fixed zero-parameter Cosh/Geo/non-Cosh and native-span controls;
4. three-seed held-out base \(1\)M and
   \(d_{\mathrm{head}}=128\) robustness;
5. bounded \(\tau\) evidence as an operating prior;
6. target-aware FMRoPE as the required deployment boundary.

### Submitted mechanism corroboration, not pure-shape attribution

Use A.6 and A.14 after the exact-range result, not instead of it:

- **A.6, functional surrogate validation:** the allocation derived from
  \(\mathcal C_{\mathrm{app}}\) lowers the collision score under the exact
  oscillatory RoPE kernel in all 12 tested configurations by 24–92% and raises
  effective rank by 24–570%.
- **A.14, language-model-free mechanism isolation:** across the 36 tested
  configurations, both fixed EVQ and a numerical collision-only search reduce
  exact-kernel off-diagonal energy relative to geometric allocation. The
  representative table's “−65%/−66%” annotations are oracle-versus-Geo values,
  not a uniform EVQ effect.

These submitted diagnostics establish that the surrogate direction survives
contact with the exact kernel and that non-geometric placement changes its
coherence structure. Neither holds sampled extrema and span fixed. The new
three-seed exact-range experiment is therefore the trained-model attribution
test that removes this remaining confound.

### Evidence breadth across settings

The submission is not confined to one small causal-MHA model, but the evidence
must be described by tier:

- **controlled primary breadth:** 454M three-seed MHA and 432M three-seed MLA;
- **larger/training-stage support:** 750M continued pretraining and 454M
  progressive context training, both single seed;
- **cross-modal support:** 129M two-seed shared-run video DiT and a directionally
  consistent 382M single-seed scale-up;
- **mature-model support:** single-seed LoRA adaptation of
  LLaMA-3-8B-Instruct;
- **capability-adjacent support:** QuALITY gold-answer NLL, while four-option
  accuracy remains near chance at the tested model scale.

The reviewer-safe correction is:

> Evidence was broader than a “small models in one architecture only” summary,
> but the strongest controlled multi-seed tier remained below 1B.

These studies must not be pooled into a universal-dominance claim. They
establish applicability only in the tested architectures, modalities,
model-training stages, and mature-model adaptation protocol. The 750M, 382M,
progressive, and submitted 8B rows remain supporting; they do not close clean
scale transfer or downstream capability.

### Evidence interpretation: capability conversion

Do not substitute PPL, routing, retrieval, or downstream accuracy for one
another. Organize the evidence as the following endpoint hierarchy:

\[
\text{frequency allocation}
\rightarrow \text{LM probability}
\rightarrow \text{gold-answer probability}
\rightarrow \text{causal remote-source use}
\rightarrow \text{answer-token rank}
\rightarrow \text{AR top-1 / task accuracy}.
\]

This is not a causal chain closed by one experiment. Different experiments
support different endpoints, and success at one level does not establish the
next. Multi-seed controlled evidence establishes allocation-to-LM-probability
effects; single-seed 8B supporting evidence extends the hierarchy to causal
remote-source use and token rank. Stable autoregressive readout and task
accuracy remain open.

## 5. Score-changing evidence ledger

| Rank / ID | Role | Artifact and status | Reviewer-facing evidence | Mandatory boundary |
| --- | --- | --- | --- | --- |
| 1. `E-MATCHED-RANGE-3S` | `AC.1`, `AC.3`; method identity | `MATCHED_RANGE_COSH_500M_3SEED_20260724.md`; `AUTHOR_CONFIRMED_AGGREGATE_PENDING_LOCAL_RAW_PROMOTION` | Same extrema/span and matched within-seed training protocol; only 30 interiors differ. Mean Cosh-minus-FMR NLL is \(-0.3159/-0.1949/-0.1674\) at 512/1K/2K; 3/3 seeds agree | Retargeted mean favors FMR at every OOD length; no CI/significance until raw promotion; exact-range arm is not raw submitted EVQ |
| 2. `E-SUBMITTED-KERNEL` | `R27bE.1`, `AC.3`; pre-existing mechanism corroboration | Submitted Theory §3 and Appendices A.6/A.14 | A.6: exact-kernel collision falls 24–92% in 12/12 configurations and effective rank rises 24–570%. A.14: EVQ and collision-only search move exact-kernel off-diagonal energy below Geo across 36 tested configurations | Not matched-range trained-model attribution; A.14 “−65%/−66%” annotations belong to the oracle column |
| 3. `E-SHAPE` | `R27bE.3/.4`, `AC.3`; fixed-schedule attribution | `EXPERIMENT_REPORT_20260724.md` §3; three-seed report-backed | Cosh improves the geometric control while matched exponential is also competitive | Cosh is one analytic schedule, not an empirical all-schedule optimum |
| 4. `E-NATIVE-SPAN` | Removes midpoint/span explanations | Report §7 and native-shape aggregate | Nonlinear allocation remains consequential under native endpoint and matched-span controls | Alternative shapes can be stronger at some lengths |
| 5. `E-HELDOUT` | `R27bE.2/.5`, `AC.2`; base/head transfer | Report §4; post-submission three-seed | Held-out base \(1\)M and \(d_{\mathrm{head}}=128\) improve at every OOD length tested | One 151.9M configuration; not a production-scale closure |
| 6. `E-TAU` | `R27bE.1/.4`, `AC.3`; operating-rule scope | Report §2 and `PHASE16_99RUN_RAW_REANALYSIS_20260724.md` | The practical rule often lands in a useful basin | It is fallible, not a trained-model point optimum |
| 7. `E-MHA/MLA` | Primary architectural scope | Paper/provenance primary text and 432M three-seed MLA evidence | Allocation effects appear in standard MHA and a 16-pair compressed-RoPE stress test | MLA \(d_{\mathrm{eff}}\) is an operating convention, not a theorem |
| 8. `E-VIDEO/PROGRESSIVE/CONTINUE` | Modality and training-stage scope | Supporting paper appendix and tracked reports | Bidirectional 3D-RoPE DiT, progressive context training, and continued pretraining preserve the intervention's relevance | Supporting 1–2-seed evidence; no pooled superiority |
| 9. `E-8B-ADAPT` | Mature-model applicability and conversion ladder | `theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md`; tracked-report and curated-JSON backed; single seed | 300-step rank-64 Q/K/V/O LoRA improves 16K/32K probability and causally stronger remote-source use | Native means the exact original Llama endpoint grid, not Paper-Geo or FMR; combined EVQ+LoRA substrate evidence, not pure Cosh attribution or solved QA |
| 10. `E-OLMO-1B-PENDING` | Controlled scale transfer | Pending | Public OLMo-2 1B step-0, EVQ intervention branch only, released Geo milestone control, 2.1B-token first gate | No result may be quoted; this strengthens controlled scale evidence if positive |

## 6. Mature-model applicability and the conversion problem

Use this only when scale, generality, or downstream relevance is directly
questioned:

> EVQ is not confined to small models trained from scratch. Evidence spans
> from-scratch MHA/MLA training, bidirectional 3D-RoPE video DiT, progressive
> and continued training, and adaptation of a mature Llama-3-8B-Instruct
> checkpoint. In the 8B study, the frequency grid is fixed after original
> pretraining and before a short LoRA adaptation: the control uses the exact
> original Llama endpoint grid, whereas EVQ-LoRA uses midpoint EVQ-Cosh. Only
> 300 matched steps of rank-64 Q/K/V/O LoRA produce pack-consistent 16K/32K
> NLL gains and, in a ten-case true-16K passkey probe, stronger routing to the
> remote target block over 32 frozen retrieval heads. This demonstrates
> mature-model probability and routing effects under the tested adaptation
> protocol; those effects have not yet converted into stable benchmark
> accuracy.

If space permits, the strongest compact evidence is:

- relative to matched Native-LoRA, EVQ-LoRA changes NLL by
  \(+0.390/-1.510/-2.048\) at 8K/16K/32K, with the long-range direction
  holding across all 24 packs and three domains;
- in the ten-case true-16K passkey probe, median target-block hit@16 over 32
  frozen retrieval heads changes from 18.75% to 64.06%, with EVQ winning all
  ten paired cases;
- removing the remote gold block from every attention head increases EVQ NLL
  by 1.5055 while leaving the Native control approximately unchanged;
- in the same ten-case probe, first-token median correct-token rank changes
  from 33,774.5 to 2,043.0, still far from top-1.

Call this **combined EVQ+LoRA substrate evidence**. The two arms match seed,
frozen training rows and order, 300 steps, rank-64 Q/K/V/O LoRA capacity, and
optimizer. Relative to the from-scratch exact-range experiment, however, this
protocol introduces a shared adaptation dataset, LoRA capacity, and
autoregressive readout around the grid intervention; its result therefore
includes grid-specific LoRA co-adaptation and is not pure interior-allocation
identification. Do not call attention routing exact retrieval, gold NLL
downstream accuracy, or token-rank improvement successful generation.

The concise capability boundary is:

> In this matched single-seed protocol, a fixed EVQ grid affects mature-model
> long-range probability and causal remote-source use, but those signals have
> not yet converted reliably into autoregressive top-1 decisions or aggregate
> task accuracy.

If `AC.2` directly asks for downstream benchmarks, add the relevant endpoint
results rather than substituting the mechanistic evidence:

- on QuALITY (\(n=2{,}086\)), EVQ lowers gold-option NLL at 8K and 16K, while
  four-option accuracy for every configuration remains within 23.7%–26.8%
  around the 25% chance floor;
- on the registered 303-example Qasper/NarrativeQA gate, EVQ-LoRA task-macro
  F1 is 0.1126 versus 0.2110 for Native-LoRA. The deficit is concentrated at
  or below the 8K adaptation length; above 8K all arms are at the task floor
  and the EVQ-versus-Native comparison is unresolved.

These are trigger-only boundary results, not the opening novelty evidence.

## 7. Concern-by-concern response route

### AC.1: novelty relative to FMRoPE and frequency-usage observations

Split the concern before defending:

1. **Related-work omission — valid.** FMRoPE should have been cited and directly
   compared. Concede this in the first sentence.
2. **Dead-channel novelty — not the submitted claim.** The paper cites prior
   frequency-specialization/allocation observations and uses dead channels only
   as supporting mechanism context. Its contribution list instead claims an
   explicit finite-grid optimization object and closed-form variational
   construction.
3. **Technical equivalence — not established.** The submitted Introduction and
   §2 already separate pre-training allocation from base/range selection and
   target transport. The new exact-range control tests that distinction rather
   than inventing it.
4. **Deployment ranking — concede the boundary.** Target-retargeted FMRoPE is
   stronger in the reported three-seed mean; technical non-equivalence does not
   imply additive gain or deployment superiority.

Use this sequence:

\[
\text{valid omission}
\rightarrow \text{submitted definition}
\rightarrow \text{submitted kernel corroboration}
\rightarrow \text{new exact-range attribution}
\rightarrow \text{retargeted boundary}.
\]

Core English paragraph, usable only after the `E-MATCHED-RANGE-3S` send gate
passes:

> We agree that FMRoPE is directly relevant and should have been cited and
> compared explicitly. This omission does not establish technical equivalence.
> The submitted paper does not claim novelty for the observation of dead or
> under-utilized frequency channels; it cites prior evidence that frequency
> usage is unequal and defines a different contribution: a pre-training
> finite-grid allocation problem with a closed-form, zero-learned-parameter
> inverse-CDF construction. This distinction was already stated in the
> Introduction and §2, while Table 1 and Appendices A.6/A.14 separated the
> conditional surrogate result from exact-kernel corroboration. The new
> three-seed exact-range control makes the allocation-versus-range distinction
> experimental: within each seed pair, uniform FMRoPE and Cosh match sampled
> extrema, log-span, trainable initialization, token order, optimizer, budget,
> and frozen anchors, and differ only in the 30 interior frequencies. Cosh
> minus uniform FMRoPE is \(-0.3159/-0.1949/-0.1674\) mean final-128-token NLL
> at 512/1K/2K, with all three training seeds agreeing in direction. When both
> tables are target-retargeted, the author-reported three-seed mean favors
> uniform FMRoPE. We therefore claim identification of a non-reducible
> training-time allocation variable, not additivity, replacement, or
> superiority over target-aware range transport.

Do not say “the reviewer failed to read the paper.” Treat the disputed
equivalence as a claim-definition and evidence question that the AC can
adjudicate.

### R27bE.1 / R27bE.4 / AC.3: surrogate, Cosh, and \(\tau\)

The submitted paper already separates the core epistemic layers:

1. **Conditional theorem:** Cosh is the unique minimizer of the stated convex
   \(\mathcal C_{\mathrm{app}}\), under its explicit coordinate, positivity,
   normalization, and function-space assumptions.
2. **Exact-kernel corroboration:** A.6 tests whether the surrogate-derived
   direction survives under the exact oscillatory kernel; A.14 isolates
   exact-kernel off-diagonal energy without language-model loss.
3. **Trained-model consequences:** the experiments report NLL/PPL/retrieval
   outcomes after optimization; these are empirical, not theorem-level.
4. **Operating prior:** \(\tau=d_{\mathrm{eff}}/\sqrt{L_{\mathrm{train}}}\)
   selects a practical basin; it is not the trained-model global optimum.

Agree with the remaining concern: the submitted evidence did not fully
attribute the trained-model effect among Cosh shape, finite \(\tau\), and
alternative fixed schedules. The new matched non-Cosh, neighboring-\(\tau\),
native-span, held-out base/head, and exact-range controls address that
attribution gap. Competitive non-Cosh schedules do not refute the conditional
theorem; they bound its empirical interpretation.

Core English paragraph:

> The submitted paper already separates the levels highlighted by the
> reviewer: Table 1 states the conditional status of the surrogate theorem;
> Appendix A.6 evaluates the derived schedule under the exact oscillatory
> kernel; Appendix A.14 performs a language-model-free collision-only
> diagnostic; and §§4–5 report post-training consequences. We agree that this
> map and the remaining attribution gap should be made more prominent. The new
> fixed-schedule, neighboring-\(\tau\), held-out base/head, and exact-range
> experiments test whether trained-model improvements are specific to Cosh or
> the operating rule, or reflect a broader allocation effect. They show that
> Cosh is effective but not empirically dominant over every analytic schedule,
> and that the operating rule is a useful but fallible basin selector rather
> than a global optimum.

### R27bE.3: learned-frequency / submitted “DAPE” comparison

Lead with fixed zero-parameter controls, not the historical learned row:

> Learned-frequency comparisons mix grid freedom with learned capacity and
> tuning. We therefore use the exact-range and fixed-schedule controls for
> attribution: the operator and all training variables are matched, no
> frequency parameter is learned, and only the fixed grid changes.

If the historical row must be discussed, call it **layer-shared learnable
inverse frequencies**, not a verified implementation of the DAPE attention
operator.

### R27bE.2 / R27bE.5 / AC.2: base, architecture, scale, and evaluation

Separate breadth from evidentiary strength. Do not accept “no larger or
pretrained evidence,” but do accept that the clean controlled tier is too
small:

> We agree that the strongest controlled multi-seed evidence in the submission
> remained below 1B parameters. The empirical scope was nevertheless broader
> than one small from-scratch architecture: the submitted evidence-tier table
> separately identifies 454M three-seed MHA, 432M three-seed MLA, 750M
> continued pretraining, 129M/382M video DiT, progressive context training,
> QuALITY gold-answer NLL, and single-seed adaptation of
> LLaMA-3-8B-Instruct. We do not pool these tiers or present the supporting
> rows as controlled scale closure. The new held-out base/head controls address
> robustness, while the 8B evidence remains supporting and the clean
> approximately 1B scale-transfer experiment remains pending.

Response order:

1. identify the breadth already present in the submitted evidence-tier table;
2. concede that its strongest controlled multi-seed tier remained below 1B;
3. add held-out base/head three-seed robustness;
4. preserve the evidence tiers across MHA, MLA, video DiT, progressive,
   continued training, QuALITY, and mature 8B adaptation;
5. use the probability-to-capability ladder;
6. keep OLMo-2 1B controlled scale transfer pending.

Do not frame OLMo-2 as the first test of broad applicability. Its value is
narrower and stronger: if positive, it adds controlled scale transfer on a
modern full-RoPE, large-base, approximately 1B architecture.

### AC.4: score-changing package

The compact package is:

1. concede the FMRoPE citation/comparison omission;
2. show the score-changing submitted-evidence correction matrix;
3. recover the method identity already stated in the submitted paper;
4. connect submitted A.6/A.14 exact-kernel evidence to the mechanism;
5. lead new evidence with exact-range three-seed identification;
6. add fixed-schedule and held-out robustness;
7. preserve the target-retargeted FMRoPE boundary;
8. state that empirical breadth existed while the strongest controlled tier
   remained below 1B;
9. preserve evidence tiers and state the unresolved capability-conversion gap;
10. keep OLMo-2 1B explicitly pending.

### AC-facing adjudication note

Use only if the venue provides an AC-confidential field and only after the
exact-range aggregate passes its send gate:

> We would like to separate the valid related-work concern from the resulting
> technical-equivalence conclusion. We agree that FMRoPE should have been cited
> and directly compared. At the same time, the submitted paper already defines
> EVQ as a pre-training finite-grid allocation problem rather than a scalar
> base/range rule, explicitly separates the conditional theorem, exact-kernel
> corroboration, and trained-model evidence, and does not claim novelty for
> dead-channel observations. The requested direct control is now complete
> across three training seeds: sampled extrema and log-span are identical and
> only the interior frequency nodes differ. Cosh improves fixed-range OOD NLL
> at 2x/4x/8x in all three seeds. Target-retargeted FMRoPE remains stronger, so
> we do not claim empirical additivity or replacement of target-aware range
> transport. We respectfully ask that the novelty question be assessed as
> finite-grid allocation versus base/range intervention, rather than by whether
> both works broadly study frequency usage.

This note asks the AC to adjudicate the technical variable. It must not discuss
reviewer confidence, motive, diligence, or alleged failure to read an appendix.

## 8. Optional pre-submission continuity

Use only if a reviewer suggests that allocation was invented after FMRoPE.
Before EVQ-Cosh, the project studied anchored-sigmoid frequency warping,
parameter-efficient frequency adaptation in a mature 7B model, and full
long-context task trade-offs. This chronology shows a continuous
pre-submission research program on finite frequency-grid design.

It is not the novelty proof, not evidence for the Cosh surrogate, and not the
headline 8B evidence. If mentioned, preserve the fact that the anchored
LongBench aggregate was not positive.

## 9. Response exclusions and claim boundaries

Omit unless directly triggered:

- detailed anchored schedule history or task-by-task LongBench values;
- failed sparse variants, band-pruning, residual, or EVQ-v2 explorations;
- internal adversarial reviews and GPU execution details;
- OLMo plans beyond the registered first gate.

Do not claim:

- that all range methods change only a scalar or preserve interiors;
- that separate decisions are empirically orthogonal or additive;
- that the submission never used “orthogonal” or “additive”; acknowledge that
  wording and narrow it from structural distinction to empirical non-additivity;
- that the submitted paper never discusses dead channels; the accurate point is
  that dead-channel discovery is not its claimed novelty;
- that EVQ improves target-aware FMRoPE deployment;
- that Cosh is the empirical optimum over real schedules or task losses;
- that the \(\tau\) rule predicts a trained-model global optimum;
- that routing, NLL, PPL, retrieval, generation, and task accuracy are
  interchangeable;
- that the 8B LoRA study is pure interior-allocation attribution;
- that current downstream capability conversion is solved;
- that OLMo-2 1B is complete;
- that LeRoPE proves Cosh or replaces EVQ's own scale evidence.

Do not introduce a new algorithm, EVQ-v2, or a post-hoc mechanism theory into
the formal response.

## 10. Send gate

Before an author response is sent:

1. the `AC.1` opening must first concede the FMRoPE citation/direct-comparison
   omission, then distinguish that omission from technical equivalence;
2. every claim about what the submission already states must resolve to a
   submitted section, table, or appendix; do not describe post-submission
   evidence as submitted evidence;
3. the opening must define finite training-time grid allocation before using
   performance as the novelty argument;
4. every number must resolve to a standalone report and retained provenance;
5. exact-range wording must say “within each seed pair” and list the matched
   sampled extrema/span and training variables;
6. obtain and promote a three-seed raw aggregate with per-seed contrasts,
   hashes, and training-seed confidence intervals before quoting the new
   aggregate externally;
7. until that promotion, do not add CIs, significance language, retargeted
   numeric values, or a `RAW_BACKED` label;
8. the target-retargeted FMR result must remain adjacent to the fixed-range
   identification claim;
9. related-work comparisons must distinguish object and stage and must not
   erase YaRN/LongRoPE/Resonance interior changes;
10. A.6 may support exact-kernel functional validation but not fixed-range
    attribution; A.14's “−65%/−66%” annotations must not be assigned to EVQ;
11. Cosh uniqueness must remain conditional on
   \(\mathcal C_{\mathrm{app}}\), and \(\tau\) must remain an operating prior;
12. the response must acknowledge and narrow the submitted
    “orthogonal/additive” wording rather than denying it;
13. every “already in the submission” correction must distinguish evidence
    presence from evidence strength; cite its tier, seed count, and actual
    endpoint;
14. 750M continuation, 382M video DiT, progressive training, and submitted 8B
    LoRA must remain supporting evidence, not controlled scale closure;
15. 8B evidence may answer scale/applicability only with its supporting,
    single-seed, combined-EVQ+LoRA and open-readout boundaries;
16. OLMo-2 1B remains pending until the 2.1B-token gate has a standalone
    report, raw metrics, and provenance;
17. no existing paper-table value is silently changed by a post-submission
    diagnostic;
18. do not assign claims to an unpromoted “Reviewer 2” source; use `AC.1` until
    the full review text and provenance are retained.

The response succeeds if a first-time reader can follow:

\[
\text{submitted finite-grid identity}
\rightarrow \text{submitted evidence counted at its correct tier}
\rightarrow \text{valid missing FMRoPE comparison}
\rightarrow \text{submitted exact-kernel corroboration}
\rightarrow \text{three-seed exact-range identification}
\rightarrow \text{target-retargeted boundary}
\rightarrow \text{bounded cross-setting applicability}
\rightarrow \text{explicit capability-conversion boundary}.
\]
