# ICLR 2027 narrative reflection: stronger science, weaker reading contract

- **Date:** 2026-08-21
- **Status:** internal narrative decision memo; not manuscript prose
- **Trigger:** an external DeepSeek review of the current 9-page manuscript
- **Authority:** current source, canonical evidence owners, and the validated
  PDF override every external-review interpretation
- **Goal:** recover ICLR-style conceptual force without restoring the
  scientific defects of the NeurIPS 2026 version

## 1. Executive conclusion

The current ICLR manuscript is scientifically stronger than the NeurIPS
submission but less immediately persuasive. The revision repaired the support
confound, added a three-seed fixed-support identification, replaced one-sided
geometry with full sin/cos subspace analysis, proved the exact transplant
obstruction, and added mature-scale and causal remote-source evidence. Those
are real advances.

The writing, however, was optimized across too many rounds for claim safety,
protocol identity, and reviewer-defence coverage. The result increasingly
reads like an evidence audit: readers are invited to inspect eight separately
scoped claims instead of first experiencing one discovery. This creates more
attack surfaces even when the underlying science is stronger.

The next revision must therefore change the **reading contract**, not add more
defensive sentences. It should combine the present science with the rhetorical
clarity of the earlier paper.

## 2. Why the NeurIPS version was easier to praise

The NeurIPS version offered a simple method-paper contract:

> RoPE has a finite frequency budget; EVQ-Cosh is a closed-form training-time
> allocation that improves long-context behaviour.

Its introduction named three primary stress tests and explicitly separated
supporting/exploratory evidence. The abstract paired the $31.1\%$ MLA result
with its $+1.1\%$ in-window cost. A reader could summarize the submission after
one page, and every section appeared to support the same method claim.

That clarity did not make the science sufficient. The real NeurIPS reviews
exposed overlap with FMRoPE, missing fixed-support identification, narrow or
synthetic endpoints, and uncertainty over whether the table or its sampled
range owned the effect. The ICLR revision was necessary; returning to the old
scientific architecture would recreate the reasons behind the $4/2/3$ outcome.

## 3. What regressed in the ICLR rewrite

The current first two pages ask the reader to absorb all of the following:

1. a slow-subspace redundancy phenomenon;
2. the support/allocation decomposition;
3. fixed-support causal identification;
4. the EVQ-Cosh construction;
5. full-basis effective-rank geometry;
6. weight--table co-adaptation and a frozen-transplant obstruction;
7. a $432$M--$1.485$B training and architecture ladder; and
8. $1.485$B/$8$B capability and remote-source evidence.

Each item is defensible, but the paper does not yet make them feel like one
indivisible finding. A hostile or merely hurried reader can select the weakest
link and review the paper through it: the surrogate is not LM risk, finite
$\tau$ is an operating convention, mature protocols are heterogeneous, or the
deployed Cosh table pays an in-window cost.

The central writing mistake was treating rebuttal-proofing as narrative
structure. Scientific qualifications belong in owners and at the nearest
claim boundary; they should not determine the reader's first mental model.

## 4. Triage of the DeepSeek review

The external review is useful as a reading-comprehension probe, not as an
evidence owner or edit specification.

### Real score-ceiling signals

1. **The theory-to-construction bridge remains a surrogate.** The Cosh density
   uniquely solves the stated convex continuum surrogate, not language-model
   risk. The manuscript already states this correctly. The narrative should
   make Cosh the first controlled construction on the identified axis, rather
   than imply that the surrogate exhausts the spectral-budget problem.
2. **Finite $\tau$ is an operating prior.** The scaling structure is derived;
   its prefactor is a convention and configuration ordering varies. This is
   already scoped correctly. More derivation prose will not turn it into a
   global optimum.
3. **The evidence hierarchy is no longer immediately visible.** The
   fixed-support and $432$M MLA arms are replicated owners; $750$M, $1.485$B,
   and $8$B establish persistence, stage, and capability roles. One concise
   hierarchy sentence is more effective than repeating seed labels beside
   every mature result.
4. **The deployed Cosh frontier is visible.** Its in-window/OOD pattern should
   be interpreted where the result is presented as an effective-context
   frontier shift. It should not become an apologetic abstract sentence or a
   universal law: the internal phase-chord result already shows that a
   Cosh-sized cost is not forced by the finite-budget premise.

### Already handled boundaries, not new defects

- The transplant theorem deliberately concerns exact preservation under
  fixed, position-independent Q/K maps. Together with the weight--table
  crossing, it explains why post-hoc static replacement is not equivalent to
  training with a table. It does not claim to prohibit retraining.
- Target-aware support retargeting and interior allocation interact. The body
  Discussion already states this. The target-matched appendix reversal should
  not be promoted into a second paper thesis.
- Static rank diagnoses positional-basis redundancy; it is not an LM-quality
  predictor. Table 1 is not a contradiction but the co-adaptation bridge:
  better post-hoc geometry can fail under weights trained in another basis.

### Noise or incorrect recommended actions

- **Do not add `single trajectory` beside every mature result.** That converts
  the paper back into an audit ledger. State the evidence hierarchy once and
  keep the exact seed inventory in the appendix/owner.
- **Do not foreground the target-matched reversal.** One interaction-boundary
  sentence is sufficient and accurate.
- **Do not call the $432$M MLA result a degenerate regime.** $K=16$ and
  $2\times$ extrapolation are the theory-matched scarce-budget stress test;
  teacher-forced PPL is the intended probability endpoint.
- **Do not reduce 2Wiki to “degrades more slowly.”** EVQ-Cosh retains token-F1
  from $24.84$ at $4$K to $21.48$ at $8$K while Native falls to $0.07$; the
  current claim of substantial answer overlap at $8$K is accurate.
- **Do not invent an expansion for EVQ.** If no author-confirmed expansion
  exists, define the coined method by its Cosh density and inverse-CDF warp at
  first use rather than fabricating acronym semantics.

## 5. The ICLR reading contract

ICLR reviewers should be able to repeat one discovery after scanning the
first two pages:

> A finite RoPE head spends more than one third of its rotary pairs on nearly
> duplicate slow positional planes. Holding the entire sampled support fixed,
> moving only the interior frequencies changes what the model learns. Exact
> subspace geometry explains the wasted positional budget, while the
> weight--table crossing and transplant obstruction explain why allocation
> must be present during training. EVQ-Cosh is the first closed-form,
> zero-learned-parameter construction on this axis; scarce-channel, scale, and
> remote-source experiments show that the axis matters beyond a small-model
> geometric diagnostic.

This yields the intended logical sequence:

```text
counterintuitive observation
    -> controlled research question
    -> fixed-support identification
    -> exact mechanism/theory
    -> minimal construction
    -> consequences across architecture, scale, and capability
```

The small model owns identification. The larger models do not need to recreate
that control; they show that the identified variable remains consequential.
This distinction should be felt through the order of the prose rather than
repeated as defensive metadata.

## 6. Rewrite principles for the next pass

### Abstract

- Lead with the redundancy discovery and the fixed-support question.
- State the exact identification and theoretical explanation before naming the
  construction.
- Keep one scarce-budget consequence and one mature capability consequence.
- Compress the $454$M/$750$M/$1.485$B ladder into training-stage/scale
  persistence rather than enumerating every protocol.
- Do not add a limitations sentence about in-window cost.

### Introduction

Use six functional paragraphs:

1. finite-head budget and the near-duplicate slow-plane observation;
2. support versus allocation and the fixed-support question;
3. the three-seed answer;
4. exact geometry plus co-adaptation as one mechanism;
5. EVQ-Cosh as a minimal construction, not the unique optimum; and
6. scarce-budget, scale, and remote-use consequences.

The first figure should make that sequence legible. Table 1's crossing is a
mechanism result, not a negative aside, and should be connected directly to
the obstruction theorem.

### Experiments

Retain the natural proof chain:

```text
fixed-support identification
    -> scarce-budget prediction
    -> table/range co-adaptation
    -> training-stage and scale persistence
    -> real-document and causal remote-source use
```

Do not reinstate Primary I/II/III labels. One opening sentence can identify
the replicated causal/statistical owners and the mature persistence/capability
roles without turning every paragraph into a disclaimer.

### Discussion

Keep support/allocation interaction and the exact theorem scope concise. The
Discussion should synthesize what becomes possible after separating the
coordinates, not replay the appendix's negative-result ledger.

## 7. Locked decisions

- Do not return to the NeurIPS scientific architecture.
- Do not edit the manuscript merely because an external reviewer generates a
  plausible caveat; first ask whether it changes a likely ICLR decision.
- Do not add the Cosh trade-off to the abstract.
- Do not add repeated outward `single-seed` labels.
- Do not promote the target-matched reversal.
- Do not describe the MLA stress test as degenerate.
- Do not weaken the verified 2Wiki statement.
- Do not promote the two-seed phase-chord discovery into the manuscript yet.
- If phase-chord or mature LoRA succeeds, replace weaker evidence and simplify
  the story; do not append an eleventh protocol.

## 8. Acceptance implication

The external review does not reveal a new core mathematical or empirical
failure. It reveals that the current prose does not fully convert stronger
science into an ICLR-style conceptual experience.

Subjective internal forecast:

- current scientific package under the present reading contract: roughly
  $55$--$65\%$ acceptance probability;
- after a successful narrative compression using only completed evidence:
  roughly $65$--$70\%$;
- with a clean mature-model retrofit that preserves Native in-window ability,
  improves long-range probability and downstream capability, and replicates
  at $1.485$B: approximately $75$--$85\%$.

These ranges are planning judgements, not calibrated venue statistics. Their
purpose is to rank work: narrative compression and a clean mature retrofit
have higher decision leverage than additional caveats or another unrelated
benchmark.

## 9. Immediate handoff

This memo authorizes no manuscript edit or GPU run. When work resumes:

1. collect the remaining cross-reviews;
2. classify each point as score ceiling, technical credibility, presentation,
   or noise;
3. rewrite the abstract/introduction as one discovery story;
4. preserve the exact current evidence owners and nomenclature; and
5. compile and visually inspect the full paper before accepting the rewrite.
