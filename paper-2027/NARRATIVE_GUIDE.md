# ICLR 2027 narrative and revision guide

> **September 2026 author contract.** This is the sole durable manuscript
> narrative guardrail for the 9/17 internal abstract freeze and 9/25 full-paper
> submission. It does not own facts, numbers, evidence, research priority,
> receipts, or live task state. Current state lives only in
> [`HANDOFF.md`](HANDOFF.md).

## Authority and use

Before editing the manuscript:

1. follow [`../AGENTS.md`](../AGENTS.md) for rules and claim ceilings;
2. use [`../INDEX.md`](../INDEX.md) to locate each canonical owner;
3. read the current source/PDF and [`HANDOFF.md`](HANDOFF.md), not an old review
   locator or plan;
4. use this guide only to decide what the verified science should make the
   reviewer remember.

External-model reviews, historical verdicts, and superseded plans are audit
inputs. They do not authorize edits, upgrade evidence, or define the paper.

## Non-negotiable author doctrine

These principles govern every future manuscript session. They are not stylistic
preferences that another review model may silently reverse.

1. **Write the strongest proposition we actually establish.** The paper shows
   that our owner-defined structured reallocations can produce large
   extrapolation, likelihood, downstream, retrieval, and in-window gains in
   their tested regimes. It is not trying to prove that every arbitrary change
   of $z$ helps every model.
2. **Make the scientific object visible before its consequences.** The
   support--allocation decomposition and the question it makes identifiable
   must be the first memory. Fixed-support identification secures the causal
   claim; exact theory explains the finite spectral budget; EVQ-Cosh is one
   analytic construction; frozen, adaptation, and from-training evidence then
   establishes behavioural consequence and breadth. Do not let a benchmark
   number or method name define the paper before the object is clear.
3. **Minimise reviewer reconstruction work.** A 30-second read must recover the
   support--allocation decomposition, fixed-support identification, spectral
   budget, and strongest consequence. A three-minute read must recover the
   causal control, theoretical mechanism, method identities, and empirical
   breadth. If the reviewer must combine distant caveats or tables to discover
   the claim, the writing has failed.
4. **Do not confuse a clean intervention with a universal theorem.** In the
   pure-$z$ owner, support, operator, checkpoint or training contract, gain,
   routing, data, and evaluation are held fixed as declared. That cleanly
   identifies the tested allocation change. It does not create an obligation
   to debate whether every possible $z$ perturbation must change or improve
   behaviour.
5. **Treat controls as scientific instruments, not opponents.** FMRoPE receives
   its intended $L_{\mathrm{target}}$ in the target-aware condition. The
   reversal is evidence that support and allocation interact, not a loss to
   hide or a baseline to handicap. A method need not win every condition to
   make the controlled decomposition decisive.
6. **Keep causal roles separate without demoting completed evidence.** Exact
   fixed-support studies own pure allocation attribution. Mature frozen,
   adaptation, scale, downstream, and video studies own capability,
   persistence, or breadth at their actual protocols. Do not discard strong
   systems evidence because it is not the pure causal owner, and do not borrow
   the causal owner's estimand for another route.
7. **Avoid both NeurIPS-style over-restraint and overclaiming.** State every
   strong owner-supported result plainly; preserve correct proofs and useful
   appendix detail; do not volunteer an inventory of unrelated negatives.
   Equally, never upgrade protocol-specific evidence into universality,
   significance, SOTA, a global optimum, or a predictive LM theorem.
8. **The manuscript is now a frozen design, not an open brainstorming surface.**
   A future edit must repair a current defect in truth, comprehension, score
   ceiling, or submission validity. New theory explorations, external-model
   preferences, and experiment ideas do not enter the paper merely because
   they are interesting.

## Claims we make—and questions the paper does not need to answer

The reviewer-facing propositions are:

- a finite RoPE table has distinct support and interior-allocation coordinates;
- at fixed support, the paired training intervention identifies allocation as a
  consequential causal variable;
- our frozen derived and coarse allocations produce large zero-training
  extrapolation and tested downstream gains with no parameter updates;
- EVQ-Cosh provides a separate closed-form, zero-learned-parameter
  training/adaptation construction with strong owner-scoped consequences;
- allocation remains consequential across frozen, adapted, from-training,
  scarce-channel, scale, and video regimes, with each protocol retaining its
  own estimand;
- allocation can improve in-window behaviour in completed video evidence, and
  learned allocations in LeRoPE provide attributed convergent evidence.

The following are not target propositions and must not be introduced as
reviewer-facing debates:

- every arbitrary non-geometric $z$ must change or improve every model;
- one allocation is uniquely or universally optimal;
- static geometry alone predicts trained-model quality;
- every route must share one intervention, estimator, seed unit, or mechanism;
- the proposed construction must beat a target-aware range method after that
  method is denied its intended target information;
- a method-level contribution is invalid unless it wins every support,
  checkpoint, length, and endpoint.

When a review proposes weakening the paper because one of these non-target
propositions is unproved, reject the premise unless the manuscript itself
accidentally made that proposition.

## One reviewer memory

> **A finite RoPE table decomposes into sampled support and interior allocation.
> Fixed-support interventions identify allocation as an independent,
> behaviourally consequential coordinate, while target-aware retargeting shows
> that support is a distinct interacting coordinate. Exact phase-invariant
> geometry exposes the finite spectral budget and slow-end redundancy;
> EVQ-Cosh is one analytic construction on this object, whose consequences
> persist across frozen, adapted, and from-training regimes.**

This memory fixes the hierarchy:

1. **Object — expose the two coordinates.** The decomposition
   $x_k=-\log\omega_k=a+Rz_k$ separates sampled support $(a,R)$ from interior
   allocation $z$. This is the paper identity, not a notation for a particular
   method.
2. **Identification — intervene on one coordinate.** Fixed-support controls
   establish that $z$ is independently consequential rather than a disguised
   scalar-base change; target-aware reversal identifies support as the second,
   interacting coordinate.
3. **Explanation and construction — geometry before instantiation.** Exact
   full-sin/cos geometry exposes slow-end positional redundancy. EVQ-Cosh is a
   separate analytic construction on the resulting design space.
4. **Consequence — evaluate structured reallocations.** The paper is not a
   claim that arbitrary `z` changes are useful. It studies specific,
   owner-defined structured schedules that deliver large long-context,
   zero-training, and tested downstream gains.
5. **Implication — regime dependence.** Text protocols often show an
   in-window/long-range crossover, while video-DiT shows that a structured
   reallocation can improve both training-frame and extrapolated performance.
   An in-window tax is therefore not the definition of allocation.

The scientific object leads; its strongest consequences remain prominent. Do
not make an old objection, an internal negative, or an implementation name the
grammatical subject of the paper.

## The three structured-allocation routes

Preserve the routes as distinct estimands. They support one design principle but
must not be pooled.

### 1. Fully frozen zero-training — the core route

The mature-checkpoint route is the most direct practical result and should lead
the empirical story. It uses model-relative structured reallocations with the
checkpoint frozen and no learned positional parameters:

- the **derived allocation** is the owner-defined model-relative full profile;
- the **coarse allocation** is its task-label-free movement-profile ramp
  approximation;
- the same-support geometric allocation is the pure-`z` control;
- Native and official Transformers YaRN are reference rows with their own
  support/transport identities.

The derived and coarse allocations are **not EVQ-Cosh**. The pure-`z` block holds
support, attention gain, checkpoint, evaluation rows, decoder, precision, and
hardware fixed. The broader zero-training deployment may add the declared fixed
gain and Native/long session route; report that combined system separately from
the pure allocation contrast.

This route owns the paper's strongest no-update RULER result and the surrounding
natural-text and tested downstream evidence. Evaluation-row bootstrap intervals
condition on each fixed checkpoint/task set; they are not training-seed,
checkpoint-population, or task-population uncertainty.

### 2. Matched adaptation — mature representations use a new allocation

The adaptation route uses the distinct **EVQ-Cosh** frequency substrate under a
matched low-rank adaptation contract. It establishes protocol-specific
task-family length transfer, long-position probability, routing, and causal
remote-source use in the tested mature models.

It is not a pure frozen-`z` contrast, a pretraining-scale point, unseen-task
transfer, or a pooled downstream result. The all-head source-deletion
intervention may establish causal source use in the tested adapted model; it
does not assign the gain to one interior coordinate or prove reliable top-1
generation.

### 3. From-training / co-adapted — install the structured table before learning

The from-training route uses EVQ-Cosh or its anchored fixed-support form before
or during weight learning:

- the paired three-seed exact-range study owns fixed-support causal
  identification during training;
- the scarce-channel MLA, 750M continuation, and existing 1.485B
  same-initialisation/same-scientific-recipe comparison own their separate
  architecture, persistence, and scale roles;
- video-DiT supplies modality breadth and the completed in-window-positive
  regime.

The paper's from-training scale line stops at the completed **1.485B** evidence.
No further from-scratch scale-up, seed expansion, or new from-initialisation run
is part of the September submission or the follow-up plan. Reopening that
decision requires a new author ruling, an updated durable agenda, and explicit
compute authorization.

## Scientific object and method identities

`z` changes interior frequencies. Never say the method “does not change
frequencies.” The design question is how a finite set of rotary pairs allocates
phase resolution inside its declared support.

Keep these identities exact:

- **EVQ-Cosh:** closed-form, zero-learned-parameter training/adaptation table;
  unique only for its stated convex surrogate.
- **Anchored EVQ-Cosh:** the exact-range EVQ-Cosh quantiles normalized to the
  FMRoPE control endpoints; a training-time identification arm.
- **Frozen derived allocation:** model-relative structured zero-training table;
  not EVQ-Cosh.
- **Frozen coarse allocation:** task-label-free ramp approximation to the derived
  allocation; not EVQ-Cosh and not a separate operator family.
- **Zero-training deployment:** the declared long table/gain/session policy;
  broader than the pure fixed-support table contrast.
- **Geo:** geometric training baseline. **Native:** unmodified pretrained
  checkpoint/table. They are not interchangeable.

The paper may say that its structured schedules materially improve the tested
extrapolation, likelihood, RULER, QA, retrieval, and modality endpoints at their
owner-defined scope. It may not turn that evidence into a claim that arbitrary
non-geometric schedules work, that one profile is universally best, or that all
routes share one causal estimand.

## FMRoPE: causal control, not opponent

FMRoPE is related work and the paper-faithful geometric control that separates
support selection from interior allocation.

- In the fixed-support training intervention, FMRoPE and anchored EVQ-Cosh share
  sampled extrema and log-span; changing interiors identifies the allocation
  effect.
- In its target-aware use, FMRoPE is allowed to use the declared target length
  `L_target` to retarget support. That is the method's intended range-selection
  role, not an unfair baseline or a failure of allocation.
- The target-retargeted ordering and the fixed-support result answer different
  questions. They show that support and structured allocation are distinct but
  interacting levers; they are not a method winner/loser tournament.

State the fixed-support control where needed for identification, keep detailed
target-retargeted values with their owner, and do not claim that the structured
allocation replaces target-aware range transport. Never write “we beat
FMRoPE” or imply that FMRoPE must ignore `L_target`.

## LeRoPE and in-window evidence

LeRoPE belongs in Related Work and Discussion as learned-allocation evidence in
the same broad direction: non-geometric allocation can improve in-window
behaviour. It supports the field implication that allocation is broader than an
extrapolation-only knob.

LeRoPE is not a matched comparator, mechanism validation, approximation to
EVQ-Cosh, or evidence that this paper's structured schedules are universally
in-window optimal. Attribute its result to LeRoPE and keep this paper's own
in-window-positive claim with the completed video-DiT owner.

## Theory-to-evidence contract

Keep the epistemic layers explicit:

1. the phase-invariant collision and spectral-budget identity are exact static
   geometry;
2. low-frequency collapse explains positional redundancy, not unused content
   channels or reclaimable LM capacity;
3. the weights-by-table crossing and exact transplant obstruction explain why
   installation stage and co-adaptation matter;
4. the EVQ-Cosh theorem derives a unique minimizer only for the stated convex
   surrogate;
5. trained-model interventions establish the behavioural consequences of the
   structured schedules.

Geometry motivates and explains the designs; it does not rank trained-model
quality. The finite-`tau` convention is a fallible operating prior, not a global
or continuous optimum.

## Reviewer path and body allocation

Preserve this reading order:

1. finite-table decomposition into sampled support and interior allocation;
2. fixed-support identification of $z$ and target-aware retargeting of support;
3. exact spectral-budget geometry and slow-frequency collapse;
4. the separate EVQ-Cosh analytic construction;
5. frozen, matched-adaptation, and from-training consequences, with their
   endpoint and seed scopes;
6. Related Work and Discussion: FMRoPE as support control, LeRoPE as
   learned-allocation in-window evidence, and regime-dependent allocation as the
   field implication.

Do not organize the abstract, Figure 1, or Contributions around a benchmark
leaderboard, EVQ-Cosh as a method identity, arbitrary `z`, old reviewer
defences, or an evidence ledger. The decomposition leads; causal identification
and exact geometry establish the finding; the analytic construction and
lifecycle evidence show what follows from it.

## Edit workflow

### Before an edit

- What current passage prevents the reviewer from recovering the structured
  design, zero-training result, or correct method identity?
- Which canonical owner governs it?
- Which of the three routes does it belong to?
- Does it strengthen the central result, or revive an old debate?
- Can lower-leverage prose be replaced rather than stacked?

### After an edit

- Does the 30-second reading recover support--allocation decomposition →
  fixed-support identification and retargeting → spectral-budget geometry →
  analytic construction → lifecycle consequence?
- Are EVQ-Cosh, anchored EVQ-Cosh, frozen derived/coarse allocations, Geo,
  Native, FMRoPE, YaRN-style, and the MLA wavelength-blend operator distinct?
- Are likelihood, capability, seed, row, checkpoint, and trajectory units kept
  separate?
- Is FMRoPE permitted its target-aware `L_target` role?
- Is LeRoPE used only as attributed learned-allocation in-window evidence?
- If science rather than presentation changed, was the canonical owner updated
  first?

## September boundary

The September manuscript currently uses completed evidence. Research remains
active under the user's exact compute authorization; validated new results may
enter only through an indexed owner and an explicit author promotion decision.
The completed 1.485B comparison remains the current from-training evidence
ceiling. This corrects the earlier blanket "no new submission compute" wording.
Do not admit an old panel item, theory exploration, or experiment design merely
because it appears in an archive.

Current tasks, authorization, freeze progress, validation receipts, and author
decisions belong only in `HANDOFF.md`. This guide changes only when the author
changes the durable reviewer memory or owner-supported narrative hierarchy.

## Future-session stop rules

A future session is moving in the wrong direction if it does any of the
following:

- lets a benchmark result, frozen route, or EVQ-Cosh become the paper identity
  before the support--allocation decomposition is clear;
- replaces a specific positive claim with an argument about arbitrary or
  universal $z$;
- describes completed evidence as merely protocol-limited without naming a
  material claim that the protocol cannot support;
- treats the target-aware FMRoPE reversal as an embarrassing failure, or
  prevents FMRoPE from using $L_{\mathrm{target}}$ so another method can win;
- calls the frozen derived/coarse allocation EVQ-Cosh, or attributes the
  bundled session policy's total gain to pure $z$;
- removes mature, downstream, scale, or video evidence merely because another
  experiment owns the cleanest causal identification;
- accumulates defensive caveats, failed internal probes, speculative
  objections, or old reviewer replies in the abstract, introduction, or
  discussion;
- reopens from-scratch scaling, a new seed program, or a static scalar-selector
  class that INDEX.md records as closed;
- follows an external-model rewrite before checking the current PDF, this
  contract, and the canonical owner.

Before changing reviewer-facing prose, a cold-start session must be able to
state, in its own words:

1. the one reviewer memory;
2. the strongest frozen zero-training result and its correct method identity;
3. the distinct roles of fixed-support identification, target-aware FMRoPE,
   EVQ-Cosh, LeRoPE, adaptation, and from-training evidence;
4. the exact current defect, canonical owner, and smallest proposed repair;
5. which stop rule was checked and why the edit does not violate it.

If it cannot answer all five, it has not loaded enough context to edit the
paper.
