# RoPE Has a Spectral Budget

Most RoPE extension work is framed around the overall table, scalar base/range,
or position transport while leaving the normalized interior exponent schedule
implicit or geometric. This project isolates that under-studied coordinate:
does the **interior allocation** `z` matter independently of support/base, and
how does it interact with the weights trained to use it?

## Paper core

The paper's identity is the decomposition
$x_k=-\log\omega_k=a+Rz_k$: a finite RoPE table has distinct sampled-support
$(a,R)$ and interior-allocation $z$ coordinates.

1. Paired fixed-support training interventions identify $z$ as independently
   consequential; target-aware support retargeting reverses the tested ordering
   and shows that the two coordinates interact.
2. Exact phase-invariant sin/cos geometry gives the spectral-budget identity,
   slow-frequency collapse, weight--table co-adaptation boundary, and frozen
   transplant obstruction.
3. A convex surrogate yields closed-form, zero-learned-parameter EVQ-Cosh,
   unique only within that surrogate and not the identity of the paper.
4. Structured allocations have behavioural consequences through three distinct
   routes: fully frozen zero-training, matched low-rank adaptation, and
   from-training/co-adaptation. The frozen route is the strongest practical
   no-update result; the routes retain separate estimands.

The active manuscript is `paper-2027/`; `paper/` is immutable.

## Evidence hierarchy

1. **Causal core:** fixed-support from-scratch and fully frozen pure-`z`
   controls isolate allocation from support, routing, gain, and weight updates.
2. **Strongest practical result:** fully frozen structured allocations produce
   large no-update mature-checkpoint gains.
3. **Lifecycle triangulation:** from-scratch/co-adapted, matched LoRA/adaptation,
   and mature frozen interventions show that the same coordinate matters at
   three stages without pooling their estimands.
4. **Breadth:** scarce-channel MLA, 750M/1.485B/8B, retrieval, downstream, and
   Video-DiT results establish scope at their own protocols despite limited GPU
   depth.
5. **Related evidence:** LeRoPE supports the possibility of useful in-window
   allocation, but it is not our causal owner or matched comparator.

## Current state and next action

The manuscript design is frozen and uses completed evidence; no new submission
experiment is planned. The immediate milestones are:

1. **2026-09-17:** freeze the title, abstract, author roster, and author metadata.
2. **2026-09-18, 11:59 PM AoE:** submit the official abstract and metadata.
3. **2026-09-25:** submit the fully verified paper and anonymous supplement.

The current priority is therefore an owner-by-owner audit of the title,
abstract, first-page path, Figure 1, numbers, method identities, endpoint/seed
scope, anonymity, author metadata, and live venue requirements. Build,
packaging, and final visual review follow as submission gates.

Post-submission research is separate. Its author-ordered first direction is a
deterministic static pure-`z` table on a frozen checkpoint, derived before LM
evaluation without learning or loss-based frequency search. Protocol and assay
validity are mandatory execution gates; they do not replace that direction.

## Read only what the task needs

Default cold start:

1. `AGENTS.md` — rules and evidence discipline.
2. This README — paper core and current direction.
3. [`paper-2027/HANDOFF.md`](paper-2027/HANDOFF.md) — latest Git, PDF,
   validation, authorization, and author actions.

Then expand only as needed:

- For a claim: search [`INDEX.md`](INDEX.md) for the exact question and open
  only its linked owner/raw artifact.
- For why the question changed: read
  [`TIMELINE.md`](paper-2027/research/history/TIMELINE.md).
- For manuscript work: read [`paper-2027/README.md`](paper-2027/README.md), then
  current TeX/PDF and the relevant owner.
- For a mature-checkpoint result or theory note: enter
  [`attention-aware-retrofit/`](paper-2027/research/attention-aware-retrofit/)
  only after `INDEX.md` routes the question.

Do not read the timeline, theory tree, result tree, old session summaries, or
archive wholesale during routine cold start.

Directory routing lives in [`paper-2027/README.md`](paper-2027/README.md) and,
for a specific claim, `INDEX.md`. Build/release gates live in
[`paper-2027/SUBMISSION_CHECKLIST.md`](paper-2027/SUBMISSION_CHECKLIST.md); live
receipts and machine availability live in the handoff. Missing work-machine
checks must be recorded as skipped, not recreated on the personal PC.
