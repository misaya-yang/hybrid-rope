# ICLR 2027 manuscript optimization and simulated-review decision

> **SUPERSEDED (2026-08-27):** superseded by the five-seat panel bundle
> [`external-reviews/qwen-panel-20260826/`](external-reviews/qwen-panel-20260826/)
> (decision: Major Revision). Retained as a historical reviewer-risk record.

- **Date:** 2026-08-26
- **Status:** historical manuscript-edit decision record (superseded
  2026-08-27, see banner); internal, not anonymous supplement content
- **Role:** records the smallest pre-submission manuscript changes that can
  improve reviewer comprehension or technical credibility, and simulates the
  remaining ICLR review paths
- **Not:** a numerical evidence owner, a new experiment plan, or manuscript
  prose to copy verbatim
- **Objective:** freeze a scientifically correct, reviewer-legible submission;
  do not reopen method search
- **Post-submission theory authority:**
  [`ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md`](ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md)

Every number below inherits the scope of its linked canonical owner. The active
PDF and LaTeX remain the reviewer-visible truth until the edits in this report
are actually applied, compiled, and visually verified.

---

## 1. Decision

The paper is complete enough to submit and has a coherent acceptance case. No
fatal defect was found in the fixed-support identification, full-RoPE geometry,
co-adaptation evidence, mature-scale routing, or promoted result identities.
The remaining score movement is editorial and theoretical cleanup, not new
compute.

Before submission, make one theory correction and five bounded wording or
validity corrections:

1. remove the obsolete single-power-law interpretation of `tau_*(L)` and the
   derived `p approx 0.85` diagnostic from Appendix A.11;
2. state the target-matched reversal numerically in the body and explain its
   exact claim boundary;
3. separate EVQ-Cosh evaluation from the independent mature-checkpoint derived
   profile in Contribution (iii);
4. state the fixed-checkpoint/evaluation-row unit beside the frozen-profile
   intervals instead of writing generic statistical indistinguishability;
5. remove the ambiguous learned-table extrapolation sentence and label the two
   Qwen numbers explicitly;
6. add the live ICLR 2027 author-quota and author-roster gates to the submission
   checklist.

After these changes, rebuild, run the curated package checks, inspect all pages,
perform a final number-to-owner sweep, and freeze the submission. Do not add a
new table candidate, seed, model, capability run, or theory branch.

---

## 2. Verified baseline

Read-only review on 2026-08-26 established:

- branch/upstream at review time: `main_0726` / `origin/main_0726`, clean and
  `0/0` ahead/behind at commit `9874b480af572ee0d55e71091755cbe74defa08e`;
- active PDF SHA-256:
  `5c5107126ae99d659c000fefd7b86fbd36178b4844a0b0b82e84bedd2db59c0e`;
- 31 US-Letter pages, with the main body ending on page 9;
- abstract length about 195 words;
- all 31 pages visually inspected: no clipping, overlap, missing glyphs, or
  unreadable table/figure defect at normal PDF zoom;
- current primary-source positioning of FMRoPE, LeRoPE, AdaRoPE, and Jet-Long
  is broadly accurate;
- live ICLR 2027 author guidelines still require a genuine abstract by
  2026-09-18 AoE, a full paper by 2026-09-25 AoE, at most nine submission-body
  pages, double blindness, and a mandatory AI-use statement.

This review did **not** compile, rebuild the supplement, rerun tests, or
recompute raw experiments because its inspection phase was read-only. Existing
build/package receipts remain owned by `../HANDOFF.md`.

---

## 3. The reviewer memory

The strongest bounded one-sentence memory is:

> In a representative 4K RoPE head, 46 nominal slow dimensions carry only
> 2.00 block-whitened effective positional dimensions; after separating
> support from interior allocation, a fixed-support intervention moves only
> the interior frequencies, and the same coordinate changes a released 1.485B
> checkpoint from 0.56% to 60.47% on 16K RULER.

Keep the qualifiers `representative`, `block-whitened`, `positional`,
`fixed-support`, `released checkpoint`, and the endpoint identity. Do not
shorten this into a universal statement that every RoPE head wastes one third
of its channels or that the slow bands are unused/reclaimable.

The five-step story remains:

1. **Observation:** finite geometric tables spend many slow pairs on nearly
   duplicate positional directions.
2. **Coordinate:** `x_k = a + R z_k` separates sampled support from normalized
   interior allocation.
3. **Identification:** endpoints fixed, 30 interior frequencies moved, all
   three training seeds favor the intervention at every fixed-range OOD length.
4. **Construction and mechanism:** EVQ-Cosh is one closed-form witness under a
   stated surrogate; full subspace geometry, the frozen transplant obstruction,
   and weights-by-table crossings explain the budget and co-adaptation boundary.
5. **Consequences:** architecture, continuation, scale, adapted capability,
   causal source use, and a frozen fixed-support corollary each keep their own
   estimand.

---

## 4. Required manuscript changes

### P0 — remove the stale `tau` power-law diagnostic

**Current source:** `appendix/a1_proofs.tex`, Appendix A.11 and the later
load-moment/f-divergence paragraphs.

The appendix currently fits `tau_*(L)` to a single `L^{-gamma}`, sets
`gamma=0.5` as a target, and derives an exponent-matched `p approx 0.85`.
The completed O3 derivation in
[`three_completions/optimization_notes.md`](three_completions/optimization_notes.md)
shows that the model itself does not predict a single power law:

\[
\gamma_{\rm eff}(L,b)
=\frac12-
\frac{g'(\varphi_*)}{2g(\varphi_*)\log b},
\qquad
g(\varphi)=\varphi(1-\varphi)(2-\varphi).
\]

Therefore the `p approx 0.85` diagnosis is a consequence of fitting the wrong
functional form. Remove the target-exponent interpretation, the
exponent-matched row, and the paragraph that promotes this diagnostic. Retain
the Pearson chi-square channel-load motivation and state that `Q_1(L,b)` makes
the effective exponent depend on length and base.

Do **not** replace this material with EVQ-Cosh-R or the Nystrom/Fredholm route.
Those are internal theory directions without behavioural validation and belong
to the closed/static-score research boundary, not this submission.

### P0 — pre-empt the target-matched reversal in the body

**Current source:** `sections/02_identification.tex` and the corresponding
Discussion paragraph.

Canonical owner:
[`EXACT_RANGE_151M_3SEED_RESULT_20260820.md`](EXACT_RANGE_151M_3SEED_RESULT_20260820.md)
Sections 3-4.

The body currently says only that target retargeting reverses the ordering.
The reviewer-relevant result is anchored EVQ-Cosh minus FMRoPE
`+0.060/+0.227/+0.460` NLL at `2x/4x/8x`, with `0/3` seeds favoring anchored
EVQ-Cosh. Leaving the values to the appendix invites the interpretation that
the comparison wins only against an untuned baseline.

Recommended bounded paragraph:

> Under target-matched support, anchored EVQ-Cosh trails FMRoPE by
> `+0.060/+0.227/+0.460` NLL at `2x/4x/8x`, with FMRoPE favored in all three
> seeds. The fixed-support result therefore identifies allocation `z`; it does
> not establish an additive gain over target-aware support selection.
> Conversely, after the same fourfold support move on a released checkpoint,
> geometric and non-geometric interior allocations score `0.56%` and
> `60.47%`, showing that support selection does not exhaust the design.

Do not turn this into a support-by-allocation law: the protocol identifies a
conditional ordering, not an interaction surface.

### P1 — separate EVQ-Cosh from the mature derived profile

**Current source:** `sections/01_intro.tex`, Contribution (iii).

The current sentence can be read as if the mature OLMo/Qwen derived profile is
the EVQ-Cosh table. It is not. Recommended replacement:

> (iii) We derive EVQ-Cosh as the unique minimizer of a stated convex
> surrogate and evaluate it across from-scratch and continued-training
> regimes; separate fixed-support controls show that the allocation coordinate
> remains consequential in frozen pretrained checkpoints.

This preserves both pillars without splicing their estimands.

### P1 — name the bootstrap unit instead of generic significance

**Current source:** `sections/00_abstract.tex`, `sections/01_intro.tex`,
`sections/04_experiments.tex`, and Figure 1 text where necessary.

Canonical owner:
[`attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`](attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md).

The intervals condition on one fixed checkpoint and task set with evaluation
rows resampled. They are not checkpoint-population, task-population, model, or
training-seed uncertainty. Replace generic `statistically indistinguishable`
with:

> The coarse ramp scores `61.04%`; its paired evaluation-row interval against
> the derived profile includes zero on this fixed checkpoint and task set.

In the abstract, `reproduces the recovery` is safer than `matches` if the unit
cannot fit. Preserve the exact conditional interval in the experiment paragraph
or appendix.

### P1 — repair two clarity/identity sentences

1. In `appendix/a5_identification.tex`, the learned inverse-frequency row
   improves over Geo in this protocol, so it does not directly support the
   adjacent statement that learned tables degrade more sharply than standard
   RoPE under naive extrapolation. Replace it with:

   > In this protocol, the learned table improves over Geo but remains weaker
   > than EVQ-Cosh; the comparison does not identify whether the gap comes from
   > the final table, optimization dynamics, or the positional-parameter
   > learning rate.

2. In `sections/04_experiments.tex`, replace the unlabeled Qwen pair
   `57.75%/66.50%` with `same-support geometric/derived is 57.75%/66.50%`.

### P0 validity — extend the submission checklist

The live ICLR 2027 author guide adds explicit author-level gates not currently
enumerated in `SUBMISSION_CHECKLIST.md`:

- no new author after the abstract deadline;
- no author on more than 20 submissions;
- the special quota for authors whose submissions have no eligible reciprocal
  reviewer;
- at least one registered reviewer per submission unless the official
  exemption applies;
- the six-review obligation for authors on three or more papers.

These are author confirmations, not automated repository checks. Add them as
unchecked submission-day items and preserve the existing author-profile
confirmation.

---

## 5. Historical-review closure

The current ICLR paper has materially answered the NeurIPS panel's decisive
requests:

| Historical concern | Current state |
| --- | --- |
| novelty versus FMRoPE / dead-frequency observations | direct citation, paper-faithful FMRoPE arm, bitwise fixed endpoints, explicit `z` distinction; substantially answered |
| weak scale and diagnostic-only evaluation | 432M/750M/1.485B/8B, RULER, QA, natural NLL, frozen checkpoints, and causal source use; substantially answered |
| allocation confounded with tuning/parameterization | three-seed exact range, pre-specified M4 factorial, matched exponential, same-support mature controls; substantially answered |
| surrogate-to-Cosh-to-operating-rule chain incomplete | claim ceiling is now bounded, but Appendix A.11 contains the stale power-law diagnostic; this remains the main theory cleanup |
| learned comparator identified as DAPE | corrected to a learned table; the remaining final sentence should be made protocol-local |

The old acceptance ceiling was evidence. The current ceiling is whether a
reviewer understands the identified object and its boundaries before deciding
that coarse-ramp parity or target matching makes the contribution unnecessary.

---

## 6. Simulated review panel

### Reviewer A — theory/mechanism: weak accept after P0 theory fix

**Accept case:** clean fixed-support intervention; exact full-subspace identity;
properly scoped collapse and transplant theorems; table/weights crossing
prevents static-geometry overclaim.

**Reject case:** the surrogate is not a trained-loss objective, and a stale
single-power-law appendix makes the operating-rule chain appear internally
inconsistent.

**Score mover:** remove the obsolete A.11 diagnosis and keep Cosh uniqueness
strictly conditional on the stated surrogate.

### Reviewer B — empirical/systems: borderline to weak accept

**Accept case:** the paper is no longer small-model only; the released OLMo
fixed-support contrast is visceral and controlled; endpoint identities remain
separate; 8B source blocking supplies a causal endpoint.

**Reject case:** mature protocols use limited checkpoint populations and often
one training seed; RULER is task-family evidence; natural-task outcomes are
heterogeneous.

**Score mover:** routing and honest scope, not a 7B/70B ablation. No new compute
is needed for this submission.

### Reviewer C — novelty/FMRoPE/LeRoPE: remaining weak-reject risk

**Accept case:** fixed sampled support, only interior frequencies moved, three
paired training seeds, and an independent frozen-checkpoint same-support
corollary define a narrower and cleaner object than merely producing a
non-geometric table.

**Reject case:** target-matched FMRoPE wins; a coarse ramp matches the detailed
profile; learned frequency tables already exist; static geometry is not a
behavioural ranker.

**Score mover:** state the target-matched values and claim boundary in the body,
separate EVQ-Cosh from the mature derived table, and explain that the theorem
is a budget account while trained ordering belongs to controlled experiments.

### Simulated AC synthesis

After the required edits, the clean acceptance sentence is:

> The paper identifies a previously conflated finite-table design variable
> through a three-seed fixed-support intervention, supplies exact geometry and
> co-adaptation boundaries, and demonstrates a large frozen-checkpoint
> consequence without claiming profile optimality.

Without the target-matched paragraph or the A.11 correction, the same panel can
reasonably split between weak accept and weak reject.

---

## 7. Do not change before submission

- Do not run new seeds, larger checkpoints, or capability suites.
- Do not tune `tau`, amplitude, bands, movement exponent, routing, or the
  coarse ramp on evaluation tasks already inspected.
- Do not promote EVQ-Cosh-R, Nystrom/Fredholm, per-head allocation, or the
  mature learned direction into the paper.
- Do not delete correct proofs merely because the appendix is long.
- Do not demote the 432M/454M/750M/1.485B/8B/video evidence; route each result
  to its actual role.
- Do not change the author-confirmed AI-use statement without renewed factual
  confirmation and a live policy check.
- Never edit or compile `paper/`.

---

## 8. Company-computer continuation order

1. Verify `main_0726`, upstream, clean worktree, and ahead/behind state after a
   fetch/fast-forward-only pull.
2. Read `AGENTS.md`, `INDEX.md` Sections 3.4 and 6, `paper-2027/HANDOFF.md`, and
   this report.
3. Apply only the edits in Section 4, replacing lower-leverage prose rather
   than expanding the nine-page body.
4. Rebuild the active paper only, run focused repository/navigation/package
   checks in Conda `aidemo`, regenerate the curated supplement, and visually
   inspect all pages affected by the edit plus all nine body pages.
5. Verify `paper/` is unchanged; run `git diff --check`, owner-number review,
   anonymity/private-path scan, and scoped staged-diff review.
6. Recheck live ICLR policy, author roster/quotas, dual-submission outcome,
   OpenReview profiles, and exact title/abstract equality immediately before
   upload.

The current PDF/package hashes remain the last validated build until Step 4 is
completed after source edits.

---

## Claim boundary

This report changes no scientific result and owns no numerical claim. It
selects and orders manuscript work from already completed owners. If this
report conflicts with `AGENTS.md`, `INDEX.md`, a canonical owner, or the active
manuscript, the higher authority wins.
