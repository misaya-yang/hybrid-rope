# Post-review delta note — Codex exposition pass vs. panel roadmap

**Date:** 2026-08-27 · **Editor:** editorial synthesizer of the qwen-panel-20260826 bundle
**Scope assessed:** uncommitted working-tree changes vs `f9804fb` made by Codex in two post-checkpoint passes on 2026-08-26 22:48–23:37 (abstract thesis pass + exposition/layout pass), per `research/CODEX_CLAUDE_PAPER_REVIEW_LOG.md` entries and HANDOFF receipt.

## What changed

1. Abstract: 189 → 159 source words; one numerical result group (mature fixed-support RULER 0.56% → 60.47%); "three-seed fixed-support" restored; EVQ-Cosh introduced as "constructive witness"; closing sentence returns to the title claim.
2. §4 roadmap: three number-heavy route summaries → one role-based navigation paragraph.
3. §5 discussion: role-based restructure; practical reporting consequence added; "in the tested protocols" scope retained.
4. §3 after Theorem 1: exact title interpretation added — fixed $\operatorname{tr}\Gamma=2K$ conserves nominal rotary dimension; allocation changes its distribution across positional directions.
5. App. E: `\FloatBarrier` repairing the 1.485B figure boundary.
6. Rebuild verified by Codex: 8 body / 27 total pages, zero undefined references/citations, 0pt overfull, anonymous, fonts embedded; tests 181/181 + supplement 144/144; HANDOFF receipt and review log updated. Constraints respected: no commit/push/upload/GPU.

## Assessment against the roadmap in `06_editorial_decision.md`

**Positive interactions.**
- The Theorem 1 addition substantively addresses the EIC-W7/DA-5 concern that "spectral budget" reads as a performance predictor: the budget is now defined, at the identity itself, as an availability/distribution statement. Item A5 is partially absorbed by this placement (the intro clause remains optional).
- The abstract keeps the "witness" hedge, says "comparable recovery" for the ramp (consistent with the derived-minus-ramp CIs containing zero), and drops the word "causally" from the identification sentence — all consistent with the A2 rescoping direction.
- Codex's author-approved refusal to move the 1.485B crossover figure into the body is consistent with R1-W10/DA-8 scoping.

**Implicit factual correction (verify before citing).** The HEAD (f9804fb) abstract called the frozen checkpoint "1.485B"; the frozen experiment uses **OLMo-2-0425-1B-Instruct** (App. E, `sec:frozen-fixed-support`) — the working tree's "billion-scale" is factually accurate and the HEAD phrasing was not. The panel reviewed the working-tree phrasing, so no panel finding rests on the error.

**Watch items (mild regressions, both author-directed and compensable).**
1. The YaRN 7.94% anchor left the abstract and the §4 roadmap intro. It remains in §4.1 body prose (+52.54 with CI) and Table `frozen-fixed-support`, so EIC-W5/A2's calibration function is preserved at the body level; the dual-anchor obligation should now be satisfied in §4.1, not the abstract.
2. The abstract's "Under the stated uniform separation prior" qualifier and the 23-pairs/2.00 figures were removed along with all static numbers. The abstract no longer makes a prior-specific numerical claim, so R3-W2's fix target (a prior-sensitivity table in the body) is unaffected.

**Unaffected mandatory items.** None of the starred items were addressed by this pass, as expected for an exposition pass: A1 (target-matched reversal disclosure — the abstract still reads "improves every tested OOD length in every seed" unqualified), A2 completion, B1–B3 (related work/NTK/bibliography), C1, D1–D3, D6, and the E-hygiene items.

## Decision status

Unchanged: **Major Revision**. The Codex pass neither addresses nor aggravates any validated DA CRITICAL; it improves claim-tone in two places (Theorem 1 interpretation, discussion scoping) and removes one latent factual error. The revision roadmap in `06_editorial_decision.md` §5 remains the operative checklist.
