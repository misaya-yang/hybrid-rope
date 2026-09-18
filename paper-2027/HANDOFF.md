# Current manuscript handoff

Updated 2026-09-18. This is the current continuation packet; older narrative
handoffs are [archived](../docs/archive/navigation_20260918/index.md).
Read one task route, not the whole history.

## Current state

- Title: **Beyond the Base: Frequency Allocation in RoPE**.
- Title and 167-word abstract are frozen: [submission text](title_abstract.txt),
  [final audit](research/revision_20260918/FINAL_TITLE_ABSTRACT.md).
- Delivery: [main.pdf](main.pdf), [source archive](exponent-allocation-source.zip).
  Nine main pages, 35 total; no numbers in the abstract and no first-page figure.
- September 18 integrated Kanana, distance-response theory and table cleanup,
  then chapter editing and the three-method concept Figure 1. The 432M learning
  curve and complete per-task main figure remain. Revision stages and exact
  artifacts are in the [revision owner](research/revision_20260918/README.md).
- Astra/Sol PDF-only comparisons are complete. Final abstract edits left
  pages 2–35 unchanged. No model experiment was run by the manuscript task.
- The OpenReview edit form was filled with the final abstract and previewed;
  submission/save was left to the author. Do not infer that the final submission
  occurred from the local PDF or the filled form.
- Pro's answer to the [three final-week questions](research/revision_20260918/PRO_FINAL_WEEK_QUESTIONS.md)
  has been reviewed. The author-approved corrected fresh-64 T-C
  distance-by-structured-context confirmation is complete: its `+3.125pp`
  interaction has a paired 95% interval `[-4.6875,+10.9375]pp` and ceiling/floor
  cells, so the [experiment owner](../experiments/tc_distance_competition_20260918/README.md)
  records it as unresolved. It does not change the manuscript, reopen the
  frozen title/abstract, or authorize other queues.

## NON_NEGOTIABLE_CONSTRAINTS

The full [author working contract](research/AUTHOR_WORKING_CONTRACT.md) records
writing and research corrections. The highest-risk constraints are:

1. **Preserve the scientific identity:** z is the design variable; fixed
   endpoints identify its effect. TailSpline is the main frozen construction;
   NCP/Cosh retain supporting roles and their actual evidence.
2. **Preserve the frozen title/abstract:** routine body edits do not reopen them.
3. **Preserve evidence and scope:** do not drop valid results, invent outcomes,
   conflate aggregation units, or turn a missing local file into a paper defect.
4. **Use declared authorization:** no paid run, queue restart, machine change or
   external submission follows merely from a plan or a deadline target.
5. **Version the correct baseline:** the normal new-round rule archives the
   then-current unedited PDF as the next history/vN. The author's explicit
   no-new-version instruction governed the subsequent September 18 chapter and
   abstract passes, so history still ends at v4. Do not retroactively create v5
   or relabel v4 as the baseline of those later passes. Record the baseline of
   any future round before editing; a current author override takes precedence.

## ARCHITECTURE_DECISIONS

The active chain in `main.tex` is: abstract → introduction → allocation
variables → controlled findings → positional structure/content use → explicit
allocations → model quality → related work → conclusion. Appendices A–F follow
geometry, constructions, identification, frozen results, native studies and learning.

Important conclusions to preserve:

- Range alone and total movement alone do not determine task quality.
- Complete-pair positional overlap differs from learned frequency use; lower
  normalized positional rank can coexist with better task performance.
- TailSpline exactly minimizes the declared one-sided discrete objective.
  Its phase/response results have stated conditions; they are not Nyquist,
  frame-stability or end-to-end task-optimality theorems.
- Natural QA is distinct from RULER and LM. NCP's native LM/context-use results
  are established; broad native real-task enhancement remains a research goal.
- Kanana is an official runtime YaRN comparison. Do not claim a new attention
  architecture or verified YaRN continuation-training provenance from that result.

## FILE_LEDGER

| Task | Authoritative files |
|---|---|
| Edit prose/math | `main.tex`, active `sections/` and `appendix/` includes |
| New result values | Result owner → `research/evidence/asset_registry.json` → claim map |
| Current experiment ownership | [experiment index](../experiments/index.md) |
| Current research decisions | [research index](../docs/research/next_stage_20260912/index.md) |
| Regenerate current concept figure | `figs/make_intro_claim.py`, `figs/intro_claim_inputs.json` |
| Evidence tables and retained figures | `figs/make_revision_evidence.py`, `figs/make_allocation_value.py` and their inputs |
| Compile/package | `compile.sh`, `package_source.py`, `figs/runtime_source_snapshot.json` |
| Decisions and learned lessons | `research/PAPER_REVISION_DIARY.md`; append, do not replace |
| Portable review inputs | `research/revision_20260918/portable_review_artifacts.json` |
| Cross-machine transfer | [continuation guide](../docs/maintenance/CROSS_MACHINE_CONTINUATION.md) |

## REJECTED_APPROACHES

Do not restore a three-method evidence collage, a results teaser as the
introduction concept figure, or mechanical whole-paper rewriting. Do not
replace the identified gap with a claim that previous work studied only base.
Do not force a universal theory, a task-performance waterbed law or a
checkpoint-calibrated construction. Cosh is not a separate FMRoPE contest.
NTS2/CA-NCP/Phi exploratory outcomes remain in their owners, not new paper claims.

## RISKY_REGIONS

- N/task=10 direct-baseline rows and larger panels have different sample sizes;
  small panels can overlap large ones. Keep precision and comparison identities.
- Native-QA99 has question-weighted and source-equal variants. Use the paper's
  declared primary estimand, retaining sensitivity separately.
- Source packaging uses a frozen runtime revision plus explicit additions;
  current experimental scripts may contain another task's changes.
- This working tree contains uncommitted work. Check status and preserve it.
  A PC checkout at the same commit need not contain this final manuscript.
- Old absolute review paths are historical invocation metadata. Use the
  portable artifact manifest for current access, not a Mac temporary directory.

## Next action by task

For a manuscript task, use the repository's
[editing skill](../.agents/skills/hybrid-rope-paper-editing/SKILL.md), then the
[PDF review skill](../.agents/skills/hybrid-rope-regression-review/SKILL.md).
For research execution, use the [experiment workflow](../docs/research/protocols/EXPERIMENT_WORKFLOW.md).
For a specific result, follow its owner directly. Today the paper is ready;
wait for a substantive new result or useful Pro recommendation before another
revision. Do not manufacture work to meet a daily-version cadence.
