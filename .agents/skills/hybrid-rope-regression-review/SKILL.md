---
name: hybrid-rope-regression-review
description: Independently review Hybrid-RoPE PDFs with Astra and Sol and audit a revision against its actual pre-edit baseline. Not an experiment or general documentation workflow.
---

# Hybrid-RoPE PDF regression review

Read the [current handoff](../../../paper-2027/HANDOFF.md) to identify the round
and any current no-new-version instruction. The normal rule is to snapshot
the unedited `main.pdf` as the next `history/vN.pdf` before editing, then compare
the new main.pdf with that same-round snapshot. Never use an older highest
snapshot simply because it was already there. Read-only reviews create no
version. Intermediate repairs stay in the same round.

Freeze both PDF inputs and record hashes. Launch fresh Astra and Sol reviewers
independently and concurrently with the identical
[PDF-only prompt](references/pdf-review-prompt.md). Use the requested model
identities available in the current environment; if one is unavailable, report
that specific missing review rather than claiming a dual-model pass.

Reviewers may extract text and render the designated PDFs only. They receive
no author expectations, previous scores, prior reviews, repository facts or
TeX. Four perspectives in one model are not four independent reviewers.
No target score or preferred version is prescribed.

The integrator may inspect TeX and actual reports. Check suspect equations in
rendered pages before accepting extraction complaints. Separate correctness,
lost evidence, important reproducibility details, optional extensions and
stylistic preferences. Assess the actual claim; do not require a universal
task-optimality theory or conflate missing local raw with an absent experiment.

Save review identities, findings and dispositions under the existing revision
owner. Keep immutable comparison PDFs in a repository-relative review path
when they must travel across machines; temporary absolute paths alone are not
a portable record. A later correction must be recorded separately from the
reviewed candidate. Repeat a focused comparison only if substantive new edits
could change the assessment, not to obtain a better rating.

Report regression status separately from the paper's scientific judgment.
Preservation is necessary but does not itself establish oral-level quality.
