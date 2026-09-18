---
name: hybrid-rope-paper-editing
description: Revise the Hybrid-RoPE manuscript by chapter, preserve author decisions and evidence, and deliver a verified PDF and source archive. Not an experiment launcher or blinded reviewer.
---

# Hybrid-RoPE paper editing

Use the repository's [current handoff](../../../paper-2027/HANDOFF.md) for
current locks and revision identity, and the relevant portions of the
[author contract](../../../paper-2027/research/AUTHOR_WORKING_CONTRACT.md)
for scientific and editorial decisions. These files travel with the repository;
personal memory or Mac-only skills are not prerequisites.

Read the actual include chain in `paper-2027/main.tex`. For each requested
chapter, determine its purpose, link to neighboring sections and actual reading
obstruction. Leave effective passages intact. Verify a contested number or
claim through its owner; do not start a whole-repository provenance search.

Before edits, apply the handoff's current version rule. Preserve the unedited
PDF as the comparison baseline and record its identity. Keep changes scoped
to the authorized chapters and completed evidence. Maintain the z argument,
TailSpline's main role and the supporting scientific roles of NCP/Cosh.

Use TeX and reports for technical checks; render the PDF for reading and layout.
Build from the repository root with `bash paper-2027/compile.sh`. Check the
first page, changed pages, body end, references and affected appendices. A green
build is not a substitute for readable figures or correct claim scope.

When delivering a revised manuscript, use the
[regression-review workflow](../hybrid-rope-regression-review/SKILL.md).
Disposition findings with independent judgment and repair real regressions.

Rebuild with `python3 paper-2027/package_source.py`; independently extract and
compile the archive when it is a deliverable. Check that its PDF, active TeX
and numerical inputs match the intended revision. Update the claim map when
claims change, and the nearest index when an owner changes. Append the gain,
consistent before/after editorial estimate and possible loss to the existing
revision diary. For navigation edits run `python3 scripts/check_repository_docs.py`.

Do not recompile, create a numbered snapshot or launch reviewers for a purely
documentary audit that leaves the manuscript untouched. Preserve other tasks'
dirty work; compilation and packaging do not authorize GPU runs or uploading.
