# Manuscript revision — 2026-09-09

The active paper is **Beyond the Base: Exponent Allocation in RoPE**.
The authorized task reconstructs the complete manuscript from existing evidence,
then completes ten fresh Sol reviews that read only immutable PDF snapshots.
No new model training, evaluation, downloads, or GPU work is part of this task.

The central question is how exponent distributions affect positional geometry,
learned behavior, and adjustments to frozen models. Controlled fixed-range
training and weights-by-table crossings precede the mathematical construction.
Cosh is an analytic family from a specified convex criterion. The mature-model
section studies native-relative exponent displacements, index placement, and
boundary-matched redistribution.

- Active sources: `main.tex`, `sections/`, `appendix/` and their referenced figures/tables.
- Build: `bash paper-2027/compile.sh` from the repository root.
- Figure reconstruction: `python3 paper-2027/figs/make_exponent_revision_figures.py`.
- Source archive: `python3 paper-2027/package_source.py`.
- Asset adjudication: `research/EXPERIMENT_ASSETS_TOP15_20260909.md`.
- Claim/formula/figure/citation map: `research/EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md`.
- Independent review ledger: `research/pdf-review-rounds/20260909/`.

Rounds 1 and 2 have completed; round 3 is reviewing its immutable PDF. Each
round records its input SHA256 and the primary agent's disposition of findings.
The current build has 9 body pages and 38 total pages, no unresolved references
or overfull boxes, embedded fonts, and anonymous PDF metadata. Final review and
source-archive verification remain in progress.

The older research operational updates are preserved in Git. They are historical
context for separate experiments, not instructions to resume them during this
manuscript task. The historical NeurIPS manuscript in `../paper/` is unchanged.
