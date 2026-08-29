# `paper-2027/` — active ICLR 2027 submission package

This directory contains the only active manuscript. The NeurIPS 2026 baseline
under `../paper/` is immutable and must never be edited, compiled, moved, or
regenerated.

## Authority and cold start

From the repository root, read:

1. [`../AGENTS.md`](../AGENTS.md) — rules and claim ceilings;
2. [`../INDEX.md`](../INDEX.md) — durable theory/evidence/code routing and
   research priority;
3. [`HANDOFF.md`](HANDOFF.md) — live state, current authorization, and receipts;
4. [`NARRATIVE_GUIDE.md`](NARRATIVE_GUIDE.md) — stable manuscript strategy;
5. current `main.tex`/`main.pdf`, then the canonical owner routed by
   [`research/README.md`](research/README.md).

Closed revision plans, the August author-verdict ledger, the old Codex/Claude
review log, historical rebuttal material, and external-model bundles are not
cold-start inputs.

## Stable package structure

- `main.tex` is the only manuscript entrypoint and declares the compiled
  section/appendix order.
- `sections/` and `appendix/` contain reviewer-facing source.
- `figs/`, `tables/`, and `refs/` contain compiled assets and bibliography.
- `compile.sh` / `build.mk` implement the paper-format checks.
- [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) defines stable release
  gates; live pass/fail state belongs only in the handoff.
- [`CHANGES_FROM_NEURIPS2026.md`](CHANGES_FROM_NEURIPS2026.md) owns the durable
  contribution-distinctness boundary.
- `research/` contains internal owners, receipts, analyses, archived plans, and
  preflights. It is not manuscript prose and is excluded from the anonymous
  supplement except through an explicit curated allowlist.

Current PDF/source synchronisation, page counts, hashes, supplement receipts,
policy checks, author actions, and worktree state are volatile; consult the
handoff rather than recording them here.

## Build, test, and packaging

Canonical commands and machine assignments live only in
[`../README.md`](../README.md) under “Build and validate.” The work machine owns
the `aidemo` environment, packaging, and final cross-environment validation.
The low-configuration personal PC may run local LaTeX/Tectonic builds and visual
PDF iteration alongside documentation, planning, and lightweight checks; do not
infer a repository failure from missing Conda there.

Compilation establishes layout/format health only. It does not validate
scientific claims. Packaging must use the curated ICLR profile from the
repository root; never archive the repository root and never compile `../paper/`.

## Evidence and research boundaries

Every number or claim must resolve through [`../INDEX.md`](../INDEX.md) §3 and
the named canonical/raw owner. A plan, preflight, script, code manifest, launch
log, external-model review, or historical PDF receipt is not a result.

The durable post-submission research order lives in `INDEX.md` §6. Current
authorization lives only in the handoff. No document in `research/` authorizes
GPU work, paid compute, an upload, or Git publication by itself.
