# `paper-2027/` — active ICLR 2027 package

This is the only active manuscript. `../paper/` is the immutable NeurIPS 2026
baseline and must never be edited, compiled, moved, or regenerated.

## Read for the task

1. [`../AGENTS.md`](../AGENTS.md) and [`../README.md`](../README.md) — rules and
   paper core.
2. [`HANDOFF.md`](HANDOFF.md) — latest PDF/Git/authorization state and actions.
3. Current `main.tex`/`main.pdf` — reviewer-visible wording.
4. Search [`../INDEX.md`](../INDEX.md) and open one owner only when verifying or
   changing a scientific claim.

Use [`NARRATIVE_GUIDE.md`](NARRATIVE_GUIDE.md) for stable manuscript strategy,
[`REVISION_BRIEF.md`](REVISION_BRIEF.md) for the bounded revision contract, and
[`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) for release gates. Historical
reviews, closed plans, external-model bundles, and `research/` are not default
cold-start inputs.

## Package contract

- `main.tex` is the only manuscript entrypoint; `sections/` and `appendix/`
  contain reviewer-facing source.
- `figs/`, `tables/`, and `refs/` contain compiled assets and bibliography.
- `compile.sh`/`build.mk` check format. Compilation does not validate science.
- `research/` contains internal owners, receipts, analyses, and history. It is
  excluded from the anonymous supplement except by curated allowlist.
- Packaging must use the curated ICLR profile from the repository root; never
  archive the root or compile `../paper/`.

The handoff owns live checks and machine availability. Scientific claims route
through `INDEX.md` to a direct owner. No plan, preflight, review, script, or
document in this directory authorizes compute, Git publication, or upload.
