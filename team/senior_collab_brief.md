# Senior Collaborator Brief

## Repository Contract

The repo has been reduced to five working surfaces:

- `scripts/`: core experiments, eval, and figure generation
- `docs/`: curated theory and experiment records
- `paper_draft/`: narrative, theory, figures, and anonymous draft
- `results/`: classified outputs
- `team/`: human collaboration notes

## What Is Already Stable

- Anonymous submission source compiles from `paper_draft/submission/`.
- Main figures are in `paper_draft/figs/`.
- Core experiment reports are in `docs/exp/`.
- Phase8–Phase15 core scripts are grouped under `scripts/core_text_phases/`.

## What You Should Use First

- To understand the paper story: `paper_draft/mainstory.md`
- To understand the evidence hierarchy: `paper_draft/figs/README.md`
- To run the main text experiment family: `scripts/core_text_phases/README.md`
- To see what is still missing: `team/open_gaps.md`

## Highest-ROI Next Tasks

1. Clean larger-scale downstream evaluation.
2. DSR-style retrieval evaluation.
3. Any replication or extension of the Phase 11 / Phase 14 / Phase 15 chain.

## Guardrails

- Do not promote single-seed evidence into main-paper anchors.
- Do not put exploratory notes back into the repository root.
- Keep new human-facing status notes under `team/`.
- Keep new experiment reports under `docs/exp/`.
