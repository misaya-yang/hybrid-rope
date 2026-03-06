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
- To see the newest mechanism hypothesis: `team/plans/capacity_compensation_hypothesis.md`

## Highest-ROI Next Tasks

1. Clean larger-scale downstream evaluation.
2. DSR-style retrieval evaluation.
3. Any replication or extension of the Phase 11 / Phase 14 / Phase 15 chain.
4. A scale/training-sufficiency sweep that tests whether stronger models absorb short-range positional deficits while preserving EVQ long-range gains.

## What The New Hypothesis Means Operationally

The current mechanism bet is:

- high-frequency positional detail may become partly redundant as model capacity and training increase,
- low-frequency long-range structure remains the non-redundant part,
- therefore EVQ should asymptotically lose less at `1x/2x` while still winning at `8x/16x`.

That means the most useful new experiments are not random extra runs. They are runs that separate:

1. model scale,
2. training sufficiency,
3. retrieval saturation versus exact long-range recovery.

## Guardrails

- Do not promote single-seed evidence into main-paper anchors.
- Do not put exploratory notes back into the repository root.
- Keep new human-facing status notes under `team/`.
- Keep new experiment reports under `docs/exp/`.
