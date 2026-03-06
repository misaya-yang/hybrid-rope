# EVQ-Cosh Submission Repository

This repository has been reduced to the minimum working surface for the EVQ-Cosh NeurIPS submission.

## Repository Contract

Visible top-level working directories are intentionally limited to:

- `scripts/`: core experiment, evaluation, and figure-generation code that directly supports the paper
- `docs/`: curated experiment reports, overview notes, and theory derivation material
- `paper_draft/`: core narrative, theory source of truth, figure/table matrix, and anonymous submission source
- `team/`: collaboration-facing handoff and planning material for advisor and lab coordination
- `results/`: classified experiment outputs and small evidence artifacts

Root-level entry documents:

- `README.md`: repository map and working contract
- `AIHANDOFF.md`: operational handoff for future AI or engineer sessions

Hidden infra directories such as `.git/`, `.codex/`, and `.claude/` remain because they are tooling, not research content.

## Start Here

1. `AIHANDOFF.md`
2. `paper_draft/mainstory.md`
3. `paper_draft/CORE_THEORY.md`
4. `paper_draft/figs/README.md`
5. `paper_draft/submission/main.tex`

## Directory Guide

### `scripts/`
Only keeps code that is directly relevant to submission-grade experiments.

- `scripts/train.py`: core training entrypoint retained for from-scratch / continued-pretraining flows
- `scripts/core_text_phases/`: the Phase 8–15 core text experiment chain used in the paper
- `scripts/supporting_eval/`: supporting evaluators kept for follow-up checks and extensions
- `scripts/data_prep/`: dataset preparation helpers still needed for reproducibility
- `figures/`: paper figure builders
- `video_temporal/`: temporal-only video extrapolation experiment
- `lib/rope/`: retained local RoPE schedule / injection library

### `docs/`
Curated research record, split into three parts.

- `overview/`: repo-level methodology, reproducibility, experiment registry
- `exp/`: key experiment reports only
- `theory/`: rigorous derivation material and theory validation notes

### `paper_draft/`
Paper-facing material only.

- `mainstory.md`: one-page spotlight-oriented narrative spine
- `CORE_THEORY.md`: main theory + evidence source of truth
- `SECONDARY_THEORY.md`: secondary claims and appendix-level material
- `figs/`: canonical paper figures and the theory-narrative-figure matrix
- `submission/`: anonymous NeurIPS draft source

### `team/`
Human collaboration surface.

- advisor handoffs
- lab coordination notes
- next-experiment planning material

### `results/`
Classified evidence store.

- `core_text/`: primary text experiments used in the main paper arc
- `theory/`: theory validation and mechanism artifacts
- `supporting_cross_model/`: cross-model / LoRA / longbench-adjacent supporting runs
- `supporting_video/`: video temporal transfer results
- `legacy/`: preserved but de-prioritized historical outputs

## Submission Anchors

The paper currently centers on three claims:

1. RoPE frequency allocation admits a closed-form variational solution, with geometric RoPE as the `tau = 0` degenerate limit.
2. EVQ beats learnable adaptive PE in DAPE-style extreme extrapolation.
3. The main systems result is `EVQ + YaRN >> Geo + YaRN`.

Everything else in the repository is organized around strengthening, reproducing, or qualifying those three claims.

## Working Rules

- New experiment reports go under `docs/exp/`.
- New paper-facing figures go under `paper_draft/figs/`.
- New non-core exploratory material should not be added back at the repository root.
- Large weights and checkpoints are intentionally excluded from version control.

## Build / Verify

The anonymous submission draft lives at:

- `paper_draft/submission/main.tex`

For runtime verification, use an activated Conda environment as required by the repo policy.
