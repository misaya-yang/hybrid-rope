# AI Handoff

## Mission

This repository is now a submission-oriented EVQ-Cosh workspace. The goal is not to preserve every historical branch of exploration, but to keep a clean, defensible path from theory to experiments to anonymous paper draft.

## Current Topology

- `scripts/`: only paper-core experiment, evaluation, and figure code
- `docs/`: curated experiment records and theory derivation material
- `paper_draft/`: narrative, theory source of truth, matrix docs, and submission source
- `team/`: advisor and collaborator coordination material
- `results/`: classified outputs

## Read Order for New Sessions

1. `README.md`
2. `paper_draft/mainstory.md`
3. `paper_draft/CORE_THEORY.md`
4. `paper_draft/figs/README.md`
5. `docs/exp/2026-03-04_phase11_L256_results.md`
6. `docs/exp/2026-03-03_passkey_mix_results.md`
7. `docs/exp/2026-03-06_phase15_750m_2k_to_4k_continue_results.md`

## Primary Evidence Hierarchy

### P0: main anchors
- `paper_draft/figs/fig2_evq_yarn_synergy.pdf`
- `paper_draft/figs/fig3_pe_dominant_scaling.pdf`
- `docs/exp/2026-03-03_passkey_mix_results.md`
- `docs/exp/2026-03-04_phase11_L256_results.md`
- `docs/exp/2026-03-05_phase11b_125m_results.md`

### P1: strong supporting evidence
- `results/core_text/phase15/`
- `results/core_text/phase9f_750m_2k_1b/`
- `results/core_text/phase14_yarn_passkey/`

### P2: secondary / scope-expanding evidence
- `results/supporting_video/video_temporal/`
- `results/supporting_cross_model/`
- `results/theory/`

## Submission Source

Anonymous draft source is here:

- `paper_draft/submission/main.tex`

The current draft should obey these structural rules:

- page 10 starts with `References`
- body keeps Figure 2 and Figure 3 as the two main figures
- Figure 1 stays supporting, not primary

## Current Narrative Lock

Do not drift away from these three claims unless new evidence justifies it:

1. Closed-form theory: RoPE frequency allocation is a variational inverse problem.
2. Extreme extrapolation: EVQ beats learnable PE in DAPE-style regimes.
3. Systems result: `EVQ + YaRN >> Geo + YaRN`.

## How To Improve The Paper Next

Optimize the paper under three principles:

1. **Protect the evidence hierarchy**
   - Keep `P0` as the real paper core: `EVQ+YaRN` and PE-dominant / DAPE-style extrapolation.
   - Keep `P1` as robustness support.
   - Keep `P2` as scope-expanding evidence only.
   - Do not promote single-seed or cross-modal results into body-level anchors without new multi-seed support.

2. **Protect the theorem / conjecture boundary**
   - Closed-form ODE solution and geometric limit are paper-grade theory.
   - `tau*` remains an empirical law / conjecture unless the derivation is tightened.
   - New mechanism ideas, including the capacity-compensation hypothesis, stay in `team/plans/` until they are experimentally closed.

3. **Optimize around the current submission skeleton, not around loose notes**
   - Submission source of truth: `paper_draft/submission/main.tex`
   - Submission structure and redlines: `paper_draft/NEURIPS_SUBMISSION_PLAN.md`
   - Core paper narrative: `paper_draft/mainstory.md`
   - Theory source of truth: `paper_draft/CORE_THEORY.md`
   - Figure / table matrix: `paper_draft/figs/README.md`
   - Known paper-side corrections and claim hygiene: `paper_draft/PAPER_ERROR_CORRECTIONS.md`
   - Missing evidence and collaborator-facing next steps: `team/open_gaps.md`, `team/plans/`

Practical rule: improve the paper by strengthening the current three-claim package, not by adding parallel storylines.

## What Has Been Deliberately Demoted

These are not headline claims in the current submission package:

- single-seed `+40pp` passkey outliers
- `Hybrid strict superiority`
- `video confirms tau*=2.0`
- `750M phase9f` as a primary result
- `750M continue` as a primary result

## Where To Put New Work

- New experiment report: `docs/exp/YYYY-MM-DD_<topic>.md`
- New theory derivation note: `docs/theory/`
- New submission-facing figure: `paper_draft/figs/`
- New advisor/collaboration note: `team/`
- New structured output bundle: `results/<bucket>/`

## Practical Warning

This repo has already been intentionally pruned. Do not recreate old root-level clutter. If a new artifact does not clearly belong to one of the five visible top-level directories, the default answer is that it does not belong in this repository.
