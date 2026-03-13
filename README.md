# EVQ-Cosh: Closed-Form Frequency Allocation for Long-Context Extrapolation

Official code repository for the NeurIPS 2026 submission.

**TL;DR** — We derive a closed-form variational solution for RoPE frequency allocation and show that the one-parameter EVQ-Cosh family, combined with Progressive YaRN, achieves 100% passkey retrieval at 8× training length while geometric RoPE + YaRN collapses to 61–65%.

## Quick Start

```bash
# Environment
conda create -n evq python=3.10 && conda activate evq
pip install torch transformers datasets tqdm matplotlib

# Core τ-sweep experiment (50M model, ~4 hours on consumer GPU)
python scripts/core_text_phases/run_evq_sweep.py --tier 50m

# Reproduce paper figures
python scripts/figures/fig1_neurips.py
python scripts/figures/fig2_evq_yarn_orthogonality.py
python scripts/figures/fig3_pe_dominant_scaling.py
```

## Repository Structure

```
paper/                  LaTeX source, figures, and tables
├── main.tex            NeurIPS submission entry point
├── figs/               All paper figures (PDF + PNG)
├── sections/           Section .tex files
├── appendix/           Appendix .tex files
├── tables/             Table .tex files
└── refs/               BibTeX references

scripts/                Experiment and evaluation code
├── train.py            Core training entrypoint
├── core_text_phases/   Main experiment chain (Phase 8–21)
├── figures/            Paper figure generation scripts
├── data_prep/          Dataset preparation helpers
├── supporting_eval/    Supporting evaluators
├── lib/rope/           RoPE schedule library (EVQ-Cosh, inject, learnable)
├── m4_max_36gb/        Theory numerical verification
└── mac_train/          Local training scripts

docs/                   Curated research documentation
├── exp/                Experiment reports (YYYY-MM-DD_slug.md)
├── overview/           Methodology, reproducibility, registry
└── theory/             Derivations and validation notes

results/                Experiment output artifacts
├── core_text/          Primary text experiments
├── theory/             Theory validation outputs
├── supporting_cross_model/  Cross-model comparisons
├── supporting_video/   Video temporal transfer
└── legacy/             Historical outputs

team/                   Collaboration materials
├── briefs/             Advisor and collaborator briefs
├── status/             Active gap tracking
├── plans/              Experiment plans
└── archive/            Historical handoffs
```

## Key Results

The paper validates three claims across 99 controlled runs at 50M–750M scale:

1. **Closed-form solution**: RoPE frequency allocation admits a variational solution φ_k(τ) with geometric RoPE as the τ→0 degenerate limit.
2. **EVQ > Learnable PE**: EVQ matches or exceeds DAPE-style learnable frequency allocation in extreme extrapolation, without per-run optimization.
3. **EVQ + YaRN synergy**: At 8× extrapolation, EVQ + Progressive YaRN achieves 100% passkey accuracy vs 61–65% for Geo + YaRN — a qualitative capability gap, not incremental improvement.

## Building the Paper

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Working Rules

New experiment reports go under `docs/exp/` with naming convention `YYYY-MM-DD_slug.md`. New paper figures go under `paper/figs/`. Large weights and checkpoints are excluded from version control via `.gitignore`.

## Citation

```bibtex
@inproceedings{evqcosh2026,
  title     = {EVQ-Cosh: Closed-Form Frequency Allocation for Long-Context Extrapolation},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

This repository is released for academic research purposes.
