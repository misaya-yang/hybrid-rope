# EVQ-Cosh: Closed-Form Frequency Allocation for Long-Context Extrapolation

Official code repository for the NeurIPS 2026 submission.

**TL;DR** — We derive a closed-form variational solution for RoPE frequency allocation and show that the one-parameter EVQ-Cosh family, combined with Progressive YaRN, achieves 100% passkey retrieval at 8× training length while geometric RoPE + YaRN collapses to 61–65%.

---

## Key Results

The paper validates three core claims across 99 controlled runs at 50M–750M scale:

**Claim 1 — Closed-form solution.** RoPE frequency allocation admits a variational solution φ_k(τ) with a single temperature parameter τ. Geometric RoPE is the τ→0 degenerate limit, making EVQ-Cosh a strict generalization.

**Claim 2 — EVQ ≥ Learnable PE.** In extreme extrapolation (128→8K tokens), EVQ-Cosh matches or exceeds DAPE-style learnable frequency allocation at 125M scale (3-seed), without per-run optimization.

**Claim 3 — EVQ + YaRN synergy.** At 8× extrapolation, EVQ + Progressive YaRN achieves 100% passkey accuracy versus 61–65% for Geometric + YaRN (350M, 3+3 seeds). This is a qualitative capability gap, not an incremental improvement.

---

## Quick Start

### Environment

```bash
conda create -n evq python=3.10 && conda activate evq
pip install -r requirements.txt
```

For Blackwell GPUs (RTX 5090/6000), use PyTorch ≥ 2.7.0 with CUDA 12.8.

### Run the Core Experiment

```bash
# 50M τ-sweep (~4 hours on any GPU with 4GB+ VRAM)
python scripts/core_text_phases/run_evq_sweep.py --tier 50m --seeds 42

# 125M τ-sweep (~8 hours, requires 16GB+ GPU)
python scripts/core_text_phases/run_evq_sweep.py --tier 125m --seeds 42,123,7
```

### Reproduce Paper Figures

```bash
python scripts/figures/fig1_neurips.py       # Fig 1: Frequency dynamics
python scripts/figures/fig2_evq_yarn_orthogonality.py  # Fig 2: EVQ×YaRN synergy
python scripts/figures/fig3_pe_dominant_scaling.py      # Fig 3: PE-dominant scaling
```

Outputs are saved to `paper/figs/`.

---

## Repository Structure

```
paper/                      LaTeX source, figures, and tables
├── main.tex                NeurIPS submission entry point
├── sections/               Section .tex files (introduction, theory, experiments, ...)
├── appendix/               Appendix .tex files
├── tables/                 Table .tex files (Tables 1–6)
├── figs/                   All paper figures (PDF + PNG)
└── refs/                   BibTeX references

scripts/                    Experiment and evaluation code
├── train.py                Core training entrypoint
├── core_text_phases/       Main experiment chain (Phase 8–21)
│   ├── run_evq_sweep.py    Core τ-sweep (Tables 1)
│   ├── phase14c_*.py       EVQ+YaRN synergy (Tables 2–3, Fig 2)
│   ├── phase16_*.py        99-run τ* validation (Fig 6)
│   ├── phase17c_*.py       454M flagship (Fig 4)
│   └── phase21b_*.py       QuALITY downstream (Fig 5)
├── figures/                Paper figure generation scripts
├── data_prep/              Dataset preparation helpers
├── lib/rope/               RoPE schedule library (EVQ-Cosh, Progressive YaRN)
└── supporting_eval/        Supporting evaluators

docs/                       Research documentation
├── overview/               Methodology, reproducibility, traceability map
│   ├── PAPER_CLAIMS_MAP.md ⭐ Paper↔Script↔Data navigation hub
│   ├── REPRODUCE.md        Full reproducibility guide
│   └── DATA_PREPARATION.md Data source documentation
├── exp/                    Experiment reports (YYYY-MM-DD_slug.md)
└── theory/                 Derivations and validation notes

results/                    Experiment output artifacts
├── core_text/              Primary text experiment results
├── theory/                 Theory validation outputs
└── supporting_*/           Cross-model and video experiments

team/                       Collaboration materials
├── briefs/                 Advisor and collaborator briefs
├── status/                 Gap tracking and priority matrices
└── plans/                  Experiment plans
```

---

## Paper ↔ Code Traceability

Every Figure and Table can be traced back to its generating script and source data in 3 steps. The full traceability map is at **`docs/overview/PAPER_CLAIMS_MAP.md`**.

| Paper Element | Generating Script | Phase |
|--------------|-------------------|-------|
| Table 1 (Multi-scale PPL) | `run_evq_sweep.py` | 8 |
| Table 2–3 (EVQ+YaRN) | `phase14c_multiscale_evq_yarn.py` | 14 |
| Table 4–5 (PE-dominant) | `phase11_L256_extrap.py`, `phase11b_125m_dape.py` | 11 |
| Table 6 (750M continue) | `phase15_750m_*.py` | 15 |
| Fig 1 (Frequency dynamics) | `fig1_neurips.py` | — |
| Fig 2 (EVQ×YaRN synergy) | `fig2_evq_yarn_orthogonality.py` | — |
| Fig 3 (PE-dominant scaling) | `fig3_pe_dominant_scaling.py` | — |
| Fig 4 (454M flagship) | `phase17c_*.py` | 17 |
| Fig 5 (Downstream QA) | `phase21b_quality_eval_clean.py` | 21 |
| Fig 6 (τ* validation) | `phase16_formula_optimality_sweep.py` | 16 |

---

## Reproducibility

Hardware requirements and full reproduction paths are documented in **`docs/overview/REPRODUCE.md`**:

| Path | Hardware | Time |
|------|----------|------|
| Quick validation (50M) | Any GPU 4GB+ or Apple M-series | ~4 hours |
| Core results (125M/350M) | NVIDIA GPU 16GB+ | ~24 hours |
| Full reproduction (454M/750M) | A100/RTX 4090+ | ~72 hours |

All training data is streamed from HuggingFace Hub (no pre-download required). See `docs/overview/DATA_PREPARATION.md` for details.

---

## Building the Paper

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

---

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
