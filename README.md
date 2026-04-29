# EVQ-Cosh: Closed-Form Frequency Allocation for Long-Context Extrapolation

Official code repository for the NeurIPS 2026 submission.

**TL;DR** - We derive a closed-form variational solution for RoPE frequency allocation and show that the one-parameter EVQ-Cosh family, under the same fixed YaRN scale, reaches 100% passkey retrieval at 8K on a 454M passkey-mix setting where geometric RoPE + YaRN remains at 61%.

---

## Key Results

The paper validates three core claims across controlled runs at 50M-750M scale, with explicit evidence tiers in the manuscript:

**Claim 1 - Closed-form solution.** RoPE frequency allocation admits a variational solution `phi_k(tau)` with a single temperature parameter `tau`. Geometric RoPE is the `tau -> 0` limit, making EVQ-Cosh a strict generalization.

**Claim 2 - PE-dominant diagnostic.** In a DAPE-style `128 -> 8K` protocol at 125M scale, the seed-42 EVQ-Cosh row attains lower extrapolation PPL than the Geo and DAPE-style learned-operator baselines, without adding learned positional parameters.

**Claim 3 - EVQ + YaRN matched-scale leverage.** At 4x extrapolation from `L_train=2048`, EVQ + YaRN reaches 100% passkey accuracy versus 61% for Geometric + YaRN (454M, 3 seeds per configuration). This is reported as a matched-scale substrate test, not a tuned-YaRN sweep.

---

## Quick Start

### Environment

```bash
conda create -n evq python=3.10 && conda activate evq
pip install -r requirements-lock.txt
```

For Blackwell GPUs (RTX 5090/6000), use PyTorch ≥ 2.7.0 with CUDA 12.8.

### Run the Core Experiment

```bash
# 50M τ-sweep (~4 hours on any GPU with 4GB+ VRAM)
python scripts/core_text_phases/run_evq_sweep.py --tier 50m --seeds 42 --strict_dataset --passkey_mix_ratio 0

# 125M τ-sweep (~8 hours, requires 16GB+ GPU)
python scripts/core_text_phases/run_evq_sweep.py --tier 125m --seeds 42,123,7 --strict_dataset --passkey_mix_ratio 0
```

`requirements.txt` keeps broad lower bounds for portability; `requirements-lock.txt` records the paper validation environment. Use `--strict_dataset` for paper reproduction so FineWeb-Edu failures do not silently fall back to another corpus.

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
├── sections/               Section .tex files (01_intro … 07_conclusion)
├── appendix/               Appendix .tex files
├── tables/                 Table .tex files (Tables 1–6)
├── figs/                   All paper figures (PDF + PNG)
└── refs/                   BibTeX references

scripts/                    Experiment and evaluation code
├── train.py                Legacy LoRA/Anchored-Sigmoid entrypoint
├── core_text_phases/       Main experiment chain (Phase 8–21)
│   ├── run_evq_sweep.py    Core τ-sweep (Table 1)
│   ├── phase14c_*.py       EVQ+YaRN synergy (Tables 2–3, Fig 2)
│   ├── phase16_*.py        99-run τ* validation (Fig 6)
│   ├── phase17c_*.py       454M flagship (Fig 4)
│   └── phase21b_*.py       QuALITY downstream (Fig 5)
├── figures/                Paper figure generation scripts
├── data_prep/              Dataset preparation helpers
├── lib/rope/               RoPE schedule library (EVQ-Cosh and scaling baselines)
├── video_temporal/         Video DiT temporal extrapolation experiments
└── supporting_eval/        Supporting evaluators (LongBench, NIAH, passkey)

experiments/                Standalone experiment packages
└── lora_evq_v2/            LLaMA-3-8B LoRA fine-tuning with EVQ-Cosh
    ├── train_evq_lora.py   Training script (bf16, r=64, τ=1.414)
    ├── eval_evq_lora.py    PPL / Passkey / LongBench evaluation
    ├── eval_ruler.py       RULER benchmark (6 tasks, 4K–32K)
    ├── eval_pe_probes.py   Custom PE probing tasks (4 tasks)
    ├── compare_results.py  Base vs EVQ comparison & LaTeX table
    └── run.sh              One-click runner

docs/                       Research documentation
├── overview/               Methodology, reproducibility, traceability map
│   ├── PAPER_CLAIMS_MAP.md Paper↔Script↔Data navigation hub
│   ├── REPRODUCE.md        Full reproducibility guide
│   └── DATA_PREPARATION.md Data source documentation
├── exp/                    Experiment reports (YYYY-MM-DD_slug.md)
└── theory/                 Derivations and validation notes

results/                    Experiment output artifacts (gitignored, synced via rsync)

tests/                      Unit tests
```

---

## Paper ↔ Code Traceability

Every Figure and Table can be traced back to its generating script and source data in 3 steps. The full traceability map is at **`docs/overview/PAPER_CLAIMS_MAP.md`**.

| Paper Element | Generating Script | Phase |
|--------------|-------------------|-------|
| Table 1 (Multi-scale PPL) | `run_evq_sweep.py` | 8 |
| Table 2–3 (EVQ+YaRN) | 454M aggregate + `phase14c_multiscale_evq_yarn.py` supporting check | 14 |
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
| Core results (125M/454M) | NVIDIA GPU 16GB+ | ~24+ hours |
| Full reproduction (454M/750M) | A100/RTX 4090+ | ~72 hours |

All training data is streamed from HuggingFace Hub (no pre-download required). See `docs/overview/DATA_PREPARATION.md` for details.

## Supplement Packaging

Do not zip the working directory directly: local `results/`, `internal/`, historical runbooks, LaTeX build products, caches, and private machine paths are not part of the reviewer supplement. Build the curated archive with the leak-checking packager:

```bash
python scripts/package_supplement.py
```

The packager copies only the public paper/source paths and fails if common identity or server-path strings are found. Include only curated result JSONs needed by figure scripts, not full checkpoints or local experiment dumps.

---

## Building the Paper

```bash
cd paper
bash compile_aidemo.sh
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
