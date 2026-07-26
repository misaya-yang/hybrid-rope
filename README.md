# EVQ-Cosh: Closed-Form Frequency Allocation for Long-Context Extrapolation

Official code repository for the NeurIPS 2026 submission.

**TL;DR** - We derive a closed-form variational solution for RoPE frequency allocation and show that the one-parameter EVQ-Cosh family, under the same fixed YaRN scale, reaches 100% PK retrieval at 8K on a 454M passkey-mix setting where geometric RoPE + YaRN remains at 61%. PK/passkey denotes teacher-forced NLL-gap retrieval unless explicitly marked as autoregressive exact match.

**Active branch:** `main_0726` is the rebuttal-focused workspace. `main` and
`backup/main-restored-paper-20260726` preserve the restored submitted-paper
baseline from before this branch was pruned.

---

## For Agents and Maintainers

Read in this order before changing the repository:

1. **`AGENTS.md`** — scientific, anonymity, editing and Git rules.
2. **`rebuttal/rebuttal_0723/README.md`** — current-cycle control room.
3. **`rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`** — retained reviewer/AC concerns.
4. **`rebuttal/rebuttal_0723/01_REBUTTAL_PLAYBOOK.md`** — evidence and response routing.
5. **`REPO_MAP.md`** — directory responsibilities and source-of-truth matrix.
6. **`docs/overview/RESULT_PROVENANCE_MANIFEST.md`** — reviewer-safe result provenance.

Do not create another rebuttal directory, another root provenance manifest, or another paper PDF. Current experiment code is indexed through `paper_experiments/MANIFEST.json`; canonical editable files remain under `scripts/` and `experiments/`.

---

## Key Results

The paper validates three core claims across controlled runs at 50M-750M scale, with explicit evidence tiers in the manuscript:

**Claim 1 - Closed-form solution.** RoPE frequency allocation admits a variational solution `phi_k(tau)` with a single temperature parameter `tau`. Geometric RoPE is recovered as the `tau -> 0` limit of the pure-tether sub-family; EVQ-Cosh changes the training-time allocation family without adding learned parameters.

**Claim 2 - PE-dominant diagnostic.** In the reported `128 -> 8K` protocol at 125M scale, the seed-42 EVQ-Cosh row attains lower extrapolation PPL than Geo and a 32-parameter learnable-frequency control, without adding learned positional parameters. The historical row label must not be presented as a faithful end-to-end DAPE reproduction.

**Claim 3 - EVQ + YaRN matched-scale leverage.** At 4x extrapolation from `L_train=2048`, EVQ + YaRN reaches 100% PK retrieval versus 61% for Geometric + YaRN (454M, 3 seeds per configuration). This is reported as a matched-scale substrate test, not a tuned-YaRN sweep.

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

### Paper Figure Sources

Figure-generation code remains under `scripts/figures/`, and its source/data
mapping remains in `docs/overview/PAPER_CLAIMS_MAP.md`. Do not run a command
that writes into `paper/` on this branch: the complete submitted-paper tree is
immutable.

Figure 3 panel (a) has a curated JSON fallback in `data/curated/`; panels (b,c) require regenerated local Phase 11 result JSONs and are documented in `docs/overview/REPRODUCE.md`.

---

## Repository Structure

```
paper/                      LaTeX source, figures, and tables
├── main.tex                NeurIPS submission entry point
├── main.pdf                only retained submission PDF
├── sections/               active sections (01, 02, 03, 05, 06)
├── appendix/               Appendix .tex files
├── tables/                 current Table .tex sources
├── figs/                   paper figures and reproducibility previews
└── refs/                   BibTeX references

rebuttal/                   single internal rebuttal control room
├── README.md               current/archive routing index
├── rebuttal_0723/          official review, AC metareview, current reports and runners
│   ├── README.md           current-cycle evidence and action index
│   ├── 00_REVIEWER_SCORES_AND_AC_METAREVIEW.md
│   ├── 01_REBUTTAL_PLAYBOOK.md
│   ├── 02_RESPONSE_QUESTIONS_AND_OUTCOMES.md
│   ├── theory_results/     standalone evidence owners and theory boundaries
│   └── experiments/        rebuttal-only code and launchers
└── pre_rebuttal/           retained audits, result notes and reusable experiment packages
    └── README.md           archive/reuse index

data/
├── curated/                tracked reviewer-safe result artifacts
└── other paths             historical sources or ignored local caches; see data/README.md

scripts/                    Experiment and evaluation code
├── train.py                Legacy LoRA/Anchored-Sigmoid entrypoint
├── core_text_phases/       Main experiment chain (Phase 8–21)
│   ├── run_evq_sweep.py    Core τ-sweep (raw PPL table)
│   ├── phase14c_*.py       EVQ+YaRN synergy and capability tables
│   ├── phase16_*.py        99-run τ* validation (Fig 6)
│   ├── phase17c_*.py       454M supporting progressive check (Fig 4)
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

paper_experiments/          Manifest-driven index of all paper experiment code
├── README.md               Scope, caveats, rebuild, and standalone-export guide
├── MANIFEST.json           Experiment families, canonical paths, and SHA-256 hashes
└── code/                   Repository-relative links to 94 canonical source files

docs/                       Research documentation
├── overview/               current provenance, claims, reproduction and audit index
│   ├── PAPER_CLAIMS_MAP.md Paper↔Script↔Data navigation hub
│   ├── RESULT_PROVENANCE_MANIFEST.md  reviewer-safe evidence authority
│   ├── REPRODUCE.md        Full reproducibility guide
│   └── DATA_PREPARATION.md Data source documentation
├── exp/                    Experiment reports (YYYY-MM-DD_slug.md)
├── theory/, tau_algor/     derivations and historical theory validation
└── archive/                explicitly historical docs

results/                    tracked historical evidence + ignored new raw outputs

internal/                   historical/private archive; never a current public entry point

tests/                      Unit tests
```

---

## Paper ↔ Code Traceability

Every Figure and Table can be traced back to its generating script and source data in 3 steps. The full traceability map is at **`docs/overview/PAPER_CLAIMS_MAP.md`**.

For one-folder access to all reported primary and supporting experiment code, use **`paper_experiments/`**. Rebuild its links and hash manifest with `python scripts/build_paper_experiment_workspace.py`; the canonical editable files remain under `scripts/` and `experiments/`.

| Stable Paper Asset | Generating Script | Phase |
|--------------------|-------------------|-------|
| Multi-scale raw PPL table | `run_evq_sweep.py` | 8 |
| EVQ+YaRN systems table | 454M aggregate + `phase14c_multiscale_evq_yarn.py` supporting check (not the full Table 2 rerun) | 14 |
| Capability/passkey robustness table | `scripts/supporting_eval/eval_passkey_scratch.py` helpers + `data/curated/table2_evq_yarn_454m_passkey_10pct.json` | 14 |
| PE-dominant tables | `phase11_L256_extrap.py`, `phase11b_125m_dape.py` | 11 |
| 750M continue table | `phase15_750m_*.py` | 15 |
| Frequency dynamics / multiscale waterbed figures | `fig1_neurips.py` | — |
| EVQ×YaRN synergy figure | `fig2_evq_yarn_orthogonality.py` | — |
| PE-dominant scaling figure | `fig3_pe_dominant_scaling.py` | — |
| 454M supporting progressive figure | `phase17c_*.py` | 17 |
| Downstream QA figure | `phase21b_quality_eval_clean.py` | 21 |
| τ* validation figure | `phase16_formula_optimality_sweep.py` | 16 |

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

Do not zip the working directory directly: local result dumps, historical runbooks, LaTeX build products, caches, and private machine paths are not part of the reviewer supplement. Build the curated archive with the leak-checking packager:

```bash
python scripts/package_supplement.py
```

The packager copies only the public paper/source paths and fails if common identity or machine-path strings are found. Include only curated result JSONs needed by figure scripts, not full checkpoints or local experiment dumps.

---

## Submitted Paper Archive

`paper/` is restored from the submitted-paper tree and is read-only on
`main_0726`: do not modify, format, compile, regenerate, move, or partially
replace it. `paper/main.pdf` is the retained submission PDF.

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
