# Submission / Core Asset Checklist

> Date: 2026-03-06
> Scope: current anonymous submission package and the core experiment assets that support its main claims
> Standard: file-existence and wiring check only; this is not a claim-strength review

---

## 1. Submission Chain

### 1.1 Required source files

| Item | Status | Path |
|------|--------|------|
| Main anonymous draft | OK | `paper_draft/submission/main.tex` |
| NeurIPS style | OK | `paper_draft/submission/neurips_2025.sty` |
| References database | OK | `paper_draft/submission/refs/references.bib` |
| Section files | OK | `paper_draft/submission/sections/*.tex` |
| Tables | OK | `paper_draft/submission/tables/*.tex` |
| Appendix files | OK | `paper_draft/submission/appendix/*.tex` |

### 1.2 Figure wiring

| Figure | Status | Path |
|--------|--------|------|
| Figure 2 main empirical figure | OK | `paper_draft/figs/fig2_evq_yarn_synergy.pdf` |
| Figure 3 PE-dominant / scaling-law figure | OK | `paper_draft/figs/fig3_pe_dominant_scaling.pdf` |
| Figure 1 supporting dynamics figure | OK | `paper_draft/figs/fig1_frequency_dynamics.pdf` |

### 1.3 Build status

- Anonymous draft compiles successfully from `paper_draft/submission/`
- `main.pdf` is produced locally
- LaTeX build artifacts exist locally but are ignored by `.gitignore`

**Conclusion**: no blocking file gap was found in the current submission source chain.

---

## 2. Core Evidence Chain

### 2.1 Main text anchors

| Evidence | Status | Path |
|----------|--------|------|
| Passkey mix multi-seed | OK | `docs/exp/2026-03-03_passkey_mix_results.md` |
| Phase 11 L=256 | OK | `docs/exp/2026-03-04_phase11_L256_results.md` |
| Phase 11b 125M | OK | `docs/exp/2026-03-05_phase11b_125m_results.md` |
| Phase 15 750M continue | OK | `docs/exp/2026-03-06_phase15_750m_2k_to_4k_continue_results.md` |

### 2.2 Core result bundles

| Result bundle | Status | Path |
|---------------|--------|------|
| Phase 14 EVQ+YaRN / passkey | OK | `results/core_text/phase14_yarn_passkey/` |
| Phase 11 L=256 | OK | `results/core_text/phase11/` |
| Phase 11b 125M | OK | `results/core_text/phase11b/` |
| Phase 15 750M continue | OK | `results/core_text/phase15/` |
| Phase 9f 750M historical support | OK | `results/core_text/phase9f_750m_2k_1b/` |

### 2.3 Core experiment scripts

| Experiment | Status | Path |
|------------|--------|------|
| Main training entrypoint | OK | `scripts/train.py` |
| EVQ+YaRN multiscale text eval | OK | `scripts/core_text_phases/phase14c_multiscale_evq_yarn.py` |
| Phase 11 L=256 extrapolation | OK | `scripts/core_text_phases/phase11_L256_extrap.py` |
| DAPE-style 125M regime | OK | `scripts/core_text_phases/phase11b_125m_dape.py` |
| Phase 15 750M continue eval | OK | `scripts/core_text_phases/phase15_750m_2k_to_4k_continue_ckpt_eval.py` |
| Video temporal support | OK | `scripts/video_temporal/run_video_temporal.py` |

**Conclusion**: the current main-paper claim chain has a complete file trail from source code to result bundle to written experiment report.

---

## 3. Supporting-but-Non-Primary Assets

| Asset | Status | Path |
|-------|--------|------|
| Video temporal results | OK | `results/supporting_video/video_temporal/` |
| Cross-model / LoRA / external-model support | OK | `results/supporting_cross_model/` |
| Theory validation artifacts | OK | `results/theory/` |
| Team collaboration material | OK | `team/` |

These assets are preserved and organized, but they are not required for the minimal submission chain to compile and remain auditable.

---

## 4. Non-File Gaps Still Worth Tracking

These are not missing files. They are evidence or process gaps that still matter:

1. **Phase 15 downstream fairness gap**
   - The EVQ side of the LongBench-style downstream comparison did not complete due remote dataset download failure.
   - Current status: supporting paragraph only, not a primary table anchor.

2. **Video remains secondary**
   - The video temporal support is present and organized, but it is still a supporting cross-modal signal rather than a co-primary main-text anchor.

3. **Single-seed larger-scale evidence**
   - `phase15` and `phase9f` remain supporting evidence rather than primary anchors.

---

## 5. Final Verdict

### Submission package
- **No blocking missing files found**

### Core experiment package
- **No blocking missing files found**

### Remaining red flags
- None at the file-availability level
- Remaining issues are claim-strength / evidence-tier questions, not repository-structure questions
