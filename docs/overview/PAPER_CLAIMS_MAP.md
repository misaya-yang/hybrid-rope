# Paper ↔ Experiment ↔ Script ↔ Results Traceability Map

> 这是整个仓库的导航中枢。从任何论文 Figure/Table 出发，都能在 3 步内找到生成脚本和原始数据。

---

## Figures

| Figure | 论文位置 | 描述 | 生成脚本 | 数据来源 | 输出文件 |
|--------|---------|------|---------|---------|---------|
| Fig 1 | Appendix (a3) | Frequency dynamics + 750M training curves | `scripts/figures/fig1_neurips.py` | Phase 9F (750M, L=2048) inline data | `paper/figs/fig1_frequency_dynamics.pdf` |
| Fig 2 | §5 Experiments | EVQ × YaRN orthogonal synergy | `scripts/figures/fig2_evq_yarn_orthogonality.py` | `docs/exp/2026-03-03_passkey_mix_results.md` (inline hardcoded) | `paper/figs/fig2_evq_yarn_synergy.pdf` |
| Fig 3 | §5 Experiments | PE-dominant regime & scaling law | `scripts/figures/fig3_pe_dominant_scaling.py` | `data/evq_128tok_results/results_phase6.json` + `results_checkpoint.json` | `paper/figs/fig3_pe_dominant_scaling.pdf` |
| Fig 4 | §5 Experiments | Phase 17c 454M flagship | `scripts/core_text_phases/phase17c_*.py` | `results/evq_phase17c_results/` | `paper/figs/fig4_phase17c_flagship.pdf` |
| Fig 5 | §5 Experiments | Downstream QA (Gold NLL) | `scripts/core_text_phases/phase21b_quality_eval_clean.py` | `results/core_text/phase21b/` | `paper/figs/fig5_downstream_qa.pdf` |
| Fig 6 | §5 Experiments | τ* formula validation (99-run) | `scripts/core_text_phases/phase16_formula_optimality_sweep.py` | `results/core_text/phase16/` | `paper/figs/fig6_tau_formula_validation.pdf` |
| Fig 7 | §5 Experiments | Multiscale waterbed trade-off | `scripts/figures/fig1_neurips.py` (subplot) | Multi-tier PPL results | `paper/figs/fig7_multiscale_waterbed.pdf` |
| Attn Viz | Appendix | Attention distance distribution | `scripts/core_text_phases/visualize_attention_distance.py` | 750M checkpoints (EVQ vs Geo) | `paper/figs/attn_*.pdf` |
| τ-sweep | Appendix | τ sweep curves (PPL, freq, collision, cross-scale) | `scripts/core_text_phases/evq_analysis.py` | `results/core_text/D_summary.json` | `paper/figs/fig_tau_sweep_*.pdf` |

---

## Tables

| Table | 论文位置 | 描述 | LaTeX 文件 | 数据来源脚本 | 实验报告 |
|-------|---------|------|-----------|------------|---------|
| Table 1 | §5.3 | Multi-scale raw PPL (50M-350M) | `paper/tables/table1_multiscale_raw_ppl.tex` | `run_evq_sweep.py` (50M/125M/350M tiers) | `docs/exp/2026-02-27_evq_tau_sweep_results.md` |
| Table 2 | §5.1 | EVQ+YaRN main systems result | `paper/tables/table2_evq_yarn_main.tex` | `phase14c_multiscale_evq_yarn.py` | `docs/exp/2026-03-03_passkey_mix_results.md` |
| Table 3 | §5.3 | Capability preservation (passkey robustness) | `paper/tables/table3_capability_passkey.tex` | `eval_passkey.py` | `docs/exp/2026-03-03_passkey_mix_results.md` |
| Table 4 | §5.2 | PE-dominant extreme extrapolation | `paper/tables/table4_pe_dominant.tex` | `phase11c_454m_scaling.py` + `phase11b_125m_dape.py` | `docs/exp/2026-03-04_phase11_L256_results.md` |
| Table 5 | §5.2 | Phase 11 leverage (YaRN asymmetry @L=256) | `paper/tables/table5_phase11_leverage.tex` | `phase11_L256_extrap.py` + `phase11_yarn_eval.py` | `docs/exp/2026-03-04_phase11_L256_results.md` |
| Table 6 | §5.3 | 750M continued pretraining evidence | `paper/tables/table6_750m_continue_supporting.tex` | `phase15_750m_2k_to_4k_continue_ckpt_eval.py` | `docs/exp/2026-03-06_phase15_750m_2k_to_4k_continue_results.md` |
| Table A1 | Appendix A2 | Reproducibility snapshot | `paper/appendix/a2_experiment_details.tex` (inline) | — | — |

---

## Core Claims → Evidence Chain

| ID | Claim | Primary Evidence | Scripts | Seeds | Risk |
|----|-------|-----------------|---------|-------|------|
| **C1** | τ*=d_head/√L scaling law | Phase 16 formula sweep (99 runs, 50M/125M) | `phase16_formula_optimality_sweep.py` | 3+ seeds × multi-τ | ✅ Low |
| **C2** | EVQ ≥ Learnable PE (DAPE-style) | Phase 11b 125M extreme extrap (128→8K) | `phase11b_125m_dape.py` | 3 seeds | ✅ Low |
| **C3** | EVQ+YaRN >> Geo+YaRN | Phase 14c 350M passkey mix | `phase14c_multiscale_evq_yarn.py` | 3+3 seeds | ✅ Low-Medium |
| **C4** | 454M Stage 2-3 continued pretrain | Phase 17c 454M (1024→2048) | `phase17c_454m_1024_to_2048_continue.py` | seeds 42-44 | ⚠️ HIGH (single-config) |
| **C5** | 750M scale-up confirmation | Phase 15 750M (2K→4K) | `phase15_750m_2k_to_4k_continue_ckpt_eval.py` | single-seed | ⚠️ HIGH (single-seed) |
| **C6** | Downstream NLL advantage | Phase 21b QuALITY eval | `phase21b_quality_eval_clean.py` | n=2086 | ✅ Low (Gold NLL) |

---

## Theory → Validation

| Theorem/Proposition | 论文位置 | Numerical Validation | Script |
|---------------------|---------|---------------------|--------|
| Thm 1: EVQ-cosh closed-form | §3 | Phase 16 (99-run) confirms τ* optimality | `phase16_formula_optimality_sweep.py` |
| Thm 2: τ=0 ≡ geometric | §3 | Verified: `evq_cosh_inv_freq(64, 0.0) == geometric` | `run_evq_sweep.py:151-152` |
| Waterbed inequality | §4 | All tiers show bounded short-range cost | `evq_analysis.py` waterbed plot |
| Phase collision reduction | §4 | Collision scores decrease with optimal τ | `run_evq_sweep.py` collision analysis |

---

## Video/DiT Claims (2026-03-16, new)

| ID | Claim | Primary Evidence | Scripts | Method | Risk |
|----|-------|-----------------|---------|--------|------|
| **V1** | EVQ-Cosh generalizes to DiT (bidirectional attention) | 129.6M h2h: τ=1.5 wins -21%/-35% | `run_dit_temporal.py` | Head-to-head | ✅ Low (same-run) |
| **V2** | DiT needs different τ*: τ*_DiT ≈ 0.53 × τ*_AR | τ sweep: only 1.5 works, 0.3/0.7/1.2 fail | `run_dit_temporal.py --tau` | Head-to-head | ⚠️ Medium (single-model) |
| **V3** | Sharp phase transition at τ∈(1.2, 1.5) | h2h: τ=1.2 is 2.8x worse, τ=1.5 is 21% better | `run_dit_temporal.py` | Head-to-head | Need fine-grained sweep |
| **V4** | Teacher-forced: EVQ +5.4% top-5 accuracy | VideoGPT 268.7M, N=2000, extrap region | `eval_temporal_precision.py` | Teacher-forced | ✅ Low (large N) |
| **V5** | Advantage scales with temporal frequency | P=16: +8.48%, P=24: +7.63%, P=32: +6.25% | `eval_temporal_precision.py` | FFT decomposition | ✅ Low |
| **V6** | Dead channel mechanism: base reduction eliminates phase transition | base=1000 h2h: τ=1.2≈τ=1.5, both -48% vs Geo | `run_dit_temporal.py --base 1000` | Head-to-head | ✅ Low (mechanistic) |

### DiT Appendix Tables (paper/appendix/a2_experiment_details.tex)

| Table | 描述 | 数据来源 | Key Numbers |
|-------|------|---------|-------------|
| `tab:video-temporal` | VideoGPT temporal extrap (PPL, FVD) | `results/supporting_video/` | PPL -27.3%, FVD -1.5% |
| `tab:dit-h2h` | DiT dual-seed h2h (train/all/far MSE) | `results/video_dit/westd_20260316/` | mean -21%/-15%/-32% |
| `tab:dit-inference` | DiT inference methods (YaRN/RIFLEx/raw) | `results/video_dit/westd_20260316/riflex_eval_*.json` | EVQ wins all: -37%/-28%/-59% |
| `tab:dit-timestep` | DiT multi-timestep (t=0.2/0.5/0.8) | Same as dit-inference | -4%/-36%/-27% |
| `tab:temporal-precision` | Teacher-forced accuracy by region | `results/supporting_video/temporal_precision/` | +3.14% top-1, +5.40% top-5 |
| `tab:freq-decomp` | Frequency-resolved accuracy delta | Same | P=16: +5.07%, P=32: +3.55% |
| `tab:quality-nll` | QuALITY Gold NLL (appendix a3) | `results/core_text/phase21b/` | -30.1% @8K |
| `tab:dit-base1000` | Dead channel validation (base=1000 h2h) | `results/video_dit/westd_20260316/base1000_h2h/` | τ=1.2≈τ=1.5, both -48% far |

### Video Reports & Data

| Experiment | Report | Data | Scripts |
|-----------|--------|------|---------|
| DiT 38.8M + 129.6M (cross-run) | `results/video_dit/REPORT_FINAL.md` (v2) | `results/video_dit/20260316_{002758,medium}/` | `run_dit_temporal.py` |
| DiT τ sweep (cross-run) | `results/video_dit/TAU_SWEEP_HANDOFF.md` | `results/video_dit/20260316_tau_sweep/` (server) | `run_dit_temporal.py --tau` |
| DiT head-to-head | `results/video_dit/REPORT_FINAL.md` (v2, Part II) | Verbal + server logs | `run_dit_temporal.py` |
| VideoGPT teacher-forced | `results/supporting_video/temporal_precision_report.md` | `results/supporting_video/temporal_precision/` | `eval_temporal_precision.py` |
| Phase collision analysis | — | `results/video_dit/phase_collision_analysis.json` | Theory computation |
| DiT theory analysis | `DiT_frequency_allocation_analysis.md` (root) | — | — |

---

## Experiment Phase Map (Script → Paper)

| Phase | Question | Paper Role | Key Scripts | → Figure/Table |
|-------|----------|-----------|-------------|----------------|
| 8 | Raw EVQ τ-sweep | Theory foundation | `phase8d_scaling_law.py`, `phase8f_multi_seed.py` | Fig 6, Table 1 |
| 11 | PE-dominant regime | **Primary anchor** | `phase11_L256_extrap.py`, `phase11_yarn_eval.py`, `phase11b_125m_dape.py`, `phase11c_454m_scaling.py` | Fig 3, Tables 4-5 |
| 14 | EVQ+YaRN synergy | **Primary anchor** | `phase14c_multiscale_evq_yarn.py` | Fig 2, Tables 2-3 |
| 15 | 750M scale-up | Supporting | `phase15_750m_2k_to_4k_continue_ckpt_eval.py` | Table 6 |
| 16 | τ* formula validation | Theory confirmation | `phase16_formula_optimality_sweep.py` | Fig 6 |
| 17c | 454M continued pretrain | Flagship demo | `phase17c_454m_1024_to_2048_continue.py` | Fig 4 |
| 21b | Downstream QA | Downstream evidence | `phase21b_quality_eval_clean.py` | Fig 5 |

---

## Directory Quick Reference

| 需求 | 路径 |
|------|------|
| 论文 LaTeX 源码 | `paper/main.tex` |
| 所有论文图片 | `paper/figs/` |
| 论文表格 | `paper/tables/` |
| 出图脚本 | `scripts/figures/fig*.py` |
| 核心实验代码 | `scripts/core_text_phases/` |
| RoPE 库 | `scripts/lib/rope/` |
| 实验报告 | `docs/exp/` (YYYY-MM-DD_slug.md) |
| 实验结果数据 | `results/core_text/` |
| 复现指南 | `docs/overview/REPRODUCE.md` |
| AI Handoff | `internal/AIHANDOFF.md` |
| 协作材料 | `team/` |
