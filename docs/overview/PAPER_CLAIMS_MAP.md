# Paper ↔ Experiment ↔ Script ↔ Results Traceability Map

> 这是整个仓库的导航中枢。从任何论文 Figure/Table 出发，都能在 3 步内找到生成脚本和原始数据。

---

## Figures

> Use figure asset names as the stable handle; compiled figure numbers may shift when front-matter or appendix floats move.

| Stable asset | 论文位置 | 描述 | 生成脚本 | 数据来源 | 输出文件 |
|--------------|---------|------|---------|---------|---------|
| Method overview | Theory and Method | EVQ-Cosh allocation schematic and collision-envelope intuition | manual/static asset | paper diagram source | `paper/figs/fig_method_overview.pdf` |
| EVQ × YaRN | Experiments / Primary I | EVQ × YaRN orthogonal synergy | `scripts/figures/fig2_evq_yarn_orthogonality.py` | `data/curated/table2_evq_yarn_454m_passkey_10pct.json` | `paper/figs/fig2_evq_yarn_synergy.pdf` |
| PE-dominant scaling | Appendix supporting PE-dominant section | PE-dominant regime & scaling-law check | `scripts/figures/fig3_pe_dominant_scaling.py` | `data/curated/fig3_extreme_128.json` fallback for panel (a); regenerate Phase 11 sweeps for panels (b,c) | `paper/figs/fig3_pe_dominant_scaling.pdf` |
| Progressive training | Appendix experiment details | Phase 17c 454M supporting/progressive pattern | `scripts/core_text_phases/phase17c_*.py` | `results/evq_phase17c_results/` | `paper/figs/fig4_phase17c_flagship.pdf` |
| Downstream QA | Appendix supporting results | Downstream QA (Gold NLL) | `scripts/core_text_phases/phase21b_quality_eval_clean.py` | `results/core_text/phase21b/` | `paper/figs/fig5_downstream_qa.pdf` |
| τ* validation | Appendix theory validation | τ* operating-rule validation (99-run sweep basin) | `scripts/core_text_phases/phase16_formula_optimality_sweep.py` | `results/core_text/phase16/` | `paper/figs/fig6_tau_formula_validation.pdf` |
| Multiscale waterbed | Appendix supporting results | Multiscale waterbed trade-off | `scripts/figures/fig1_neurips.py` (subplot) | Multi-tier PPL results | `paper/figs/fig7_multiscale_waterbed.pdf` |
| Attn Viz | Appendix supporting results | Attention distance distribution | `scripts/core_text_phases/visualize_attention_distance.py` | 750M checkpoints (EVQ vs Geo) | `paper/figs/attn_*.pdf` |
| τ-sweep | Appendix supporting results | τ sweep curves (PPL, freq, collision, cross-scale) | `scripts/core_text_phases/evq_analysis.py` | `results/core_text/D_summary.json` | `paper/figs/fig_tau_sweep_*.pdf` |

---

## Tables

> Use the LaTeX file/label as the stable handle. Compiled table numbers are intentionally omitted here because floats may move between builds.

| Stable table source | Stable PDF location | 论文位置 | 描述 | 数据来源脚本 | 实验报告 |
|---------------------|---------------------|---------|------|------------|---------|
| `paper/tables/table_epistemic_map.tex` | Main body table | Theory and Method | Epistemic status of derivation components | derivation notes | — |
| `paper/tables/table_evidence_tier.tex` | Main body table | Experiments setup | Evidence tier by setting | paper-source classification | — |
| `paper/tables/table2_evq_yarn_main.tex` | Main body table | Primary I | EVQ+YaRN main systems result | 454M curated aggregate/provenance + `phase14c_multiscale_evq_yarn.py` supporting check | `data/curated/table2_evq_yarn_454m_passkey_10pct.json` |
| `paper/tables/table4_pe_dominant.tex` | Main body table | Primary II | PE-dominant extreme extrapolation | `phase11b_125m_dape.py`; `phase11c_454m_scaling.py` supports Fig. 3 scaling panels | `data/curated/fig3_extreme_128.json` + Phase 11 regeneration |
| `paper/tables/table_method_comparison.tex` | Appendix table | Appendix method comparison | Compact method comparison / positioning | literature survey | — |
| `paper/appendix/a2_experiment_details.tex` (inline) | Appendix table | Appendix experiment details | Reproducibility snapshot | — | — |
| `paper/tables/table6_750m_continue_supporting.tex` | Appendix table | Appendix experiment details | 750M continued-pretraining support | `phase15_750m_2k_to_4k_continue_ckpt_eval.py` | `docs/exp/2026-03-06_phase15_750m_2k_to_4k_continue_results.md` |
| `paper/tables/table5_phase11_leverage.tex` | Appendix table | Appendix supporting results | Phase 11 leverage (YaRN asymmetry at L=256) | `phase11_L256_extrap.py` + `phase11_yarn_eval.py` | `docs/exp/2026-03-04_phase11_L256_results.md` |
| `paper/tables/table1_multiscale_raw_ppl.tex` | Appendix table | Appendix supporting results | Multi-scale raw PPL (50M-750M) | `run_evq_sweep.py` (50M/125M tiers plus curated larger rows) | `docs/exp/2026-02-27_evq_tau_sweep_results.md` |
| `paper/tables/table3_capability_passkey.tex` | Appendix table | Appendix supporting experiments | Capability preservation / passkey robustness | `scripts/supporting_eval/eval_passkey_scratch.py` helpers + curated aggregate | `data/curated/table2_evq_yarn_454m_passkey_10pct.json` |

---

## Core Claims → Evidence Chain

| ID | Claim | Primary Evidence | Scripts | Seeds | Risk |
|----|-------|-----------------|---------|-------|------|
| **C1** | EVQ-Cosh is the exact inverse-CDF minimizer of the stated broadband surrogate; τ is a semi-analytic operating rule, not a global optimum | Theory + Phase 16 formula sweep (99 runs, 50M/125M) | `phase16_formula_optimality_sweep.py` | 3+ seeds × multi-τ | ✅ Low |
| **C2** | PE-dominant DAPE-style diagnostic: EVQ has lower seed-42 8K PPL than Geo/DAPE without learned PE parameters | Phase 11b 125M extreme extrap (128→8K) | `phase11b_125m_dape.py` | 1--3 seeds by row | ⚠️ Medium (diagnostic scope) |
| **C3** | EVQ increases fixed-scale YaRN leverage vs Geo+YaRN | 454M passkey-mix aggregate/provenance; Phase 14c provides 50M/125M supporting rerun | `data/curated/table2_evq_yarn_454m_passkey_10pct.json`; `phase14c_multiscale_evq_yarn.py` is supporting only | 3+3 seeds | ✅ Low for traceability; medium for scope |
| **C4** | MLA scarce-channel stress test is the third primary empirical anchor | 432M MLA 3-seed run; matched-scale Geo+YaRN comparison | MLA scripts / curated aggregate | 3 seeds | ⚠️ Medium (architecture-specific convention) |
| **S1** | 454M Stage 2-3 continued pretrain | Phase 17c 454M (1024→2048) | `phase17c_454m_1024_to_2048_continue.py` | single seed | Supporting only |
| **S2** | 750M scale-up confirmation | Phase 15 750M (2K→4K) | `phase15_750m_2k_to_4k_continue_ckpt_eval.py` | single-seed | Supporting only |
| **S3** | Downstream NLL advantage | Phase 21b QuALITY eval | `phase21b_quality_eval_clean.py` | n=2086 | Supporting / downstream check |

---

## Theory → Validation

| Theorem/Proposition | 论文位置 | Numerical Validation | Script |
|---------------------|---------|---------------------|--------|
| Thm 1: EVQ-cosh closed-form | Theory and Method | Phase 16 (99-run) validates the τ* operating-rule basin | `phase16_formula_optimality_sweep.py` |
| Thm 2: $\tau{\to}0$ geometric limit | Theory and Method | Verified on the midpoint grid used by EVQ experiments; `geometric_inv_freq` remains the standard endpoint-grid RoPE baseline | `scripts/lib/rope/schedules.py` |
| Waterbed inequality | Theory and appendix proof | All tiers show bounded short-range cost | `evq_analysis.py` waterbed plot |
| Phase collision reduction | Theory and appendix validation | Collision scores decrease near the selected operating τ | `run_evq_sweep.py` collision analysis |

---

## Supporting Video/DiT Observations (2026-03-16)

These rows are appendix supporting/exploratory evidence only; no abstract or introduction claim should depend on them.

| ID | Supporting observation | Evidence | Scripts | Method | Scope |
|----|------------------------|----------|---------|--------|-------|
| **V1** | EVQ-Cosh generalizes to DiT (bidirectional attention) | 129.6M h2h: τ=1.5 wins -21%/-35% | `run_dit_temporal.py` | Head-to-head | Supporting |
| **V2** | DiT needs different τ*: τ*_DiT ≈ 0.53 × τ*_AR | τ sweep: only 1.5 works, 0.3/0.7/1.2 fail | `run_dit_temporal.py --tau` | Head-to-head | ⚠️ Medium (single-model) |
| **V3** | Sharp phase transition at τ∈(1.2, 1.5) | h2h: τ=1.2 is 2.8x worse, τ=1.5 is 21% better | `run_dit_temporal.py` | Head-to-head | Need fine-grained sweep |
| **V4** | Teacher-forced: EVQ +5.4% top-5 accuracy | VideoGPT 268.7M, N=2000, extrap region | `eval_temporal_precision.py` | Teacher-forced | ✅ Low (large N) |
| **V5** | Advantage scales with temporal frequency | P=16: +8.48%, P=24: +7.63%, P=32: +6.25% | `eval_temporal_precision.py` | FFT decomposition | ✅ Low |
| **V6** | Dead channel mechanism: base reduction eliminates phase transition | base=1000 h2h: τ=1.2≈τ=1.5, both -48% vs Geo | `run_dit_temporal.py --base 1000` | Head-to-head | ✅ Low (mechanistic) |
| **V7** | Dead channels are systemic across all major video DiTs | CogVideoX 50%, Wan2.1 42%, Latte 42%, HunyuanVideo 38%, Open-Sora 31% | Pure math (θ_k × T_train < 0.1 rad) | Analytical | ✅ Low (mathematical fact) |
| **V8** | EVQ robust to base; GEO fragile + non-monotonic | Base sweep 6pt: EVQ 1.9× range vs GEO 11.8× (YaRN); base=50K gap 7.2× | `run_dit_temporal.py --base` | Head-to-head | ✅ Low (6-point sweep) |
| **V9** | EVQ advantage is training-time, not YaRN artifact | Without YaRN: EVQ -35% to -56% at base≥500 | Same | Head-to-head | ✅ Low |
| **V10** | Frequency allocation is pure extrapolation effect | 32f eval: ALL 12 configs within 0.0093–0.0102 (<6%); 128f range +18% to -86% | Same checkpoints, 32f eval | Isolation experiment | ✅ Low (decisive) |
| **V11** | EVQ advantage persists at 3× scale (382M DiT) | 382M h2h: YaRN far -35%, noYaRN far -64%, training loss identical (+0.3%) | `run_dit_temporal.py` (382M config) | Head-to-head | ⚠️ Medium (single-seed) |

### Video/DiT Appendix Tables

Only the rows below are present in the current paper appendix; other video/DiT reports remain external supporting notes and are not packaged as paper tables.

| Table | 描述 | 数据来源 | Key Numbers |
|-------|------|---------|-------------|
| `tab:dit-h2h` | DiT dual-seed h2h (train/all/far MSE) | `results/video_dit/westd_20260316/` | mean -21%/-15%/-32% |
| `tab:quality-nll` | QuALITY Gold NLL (appendix a3) | `results/core_text/phase21b/` | -30.1% @8K |
| `tab:dit-base1000` | Dead channel validation (base=1000 h2h) | `results/video_dit/westd_20260316/base1000_h2h/` | τ=1.2≈τ=1.5, both -48% far |
| `tab:dead-channels` | Dead-channel counts across video models | analytical channel count | 32--50% temporal channels dead |

### Video Reports & Data

| Experiment | Report | Data | Scripts |
|-----------|--------|------|---------|
| DiT 38.8M + 129.6M (cross-run) | `results/video_dit/REPORT_FINAL.md` (v2) | `results/video_dit/20260316_{002758,medium}/` | `run_dit_temporal.py` |
| DiT τ sweep (cross-run) | `results/video_dit/TAU_SWEEP_HANDOFF.md` | Regenerate from the listed script; raw run logs are excluded from the compact supplement | `run_dit_temporal.py --tau` |
| DiT head-to-head | `results/video_dit/REPORT_FINAL.md` (v2, Part II) | Summary statistics only; raw run logs are excluded from the compact supplement | `run_dit_temporal.py` |
| VideoGPT teacher-forced | `results/supporting_video/temporal_precision_report.md` | `results/supporting_video/temporal_precision/` | `eval_temporal_precision.py` |
| Phase collision analysis | — | `results/video_dit/phase_collision_analysis.json` | Theory computation |
| DiT theory analysis | `DiT_frequency_allocation_analysis.md` (root) | — | — |

---

## Experiment Phase Map (Script → Paper)

| Phase | Question | Paper Role | Key Scripts | → Figure/Table |
|-------|----------|-----------|-------------|----------------|
| 8 | Raw EVQ τ-sweep | Theory foundation | `phase8d_scaling_law.py`, `phase8f_multi_seed.py` | `table1_multiscale_raw_ppl.tex`, `fig6_tau_formula_validation.pdf` |
| 11 | PE-dominant regime | **Primary anchor for Table 4; supporting for L=256/454M scaling panels** | `phase11b_125m_dape.py`, plus `phase11_L256_extrap.py`, `phase11_yarn_eval.py`, `phase11c_454m_scaling.py` as supporting checks | `table4_pe_dominant.tex`, `table5_phase11_leverage.tex`, `fig3_pe_dominant_scaling.pdf` |
| 14 | EVQ+YaRN synergy | **Primary anchor (454M aggregate); supporting 50M/125M rerun** | `phase14c_multiscale_evq_yarn.py` supports the trend but is not the full Table 2 rerun | `table2_evq_yarn_main.tex`, `table3_capability_passkey.tex`, `fig2_evq_yarn_synergy.pdf` |
| 15 | 750M scale-up | Supporting | `phase15_750m_2k_to_4k_continue_ckpt_eval.py` | `table6_750m_continue_supporting.tex` |
| 16 | τ* formula validation | Theory confirmation | `phase16_formula_optimality_sweep.py` | `fig6_tau_formula_validation.pdf`, `table_lambda_cv.tex` |
| 17c | 454M continued pretrain | Supporting progressive-training check | `phase17c_454m_1024_to_2048_continue.py` | `fig4_phase17c_flagship.pdf` |
| 21b | Downstream QA | Downstream evidence | `phase21b_quality_eval_clean.py` | `fig5_downstream_qa.pdf` |

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
