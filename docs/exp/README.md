# Experiment Reports Index

实验报告归档目录。所有报告使用 `YYYY-MM-DD_slug.md` 命名。

完整的 Paper ↔ Experiment 映射见 `docs/overview/PAPER_CLAIMS_MAP.md`。

---

## 核心 Anchor 报告 ⭐

以下报告直接支撑论文主要 claim，是审稿复现的关键依据:

| 报告 | Claim | → Paper | Seeds | 置信度 |
|------|-------|---------|-------|-------|
| `2026-03-03_passkey_mix_results.md` | C3: EVQ+YaRN 100% vs 61-65% | **Fig 2, Tables 2-3** | 3+3 | ★★★★★ |
| `2026-03-04_phase11_L256_results.md` | C2: EVQ ≥ Learnable PE | **Fig 3, Tables 4-5** | 3 | ★★★★★ |
| `2026-03-05_phase11b_125m_results.md` | C2: EVQ vs DAPE | **Fig 3 panel (a)** | 3 | ★★★★★ |
| `2026-03-09_phase16_formula_optimality_sweep_results.md` | τ* = d_head/√L | **Fig 6** | 99 runs | ★★★★★ |
| `2026-03-11_phase17c_2048_continue_results.md` | C3/C4: 454M flagship | **Fig 4** | 42-44 | ★★★★ |
| `2026-03-12_phase21b_454m_full_eval_report.md` | C5: Downstream NLL | **Fig 5** | n=2086 | ★★★★ |

---

## 全部报告 (按时间排序)

| 报告 | 描述 | → Paper | 类别 |
|------|------|---------|------|
| `2026-02-24_128tok_baseline_report.md` | 128-token 基线实验 | Background | 基础 |
| `2026-02-24_128tok_baseline_results.md` | 128-token 基线结果数据 | Background | 基础 |
| `2026-02-25_phase6_initial_results.md` | Phase 6 初始 EVQ 结果 | Early validation | 基础 |
| `2026-02-26_full_experiment_report.md` | 完整实验报告 (早期) | Historical | 历史 |
| `2026-02-27_evq_tau_sweep_results.md` | τ-sweep 基础 + scaling law 信号 | **Table 1, Fig 6** | ⭐ 核心 |
| `2026-03-01_video_temporal_transfer.md` | 视频时序迁移实验 | Appendix | 跨模态 |
| `2026-03-03_passkey_mix_results.md` | Multi-seed passkey + EVQ×YaRN | **Fig 2, Tables 2-3** | ⭐ 核心 |
| `2026-03-03_phase9f_50pct_checkpoint_report.md` | 750M checkpoint dynamics | Supporting | 规模 |
| `2026-03-04_phase11_L256_results.md` | Phase 11 PE-dominant regime | **Fig 3, Tables 4-5** | ⭐ 核心 |
| `2026-03-05_phase11b_125m_results.md` | 125M DAPE-style 对比 | **Fig 3 panel (a)** | ⭐ 核心 |
| `2026-03-06_phase15_750m_2k_to_4k_continue_results.md` | 750M 续训 (2K→4K) | **Table 6** | 规模 |
| `2026-03-09_phase16_formula_optimality_sweep_results.md` | τ* formula 99-run 验证 | **Fig 6** | ⭐ 核心 |
| `2026-03-09_phase17_evq_yarn_overlay_results.md` | EVQ+YaRN overlay | Supporting | 分析 |
| `2026-03-10_phase17b_1024_continue_vs_512_baseline.md` | 454M 512→1024 续训 | Fig 4 (Stage 2) | 续训 |
| `2026-03-10_theory_numerical_verification.md` | 理论数值验证 | §3 Theory | 理论 |
| `2026-03-11_phase17c_2048_continue_results.md` | 454M 1024→2048 续训 | **Fig 4** | ⭐ 核心 |
| `2026-03-11_test3_broadband_r2_validation.md` | Broadband R² 验证 | Theory appendix | 理论 |
| `2026-03-12_phase21_quality_downstream_report.md` | QuALITY 下游初步 | Fig 5 | 下游 |
| `2026-03-12_phase21b_454m_full_eval_report.md` | 454M QuALITY 完整评估 (n=2086) | **Fig 5** | ⭐ 核心 |

---

## 推荐阅读顺序

| 步骤 | 报告 | 为什么 |
|------|------|-------|
| 1 | `2026-03-03_passkey_mix_results.md` | 论文最强结果 (100% vs 61-65%) |
| 2 | `2026-03-04_phase11_L256_results.md` | PE-dominant regime 主要证据 |
| 3 | `2026-03-11_phase17c_2048_continue_results.md` | 454M flagship 续训链 |
| 4 | `2026-03-12_phase21b_454m_full_eval_report.md` | 下游 Gold NLL 证据 |
| 5 | `2026-03-09_phase16_formula_optimality_sweep_results.md` | 99-run 理论验证 |

---

## 归档规则

- 每个报告必须能追溯到论文 Figure/Table、理论文档或协作 brief
- 使用 `YYYY-MM-DD_slug.md` 命名格式
- 核心报告标注 ⭐，supporting 报告保留但降低优先级
- 不再被论文任何部分引用的报告应移至 `internal/`
