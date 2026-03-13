# Core Text Phases — 实验主链

Phase 8–21 核心文本实验链，直接支撑论文所有 claims。本目录是整个仓库最核心的代码。

完整的 Figure/Table → Script → Data 追溯见 `docs/overview/PAPER_CLAIMS_MAP.md`。

---

## Phase Map (含论文追溯)

| Phase | 核心问题 | 论文角色 | 主脚本 | → Paper |
|-------|---------|---------|-------|---------|
| **8** | EVQ raw scaling law | Theory foundation | `run_evq_sweep.py`, `phase8d_scaling_law.py`, `phase8f_multi_seed.py` | Table 1, Fig 6 |
| **11** | PE-dominant regime + YaRN interaction | **Primary anchor** | `phase11_L256_extrap.py`, `phase11_yarn_eval.py`, `phase11b_125m_dape.py`, `phase11c_454m_scaling.py` | **Fig 3, Tables 4-5** |
| **13** | Downstream NLL probe | Supporting | `phase13a_longbench_nll.py` | Appendix |
| **14** | EVQ + YaRN >> Geo + YaRN | **Primary anchor** | `phase14c_multiscale_evq_yarn.py`, `phase14d_125m_tinystories_10pct.py` | **Fig 2, Tables 2-3** |
| **15** | Larger-scale continued pretrain | Supporting scale-up | `phase15_750m_2k_to_4k_continue_ckpt_eval.py`, `phase11e_continued_pretrain.py` | Table 6 |
| **16** | τ* formula validation | Theory confirmation | `phase16_formula_optimality_sweep.py` | **Fig 6** (99-run) |
| **17b** | 454M L=512→1024 continue | Stage 2 | `phase17b_454m_512_to_1024_continue_ckpt_eval.py` | Fig 4 (middle) |
| **17c** | 454M L=1024→2048 continue | **Flagship demo** | `phase17c_454m_1024_to_2048_continue.py`, `phase17c_extended_eval.py` | **Fig 4** (final) |
| **18** | RoPE base 泛化 | Appendix ablation | `phase18_base_generalization_sweep.py` | Appendix |
| **21b** | QuALITY downstream eval | Downstream evidence | `phase21b_quality_eval_clean.py`, `phase21b_scrolls_finetune.py` | **Fig 5** |

---

## 脚本分类

### 实验脚本 (`phase*.py`)

每个实验脚本的 docstring 头部包含标准化的 **Paper Role / Input / Output / Seeds** 字段，可通过 `grep "Paper Role" *.py` 快速检索。

| 脚本 | Paper Role | 模型规模 | Seeds |
|------|-----------|---------|-------|
| `run_evq_sweep.py` | Table 1, Fig 6 — multi-scale τ-sweep | 50M/125M/350M/500M | multi |
| `phase11_L256_extrap.py` | Fig 3 (b,c) — PE-dominant raw PPL | 350M | 42,123,7 |
| `phase11_yarn_eval.py` | Fig 3 (c) — +YaRN PPL curves | 350M | 42,123,7 |
| `phase11b_125m_dape.py` | Fig 3 (a), Table 4 — EVQ vs DAPE | 125M | 42,123,7 |
| `phase11c_454m_scaling.py` | Table 4 — 454M token scaling | 454M | single |
| `phase11e_continued_pretrain.py` | Supporting — Geo→EVQ retrofit | 454M | configurable |
| `phase11f_token_scaling_454m.py` | Supporting — token scaling trend | 454M | configurable |
| `phase13a_longbench_nll.py` | Appendix — LongBench NLL | 750M | 42 |
| `phase14c_multiscale_evq_yarn.py` | **Fig 2, Tables 2-3** — EVQ+YaRN synergy | 125M-350M | 3+3 |
| `phase14d_125m_tinystories_10pct.py` | Supporting — TinyStories 10% mix | 125M | — |
| `phase15_750m_2k_to_4k_continue_ckpt_eval.py` | Table 6 — 750M continued-pretrain | 750M | 42 |
| `phase16_formula_optimality_sweep.py` | **Fig 6** — 99-run τ* validation | 50M/125M | 3 seeds × 9 configs |
| `phase17b_454m_512_to_1024_continue_ckpt_eval.py` | Fig 4 (Stage 2) | 454M | 42 |
| `phase17b_full_grid_eval.py` | Fig 4 — full grid eval | 454M | — |
| `phase17c_454m_1024_to_2048_continue.py` | **Fig 4** (Stage 3) — flagship | 454M | 42,43,44 |
| `phase17c_extended_eval.py` | Fig 4 — extended 2K-48K eval | 454M | — |
| `phase17c_resume_evq.py` | Fig 4 — resume interrupted training | 454M | 42 |
| `phase18_base_generalization_sweep.py` | Appendix — base sweep | configurable | single |
| `phase18_simple.py` | Appendix — simplified base sweep | configurable | single |
| `phase21b_quality_eval.py` | Fig 5 — QuALITY eval (legacy) | 454M | — |
| `phase21b_quality_eval_clean.py` | **Fig 5** — QuALITY eval (canonical) | 454M/750M | n=2086 |
| `phase21b_scrolls_finetune.py` | Downstream — SCROLLS finetune | configurable | — |

### 评估工具 (`eval_*.py`)

可复用的评估脚本，被多个 Phase 调用。

| 脚本 | 功能 | 被哪些 Phase 使用 |
|------|------|-----------------|
| `eval_passkey.py` | Passkey 检索准确率 | Phase 14c, 17b, 17c |
| `eval_multi_needle.py` | Multi-needle NIAH | Phase 15, 17b |
| `eval_longbench_nll.py` | LongBench 分任务 NLL | Phase 13a, 21a |
| `eval_pe_baselines.py` | PE 基线对比 | Phase 11 |
| `eval_super_extrap.py` | 极端外推测试 | Phase 11 |
| `eval_dsr.py` | Distance Sensitivity Ratio | 独立评估 |

### 分析与可视化

| 脚本 | 功能 | 输出 |
|------|------|------|
| `evq_analysis.py` | τ-sweep 分析 + waterbed 绘图 | Appendix figures |
| `visualize_attention_distance.py` | Attention distance 分布可视化 | paper/figs/attn_*.pdf |

---

## 实验数据流

```
FineWeb-Edu (streaming)
    ↓
scripts/data_prep/prepare_mixed_prior_dataset_v1.py → .pt 文件
    ↓
run_evq_sweep.py (Phase 8)     → results/core_text/evq_sweep/
    ↓ 基础模型 checkpoint
phase11*.py (Phase 11)          → results/core_text/phase11*/
phase14c (Phase 14)             → results/core_text/phase14c/
    ↓ 续训 checkpoint
phase17b (Stage 2)              → results/core_text/phase17b/
phase17c (Stage 3)              → results/core_text/phase17c/
    ↓ 下游评估
phase21b (QuALITY)              → results/core_text/phase21b/
    ↓ 出图
scripts/figures/fig*.py         → paper/figs/fig*.pdf
```

---

## 阅读顺序

| 步骤 | 脚本 | 目的 |
|------|------|------|
| 1 | `run_evq_sweep.py` | 理解模型架构 + EVQ 实现 + 训练循环 |
| 2 | `phase11_L256_extrap.py` | PE-dominant regime 核心逻辑 |
| 3 | `phase14c_multiscale_evq_yarn.py` | YaRN 集成 + passkey 评估 |
| 4 | `phase17c_454m_1024_to_2048_continue.py` | 续训流程 |
| 5 | `phase21b_quality_eval_clean.py` | 下游评估 |

---

## 命名规则

- `phaseXX_*.py`: Phase 实验脚本 (包含训练 + 评估)
- `eval_*.py`: 可复用评估工具 (被多个 Phase 调用)
- `visualize_*.py`: 可视化脚本

## Docstring 规范

每个脚本的 docstring 应包含以下标准字段:

```python
"""
Phase XX: 简短描述

详细描述...

Paper Role:  Fig X / Table Y — 具体贡献
Input:       所需输入数据/checkpoint
Output:      输出路径和格式
Seeds:       随机种子配置 (如适用)
"""
```
