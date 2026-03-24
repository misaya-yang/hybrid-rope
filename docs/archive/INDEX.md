# τ* 理论文档索引

> 最后更新: 2026-03-22
> 本文件夹收录所有关于 τ 参数理论推导的分析文档。

---

## 文件总览

### 核心理论（按阅读顺序）

| # | 文件 | 来源 | 语言 | 内容 |
|---|------|------|------|------|
| 1 | `TAU_SCALING_DERIVATION.md` | Claude Code | EN | **起点**：完整的 τ* scaling law 理论分析。测试了 12 个静态目标 × 18 种配置，发现无法复现 L^{-0.5}。结论：broadband surrogate 丢失信息 |
| 2 | `TAU_HABITABLE_ZONE.md` | Claude Code | CN | τ 的"宜居带"理论。推导 τ_floor = 4/√K，解释为什么 τ≈1.5 是跨架构的 sweet spot |
| 3 | `TAU_UNIFIED_THEORY.md` | Claude Code + Codex | CN | 统一理论：连续最优 × 离散约束 × 训练动力学。最终公式 τ* = max(d_head/√L, 1.4)，18 组实验验证 |

### 本次研究（2026-03-22，Cowork session）

| # | 文件 | 内容 |
|---|------|------|
| 4 | `TAU_THEORY_RESEARCH_REPORT_2026-03-22.md` | **系统性综述**：跨 MHA/MLA/DiT 三架构的 τ 理论总证据表，分 A-F 六节（证据概览、现象总结、τ理论分析、有效条件、论文措辞、实验建议） |
| 5 | `TAU_FIRST_PRINCIPLES_ANALYSIS_2026-03-22.md` | **第一性原理诊断**：追踪推导链 6 步中 L 信息的流向，定位断裂在 Step 3→4 (broadband projection)。提出三条修补路径 (A: 三参数 surrogate, B: 自洽, C: 训练动力学) |
| 6 | `TAU_STATIC_VS_DYNAMIC_EXPERIMENT_2026-03-22.md` | **数值判决实验**：在 2 种 prior × 7+ 静态目标上直接优化 τ。结论：L^{-0.5} 约 1/3 来自静态理论（自洽 surrogate 贡献 L^{-0.17}），2/3 来自训练动力学 |

### 跨架构 & 论文

| # | 文件 | 来源 | 内容 |
|---|------|------|------|
| 7 | `unified_tau_star_theory_v2.md` | Claude + Codex | 统一公式的英文论文版本 v2（基于 v1 + Codex audit 修订） |
| 8 | `mla_linear_vs_sqrt_correction_v1.md` | Codex | MLA 的 d_qk/d_rope 修正应该是线性还是平方根？结论：线性更合理 |
| 9 | `tau_star_paper_ready_v1.md` | Claude Code | τ* 定理的论文就绪表述（MHA baseline case） |

### 归档

| 文件 | 原因 |
|------|------|
| `archive/unified_tau_star_theory_v1.md` | 被 v2 取代 |

---

## 来源说明

- **Claude Code**: 由 Claude Code CLI 在终端中生成，通常是英文，格式偏论文/appendix 风格
- **Codex**: 由 OpenAI Codex 生成，通常做 full-repo audit 和交叉验证
- **Cowork**: 本次 Cowork session (2026-03-22) 生成，中文，偏研究分析风格

## 关键结论演进

```
TAU_SCALING_DERIVATION (Claude Code, 3/21)
  → "L^{-0.5} 无法从静态目标推导" (12 个目标全部失败)

TAU_HABITABLE_ZONE (Claude Code, 3/21)
  → "τ 有宜居带 [1.0, 2.5]，floor = 4/√K"

TAU_UNIFIED_THEORY (Claude Code + Codex, 3/21)
  → "统一公式 τ* = max(d_head/√L, 1.4)"

TAU_FIRST_PRINCIPLES_ANALYSIS (Cowork, 3/22)
  → "断裂在 broadband projection，提出自洽修补路径"

TAU_STATIC_VS_DYNAMIC_EXPERIMENT (Cowork, 3/22)
  → "自洽 surrogate: L^{-0.17}，填了 1/3 gap"
  → "L^{-0.5} 主要是训练动力学涌现属性"
  → "类比：η* ∝ 1/√T (SGD) 也不来自 loss landscape"
```

## 实验代码

- `scripts/analysis/tau_scaling_analysis.py` — 原始 broadband 系数分析 (α,β vs L)
- `scripts/tau_fp_final.py` — 本次新增：两种 prior × 多目标直接优化实验
