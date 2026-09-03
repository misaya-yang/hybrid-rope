# tau_algor/ — τ 理论历史推导（2026-03）

**状态：只读历史层。** 当前理论权威是 [`INDEX.md`](../../INDEX.md) §1。
本目录记录 τ scaling law 的原始推导过程，其中若干结论已被 2026-08 的
`optimization_notes.md` 明确修正。**逐条取代关系由本目录拥有，见下表**；
`INDEX.md` §1 只保留冷启动需要的当前边界，不复制该表。

引用本目录任何文件时必须标注为历史推导，不得作为当前 claim 依据。
`tau-theory-assistant` skill 使用本目录，因此文件保留原位、原名。

## 已被取代的结论（重要）

| 本目录的结论 | 现状 |
| --- | --- |
| $\tau_*\propto L^{-\gamma}$ 单一幂律，$\gamma\approx0.465$ | 被 O3 解释掉：$\tau_*$ 非幂律，是拟合形式错误 |
| $p\approx0.85$ 指数匹配诊断 | 被 O3 撤销 |
| $\tau_*=\max(d_{\rm head}/\sqrt L,\;1.4)$ 作为普适律 | 降级为 operating prior，见 `AGENTS.md` claim 上限「Finite tau」 |
| broadband surrogate 作为 collision 模型 | 被 full sin/cos 二维子空间与白化 cross-Gram 取代 |
| 重尾先验下最优分配为 arcsine 型 | 被 O5 数值证伪 |

## 文件

### 核心推导（按原阅读顺序）

| 文件 | 内容 |
| --- | --- |
| `TAU_SCALING_DERIVATION.md` | 起点：12 静态目标 × 18 配置，无法复现 $L^{-0.5}$；结论 broadband surrogate 丢信息 |
| `TAU_HABITABLE_ZONE.md` | τ 宜居带；$\tau_{\rm floor}=4/\sqrt K$ |
| `TAU_UNIFIED_THEORY.md` | 连续最优 × 离散约束 × 训练动力学，18 组实验 |
| `TAU_EXACT_DERIVATION_2026-03-23.md` | 精确推导版本 |
| `TAU_REGIME_THEORY_2026-03-24.md` | 分区理论 |
| `TAU_THEORY_DEEP_ANALYSIS_2026-03-24.md` | 最长的综合分析 |

### 诊断与实验

| 文件 | 内容 |
| --- | --- |
| `TAU_FIRST_PRINCIPLES_ANALYSIS_2026-03-22.md` | 第一性原理诊断：定位 L 信息断裂在 Step 3→4（broadband projection） |
| `TAU_STATIC_VS_DYNAMIC_EXPERIMENT_2026-03-22.md` | 数值判决：$L^{-0.5}$ 约 1/3 来自静态理论，2/3 来自训练动力学 |
| `TAU_STIFFNESS_DERIVATION_2026-03-24.md`、`PROMPT_STIFFNESS_P_DERIVATION.md` | stiffness 与 $p$ 的推导 |
| `TAU_SOFTMAX_TRANSPORT_THEORY_2026-03-23.md` | softmax transport |

### 跨架构与论文版

| 文件 | 内容 |
| --- | --- |
| `unified_tau_star_theory_v2.md` | 统一公式英文论文版 v2（v1 见 `../archive/`） |
| `mla_linear_vs_sqrt_correction_v1.md` | MLA 的 $d_{qk}/d_{\rm rope}$ 修正：线性 vs 平方根，结论线性 |

### 会话提示词

`PROMPT_FOR_GPT5.md`、`PROMPT_FOR_GPT5_V2.md` 是当时的外部模型提示词，
不是结论文档。

## 相关目录

- `../archive/` — 明确退役的 τ 文档（`unified_tau_star_theory_v1.md` 已被 v2
  取代；`tau_star_paper_ready_v1.md`；`TAU_THEORY_RESEARCH_REPORT_2026-03-22.md`）
- `../theory/` — 早期理论推导与数值验证
- `../../paper-2027/research/three_completions/` — 当前理论优化层（O1–O8）
