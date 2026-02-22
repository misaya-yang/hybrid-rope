# 图表规划 (Figure and Table Plan)

> 最后更新：2026-02-22
> 本文档定义了最终 LaTeX 论文草稿所需的核心图表清单，并严格绑定至仓库的客观数据源。

## 1. 核心图 (Figures)

### Figure 1: The Shape Matters (频谱形状概念与整体性能收益)
- **描述**: 跨不同大语言模型规模 (50M, 100M, 350M) 的 PPL-Length 外推曲线。展现 Geometric Base 策略在 16K 后的崩溃，以及 Anchored Hybrid 方案的平稳延伸。
- **数据溯源**: 
  - 50M: `results/evidence_chain_50m_3cfg3seed/results.json`
  - 100M: `artifacts/a100_2026-02-13/data/100m_scaling/`
  - 350M: `artifacts/a100_2026-02-13/data/350m_final/results.json`
- **对应理论段落**: Introduction / Method - 频率形态重塑如何根本性延缓 Phase Collision。

### Figure 2: Phase Collision Mechanism & Distance Distribution
- **描述**: 双坐标轴图。左轴展现真实注意力的距离衰减偏好近似 Power-law (D(Δ) ∝ Δ^(-γ))；右轴展现 Theta 极度放大时，几何频带的 Phase 聚集与排序反转。
- **数据溯源**: `results/anchored_sigmoid_v3_followup/` 与 `results/attention_distribution/` (L2 $\gamma=1.31$, avg $\gamma=0.72$)
- **对应理论段落**: Theoretical Proofs - D(Δ) 分布决定理论相变点。

### Figure 3: 8B LoRA Training Dynamics / Context PPL
- **描述**: 在 Llama-3-8B 严格公平的 `inv_freq.copy_()` 架构下，四种策略 (Baseline, PI, YaRN, Anchored Hybrid) 的训练 Loss 下降曲线以及对应的多档位 PPL 柱状图。
- **数据溯源**: (进行中) `results/overnight_8h/summary/loss_curves.csv` (待产出)
- **对应理论段落**: Experiments - 大规模微调下的真实对等扩展能力。

### Figure 4: NIAH Accuracy Heatmaps (4-Grid)
- **描述**: 4个 11x10 的热力图矩阵，分别展示四大策略在 4K-32K 针刺测试中的找回精确度。展现中频饿死与高频保护的微观效果差异。
- **数据溯源**: (进行中) `results/overnight_8h/summary/niah/` (待产出)
- **对应理论段落**: Experiments - 长程推理的精度解析。

---

## 2. 核心表 (Tables)

### Table 1: 从零预训练的主线 PPL 矩阵
- **描述**: 综合对比了所有规模和核心配置在外推窗口上的 PPL，并提供多随机种子 (3-seed) 的置信区间。
- **涉及列**: Model Size, Train Tokens, Target Eval Context, Baseline PPL (Geo), Hybrid PPL, Relative Improvement.
- **数据溯源**: 参见 `docs/EXPERIMENT_REGISTRY.md` 中的 Tier 1。

### Table 2: 实施海市蜃楼与频谱失效模式图谱 (Implementation Mirages & Failures)
- **描述**: 首创于同类文献，真实记录在不同规模和不当操作下的架构崩溃或泛化失败，并给予严格归因。体现极为深刻的方法论。
- **包含条目**: 
  - Zero-shot 频域调换崩溃 (缺乏自适应微调)
  - 过度扩大 Base 的灾难 (Base=300K 毁损高频, PPL劣于标准)
  - 非公平注入 (Scaling vs Array Patch 导致的系统误差)
- **数据溯源**: `knowledge_base/03_负结果与风险复盘.md` 与 `docs/EXPERIMENT_REGISTRY.md` 的 Tier 4 黑名单。
- **对应理论段落**: Discussion & Limitations - Engineering vs Mathematical boundaries.
