# 术语与协议规范 (Terms & Protocols)

> 最后更新：2026-02-22
> 本文档定义本仓库/论文中所有实验的统称、度量指标计算方式、以及公平对比协议的判定标准。所有用于论文的数据必须符合本协议。

## 1. 核心术语定义 (Terminology)

- **Standard / Geo**: 基于传统 LLaMA 的 RoPE 频率分配方式（即 `theta` 几何递减缩放）。
- **Hybrid (Anchored Hybrid)**: 本文提出的频率设计方式，其核心是“高频保持绝对不变 (anchored)”+“中低频采用非线性或凸组合映射 (hybrid/sigmoid)”。
- **Phase Collision / D(Δ)**: 我们提出的理论框架，指出距离函数 D(Δ) 会因为 theta 的极度放大而缩小甚至排名反转，进而导致注意力崩塌。这是解释“为什么无限大 theta 不好”的核心理论。

## 2. 评测指标与长度定义 (Metrics & Context Lengths)

### 2.1 PPL 计算口径
为避免 Definition Drift，所有长文本 PPL 评测需遵守：
- **数据集**: TinyStories (Streaming) / Wikitext
- **Tokenizer**: 统一使用原模型 Tokenizer (如 gpt-neox-20b 或 llama3)，固定 `add_special_tokens=False` (除非特殊说明)。
- **Slicing (滑动窗口验证)**: 给定一个评测长度 `L`，取连续的 `EVAL_CHUNKS` 个组（如 Chunk0: 0~L, Chunk1: L~2L），确保同一评测流数据确定不变。
- **目标对比长度**: 主看模型训练外推的泛化性能，核心指标报告 `PPL@16K` 与 `PPL@32K`。

### 2.2 NIAH (Needle In A Haystack) 口径
- 采用 4 宫格或标准热力图评测。以 Accuracy 表示对应深度与对应文档长度下的召回率。长程评测必须跨越 4K, 8K, 16K, 32K 边界点。

### 2.3 LongBench 任务解释
- 计算采用官方的 F1 或 Rouge-L 指标。主要考察模型不仅能 recall，且能进行长程 reasoning 的能力。

### 2.4 Seed 稳健性统计
对于小规模 (50M/100M) 从零训练，单次 run 极易受 data streaming 取样噪声影响（“lucky seed”），必须执行 **多种子实验**。
- **规定**: PPL 等核心比较指标必须基于至少 3 个 seed 的结果计算 `mean ± std`。单 seed 结果可作为趋势参考，但不能作为论文主张的坚实证据。

## 3. 公平协议判定标准 (Fairness Protocol)

对于 LoRA 或 Fine-tuning 的拓展基准测试（如针对 8B 及以上大模型），**严禁存在实现层面带来的偏差**。

### ✅ 符合公平标准 (Fair)
- 必须统一通过 **`inv_freq.copy_()` buffer 覆写** 的方式在所有基线间进行频率修改。
- `rope_scaling` 必须禁用或统一设置为 `None`。
- 其他基线如 YaRN/PI 等的缩放因子应当手动计算并注入相同的 `inv_freq` buffer 中，以确保前向传播、长程注意力切片等系统环境一模一样。

### ❌ 不符合公平标准 (Unfair)
- 基线 (PI/YaRN) 走 HuggingFace 官方的 `rope_scaling` API，而新方法 (Hybrid/Sigmoid) 走自定义 forward monkey patch。这会导致底层 Kernel (如 Flash Attention 处理方式) 发生变化，造成 loss/PPL 不可比。

---

## 4. 禁止引用清单 (Do Not Cite List)

以下数据因违反公平协议、未调优或存在已知严重缺陷，**明确禁止在论文正面结论或任何公共沟通中引用为有效数据**。它们只能作为 Limitations / Failure 分析使用。

| 实验/数据点 | 废弃原因 | 替代方案 |
|-------------|----------|----------|
| **2026-02-13 版的 8B LoRA YaRN/PI vs Hybrid** | 使用 monkey patch vs rope_scaling，不符合公平协议；且 Hybrid 缺高频保护参数导致中频饿死。 | 参见 `EXP_8B_FAIR_LORA` (2026-02-22 夜间执行的公平架构实验)。 |
| **50M Base=300K Sigmoid vs Standard** | Base 选型不当触发的局部失效 (Sigmoid PPL 劣于 Standard)。 | 列入 Limitations，作为“频率重划分必须正确挑选 base” 的反例。 |
| **Zero-shot 直接替换频率 (无 FT)** | 相位未经过微调适应，导致极大损失和注意力崩塌。 | 列入 Limitations，证明本方法必须依赖少量步骤的相位适配(微调)。 |

> 附录：本文档为论文叙事提供底线保障，凡是违背的必须降级为 draft/trash。
