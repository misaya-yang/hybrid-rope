# 论文全景评估：NeurIPS 可行性分析 (All in One)

> 最后更新：2026-02-22
> 基于全量代码仓库和实验数据的系统性顶会发文评估。

## 1. TL;DR

- **总判断**：有极高希望冲击 NeurIPS，但必须利用今晚的 8B overnight 数据补齐最重要的一环。
- **核心竞争力**：(1) 将频率分布视为独立的形状设计维度；(2) 横跨 50M 到 350M 及至 8B 的完整代码开源的从零至微调验证；(3) 透明且深刻的失效分析与重灾排查机制。
- **当前瓶颈**：8B 级公平竞技数据的缺失 (正在弥补)。

## 2. Claims (核心卖点主张)

- **Claim 1 (Novelty)**: 区别于过去单一放大 $\theta$ 的方法，本文首创将高频“硬性锚定”(Rigid Core Anchoring) 结合中低频的重新映射作为扩展文本长度的最优解。
  - *审稿维度*: Novelty & Effectiveness.
- **Claim 2 (Theoretical Foundation)**: 提出基于基带频率压缩引起的距离测度分布缩小乃至逆序的“Phase Collision”分析框架，精准预判崩塌边界点。
  - *审稿维度*: Theory.
- **Claim 3 (Transparency)**: 长文本外推能力的失效，往往来源于框架实现层面的基底不统一 (如 scaling API vs patching)，而不是数学本质。
  - *审稿维度*: Limitations & Engineering. 

## 3. Evidence Map

| 实验组别 | 核心证据提取 | 来源登记 |
|----------|--------------|----------|
| **从零训练主轴** | 多种子验证下 16K PPL的绝对及相对优势 ( 5% 至 13.7% ) | `EXP_50M_3SEED`, `EXP_350M_FINAL` |
| **机理验证大盘** | PPL随着长度激增的曲线坡度远小于基线 (改善 64%+) | `EXP_PHASE4_124M` |
| **8B Fair Pipeline** | 统一采用 `inv_freq.copy_()` | `EXP_8B_FAIR_LORA` (运行中) |

## 4. Failure & Fix (失败模式与防守应对)

- **Failure (审稿人可能的攻击)**: "如何确定你们的方法在更大参数或更先进模型同样生效？"
  - *Fix*: 我们的实验不是零散的改动，而是从 `TinyStories/50M` 逐级爬升至 `Llama3-8B`。只要证明在控制变量的公平协议下能胜过 PI/YaRN 即可。
- **Failure (负结果应对)**: "你们的方法改变了高频会造成 Zero-shot 毁灭。"
  - *Fix*: 我们在稿件中将主动披露：频率重构必须搭配前置的自适应微调，这不应当做缺点，而是注意力相位的本质特征。

## 5. What to write in paper (各部分贡献点映射)

| Section | 内容架构与关联实验 | 对应理论 |
|---------|--------------------|----------|
| **Intro & Background** | 引出距离表达与 Phase Collision 问题 | `06_phase_collision_D_analysis.md` |
| **Method** | Anchored Hybrid 的数学形式与高频保护定理 | `07_frequency_design_theory.md` |
| **Training (Scratch)** | 50M - 350M 的 PPL 跨度验证 | `01_已完成实验核心数据.md` |
| **Fine-Tuning (LoRA)** | 8B 参数级别上的公平对比 | `08_8b_experiment_analysis.md` |
| **Limitations / Failure** | 不正规评测引发的工程海市蜃楼、基础频率失效边界 | `03_负结果与风险复盘.md` |

## 6. Open Questions (下一步行动闭环)

- 明天一早首要任务：提取 `overnight` 的 8B 数据，并在 `docs/RESULTS.md` 与 Paper Draft 中立即生效数据。
- 是否需要跑极小样本量的 LongBench 验证一下 F1 分数作为附录补充？
