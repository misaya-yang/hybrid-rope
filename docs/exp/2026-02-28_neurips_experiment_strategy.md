# NeurIPS 2026 实验策略：从零训练 vs 微调的可行性分析

> **日期**: 2026-02-28
> **背景**: A2 EVQ τ=1.5 在 Llama-3-8B LoRA 微调中崩溃，需要重新设计实验路径
> **目标**: 用最低成本拿到 NeurIPS 可接受的实验证据

---

## 1. 直接回答：从零训练 1.5B 做 LongBench 不可行

| 需求 | 量级 | 单卡 96GB 耗时 |
|------|------|---------------|
| Chinchilla 最优预训练 | 30B tokens | ~70-100 天 |
| 足够的语言能力做复杂QA | 需要高质量多领域语料 | 数据工程量巨大 |
| 指令跟随能力 | 需额外 SFT 阶段 | 再加数天 |

**但这不是正确的问题。** 正确的问题是：NeurIPS PE 论文到底需要什么级别的实验？

---

## 2. NeurIPS 2024-2025 同类论文实验标准（实证调研）

### 2.1 已接受论文使用的模型规模和评测

| 论文 | 会议 | 模型规模 | 是否从零训练 | PPL | Passkey/NIAH | LongBench | 标准NLU |
|------|------|---------|-------------|-----|-------------|-----------|---------|
| **DAPE** | NeurIPS 2024 | **125M** | **从零训练** | Arxiv,Books3 | 否 | 否(用其他下游) | 否 |
| Base of RoPE | NeurIPS 2024 | **2B** | **从零训练** + 7B微调 | 是 | Long-eval | 否 | 否 |
| CREAM | NeurIPS 2024 | 7B | 微调 LLaMA-2 | 是 | Passkey | LongBench | 否 |
| YaRN | ICLR 2024 | 7B, 13B | 微调 LLaMA-2 | PG19,Proof-pile | Passkey | 否 | Open LLM |
| LongRoPE | ICML 2024 | 7B | 微调 LLaMA-2 | PG19,Proof-pile | Passkey | 否 | Open LLM |
| LongRoPE2 | ICML 2025 | 3.8B, 8B | 微调 | 是 | NIAH+RULER | LongBench等 | 是 |
| CLEX | ICLR 2024 | 7B, 13B | 微调 LLaMA-2 | RedPajama-Book | 否 | LongBench | 否 |
| Scaling Laws of RoPE | ICLR 2024 | 7B, 13B | 微调 | Books3 | 否 | 否 | 否 |

### 2.2 关键发现

1. **DAPE 是最重要的先例**：NeurIPS 2024 接受了仅用 **125M 从零训练** 模型的 PE 论文。
   训练长度 128，测试到 8192。评估用 PPL + 下游任务（不含 LongBench）。
   → **与我们的设置几乎完全匹配！**

2. **"Base of RoPE" 也是先例**：NeurIPS 2024，**2B 从零训练** + 7B 微调验证。
   关键发现：PPL 低不代表检索能力好 → **必须加 passkey/NIAH 评测**

3. **纯 PPL 已不被接受**：2024 年有专门论文 (ICLR 2025) 论证 PPL 对长上下文评估不可靠。
   → **至少需要 PPL + 一个合成检索任务**

4. **LongBench 不是必须的**：多数 NeurIPS/ICLR 接受的 PE 论文不含 LongBench。
   → **不需要为 LongBench 训练大模型**

---

## 3. 推荐策略：从零训练 scaling + 合成任务

### 3.1 核心方案（最低成本、最高确定性）

利用已有的 `run_evq_sweep.py` 基础设施，扩展实验维度：

| 实验 | 模型 | 训练量 | 评估 | 单卡96GB 耗时 | 优先级 |
|------|------|--------|------|--------------|--------|
| 50M τ-sweep | 50M x 8τ | 50M tok/run | PPL@2K/4K/8K/16K | **已完成** | — |
| 125M 双种子 | 125M x 2τ x 2seed | 100M tok/run | PPL@2K/4K/8K/16K | **已完成** | — |
| **350M τ=0 vs τ=1.5** | 350M x 2 | 350M tok/run | PPL + Passkey | ~1-2 天 | **P0** |
| **1B τ=0 vs τ=1.5** | 1B x 2 | 1B tok/run | PPL + Passkey | ~3-5 天 | **P1** |
| **Passkey 评测** | 所有已训模型 | 0（推理only） | Passkey@2K~32K | 数小时 | **P0** |
| **NIAH 评测** | 350M + 1B | 0（推理only） | Needle@多深度 | 数小时 | **P1** |

### 3.2 为什么这个方案足够

1. **Scaling law 证据**：50M → 125M → 350M → 1B，展示 EVQ 优势随规模放大
   - 已有数据：50M -10.9%, 125M -18.9%
   - 预期 350M 和 1B 进一步放大

2. **Passkey retrieval 直接测试位置编码质量**：
   - 不依赖语言理解能力，纯测位置感知
   - 小模型（甚至 137M）已被证明能做 passkey
   - EVQ 应该在此任务上展现明确优势（更好的频率分配 → 更好的位置辨别）
   - 已是 PE 论文的标准评测

3. **理论-实验闭环**：
   - EVQ warp curve → 预测中频维度旋转速度变化 → passkey 验证位置辨别
   - Phase collision score → 预测低频碰撞减少 → 长距离 PPL 验证
   - Waterbed inequality → 理论界 → PPL 数据验证

### 3.3 关于训练数据

当前使用 TinyStories。对 NeurIPS 有两种策略：

**策略 A：继续用 TinyStories（简单，可辩护）**
- 优点：已有基础设施，结果可复现
- 辩护方式：「控制变量实验，数据分布不影响相对比较」
- 风险：审稿人可能质疑 generalizability

**策略 B：换用标准语料（更强，推荐）**
- RedPajama / SlimPajama 子集 → 学术界主流选择
- PG19 / Proof-pile → YaRN/LongRoPE 使用的标准语料
- 需要修改数据加载代码，但工作量不大
- 显著增强论文说服力

**推荐**：350M 和 1B 实验用 SlimPajama 或 RedPajama 子集。
50M/125M 的 TinyStories 结果保留作为 preliminary / ablation。

---

## 4. 可选加强方案

### 4.1 方案 B：1.5B Continued Pretraining（全参数，非 LoRA）

| 参数 | 值 |
|------|-----|
| 基座 | Qwen2.5-1.5B 或 Phi-3-mini-3.8B |
| 训练方式 | **全参数 continued pretraining**（不是 LoRA） |
| 训练量 | 1-5B tokens |
| 数据 | SlimPajama / RedPajama |
| 评估 | PPL + Passkey + 可选 LongBench |

**与 8B LoRA 的关键区别**：
- 全参数训练有足够容量适应新频率（vs LoRA 只有 0.1% 参数）
- 1.5B 在 96GB 上全参 bf16 训练完全可行（~18GB 显存）
- 持续预训练让模型逐步适应 EVQ，而非冷启动

**耗时**：~4-7 天（1-2B tokens）。适合在 scaling 实验之后作为锦上添花。

### 4.2 方案 C：重试 8B LoRA + 小 τ

| 参数 | 值 |
|------|-----|
| tau | 0.3 或 0.5 |
| 其余 | 与 A1/A2 完全一致 |
| 耗时 | ~6-8 小时/run |

中频扰动降到 ~1.3x-1.7x，可能在 LoRA 可适应范围内。成本很低，值得尝试。

---

## 5. 论文实验章节规划（基于推荐方案）

```
Section 5: Experiments

5.1 Setup
    - EVQ-Cosh 实现（单参数 τ）
    - 从零训练：50M / 125M / 350M / 1B
    - 训练数据：[SlimPajama subset / TinyStories]
    - Baselines: geometric (τ=0), 8-point τ-sweep

5.2 Scaling Validation (Table 1)
    - 50M 全 τ-sweep（8 个 τ 值）
    - PPL@2K/4K/8K/16K
    - Phase collision scores

5.3 Cross-Scale Consistency (Table 2)
    - 50M / 125M / 350M / 1B，τ=0 vs τ=1.5
    - PPL 改善随规模放大
    - 双种子验证（125M+）
    - Waterbed 不等式验证

5.4 Positional Retrieval (Table 3 / Figure)
    - Passkey retrieval accuracy @ 2K/4K/8K/16K/32K
    - EVQ vs geometric 直接对比
    - 可选：Needle-in-a-haystack 热力图

5.5 [可选] Continued Pretraining (Table 4)
    - Qwen2.5-1.5B + EVQ continued pretraining
    - PPL + passkey + 下游任务
```

---

## 6. 时间线（9 周到 NeurIPS DDL）

| 周次 | 任务 | 交付物 |
|------|------|--------|
| Week 1 | 实现 passkey 评测 + 跑已有模型 passkey | passkey 结果表 |
| Week 1-2 | 350M τ=0/τ=1.5 从零训练 | PPL + passkey 结果 |
| Week 2-3 | 1B τ=0/τ=1.5 从零训练 | PPL + passkey 结果 |
| Week 3-4 | [可选] 1.5B continued pretraining | 下游验证 |
| Week 3-4 | [可选] 8B LoRA τ=0.3 快速验证 | gate 结果 |
| Week 4-6 | 论文实验章节 + 图表 | Section 5 draft |
| Week 6-8 | 全文修改 + 审稿模拟 | 完稿 |
| Week 9 | buffer + 提交 | 提交 |

---

## 7. 成本估算

| 实验 | GPU 小时 | 算力成本（约） |
|------|---------|-------------|
| 350M x 2 runs | ~48h | ~100 元 |
| 1B x 2 runs | ~120h | ~250 元 |
| Passkey 评测 | ~8h | ~15 元 |
| [可选] 1.5B CT | ~168h | ~350 元 |
| [可选] 8B LoRA τ=0.3 | ~8h | ~15 元 |
| **核心方案总计** | **~176h** | **~365 元** |
| **含可选方案总计** | **~352h** | **~730 元** |

---

## 8. 最终建议

### 不要做
- 从零预训练 1.5B 做 LongBench（时间不够，成本太高）
- 继续在 8B + τ=1.5 + LoRA 上投入（已证明不可行）
- 只报告 PPL（审稿人会打回）

### 必须做
- 添加 **passkey retrieval** 评测（PE 论文标准要求，且小模型可做）
- 至少扩展到 **350M** 规模（展示 scaling）
- **双种子验证**（统计显著性）

### 建议做
- 扩展到 **1B** 规模（更强的 scaling evidence）
- 换用 **SlimPajama/RedPajama** 语料（审稿人更认可）
- 快速试一轮 **8B LoRA τ=0.3**（成本极低，成功了是强加分项）

### 锦上添花
- 1.5B continued pretraining（如果时间和预算允许）
- Needle-in-a-haystack 热力图（视觉效果好）
- RULER 子集评测

---

## 9. 参考先例

| 论文 | 与我们的相似度 | 关键策略 |
|------|-------------|---------|
| **DAPE (NeurIPS 2024)** | ★★★★★ | 125M 从零训练，PPL + 下游，无 LongBench |
| Base of RoPE (NeurIPS 2024) | ★★★★ | 2B 从零 + 7B 微调，PPL + Long-eval |
| Scaling Laws of RoPE (ICLR 2024) | ★★★ | 主要 PPL，强理论补偿 |
| YaRN (ICLR 2024) | ★★ | 7B 微调，PPL + passkey |
