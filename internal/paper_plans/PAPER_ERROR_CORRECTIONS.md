# 论文初稿错误纠正清单

> 日期：2026-03-01
> 文件：`evq_neurips2026_draft.tex`
> 目的：在下一版论文更新前，彻底记录所有事实性错误，防止再犯

---

## 错误 1：LLaMA-3-8B LoRA 实验不应写入论文

### 论文中写了什么
§5.3 (lines 488-507) 写了 "Llama-3-8B (5 schedules, 600 steps)"，声称 EVQ 在 LongBench mean 上 +14.5%，并给了一个 5 方法对比表。

### 真实情况
- LLaMA-3 8B 实验来自 `llama8b_fair_lora_suite_20260214`，只有 600 steps 的 LoRA
- 该实验被标记为问题实验（参见 `03_负结果与风险复盘.md`、`08_8b_experiment_analysis.md`）
- Hybrid 在此实验中 PPL@16K=11.875，远逊于 YaRN 的 6.057 和 PI 的 6.137
- LongBench 只跑了 6 个任务（不是 21 个），分数都极低（<0.2）
- 整个 LLaMA-3 实验由于协议问题（早期不公平比较遗留），**没有意义，不应写入论文**

### 纠正方案
**删除整个 LLaMA-3-8B LoRA 小节和对应的 Table 3。** 论文的 LoRA 部分只保留 Qwen-2.5-7B。

---

## 错误 2：LongBench 任务数量写错

### 论文中写了什么
Table 3 caption 写 "(600 steps, 6 LongBench tasks)"，暗示整个 LongBench 只有 6 个任务。

### 真实情况
- LLaMA-3 的 LongBench 确实只跑了 6 个任务（qasper, hotpotqa, 2wikimqa, multi_news, gov_report, narrativeqa）——这本身就是问题之一
- **Qwen-2.5-7B 跑了完整的 LB-21（21 个任务）**，包括中文任务、代码任务、检索任务等
- Qwen 数据在 `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed{42,1337}/`
- 双种子结果：
  - seed42: baseline avg 44.44, hybrid avg 44.08, delta = **-0.35**
  - seed1337: baseline avg 44.47, hybrid avg 44.05, delta = **-0.42**

### 纠正方案
Qwen 部分改为 "full LB-21 (21 tasks)"。去掉 LLaMA 的 6 任务表述。

---

## 错误 3：Qwen 结果不是"总体小退化"那么简单

### 论文中写了什么
"aggregate score shows a small regression (−0.39 points)"

### 真实情况
这个 -0.39 的数字大致正确（实际 seed42=-0.35, seed1337=-0.42），但问题是：
- 这是 **anchored_sigmoid** 方法（旧方法），不是 EVQ τ
- 当前论文是 V5 (EVQ/τ)，不应该直接把 V4 (anchored_sigmoid) 的 Qwen 数据写成 EVQ
- Qwen 实验用的是 TinyStories 数据训练 LoRA 400 steps，不是 FineWeb
- Qwen 本身 base θ=10^6，原生支持 32K+ 上下文，用短步 LoRA 去改它的频率基本没收益

### 纠正方案
如果要保留 Qwen 数据，必须明确：
1. 这是 anchored_sigmoid 方法，EVQ τ 的 8B 实验尚未做
2. 训练数据是 TinyStories（不匹配 Qwen 的预训练分布）
3. 400 steps 极短，与 LongRoPE2 的 10B token 不可比
4. 21 个任务，不是 6 个

---

## 错误 4：Wikitext 跨模型实验存在问题

### 论文中可能引用了什么
cross_model_wikitext_v1 的数据

### 真实情况
该实验是 **zero-shot 直接替换频率**（无训练），在 wikitext-103 validation 上 eval：
- LLaMA-3 geo_10k: PPL@16K = **12111**（崩溃）→ sigmoid: PPL@16K = **12.57**（稳定）
- Qwen orig: PPL@16K = **6.98** → Qwen geo_100k: PPL@16K = **7.16**
- 这个实验验证的是 **phase collision 现象**（LLaMA 用小 θ 会崩），不是 EVQ 的效果
- Wikitext 评测本身在 NeurIPS 审稿人看来可能不够 compelling（太旧）

### 纠正方案
该实验可以用在 Phase Collision 的 motivation 部分，但不能作为 EVQ 效果的证据。

---

## 错误 5：50M 实验中 τ=2.0 其实更差

### 论文中写了什么
暗示 τ 越大越好（基于 128-tok 的 sweep 数据）

### 真实情况（极关键！）
**50M TinyStories 从零训练（2K tokens, 500K steps）的 τ sweep：**

| τ | PPL@16K | Δ vs Geo |
|---|---------|----------|
| 0.0 (Geo) | 33.316 | — |
| 0.2 | 42.314 | **+27.0%** |
| 0.4 | 33.298 | -0.1% |
| 0.6 | 37.978 | +14.0% |
| 0.8 | 36.306 | +9.0% |
| 1.0 | 37.369 | +12.2% |
| **1.5** | **29.697** | **-10.9%** |
| 2.0 | 35.646 | **+7.0%** |

**τ=2.0 比 geometric 更差！** Peak 在 τ=1.5。

而 128-tok 实验（FineWeb, 125M, 15M tokens）的 sweep：
| τ | PPL@8K | Δ vs Geo |
|---|--------|----------|
| 0.0 | 513.7 | — |
| 1.0 | 477.5 | -7.1% |
| 1.5 | 419.7 | -18.3% |
| 2.0 | 406.1 | -20.9% |
| 2.5 | 383.3 | -25.4% |

这里 τ=2.5 还在改善。**两组实验的 τ* 完全不同。**

### 原因分析
- 50M 实验是 **2K token 训练**，模型容量更大（相对 128 tok），部分频率已被模型权重吸收
- 128-tok 实验是 **PE-dominant regime**，模型太弱无法补偿，PE 参数直接决定外推质量
- τ* 取决于训练 regime：短上下文（PE-dominant）→ 高 τ* ≈ 2.7；长上下文（model-dominant）→ 低 τ* ≈ 1.5

### 纠正方案
论文不能说 "τ* ≈ 2.7 是普适最优"。正确说法：
- τ* 取决于训练 regime 和 model capacity
- 128-tok regime: τ* ≈ 2.7（外推 PPL 最优）
- 2K-tok regime: τ* ≈ 1.5（50M 数据支持）
- 实际部署建议：先用 τ=1.5 作为安全默认，然后用 mini-sweep 调优

---

## 错误 6：LoRA 步数问题需要正面回应

### 审稿人可能的质疑
"LoRA 只有 400-600 步，太短了。LongRoPE2 用了 10B token，Meta 用了 80B。增益在更长微调后可能消失。"

### 正确的回应策略
1. **承认限制**：计算资源限制了 LoRA 规模
2. **论证相关性**：引用 2024-2025 年被接受的同类工作（CLEX、ReRoPE 等），它们也使用了类似规模的 LoRA
3. **EVQ 的核心贡献不在 LoRA**：EVQ 的价值是理论框架（变分反问题、cosh 族的 Euler-Lagrange 推导）+ 从零训练的 PE quality 验证
4. **LoRA 是补充证据**：不是 claim 的核心支撑

---

## 错误 7：缺少的消融实验

### 审稿人会问的
1. Fisher 脉冲 vs 纯 cosh 的消融
2. τ 学习率敏感性分析
3. HoPE、PaTH、CREAM 等 2025 新基线

### 现状
- Fisher 消融和 τ lr 敏感性：未做
- 新基线对比：未做，后续会补

### 纠正方案
不要在论文中声称已经做了这些实验。在 Limitations 中诚实写明。

---

## 总结：论文中需要删除/修改的部分

| 位置 | 问题 | 操作 |
|------|------|------|
| §5.3 LLaMA-3-8B LoRA 段落 + Table 3 | 实验无意义 | **整段删除** |
| §5.3 Qwen "6 tasks" | 任务数错误 | 改为 "21 tasks (LB-21)" |
| §5.3 Qwen "EVQ" | 方法名错误 | 改为 "anchored_sigmoid" 或注明方法差异 |
| Abstract/Intro 中的 "Llama-3-8B" 引用 | 不应引用已删除实验 | 删除相关提及 |
| τ* ≈ 2.7 的说法 | 只适用于 128-tok regime | 加限定条件 |
| "6 LongBench tasks" 的所有出现 | Qwen 是 21 | 全部修正 |
| Appendix H 中的 LLaMA LoRA 超参 | 对应已删除实验 | 删除或替换 |

---

*本文档在下次更新论文前必须逐条核对。*
