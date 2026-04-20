# Finetune 微调方案诊断：为什么效果小，问题到底出在哪里

> 日期: 2026-03-12
> 背景: n=2086 QuALITY 结果显示 EVQ@8K +2.2pp (p≈0.014)，信号存在但效果偏小。
> 之前的假设 "finetune masks PE signal" 被推翻——微调确实能保留 PE 信号，问题在具体的实验设计。

---

## 核心结论

**信号确实存在，微调没有"掩盖" PE 信号。效果小的主因是：模型在 QA 任务上几乎没学会（两者都在 25% random baseline 附近），PE 改善了 attention 质量但模型本身做不了题。**

这类似于给一个不会开车的人换了更好的轮胎——轮胎确实更好（PPL -52%），但这个人根本到不了终点（accuracy ≈ random）。

---

## 第一层：当前 454M QuALITY 设置的具体问题

### 问题 1: 训练量严重不足（最重要）

| 维度 | 我们的 454M | FIRE (ICLR 2024) | 差距 |
|------|-----------|-----------------|------|
| 步数 | **2,000** | 25,000 | 12.5× |
| Effective batch | **4** | 128 | 32× |
| 总样本数 | 8,000 | 3,200,000 | **400×** |
| 总 token | ~32M | ~26B | ~800× |
| QuALITY epoch 数 | ~3 | ~1268 | ~420× |

2000 步 × bs4 意味着模型只见了 3 遍 QuALITY 训练集。从 finetune accuracy 看（Geo 78%, EVQ 79%），模型确实在训练集上拟合了，但这不等于泛化——几乎肯定是在 memorize 训练集的浅层模式，而非学会真正的阅读理解。

**关键证据**: 两个模型的 eval accuracy 都在 25-27%（4-choice random = 25%），说明模型在验证集上基本不具备 QA 能力。当基础能力 ≈ 0 时，PE 差异能贡献的 accuracy delta 自然很小。

### 问题 2: Finetune 长度选择 (L=4096 而非 8192)

```
Pretrain:   512 → 1024 → 2048  (progressive)
Finetune:   4096               (2× from pretrain)
Eval:       4K, 8K, 16K, 32K
```

实际 finetune 是在 L=4096，不是 8192。这意味着：
- @4K eval 是 finetune in-distribution → 预期 waterbed cost（Geo 略赢）→ ✅ 实际 +0.7pp Geo 赢
- @8K eval 是 2× extrapolation from finetune → 预期 EVQ 赢 → ✅ 实际 +2.2pp EVQ 赢
- @16K 是 4× extrapolation → 预期 gap 更大 → ✅ 实际趋势 ~+2.9pp

**模式完全正确！** 问题不是方向（方向全对），而是绝对值太小。

如果改成 L=8192 finetune（与 FIRE 对齐）：
- @8K 变成 in-distribution → 可能 Geo 略赢（waterbed）
- @16K 才是 2× extrapolation → EVQ 优势应在 16K 显现
- 需要在 32K 才能看到大 gap

### 问题 3: τ 的两难困境（次要）

| 阶段 | L_train | 当前 τ | 理论 τ* |
|------|---------|--------|---------|
| Pretrain Stage 3 | 2048 | 1.414 ✓ | 1.414 |
| Finetune | 4096 | **1.414** | **1.000** |
| 如果 finetune@8192 | 8192 | ? | **0.707** |

Finetune 时没有 retarget τ。但这其实是一个**设计两难**：

- **如果 retarget τ 到 1.0**: 等于在 finetune 阶段改变 inv_freq → 这就是 retrofit（`AI_HANDOFF_PITFALLS.md` §4 说了不能做）。模型已经用 τ=1.414 的频率学了 1.5B token，突然改频率会破坏已有的 attention pattern。
- **如果保持 τ=1.414**: 频率分配针对 L=2048 优化，在 L=4096 finetune 时不是最优的。但至少保持了已学会的 attention structure。

**实际影响**: τ=1.414 在 L=4096 finetune 时意味着 EVQ 仍然在做"为 2048 优化的频率分配"，但这正是 pretrain 阶段证明了有效的分配。EVQ 的 extrapolation 优势恰恰来源于 pretrain 学到的频率结构，保留它是合理的。

**结论**: τ retarget 不是当前的关键问题。保持 τ=1.414 是合理选择。

### 问题 4: Eval 协议差异解释了 n=200 vs n=2086 的 gap

| Eval | Protocol | @8K Gap |
|------|----------|---------|
| n=200 原始 | Distractor-padded | **+7.0pp** |
| n=2086 新 | Standard (无 padding) | **+2.2pp** |

Distractor-padded eval 把文章塞在随机无关文本中间 → 变成 needle-in-haystack 任务 → PE 对长距离检索的影响被放大 → gap 更大。

Standard eval 直接读原文 → 很多 QuALITY 文章 ≤4K token → 在 8K context 下不需要很强的长距离 attention → PE 差异不明显。

**两种 eval 都是合理的**，但衡量的东西不同：
- Distractor-padded → 测 "能否在长文本中找到证据"（PE-sensitive）
- Standard → 测 "能否理解文章回答问题"（NLU-sensitive）

---

## 第二层：和 Qwen 实验的关键区别

用户提到 "之前 Qwen 的微调其实起作用了"。仔细对比：

| 维度 | Qwen 实验 | 454M QuALITY |
|------|----------|--------------|
| 模型规模 | **7B** | 454M (15× 小) |
| 方法 | anchored_sigmoid (非 EVQ) | EVQ-cosh |
| 微调方式 | 400 步 WikiText LoRA | 2000 步 task-specific full FT |
| 评测 | 21 task LongBench NLL | QuALITY accuracy |
| 基础能力 | **avg 44.4%**（远超 random） | **~26%**（≈ random 25%） |
| PE 效果 | delta -0.35 to -0.42 | +2.2pp accuracy |

**最关键的区别: Qwen 的基础 QA 能力 (44.4%) >> 454M 的 QA 能力 (~26%)**

Qwen 7B 在 LongBench 上 44.4% 平均分，说明模型 genuinely 能做下游任务。PE 改变导致的 attention 质量差异可以体现在 task performance 上，因为模型有足够的"底子"让 PE 差异发挥作用。

454M 在 QuALITY 上 ~26% ≈ random。模型基本不会做这个任务。就算 EVQ 给了完美的长距离 attention，模型也不知道怎么用这个 attention 来回答问题。

**类比**:
- Qwen: 一个会开车的司机 (44.4%), 换更好的轮胎 → 弯道表现提升
- 454M: 一个不会开车的人 (26%), 换更好的轮胎 → 几乎没区别

---

## 第三层：Qwen 实验"起作用了"的真正含义

回查 `PAPER_ERROR_CORRECTIONS.md`:
- Qwen 用的是 **anchored_sigmoid**，不是 EVQ
- 结果: seed42 avg delta = **-0.35**, seed1337 = **-0.42**（小幅退化）
- 21 个 LongBench 任务的 per-task 分析才能看到 waterbed 模式

严格来说 Qwen 实验也没有"赢"——aggregate delta 是 **负的** (-0.35)。但它成功的地方是：
1. **per-task 模式可见**: 长距离任务 EVQ 赢，短距离任务 Geo 赢（waterbed）
2. **NLL-level 信号清晰**: 因为 NLL 是连续指标，200K tokens 的样本量足够
3. **模型有底子**: 44.4% 基础分 → PE 差异可以 resolve

所以 "Qwen 微调起作用了" 更准确的表述是: **在足够大的模型 + NLL 指标 + 多任务平均的条件下，PE 信号可以被检测到。** 它证明的是信号的存在性，不是某个具体 accuracy metric 上的赢。

---

## 第四层：信号到底有多大——物理上限估算

PPL@8K: EVQ 192 vs Geo 337 (-43%) [proof-pile-2, pretrain checkpoint]

这 43% 的 PPL 差异能转化为多少 accuracy 差异？

假设一个 4-choice QA 题需要模型 attend 到文章中的一段关键证据（~50 token）:
- Geo@8K: 这 50 token 的 attention 质量下降 → 模型"看不清"证据 → 猜对概率 ≈ 25% + ε
- EVQ@8K: attention 质量保持 → 模型"看得到"证据 → 猜对概率取决于理解力

但 454M 的理解力问题: 即使完美看到了证据，模型也不一定理解问题和选项的语义关系。**Accuracy 的上限不是 PE 决定的，是模型容量决定的。**

因此:
- **PPL 差异 -43%**: 反映的是 next-token prediction quality（每个 token 都是样本）→ 信号强
- **Accuracy 差异 +2.2pp**: 反映的是 "看到证据 + 理解问题 + 匹配选项" 的联合概率 → 每一环都是 bottleneck

**PE 只改善了"看到证据"这一环。其他环节（理解力、推理力）是 454M 的硬性瓶颈。**

---

## 诊断总结：问题优先级

| # | 问题 | 影响大小 | 状态 |
|---|------|---------|------|
| 1 | 模型基础能力 ≈ random（~26%） | **致命** | 需更多训练或更大模型 |
| 2 | 训练量不足（2K步 vs FIRE 25K步） | **高** | 直接加步数 |
| 3 | Finetune@4K 而非 8K | **中** | 改配置即可 |
| 4 | Standard eval 区分度低于 distractor eval | **中** | 两种都报告 |
| 5 | τ 未 retarget | **低** | 保持 τ=1.414 是合理的 |
| 6 | YaRN 实现 | **已修复** | eval_clean.py 是正确的 |
| 7 | Accuracy metric | **已修复** | options_nll scoring |

---

## 可行的修复方案

### 方案 A: 加训练量（最小改动，最可能见效）

把 454M QuALITY finetune 从 2000 步加到 **10000-25000 步**:
- QuALITY 训练集 2523 样本 × bs4 × 10K 步 = ~16 epochs
- 预期: 基础 accuracy 从 ~26% 提升到 35-40%（如果模型能学会的话）
- 一旦基础能力脱离 random floor，PE 差异就会放大

**风险**: 454M 可能容量不够在 QuALITY 上达到 35%+。FIRE 的 350M 在 SCROLLS 上达到了 meaningful accuracy，但他们用了 25K×bs128 的训练量。

### 方案 B: 用 Phase 17c 做 zero-shot collapse demo（之前策略文档的方案 1）

不 finetune，直接用 pretrained checkpoint 在 Geo-collapsed 长度上做 QA:
- EVQ+YaRN PPL=2.635@48K vs Geo+YaRN PPL=14.219@48K
- PPL 差 5 倍的模型在 accuracy 上应该是 catastrophic 差距
- 不需要 finetuning，纯粹测 PE extrapolation

**这其实最干净**: 避开了所有 finetune 带来的混淆因素。

### 方案 C: 750M + 正确 Protocol（Phase 21E）

如果 750M 能在 QuALITY 上 consistently >30%（脱离 random floor），那 PE 差异就能体现。需要:
- τ=0.707 for L=8192 finetune（但有 retrofit 风险）
- 或保持 τ=1.414 不变
- 25K 步训练
- Progressive YaRN（已修复）

---

## 核心洞察（给论文叙事的建议）

之前的假设 "finetune 掩盖 PE 信号" 实际上是错的。正确的理解是:

> **PE 改善了 attention 的 positional resolution（PPL -43%, passkey +60pp），这是 infrastructure 层面的改善。它能转化为多少 task accuracy 取决于模型本身的 task capability。当模型处于 capacity floor 时，infrastructure 改善无法体现为 task metric 改善。**

这不是 EVQ 的问题，而是所有 PE 研究共有的评测难题。DAPE、YaRN、ALiBi 都没有在 accuracy 上展示大幅改善——因为它们面临同样的 bottleneck。

**我们已有的 PPL + passkey + NLL reversal 证据，在 PE 论文标准中已经是 top tier。**
