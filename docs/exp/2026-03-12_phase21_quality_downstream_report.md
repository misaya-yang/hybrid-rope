# Phase 21 QuALITY Downstream Report

> 日期: 2026-03-12
> 状态: **INDEPENDENT REPORT**
> 范围: 仅记录本轮 QuALITY downstream 实验、清洗、结果与问题定位
> 结论级别: **pilot / diagnostic**, 不是 paper-ready primary anchor

---

## 0. 执行摘要

这轮 QuALITY downstream 的主要结论不是“EVQ 已经在真实任务上稳定胜出”，也不是“之前的 PPL / passkey 结论被推翻”。

更准确的判断是：

1. **QuALITY 作为任务方向是对的**：它是长文档 QA，方向上符合我们希望测的长距离证据定位能力。
2. **当前 pipeline 不够干净**：训练目标、评测目标、checkpoint 血统、统计功效都存在问题。
3. **旧版 accuracy 结果不可用**：最早那批 QuALITY accuracy 包含多个实现层面的错误与协议错位。
4. **在对齐后的 `gold_answer_nll` 指标下，确实能看到部分长程结构性信号**，但曲线并不稳定，不能直接当论文主结论。

因此，这轮实验更适合作为：

- 一个 **protocol stress-test**
- 一个 **bug / design audit**
- 一个 **决定下一轮如何重做 clean downstream anchor 的依据**

不适合作为：

- “EVQ downstream 已被验证”的主结论
- 或 “EVQ downstream 不成立”的反结论

---

## 1. 目标与背景

本轮实验的目标，是验证此前在更受控设置中观察到的 EVQ 长程优势，能否迁移到真实 downstream QA 任务。

已有 strongest evidence 来自：

- raw text extrapolation PPL
- passkey / AR exact
- EVQ + YaRN 相比 Geo + YaRN 的长程优势

QuALITY 被选为主任务，是因为它属于长文档 QA，理论上比 GovReport 这类 summarization 更接近 EVQ 的优势区间。

---

## 2. 本轮实际对象

### 2.1 Checkpoint

本轮使用的是 `750M` Phase 15 系列 checkpoint：

- Geo checkpoint
- EVQ `tau=1.5, r=0` checkpoint

需要强调的是，这不是一组“从头纯 Geo vs 从头纯 EVQ”的完全干净预训练对照。
Phase 15 文档记录的初始化是：

- init ckpt: `phase9 geo_750m_2k_1bdata_ckpt step_15258.pt`
- continue: `2048 -> 4096`

因此这组 750M EVQ 模型本质上是 **Geo 2K checkpoint 上的 EVQ retrofit / continue**，不是最终论文里最理想的 clean anchor。

### 2.2 Finetune 范式

当前 QuALITY finetune 采用的是 free-form causal LM 方式：

- 输入：`Read the following and answer the question.\n\n{raw_input}\n\nAnswer:`
- 目标：真实答案文本（不是固定的 `A/B/C/D` 标签）

这意味着：

- 模型学到的是 free-form answer generation
- 不是标准 multiple-choice classifier

### 2.3 样本规模

本轮 pilot 实际配置：

- train samples: `2523`
- validation samples: `2086`
- finetune pilot: `2000 steps`
- 单 seed
- 评测通常只取 `200` 个 validation 样本

这是一个方向性探测设置，不是 high-power anchor。

---

## 3. 发现的关键问题

### 3.1 旧版 QuALITY accuracy 指标本身无效

训练脚本最早对 QuALITY 的“accuracy”是这样算的：

- 生成最多 16 个 token
- 只看生成结果的 **首字符** 是否等于 gold answer 的首字符

这对多选题是坏指标。

我对官方 validation 集重算后得到：

- `2086` 个样本里，`83.84%` 的题目四个选项首字符不唯一
- `73.15%` 的题目四个选项连首个词都不唯一

所以此前类似：

- `geo 80.5%`
- `evq 82.0%`

这样的“finetune accuracy”，不能作为可信结论。

### 3.2 旧版长长度 eval 曾经存在真实代码 bug

在远端旧版脚本里，generation eval 存在硬编码：

- context 超过 `8192` 时只保留最后 `8192`

这会让 `16K+` 的 QA 输入在 generation 时直接丢掉前面的长文档内容。
因此最早那批极低 accuracy 不能直接解释为模型真实能力。

### 3.3 旧版所谓 YaRN 实现其实是 NTK-aware

远端旧版 QuALITY 脚本里，所谓 `apply_yarn_scaling()` 实际是“全通道同因子缩放”，属于 NTK-aware 风格，不是真正的 progressive YaRN。

这对 EVQ 特别不利，因为它会覆盖 EVQ 原本的频率分配结构。

本轮后续 clean eval 已修成真正的 per-channel progressive YaRN。

### 3.4 EVQ 的 `tau` 没有按 `8K` finetune stage 重新 retarget

这是一个更具体、也更容易被忽略的设计不对称。

当前 `750M` downstream finetune 用的不是一个“随 stage 更新的 EVQ 频率分配”，而是直接沿用了已有 checkpoint 的固定 `tau`。

本轮使用的 Phase 15 文档写的是：

- `EVQ tau=1.5, r=0`

而按我们自己在 progressive protocol 中写过的 scaling：

- `L=2048 -> tau*=64/sqrt(2048)=1.414`
- `L=4096 -> tau*=64/sqrt(4096)=1.0`
- `L=8192 -> tau*=64/sqrt(8192)=0.707`

也就是说，如果把 `8K finetune` 看成一个新的训练 stage，那么当前 EVQ 其实是带着一个 **对更短训练长度更激进的 warp** 去适应 `8K` 任务数据。

这对 Geo 不构成同类问题，因为：

- Geo 永远是 `tau=0`

所以当前实验存在一个真实的 asymmetric burden：

- Geo: 没有 length-dependent hyperparameter mismatch
- EVQ: 有

这很可能解释了为什么我们看到：

- `8K/12K/16K` 的 in-dist / near-extrap 区间整体偏 Geo
- 但更远处 `24K/32K/40K` EVQ 又能重新冒出优势

换句话说，这轮实验更像是在测：

- **fixed-tau EVQ carried into 8K SFT**

而不是：

- **stage-aware retargeted EVQ at 8K**

因此，当前 downstream negative signal 不能直接被读成“按我们自己的 length-aware EVQ rule，工程上也不行”；它更准确地是在说：

- **固定旧 `tau` 直接拿去做 8K SFT，没有证明工程收益**

### 3.5 训练目标与评测目标错位

这是本轮最大的结构性问题。

训练时学的是：

- `raw_input -> free-form answer`

但最开始拿来评估的却是：

- 多选 accuracy
- 甚至是 `4 option NLL`

这两者不是同一个任务。

这也是为什么会出现下面这种明显不合理的现象：

- 旧版 finetune “accuracy” 看起来接近 `80%`
- 但 `8K in-dist` 的 option-NLL accuracy 却只有 `~27%`

这个差异不是模型突然不会做题，而是 **scorer 和训练目标根本没对齐**。

### 3.6 原来的 “clean @8K” 其实也不是 in-distribution baseline

在 `article_pad_extrap` 协议里，evaluator 会：

- 保留 prompt 模板
- 但把 article 区域自动填满到目标长度

对 QuALITY validation 前 `200` 个样本，本地重算得到：

- `8K` 时 `200/200` 样本都会被 padding
- median article budget: `8113`
- median article length: `6900`
- median distractor budget: `1203`

所以之前那个 “clean @8K” 其实仍然是 stress test，不是 in-dist baseline。

### 3.7 Finetune 数据本身存在中间截断

当前 finetune 预处理对所有超过 `seq_len` 的样本采用 middle truncation：

- 保留前半段
- 保留后半段
- 中间内容被丢弃

我按当前 QuALITY prompt 模板重算了 train split：

- total train samples: `2523`
- `>8192` 的样本: `254`
- truncation fraction: `10.07%`

也就是说，大约十分之一的训练样本在 finetune 时就会丢失中间证据。
对需要精确证据定位的 QA，这会直接增加训练噪声。

### 3.8 统计功效不足

当前 pilot 的规模太小：

- 单 seed
- `2000` steps
- `200` eval samples

这会导致：

- accuracy 方差大
- NLL 也容易受少数 hard examples 影响
- 长度曲线不够平滑

因此，这轮实验最多能做方向性判断，不能做强结论。

---

## 4. 本轮清洗后的评测协议

为避免继续混淆，本轮最终保留两种 clean protocol：

### 4.1 `in_dist_nopad`

- 只用于真正的 `8K` in-distribution baseline
- 不插 distractor
- 保持训练 prompt 范式

### 4.2 `article_pad_extrap`

- 用于 `12K+` 外推
- 只在 article 区域插入 distractor
- 保留训练 prompt 模板，不重排 question / options / answer 逻辑

### 4.3 最终主 scorer

对于当前 free-form finetune checkpoint，最终采用：

- `gold_answer_nll`

即 teacher-forced 计算真实答案文本的条件 NLL。

这里：

- `NLL` 越低越好

这样至少保证了训练目标和评测目标一致。

---

## 5. 结果

## 5.1 真正的 `8K` in-distribution baseline

`protocol = in_dist_nopad`
`scoring = gold_answer_nll`

| Length | Geo | EVQ |
|--------|-----|-----|
| 8K in-dist | `0.4903` | `0.5861` |

结论：

- 在当前训练范式和当前 checkpoint 上，`8K` 训练内是 `Geo` 更好
- 因此不存在“8K 已经明显 EVQ 胜”的证据

---

## 5.2 Clean padded extrapolation: 12K / 16K / 24K

`protocol = article_pad_extrap`
`scoring = gold_answer_nll`

| Length | Geo | EVQ | Winner |
|--------|-----|-----|--------|
| 12K | `0.7800` | `0.8638` | Geo |
| 16K | `1.6550` | `1.9537` | Geo |
| 24K | `4.8131` | `4.1510` | EVQ |

结论：

- `12K/16K`：Geo 更好
- `24K`：EVQ 反超

这说明如果 EVQ 优势存在，它在这条 pipeline 下不是在近外推区间出现，而是更晚才显现。

---

## 5.3 极限长度 raw vs +YaRN: 32K / 40K / 48K

### Raw

| Length | Geo raw | EVQ raw | Winner |
|--------|---------|---------|--------|
| 32K | `4.7718` | `3.2310` | EVQ |
| 40K | `6.6160` | `6.0479` | EVQ |
| 48K | `6.6547` | `7.4477` | Geo |

### +YaRN

| Length | Geo + YaRN | EVQ + YaRN | Winner |
|--------|------------|------------|--------|
| 32K | `0.7180` | `0.6607` | EVQ |
| 40K | `0.9351` | `0.6880` | EVQ |
| 48K | `0.7661` | `0.8021` | Geo |

结论：

1. `+YaRN` 对两边都非常有用，NLL 下降一个量级。
2. `32K / 40K` 上，无论 raw 还是 +YaRN，都是 EVQ 更好。
3. `48K` 上出现翻转，Geo 重新更好。

这说明：

- 当前 downstream 曲线并不是“单调 EVQ 胜”
- 也不是“Geo 全程胜”
- 而是一个 **不稳定、会翻转的 crossover pattern**

---

## 6. 如何解读

### 6.1 这不推翻之前的 PPL / passkey

此前 strongest evidence 的性质是：

- 更受控
- 更 PE-dominant
- 更少掺入 answer calibration / generation style / task format 误差

而这轮 QuALITY downstream 测的是复合能力：

- 长距检索
- 题目理解
- 答案生成
- 输出校准
- 任务格式适配

因此它不能直接反向推翻前面那些更干净的证据。

### 6.2 这也不能支持“EVQ downstream 已被验证”

虽然在对齐后的 `gold_answer_nll` 指标下，EVQ 确实在部分长度赢了：

- `24K`
- `32K`
- `40K`

但整条曲线并不稳定：

- `8K/12K/16K`: Geo 更好
- `24K/32K/40K`: EVQ 更好
- `48K`: Geo 更好

因此不能把这轮写成稳定、单调、可复现的 downstream anchor。

另一个重要 caveat 是：

- 当前这条曲线测到的还是 **fixed-tau EVQ**
- 不是按 `tau*(L_train)` 在 `8K` stage 重新 retarget 过的 EVQ

因此它对“EVQ 作为 length-aware rule 是否在工程上有效”的判定，仍然是不完整的。

### 6.3 当前最合理的定位

这轮实验最适合被定位为：

- QuALITY downstream pilot
- protocol / implementation audit
- downstream transfer is highly protocol-sensitive 的证据

而不是最终 headline result。

---

## 7. 我对“哪里出了问题”的判断

如果只选一个最核心的问题，我会选：

**当前 QuALITY pipeline 同时犯了两类最伤解释性的错：**

- 把 free-form generation finetune，当成了 multiple-choice calibrated evaluation 来读
- 在 `8K` finetune stage 没有按 length-aware rule 重新 retarget `tau`

这两点叠在一起，基本足以把下游结论读歪。

如果展开成问题树，排序如下：

1. **训练目标 vs 评测目标错位**
2. **EVQ `tau` 没有按 `8K` finetune stage 重新 retarget**
3. **旧版 QuALITY accuracy 指标无效**
4. **旧版长长度 eval 存在真实 bug**
5. **8K baseline 最初并不干净**
6. **750M checkpoint 血统不够 clean**
7. **训练样本有 10% 中间截断**
8. **pilot 统计功效不足**

也就是说，当前问题不是“QuALITY 这个任务选错了”，而是：

**任务方向是对的，但实现和协议没有 lock 到前面那些 strongest evidence。**

---

## 8. 这轮实验现在能支持什么说法

### 可以支持

- QuALITY downstream 对 protocol 非常敏感
- 旧版 accuracy 结果不可信，必须丢弃
- 用训练对齐的 `gold_answer_nll` 后，确实能恢复出部分长度相关结构
- `+YaRN` 对两边都非常有帮助
- EVQ 在部分长长度区间存在优势，但不是稳定单调优势

### 不能支持

- “EVQ downstream 已被 clean 证明”
- “之前所有 PPL / passkey 结论都被推翻”
- “Geo definitively 更好”
- “EVQ definitively 更好”

---

## 9. 建议的后续动作

如果要把 QuALITY 变成可用于论文的 clean downstream anchor，下一轮至少要补：

1. **clean checkpoint pair**
   - 用更干净的 Geo vs EVQ 对照，不要再依赖当前这组 retrofit 750M

2. **protocol-locked finetune / eval**
   - 要么训练和评测都走 free-form + gold-answer NLL
   - 要么重训成 option-letter supervision，然后统一用 logits / option NLL

3. **true 8K baseline**
   - 固定保留 `in_dist_nopad`

4. **多 seed + 更大 eval**
   - 至少 `3 seeds`
   - eval 样本数至少扩大到 `1000+`

5. **paired analysis**
   - 保留每题 per-example NLL
   - 做 paired bootstrap / confidence interval

---

## 10. Bottom Line

本轮 QuALITY downstream 的最大价值，不是给出一个漂亮 headline，而是把问题暴露清楚了：

- 旧结果里哪些不能用
- 哪些实现会系统性污染结论
- 当前 750M pilot 为什么和此前 PPL / passkey story 对不上

这轮实验告诉我们的不是“EVQ 不行”，而是：

**如果要做 downstream anchor，必须先把 protocol 锁死；否则任务噪声、scorer 错位和 checkpoint impurity 会把真正的 PE 信号洗掉。**
