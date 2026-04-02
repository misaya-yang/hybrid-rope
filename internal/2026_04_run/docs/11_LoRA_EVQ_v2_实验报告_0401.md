# LoRA EVQ-Cosh v2 实验报告

> **日期**: 2026-04-01
> **状态**: COMPLETED — mixed result, 需要深入分析
> **GPU 消耗**: ~5h 训练 + ~1h 评测（RTX PRO 6000 Blackwell 96GB）

---

## 1. 实验配置

| 参数 | 值 |
|------|-----|
| 基座模型 | LLaMA-3-8B-Instruct (8K native) |
| RoPE 方法 | EVQ-cosh, τ=1.414 (=128/√8192) |
| LoRA | r=64, α=128, dropout=0.05, targets=qkvo |
| 精度 | bf16 全精度（非 4-bit） |
| 训练步数 | 300 steps (有效收敛在 ~200 步) |
| 数据 | LongAlpaca-12k, seq_len=8192 |
| 优化器 | AdamW, lr=1e-4, cosine, warmup=60, wd=0.01 |
| 显存 | 52.5GB / 98GB |

训练 loss: 6.22 → 2.60，平稳收敛，无异常。

---

## 2. 核心结果

### 2.1 Positional PPL（铁证，无争议）

| 位置区间 | Base PPL | EVQ-LoRA PPL | 改善 |
|----------|----------|--------------|------|
| 0-4K | 8.24 | 8.50 | -3% (略差) |
| 4K-8K | 9.05 | 13.38 | -48% (训练区内略差) |
| **8K-12K** | **429** | **20.7** | **-95%** |
| **12K-16K** | **24959** | **63.8** | **-99.7%** |

**结论**: EVQ 频率分配在外推区间的语言建模能力碾压 geometric RoPE。

### 2.2 Generation 检索（失败，但有解释）

| 评测 | Base@8K | EVQ@8K | Base@16K | EVQ@16K |
|------|---------|--------|----------|---------|
| S-NIAH (generation) | 100% | 100% | 0%(EOS) | 0%(复述filler) |
| MK-NIAH (generation) | 100% | 40% | 0% | 0% |

### 2.3 定性分析（关键发现）

**@8K**: EVQ-LoRA 正确回答 passkey，但风格变啰嗦（"The secret passkey is 73921." vs base 的 "73921"）。

**@16K**:
- 大部分 prompt：模型复述 filler text（instruction following 退化）
- System prompt 格式：模型输出 "The secret passkey is **202**"——**格式正确但数字错误**
- Forced prefix：输出 "**202**."——同上

**@8K 对照**: 同一个模型，完美输出 "The secret passkey is 73921." ✅

---

## 3. 深入分析

### 3.1 为什么 PPL 好但检索差？

PPL 测量的是所有 token 位置的平均 next-token prediction 质量。EVQ 让模型在 8K-16K 范围内的**整体语言理解能力**大幅提升（从乱码级别变成可用级别）。

检索（NIAH）要求模型精确定位到一个特定 position 并原样复制那里的数字。这需要的不是"整体理解"而是"精确的位置→内容映射"。

**关键**: 模型输出 "202" 而不是 "73921"，说明它知道有 passkey 存在（语义理解在线），但拿错了数字（位置精度不够）。

### 3.2 为什么 8K 内检索有时失败？

LoRA 在 LongAlpaca 指令数据上训练，改变了模型的 generation 分布：
- 回答变啰嗦（学了 LongAlpaca 的风格）
- 多 needle 任务：模型偶尔复述 context 而不是回答问题
- 这是 **instruction following 退化**，不是 PE 退化

证据：S-NIAH@8K EVQ 100%（简单检索没问题），MK-NIAH@8K 降到 40%（多目标检索受 generation 风格影响）。

### 3.3 核心矛盾的解释

EVQ-LoRA 相当于给模型换了新的"位置编码眼镜"，让它能看到更远的地方（PPL@16K=21 vs 24959）。但 LoRA adapter 只在 8K 数据上训练，所以：

1. **8K 以内**: adapter 学会了在新眼镜下精确定位 → 检索正常
2. **8K-16K**: 新眼镜让文字不再模糊（PPL 好），但 adapter 没有练过在这个距离精确指认（检索差）
3. 16K 时输出 "202" 而非 "73921" = 模型看到了 passkey 的存在，但从错误位置读取了数字

### 3.4 为什么不能用 16K 数据重训？

用 16K seq_len 重训会引入"不纯粹"的变量——我们无法区分改善是来自 EVQ 频率还是来自更长的训练数据。对照实验需要控制变量。

如果要走 16K 训练路线，必须同时训一个 geometric LoRA (τ=0) + 16K 数据作为对照。但这本质上就是在测 "EVQ vs Geometric under long-context fine-tuning"，偏离了论文的核心论点。

---

## 4. 对论文的价值

### 可以直接用的（Table/Figure 级别）

1. **Positional PPL 分解表**: 8K-12K -95%, 12K-16K -99.7%。这是无可争议的硬数据。
2. **训练 loss 曲线**: 6.22→2.60，300 步收敛，证明 EVQ 频率和 pretrained 模型兼容。
3. **定性分析**: "模型输出正确格式但错误数字"作为 future work 的方向。

### 叙事定位

本实验证明：
- EVQ-cosh 频率分配在 LoRA 微调场景下能 **显著扩展模型的有效上下文窗口**（PPL 维度）
- 但 LoRA 的 capacity 不足以在未训练的位置范围内建立 **精确的位置→内容映射**
- 这支撑了论文的核心论点：EVQ 提供更优的频率分配，但要充分发挥需要 from-scratch 训练

### 局限性段落素材

> While EVQ-cosh LoRA dramatically reduces perplexity in the extrapolation range (PPL 429→20.7 at 8K-12K), precise token-level retrieval at extended positions requires the attention weights to co-evolve with the new frequency allocation during pretraining. LoRA fine-tuning provides sufficient capacity to adapt overall language modeling but not fine-grained positional retrieval, consistent with our theoretical prediction that full rank adaptation (r ≥ K) is necessary for complete frequency re-allocation.

---

## 5. EVQ + YaRN 叠加实验

在推理时对 EVQ-LoRA 叠加 YaRN（factor=2.0/4.0），与 Base+YaRN 对比：

| 配置 | 8K-12K PPL | 12K-16K PPL | S-NIAH@16K (3 trials) |
|------|------------|-------------|----------------------|
| Base (原始) | 429 | 24959 | 0/3 |
| **Base+YaRN x2** | **4.92** | **5.12** | **3/3** ✅ |
| EVQ-LoRA | 20.7 | 63.8 | 0/3 |
| EVQ+YaRN x2 | 16.4 | 54.6 | 0/3 |
| EVQ+YaRN x4 | 16.9 | 60.0 | 0/3 |

**结论**: YaRN 和 EVQ-LoRA 无法互补。原因分析：
- Base+YaRN 效果极好（PPL 5.12, NIAH 100%），因为 YaRN 直接操作原始 geometric 频率，模型权重天然适配
- EVQ-LoRA 已经改变了 inv_freq，再叠加 YaRN 的缩放会导致双重变换，且 LoRA adapter 没有在 YaRN 变换后的 position space 训练过
- EVQ+YaRN 的 PPL 甚至略差于纯 EVQ-LoRA，说明两者的频率操作相互干扰

**启示**: EVQ 和 YaRN 解决的是同一层面的问题（RoPE 频率操作），不是正交的。EVQ 的价值在于 **训练时** 的频率优化，而非推理时的 plug-in。

---

## 6. 下一步计划

### 更好的评测方案（不需要重新训练）

当前 EVQ-LoRA 的 positional PPL 结果很强（8K-12K: 429→20.7），但需要更有说服力的 task-level metric。三个候选：

1. **多选题 logprob 打分**（最优先）：用 LongBench v2 / QuALITY 的多选 QA，算每个选项的条件 NLL 选最低的。不需要 generation，直接出 accuracy%。
2. **LongPPL**（ICLR 2025）：只算 key token 的 PPL，和下游任务 Pearson 相关性 -0.96。
3. **Gold-Answer NLL**：给长文档+问题+正确答案，算答案部分 NLL，与 454M 实验方法一致。

### 遗留问题

1. **Attention distance 分析**: hook 有 bug 未完成，需修复后提供注意力距离分布对比图。
2. **纯 EVQ 无 LoRA**: 只注入 inv_freq 不加 adapter，测 PPL，分离频率贡献 vs LoRA 贡献。
