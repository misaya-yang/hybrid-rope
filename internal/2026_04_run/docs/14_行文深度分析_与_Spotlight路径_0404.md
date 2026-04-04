# EVQ-Cosh 行文深度分析 + Spotlight 路径规划

> **日期**: 2026-04-04
> **对比材料**: mainstory.md v25 vs paper/ 全部 tex vs 六大理论问题 tex
> **目标**: (1) 行文优化空间 (2) Spotlight 所需的 LoRA 加强 + 理论问题解决路径

---

## 第一部分：行文分析——Paper vs Mainstory 的差距

### 一、Mainstory 中有但 Paper 中缺失或弱化的关键素材

这是最直接的提分空间——数据已有，只需要写进去或写得更好。

#### 1.1 MLA YaRN FT 13.6pp Structural Reversal（最大遗漏）

**Mainstory §6.7** 详细记录了 Phase 18 的核心发现：EVQ raw 在 fully-trained (1B tokens) 下输 11.1%，但 EVQ+YaRN+FT 赢 2.5%，产生 13.6pp structural reversal。这是整篇论文**回应 "undertraining objection" 的终极武器**。

**Paper 现状**: §5 完全没有这个数据。Appendix MLA 段落只覆盖了 standalone + YaRN inference 结果。

**行文建议**: 这个数据的叙事威力在于——即使你承认 EVQ standalone 在充分训练后优势减弱，composition advantage 依然是 structural 的。这直接把 reviewer 从 "EVQ might be an undertraining artifact" 的攻击路径上封死。

建议在 §5 MLA paragraph 末尾加 2-3 句，或在正文某处加一个 mini table：

```
| Training | EVQ vs GEO raw@2× | EVQ vs GEO +YaRN+FT@target | Swing |
|----------|-------------------|----------------------------|-------|
| 500M tokens | -31.1% EVQ wins | -39.7% EVQ wins | +8.6pp |
| 1B tokens  | +11.1% EVQ loses | -2.5% EVQ wins  | 13.6pp |
```

#### 1.2 "Four-Layer Signal Gradient" 叙事

Mainstory §Claim 5 给出了一个极其清晰的叙事框架：

> PPL −52% → Gold NLL −30% → passkey +60pp → accuracy +2.2pp

这是 "PE 改进如何逐层传导到下游" 的完美故事。Paper 目前有 QuALITY 数据（Gold NLL −30%@8K），有 passkey 数据（100% vs 61%），但**没有把它们串成一条叙事链**。审稿人需要自己把 Table 1、Table 2、Table 7、Appendix QuALITY 表格的数字拼起来。

**行文建议**: 在 §5 最后一段（或 Conclusion 前）加一段 synthesis，把信号梯度显式写出来。这不需要新数据，只需要一句 connecting sentence：

> "The signal gradient across evaluation layers — PPL $-52\%$ (raw next-token) $\to$ Gold NLL $-30\%$ (task-specific answer confidence) $\to$ passkey $+60$pp (exact retrieval) $\to$ accuracy $+2.2$pp (argmax at capacity floor) — is consistent with the expected behavior of an infrastructure-level improvement."

#### 1.3 Phase 17b 发现：EVQ raw 超越 EVQ+YaRN

Mainstory §5.4 记录了一个被 paper 忽略的关键发现：progressive training 后，EVQ raw PPL@16K (11.2) 竟然低于 EVQ+YaRN (16.8)——模型已经**内化了 EVQ 的分配优势**，使得 inference-time scaling 变得多余。

> Training-inference equivalence: evq_512+YaRN ≈ evq_1024_cont raw (@16K: 11.6 vs 11.2)

这个发现的 paper impact 非常大：它意味着 EVQ + progressive training 可以**完全替代 YaRN**，节省推理成本。目前 paper progressive training 段落只提了 PPL 数字，没有点出这个 implication。

**行文建议**: 在 progressive training paragraph 加一句：

> "After two stages of progressive extension, EVQ raw surpasses Geo+YaRN (PPL 11.2 vs 16.8 at 16K), demonstrating that progressive training can internalize the allocation benefit, potentially eliminating inference-time scaling overhead entirely."

#### 1.4 Reviewer Defense Table

Mainstory §9 有一个极其完整的 reviewer defense table（14 个预期攻击 + 预备回应）。Paper 没有等价物。虽然这不直接进论文，但它应该变成 **rebuttal playbook** 或至少影响 paper 的 preemptive defense writing。

**已有但可以更好利用的 defense points**:

- "Only base=500K" → Phase 18 数据（两个 base 下 EVQ PPL 几乎相同：192.4 ≈ 191.9）目前只在正文一句带过。这个 "EVQ partially compensates for suboptimal base choices" 的发现值得一个 explicit sentence。
- "Models too small" → Mainstory 明确指出这是 PE 文献中最大的 from-scratch 研究（5 scales, 50M-750M），DAPE 只有 125M。**Paper 目前没有显式做这个对比**。建议在 Setup 或 Related Work 加一句。

### 二、Paper 当前行文中可以"化弱为强"的点

#### 2.1 τ* 的 Semi-Analytic Nature → "Even a first-order approximation crushes the default"

当前 paper §3.7 的措辞是：

> "The scaling law is therefore a semi-analytic result rather than a parameter-free theorem."

这是一个**防御性表述**。完全准确，但给审稿人提供了攻击表面。

**化弱为强的改法**: 保持 honest disclosure（不要删这句），但在它之后立刻接一个 **offensive statement**:

> "Nevertheless, the semi-analytic formula $\tau^* = d_{\mathrm{head}}/\sqrt{L}$ falls within a shallow optimality basin ($< 1\%$ PPL gap across 27 configurations; Figure X). The practical implication is that even a first-order variational approximation, calibrated by a single O(1) constant, produces improvements of 14–32% over the 60-year-old geometric default — suggesting that the frequency allocation axis contains far more optimization headroom than previously recognized."

这把 "semi-analytic" 从弱点变成了 story 的一部分：如果一个近似解就能赢这么多，说明这个 axis 的潜力有多大。

#### 2.2 Waterbed Inequality → 把 ≤1% in-range cost 写成 feature

当前 paper 在 Table 1 和 §5 讨论了 waterbed cost（≤1% in-range），但语气是 neutral 的。

Mainstory §4.4 给了一个更好的 framing：

> "The high-frequency band has large redundancy (adjacent channels encode nearly identical short-distance information), so compression is nearly free."

Paper 可以更 explicit 地说：waterbed cost is **asymmetrically bounded** by high-frequency redundancy。这不是 "EVQ 付出了代价"，而是 "EVQ 发现了 geometric RoPE 的浪费并回收了它"。

#### 2.3 MLA 结果在 Abstract 中缺失

当前 Abstract 列了 EVQ+YaRN (100% vs 61%)、PE-dominant (333.7 vs 455.3)、MLA (-31.1%)——等等，检查一下... 实际上 Abstract 最后一句已经包含了 MLA：

> "On Multi-head Latent Attention (MLA, DeepSeek-V2/V3), where only 16 frequency channels carry positional information, EVQ alone outperforms Geo+YaRN at 2× extrapolation (−31.1% PPL, 3 seeds)."

好，这个已经有了。但 **"EVQ alone outperforms Geo+YaRN"** 这个 punchline 在正文中的呈现不够突出。在 Intro §1 的 MLA paragraph 里说了 "EVQ alone reduces 2× extrapolation PPL by 31.1% and outperforms GEO+YaRN at 16K"，但这个 "a single training-time parameter change beats the best inference-time method" 的 framing 太重要了，值得加粗或给一个 standalone sentence。

### 三、Paper 行文中的具体风格/结构问题

#### 3.1 §3 Theory 的层级标注不够

Paper 当前 §3 从 §3.1 到 §3.8 是一个平铺结构。但理论内部有三个清晰的 **epistemic level**：

1. **Exact**：Variational formulation → surrogate → ODE → closed-form (Theorem 1, 2)
2. **Exact conditional on surrogate**：Waterbed inequality (Proposition)
3. **Semi-analytic**：τ* scaling law (Proposition 3)

Paper 目前没有显式标注这三层。审稿人从 Theorem 1 读到 Proposition 3 时，如果没有注意到 "semi-analytic" 这个词，会以为全部都是 exact 的，然后在 Limitations 里发现 λ 是 calibrated 的，产生 disappointment。

**建议**: 在 §3.7 开头加一个 framing sentence：

> "The preceding results are exact conditional on the broadband surrogate. The remaining step — selecting the optimal $\tau$ for a given $(d_{\mathrm{head}}, L)$ — involves an additional approximation layer."

#### 3.2 §5 Experiments 的 headline result 不够突出

§5 开头是 "All core experiments are from-scratch and keep the model architecture fixed." 这是 setup 信息，不是 punchline。NeurIPS 的优秀实验部分通常在第一句就亮出最强结果。

**建议**: 把 §5.2 的 headline 提到 §5 开头第一段：

> "The paper's central systems result is that EVQ unlocks the full potential of inference-time scaling: EVQ+YaRN reaches 100% retrieval at 8K (6 seeds) where Geo+YaRN plateaus at 61%, with PPL improvements of 14–32% at 8–16K (Table 2, Figure 2)."

然后再说 setup details。让审稿人在第一句就知道 "这个实验部分的 headline 是什么"。

#### 3.3 Conclusion 太短

当前 Conclusion 只有 3 行。对于一个有 theory + experiments + MLA + progressive training + LoRA 的论文，3 行太短了。

Mainstory §11 (Impact) 有很好的素材但太长。建议从中提取 2 句加入 Conclusion：

> "For MLA architectures now compressing RoPE to 16 frequency channels, principled allocation transitions from a refinement to a prerequisite — each channel's placement directly determines long-context capability. The variational framework opens the allocation axis to systematic optimization, with EVQ-cosh providing the first closed-form entry point."

---

## 第二部分：Spotlight 路径——LoRA 加强 + 六大理论问题

### 从 Weak Accept 到 Spotlight 需要什么？

NeurIPS Spotlight 的标准大致是：**strong accept from at least one reviewer + no reject**。这意味着论文需要让至少一个审稿人觉得 "this is a significant contribution that I'm excited about"，同时没有审稿人觉得 "this has a fatal flaw"。

当前论文的 Weak Accept 定位是因为：theory is elegant + experiments are solid，但没有一个 "wow factor" 让审稿人兴奋到给 8。

**两个 Spotlight 级别的 upgrade path**：

### Path A：LoRA 实验做成完整的 Cross-Method Benchmark

目前 LoRA 是一个 "mixed result"（PPL 很好，generation 不行）。如果你的 3-method baseline 对照实验（4月3日计划）完成，且 EVQ ≥ YaRN in LoRA setting，这会变成一个非常强的 story：

> "EVQ-cosh provides the best frequency allocation for long-context adaptation, whether from scratch (MHA/MLA, 5 scales) or via LoRA (LLaMA-3-8B). It matches or outperforms search-based methods (YaRN) and learning-based methods (DAPE) with zero additional parameters and zero hyperparameter tuning."

**具体需要**:

1. **Geo vs EVQ vs YaRN LoRA 3-seed 对比表**（正在进行）
   - 如果 EVQ PPL@8K-16K ≥ YaRN → 论文叙事极强
   - 如果 EVQ < YaRN → 仍有价值（closed-form vs search），但叙事需调整

2. **Positional PPL 分区间数据更新 Table 7**
   - v2 的 0-4K/4K-8K/8K-12K/12K-16K 分解比当前粗粒度表更有说服力

3. **Gold-Answer NLL 评测**（不需要 generation，纯 logprob）
   - 这是解决 "generation 失败" 问题的 workaround——在 logprob 空间展示 EVQ 优势

**对 Spotlight 的贡献**: 让论文从 "from-scratch only" 扩展到 "from-scratch + post-hoc"，大幅拓宽 practical applicability。

### Path B：六大理论问题的针对性解决

六大问题的 Spotlight 优先级排序：

| 问题 | Spotlight 价值 | 难度 | 建议 |
|------|-------------|------|------|
| Q1: λ 闭合 | ★★★★★ | 极高 | **不追求完全闭合**，改为写入代理自洽性定理 + λ CV table + "transport constant" 重新定位 |
| Q2: 代理有效性边界 | ★★★ | 中 | 12-config functional validation 已在论文中。补一个显式的 R² 条件表 |
| Q3: DiT 0.53× | ★★ | 高 | 保留为 acknowledged limitation + future work |
| Q4: LoRA 相变 | ★★★★ | 中 | **这是 Spotlight 加分项**。r_c = d_head/2 的理论 + r-sweep 实验已有 |
| Q5: Progressive 放大 | ★★★ | 高 | 3-seed 复现后自动 partially resolved |
| Q6: τ* at L≥4096 | ★★ | 中 | 无法短期实验验证，保留为 future work |

**Spotlight 级别的理论加强**:

#### A. 代理自洽性定理写入 Appendix（Q1 partial）

λ 闭合分析报告中的新定理：

$$\tau^2 T_2(\tau) + T_1(\tau) = \tau \coth(\tau)$$

这个恒等式证明 τ = √(β/α) 是代理泛函的**精确驻点**，不是近似。把它写入 Appendix 作为 Theorem（或 Proposition），然后在正文 §3 加一句引用。

**价值**: 把理论的 "exact conditional on surrogate" 层级进一步加固。审稿人看到这个恒等式的数值精度（误差 < 10^{-16}）会被 impress。

#### B. LoRA Phase Transition 理论写入正文（Q4）

Mainstory §4.7.5 已经有了完整的理论：

- Frozen 权重引入 coupling stiffness S_frozen ∝ τ²
- r < K = d_head/2 时系统 degenerate，τ* → 0
- r = K 时突变恢复

这个理论的 Paper 价值在于：它**预测了一个 sharp phase transition that was subsequently confirmed experimentally**（r=48 PPL 崩溃，r=64 正常）。如果能在 paper 里写成 Prediction → Confirmation 的形式，这是一个非常 clean 的 theory-experiment loop。

**建议**: 在 §5 LoRA paragraph 前加一段 theoretical prediction：

> "The softmax transport analysis predicts a phase transition at $r_c = K = d_{\mathrm{head}}/2$: below this rank, the frozen weight coupling dominates the variational balance, forcing $\tau^* \to 0$ and rendering EVQ infeasible."

然后 Table 7 的 r=16 崩溃（PPL 77.1）变成了 **confirmation of a theoretical prediction**，而不是一个 limitation。

#### C. Collision Score 消融写入 Appendix（Q2 partial）

tau_diagnostic 数据已有（collision-only vs full objective）。这给 "EVQ 到底改善了什么" 一个 mechanistic answer：off-diagonal energy 降低 63-68%，mutual coherence 从 0.99 降到 0.6。

### Spotlight 路径的时间线

| 周 | 任务 | Spotlight 贡献 |
|----|------|--------------|
| 4/4-4/10 | 行文重构（§3 层级标注、§5 headline、Conclusion 扩展） | 消除 "disappointment" 风险 |
| 4/4-4/10 | 代理自洽性定理 + λ CV table 写入 Appendix | 理论加固 |
| 4/4-4/10 | LoRA phase transition theory 写入正文 | Prediction→Confirmation loop |
| 4/4-4/18 | 3-method LoRA baseline 完成 + positional PPL 更新 | LoRA story 完整化 |
| 4/4-4/18 | Progressive 3-seed 补跑 | 消除 single-seed 弱点 |
| 4/18-4/25 | 合并所有新数据 + 全文 polish | 最终版 |

### Spotlight 概率估计

| 场景 | 概率 |
|------|------|
| 当前论文 → Spotlight | ~5% |
| + Progressive 3-seed + 行文重构 | ~12% |
| + LoRA 3-method 成功 (EVQ ≥ YaRN) + 理论加强 | ~20-25% |
| + LoRA 3-method 成功 + Gold-Answer NLL 成功 + 理论加强 | ~25-30% |

Spotlight 的瓶颈不是论文质量，而是**规模 ceiling**（50M-750M vs 1B+）和**下游 benchmark 覆盖**。这两个在投稿前无法解决，但如果 LoRA story 做好了，它变相扩展了 scale evidence（LLaMA-3-8B 是 8B 规模）和 practical applicability。

---

## 第三部分：具体 LaTeX 修改建议汇总

以下按优先级排列，都是不需要新数据的纯行文改动：

### P0 级（不做会影响分数）

1. **§3.7 层级标注**: 在 Proposition 3 前加 framing sentence 区分 exact / semi-analytic
2. **§5.1 开头**: 把 headline result 提前到第一句
3. **Abstract MLA**: 已有，OK
4. **Conclusion 扩展**: 加 2 句 MLA implication + future work

### P1 级（做了明显加分）

5. **§5 加 signal gradient 叙事**: 一句 connecting sentence
6. **§5 progressive paragraph 加 "EVQ raw surpasses EVQ+YaRN"**: 一句
7. **§5 MLA paragraph 加 Phase 18 structural reversal mini table**: 2-3 行
8. **§3.7 τ* 后加 offensive framing**: "even a first-order approximation..." 一句
9. **§5.4 waterbed discussion 加 "high-frequency redundancy"**: 一句 reframe
10. **Related Work 加 scale comparison**: "the broadest from-scratch PE study" 一句

### P2 级（Nice to have）

11. **LoRA phase transition theoretical prediction paragraph**: 需要写 ~5 句理论 + 对接 Table 7
12. **代理自洽性定理 Appendix section**: ~0.5 page LaTeX
13. **λ CV table Appendix**: 一个小表

---

*本文档为行文分析 + Spotlight 路径规划。不包含对论文 tex 文件的直接修改。*
