# EVQ-Cosh 论文行文策略分析

> 2026-03-31 · 基于当前论文全文 + 理论分析结果

---

## 一、核心矛盾：论文到底在 claim 什么？

当前论文的 claim 链条是：

1. 频率分配是 RoPE 的一个 fundamental design axis（正确，没争议）
2. 我们从变分问题推出了 closed-form 的 EVQ-cosh 族（正确，但有 surrogate 近似）
3. τ* = d_head/√L 是最优温度的 scaling law（**这里有裂缝**）
4. 实验全面验证了 EVQ 的优越性（数据是好的）

**裂缝在第3步。** 论文目前的写法是 "semi-analytic result"，limitations 里也老实交代了 λ 是 calibrated 的。但问题是：审稿人读完理论部分，会期待一个 **理论驱动的最优解**，然后发现 τ* 其实是半拟合的。这个落差才是 borderline 的根源。

---

## 二、当前行文的具体问题

### 2.1 理论部分的过度承诺

03_theory.tex 的行文结构是：variational problem → surrogate → ODE → closed-form → τ* scaling law。读起来像是一气呵成的推导，给人的印象是 τ* 是从第一性原理得出的。但实际上：

- surrogate K_app 是一个 **功能性近似**，不是逐点保真的
- ODE 的解是 exact **conditional on surrogate**
- τ 的含义是 √(β/α)，但 α, β 没有解析表达式
- scaling law 的 L^(-1/2) 指数来自 softmax transport，但 λ 是拟合的
- 最终 τ* = d_head/√L 是把 λ≈1.17 吸收进去后的简化

**问题：** 论文没有清楚地区分"推导出来的部分"和"拟合出来的部分"。Theorem 1, 2 和 Proposition 3 之间的逻辑层级不够分明。审稿人如果仔细读，会觉得理论的 "exact" 和实际的 "semi-analytic" 之间有 gap。

### 2.2 实验结果的叙事依赖 composition

论文最强的实验结果几乎都是 EVQ + YaRN 的组合：
- 100% retrieval at 8K（EVQ+YaRN vs Geo+YaRN）
- PPL 14-32% improvement（组合）
- 48K functional context（组合 + progressive）

**EVQ 单独的提升** 相对温和：
- MLA 上 EVQ alone -31.1% PPL（这个很强）
- PE-dominant 333.7 vs 455.3（强，但是极端设置）
- Multi-scale in-range cost ≤1%（好，但不是 exciting 的数字）

审稿人可能会问：EVQ 到底是一个好的 frequency allocation，还是只是一个让 YaRN 能 work 的 enabler？如果是后者，贡献就被弱化了。

### 2.3 "金玉其外"的具体表现

你说的"金玉其外败絮其中"，我理解具体指的是：

1. **τ* 不是最优**：Phase 16 的 99 组实验验证了 scaling law，但指数偏差 6%，λ 的 CV 是 11.1%。这意味着公式给的 τ 在某些配置下并非 empirical best
2. **surrogate 的 R² 只有 0.25-0.73**：functional validation 通过了（collision reduction 24-92%），但 pointwise fidelity 很差
3. **LoRA 完全失败**：r < d_head/2 时崩溃，限制了实际应用场景
4. **DiT 需要 post-hoc 0.53× 修正**：跨模态时公式直接用不了
5. **LongBench 各有胜负**：下游 benchmark 没有 consistent win

这些单独看都是 acceptable limitations，但组合在一起，审稿人可能觉得：理论看上去很完整，但实际上处处有 gap。

---

## 三、反面视角：为什么"非最优解赢了 Geo"反而可能更强

这是你提到的另一面。如果换个 narrative 角度：

**Geometric RoPE 是 60 多年来位置编码的默认选择（从 sinusoidal PE 到 RoPE 一脉相承）。EVQ 用一个 principled but imperfect 的变分解就打赢了这个默认选择，说明频率分配这个 design axis 的优化空间远比人们想象的大。**

这个 narrative 的力量在于：
- 不需要 claim τ* 是最优的
- 重点放在 "frequency allocation 这个 axis 本身是重要的" 这个 insight 上
- EVQ 是这个 axis 上的 **第一个 principled attempt**，不需要是最后一个
- τ* 的 semi-analytic nature 变成了 "even a first-order approximation already wins"

**对应的 reviewer 心理模型：**
- 好 reviewer：这个 direction 是对的，first principled approach 已经 work 了，贡献是 opening the door
- 坏 reviewer：理论有 gap，实验没有 SOTA，borderline reject

---

## 四、行文修改策略（暂不执行）

### Strategy A：强化 "principled framework" 叙事，弱化 "optimal formula" 叙事

**核心调整：** 把论文从 "我们找到了最优 τ*" 重新定位为 "我们建立了频率分配的变分框架，并证明该框架的一阶近似就已经超过 Geometric baseline"。

具体改动点：

1. **Abstract：** 当前写法 "derive a broadband surrogate whose stationary density satisfies a closed-form ODE" 暗示了 exactness。可以调整为更强调 framework 而非 formula。

2. **Theory section (§3)：**
   - 把 surrogate 的角色从 "approximation step" 提升为 "the key modeling insight"
   - 显式区分三个层次：(i) variational formulation（exact），(ii) surrogate + ODE（exact conditional on surrogate），(iii) τ* scaling law（semi-analytic, empirically calibrated）
   - 加一段话：even if τ* is not the global optimum, the variational framework identifies the correct direction of improvement

3. **τ* 呈现方式：**
   - 当前：τ* = d_head/√L 放在 Proposition 里，读起来像定理
   - 调整：明确标记为 "empirical scaling law consistent with the variational prediction"，加 λ cross-validation table 到 appendix

4. **Conclusion：** 加一句 "the gap between the semi-analytic τ* and the empirical optimum is small (< 1% PPL), suggesting that the variational framework captures the essential structure even in its current approximate form"

### Strategy B：补一个关键实验来封口

如果 τ* 确实不是最优的，那最好的防御是：**show that the empirical optimum is close to τ***。

Phase 16 的数据已经部分做了这个（99 runs, < 1% PPL gap in the optimality basin），但论文里没有充分利用这个结果。建议：

- 把 τ* 的 optimality basin 分析从 appendix 提到 body
- 画一张 τ vs PPL 的 landscape 图，标注 τ* 的位置和 basin 宽度
- 这张图的 message 是：τ* 不需要是全局最优，因为它落在一个宽的浅 basin 里

### Strategy C：正面 address "非最优解赢了" 的 narrative

在 discussion 或 conclusion 中加一段：

> The fact that a semi-analytic approximation to the variational optimum already yields substantial improvements over geometric allocation suggests that the frequency allocation axis has been severely under-optimized in current practice. This parallels the history of learning rate scheduling: even a coarse cosine schedule outperforms constant learning rate by a wide margin, and the precise schedule matters less than the principle of annealing itself. Similarly, the precise value of τ* matters less than the principle of collision-aware redistribution.

这个类比把 "不是最优" 变成了 "principle > formula" 的 positive message。

---

## 五、当前论文的真实定位

诚实地说，这篇论文的核心贡献是：

| 贡献 | 强度 | 审稿风险 |
|------|------|----------|
| 提出频率分配作为 PE design axis | ★★★★★ | 低 — 这是 genuine insight |
| Variational framework + EVQ-cosh 族 | ★★★★ | 中 — surrogate 是软肋 |
| τ* scaling law | ★★★ | 高 — semi-analytic，审稿人可能不买账 |
| EVQ+YaRN composition 实验 | ★★★★ | 中 — 结果很好但依赖组合 |
| MLA 实验 | ★★★★★ | 低 — EVQ alone -31.1%，最干净的结果 |
| 跨架构/跨模态 | ★★★ | 中 — DiT 需要 post-hoc correction |

**最强的两个卖点：** (1) frequency allocation as a design axis, (2) MLA result (EVQ alone wins big)

**最大的软肋：** τ* 的 semi-analytic nature + LongBench 不 consistent

---

## 六、对师兄 LongRoPE2 建议的看法

师兄说的 EVQ + LongRoPE2 > Geo + LongRoPE2 实验，从论文策略角度分析：

**好处：**
- 如果 work，证明 EVQ 的价值跨 extension method（不只是 YaRN）
- LongRoPE2 是 ICML 2026，审稿人认可度高
- 和现有的 EVQ+YaRN composition 叙事一致

**风险：**
- LongRoPE2 的 evolutionary search 可能把初始频率差异 "搜掉"
- 计算开销大
- 如果效果不明显，反而削弱 EVQ 的 narrative

**判断：** 如果有时间和算力，这是一个 nice-to-have 的加分实验。但**不是解决核心问题的关键**。核心问题是 τ* 的理论自洽性，不是缺一个新 baseline。

---

## 七、建议的优先级排序

1. **P0（行文调整）：** 重构 narrative，从 "optimal formula" 转向 "principled framework"
2. **P0（补数据展示）：** τ* optimality basin 图从 appendix 提到 body
3. **P1（理论补充）：** λ ≈ 1.17 的 cross-validation table 加到 appendix
4. **P2（可选实验）：** EVQ + LongRoPE2 如果时间允许
5. **P2（理论完善）：** 小 τ 渐近分析写进 appendix 作为 theoretical note

---

*本文档为内部策略分析，暂不对论文做任何改动。*
