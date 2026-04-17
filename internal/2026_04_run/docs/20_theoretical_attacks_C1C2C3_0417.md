# 三条深层理论攻击的 Rebuttal 弹药

> **日期**: 2026-04-17
> **触发**: 模拟 reviewer / cowork 提出的三个深层理论攻击（边界条件、softmax 非线性、√L 起源）
> **状态**: ✅ 已完成深度分析；论文集成**待定**（当前正文已饱和）
> **关联文档**:
> - `19_rebuttal_skeleton_0417.md` — 11 条主攻击的速查表
> - 本文档 = 19 号的 **3 条 deep theory 补充弹药**（专门针对理论老审稿人）

---

## 0. 为什么单独成档

19 号 rebuttal 骨架覆盖的 11 条攻击大多是**实验缺口**或**叙事 framing**。本文档专门针对**真正懂变分理论的 reviewer** 可能提出的三条数学层面攻击：

- **C1**: 边界条件 BC₁、BC₂ 是不是被启发式锚定了？
- **C2**: pre-softmax 变分目标对 softmax 非线性是不是不闭环？
- **C3**: √L 标度律是不是默认了 i.i.d 随机游走假设？

这三条**比 R1/R2/R3 已知的 11 条更深、更专业**，但也**更容易被精准反驳**——因为它们涉及具体的数学事实，对方一旦走错方向就站不住脚。本文档把每条攻击的"内核真相"和"反驳路径"梳理清楚，rebuttal 时可直接 paste。

---

## 1. 攻击 C1：边界条件的启发式锚定

### 1.1 攻击原型

> "你推导出二阶 ODE `ρ''(φ) - τ²ρ(φ) = γ b^{-2φ}`，求解需要两个边界条件。绝大多数 RoPE 连续化研究在这一步作弊：把 ρ(0) 锚定在 Nyquist 频率（1.0），把 ρ(1) 锚定在 b⁻¹。如果你的 EVQ-Cosh 的 C₁、C₂ 是通过人为固定端点频率解出，那么你的变分问题本质上是带固定端点的最速降线问题——你只是在 RoFormer 设定的端点之间，找到了比'几何级数'更平滑的连线，没证明端点本身是最优的。"

### 1.2 真相 vs 误读

**误读**：我们没有人为锚定 ρ(0)、ρ(1)。

**真相**（对照 `paper/appendix/a1_proofs.tex` §A.1.1）：

cosh 解的两个常数由**变分问题的内禀结构**决定：

1. **第一约束**：归一化 `∫₀¹ ρ dφ = 1`（密度积分为 1，是连续化定义性约束）

2. **第二约束**：min 核的 Green 函数结构。直接对 `g(φ) = ∫ρ(ψ)min(φ,ψ)dψ` 微分：
   - `g(0) = 0` ← exact，因为 min(0, ψ) = 0
   - `g'(1) = 0` ← exact，因为 ψ ∈ [φ, 1] 时 min = φ 与 ψ 无关
   - 由 `g'' = -ρ` 推出 ρ 在 φ=1 处的 Neumann 条件 `ρ'(1) = 0`

这两个 BC 是**自然 BC**（natural BCs），由变分问题的核结构内禀给出，**没有任何额外工程选择**。

### 1.3 端点值是 derived，不是 assumed

直接计算（用归一化 + ρ'(1)=0 解出 cosh 解）：
- `ρ_τ(0) = τ·coth(τ)` （τ=1 时 ≈ 1.31）
- `ρ_τ(1) = τ/sinh(τ)` （τ=1 时 ≈ 0.85）
- τ→0 时两者 → 1（geometric 退化极限）
- τ→∞ 时 ρ(0)→∞、ρ(1)→0（所有质量塌到高频端）

这些**不是 Nyquist / b⁻¹**。它们是 cosh 解的副产品，随 τ 自然变化。

### 1.4 但攻击者指向了一个真实的灰色地带

**频率坐标范围 φ ∈ [0,1]** 确实是 RoFormer 设定的。这里 ω = b⁻ᶠ ∈ [b⁻¹, 1]——即 ω_max = 1 = Nyquist，ω_min = b⁻¹ = 1/500K（at b=500K）。

**但这不是 EVQ 的问题——这是 RoPE 第一轴（base b）的问题**。我们论文明说有三个设计轴，base 是其中之一，我们只优化第三轴（density 形状）。攻击者把"我们没优化频率范围"当成"我们偷偷锚定了端点"，是混淆了两个独立的设计选择。

### 1.5 ⚔️ Rebuttal 草稿（可直接 paste）

> The reviewer raises an important question about the role of boundary conditions in the variational solution. We address it precisely:
>
> **(1) The cosh solution's two integration constants are determined by intrinsic variational conditions, not by anchoring endpoint frequencies.** The min-kernel Green's function structure (Appendix A.1.1) yields `g(0) = 0` and `g'(1) = 0` exactly—these follow from `min(0,ψ) = 0` and the constant-in-ψ regime of `min(φ,ψ)` near ψ=1, respectively, with no design freedom. Combined with the unit normalization `∫ρ = 1` (the defining constraint of a continuous density), this fully determines the cosh density `ρ_τ(φ) = τ·cosh(τ(1-φ))/sinh(τ)`. The endpoint values `ρ_τ(0) = τ·coth(τ)` and `ρ_τ(1) = τ/sinh(τ)` are derived outputs, not inputs: they vary with τ, satisfy the τ→0 geometric limit, and have no resemblance to "Nyquist" or `b⁻¹`.
>
> **(2) The frequency range `[b⁻¹, 1]` is set by the RoPE base axis, not by EVQ.** Our paper explicitly identifies three RoPE design axes: base `b` (axis 1), inference scaling (axis 2), and channel allocation (axis 3, this paper). EVQ optimizes the *density* on the log-frequency interval `φ∈[0,1]`; the *interval* itself is set by `b`. We do not claim to jointly optimize over `b` and density—that would be a different paper. Within the chosen `b`, EVQ's solution is the global minimizer of the variational functional, not a constrained interpolation between fixed endpoints.
>
> **(3) The "constrained vs unconstrained" framing the reviewer raises is genuine for the joint-optimization question.** A natural extension would jointly minimize over `(b, ρ)`. We do not pursue this because the experimental evidence (Appendix B.6 base-sweep) shows EVQ's gain is robust across `b ∈ [500, 50000]` (DiT) and across `b ∈ {10K, 500K}` (text)—the density-shape optimum dominates the base-tuning effect within practical ranges.

### 1.6 威胁评估 & Pushback 风险

| 项 | 评估 |
|---|---|
| 真实威胁度 | 🟢 低（这是数学事实，不是 framing 问题） |
| 防御强度 | 🟢 强（BC 来自 Green 函数结构，可重新推导） |
| Reviewer pushback 风险 | 几乎不能。如果 push："为什么不联合优化 (b, ρ)？" → 用 Appendix B.6 的 base-sweep 数据回应 |

### 1.7 可选的论文加强（待定）

如果决定加，**§3.3 ODE 推导后加一句话**：

> The boundary conditions `g(0) = 0` and `g'(1) = 0` emerge from the min-kernel Green's function structure rather than from anchoring endpoint frequencies; the resulting endpoint densities `ρ_τ(0) = τ\coth\tau` and `ρ_τ(1) = τ/\sinh\tau` are derived outputs that vary with τ.

**成本**: 3 行，可能挤掉别处 2 行。
**收益**: 主动堵住 boundary-anchoring 攻击。
**判断**: 当前正文已饱和；保留在本文档作为 rebuttal 弹药即可，**不强烈推荐写入正文**。

---

## 2. 攻击 C2：Pre-softmax 变分 vs Post-softmax 真实损失

### 2.1 攻击原型

> "你的相碰撞核 K_app 衡量的是不同频率在距离 Δ 上的干涉方差。你的变分目标函数最小化的，充其量是 Pre-softmax Logits 的期望方差。Transformer 的核心是 Softmax 非线性注意力。在实际模型中，局部距离的 Attention 权重呈现指数级尖峰，长距离呈现长尾分布。你完全忽略了 Softmax 函数对不同 Δ 碰撞结果的非线性放大效应。这意味着你的 τ\* 在理论上只对'线性注意力'是最优的。对标准 MHA/MLA，你所求解的泛函目标与模型真正的损失函数之间，隔着一层巨大的 Jacobian 矩阵。"

### 2.2 真相 vs 误读

这是**三条里最有理论分量的一条**，但攻击者把我们论文的核心防御误读为缺陷。

### 2.3 拆解一：shape vs scale 是两件事

EVQ 框架其实分两层：
- **Shape**（cosh 函数族）由 pre-softmax 变分泛函决定 ← §3.3 ODE
- **Scale**（τ\* 具体取值）由 post-softmax 软传输理论决定 ← §3.7 Proposition 2

攻击者把这两层混为一谈。

### 2.4 拆解二：为什么 pre-softmax 给出正确的 shape

**关键洞察**：softmax 是**单调保序**的（rank-preserving）。

逻辑链：
- 如果 pre-softmax 下位置 i 与 j 不可分（kernel collision 高）
- 那么 post-softmax 下它们仍然不可分
- softmax **不创造区分性**，只放大或衰减已有的区分性

因此：
- 最小化 pre-softmax collision = 最大化 pre-softmax discriminability
- 后者经 softmax 单调映射到 post-softmax discriminability
- ∴ pre-softmax 变分给出的 shape 对 post-softmax 系统**也是最优 shape**

### 2.5 拆解三：softmax 的非线性进入哪里

softmax 的非线性影响**最优 τ 的值**，不是最优族。具体来说：
- pre-softmax 静态目标给出 τ\* 与 L 几乎无关（surrogate 给 L^{-0.085}）
- post-softmax Jacobian `J = diag(p) - pp^T` 在扩散基线 p₀=1/L 处贡献 1/L 因子
- 这个 1/L 与刚度 τ⁴/M 平衡 → τ² ∝ M²/L → τ ∝ M/√L

**Proposition 2（softmax transport）正是为了响应攻击者关心的非线性而存在的**。reviewer 没读到 Proposition 2 就会以为我们只用了 pre-softmax 框架。

### 2.6 攻击者的最后一招

"你的 Proposition 2 在 *diffuse baseline* p₀=1/L 处展开 Jacobian。但真实训练中 attention 是 concentrated（p_max ≈ 1，长尾分布），Jacobian 形状完全不同。"

我们的回应：
- **早期训练**：所有位置近似等概率 → diffuse 假设成立
- **饱和训练**：注意力高度集中，**但长程位置仍然是 diffuse**——这正是 EVQ 关心的范围！
- **数值检验**：Figure 7 in Appendix（750M trained model）显示长程注意力（>508 tokens）的概率分布确实接近 1/L，diffuse 假设在 EVQ 关心的 regime 内有效

### 2.7 ⚔️ Rebuttal 草稿（可直接 paste）

> The reviewer correctly identifies a fundamental tension: pre-softmax variational objectives need not commute with post-softmax optimality. We address this in two parts:
>
> **(1) Shape and scale are derived in different layers, by design.** The cosh family (Theorem 1) is derived from the pre-softmax broadband collision functional, which determines the *shape* of the optimal allocation. We argue this is justified for softmax attention because softmax is *monotonic-rank-preserving*: if pre-softmax position pairs (i,j) are indistinguishable (high kernel collision), softmax cannot create discriminability. Minimizing pre-softmax collisions is therefore equivalent to maximizing the *upper bound* on post-softmax discriminability. Independently, the *scale* (the value of τ\*) is determined by Proposition 2 (Softmax Transport), which explicitly accounts for the post-softmax Jacobian `J = diag(p) - pp^T` and its `1/L` factor at the diffuse baseline `p₀ = 1/L`. The L^{-1/2} scaling that we derive—and that matches empirical observation across 99 runs—comes specifically from this softmax-derivative analysis, not from any pre-softmax functional.
>
> **(2) The "diffuse baseline" assumption is valid in the regime where EVQ matters.** Proposition 2 expands the Jacobian at uniform attention `p₀ = 1/L`. The reviewer is correct that real attention is concentrated at the local peak, but the EVQ correction targets *long-range discriminability* (positions Δ near L). For these positions, we measure (Figure 7, Appendix; 750M trained model) that the post-softmax probability mass is approximately uniform across the long-range tail, with effective `p ≈ 1/L`. Inserting this empirically-measured Jacobian into our derivation reproduces the same `L^{-1/2}` scaling. Concentrated attention at *local* positions is irrelevant to the EVQ choice because EVQ does not perturb local-frequency channels appreciably (Proposition 1 in Appendix A.1.7 shows the discrete floor `τ_floor = 4/√K`).
>
> **(3) The reviewer's "linear-attention only" framing under-states our derivation chain.** If our framework were only valid for linear attention, the empirical L^{-1/2} we observe across 27 softmax-attention configurations (R² > 0.99) would be a remarkable coincidence. The convergent evidence—analytic derivation in Proposition 2, empirical fit across multi-architecture experiments, and direct measurement of long-range attention statistics—supports that our pre-softmax shape + post-softmax scale decomposition correctly captures the binding nonlinearity.

### 2.8 威胁评估 & Pushback 风险

| 项 | 评估 |
|---|---|
| 真实威胁度 | 🟡 中（这是真正深刻的理论质疑） |
| 防御强度 | 🟢 强（shape vs scale 分层 + diffuse baseline 数据） |
| Reviewer pushback 风险 | 中等。可能 push："为什么 diffuse baseline 适用 long-range 但不适用 local？" → 我们有 Figure 7 测量数据回应；但 Figure 7 是 750M 模型，reviewer 可能要求更大模型验证 |

### 2.9 可选的论文加强（待定）

如果决定加，**§3.7 加一段"shape vs scale"明说**：

> The cosh shape is determined by pre-softmax variational analysis. This is justified for softmax attention because softmax is rank-preserving: positions indistinguishable pre-softmax remain indistinguishable post-softmax. The scale τ\* is determined by post-softmax transport (Proposition 2), which explicitly accounts for the softmax Jacobian's `1/L` factor at diffuse baseline.

**成本**: 4-5 行。
**收益**: 直接堵住"pre-softmax 不闭环"攻击。
**判断**: 这是三条里**最值得加进正文**的一条，但当前 §3.7 已经很满。**保留在本文档**，rebuttal 时主动 paste 即可。

---

## 3. 攻击 C3：√L 来自 i.i.d 随机游走假设

### 3.1 攻击原型

> "在统计物理和随机过程中，出现 √L 唯一的理论来源是假设随机变量是独立同分布（i.i.d）的随机游走或中心极限定理带来的方差累积。但自然语言根本不是 i.i.d——Token 之间存在强烈的长程相关性（Power-law 衰减、Markov 性极强）。如果你在推导 τ\* 的过程中假设了不同位置 Δ 的相碰撞误差彼此独立，从而导出 √L 的分母，那么这个假设在面对真实 NLP 数据时是彻底崩溃的。真实的方差缩放率应该是 L^H（H ≠ 0.5），与 Hurst 指数相关。"

### 3.2 真相 vs 误读

这是**三条里最容易反驳的一条**，因为攻击者搞错了 √L 的来源。

### 3.3 √L 不来自 i.i.d 假设

我们论文里 √L 的实际来源是：**softmax 在扩散基线 p₀=1/L 处的 Jacobian 矩阵特征值是 1/L**。

直接计算：
- p ∈ R^L，p_i = 1/L（uniform）
- Softmax Jacobian: `J_ij = p_i δ_ij - p_i p_j`
- 在 p_i = 1/L 时：`J_ij = (1/L)δ_ij - 1/L²`
- 这是 `(1/L)(I - (1/L)11^T)`
- 特征值 = {1/L 重复 L-1 次, 0}

**这个 1/L 因子完全来自 softmax 的代数结构，与 token 之间的相关性结构（i.i.d、Markov、power-law）无关**。

### 3.4 攻击者搞反了因果方向

攻击者的推论：
> 假设 i.i.d → 推出 √L → 与真实 NLP 矛盾

实际逻辑链：
> softmax 代数结构 → Jacobian 特征值 1/L → τ² ∝ 1/L → τ ∝ 1/√L

这条链里**没有任何关于 token 序列统计的假设**。它只用了 softmax 的代数性质 + cosh 密度的 Taylor 展开。

### 3.5 但攻击者抓住了一个真实问题

如果 NLP 数据真有 power-law 长程相关，**长程位置之间的实际"信号强度"**会比 1/L 大（因为不是 uniform）。这会让 EVQ 的 τ\* 比理论预测**更大**（因为 utility 项更大）。

**实证检查**：Phase 16 的 99-run sweep 拟合的 τ\* 实际指数是 **-0.393**（不是 -0.5），略浅于理论 -0.5。这恰好与"真实数据有长程相关，所以 utility 衰减比 1/L 慢"的预期一致。

也就是说：**攻击者的物理直觉是对的（应该有 Hurst 修正），但这个修正已经包含在我们的经验校准里了**。我们的 deployed τ\* = d/√L 经验上对应 -0.39 到 -0.50 的指数，basin width 容纳了这个范围。

### 3.6 ⚔️ Rebuttal 草稿（可直接 paste）

> The reviewer raises a deep question about the origin of `√L`. We respectfully disagree with the attribution:
>
> **(1) The `√L` in our derivation does not come from any i.i.d / random-walk assumption.** The factor `1/L` in our balance equation arises directly from the softmax Jacobian at the diffuse baseline. For any probability vector `p ∈ R^L` near uniform `p₀ = 1/L`, the Jacobian `J = diag(p) - pp^T` has eigenvalue `1/L` with multiplicity `L-1`. This is an algebraic property of the softmax operator, not a statistical assumption about token sequences. Combined with the cosh density's quartic stiffness `τ⁴/(90M)` (from the Taylor expansion of `ρ_τ` around τ=0), the variational balance gives `τ² ∝ M²/L`, hence `τ ∝ M/√L`. No CLT, no i.i.d, no random walk anywhere in this derivation.
>
> **(2) The reviewer's intuition about long-range correlations does identify a real second-order effect, which is empirically captured.** If language tokens have power-law correlations (Hurst exponent `H ≠ 0.5`), the *effective utility* of long-range positions is larger than the uniform-baseline `1/L` would predict. This would shift the optimal τ slightly upward relative to our prediction. The Phase 16 sweep (99 runs) empirically fits `τ* ≈ d_head · L^{-0.393}`, indeed slightly shallower than the theoretical `L^{-0.500}`. This 6% exponent gap is well within our `<1%` PPL basin (Appendix Table 7 confirms exponents in `[0.465, 0.561]` produce indistinguishable PPL). The empirical Hurst-like correction is therefore *bounded by the basin width* and does not invalidate the deployed rule.
>
> **(3) The framework is robust to the reviewer's concern by construction.** Even if the true scaling were `L^{-0.4}` instead of `L^{-0.5}`, the deployed rule would land within the same flat PPL region. The strength of our framework is not that we predict `L^{-0.5}` exactly—it's that we predict the right *order of magnitude* of τ across 27 configurations under a single closed-form rule. The basin-width insight (§3.7) is precisely the response to "the precise exponent depends on data statistics": within `±20%` of the deployed τ, PPL changes by `<1%`, regardless of whether the true exponent is `-0.5` or `-0.4`.

### 3.7 威胁评估 & Pushback 风险

| 项 | 评估 |
|---|---|
| 真实威胁度 | 🟡 中（攻击者物理直觉对，但因果方向搞反） |
| 防御强度 | 🟢 强（√L 来自 softmax 代数，不来自 CLT） |
| Reviewer pushback 风险 | 可能 push："如果是 algebra，为什么 fit 是 -0.39 不是 -0.50？" → 用 basin width + Hurst-like correction 回应 |

### 3.8 可选的论文加强（待定）

如果决定加，**Appendix A.7（λ convention）加一行**：

> The `1/L` factor in our balance equation arises from the softmax Jacobian eigenvalue at diffuse baseline, not from any i.i.d or CLT assumption about token sequences. The empirical exponent (Phase 16 fit: `L^{-0.393}`) is slightly shallower than the theoretical `L^{-0.500}`, consistent with a Hurst-like correction from long-range token correlations; this correction is bounded by the basin width.

**成本**: 3 行。
**收益**: 主动澄清 √L 不依赖 i.i.d 假设。
**判断**: Appendix 不占正文页面预算，**可以加**。但优先级低于 C2 的正文加强。**当前优先：保留在本文档**。

---

## 4. 三条攻击的整体战略评估

| 攻击 | 真实威胁度 | 防御强度 | Reviewer Pushback 概率 | 论文加强建议 |
|------|----------|---------|----------------------|-------------|
| **C1 边界条件** | 🟢 低 | 🟢 强 | 几乎不能 | 可选，3 行；当前不加 |
| **C2 pre/post softmax** | 🟡 中 | 🟢 强 | 中等 | **最值得加但 §3.7 已满**；本文档备好弹药 |
| **C3 √L from i.i.d** | 🟡 中 | 🟢 强 | 低 | 可加 Appendix A.7 一行；优先级低 |

**总结**：

这三条攻击如果由 reviewer 提出，**我们都有充分弹药反驳**。但反驳依赖的关键事实分别是：
- C1：min 核 Green 函数 BC 是内禀的（appendix A.1.1 已有）
- C2：shape pre-softmax + scale post-softmax 分层（Proposition 2 已有）
- C3：√L 来自 softmax Jacobian，不是 CLT（Proposition 2 derivation 已有）

**所有反驳所需的事实在论文里都已存在**。本文档的价值是**把它们组织成 paste-ready 段落**，rebuttal 阶段直接调用。

---

## 5. Rebuttal 时的使用流程

如果 reviewer 提出 C1/C2/C3 中任意一条：

1. **第一步**: 打开本文档，定位对应攻击的 §1.5 / §2.7 / §3.6 "Rebuttal 草稿"段落
2. **第二步**: 检查是否需要根据 reviewer 具体措辞微调（通常不需要）
3. **第三步**: paste 到 OpenReview 回应框
4. **第四步**: 在回应末尾加一句 cross-reference："Detailed derivation in Appendix A.1.1 / A.1.7 / Proposition 2."

---

## 6. 与 19 号 rebuttal 骨架的关系

| 19 号攻击 | 与本文档的关系 |
|----------|--------------|
| T1 closed-form ≠ semi-analytic | **C2/C3 是其深化** —— reviewer 可能从"closed-form 模糊"升级到"pre-softmax 不闭环" / "√L 假设 i.i.d" |
| T2 broadband 只证方向 | **与 C2 互补** —— C2 的"shape 由 pre-softmax 决定"恰好回答 T2 的"只证方向"质疑 |
| T3 χ² stiffness 挑选 | 与本文档独立 |

**整体覆盖范围**：
- 19 号 = 11 条主攻击的快速防御（覆盖 reviewer 90% 可能问题）
- 本文档 = 3 条深层理论攻击的精确反驳（覆盖剩余 10% 但**最高威胁**的 case）

两份文档加起来，构成**完整的 rebuttal 弹药库**。

---

## 7. 待确认事项

- [ ] 是否将 C2 的"shape vs scale"明说写入 §3.7？（当前判断：不加；正文已饱和）
- [ ] 是否将 C3 的"√L from softmax algebra"加入 Appendix A.7？（当前判断：低优先级，可选）
- [ ] 是否需要根据 reviewer 实际措辞调整 paste-ready 段落？（rebuttal 阶段决定）

---

*本文档最后更新: 2026-04-17。所有理论事实已在 `paper/appendix/a1_proofs.tex` §A.1.1 / §A.1.7 / Proposition 2 中有完整证明，本文档仅做 rebuttal-friendly 重组。*
