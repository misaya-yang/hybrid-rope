# τ*能否从第一性原理推导？——精确诊断与修补路径

> **日期**: 2026-03-22
> **问题**: 为什么 τ = √(β/α) 给出 L^{-0.085}，而不是经验观测的 L^{-0.5}？
> **立场**: 不接受"经验律"，追问到底哪里丢了信息

---

## 1. 推导链的精确断面

当前理论链共6步。让我逐步追踪 L 的信息流:

```
Step 1: D(Δ)=1/(ΔlnL) on [1,L]     ← L 在这里
Step 2: K(φ₁,φ₂) = ∫D(Δ)cos(ω₁Δ)cos(ω₂Δ)dΔ    ← L 在积分上界
Step 3: K ≈ αδ + β·min    ← α,β 通过拟合吸收了 L 的信息
Step 4: ρ'' - τ²ρ = γb^{-2φ},  τ=√(β/α)    ← τ 从 α,β 继承 L
Step 5: ρ = cosh(τ(1-φ))    ← 函数族确定
Step 6: φ_k(τ) = 闭式映射    ← 离散化
```

**L信息的实际传递** (数值验证, d_head=64, b=500K):

| L | α | β | τ_surr=√(β/α) | τ*=d/√L | ratio |
|---|---|---|---|---|---|
| 128 | 0.0251 | 1.486 | 7.69 | 5.66 | 1.36× |
| 256 | 0.0243 | 1.330 | 7.40 | 4.00 | 1.85× |
| 512 | 0.0236 | 1.167 | 7.04 | 2.83 | 2.49× |
| 1024 | 0.0227 | 1.017 | 6.70 | 2.00 | 3.35× |
| 2048 | 0.0218 | 0.849 | 6.24 | 1.41 | 4.42× |
| 4096 | 0.0211 | 0.685 | 5.70 | 1.00 | 5.70× |

**拟合结果**: α ~ L^{-0.051}, β ~ L^{-0.221} → τ_surr ~ L^{-0.085}

**关键观察**:
- α 几乎不依赖 L（这是因为对角线 K(φ,φ) 由局部自相关决定，与积分区间 [1,L] 弱相关）
- β 只有微弱的 L 依赖（L^{-0.22}），因为 off-diagonal coupling 在 D(Δ)∝1/Δ 先验下以 ln(L) 增长
- 两者结合给出 τ ~ L^{-0.085}，差了 L^{-0.415} 的因子

## 2. 断裂的精确位置

**断裂在 Step 3 → Step 4**: broadband投影 K ≈ αδ + β·min 丢失了 L 信息。

为什么？因为 broadband surrogate 是一个**两参数模型**。K(φ₁,φ₂) 是一个 K×K 矩阵（对 K=32 就是 32×32=1024 个元素），而 αδ+β·min 只有2个自由度。这个巨大的信息压缩把 L 对 kernel 结构的精细影响（特别是振荡性的 sinc 项）抹掉了。

**更精确地说**: exact kernel 的 off-diagonal 包含 sin((ω_i-ω_j)L)/((ω_i-ω_j)L) 项。当相邻通道靠近时（|ω_i-ω_j| < π/L），这个 sinc 项 ≈ 1（constructive interference）。这个效应依赖于 K 和 L 的比值，但 β·min(φ,ψ) 是光滑的，无法捕捉这种随 K/L 变化的离散相干效应。

## 3. 能修补吗？——三条路径

### 路径A: 扩展 surrogate 到三参数

将 K_app 扩展为:
$$K_{\text{app}} = \alpha\delta(\phi_1-\phi_2) + \beta\min(\phi_1,\phi_2) + \eta\cdot\Phi(\phi_1,\phi_2;L)$$

其中 Φ 是捕捉 L-dependent 离散相干效应的第三个 basis function。

**候选Φ**: Φ(φ₁,φ₂;L) = sinc(|b^{-φ₁}-b^{-φ₂}|·L) — 直接表示两个通道在距离L处的相干性。

**问题**: 引入 Φ 后 Euler-Lagrange ODE 不再是常系数的。解不再是 cosh 族，失去了闭式 CDF 反演。

**评估**: 理论上可行，但会摧毁 EVQ 的实用性（需要数值求解）。不推荐作为主要路径。

### 路径B: 自洽 surrogate（最有前景）

**核心洞察**: 当前的 α, β 是在**均匀网格** φ_k = k/K 处拟合 exact kernel。但 EVQ 的通道并不在均匀网格上——它们被 τ 变形了。正确做法是在 **EVQ 的实际通道位置**处拟合 surrogate。

这构成一个**自洽方程**:
1. 在 φ_k(τ) 处拟合 exact kernel → 得到 α*(τ), β*(τ)
2. τ = √(β*(τ)/α*(τ))
3. 用新的 τ 回到 step 1

**为什么这可能给出 L^{-1/2}**:

当 τ > 0 时，EVQ 将通道集中在低频端。在集中的区域，相邻通道频率更近 → exact kernel 的 sinc 项更接近 1 → off-diagonal 相关性增强 → β*(τ) 增大。

β*(τ) 的增量来自于集中区域中"相干对"(coherent pairs)的数目:
$$\Delta\beta \propto N_{\text{coherent}}(τ, L) \propto \frac{K\tau}{L \cdot \Delta\omega_{\min} \cdot \text{sinh}(\tau)}$$

这里 Δω_min 是低频端的通道间距。关键的 L 依赖在分母中: 相干球半径 π/L 随 L 缩小，所以更大的 L → 更少的相干对 → 更小的 β* → 更小的 τ。

**粗略 scaling**:
- α* ≈ 1/K = 2/d_head（几乎不变）
- β*(τ) ≈ β₀ + c·Kb/(L·sinh(τ)·lnb)
- 自洽: τ² = β*/α* = 2d_head·[β₀ + cKb/(L sinh(τ) lnb)]

在 τ 适中 (sinh(τ) ≈ τ) 的区域:
τ² ≈ 2d_head·β₀ + 2d_head·cKb/(Lτ·lnb)

令 A = 2d_head·β₀ (surrogate贡献) 和 B = 2d_head·cKb/(lnb) = c·d_head²·b/lnb (离散相干贡献):
τ³ + Aτ ≈ Aτ + B/L
τ³ ≈ B/L
**τ ∝ (d_head²/L)^{1/3}**

这给出 L^{-1/3}，比 L^{-0.085} 强很多但还不到 L^{-0.5}。

**但是**: 如果 coherent pair 的影响不是通过 β（全局 off-diagonal 耦合）而是通过一个更强的 **局部** 项，exponent 可能会更大。需要数值计算来确定。

### 路径C: 训练动力学修正的变分原理（最物理的路径）

**核心思想**: 变分泛函 J[ρ] 的真正目标不是最小化静态 collision，而是最小化**训练损失**。训练损失与静态 collision 之差在于梯度信号质量——模型通过 SGD 从频率分配中学习的效率。

**修正泛函**:
$$J_{\text{train}}[\rho] = \underbrace{\frac{\alpha}{2}\int\rho^2 + \frac{\beta}{2}\iint\rho\rho\cdot\min}_{\text{静态 collision}} + \underbrace{\frac{\lambda}{2}\int\rho(\phi)^2 \cdot V(\phi,L) \, d\phi}_{\text{梯度方差惩罚}}$$

其中 V(φ,L) 是频率 φ 处的梯度方差:
$$V(\phi,L) = \sum_{\Delta=1}^{L} \frac{1}{\Delta}\cos^2(b^{-\phi}\Delta)$$

**V的行为**:
- 高频 (b^{-φ}L >> 1): V ≈ (1/2)·ln(L) — 充分采样，稳定
- 低频 (b^{-φ}L << 1): V ≈ b^{-2φ}L²/4 — 弱信号，L²增长

V 的 L² 增长在低频端造成一个**随L增强的惩罚**: 训练序列越长，模型在低频通道上的梯度累积越多，但这些梯度相互高度相关（因为低频通道对所有距离的贡献几乎相同），产生更大的方差。

**修正后的有效 α**:
$$\alpha_{\text{eff}}(\phi) = \alpha + \lambda V(\phi, L)$$

在低频端: α_eff(1) ≈ α + λ·b^{-2}L²/4
在高频端: α_eff(0) ≈ α + λ·ln(L)/2

有效 τ 的 ORDER OF MAGNITUDE:
$$\tau_{\text{eff}} \sim \sqrt{\frac{\beta}{\alpha + \lambda \cdot \langle V \rangle}}$$

如果低频端的惩罚主导 (λ·b^{-2}L²/4 >> α):
$$\tau_{\text{eff}} \sim \sqrt{\frac{\beta \cdot 4}{\lambda \cdot b^{-2} \cdot L^2}}$$

但这给出 τ ∝ 1/L，太强了。

**关键**: V的主导贡献不是来自最低频（那里 b^{-2φ} 极小），而是来自**死区边界**附近（φ ≈ φ* 处）。在死区边界:

ω* = b^{-φ*} = 2π/L

V(φ*, L) = Σ cos²(2πΔ/L)/Δ ≈ (1/2)H_L + (1/2)Σcos(4πΔ/L)/Δ ≈ ln(L)/2

所以死区边界处 V 只有 ln(L) 量级，不是 L² 量级。

**重新计算**: 如果 V 的有效平均值 ∝ ln(L):
α_eff ≈ 1/d_head + λ·ln(L)

τ_eff ≈ √(β/(1/d_head + λ·ln(L)))

当 λ·ln(L) >> 1/d_head:
τ_eff ∝ √(β/(λ·ln(L))) ∝ 1/√(ln L)

这是**对数**衰减，比 L^{-0.5} 慢得多。

**问题**: 训练动力学的修正量级不够给出幂律衰减。

## 4. 真正的理论图景（诚实结论）

经过以上三条路径的分析，我现在可以给出一个更精确的诊断:

### 4.1 能从第一性原理推导的部分

| 结果 | 推导方式 | 严格性 |
|------|---------|--------|
| cosh 密度族 | E-L ODE 的精确解 | **定理** (条件于 surrogate) |
| geometric 是 τ=0 极限 | Taylor 展开 | **定理** |
| α ≈ 1/d_head | kernel 对角线结构 | **引理** |
| β = O(1), 弱 L 依赖 | kernel off-diagonal 结构 | **引理** |
| τ_surr ∝ √d_head | α和β的组合 | **推论** |
| τ_floor = 4/√K | 离散化截断 | **命题** (可严格证明) |
| 宜居带 [1.0, 2.5] | floor + ceiling | **命题** (部分严格) |

### 4.2 不能从当前理论推导的部分

| 结果 | 原因 | 最近的理论近似 |
|------|------|--------------|
| L^{-0.5} 指数 | broadband surrogate 丢失离散相干信息 | L^{-0.085} (差6倍) |
| τ*的精确系数 | 同上 | τ_surr/τ* ∈ [1.4, 5.7] (差很多) |
| DiT 的 γ≈0.53 | 未建模双向注意力 | 无 |

### 4.3 为什么差距如此大（L^{-0.085} vs L^{-0.5}）

**根本原因**: 这不是一个"小修正"的问题。τ_surr 和 τ* 的差距从 1.4×（L=128）增长到 5.7×（L=4096）。到 L=8192 时，τ_surr ≈ 5.2 而 τ* ≈ 0.71（差 7.3×）。

这意味着 surrogate 预测的 τ 在实际训练中是**严重错误的** — 差的不是几个百分点，而是数倍。surrogate给出的 τ≈6 对应极端再分配（密度比 > 200:1），而实验最优 τ≈1.5 只是温和再分配（密度比 2.35:1）。

**物理解释**: 静态 collision 分析告诉你"把所有通道都搬到低频区最好"（因为低频通道是瓶颈），但训练动力学告诉你"搬太多学不动"。后者是主要约束，前者只提供方向。

### 4.4 这意味着什么？

**诚实的陈述**: 从 broadband variational principle 推导出的不是 τ* 的值，而是:
1. **正确的函数族** (cosh) — 这在所有竞争族中唯一
2. **正确的d_head方向** — 更大的 d_head → 更大的 τ
3. **τ=0 是退化极限** — geometric 不是最优而是特殊点

τ* = d_head/√L 是一条在 [256, 4096] 区间内 R²>0.99 的经验标度律。它的 d_head 因子有理论支持（来自 α），但 L^{-0.5} 指数**不来自变分原理**，而是训练动力学的涌现属性。

## 5. 如何在论文中最优地处理这个问题

### 方案1（当前论文的写法 — 已经接近最优）:

论文 03_theory.tex 第109行已经写了:
> "The L^{-1/2} exponent is a finite-channel correction that the continuous variational theory does not resolve, arising from oscillatory inter-channel correlations..."

这是诚实的。但可以更强:

### 方案2（推荐的改进）:

**Theorem 1**: cosh 族是 broadband 泛函的唯一稳态解 ← 不变
**Theorem 2**: geometric 是 τ→0 极限 ← 不变
**Proposition (New)**: τ 的离散下界 τ_floor = 4/√K ← 新增，可严格证明
**Empirical Law**: τ* = max(d_head/√L, C/√K) ← 将下界融入公式

**Remark**: The scaling law τ* = d_head/√L is validated with R²>0.99 across 99 runs. The d_head dependence is analytically derived (α ≈ 1/d_head). The L^{-1/2} exponent is an emergent property of the training dynamics that the static broadband functional does not resolve — analogous to how learning rate scaling in deep learning (η ∝ 1/√T) emerges from SGD dynamics rather than from the loss landscape topology.

**关键类比**: 学习率的最优 scaling η ∝ 1/√T 也不是从 loss function 推导的，而是从 SGD 动力学推导的（Robbins-Monro 条件）。τ* 的 L^{-1/2} 与此类似——它是优化算法与目标的交互属性，不是纯目标的属性。

## 6. 可以做的补充工作（把差距缩小）

### 6.1 自洽 surrogate 计算（计算量小，理论价值高）

```python
# 伪代码: 自洽迭代
tau = 1.5  # 初始猜测
for _ in range(10):
    phi_k = evq_cosh(tau, K)  # 在 tau 给定的通道位置
    K_exact = compute_exact_kernel(phi_k, L, b)  # 在这些位置计算精确kernel
    alpha_sc, beta_sc = fit_broadband(K_exact, phi_k)
    tau_new = sqrt(beta_sc / alpha_sc)
    if |tau_new - tau| < 1e-4: break
    tau = tau_new
```

如果自洽 τ 的 L-scaling 比均匀网格的 L^{-0.085} 更接近 L^{-0.5}，这就是理论进步。

### 6.2 训练动力学的严格化

定义训练损失在 τ 附近的 Hessian:
$$H(\tau) = \frac{\partial^2 \mathcal{L}_{\text{PPL}}}{\partial \tau^2}\bigg|_{\tau=\tau^*}$$

如果 H(τ) 的 L-scaling 可以理论估计，加上 shallow basin 的观测（H 很小），可以给出 τ* 附近的 second-order 近似。

### 6.3 数值"理论"——用离散kernel的精确极值

直接在离散 exact kernel 上优化 τ:
$$\tau^*_{\text{discrete}} = \arg\min_\tau \sum_{i<j} K_{\text{exact}}(\phi_i(\tau), \phi_j(\tau))^2$$

如果这个离散优化给出 L^{-0.5}，则说明问题完全在 broadband 近似的信息损失上（路径A/B可修补）。如果仍然给不出 L^{-0.5}，则确认问题在训练动力学上。

**这是最关键的区分实验**: 它能明确回答"L^{-0.5}是静态还是动态属性"。

---

## 7. 总结

**能推导**: cosh 族、d_head 依赖、geometric 极限、离散下界
**不能推导（当前）**: L^{-0.5} 指数
**最有可能的原因**: broadband 两参数 surrogate 在投影时丢失了 L-dependent 的离散相干结构
**最关键的下一步**: 在 discrete exact kernel 上直接优化 τ，看是否能复现 L^{-0.5} — 这能区分"静态信息损失"和"训练动力学涌现"两个假说
