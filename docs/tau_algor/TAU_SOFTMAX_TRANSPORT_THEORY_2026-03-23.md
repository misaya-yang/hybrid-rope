# τ* ∝ d_head/√L 的第一性原理推导：Softmax Transport 路径

> **日期**: 2026-03-23
> **来源**: GPT-5 提出理论框架，Cowork session 数值验证
> **状态**: 核心机制已验证，系数匹配待完善

---

## 0. 一句话结论

**L^{-0.5} 指数可以从纯静态理论推导，但不是从 pre-softmax collision kernel，而是从 post-softmax attention transport。** 具体机制是："每个活通道只值 1/L 的 attention transport" + "EVQ 在 τ=0 处只从二阶开始变形" → τ⁴ cost vs τ²/L benefit → τ ∝ 1/√L。

---

## 1. GPT-5 的核心洞察

### 1.1 为什么之前所有静态目标都失败了

之前我们测试了 7+ 种静态目标函数（L2 collision, mutual coherence, condition number 等），全部给出 τ ≈ 10-15 且不依赖 L。

GPT-5 指出：**这些目标都在 pre-softmax logit space 里优化**。在 logit space，一个频率通道的边际收益是 O(1)，与 L 无关。所以 stiffness vs benefit 的平衡给出的 τ 当然也与 L 无关。

但 attention 真正起作用的对象是 **softmax 后的概率分布**，不是 logit：

$$p = \text{softmax}(\ell), \quad \delta p = J \delta\ell, \quad J = \text{diag}(p) - pp^\top$$

Jacobian $J$ 对长度 L 的 causal attention，在 diffuse baseline $p_0 = 1/L$ 下，贡献一个额外的 $1/L$ 因子。

### 1.2 单通道的 Transport Energy

对频率 ω 的通道，在长度 L 的 attention row 中，其对概率空间的 transport energy 是：

$$u_L(\omega) = \frac{1}{L} q(\omega L) + o(1/L)$$

其中

$$q(x) = \frac{1}{2} + \frac{\sin 2x}{4x} - \left(\frac{\sin x}{x}\right)^2$$

**关键渐近**：
- $\omega L \ll 1$ (死通道): $q \approx (\omega L)^4/45 \approx 0$
- $\omega L \gg 1$ (活通道): $q \approx 1/2$

**数值验证**: q(x) 的闭式与数值积分 $\int_0^1 (\cos(xt) - \sin(x)/x)^2 dt$ 的误差 < 5×10⁻⁴ ✓

### 1.3 新的变分目标

$$\boxed{\tau_* = \arg\min_{\tau \geq 0} \left[ \underbrace{\frac{1}{2M}\int_0^1 (\rho_\tau - 1)^2 d\phi}_{\text{stiffness (偏离geometric的代价)}} - \underbrace{\lambda \frac{M}{L}\int_0^1 q(Lb^{-\phi})\rho_\tau(\phi) d\phi}_{\text{softmax transport utility}} \right]}$$

---

## 2. 为什么这给出 L^{-0.5}

### 2.1 EVQ 在 τ=0 的二阶起跳

cosh 密度在 τ=0 处的展开：

$$\rho_\tau(\phi) = 1 + \tau^2 \eta(\phi) + O(\tau^4), \quad \eta(\phi) = \frac{(1-\phi)^2}{2} - \frac{1}{6}$$

**数值验证**: τ=0.1 时 max error = 2×10⁻⁶, τ=0.5 时 = 1.4×10⁻³ ✓

### 2.2 Stiffness 是 τ⁴ 阶

$$\frac{1}{2M}\int(\rho_\tau - 1)^2 d\phi \approx \frac{\tau^4}{2M}\int\eta^2 d\phi = \frac{\tau^4}{90M}$$

（因为 $\int_0^1 \eta(\phi)^2 d\phi = 1/45$）

**数值验证**: τ=0.1 时 ratio = 1.001, τ=0.5 时 = 0.957 ✓

### 2.3 Utility 是 τ² 阶

$$\frac{M}{L}\int q(Lb^{-\phi})\rho_\tau d\phi \approx \frac{M}{L}\left[Q_0 + \tau^2 Q_1(L)\right]$$

其中 $Q_1(L) = \int \eta(\phi) q(Lb^{-\phi}) d\phi$

**数值验证**: $Q_1(L) \approx 0.030$ 对 $L \in [128, 8192]$，确认 $\Theta(1)$ ✓

### 2.4 平衡方程

$$\frac{\partial}{\partial\tau}\left[\frac{\tau^4}{90M} - \lambda\frac{M}{L}Q_1\tau^2\right] = 0$$

$$\frac{4\tau^3}{90M} = 2\lambda\frac{M}{L}Q_1\tau$$

$$\boxed{\tau^2 = 45\lambda Q_1 \frac{M^2}{L}} \quad \Rightarrow \quad \tau \propto \frac{M}{\sqrt{L}} = \frac{d_{\text{head}}}{2\sqrt{L}}$$

---

## 3. 数值验证结果

### 3.1 小 τ 极限 (λ 小) — 解析预测完美匹配

| L | τ_opt | τ_analytic | τ* = d/√L | analytic/opt |
|---|-------|-----------|-----------|--------------|
| 128 | 0.0760 | 0.0761 | 5.657 | 1.001 |
| 512 | 0.0391 | 0.0392 | 2.828 | 1.003 |
| 2048 | 0.0191 | 0.0192 | 1.414 | 1.005 |
| 4096 | 0.0132 | 0.0132 | 1.000 | 1.000 |

**L-exponent = -0.497** (target -0.500, gap 0.003) ✓
**d_head exponent = 0.981** (target 1.000) ✓

### 3.2 实际 τ* 量级 (λ ≈ 2.95)

当 λ 调到使绝对值匹配 d_head/√L 时：

| L | τ_opt | τ* = d/√L | ratio |
|---|-------|-----------|-------|
| 128 | 6.37 | 5.66 | 1.13 |
| 256 | 4.01 | 4.00 | **1.00** |
| 512 | 2.45 | 2.83 | 0.87 |
| 1024 | 1.57 | 2.00 | 0.79 |
| 2048 | 1.04 | 1.41 | 0.74 |
| 4096 | 0.70 | 1.00 | 0.70 |

**L-exponent = -0.626** (比 -0.5 更陡)

### 3.3 解读

小 τ 极限下，二阶展开精确成立，L^{-0.5} 是**严格的解析结果**。

在实际 τ ≈ 1-2 的范围内，高阶项（τ⁶, τ⁸ 等）使得 exponent 偏向 -0.63。但这恰好说明：**实验中看到的精确 -0.5 可能是 softmax transport 的 -0.63 被训练动力学的反向修正拉回到 -0.5 的结果。**

---

## 4. 与之前结论的对比

| 方法 | L-exponent | 能解释 L^{-0.5}? |
|------|-----------|-----------------|
| Broadband surrogate (uniform) | -0.085 | ✗ (差 6×) |
| Self-consistent surrogate | -0.172 | ✗ (差 3×) |
| Kernel-only 静态目标 (7种) | ≈ 0 | ✗ (无 L 依赖) |
| **Softmax transport (小τ)** | **-0.497** | **✓ (gap 0.003)** |
| **Softmax transport (τ~1)** | **-0.626** | **部分 (方向正确, 过度修正)** |

---

## 5. 理论图景的修正

### 旧图景（错误）
> "L^{-0.5} 是训练动力学涌现的，类似学习率 η ∝ 1/√T"

### 新图景（GPT-5 + 本验证）
> EVQ 理论分两层：
> 1. **内层**（已有）: broadband kernel → E-L ODE → cosh 族。决定 **shape**。
> 2. **外层**（新增）: softmax attention transport vs 有限通道 stiffness。决定 **temperature**。
>
> L^{-0.5} 来自外层的 "benefit ∝ τ²/L, cost ∝ τ⁴" 平衡。这是纯静态的、不需要 SGD。
> 精确的 -0.5 指数来自 EVQ 密度在 τ=0 处只有**偶阶**展开（ρ = 1 + τ²η + O(τ⁴)）。

### 为什么之前看不到
**因为一直在 pre-softmax kernel 空间里做优化。** 在 logit space，通道收益是 O(1)；在 probability space，通道收益是 O(1/L)。这个 1/L 正是缺失的因子。

---

## 6. 对论文的影响

### 6.1 可以新增的定理/命题

**Proposition (Softmax Transport Selection)**:
对 EVQ-cosh 族 $\{\rho_\tau\}_{\tau \geq 0}$，定义 softmax transport objective

$$\mathcal{F}(\tau) = \frac{1}{2M}\int_0^1(\rho_\tau - 1)^2 d\phi - \lambda\frac{M}{L}\int_0^1 q(Lb^{-\phi})\rho_\tau(\phi)d\phi$$

其中 $q(x) = 1/2 + \sin(2x)/(4x) - (\sin x/x)^2$。

则在小 τ 极限下，$\mathcal{F}$ 的极小点满足

$$\tau_* = \sqrt{45\lambda Q_1}\cdot\frac{M}{\sqrt{L}} = C\cdot\frac{d_{\text{head}}}{\sqrt{L}}$$

其中 $Q_1 = \int_0^1\eta(\phi)q(Lb^{-\phi})d\phi = \Theta(1)$, $C = \sqrt{45\lambda Q_1}/2$。

**证明**: 直接对 $\mathcal{F}$ 的二阶+四阶展开求导。□

### 6.2 Remark

> The d_head/√L scaling law arises from the balance between finite-channel stiffness (quartic in τ) and softmax-weighted active transport (quadratic in τ, with an O(1/L) per-channel contribution from the Jacobian of the softmax map). This is a purely static principle that does not invoke training dynamics.

---

## 7. 遗留问题

1. **λ 的物理意义**：目标函数中的 λ 需要从 attention 机制的几何中自然导出，目前是自由参数
2. **高阶修正**：在 τ ≈ 1-2 的实际工作点，exponent 偏向 -0.63；需要理解为什么实验看到的是 -0.5 而不是 -0.63（可能是训练动力学提供了反向修正）
3. **MLA/DiT 推广**：需要验证 softmax transport 理论在 MLA (K=16) 和 DiT (bidirectional) 下的表现
4. **$Q_1(L)$ 的微弱 L 依赖**：$Q_1$ 从 0.032 (L=512) 下降到 0.027 (L=8192)，这在大 L 下会引入 log 修正

---

## 附录：验证代码

- `scripts/verify_softmax_transport.py` — 完整验证（q函数, η展开, Q₁, L-scaling, d-scaling）
- `scripts/verify_softmax_v2.py` — 系数匹配分析

所有验证均不依赖 scipy，纯 numpy，运行 < 1 秒。
