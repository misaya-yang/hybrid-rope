# τ* 精确推导：从 Softmax Transport 到不截断的平衡方程

> **日期**: 2026-03-23
> **作者**: Cowork session (基于 GPT-5 softmax transport 理论框架)
> **状态**: 完整推导 + 数值验证

---

## 0. 核心结论

**精确结果**（无截断）：softmax transport 变分目标的完整平衡方程给出

$$\tau_* = \tau_0 \cdot \sqrt{\frac{u(\tau_*)}{c(\tau_*)}}, \quad \tau_0 = \sqrt{45\lambda Q_1}\,\frac{M}{\sqrt{L}}$$

其中 $c(\tau)$ 和 $u(\tau)$ 是精确已知的修正因子（见 §3）。

**关键发现**：
1. 前导阶 τ₀ ∝ d_head/√L 严格成立（L-exponent = -0.500，gap = 0.003）
2. 完整方程的全局 L-exponent = **-0.626**（非 -0.5）
3. 局部指数在 L=4096 处为 -0.589，在 L=256 处为 -0.608
4. **实验中的 -0.5 可能是静态 -0.63 被训练动力学修正后的综合效应**

---

## 1. 变分目标（精确形式）

### 1.1 定义

$$\mathcal{F}(\tau) = \frac{C(\tau)}{M} - \lambda\frac{M}{L}\,U(\tau, L)$$

其中

$$C(\tau) = \frac{1}{2}\int_0^1 (\rho_\tau(\phi) - 1)^2\,d\phi \quad\text{(stiffness)}$$

$$U(\tau, L) = \int_0^1 q(Lb^{-\phi})\,\rho_\tau(\phi)\,d\phi \quad\text{(transport utility)}$$

### 1.2 基本构件

**EVQ-cosh 密度**：
$$\rho_\tau(\phi) = \frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau}$$

**单通道 transport energy**：
$$q(x) = \frac{1}{2} + \frac{\sin 2x}{4x} - \left(\frac{\sin x}{x}\right)^2$$

渐近行为：$q(x) \to x^4/45$ ($x \ll 1$)，$q(x) \to 1/2$ ($x \gg 1$)。

**参数**：$M = d_{\text{head}}/2$，$b = \theta_{\max}/\theta_{\min}$。

---

## 2. C(τ) 精确闭式（已证明 + 验证）

### 2.1 推导

$$\int_0^1 \rho_\tau^2\,d\phi = \frac{\tau^2}{\sinh^2\tau}\int_0^1 \cosh^2(\tau(1-\phi))\,d\phi$$

令 $x = \tau(1-\phi)$：

$$= \frac{\tau}{\sinh^2\tau}\int_0^\tau \cosh^2(x)\,dx = \frac{\tau}{\sinh^2\tau}\left[\frac{\tau}{2} + \frac{\sinh 2\tau}{4}\right]$$

$$= \frac{\tau^2}{2\sinh^2\tau} + \frac{\tau\coth\tau}{2}$$

因此

$$\boxed{C(\tau) = \frac{\tau^2}{4\sinh^2\tau} + \frac{\tau\coth\tau}{4} - \frac{1}{2}}$$

**数值验证**（n=100000 积分点）：

| τ | C_closed | C_numerical | rel_err |
|---|---------|-------------|---------|
| 0.01 | 1.1×10⁻¹⁰ | 1.1×10⁻¹⁰ | 1.4×10⁻⁷ |
| 1.0 | 0.00927424 | 0.00927424 | 2.8×10⁻¹⁰ |
| 5.0 | 0.75124861 | 0.75124861 | 1.1×10⁻⁹ |
| 10.0 | 2.00000022 | 2.00000022 | 3.8×10⁻⁹ |

**全部 ✓，误差 < 10⁻⁷**

### 2.2 C'(τ) 闭式

$$C'(\tau) = \frac{\tau}{4\sinh^2\tau}(1 - 2\tau\coth\tau) + \frac{\coth\tau}{4}$$

验证：与有限差分的吻合精度 < 4×10⁻¹¹。

### 2.3 渐近展开

$$C(\tau) = \frac{\tau^4}{90} + O(\tau^6), \quad C'(\tau) = \frac{2\tau^3}{45} + O(\tau^5)$$

---

## 3. 平衡方程与修正因子

### 3.1 精确平衡方程

$$\mathcal{F}'(\tau_*) = 0 \quad\Longleftrightarrow\quad \frac{C'(\tau_*)}{M} = \lambda\frac{M}{L}\,U'(\tau_*, L)$$

即

$$\boxed{C'(\tau_*) = \frac{\lambda M^2}{L}\,U'(\tau_*, L)}$$

这是关于 τ* 的**隐式方程**，通过二分法求解（不做任何截断）。

### 3.2 定义修正因子

将 C' 和 U' 分解为"前导阶 × 修正"：

$$c(\tau) \equiv \frac{C'(\tau)}{2\tau^3/45} \quad\text{(stiffness 修正)}$$

$$u(\tau, L) \equiv \frac{U'(\tau, L)}{2Q_1(L)\,\tau} \quad\text{(utility 修正)}$$

其中 $Q_1(L) = \int_0^1 \eta(\phi)\,q(Lb^{-\phi})\,d\phi \approx 0.030$。

两个修正因子在 τ → 0 时均趋向 1。代入平衡方程：

$$c(\tau_*)\cdot\frac{2\tau_*^3}{45} = \frac{\lambda M^2}{L}\cdot u(\tau_*, L)\cdot 2Q_1\tau_*$$

$$\tau_*^2 = \frac{45\lambda Q_1 M^2}{L}\cdot\frac{u(\tau_*)}{c(\tau_*)}$$

$$\boxed{\tau_* = \tau_0\cdot\sqrt{\frac{u(\tau_*)}{c(\tau_*)}}, \quad \tau_0 = \sqrt{45\lambda Q_1}\,\frac{M}{\sqrt{L}}}$$

### 3.3 修正因子的数值

c(τ) 是纯解析的（用 C' 闭式）：

| τ | c(τ) | u(τ, L=2048) | u/c |
|---|------|-------------|-----|
| 0.01 | 1.000 | 1.000 | 1.000 |
| 0.5 | 0.932 | 0.949 | 1.018 |
| 1.0 | 0.763 | 0.818 | 1.072 |
| 1.414 | 0.598 | 0.680 | 1.136 |
| 2.0 | 0.393 | 0.489 | 1.245 |
| 3.0 | 0.178 | 0.251 | 1.410 |
| 5.0 | 0.045 | 0.061 | 1.364 |

**关键观察**：
- **c(τ) 衰减更快**：stiffness 的高阶修正比 utility 的更强
- **u/c > 1**：实际 τ* 总是**大于**前导阶 τ₀
- 在 τ ≈ 1.4 (L=2048) 处，u/c ≈ 1.14，修正仅 7%

---

## 4. 数值结果

### 4.1 λ 校准

在 L=2048 处匹配 τ* = d_head/√L = 1.414：

$$\lambda = \frac{C'(1.414) \times L}{M^2 \times U'(1.414, L)} = 2.564$$

### 4.2 全局精确解

| L | τ*(exact) | τ₀(leading) | d/√L | τ*/τ₀ | τ*/(d/√L) |
|---|-----------|-------------|------|--------|-----------|
| 128 | 8.151 | 5.270 | 5.657 | 1.547 | 1.441 |
| 256 | 5.496 | 3.810 | 4.000 | 1.443 | 1.374 |
| 512 | 3.508 | 2.714 | 2.828 | 1.293 | 1.240 |
| 1024 | 2.189 | 1.909 | 2.000 | 1.147 | 1.094 |
| 2048 | 1.414 | 1.327 | 1.414 | 1.066 | 1.000 |
| 4096 | 0.938 | 0.911 | 1.000 | 1.029 | 0.938 |
| 8192 | 0.626 | 0.618 | 0.707 | 1.013 | 0.885 |

**全局 L-exponent = -0.626**（经 bisection 和 golden section 两种方法交叉验证）

### 4.3 局部指数

| L | τ* | local exponent d ln τ*/d ln L |
|---|-----|------------------------------|
| 256 | 5.496 | -0.608 |
| 512 | 3.508 | -0.664 |
| 1024 | 2.189 | -0.655 |
| 2048 | 1.414 | -0.612 |
| 4096 | 0.938 | -0.589 |

局部指数在 -0.59 到 -0.66 之间变化，不是常数 → 不是严格的幂律。

### 4.4 d_head 标度

| d_head | M | τ*(exact) | d/√L | ratio |
|--------|---|-----------|------|-------|
| 32 | 16 | 0.674 | 0.707 | 0.954 |
| 64 | 32 | 1.414 | 1.414 | 1.000 |
| 128 | 64 | 3.172 | 2.828 | 1.121 |
| 256 | 128 | 5.838 | 5.657 | 1.032 |

**d_head exponent = 1.051**（接近 1.0）

---

## 5. 理论解读

### 5.1 为什么前导阶给出 -0.5 而完整方程给出 -0.63？

前导阶分析（τ → 0）：
- Stiffness cost: C(τ) ≈ τ⁴/90 → C'(τ) ≈ 2τ³/45
- Transport benefit: (M/L)U(τ,L) ≈ (M/L)(Q₀ + Q₁τ²) → (M/L)·2Q₁τ

τ⁴ vs τ²/L 的平衡**严格**给出 τ ∝ 1/√L。

但在 τ ≈ 1-2 的实际工作点：
- C'(τ) 增长**慢于** τ³（因为 c(τ) < 1 且递减）
- U'(τ,L) 也衰减（因为 u(τ,L) < 1）
- 但 **c(τ) 衰减比 u(τ) 更快**
- 因此 τ* > τ₀，且在大 L（小 τ）端接近 τ₀，小 L（大 τ）端远离 τ₀
- 这使得 log-log 斜率比 -0.5 更陡

### 5.2 差距 0.13 的来源

$$\text{L-exponent} = -0.5 \underbrace{- 0.13}_{\text{高阶 stiffness 修正}}$$

精确地说，偏差来自 c(τ)/u(τ) 比值随 τ（因而随 L）的变化：

$$\frac{d\ln\tau_*}{d\ln L} = -\frac{1}{2} + \frac{1}{2}\frac{d\ln[u(\tau_*)/c(\tau_*)]}{d\ln L}$$

第二项在 τ ≈ 1 时约为 -0.13。

### 5.3 与实验 L^{-0.5} 的调和

两种可能的解释：

**(A) 训练动力学修正**：SGD 的隐式正则化提供了一个反向修正，将 -0.63 拉回 -0.5。这与之前的分解 L^{-0.5} = L^{-0.085}(kernel) × L^{-0.087}(self-consistent) × L^{-0.328}(dynamic) 一致——动力学部分贡献了 -0.33 ≈ (-0.5) - (-0.17)。在 softmax transport 框架下，动力学只需贡献 +0.13。

**(B) λ 的 L 依赖**：如果 λ 本身弱依赖于 L，例如 λ ∝ L^{0.26}，则可精确补偿。物理上，这可能来自 attention temperature 或 positional susceptibility 随序列长度的变化。

---

## 6. 实用公式

### 6.1 精确形式（推荐用于实现）

$$\tau_* = \text{bisect}\left\{C'(\tau) = \frac{\lambda M^2}{L}\,U'(\tau, L)\right\}$$

其中 λ ≈ 2.56 由单个 (L, d_head) 参考点校准。

### 6.2 近似形式

**幂律近似**（max error 3%，对 d_head=64）：
$$\tau_* \approx 0.94 \times \left(\frac{d_{\text{head}}}{\sqrt{L}}\right)^{1.25}$$

**自洽迭代**（任意精度）：
$$\tau_{n+1} = \tau_0\sqrt{\frac{u(\tau_n)}{c(\tau_n)}}, \quad \tau_0 = \sqrt{45\lambda Q_1}\,\frac{M}{\sqrt{L}}$$

通常 3-5 次迭代收敛。

### 6.3 简化推荐

对于绝大多数实际场景：

$$\boxed{\tau_* \approx \frac{d_{\text{head}}}{\sqrt{L}}}$$

前导阶公式在 L ∈ [1024, 8192] 范围内误差 < 12%，且给出正确的标度趋势。精确数值可通过 §6.1 的隐式方程获得。

---

## 7. 完整推导链

```
Softmax Jacobian J₀ = (1/L)Π₀
    ↓ (1/L factor)
单通道 transport energy: u_L(ω) = q(ωL)/L
    ↓ (对 EVQ 频率求和)
Transport utility: U(τ,L) = ∫q(Lb^{-φ})ρ_τ(φ)dφ
    ↓
变分目标: F(τ) = C(τ)/M - λ(M/L)U(τ,L)
    ↓ (对 τ 求导)
平衡方程: C'(τ*) = λ(M²/L)U'(τ*,L)
    ↓ (分解修正因子)
精确公式: τ* = τ₀√(u/c),  τ₀ = √(45λQ₁)·M/√L
    ↓ (前导阶 c→1, u→1)
Scaling law: τ* ∝ d_head/√L
```

---

## 8. 遗留问题

1. **λ 的自然导出**：λ = 2.56 是校准值；需从 attention 机制的几何中导出
2. **0.13 的精确来源**：训练动力学 vs λ(L) 依赖，需要实验区分
3. **MLA/DiT 推广**：c(τ) 和 u(τ) 的表达式不依赖于 attention 类型，但 q(x) 可能不同
4. **Q₁ 的弱 L 依赖**：Q₁ 从 0.032(L=512) 下降到 0.027(L=8192)，引入 log 修正

---

## 附录 A：验证代码

- `scripts/tau_exact_derivation.py` — 完整精确推导 + 数值验证
- `scripts/tau_refined_formula.py` — 多种近似公式的系统比较
- `scripts/verify_softmax_transport.py` — 前导阶验证
- `scripts/verify_softmax_v2.py` — 系数匹配分析

所有代码纯 numpy，无外部依赖，运行 < 1 秒。
