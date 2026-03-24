# 从第一性原理推导 Stiffness Functional 的 f-divergence 指数 p

> **给 Claude Code / GPT-5 的专项推导任务**
> **前置要求**: 请先完整阅读以下文件（按优先级）：
> 1. `docs/tau_algor/TAU_THEORY_DEEP_ANALYSIS_2026-03-24.md` §12 — χ² stiffness 突破
> 2. `docs/tau_algor/TAU_SOFTMAX_TRANSPORT_THEORY_2026-03-23.md` — Softmax Transport 完整框架
> 3. `paper/sections/03_theory.tex` — 论文正式理论部分
> 4. `scripts/analysis/verify_softmax_transport.py` — 现有验证脚本

---

## 0. 背景：我们已经解决了什么，还差什么

### 已解决
Softmax Transport 变分原理：

$$\mathcal{F}(\tau) = S_p(\tau) - \lambda \cdot U(\tau, L)$$

其中 Utility 来自 post-softmax attention transport（含 1/L 因子），Stiffness 用 f-divergence 家族：

$$S_p(\tau) = \frac{1}{M}\int_0^1 \frac{(\rho_\tau(\phi) - 1)^2}{\rho_\tau(\phi)^p} \, d\phi$$

ρ_τ(φ) = τ·cosh(τ(1-φ))/sinh(τ) 是归一化 cosh 密度。

**核心数值结果**（精确积分，非小 τ 展开）：

| p | L-exponent γ | Gap from -0.5 |
|---|-------------|--------------|
| 0.00 (L²) | -0.626 | 0.126 |
| 0.50 | -0.561 | 0.061 |
| **0.75** | **-0.508** | **0.008** |
| 0.90 | -0.481 | 0.019 |
| 1.00 (χ²) | -0.465 | 0.035 |

**结论**: p ∈ [0.75, 0.80] 精确给出 L^{-0.5}。

### 未解决（本 prompt 的目标）
**p 的值是从哪里来的？** 目前 p 是数值拟合出来的。如果 p 可以从第一性原理推导，理论就从"区间预测 [-0.51, -0.47]"升级为"点预测 -0.5"，即 **定理级别**。

---

## 1. 为什么 p 不是任意的——物理约束

Stiffness functional S_p(τ) 不是数学上可以任意选取的——它必须反映 **频率分配偏离均匀时的真实物理代价**。这个代价来自 Transformer 的 attention 机制：

- **密度偏低区域** (ρ < 1, 高频通道被稀释)：每对相邻 token 都需要高频通道来区分，损失一个高频通道的代价与 1/ρ 成正比——通道越稀疏，每个残存通道承受的负担越大
- **密度偏高区域** (ρ > 1, 低频通道被聚集)：多出的低频通道只带来边际递减的冗余

这个不对称性恰好是 f-divergence 中 1/ρ^p 权重的物理来源。但 **精确的 p 值应该由 softmax attention 的几何决定，而非手动选择**。

---

## 2. 推导方向（5 条路径，请逐一尝试）

### 路径 A（最有希望）：Softmax Fisher Metric 诱导的 Stiffness

**核心思路**：Softmax attention 的概率分布 p = softmax(ℓ) 定义了一个关于频率参数 φ 的统计流形。该流形上的自然度量是 **Fisher Information Metric**。Stiffness 应该是频率密度 ρ(φ) 偏离 uniform 的 Fisher-Rao 距离，而非任意 f-divergence。

**具体推导**：

1. 考虑 causal attention row 的 logit 向量 ℓ ∈ ℝ^L，其中位置 t 的 logit 包含所有频率通道的贡献：

   $$\ell_t = \sum_{k=1}^{K} \cos(\omega_k \Delta_t)$$

   频率分配 {ω_k} 由密度 ρ_τ(φ) 决定。

2. Softmax 映射 σ: ℓ → p = softmax(ℓ) 的 Jacobian 是 J = diag(p) - pp^T。

3. 频率密度从 ρ₀ = 1 (uniform) 变到 ρ_τ 时，logit 空间的变化 δℓ 通过 J 映射到概率空间的变化 δp = J·δℓ。

4. **问题**：在概率单纯形上的 Fisher-Rao 度量下，这个 δp 的长度是多少？

   Fisher-Rao 度量：ds² = Σ (dp_i)² / p_i

   关键：p_i 在 diffuse baseline 下 = 1/L，所以 Fisher metric 的局部结构包含 1/p ∝ L 的因子。

5. 将上述 Fisher-Rao 距离**对所有 attention rows 求平均**，得到关于 ρ_τ 的泛函 S_Fisher(τ)。

6. **预期**：S_Fisher 应该形如 ∫ g(ρ)(ρ-1)² dφ，其中 g(ρ) 由 softmax Jacobian 和 Fisher metric 的交互决定。如果 g(ρ) ∝ 1/ρ^p，那 p 就被确定了。

**具体要回答的问题**：
- Fisher-Rao 距离在 δρ = ρ_τ - 1 的二次近似下是否给出一个明确的 p 值？
- 如果 p 不是常数而是 ρ 的函数 p(ρ)，在 ρ ∈ [1/cosh(τ), cosh(τ)/1] 的范围内有效平均是多少？
- 这个有效 p 是否接近 0.75？

### 路径 B：KL Divergence 在密度空间的投影

**核心思路**：最自然的 stiffness 应该是频率分配 ρ_τ 相对于 uniform ρ₀=1 的 KL divergence。

$$D_{KL}(\rho_\tau \| \rho_0) = \int_0^1 \rho_\tau \ln \rho_\tau \, d\phi$$

但这不是 (ρ-1)² 形式的 f-divergence。**问题**：

1. 对 KL divergence stiffness，dF/dτ = 0 给出的 τ_opt(L) 的 L-exponent 是多少？
2. KL 的二阶 Taylor 展开是 (1/2)∫(ρ-1)²/ρ dφ = (1/2)S_{p=1}(τ)，即 χ²/2。这解释了为什么 χ² (p=1) 是信息论自然选择。但实验给出 p ≈ 0.75 而非 1.0——差距 0.03 的 L-exponent 是否来自 KL 中**三阶及以上项**的贡献？
3. 请数值计算：用**完整 KL**（而非 χ² 近似）作为 stiffness 时的 L-exponent。如果比 χ² 的 -0.465 更接近 -0.5，那 KL 就是正确的 stiffness。

### 路径 C：Attention 有效通道数的变分原理

**核心思路**：频率分配的"好坏"本质上取决于**有效独立通道数**。定义：

$$K_{eff}(\tau, L) = \frac{\left(\sum_k u_k(\omega_k, L)\right)^2}{\sum_k u_k^2(\omega_k, L)}$$

其中 u_k 是通道 k 在 softmax 后的 transport energy。这是一个 participation ratio。

**推导**：
1. K_eff 关于 τ 的变分导数给出一个类似于 f-divergence 的 stiffness，但 p 值由 u_k 的分布决定
2. 如果活通道的 u_k 近似相等（≈ 1/2），而死通道 u_k ≈ 0，那 K_eff 的梯度结构可能自然给出一个特定的 p
3. 请分析 K_eff 的极大化是否等价于 S_p stiffness，如果是，p 的值是什么

### 路径 D：Rényi Divergence 视角

f-divergence 家族 S_p(τ) = ∫(ρ-1)²/ρ^p 可以与 **Rényi divergence** 联系：

$$D_\alpha(\rho \| 1) = \frac{1}{\alpha - 1} \ln \int \rho^\alpha \, d\phi$$

注意 α → 1 时退化为 KL。**问题**：

1. 是否存在一个 Rényi order α* 使得其 Taylor 展开到二阶精确给出 S_{p=0.75}？
2. 如果存在，α* 是否有独立的信息论意义（如 min-entropy, collision entropy 等）？
3. 特别关注 α = 1/2（Hellinger distance 的推广）——它对应什么 p？

### 路径 E：离散通道的有效连续极限

**核心思路**：实际系统有 K=32 或 K=64 个离散通道。连续极限 (K→∞) 中的 L² stiffness 可能在有限 K 下被修正。

1. 对 K 个离散通道，stiffness 实际是：
   $$S_{discrete} = \frac{1}{K} \sum_{k=1}^{K} f(\rho_k)$$
   其中 ρ_k = ρ_τ(φ_k) 是离散网格上的密度。

2. 当 ρ_k 的分布不均匀时，离散求和与连续积分之间有**Euler-Maclaurin 修正项**，这些修正项依赖于 ρ' 和 ρ''。

3. **问题**：在 K=32 的典型配置下，离散修正是否等效于将 p 从 0 (L²) 移到 ~0.75？

---

## 3. 验证标准

对于任何声称推导出 p 值的路径，必须满足：

### 3.1 数值一致性
- 用推导出的 p 值代入 S_p(τ)，在 L ∈ [128, 4096] 上优化 τ，拟合 L-exponent
- 要求 |γ - (-0.5)| < 0.02

### 3.2 自洽性
- 推导过程不能使用 L^{-0.5} 这个结果作为输入
- p 值必须只依赖于 attention 机制的几何/信息论性质

### 3.3 可预测性
- 同一个推导应该能预测：当 d_head 改变时 p 是否改变？当 K 改变时呢？
- 如果 p = p(K)，那 K=16 (MLA) 和 K=64 (大模型) 的预测是什么？

### 3.4 写出 Proposition
如果成功，写出可直接插入论文的 Proposition：

> **Proposition (Stiffness Functional Selection):**
>
> 在 softmax attention 机制下，频率分配 ρ 偏离 uniform 的自然代价泛函由 [你的推导] 唯一确定为 S_{p*}(τ) = (1/M)∫(ρ-1)²/ρ^{p*} dφ，其中 p* = [具体值]。
>
> 结合 Softmax Transport Selection (§12.6)，这给出：
>
> τ* = C · d_head / L^{1/2}
>
> 其中 C = [明确表达式]，无自由参数。

---

## 4. 已知约束（用于排除错误方向）

1. **p < 0 不可能**：p < 0 意味着高密度区域被加权惩罚，这与物理直觉矛盾（高密度 = 低频冗余 = 温和代价）
2. **p > 1.5 不可能**：p > 1.5 给出 |γ| > 0.1 的 gap，且 S_p 在 ρ→0 时发散太快
3. **p 不应显著依赖于 τ**：因为 stiffness 度量的是"偏离代价"，不应随被度量的对象自身变化
4. **p 应当在 [0.5, 1.0] 区间**：物理直觉（χ² 型不对称惩罚）和数值约束共同限制

---

## 5. 如果所有路径失败

如果五条路径都无法给出精确的 p，请：

1. **证明 p 不可从纯 attention 几何确定**——即 stiffness functional 的选择本质上包含一个来自训练动力学的自由度
2. **给出 p 的最紧 bound**：基于信息几何，p ∈ [p_min, p_max] 的理论约束
3. **分析 0.03 gap 的物理来源**：χ² (p=1) 给出 -0.465，精确 -0.5 需要 p=0.75。差距 0.03 的 L-exponent 在实验精度 ±0.02 内可能不可区分——请对 99 次实验的 bootstrap confidence interval 做分析，判断 p=1 (γ=-0.465) 是否在实验误差内被接受

---

## 6. 参考数值（验证用）

### χ² Stiffness 闭式
$$S_{\chi^2}(\tau) = \frac{1}{M}\left[\frac{\sinh(\tau)\cdot\arctan(\sinh(\tau))}{\tau^2} - 1\right]$$

### L² Stiffness 闭式
$$S_{L^2}(\tau) = \frac{1}{2M}\left[\frac{\tau(2\tau + \sinh 2\tau)}{4\sinh^2\tau} - 1\right]$$

### Utility 项
$$U(\tau, L) = \frac{M}{L}\int_0^1 q(Lb^{-\phi}) \cdot \frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau} \, d\phi$$

其中 q(x) = 1/2 + sin(2x)/(4x) - (sin(x)/x)²，b = 500000。

### cosh 密度
$$\rho_\tau(\phi) = \frac{\tau \cosh(\tau(1-\phi))}{\sinh(\tau)}$$

归一化：∫₀¹ ρ_τ dφ = 1 ✓

### 标准配置
- d_head = 64, K = 32 (= d_head/2), M = K = 32
- b = 500000 (base frequency)
- L ∈ {128, 256, 512, 1024, 2048, 4096}
- 实验 L-exponent = -0.500 ± 0.02 (99 runs, R² > 0.99)

---

*一句话总结：我们已经知道 f-divergence stiffness S_p 中 p ∈ [0.75, 0.80] 精确给出 L^{-0.5}。请从 softmax attention 的信息几何中推导出 p 的精确值，将 τ* = d_head/√L 从经验律升级为定理。*
