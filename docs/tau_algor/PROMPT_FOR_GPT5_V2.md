# τ* = d_head/√L 的第一性原理推导：终极问题

> **给 GPT-5 Pro 的深度推导任务**
> **前置要求**: 请先完整阅读本仓库的以下文件（按优先级排序）：
> 1. `docs/tau_algor/TAU_SOFTMAX_TRANSPORT_THEORY_2026-03-23.md` — 最接近成功的推导路径
> 2. `docs/tau_algor/TAU_FIRST_PRINCIPLES_ANALYSIS_2026-03-22.md` — 所有失败路径的精确诊断
> 3. `docs/tau_algor/TAU_STATIC_VS_DYNAMIC_EXPERIMENT_2026-03-22.md` — 数值判决实验
> 4. `docs/tau_algor/TAU_SCALING_DERIVATION.md` — 完整理论推导链
> 5. `docs/tau_algor/TAU_HABITABLE_ZONE.md` — 离散化截断与宜居带
> 6. `docs/tau_algor/unified_tau_star_theory_v2.md` — 跨架构统一理论框架
> 7. `paper/sections/03_theory.tex` — 论文正式理论部分
> 8. `paper/appendix/a1_proofs.tex` — 完整证明
> 9. `docs/theory/THEORY_IRONCLAD.md` — 理论铁桶级参考

---

## 0. 问题的重要性——为什么这不只是一个系数问题

这不是一个"锦上添花"的理论打磨。**这是整篇论文的天花板决定因素。**

当前所有实验（领先 baseline 1%-81%）都**不在最优 τ 下运行**——我们用的是一个经验公式 τ* = d_head/√L，但这个公式在大 L 时给出明显偏小的 τ（例如 L=8192, d_head=64 时公式给 0.71，实际甜点在 ~1.4）。这意味着：

1. **当前所有正面结果是 EVQ 真实能力的下界**——如果能找到精确的 τ*，实验会更强
2. **LoRA 微调的灾难性失败**（LLaMA-8B LoRA PPL 从 11.8 爆炸到 77.1）可能是因为用了错误的 τ——fine-tuning regime 的最优 τ 可能完全不同于 pre-training
3. 如果能建立一个**统一的 τ* 理论**，覆盖 pre-training、SFT、LoRA、不同架构（MHA/MLA/DiT），那这篇论文从 poster 变成 oral 甚至 best paper

---

## 1. 已确立的理论（不需要重新推导）

### 1.1 变分推导链（Theorem 级）

```
D(Δ) ∝ 1/Δ  →  K(φ₁,φ₂) = ∫D(Δ)cos(ω₁Δ)cos(ω₂Δ)dΔ
→  K ≈ αδ(φ₁-φ₂) + β·min(φ₁,φ₂)   [broadband surrogate, R²>0.99]
→  Euler-Lagrange: ρ'' - τ²ρ = γb^{-2φ}
→  Pure-tether solution: ρ_τ(φ) = τ·cosh(τ(1-φ))/sinh(τ)   [归一化]
→  CDF inversion: φ_k(τ) = 1 - (1/τ)arcsinh((1-u_k)sinh(τ))
```

关键点：
- 只有**一个近似**（broadband surrogate），其余全部精确
- cosh 族不是 ansatz，是 Euler-Lagrange ODE 的唯一稳态解
- Geometric RoPE 是 τ→0 的退化极限（Theorem 2）
- τ = √(β/α) 是 surrogate 参数的精确关系

### 1.2 已知的 α, β scaling

数值拟合（d_head=64, b=500K, L ∈ [128, 4096]）：
- α ≈ 1/d_head，几乎不依赖 L（α ~ L^{-0.051}）
- β = O(1)，弱 L 依赖（β ~ L^{-0.221}）
- 因此 τ_surr = √(β/α) ~ L^{-0.085}

**缺口：** 经验值 τ* ~ L^{-0.5}，surrogate 只给出 L^{-0.085}。差了 L^{-0.415}。

### 1.3 离散化下界（Proposition 级）

τ_floor = 4/√K：当 τ < 4/√K 时，中间通道位移不到一个网格间距，EVQ 等价于 geometric。

### 1.4 经验验证

τ* = d_head/√L 在 99 次训练实验中 R² > 0.99（27 配置 × 3+ seeds）。

---

## 2. 已失败的路径（请不要重复）

### 2.1 所有 pre-softmax 静态目标（7+ 种）

| 目标 | 最优 τ | L-exponent |
|------|--------|-----------|
| L2 off-diagonal collision | ~12 | +0.01 (无 L 依赖) |
| Weighted L2 collision | ~9 | -0.03 |
| Mutual coherence | ~13 | +0.00 |
| Condition number | ~13 | +0.00 |
| Effective rank | ~12 | +0.09 (反向) |
| Position Gram log-det | ~? | +1.55 (反向) |
| Cross-entropy -Σlog(1-R²) | ~? | -0.22 (不够) |

**结论：所有 pre-softmax kernel 空间的静态目标都给 τ ≈ 10-15，不依赖 L。** 因为在 logit space，一个频率通道的边际收益是 O(1)，与 L 无关。

### 2.2 自洽 surrogate（路径 B）

在 EVQ 实际通道位置（而非均匀网格）处拟合 α*, β*，迭代至收敛：
- 结果：L^{-0.172}（从 -0.085 改善到 -0.172）
- 仍差 0.328 到目标 -0.5
- 在 1/Δ prior 下发散（无稳定不动点）

### 2.3 三参数扩展 surrogate（路径 A）

K_app = αδ + β·min + η·Φ(L)：引入第三个 L-dependent basis function。
- 问题：Euler-Lagrange ODE 不再是常系数，失去 cosh 闭式解
- 放弃原因：会摧毁 EVQ 的实用性

### 2.4 训练动力学修正（路径 C）

添加梯度方差惩罚 λ∫ρ²V(φ,L)dφ：
- V 在死区边界处只有 ln(L) 量级，不是 L² 量级
- 修正后 τ_eff ∝ 1/√(ln L)——对数衰减，不是幂律

### 2.5 维度分析

K_ind^max ∝ √(K×L) 给出 τ ∝ √(L/K)——**方向相反**！

---

## 3. 最接近成功的路径：Softmax Transport Theory

### 3.1 核心洞察（来自你上次的推导）

**关键：从 pre-softmax kernel 换到 post-softmax attention transport。**

在 softmax 后的概率空间中，Jacobian J = diag(p) - pp^T 在 diffuse baseline p₀ = 1/L 下引入一个额外的 1/L 因子。这正是之前所有静态目标缺失的 L 依赖。

### 3.2 变分目标

$$\mathcal{F}(\tau) = \underbrace{\frac{1}{2M}\int_0^1 (\rho_\tau - 1)^2 d\phi}_{\text{stiffness}} - \underbrace{\lambda \frac{M}{L}\int_0^1 q(Lb^{-\phi})\rho_\tau(\phi) d\phi}_{\text{softmax transport utility}}$$

其中单通道 transport energy：
$$q(x) = \frac{1}{2} + \frac{\sin 2x}{4x} - \left(\frac{\sin x}{x}\right)^2$$
渐近：q → 0 (dead channel, ωL≪1), q → 1/2 (active channel, ωL≫1)

### 3.3 小 τ 展开的结果

ρ_τ = 1 + τ²η + O(τ⁴)，η(φ) = (1-φ)²/2 - 1/6

- Stiffness: ∫(ρ-1)² ≈ τ⁴/90M（τ⁴ 阶，因为 ρ-1 ~ τ²）
- Utility 增量: ~ τ²Q₁(L)，Q₁ = ∫η·q·dφ ≈ 0.030 = Θ(1)
- 平衡：4τ³/(90M) = 2λ(M/L)Q₁τ → **τ² = 45λQ₁ M²/L → τ ∝ M/√L = d_head/(2√L)** ✓

**小 τ 极限下：L-exponent = -0.497（目标 -0.5，误差 0.003）** ✓

### 3.4 问题：实际工作点 τ~1.5 时展开不准

在 λ 调到使 τ_opt 匹配实际量级（~1-2）时：

| L | τ_opt | τ* = d/√L | ratio |
|---|-------|-----------|-------|
| 128 | 6.37 | 5.66 | 1.13 |
| 256 | 4.01 | 4.00 | 1.00 |
| 512 | 2.45 | 2.83 | 0.87 |
| 1024 | 1.57 | 2.00 | 0.79 |
| 2048 | 1.04 | 1.41 | 0.74 |
| 4096 | 0.70 | 1.00 | 0.70 |

**L-exponent 变成 -0.626**（比 -0.5 更陡）。

原因：τ~1.5 时 cosh(1.5) ≈ 2.35，ρ_τ - 1 不再是 τ² 的小量。高阶项（τ⁶, τ⁸...）使得 stiffness 增长比 τ⁴ 更慢，utility 增长比 τ² 更慢，平衡点偏移。

---

## 4. 核心推导任务

### 任务 A（最高优先级）：用精确 cosh 积分重做 Softmax Transport 优化

**不做小 τ 展开。** 直接用归一化的 cosh 密度：

$$\rho_\tau(\phi) = \frac{\tau \cosh(\tau(1-\phi))}{\sinh(\tau)}$$

Stiffness 可以精确计算：
$$S(\tau) = \frac{1}{2M}\int_0^1 \left(\frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau} - 1\right)^2 d\phi$$

展开：
$$S(\tau) = \frac{1}{2M}\left[\frac{\tau^2}{\sinh^2\tau}\left(\frac{1}{2} + \frac{\sinh 2\tau}{4\tau}\right) - 2\frac{\tau}{\sinh\tau}\cdot\frac{\sinh\tau}{\tau} + 1\right]$$

$$= \frac{1}{2M}\left[\frac{\tau}{2\sinh^2\tau}\left(\tau + \frac{\sinh 2\tau}{2}\right) - 1\right]$$

请精确化简这个表达式，然后数值计算 Utility 项：
$$U(\tau, L) = \frac{M}{L}\int_0^1 q(Lb^{-\phi}) \cdot \frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau} \, d\phi$$

然后求解：
$$\frac{dS}{d\tau} = \lambda \frac{dU}{d\tau}$$

**关键问题：** 在这个精确框架下，τ_opt(L) 的 L-scaling 是多少？是 -0.5、-0.626、还是其他值？

如果精确积分给出的 exponent 比 -0.626 更接近 -0.5，那说明 small-τ 展开引入的误差恰好歪曲了 L-scaling。如果仍然是 -0.626 左右，那说明 Softmax Transport 框架本身在有限 τ 处有系统偏差，需要额外机制。

### 任务 B：λ 的物理确定

Softmax Transport 目标中的 λ 目前是自由参数。请尝试从以下方向确定其物理值：

1. **Attention entropy 约束**：总 transport 应与 causal attention 的熵预算一致。在 p₀ = 1/L 的 baseline 下，attention 的最大 transport capacity 是 H = ln(L)。λ 可能是 1/ln(L) 量级。

2. **Scale matching**：在 τ=0 处，stiffness = 0，utility > 0。λ 应当使得在 τ*（经验值）处 stiffness ≈ utility 的同量级。用实验值 τ* ≈ 1.5, M=32, L=2048 反解 λ。

3. **自洽条件**：如果 λ 本身依赖于 τ（因为 softmax Jacobian 在非均匀分配下会变化），那 F(τ) 的一阶条件变成一个更复杂的方程。请分析 λ(τ) 是否存在自洽解。

4. **关键猜想**：如果 λ 从 attention 几何中自然导出且不含自由参数，那整个 τ* = d_head/√L 就变成了一个**定理**而非经验律。

### 任务 C：训练范式相关的 τ* 推导

如果 Softmax Transport 框架成立，不同训练范式下的 τ* 应该不同，因为：

1. **Pre-training**：模型从随机初始化学习，所有通道的 gradient signal 从零开始建立。Utility 项的 L 对应实际训练长度 L_train。

2. **SFT（全参数微调）**：模型已有 pre-trained attention patterns。频率分配的变化需要"重写"已有模式。Stiffness 项应该有一个额外的**inertia penalty**——偏离 pre-trained 分配的代价更高。
   - 猜想：τ*_SFT < τ*_pretrain（因为惯性增加了有效 stiffness）
   - 或者：τ*_SFT 需要考虑 L_pretrain 和 L_SFT 的比值

3. **LoRA（低秩适配）**：只有 rank-r 的低秩更新。模型能够适应新频率分配的自由度受限。
   - 猜想：LoRA 的有效 stiffness ∝ d_head/r（rank 越低，适应能力越弱，需要更保守的 τ）
   - 这可能解释 LLaMA-8B LoRA 的灾难性失败——如果 τ 对于 LoRA 的有效 stiffness 来说太激进

4. **MLA 架构**：只有 d_rope 个维度参与 RoPE（其余 d_nope 个是 content-based）。Softmax score 被 positional 和 content 部分联合决定。
   - Utility 项应该按 d_rope/d_qk 缩放（positional 部分的贡献被稀释）
   - 推导：这是否自然给出 τ*_MLA = (d_qk/d_rope) × d_rope/(2√L)？

### 任务 D：寻找全新的理论路径

如果 Softmax Transport 的精确积分仍然不能给出 L^{-0.5}，请考虑以下全新方向：

1. **Fisher Information 路径**：RoPE 频率分配定义了一个关于位置 t 的统计模型。Fisher Information I(t) = E[(∂logp/∂t)²] 关于频率分配有闭式表达。最优频率分配应该最大化 ∫I(t)D(t)dt（加权 Fisher Information）。
   - 在 causal attention 下，I(t) 依赖于 L（因为 softmax 的归一化常数）
   - 这条路径可能自然给出 L 依赖

2. **Mutual Information 路径**：I(position; attention_output) 在给定频率分配 ρ 下的表达式。最大化这个关于 ρ 的变分问题。
   - 关键：MI 在 softmax 后的概率空间中计算，自然包含 1/L 因子

3. **随机矩阵理论**：K×K 的 exact kernel 矩阵 K(φ_i, φ_j) 的谱分布如何随 L 变化？特别是最小特征值的 scaling——这决定了频率分配的"条件"好坏。
   - Marchenko-Pastur 分布可能适用于某个 regime

4. **Neural Tangent Kernel**：Transformer with RoPE 的 NTK 包含频率分配的信息。NTK 的有效 rank 随 τ 和 L 的 scaling 可能直接给出 τ* 的理论值。

5. **Optimal Transport 路径**：将 geometric 分配视为源分布，将"理想分配"视为目标分布。Wasserstein 距离 W₂(ρ_geo, ρ_ideal) 的最小化可能给出不同的变分问题。

---

## 5. 输出要求

### 5.1 对于每条尝试的路径

- 写出完整的数学推导（不要跳步）
- 给出 L-exponent 的解析表达式（如果可能）或数值预测
- 如果失败，精确说明在哪一步失败以及为什么
- 与已知的失败路径对比，确认不是重复

### 5.2 对于成功的路径（如果存在）

- 写出可以直接放入论文的 Proposition/Theorem 陈述
- 给出严格的证明或证明 sketch
- 明确所有假设和近似
- 给出数值验证方案（Python 代码 sketch）
- 分析其对 SFT/LoRA/MLA/DiT 的推广

### 5.3 如果确认 L^{-0.5} 无法从纯静态理论推导

- 给出当前可达的最紧 bound（比 L^{-0.172} 更紧）
- 明确需要什么额外的物理假设才能补上 gap
- 对 Softmax Transport 在有限 τ 下的 L^{-0.626} 结果：分析高阶修正的结构，判断是否存在一个 resummation 方案使其收敛到 -0.5

### 5.4 关于 λ

- 如果能从第一性原理确定 λ，写出完整推导
- 如果 λ 必须是自由参数，分析 τ_opt 对 λ 的敏感性
- 特别关注：λ 是否可能本身依赖于 L？如果 λ ∝ L^a，那 τ_opt 的 L-exponent 会从 -0.5 变成什么？

---

## 6. 背景数据（验证用）

### 经验 τ* 值

| d_head | L | τ* (经验) | d/√L | 来源 |
|--------|---|-----------|------|------|
| 64 | 128 | ~5.7 | 5.66 | Phase 16 sweep |
| 64 | 256 | ~4.0 | 4.00 | Phase 11 |
| 64 | 512 | ~2.8 | 2.83 | Phase 16 |
| 64 | 1024 | ~2.0 | 2.00 | Phase 16 |
| 64 | 2048 | ~1.4 | 1.41 | Phase 17 |
| 32 | 8192 | ~1.4 | 0.35 | MLA (注意：公式预测 0.35 但实际甜点 ~1.4) |

### Softmax Transport 数值验证

小 τ 极限（λ 小）：L-exponent = -0.497 ✓
实际 τ~1-2（λ ≈ 2.95）：L-exponent = -0.626 ✗

### 静态目标 L-exponent 排名

1. 自洽 surrogate: -0.172
2. L2 + μτ² 正则化: -0.173
3. 均匀 surrogate: -0.085
4. 其他所有: ≈ 0

---

## 7. 一句话总结

**我们已经推导出 RoPE 最优频率分配的函数形式（cosh 族，唯一解）和参数方向（τ ∝ √d_head）。唯一缺失的是 L^{-0.5} 指数的第一性原理推导。Softmax Transport 理论在 τ→0 极限下精确给出 -0.497，但在实际工作点 τ~1.5 处偏到 -0.626。请你要么（a）在精确积分下修复这个偏差，要么（b）找到一个全新的变分原理直接给出 -0.5，要么（c）证明 -0.5 不可从任何静态理论推导并给出最紧的 bound。**

**附加高价值目标：如果你的框架成功，请推广到 SFT 和 LoRA regime，预测不同训练范式下的最优 τ*。这将统一我们目前观察到的所有实验结果（包括 LoRA 的灾难性失败）。**
