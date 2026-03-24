# τ* 的训练 Regime 依赖：SFT 与 LoRA 下的 Softmax Transport 理论

> **日期**: 2026-03-24
> **前置**: TAU_STIFFNESS_DERIVATION_2026-03-24.md (χ² stiffness 推导)
> **结论**: Pre-trained 模型的权重-频率耦合创造了额外的 τ² stiffness，导致 LoRA regime 下存在 **相变**——当 r < K = d_head/2 时 EVQ 不可行 (τ* ≈ 0)，完美解释了 LLaMA-8B LoRA r=16 的 PPL 77.1 灾难。

---

## 1. 统一框架

所有 regime 共享同一个 Softmax Transport 目标：

$$\mathcal{F}(\tau) = S_{\text{total}}(\tau) - \lambda \cdot U(\tau, L)$$

**不同 regime 改变的是 Stiffness，不是 Utility**。

| Regime | S_total(τ) | τ* |
|--------|-----------|-----|
| Pre-train (from scratch) | S_χ²(τ) | d_head/√L |
| SFT (full param, T steps) | S_χ²(τ) + Λ·e^{-Tησ}·τ²/d | → d_head/√L as T→∞ |
| LoRA (rank r) | S_χ²(τ) + (1-r/K)·Λ₀·τ²/d | **相变**: r < K → τ*≈0 |

Utility 不变：$U(\tau, L) = \frac{M}{L}\int_0^1 q(Lb^{-\phi})\rho_\tau(\phi)\,d\phi$。

---

## 2. 权重-频率耦合 Stiffness

### 2.1 物理机制

Pre-trained 权重 W₀ 经过数十亿梯度更新后，与 geometric RoPE 的频率结构**强耦合**：

- Q, K 投影矩阵的每个 2D block (对应频率通道 k) 已学会以特定幅度调制位置信号
- Attention 模式的长/短程平衡已校准于 geometric 频率间距
- LayerNorm 和残差流的统计量已适应 geometric 的 logit 分布

当频率分配从 geometric (τ=0) 切换到 EVQ (τ>0)：

1. **通道位移**: 中间通道位移 |Δφ_mid| = τ²/16。在 τ=1.414 时 ≈ 8 个通道（K=64 中）
2. **Logit 失配**: 冻结权重产生的 attention logit 偏移 δℓ(Δ) ∝ τ·(frozen channel count)^{1/2}
3. **Attention 崩塌**: softmax(ℓ + δℓ) 的概率分布被严重扭曲

### 2.2 数学形式

冻结权重的 logit 失配方差：

$$\text{Var}(\delta\ell) \propto \frac{K_{\text{frozen}}}{d_{\text{head}}} \cdot \tau^2$$

这贡献一个**二次** stiffness：

$$S_{\text{frozen}}(\tau;\, r) = \Lambda_0 \cdot \left(1 - \frac{r}{K}\right) \cdot \frac{\tau^2}{d_{\text{head}}}$$

其中 Λ₀ 是权重-频率耦合强度（对 well-trained 模型，Λ₀ >> 1）。

**关键**: S_frozen ∝ τ² 而 S_χ² ∝ τ⁴（小 τ 时）。当 Λ₀ 足够大时，S_frozen 在所有有意义的 τ 处**主导** S_χ²。

---

## 3. LoRA 相变

### 3.1 相变机制

LoRA (rank r) 限制权重更新为 W = W₀ + BA (B ∈ ℝ^{d×r}, A ∈ ℝ^{r×d})。每个频率通道占据 Q/K 的一个 2D block，适应需要 ≥ 1 个 LoRA 方向。

**适应容量**：

| 量 | 值 (LLaMA-8B) |
|---|---|
| d_head | 128 |
| K = d_head/2 (频率通道数) | 64 |
| LoRA rank r | 16 |
| 可适应通道 min(r, K) | 16 |
| **适应比** r/K | **25%** |
| **冻结比** 1 - r/K | **75%** |

### 3.2 LoRA 总 Stiffness

$$S_{\text{LoRA}}(\tau;\, r) = \underbrace{S_{\chi^2}(\tau)}_{\sim\, \tau^4/M} + \underbrace{\Lambda_0\!\left(1-\frac{r}{K}\right)\!\frac{\tau^2}{d_{\text{head}}}}_{\text{frozen-weight penalty}}$$

平衡方程 ∂F/∂τ = 0：

$$\underbrace{S_{\chi^2}'(\tau)}_{\sim\, 4\tau^3/M} + \underbrace{\frac{2\Lambda_0(1-r/K)\tau}{d_{\text{head}}}}_{\text{dominant when } \Lambda_0 \gg 1} = \lambda \cdot \underbrace{\frac{2\tau Q_1 M}{L}}_{\text{utility gradient}}$$

**当 S_frozen 项主导时**，τ 在等式两边同时被消除：

$$\frac{2\Lambda_0(1-r/K)}{d_{\text{head}}} \lessgtr \frac{2\lambda Q_1 M}{L}$$

这是一个**不依赖于 τ 的不等式**：

- **左 > 右**: stiffness 永远主导 → τ* = 0（EVQ 不可行）
- **左 < 右**: utility 主导 → τ* → ∞（需要 S_χ² 的 τ⁴ 项来约束）

**这就是相变**: 不存在中间态的 "最优 τ"。系统要么在 τ=0 相（EVQ 无用），要么在 τ>0 相（EVQ 有效）。

### 3.3 临界 rank

相变条件 (左 = 右)：

$$r_c = K \cdot \left(1 - \frac{\lambda Q_1 M d_{\text{head}}}{\Lambda_0 L}\right)$$

对 well-trained 模型（Λ₀ >> λQ₁Md/L）：

$$r_c \approx K = \frac{d_{\text{head}}}{2}$$

**预测**: LoRA 需要 r ≈ K = d_head/2 才能启用 EVQ。

| 模型 | d_head | K | r_c ≈ K | LoRA r=16 |
|------|--------|---|---------|-----------|
| LLaMA-8B | 128 | 64 | ~64 | r/K = 0.25 ❌ |
| LLaMA-1B | 64 | 32 | ~32 | r/K = 0.50 ⚠️ |
| 50M (实验) | 64 | 32 | ~32 | from-scratch ✅ |

### 3.4 LLaMA-8B LoRA r=16 的灾难解释

| 量 | 值 |
|---|---|
| τ applied | d_head/√L = 128/√8192 = 1.414 |
| 通道位移 | τ²/16 × K = 8.0 个通道 |
| 可适应通道 | 16 (LoRA r=16) |
| **不可适应的位移通道** | **48 个冻结通道被错误位移** |
| PPL | 11.8 → **77.1** (ΔCE = 1.88 nats) |

48 个冻结通道仍在"期待"它们的 geometric 频率，但 EVQ 已经把频率移走了。这创造了系统性的 attention logit 失配，导致 PPL 爆炸。

---

## 4. SFT (全参数微调)

### 4.1 耦合衰减公式

全参数 SFT 可以在所有 d_head 维度上适应，但需要足够的梯度步数。耦合 stiffness 随训练指数衰减：

$$S_{\text{coupling}}(T) = \Lambda_0 \cdot \frac{\tau^2}{d_{\text{head}}} \cdot e^{-T\eta\sigma}$$

其中 T = SFT 步数，η = 学习率，σ = 频率敏感方向的 Hessian 谱隙。

### 4.2 SFT τ* 公式

$$\boxed{\tau^*_{\text{SFT}}(T) \approx \frac{\tau^*_{\text{pretrain}}}{\sqrt{1 + \Lambda \cdot e^{-T\eta\sigma}}}}$$

其中 Λ = Λ₀·d_head/(M·S_χ²(τ*)) 是无量纲耦合强度。

**极限行为**:
- T → 0: τ* → τ*_pretrain / √(1+Λ) → 0 (如果 Λ >> 1)
- T → ∞: τ* → τ*_pretrain = d_head/√L

**物理含义**: 全参数 SFT 可以渐近恢复 pre-training 的 τ*，但所需步数取决于 Λ。对 well-trained 大模型 (Λ >> 1)，可能需要接近原始 pre-training 的步数才能在频率敏感子空间中完成适应。

### 4.3 Continued pretraining 的成功解释

750M 实验 (continued 2K→4K, PPL@16K -45.9%) 成功的原因：

**base 模型已是 EVQ (τ=1.5) from-scratch 训练的**。所以：
- S_coupling(τ ≈ 1.5) ≈ 0（权重已适应 EVQ 频率）
- 继续训练只需微调 content 方向，不需要重新适应频率
- 等效于 Λ ≈ 0 的 regime → τ* ≈ d_head/√L 正常工作

---

## 5. LoRA rank sweep 预测

$$\tau^*_{\text{LoRA}}(r) \approx \begin{cases} 0 & r < K/4 \text{ (EVQ 不可行)} \\ \tau^*_{\text{pretrain}} \cdot r/K & K/4 \leq r < K/2 \text{ (边际)} \\ \tau^*_{\text{pretrain}} \cdot \sqrt{r/K} & K/2 \leq r < K \text{ (部分 EVQ)} \\ \tau^*_{\text{pretrain}} & r \geq K \text{ (完全 EVQ)} \end{cases}$$

具体预测 (LLaMA-8B, d_head=128, K=64, L=8192):

| r | r/K | 预测 τ* | 预期效果 |
|---|-----|--------|---------|
| 4 | 0.06 | 0 | ❌ EVQ 不可行 |
| 8 | 0.12 | 0 | ❌ EVQ 不可行 |
| **16** | **0.25** | **~0** | **❌ 实测 PPL 77.1** |
| 32 | 0.50 | ~1.0 | ⚠️ 边际，需验证 |
| 48 | 0.75 | ~1.2 | ✓ 部分 EVQ |
| **64** | **1.00** | **1.41** | **✅ 完全 EVQ (未测)** |

**可测试预测**: r=64 的 LoRA 应该能支撑 EVQ τ≈1.4，PPL 恢复正常。

---

## 6. 论文建议

### 6.1 Proposition

> **Proposition (Regime-Dependent Temperature Selection):**
>
> 将 softmax transport 目标推广到包含权重-频率耦合的 regime：
>
> $$\mathcal{F}(\tau;\, r, T) = S_{\chi^2}(\tau) + \underbrace{\Lambda_0\!\left(1-\frac{\min(r,K)}{K}\right)\!\frac{\tau^2}{d_{\text{head}}} \cdot e^{-T\eta\sigma}}_{\text{weight-frequency coupling}} - \lambda U(\tau, L)$$
>
> 则：
> - **Pre-training** (Λ₀=0): τ* = d_head/√L [T3, 99 runs]
> - **LoRA** (r < K, T 有限): 当 Λ₀(1-r/K) >> λQ₁Md/L 时 τ* → 0（**相变**，EVQ 不可行）
> - **SFT** (r=d, T 步): τ* → d_head/√L 当 T → ∞

### 6.2 对论文 Limitations 部分的贡献

> **Limitation (fine-tuning regime):** EVQ-cosh requires from-scratch training or very early injection. Applying EVQ to a pretrained model via LoRA (rank r < d_head/2) creates a weight-frequency coupling penalty that overwhelms the transport utility, driving the optimal τ to zero. This is a direct consequence of the fact that pretrained Q/K projections have been calibrated for geometric RoPE's frequency structure; the low-rank LoRA subspace cannot provide sufficient degrees of freedom (≥K = d_head/2) to remap this structure. A practical workaround is to pretrain with EVQ from initialization and then fine-tune normally.

### 6.3 可测试预测（强化论文的 falsifiable predictions）

1. **LoRA r=64**: 对 LLaMA-8B 应能支撑 EVQ (τ≈1.4)，PPL 不爆炸
2. **LoRA rank sweep**: PPL 的 r-依赖应有一个在 r ≈ K/2 附近的相变
3. **Full SFT, long training**: 足够多步的全参数 SFT 应能恢复 EVQ 效果

---

## 附录: 为什么 S_frozen ∝ τ² 而非 τ⁴

S_χ²(τ) ∝ τ⁴ 来自 cosh 密度的对称性：ρ_τ = 1 + τ²η + O(τ⁴)，所以 (ρ-1)² ∝ τ⁴。

S_frozen(τ) ∝ τ² 来自不同机制：冻结权重的 attention logit 偏移与频率变化量**线性**相关：

$$\delta\ell(\Delta) = \sum_{k \in \text{frozen}} \frac{q_k k_k}{\sqrt{d}} [\cos(\omega_k^{EVQ}\Delta) - \cos(\omega_k^{GEO}\Delta)]$$

Taylor 展开：cos(ω_k^{EVQ}Δ) - cos(ω_k^{GEO}Δ) ≈ -sin(ω_k Δ) · δω_k · Δ

其中 δω_k ∝ τ（频率变化与 τ 线性相关）。所以 δℓ ∝ τ，Var(δℓ) ∝ τ²。

这个 **τ² vs τ⁴ 的阶数差异** 是相变的根本原因：冻结权重的惩罚在小 τ 处增长更快，使得在 τ→0 处 stiffness 梯度就已经主导 utility 梯度，不存在内部极值点。

---

*本文档将 softmax transport 框架从 pre-training 推广到 SFT/LoRA regime，解释了 pretrained model EVQ 失败的根本机制（权重-频率耦合 + LoRA rank 相变），并给出可测试的预测。*
