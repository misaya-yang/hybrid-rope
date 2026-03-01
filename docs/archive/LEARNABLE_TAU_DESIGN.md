# Learnable τ: 完整设计方案

> **日期**: 2026-03-01
> **状态**: 核心设计文档
> **核心论点**: 理论提供参数化形式（EVQ-Cosh 族），学习找到最优参数（τ*），两者共同构成完整方法

---

## 0. 为什么必须做 Learnable τ

### 痛点
- τ = √(β/α) 取决于数据的距离先验 D(Δ)
- 即使在 3 个数据集上验证了 τ 预测，实际 LLM 训练数据是 **千变万化的混合体**
- 让用户先测 D(Δ) 再算 τ → 工程上不现实
- 手动 sweep τ → 跟调参没区别，理论白推了

### 解决方案
- **理论给出函数族** φ_k(τ) = 1 - (1/τ)arcsinh((1-u_k)sinh τ)
- **SGD 找到最优参数** τ* 在训练中自动收敛
- **零手动调参**：drop-in replacement for standard RoPE

### 论文叙事升级

```
之前: theory → formula → sweep → "我们推导了正确的函数形式"
现在: theory → formula → learnable → "我们推导了正确的函数族，训练自动找到最优点"
```

**这不是稀释理论贡献，而是强化它**——理论的价值体现在：
1. 把 N/2 维搜索空间（每个频率独立调）压缩到 **1维**（只学τ）
2. 函数形式保证了 boundary anchoring（端点不动）
3. 收敛的 τ_learned 应该接近 τ*_theory → **理论的可检验预测**

### Learnable τ 的双重身份

**身份 1: 工程方法** — Drop-in replacement, 零调参，任何数据集自动适配

**身份 2: Landscape Probe（理论验证探测器）** — 如果 τ 从任意初始值收敛到 Algorithm 1 预测的 τ*_theory，这证明变分泛函 J[ρ] 忠实地捕获了真实 loss landscape 的极值结构。

> **论文措辞**: "We learn τ end-to-end as a landscape probe: convergence to the theory-predicted value validates that the variational functional J[ρ] faithfully captures the structure of the empirical loss."

### vs DAPE: 流形约束的优势

| | DAPE (NeurIPS 2024) | EVQ-Cosh (Ours) |
|---|---|---|
| 搜索空间 | R^{d/2}（~32-64 维） | **1D**（τ 流形） |
| 理论约束 | 无（纯数据驱动） | cosh 族 = Galerkin 投影下的最优参数化 |
| 保证 | 无 | boundary anchoring, τ→0 恢复 geometric |
| 过拟合风险 | 高维搜索 + 高频梯度弱 → Hessian 病态 | 1D 搜索 → 单盆地拟凸 |
| 可预测性 | 不可预测最终频率 | Algorithm 1 先验预测 τ* |

> **论文措辞**: "While DAPE learns d/2 independent frequencies in an unconstrained space, our variational theory compresses the search to a single parameter τ on a physically-derived manifold. The manifold constraint reduces dimensionality by d/2× while providing boundary anchoring, smooth geometric recovery, and a priori τ* prediction."

---

## 1. 数学基础

### 1.1 前向计算

给定 τ > 0:

```
φ_k(τ) = 1 - (1/τ) · arcsinh((1 - u_k) · sinh(τ))
ω_k(τ) = b^{-φ_k(τ)}
```

其中 u_k = (k + 0.5) / N, k = 0, ..., N-1 是中点量化坐标。

### 1.2 梯度解析式

```
∂φ_k/∂τ = (1/τ²) · arcsinh(A_k · sinh τ)
         - (1/τ) · A_k · cosh τ / √(1 + A_k² · sinh²τ)
```

其中 A_k = 1 - u_k。

**关键性质——边界锚定**:
- k = 0 (最高频): A_0 ≈ 1, u_0 ≈ 0 → φ_0 ≈ 0 → ∂φ_0/∂τ = 0
- k = N-1 (最低频): A_{N-1} ≈ 0, u_{N-1} ≈ 1 → φ_{N-1} ≈ 1 → ∂φ_0/∂τ = 0
- 中间频率: ∂φ_k/∂τ ≠ 0, τ 主要重分配中间频段密度

**物理含义**: 学习 τ 永远不会破坏频谱的最高和最低端点。最高频保持位置分辨力，最低频保持长距离可达性。变动集中在中间频段的密度分配。

### 1.3 τ → 0 的 Taylor 稳定性

当 τ → 0:
```
∂φ_k/∂τ → -(A_k · (1 - A_k²) / 3) · τ + O(τ³)
```

梯度自动趋零 → 在 geometric 附近不会发生剧烈跳变 → 安全初始化。

### 1.4 频率梯度的完整链

训练 loss L 对 τ 的梯度：
```
∂L/∂τ = Σ_k ∂L/∂ω_k · ∂ω_k/∂φ_k · ∂φ_k/∂τ
       = Σ_k ∂L/∂ω_k · (-ln b · ω_k) · ∂φ_k/∂τ
```

每一步都是 closed-form，不需要任何近似。

---

## 2. 参数化选择

### 2.1 为什么不直接学 τ

直接用 τ 作 nn.Parameter 的问题：
- τ 必须 > 0，但 SGD 可能把它推到负数
- τ = 0 是奇点（梯度公式需要 L'Hôpital）

### 2.2 Softplus 参数化 (推荐)

```python
τ = softplus(ψ) = log(1 + exp(ψ))
```

- ψ ∈ ℝ 无约束
- τ ∈ (0, ∞) 自动保证
- 梯度: ∂τ/∂ψ = sigmoid(ψ), 处处非零
- softplus 在 ψ >> 0 时 ≈ ψ (近线性), 在 ψ << 0 时 ≈ exp(ψ) (指数衰减)

### 2.3 初始化策略

| 策略 | ψ_init | τ_init | 适用场景 |
|------|--------|--------|---------|
| Neutral | softplus⁻¹(1.0) ≈ 0.541 | 1.0 | 不知道数据特性时 |
| Geometric | softplus⁻¹(0.01) ≈ -4.60 | ≈ 0.01 | 希望从 geometric 开始慢慢偏离 |
| Theory-guided | softplus⁻¹(τ*) | τ* | 已有 D(Δ) 估计时 |

**推荐**: τ_init = 1.0 (neutral)。理由：
- 不偏向任何先验假设
- 如果数据是 uniform prior，τ 会自己趋向 0（即 geometric）
- 如果数据是 power-law，τ 会增长到适当值
- 避免"我们精心选了初始值"的审稿攻击面

---

## 3. 训练配方 (Recipe)

### 3.1 核心参数

```yaml
# τ 学习率相关
tau_lr_multiplier: 10.0     # τ 的学习率 = 主 LR × multiplier
tau_warmup_steps: 0         # 不需要 warmup（边界锚定已保证稳定）
tau_weight_decay: 0.0       # τ 不做 weight decay（它不是过拟合源）

# 参数化
tau_init: 1.0               # softplus(ψ_init) = 1.0
tau_parameterization: "softplus"

# 监控
tau_log_interval: 100       # 每 100 步记录 τ 值
```

### 3.2 为什么 τ_lr_multiplier = 10×

τ 是一个标量，控制 N/2 个频率的联合分配。它的梯度是 N/2 个分量的平均效应，数值上比单个权重矩阵元素小得多。乘以 10× 补偿这个 scale 差异。

具体倍率需要在 50M 实验中验证。候选值: {1×, 5×, 10×, 20×}

### 3.3 为什么不需要 warmup

传统观点：频率大幅变化会破坏训练早期的学习。

但 EVQ-Cosh 的 **边界锚定** 保证了：
- 最高频和最低频不动（梯度为零）
- 中间频率的变化是连续的（no discontinuity）
- τ 的初始梯度 ∝ τ（当 τ ≈ 1 时梯度温和）

因此不需要额外 warmup。如果实验中发现训练初期不稳定，可以加 100-step warmup 作为 fallback。

### 3.4 Optimizer 集成

```python
# 两组参数，不同学习率
param_groups = [
    {"params": model.non_tau_params(), "lr": base_lr},
    {"params": [model.rope.raw_tau], "lr": base_lr * tau_lr_multiplier, "weight_decay": 0.0},
]
optimizer = AdamW(param_groups)
```

---

## 4. 实验设计（5090, 50M + 125M）

### 4.1 实验矩阵

**Phase 1: τ LR 校准 (50M, FineWeb-Edu, 3 runs)**

| Run | Method | τ_init | τ_lr_mult | 目的 |
|-----|--------|--------|-----------|------|
| A1 | Geometric (baseline) | — | — | 基线 |
| A2 | Learnable EVQ | 1.0 | 5× | 低 LR |
| A3 | Learnable EVQ | 1.0 | 10× | 中 LR |
| A4 | Learnable EVQ | 1.0 | 20× | 高 LR |

产出: 选定最优 τ_lr_mult, 观察 τ 收敛轨迹

**Phase 2: 固定 vs 可学习 (50M, FineWeb-Edu, 关键对照)**

| Run | Method | 详情 | 目的 |
|-----|--------|------|------|
| B1 | Geometric | τ = 0 (标准 RoPE) | 基线 |
| B2 | Fixed EVQ τ=0.5 | 固定 | sweep 点 1 |
| B3 | Fixed EVQ τ=1.0 | 固定 | sweep 点 2 |
| B4 | Fixed EVQ τ=1.5 | 固定 | sweep 点 3 |
| B5 | Fixed EVQ τ=2.0 | 固定 | sweep 点 4 |
| B6 | Learnable EVQ τ_init=1.0 | 学习 | **核心实验** |

产出:
- τ_learned 最终值 vs 最优 fixed τ
- PPL 对比: learnable 是否 ≈ 最优 fixed
- τ 轨迹图: 收敛行为

**Phase 3: 跨数据集 + 125M 确认 (双数据集, 双规模)**

| Run | Model | Dataset | Method | 目的 |
|-----|-------|---------|--------|------|
| C1 | 50M | TinyStories | Learnable EVQ | τ 数据依赖性 |
| C2 | 125M | FineWeb-Edu | Learnable EVQ (seed 42) | 规模不变性 |
| C3 | 125M | FineWeb-Edu | Learnable EVQ (seed 137) | 复现性 |
| C4 | 125M | FineWeb-Edu | Geometric | 125M 基线 |

产出:
- TinyStories τ_learned vs FineWeb-Edu τ_learned (应该不同)
- 50M τ_learned ≈ 125M τ_learned on 同数据集 (规模不变性, 理论预测)
- 双 seed 复现性

### 4.2 总 GPU 时间估算

| Phase | Runs | Per-run time (50M) | Per-run time (125M) | Total |
|-------|------|-------------------|--------------------| ------|
| Phase 1 | 4 × 50M | ~40 min | — | ~2.5 hr |
| Phase 2 | 6 × 50M | ~40 min | — | ~4 hr |
| Phase 3 | 1 × 50M + 3 × 125M | ~40 min | ~1.5 hr | ~5 hr |
| **总计** | **14 runs** | | | **~11.5 hr** |

在 5090 上约 1.5 天完成全部实验。

### 4.3 成功标准 (论文可发表的最低要求)

1. **τ 收敛**: τ 在训练过程中单调收敛或振荡收敛到稳定值 (std < 0.05 over last 20% steps)
2. **τ 匹配 sweep**: |τ_learned - τ_best_fixed| < 0.3
3. **PPL 匹配**: learnable EVQ 的 PPL 在最优 fixed EVQ 的 ±1% 以内
4. **数据依赖**: τ_learned(FineWeb) ≠ τ_learned(TinyStories) 差异 > 0.2
5. **规模不变**: |τ_learned(50M) - τ_learned(125M)| < 0.2 on same dataset
6. **超越 geometric**: learnable EVQ PPL < geometric PPL (至少 -3% @16K)

### 4.4 如果失败的降级方案

如果 τ 学习不收敛或不匹配 sweep:
- 检查 LR multiplier (可能需要更大/更小)
- 加 warmup (前 500 步冻结 τ)
- 换 log 参数化: τ = exp(ψ) (更强正性约束)
- **最坏情况**: 仍然可以展示 Algorithm 1 (D→τ estimation) + fixed sweep 结果，learnable 退为 Appendix 的理论贡献

---

## 5. 关键图表设计

### Figure 1: τ 收敛轨迹

```
x-axis: Training step
y-axis: τ value
Lines:
  - FineWeb-Edu 50M (blue solid)
  - FineWeb-Edu 125M seed42 (blue dashed)
  - FineWeb-Edu 125M seed137 (blue dotted)
  - TinyStories 50M (red solid)
  - horizontal dashed line: τ_best_fixed for each dataset
Annotation: "τ converges to dataset-dependent optimum regardless of model size"
```

这一张图就能同时展示：收敛性、数据依赖性、规模不变性、与 sweep 一致性。

### Figure 2: PPL vs τ 的函数 (validation)

```
x-axis: τ value
y-axis: PPL @ 16K context
Points: Fixed τ sweep results (discrete markers)
Star: Learnable τ converged point
Curve: Smooth interpolation of sweep
Annotation: "Learned τ falls near the sweep optimum"
```

### Table: 主结果表

| Model | Dataset | Method | τ | PPL@4K | PPL@16K | Δ% vs Geo |
|-------|---------|--------|---|--------|---------|-----------|
| 50M | FineWeb-Edu | Geometric | 0 | x | x | — |
| 50M | FineWeb-Edu | Fixed EVQ (best) | τ_sweep | x | x | -y% |
| 50M | FineWeb-Edu | **Learnable EVQ** | τ_learned | x | x | -z% |
| 125M | FineWeb-Edu | Geometric | 0 | x | x | — |
| 125M | FineWeb-Edu | **Learnable EVQ** | τ_learned | x | x | -w% |
| 50M | TinyStories | **Learnable EVQ** | τ_learned | x | x | -v% |

---

## 6. 论文叙事（完整版）

### §1 Introduction
"RoPE 频率分配是一个变分逆问题。我们推导出最优频率族 EVQ-Cosh，由单参数 τ 控制。τ 依赖于数据的距离先验 D(Δ)，理论上给出 τ = √(β/α)。我们进一步证明 τ 可以在训练中直接学习——理论给出函数族，梯度下降自动找到最优点。"

### §3 Theory
- 变分问题 → ODE → cosh 通解 → EVQ 公式
- τ 的物理意义 = √(β/α)
- Waterbed 不等式: 约束了任何 τ 下的性能边界

### §4 Method: Learnable EVQ-Cosh
- Algorithm 1: EVQ 频率的参数化形式
- τ 的 softplus 参数化: τ = log(1 + exp(ψ)), ψ ∈ ℝ
- 梯度解析式 + 边界锚定性质
- Theorem: Geometric RoPE 是 τ→0 特例 (理论保证安全初始化)

### §5 Experiments
- 50M/125M on FineWeb-Edu + TinyStories
- τ 收敛轨迹 → 数据依赖性 + 规模不变性
- PPL 对比: learnable ≈ best sweep >> geometric
- Waterbed 验证 (如果做 8B LoRA downstream tasks)

### §6 Discussion
- τ_learned ≈ τ*_theory → 理论预测能力的验证
- 可学习 τ 的工程价值: 零手动调参
- B* = ln D 统一 ALiBi/T5 (future work)

---

## 7. 与之前方案的对比

| 维度 | 之前 (Fixed τ + sweep) | 现在 (Learnable τ) |
|------|----------------------|-------------------|
| 审稿人攻击面 | "τ=1.5 怎么选的" | "训练自动选的" |
| 工程价值 | "先测 D(Δ) 再算 τ" | "直接 drop-in" |
| 理论价值 | "理论推出了正确的函数形式" | "理论推出了函数族 + 学习验证了预测" |
| 实验复杂度 | τ sweep (5 runs) | 单次训练 (1 run) |
| 可复现性 | 依赖 sweep 设置 | 完全自动 |
| NeurIPS 吸引力 | "Understanding paper" | "Method + theory paper" |

---

## 8. 潜在审稿人问题及回答

### Q: "为什么不直接让每个频率独立学习？"

A: N/2 个独立可学习频率有两个致命问题：
1. 搜索空间从 1D (τ) 爆炸到 N/2 维 (~32-64D)，需要大量数据才能收敛
2. 失去了理论约束——无法保证 boundary anchoring, 无法保证 waterbed 结构
3. 我们的方法用 1 个参数控制整个频谱，是理论指导下的降维

### Q: "Learnable τ 跟 DAPE 的可学习频率有什么区别？"

A: DAPE 让每个频率独立可学习（无理论约束），相当于 N/2 维搜索。我们用变分理论把搜索空间压缩到 1D，理论保证了函数形式的最优性，学习只找最优 τ。这是 "理论 + 实践" vs "纯数据驱动" 的根本区别。

### Q: "如果 τ 学完了跟理论预测不一致呢？"

A: 可能的原因：(a) broadband 近似的有限 base 残差 (~11%)，(b) 离散化效应，(c) loss landscape 非凸性。但即使有小偏差，只要 learnable EVQ 持续优于 geometric 且 τ_learned 在合理范围内，方法仍然有效。偏差本身也是有价值的诊断——它量化了 broadband 近似的实际误差。

### Q: "一个标量 τ 真的够用吗？"

A: 在单一数据集上，是的。τ 控制的是 "全局耦合 vs 局部脊" 的平衡，这是一个全局属性。但混合数据集可能需要 per-domain τ 或 per-layer τ——这是自然的 future work 扩展。当前论文用全局 τ 已经足以展示方法论。

---

## 9. 代码文件清单

| 文件 | 内容 | 状态 |
|------|------|------|
| `rope/learnable_evq.py` | LearnableEVQRoPE nn.Module | 待写 |
| `rope/schedules.py` | 固定 EVQ 调度（现有） | 需对齐 u_k 中点量化 |
| `experiments/train_learnable_tau.py` | 训练脚本 with τ logging | 待写 |
| `experiments/plot_tau_trajectory.py` | τ 收敛轨迹可视化 | 待写 |
| `experiments/compute_tau_from_data.py` | Algorithm 1: D(Δ) → τ* | 待写 |

---

---

## 10. 实验结果 (128-tok PE Quality Test)

### 10.1 核心发现

| 结果 | 值 | 状态 |
|------|-----|------|
| τ_learned (3-seed mean) | **1.141 ± 0.003** | ✅ 高度一致 |
| τ_sweep optimal | **1.5** | ✅ 稳定 |
| EVQ learnable vs Geometric @8K | **-14.1%** | ✅ 显著 |
| EVQ fixed τ=1.5 vs Geometric @8K | **-18.3%** | ✅ 显著 |
| EVQ learnable (1p) vs DAPE (32p) @8K | **-3.1%** | ✅ 核心 claim |
| EVQ fixed vs DAPE @2K | **-13.9%** | ✅ 核心 claim |
| Algorithm 1 τ* prediction | 40.96 (残差 35.6%) | ❌ 失败 |

### 10.2 τ_learned vs τ_sweep gap 的解释

τ_learned = 1.14 优化的是训练 loss (PPL@128)，τ_sweep = 1.5 优化的是外推 PPL@8K。
两个目标不完全一致。τ_learned 的方向是正确的（从 1.0 向 1.5 移动了 0.14），
但因为训练目标只看 128 token 内的 loss，τ 没有动力继续增大。

**论文措辞**: "The learned value τ=1.14 represents the in-distribution optimum;
the sweep optimum τ=1.5 additionally accounts for extrapolation quality.
The gap (0.36) quantifies the divergence between training and deployment objectives."

### 10.3 Algorithm 1 失败的后续处理

Algorithm 1 的 Hilbert-Schmidt 投影在实际数据上残差 35-49%，无法给出可靠的 τ* 预测。
论文中 Algorithm 1 的角色从"实用工具"降级为"理论联系"：

- **正文**: 保留 Algorithm 1 的数学推导（展示 D(Δ) → τ 的理论映射）
- **正文**: 加 Remark 讨论投影残差和实际局限性
- **Appendix**: 报告具体残差数值，讨论改进方向

**实际 τ 获取方案**:
1. 默认 τ=1.5（跨协议/跨数据集一致）
2. Learnable τ init=1.0（自适应，给出 reasonable 值）
3. 3-point mini-sweep（15 min GPU，最精确）

---

## 11. 版本历史

- v1 (2026-03-01): 初始设计，综合 Gemini/GPT-5.2Pro/Claude 分析
- v2 (2026-03-01): 128-tok 实验结果整合，Algorithm 1 失败处理
