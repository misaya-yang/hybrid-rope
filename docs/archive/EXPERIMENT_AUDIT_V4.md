# 深度审核：实验困境诊断与破局方案 (v4)

> **日期**: 2026-03-01
> **审核者**: Claude Opus 4.6
> **状态**: 关键决策文档——GPU 实验前必读

---

## 一、问题诊断：为什么 Learnable τ 在 2K 训练中不收敛？

### 1.1 实验观察（Sanity Check, 50M, 15M tokens, FineWeb-Edu）

| τ_init | τ_final | 方向 | 诊断 |
|--------|---------|------|------|
| 0.01   | 0.0004  | 继续缩小 → 卡在 softplus 死区 | softplus(ψ≈-8) 梯度 ≈ 0 |
| 1.0    | 0.92    | 微调 | 几乎不动 |
| 2.0    | 1.73    | 微调 | 几乎不动 |

**核心矛盾**：三个初始值收敛到三个不同终值。Loss landscape 在 τ∈[0.6, 1.7] 几乎完全平坦。

### 1.2 根因分析：为什么 landscape 是平的？

**答案：2K 训练上下文太长，模型权重吸收了 PE 差异。**

推理链：

1. **模型容量过剩**：125M 参数模型处理 2048 token 上下文时，注意力权重 W_Q, W_K, W_V 有充足的学习能力。给定任何 reasonable 的频率分配（τ∈[0.5, 2.0]），模型都能通过调整权重矩阵来补偿 PE 的差异。

2. **梯度信号被淹没**：τ 的梯度 = Σ_k (∂L/∂ω_k)·(-ln b·ω_k)·(∂φ_k/∂τ)。其中 ∂L/∂ω_k 是每个频率对 loss 的影响。当模型权重已经适应了当前频率时，∂L/∂ω_k 很小 → τ 的梯度也很小。

3. **训练内 PPL 差异极小**：PPL@2K 在 τ∈[0, 2.0] 范围内仅差 3.29-4.21（50M 模型），不到 2% 相对差异。优化器没有足够的 signal 来区分不同的 τ。

4. **外推 PPL 的差异只在训练外可见**：τ=1.5 的优势（-10% ~ -18% @16K）只在 8x 外推时体现，但**训练过程中模型从不"看到"16K**，所以这个 signal 不会反向传播到 τ。

### 1.3 类比理解

想象你在一个黑暗房间里，靠回声定位寻找出口。如果房间只有 3 米（训练 2K），你的回声不管怎么发都能清楚地反射回来——频率分配无关紧要。但如果房间有 30 米（外推 16K），高频回声衰减殆尽，你必须把更多能量分配给低频——这时 τ 才有差异化价值。

**但你在 3 米房间里训练时，永远感受不到 30 米房间的需求。这就是 τ 梯度为零的根本原因。**

---

## 二、DAPE 如何做到外推的？关键发现

### 2.1 DAPE 的实验设置

| 参数 | DAPE 的选择 | 我们当前的选择 |
|------|-----------|-------------|
| **训练序列长度** | **128 tokens** | 2048 tokens |
| 测试序列长度 | 最长 8192 | 最长 16384 |
| 外推比 | **64×** | 8× |
| 模型规模 | 125M | 50M-125M |
| 可学习参数数量 | d/2 ≈ 32-64 个频率 | 1 个（τ） |

### 2.2 为什么 128 token 训练能让频率学习有效？

**关键洞察：在极短上下文训练中，PE 对 loss 的影响被放大。**

1. **PE 的相对影响力与上下文长度成反比**：
   - 128 tokens：每个注意力头只有 128×128 = 16384 个位置对。PE 的旋转角度直接决定了哪些 token 能有效注意到哪些 token。模型权重无法绕过 PE 的限制。
   - 2048 tokens：4M 个位置对。模型有足够的冗余来适应任何 reasonable PE。

2. **频率的边际效应在短上下文中更大**：
   - 128 tokens 时，一个频率 ω_k 的变化直接影响所有 128 个位置的 attention pattern
   - 这意味着 ∂L/∂ω_k 更大 → τ 的聚合梯度更大 → 学习信号更强

3. **DAPE 的 d/2 独立参数 vs 我们的 1 个 τ**：
   - DAPE 的每个频率独立学习，所以即使某些频率的梯度弱，其他频率的强梯度仍能推动学习
   - 我们的 τ 是所有频率梯度的加权和，弱信号被聚合后更弱

### 2.3 DAPE 外推的实质

DAPE 不是在测试"模型在长上下文上表现如何"，而是在测试：

> **哪种 PE 让模型对未见过的位置最鲁棒？**

128→8192 的 64× 外推意味着 97% 的测试位置在训练中从未出现过。这是一个**纯粹的 PE 泛化测试**——模型的权重只见过 [0, 127] 的位置，PE 的数学结构决定了 [128, 8191] 的位置编码是否仍然有意义。

**这恰好是 EVQ-Cosh 理论最有优势的场景**：频率的数学结构（cosh 族）保证了外推时频率间距的合理性，而 DAPE 的 ad-hoc 学习无法保证这一点。

---

## 三、破局方案：双轨实验策略

### 方案 A：DAPE 复刻实验（纯 PE 质量测试）

**核心思路**：在 DAPE 自己的地盘上击败它。

```
训练: 128 tokens, 125M 模型, FineWeb-Edu
测试: 128 / 256 / 512 / 1024 / 2048 / 4096 / 8192 tokens

Methods:
1. Geometric RoPE (baseline)
2. EVQ-Cosh τ=1.0 (fixed)
3. EVQ-Cosh τ=1.5 (fixed)
4. EVQ-Cosh learnable τ (init=1.0)
5. DAPE-style: d/2 independent learnable frequencies (复现 DAPE)
```

**为什么这能解决我们的问题**：

1. **τ 会收敛**：128 token 训练下，PE 的梯度信号极强。τ 的 landscape 不再平坦。
2. **直接与 DAPE 对比**：在同样的实验协议下，如果 1 参数的 EVQ 打平或击败 d/2 参数的 DAPE，这是理论降维能力的直接验证。
3. **外推是天然的**：128→8192 就是 64× 外推，不需要任何特殊训练。

**理论预测**：
- EVQ learnable τ 应该收敛到某个 τ*
- EVQ (1 param) 应该 ≈ DAPE (32 params) 的 PPL，因为 cosh 流形已经接近最优
- EVQ 在最极端的外推（8192+）可能优于 DAPE，因为 cosh 的数学结构提供了更好的外推保证

**GPU 成本**：128 token 训练极快。5 runs × ~20 min = ~2 小时。

### 方案 B：上下文扩展实验（实际应用场景）

**核心思路**：模拟真实 LLM 的"预训练→上下文扩展"工作流。

```
Phase 1: 预训练 (geometric RoPE)
  - 125M 模型, 2K 上下文, 100M tokens, FineWeb-Edu
  - 输出: checkpoint_base.pt

Phase 2: 上下文扩展 (不同方法)
  - 从 checkpoint_base.pt 继续训练
  - 上下文扩展到 8K, 再训练 10-20M tokens

  Methods:
  1. Geometric (直接用 base 500000 → base 变大? 或者不变)
  2. PI (Position Interpolation): 频率 /= 4
  3. YaRN: PI + 高频保护
  4. EVQ-Cosh fixed τ=1.5
  5. EVQ-Cosh learnable τ (init=1.0)

Phase 3: 评估
  - 训练内: PPL@2K, PPL@4K, PPL@8K
  - 外推: PPL@16K, PPL@32K
```

**为什么这能解决我们的问题**：

1. **τ 在 8K continue-train 时有梯度**：模型需要处理 8K 上下文，频率分配直接影响 4K-8K 范围的注意力质量 → τ 有信号
2. **直接与 PI/YaRN 对比**：这是实践中最常见的场景，审稿人关心的核心问题
3. **利用已有理论**：EVQ 的 cosh 族在 context extension 时给出理论最优频率再分配

**理论预测**：
- EVQ learnable 应该在 PPL@8K（训练内）上 ≈ 最优 fixed τ
- EVQ 在 PPL@16K（2× 外推）上应该优于 PI/YaRN
- τ_learned 应该 > 训练前（因为需要更多高频来覆盖更长上下文）

**GPU 成本**：1 次预训练（~1.5h）+ 5 次 continue-train（~30min each）= ~4 小时。

### 推荐策略：A 优先，B 补充

**方案 A 是核心**，原因：
1. 成本最低（~2h GPU），失败风险最小
2. 直接对标 DAPE（NeurIPS 2024 accepted），结果有说服力
3. 如果 τ 在 128 token 训练中收敛，**三重验证叙事恢复**
4. 128→8192 的外推比任何其他方案都更极端，展示 EVQ 的外推能力

**方案 B 是加分项**，原因：
1. 更贴近实际应用场景（审稿人可能问"这在实践中怎么用"）
2. 与 PI/YaRN 的直接对比增强论文实验的完整性
3. 但不是必须的——DAPE 没做这个也中了 NeurIPS

---

## 四、方案 A 的理论审计

### 4.1 EVQ-Cosh 在 128 token 训练下的理论保证

**Q: 128 token 训练时 broadband 分解还成立吗？**

broadband 近似要求 b·L >> 1，其中 L 是最大距离。L=128 时：
- b=500000: b·L = 6.4×10⁷ >> 1 ✓
- b=10000: b·L = 1.28×10⁶ >> 1 ✓

结论：broadband 分解在 L=128 下完全有效。实际上比 L=2048 时更好（残差 O(1/(L·ln b)) 更小）。

**Q: Algorithm 1 对 128 token 数据的预测可靠吗？**

D(Δ) 统计只涉及 [1, 128] 距离范围。在这个短距离范围内，自然语言的距离分布非常接近几何衰减（Zipf），远离 uniform。这意味着：
- β/α > 0 → τ* > 0
- Algorithm 1 应该给出一个有意义的 τ* 预测

**Q: 外推到 8192 时，EVQ 频率分配还合理吗？**

EVQ-Cosh 的频率 ω_k = b^{-φ_k(τ)} 在所有位置上都是良定义的。关键是：
- 最低频 ω_{N-1} ≈ b^{-1} ≈ 2×10⁻⁶：周期 ≈ 3×10⁶ >> 8192 ✓
- 最高频 ω_0 ≈ 1：能分辨相邻位置 ✓
- 中间频率由 cosh 分配：在高频端密度更大 → 短距离分辨率更好

**与 geometric RoPE 的区别在外推时放大**：geometric 在所有频率上等间距，而 EVQ 把更多频率集中在高频端。在外推到 8192 时：
- geometric 的中频率可能恰好出现周期共振（position mod period ≈ 相同相位）
- EVQ 的频率分配避免了这种共振（cosh 分配打破了等间距的规律性）

### 4.2 与 DAPE 的理论对比

| 维度 | DAPE | EVQ-Cosh |
|------|------|----------|
| 参数维度 | d/2 ≈ 32 | 1 (τ) |
| 数学保证 | 无 | cosh = Galerkin 投影最优 |
| 外推结构 | 无约束 → 频率可能溢出 | boundary anchoring → 端点锁定 |
| 过拟合风险 | 32D 搜索 + 有限数据 | 1D 搜索，几乎不可能过拟合 |
| 先验预测 | 不可能 | Algorithm 1: D(Δ) → τ* |

**频率覆盖分析（关键发现）**：

在 128 token 训练中，频率 ω_k = b^{-u_k} (b=500000) 的周期覆盖情况：

| 频率索引 k | 周期 (tokens) | 128 内完整周期数 | 梯度信号 |
|-----------|--------------|----------------|---------|
| k=0 (最高频) | ~8 | ~16 | 强 |
| k=8 | ~205 | ~0.6 | 弱 |
| k=16 | ~5454 | ~0.02 | 几乎为零 |
| k=24+ | >145000 | ~0 | 零 |

**关键推论**：128 token 训练中，只有前 ~10 个高频通道有梯度信号。低频通道（k>10）在训练中不学习任何东西。

**这恰好是 EVQ 相对于 DAPE 的核心优势所在**：
- DAPE 的低频通道在 128 训练中没有梯度 → 停留在初始化位置 → 外推时这些频率是随机的
- EVQ 的低频通道由 cosh 数学结构确定（不依赖梯度）→ 即使训练中无信号，外推时仍然有数学保证
- τ 只需从有梯度的高频通道学习，然后 cosh 结构自动推断低频的正确位置

**预期结果**：
- 在 128→512 (4×外推) 范围内，EVQ ≈ DAPE（高频主导，两者都有梯度）
- 在 128→4096+ (32×+外推) 范围内，EVQ > DAPE（低频主导，EVQ 有数学保证而 DAPE 没有）
- 总体 EVQ (1 param) ≈ 或 > DAPE (32 params)，且外推越极端优势越大

### 4.3 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| τ 在 128 训练中仍不收敛 | 低（10%）| 高 | 增大 tau_lr_mult 到 100×；128 tokens 下梯度理论上很强 |
| EVQ 明显输给 DAPE | 低（15%）| 中 | 调整 τ_init；即使输也说明 cosh 族有局限性，这本身是有价值的发现 |
| 外推 PPL 全部爆炸 | 中（25%）| 高 | 降低外推比到 16× (128→2048)；如果 DAPE 也爆炸则不是我们的问题 |
| 128 token 结果审稿人不认可 | 低（10%）| 低 | DAPE 用 128 中了 NeurIPS；加方案 B 作为补充 |

---

## 五、方案 A 的完整实验协议

### 5.1 Phase 0: Algorithm 1 先验预测 (CPU, 5 min)

```python
# 在 FineWeb-Edu 上测量 D̂(Δ) for Δ ∈ [1, 128]
# 然后用 Algorithm 1 计算 τ*_128

from rope.learnable_evq import measure_distance_distribution, estimate_tau_from_distance_prior

# 限制 max_delta = 128 (匹配训练长度)
D_hat = measure_distance_distribution(token_ids, max_delta=128)
tau_star, alpha, beta, residual = estimate_tau_from_distance_prior(D_hat, base=500000.0)
# 记录: τ*_128_FW = ?
```

这个值是实验前的 **盲预测**。如果 τ_learned 收敛到 τ*_128_FW 附近，三重验证成立。

### 5.2 Phase 1: 核心对比 (5 runs, ~2h)

```yaml
Common config:
  model: 125M (vocab=50304, hidden=768, layers=12, heads=12, head_dim=64)
  train_seq_len: 128
  train_tokens: 15M (足以在 128 token 下达到收敛)
  dataset: fineweb-edu
  base: 500000.0
  seed: 42
  eval_lengths: [128, 256, 512, 1024, 2048, 4096, 8192]

Runs:
  A1: Geometric RoPE (τ=0)                    # baseline
  A2: Fixed EVQ τ=1.0                         # fixed mid
  A3: Fixed EVQ τ=1.5                         # fixed high
  A4: Learnable EVQ (τ_init=1.0, lr_mult=100) # 核心实验
  A5: Learnable EVQ (τ_init=0.01, lr_mult=100)# 鲁棒性
```

**注意**: τ_lr_mult 从 10 提升到 100。原因：128 token 训练步数少（15M/128/batch ≈ 几千步），需要更大 LR 让 τ 有足够的更新量。

### 5.3 Phase 2: 与 DAPE 直接对比 (可选, 2 runs, ~1h)

```yaml
Runs:
  B1: DAPE-style (d/2=32 independent learnable frequencies, lr_mult=10)
  B2: DAPE-style (d/2=32 independent learnable frequencies, lr_mult=100)
```

实现 DAPE：将 inv_freq 变成 nn.Parameter(shape=(32,))，直接学习每个频率的 log 值。

### 5.4 Phase 3: 多 seed 确认 (如果 Phase 1 成功, 2 runs, ~1h)

```yaml
Runs:
  C1: Learnable EVQ (τ_init=1.0, seed=137)  # 复现性
  C2: Learnable EVQ (τ_init=1.0, seed=256)  # 第三 seed
```

### 5.5 产出清单

- [ ] τ 收敛轨迹图 (Figure τ_trajectory)
- [ ] PPL vs eval_length 表 (所有方法)
- [ ] τ*_theory vs τ_learned 对比
- [ ] EVQ vs DAPE 的参数效率对比
- [ ] 外推 PPL 的方法排序

### 5.6 成功标准

| 标准 | 最低要求 | 理想结果 |
|------|---------|---------|
| τ 收敛 | 两个 init 收敛到 |τ_A4 - τ_A5| < 0.5 | < 0.2 |
| EVQ vs Geometric | PPL@2048 改善 ≥ 3% | ≥ 10% |
| Learnable vs Fixed | PPL 在最优 Fixed ±2% | ±1% |
| EVQ vs DAPE | PPL@2048 差距 < 5% | EVQ ≤ DAPE |
| Theory match | |τ_learned - τ*_theory| < 1.0 | < 0.3 |

---

## 六、已有实验结果的重新解读

### 6.1 TinyStories 固定 τ 结果（有效，不需要重做）

| Model | τ | PPL@2K | PPL@16K | Δ% vs Geo@16K |
|-------|---|--------|---------|---------------|
| 50M | 0.0 | 4.146 | 33.316 | — |
| 50M | 1.5 | 4.134 | 29.697 | **-10.9%** |
| 125M | 0.0 (seed42) | 3.346 | 34.153 | — |
| 125M | 1.5 (seed42) | 3.290 | 27.699 | **-18.9%** |

**这些结果仍然有效**：它们证明了 EVQ frequency allocation 的结构性优势。论文中可以作为"固定 τ sweep"的证据。

**关键解读**：
- PPL@2K 差异 < 2%（模型补偿了 PE 差异）
- PPL@16K 差异 10-18%（模型无法补偿外推时的 PE 差异）
- 这恰好支持了 §1.2 的分析：**训练内 landscape 平坦 + 外推时差异放大**

### 6.2 Learnable τ Sanity Check 结果（诊断性，不是失败）

τ 不收敛 **不是 EVQ 理论的失败**，而是**实验设计的错配**：
- 理论说 τ 控制频率分配
- 实验在 2K 训练中无法区分不同的频率分配
- 解决方案：改变实验协议（128 token 训练 或 context extension）

### 6.3 Waterbed 验证（仍然有效）

8B/7B LoRA 下游任务结果（Retrieval ↑, Multi-hop ↓）仍然是论文最有力的理论-实验闭环，不受当前问题影响。

---

## 七、论文叙事 v4（适应新实验策略）

### 旧叙事 (v3, 已失效):
```
Theory → EVQ 族 → Learnable τ 自动收敛到 τ*_theory → 三重验证
```
三重验证在 2K 训练中失效（τ 不收敛）。

### 新叙事 (v4):
```
Theory → EVQ 族 → 在 PE-dominant regime (短上下文或 context extension) 中，
learnable τ 收敛到理论预测值 → 单参数 vs DAPE 的 d/2 参数同等性能 →
理论的价值 = 降维 + 外推保证
```

**核心论文结构**:

1. **§1 Introduction**: RoPE 频率分配是变分逆问题，EVQ-Cosh 是精确解。
2. **§3 Theory**: ODE → cosh 通解 → Galerkin 投影 → Algorithm 1。
3. **§4 Method**: Learnable τ + softplus + boundary anchoring。
4. **§5.1 Experiment: PE Quality Test** (方案 A):
   - 128 token training → 8192 extrapolation
   - EVQ learnable τ ≈ τ*_theory（三重验证）
   - EVQ (1 param) ≈ DAPE (32 params)（降维验证）
5. **§5.2 Experiment: From-Scratch Scaling** (已有):
   - 50M-350M TinyStories, fixed τ sweep
   - EVQ consistently beats geometric at extrapolation
6. **§5.3 Experiment: Waterbed Verification** (已有):
   - 8B/7B downstream tasks
   - Retrieval ↑, Multi-hop ↓ (exactly as predicted)
7. **§5.4 Experiment: Context Extension** (方案 B, 如果时间允许):
   - 2K geometric → 8K with EVQ
   - Practical application demo
8. **§6 Discussion**: τ 在 PE-dominant regime 收敛，在 model-dominant regime (长上下文训练) 无差异 → 这是一个关于 PE 与模型能力交互的见解。

---

## 八、关于 DAPE 复现的实现要点

### 8.1 DAPE 的核心实现（需要写的代码）

```python
class DAPEFrequencies(nn.Module):
    """DAPE-style: d/2 independent learnable frequency scales."""
    def __init__(self, dim, base=500000.0):
        super().__init__()
        n_freqs = dim // 2
        # Initialize as geometric (log-linear)
        u = (torch.arange(n_freqs, dtype=torch.float64) + 0.5) / n_freqs
        log_freqs = -u * math.log(base)
        self.log_freqs = nn.Parameter(log_freqs)  # d/2 个独立参数

    def get_frequencies(self):
        return torch.exp(self.log_freqs)
```

### 8.2 run_evq_sweep.py 的修改

需要加一个 `--dape` flag：
- 用 DAPEFrequencies 替代 EVQ
- 所有 d/2 个 log_freqs 都可学习
- lr_mult 适用于全部频率参数

### 8.3 训练序列长度的修改

需要加一个 `--seq_len` flag（当前硬编码为 2048）：
- 改为可配置
- 128 token 时 batch_size 可以大幅增加（内存充裕）

---

## 九、时间表

| 阶段 | 工作 | 时间估计 |
|------|------|---------|
| D0: 代码修改 | seq_len 可配置 + DAPE 实现 + eval_lengths 扩展 | 1-2 小时 |
| D0: Algorithm 1 | 在 FineWeb-Edu 上测量 D̂(Δ)@128 → τ*_128 | 10 min |
| D1: Phase 1 | 5 runs × 128 tokens | ~2 小时 |
| D1: Phase 2 | 2 runs (DAPE) | ~1 小时 |
| D2: Phase 3 | 2 runs (multi-seed) | ~1 小时 |
| D3: 方案 B (可选) | 1 pretrain + 5 continue-train | ~4 小时 |
| **总计** | | **~8-10 小时** |

---

## 十、最终建议

1. **立即执行方案 A**。成本最低（~2h GPU），风险最低，理论对齐最好。
2. **不要做更多 2K 训练的 learnable τ 实验**。Landscape 平坦是结构性问题，换 LR/warmup/参数化都无法解决。
3. **保留已有的 TinyStories + LoRA 结果**。它们证明了不同的东西（频率分配的结构性优势 + waterbed 验证）。
4. **方案 B 作为锦上添花**。如果方案 A 成功且还有 GPU 时间，方案 B 增加论文的实用性论证。

---

*审核完成: 2026-03-01*
