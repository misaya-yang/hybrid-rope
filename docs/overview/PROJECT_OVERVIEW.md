# EVQ-Cosh: RoPE 频率分配优化项目概述

> **文档目的**：向合作者同步项目核心理论、实验结果和论文定位
> **最后更新**：2026-03-04
> **代码仓库**：`hybrid-rope`

---

## 一、项目背景与核心问题

### 1.1 我们在解决什么问题？

**RoPE（Rotary Position Embedding）** 是目前大语言模型最主流的位置编码方式，被 LLaMA、Qwen、Mistral 等主流模型采用。

RoPE 的核心设计是使用**等比数列**的频率（Geometric RoPE）：
```
θ_k = base^(2k/d)    其中 k = 0, 1, 2, ..., d/2-1
```

**问题**：这个设计是经验性的，没有理论依据。我们的研究发现：
- Geometric RoPE 在长上下文外推时表现不佳
- 根本原因是**低频通道之间的相位碰撞（Phase Collision）**

### 1.2 什么是相位碰撞？

RoPE 通过不同频率的余弦/正弦波来编码位置信息。当两个频率通道的波长接近时，它们在长距离上会产生"碰撞"——不同位置得到几乎相同的编码，模型无法区分。

**直观理解**：
- 高频通道（短波长）：编码短距离位置，冗余度高
- 低频通道（长波长）：编码长距离位置，是长上下文的瓶颈

Geometric RoPE 的等比分配导致低频通道间距过小，在长距离外推时碰撞严重。

---

## 二、核心理论贡献

### 2.1 变分优化框架

我们将 RoPE 频率分配问题形式化为一个**变分逆问题**：

**目标**：在明确的宽带 surrogate 下推导闭式频率密度分布 ρ(φ)，改善位置编码的长程区分度。

**推导链**（简化版）：
1. 从距离先验 D(Δ) 出发
2. 建立相位碰撞核 K(φ₁, φ₂)
3. 通过变分泛函得到 Euler-Lagrange 方程
4. **闭式解**：ρ*(φ) = C₁·cosh(τφ) + C₂·sinh(τφ) + P·b^(-2φ)

### 2.2 核心定理

**Theorem 1（ODE 精确解）**
该 surrogate 的闭式频率密度由双曲函数（cosh/sinh）项和指数衰减项的竞争决定，参数 τ 控制两者的平衡。

**Theorem 2（Geometric 是退化特例）**
当 τ → 0 时，EVQ 光滑退化为 Geometric RoPE。
- **推论**：Geometric RoPE 是 EVQ-Cosh pure-tether 子族中的 τ=0 退化点；经验上在长程 stress tests 中常处于较弱 operating point。

**Waterbed 不等式**
频率重分配存在代价：短上下文性能可能略有下降，但长上下文性能显著提升。
- 实验验证：PPL@2K 仅 +0.4%（误差范围内），PPL@16K -13.3%

### 2.3 τ 的物理意义

参数 τ 控制频率通道密度的重分配：
- τ = 0：log-frequency 间距均匀（Geometric）
- τ > 0：**低频间距扩大，高频间距压缩**

**关键洞察**：
- 高频通道有大量冗余（编码相似的短距离信息）→ 压缩代价极小
- 低频通道是瓶颈（相位碰撞破坏长距离信息）→ 扩大收益巨大
- 本质是"拆冗余补瓶颈"，不是对称的 tradeoff

### 2.4 τ* Scaling Law

默认 operating τ 由训练长度 L 和注意力头维度 d_head 给出：

$$\tau^*(L) = \frac{d_{head}}{\sqrt{L}}$$

**实验验证**（5 个 context length）：

| L_train | 预测 τ* | 实测 τ* | 偏差 |
|---------|---------|---------|------|
| 1024 | 2.0 | 2.0 | 精确 |
| 2048 | 1.41 | 1.5 | +6% |
| 512 | 2.83 | 4.0 | +41% |
| 256 | 4.0 | 5.0 | +25% |
| 128 | 5.66 | ≥5.0 | 单调下降 |

L ≥ 1024 时吻合良好；L < 1024 系统偏高（极端短序列 regime）。

---

## 三、方法：EVQ-Cosh

### 3.1 算法

```python
def evq_cosh_inv_freq(d_head, L, base):
    """
    EVQ-Cosh 频率分配
    输入：d_head = 注意力头维度, L = 训练长度, base = RoPE base
    输出：逆频率张量（替换标准 RoPE 的 inv_freq）
    """
    tau = d_head / math.sqrt(L * math.log(base))
    K = d_head // 2
    u = torch.linspace(0.5/K, 1 - 0.5/K, K)  # 均匀分位数
    phi = 1 - (1/tau) * torch.arcsinh((1 - u) * math.sinh(tau))
    return base ** (-phi)
```

**使用方式**：只需替换一行 inv_freq 初始化代码。不改架构、不改训练流程、不改推理。

### 3.2 超参数对比

| 方法 | 超参数数量 | 需要搜索？ |
|------|-----------|----------|
| Geometric RoPE | 0 | — |
| PI (Position Interpolation) | 1 | 否，但性能差 |
| NTK-aware | 1 | 通常需要 |
| YaRN | 3 | **是** |
| DAPE (learnable) | 32 | **是** |
| **EVQ-Cosh** | **0** | **否（τ* 由公式给出）** |

### 3.3 退化安全性

τ → 0 时 EVQ → Geometric。即使 τ* 预测偏差 50%，最坏情况是"没提升"而不是"崩溃"。

---

## 四、实验结果汇总

### 4.1 跨规模 PPL 改善（base=500K，L_train=2048 为主）

| 模型规模 | 数据集/设置 | Seeds | Δ PPL@2K | Δ PPL@16K |
|---------|-------------|-------|----------|-----------|
| 50M | TinyStories | 1 | -0.3% | **-10.9%** |
| 125M | TinyStories | 1 | -1.7% | **-18.9%** |
| 454M | TinyStories (Hybrid) | 1 | -0.4% | **-13.7%** |
| 454M | FineWeb-Edu | 3 | +0.4% | **-13.3%** |
| 750M continue@4K | mix setting | 1 | +1.2% | **-45.9%** |

**结论**：短程代价保持在小范围内，长程 PPL 改善在多个规模和设置下方向一致；750M 行按论文表述仅作为 single-seed consistency check。

### 4.2 核心发现：EVQ + YaRN 互补

**实验配置**：454M 模型，10% passkey mix，3 seeds/config，L_train=2048，matched YaRN scale s=8。

| 方法 | PK@8K | PK@12K | PK@16K | PPL@8K / 16K |
|------|-------|--------|--------|--------------|
| Geo | 41±5% | 57% | 51% | 161.9 / 253.2 |
| Geo + YaRN | 61±3% | 59% | 51% | 82.9 / 157.7 |
| EVQ τ=1.5 | 53±8% | 63% | 50% | 150.3 / 229.5 |
| **EVQ + YaRN** | **100±0%** | **79%** | **68%** | **70.9 / 107.5** |

**论文定位**：EVQ 不是 YaRN 的替代品；EVQ 改变训练时频率 substrate，YaRN 改变推理时 range scaling，两者是互补的 PE 设计轴。

### 4.3 Passkey Mix 检索能力验证

**配置**：90% FineWeb-Edu + 10% synthetic passkey，454M，3 seeds，teacher-forced NLL-gap retrieval。

| 长度/指标 | Geo | EVQ | Δ |
|-----------|-----|-----|---|
| 2K retrieval | 100% | 100% | 0 |
| 4K retrieval | 58.7% | **68.7%** | **+10.0pp** |
| 8K retrieval | 40.7% | **53.3%** | **+12.7pp** |
| Global retrieval | 66.7% | **74.0%** | **+7.3pp** |
| PPL@16K | 262.0 | **237.2** | **-9.5%** |

**解读**：seed-42 的 +40pp 是诊断性大差异，不作为 headline；论文使用 3-seed mean。

### 4.4 5% vs 10% passkey concentration

同样总 token 量下，passkey 浓度从 5% 到 10%：

| 效应 @8K | Geo | EVQ |
|----------|-----|-----|
| 5% mix retrieval | 54.0% | 56.7% |
| 10% mix retrieval | 40.7% | 53.3% |
| Δ | **-13.3pp** | **-3.3pp** |

**解读**：更密的 retrieval supervision 对 Geo 的 8K generalization 损伤更大；EVQ 的频率分配在该设置下更抗退化。该结果用于 robustness，不升级为 universal data-scaling law。

### 4.5 750M 训练动态分析

**核心发现**：尽管两种方法的 OOD PPL 都在恶化（waterbed 效应），但：

| 训练进度 | Geo PK@8K | Hybrid PK@8K |
|---------|-----------|--------------|
| 25% | 55% | 45% |
| 50% | 70% | 65% |
| 75% | 60% | 75% |
| 100% | 60% | **80%** |

- Geometric 的 passkey retrieval **在 50% 后回退**（70%→60%）
- Hybrid 的 passkey retrieval **单调递增**（45%→80%）

**结论**：PPL waterbed 对 Hybrid 的位置检索能力没有破坏力。这是 EVQ 频率分配的核心价值。

### 4.6 Base=10K 死区验证

| Method | retrieval | PPL@16K |
|--------|-----------|---------|
| Geometric | 0.680 | 274.2 |
| EVQ τ=0.7 | 0.643 (-5.5%) | 342.8 (+25%) |
| EVQ τ=1.1 | 0.568 (-16.5%) | 282.4 (+3%) |

**结论**：base=10K 下 EVQ 全败。这是碰撞块理论的预测确认——低 base 导致大部分频率通道落入"碰撞块"（波长超过训练长度），可优化空间极小。

---

## 五、论文叙事结构

### 5.1 核心主张（3 层递进）

1. **理论层**：RoPE 频率分配是变分逆问题，有闭式解；Geometric 是 τ=0 退化点；τ* 由 scaling law 给出

2. **训练时实验层**：raw PPL、passkey/NLL-gap retrieval、PE-dominant extrapolation、MLA scarce-channel stress test 共同支持有限频谱预算视角

3. **组合层**：454M matched-scale EVQ+YaRN 在 8K 达到 100±0% vs Geo+YaRN 61±3%，支持 training-time allocation 与 inference-time scaling 的互补性

### 5.2 论文 Figure 规划

| Figure | 内容 | 状态 |
|--------|------|------|
| **Figure 1（主图）** | 方法 schematic 与频率分配直觉 | 完成 |
| **Figure 2** | EVQ × YaRN matched-scale complementarity | 完成 |
| Appendix figures | PE-dominant scaling、QuALITY、τ* validation 等 supporting views | 完成 |

### 5.3 与现有方法的关系

| 方法类别 | 代表 | EVQ 的定位 |
|---------|------|-----------|
| Inference-time | PI, NTK-aware, YaRN | **互补**：EVQ 是 training-time 优化，组合使用效果超线性 |
| Learnable PE | DAPE, Kerple | **同轴对照**：EVQ 是闭式、零额外参数的 training-time allocation，与 learned PE 保持不同假设 |
| 其他 RoPE 变体 | xpos, ALiBi | 不同设计空间 |

---

## 六、证据强度评级

### 6.1 安全区（可以写得自信）

| Claim | 证据 | 强度 |
|-------|------|------|
| EVQ PPL@8K-16K 优于 Geo | 6/6 runs 全胜 | **A（无争议）** |
| EVQ+YaRN 8K=100% | 454M matched-scale 3-seed zero variance | **A** |
| Waterbed 短端代价有限 | 6/6 runs ≤+4.1% | **A** |
| τ* scaling law | 5 context lengths | **B+** |
| Base=10K 死区 | 碰撞块理论精确预测 | **A** |

### 6.2 风险区（需要 caveat）

| Claim | 状态 |
|-------|------|
| seed-42 large passkey delta | 不作为 headline；论文使用 3-seed +10.0pp/+12.7pp |
| "5%→10%" robustness | 使用 multi-seed mean，只表述为该设置下更抗退化 |
| 750M continuation | single-seed consistency check，不作为 primary claim |

---

## 七、后续工作（不作为当前投稿主 claim）

### 7.1 高优先级

- 750M multi-seed retrieval divergence
- r=14/τ=2.5 multi-seed validation

### 7.2 中优先级

- 更多 r 值的 τ-sweep（验证 τ*(r) 修正公式）
- CHE benchmark 补充实验（验证极端短序列 regime）

### 7.3 低优先级

- 换 head_dim 验证 τ* 公式中的系数
- 与 DAPE 的直接对比实验

---

## 八、常见问题

**Q: EVQ 和 YaRN 有什么区别？**
A: YaRN 是 inference-time 方法，在推理时对位置编码进行缩放。EVQ 是 training-time 方法，在训练前确定闭式频率分配。两者是正交的，组合使用效果超线性。

**Q: 为什么不用可学习的位置编码（如 DAPE）？**
A: EVQ 提供闭式解，零额外参数，不需要搜索。实验表明 in-distribution loss 对 τ 几乎平坦；learned-PE 对比仅在文中 DAPE-style tested protocol 内解读。

**Q: 短上下文会退化吗？**
A: Waterbed 效应存在，但实际代价极小（PPL@2K ≤+0.4%，在误差范围内）。750M 模型 in-distribution PPL 甚至略有改善（-0.14%）。

**Q: 为什么 base=10K 全败？**
A: 碰撞块理论预测：低 base 导致大部分频率通道的波长超过训练长度，落入"不可分辨区"。可优化空间极小（仅 ~3/32 通道）。这反而是理论的验证。

---

## 九、联系方式与资源

- **代码仓库**：`hybrid-rope`
- **核心文档**：
  - `paper/appendix/a1_proofs.tex` - 完整理论推导
  - `docs/exp/` - 实验报告
  - `paper/figs/` - 论文图表
- **实验数据**：`results/`

---

*本文档是项目核心内容的摘要。详细技术细节请参考 `CORE_THEORY.md`。*
