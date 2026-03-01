# 最终行动方案 v4（综合 Gemini + GPT-5.2 Pro + Claude 审核 + 实验诊断）

> **日期**: 2026-03-01
> **状态**: v4 更新——解决 learnable τ 不收敛问题
> **关键变更**: 训练序列长度从 2048 改为 128（对标 DAPE）

---

## 一、写进论文正文的（高优先级）

### 1. D(Δ) → τ* 的数值估计器 [正文 §4 新增, Algorithm 1]

**方法**: GPT 的非对角+对角分步拟合法（比 Gemini 的全部展平更稳健）

```
Algorithm 1: Data-Driven τ Estimation
Input: 经验距离分布 D̂(Δ), RoPE base b, 网格点数 N
1. 计算余弦变换 C(ν) = Σ_Δ D̂(Δ) cos(νΔ)
2. 构造核矩阵 K_ij = ½[C(ω_i - ω_j) + C(ω_i + ω_j)]
3. 用非对角元素 (i≠j) 回归: K_ij ≈ c₀ + β·min(φ_i, φ_j) → 解出 β, c₀
4. 用对角元素回归: K_ii ≈ c₀ + β·φ_i + α/Δφ → 解出 α
5. 输出: τ* = √(β/α)
   附带: 拟合残差 ‖K - K_approx‖_F / ‖K‖_F（broadband 近似诊断）
```

### 2. 跨数据集 τ 预测 vs 经验对照表 [正文 Table, §5]

| Dataset | D̂ source | τ* (predicted) | τ_emp (sweep) | 拟合残差 | PPL Δ@16K |
|---------|----------|----------------|---------------|---------|-----------|
| TinyStories | token 距离统计 | ? | 1.5 | ? | -10.9% |
| FineWeb-Edu | token 距离统计 | ? | ? (sweep) | ? | ? |

**这张表是论文的 killer evidence。**

### 3. 拟合残差作为 broadband 近似的诊断 [正文 §4.2 扩展]

报告 ‖K - (c₀ + βM + αI/Δφ)‖_F / ‖K‖_F，量化近似质量。
- 残差小 → broadband 分解成立 → τ* 可信
- 残差大 → 数据结构超出二阶 ODE 假设 → 解释为什么某些数据集上理论预测偏差更大

---

## 二、写进 Appendix 的

### 4. Learnable τ 梯度推导 [Appendix, 支撑正文 §4]

$$\frac{\partial \phi}{\partial \tau} = \frac{1}{\tau^2}\left[\text{arcsinh}((1-u)\sinh\tau) - \tau\cdot\frac{(1-u)\cosh\tau}{\sqrt{1+(1-u)^2\sinh^2\tau}}\right]$$

包含：
- 完整推导过程
- τ→0 时的 Taylor 稳定性分析
- 边界锚定证明 (∂φ/∂τ = 0 at endpoints)
- softplus 参数化的理论动机

> **⚠️ 策略更新 (2026-03-01)**: Learnable τ 从 "Appendix-only future work" 升级为 **正文核心贡献 (§4)**。梯度推导留在 Appendix 支撑正文。

### 5. 谱匹配法（可选，如果空间允许）[Appendix]

用 (-∂²)⁻¹ 的本征函数匹配前两个模式解 α, β。比最小二乘更有理论味道。

---

## 三、写进 Conclusion 的（一句话 future work）

### 6. B* = ln D(Δ) + const 统一 ALiBi/T5

> "Our variational framework extends naturally to additive attention biases:
> under a mean-field approximation, the optimal bias B*(Δ) = log D(Δ) + const,
> recovering ALiBi as the exponential-prior special case and T5's log-bias
> under a power-law prior."

### 7. Per-layer learnable τ

> "Allowing per-layer τ_ℓ captures the effective distance prior shift across
> layers; preliminary gradient analysis (Appendix X) shows stable training
> with boundary-anchored gradients."

### 8. Sigmoid attention（可选，如果空间够）

> "Removing softmax normalization breaks the waterbed constraint but
> introduces new regularization requirements; we leave this to future work."

---

## 四、绝对不做的

| 诱惑 | 为什么不做 |
|------|----------|
| ALiBi/T5 的实验对比 | 引火烧身，审稿人会要求完整 baseline 矩阵 |
| Sigmoid attention 实验 | 太远，另一篇论文 |
| 分数阶 ODE（重尾先验） | 纯理论延伸，没有实验支撑 |
| Per-layer τ 实验 | future work，当前全局 τ 已足够 |

---

## 五、5090 实验清单 (v4: 128-token PE Quality Test)

> **⚠️ v4 变更说明**: 2K 训练中 learnable τ 不收敛（loss landscape 平坦），
> 根因是训练上下文太长，模型权重吸收了 PE 差异。
> 改为 DAPE 风格的 128 token 训练，直接对标 DAPE (NeurIPS 2024)。
> 详细分析见 `docs/paperdraft/EXPERIMENT_AUDIT_V4.md`

### Phase 0: D(Δ) 测量 + Algorithm 1 盲预测 (~10 min, CPU)
| 序号 | 任务 | 产出 |
|------|------|------|
| 0a | 测量 FineWeb-Edu D̂(Δ) for Δ∈[1,128] | D̂_fw_128.npy |
| 0b | Algorithm 1: D̂ → τ*_128 | τ*_128_FW (盲预测) |
| 0c | 测量 TinyStories D̂(Δ) for Δ∈[1,128] | D̂_tiny_128.npy |

### Phase 1: 128-token 核心对比 (~2h GPU)
| 序号 | Method | 配置 | 目的 |
|------|--------|------|------|
| A1 | Geometric RoPE | τ=0 | baseline |
| A2 | Fixed EVQ τ=1.0 | 固定 | sweep 中值 |
| A3 | Fixed EVQ τ=1.5 | 固定 | sweep 高值 |
| A4 | **Learnable EVQ** | τ_init=1.0, lr_mult=100 | **核心实验** |
| A5 | **Learnable EVQ** | τ_init=0.01, lr_mult=100 | 鲁棒性 |

公共配置: 125M, train_seq=128, train_tokens=15M, FineWeb-Edu, base=500000, seed=42
评估长度: [128, 256, 512, 1024, 2048, 4096, 8192]

### Phase 2: DAPE 直接对比 (~1h GPU)
| 序号 | Method | 配置 | 目的 |
|------|--------|------|------|
| B1 | DAPE-style | d/2=32 独立可学习频率, lr_mult=10 | 直接对标 |
| B2 | DAPE-style | d/2=32 独立可学习频率, lr_mult=100 | LR 校准 |

### Phase 3: 多 seed 确认 (~1h GPU, 如果 Phase 1 成功)
| 序号 | Method | 配置 | 目的 |
|------|--------|------|------|
| C1 | Learnable EVQ | seed=137 | 复现性 |
| C2 | Learnable EVQ | seed=256 | 第三 seed |

### Phase 4: Context Extension (可选, ~4h GPU)
| 序号 | 步骤 | 配置 | 目的 |
|------|------|------|------|
| D0 | 预训练 | 125M, geometric, 2K, 100M tokens | 基础模型 |
| D1 | 扩展: Geometric | 继续训练 8K, 20M tokens | baseline |
| D2 | 扩展: PI | freq /= 4 | 经典方法 |
| D3 | 扩展: YaRN | PI + 高频保护 | 经典方法 |
| D4 | 扩展: EVQ fixed τ=1.5 | — | 理论方法 |
| D5 | 扩展: EVQ learnable | τ_init=1.0 | **核心** |

**总计**: Phase 0-3 (~4h) + Phase 4 (~4h) = ~8h

---

## 六、论文叙事升级路径 (v4: PE-dominant regime)

**v1**: theory → EVQ formula → sweep 找到好的 τ → improvement
**v2**: theory → D(Δ) measurement → τ prediction → sweep validates
**v3**: theory → EVQ 函数族 → learnable τ → 自动收敛 (❌ 在 2K 训练中失效)
**v4 (当前)**: theory → EVQ 函数族 → 在 PE-dominant regime 中 learnable τ 收敛 → 1 param vs DAPE 32 params

**关键见解**：PE 的频率分配在 "PE-dominant regime"（短训练上下文或 context extension）中有强梯度信号。
在 "model-dominant regime"（长训练上下文）中，模型权重吸收了 PE 差异，τ 的 landscape 平坦。
这不是理论的失败，而是关于 PE 与模型能力交互的见解。

**核心卖点**：
- 理论的价值 = 把 N/2 维搜索空间压缩到 1D（vs DAPE 的暴力搜索）
- 实验的价值 = EVQ (1 param) ≈ DAPE (32 params) at extrapolation
- 三重验证 = Algorithm 1 预测 ≈ sweep 最优 ≈ 学习收敛值（在 128-token regime）

详细设计: `docs/paperdraft/LEARNABLE_TAU_DESIGN.md`
实验审核: `docs/paperdraft/EXPERIMENT_AUDIT_V4.md`
实现代码: `rope/learnable_evq.py`
