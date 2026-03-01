# 最终行动方案（综合 Gemini + GPT-5.2 Pro + Claude 审核）

> **日期**: 2026-03-01
> **状态**: 定稿，不再改动

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

## 五、5090 实验清单（更新: Learnable τ 为核心）

### Phase 0: D(Δ) 测量 + Algorithm 1 验证 (~30 min)
| 序号 | 任务 | GPU 时间 | 产出 |
|------|------|---------|------|
| 0a | 测量 TinyStories 的 D̂(Δ) | 10 min | D̂_tiny.npy |
| 0b | 测量 FineWeb-Edu 的 D̂(Δ) | 10 min | D̂_fw.npy |
| 0c | Algorithm 1: D̂ → τ* | 1 min (CPU) | τ*_tiny, τ*_fw |

### Phase 1: 100M 对照实验 (~10.5 hr)
| 序号 | Method | 配置 | 产出 |
|------|--------|------|------|
| 1 | Geometric baseline | 标准 RoPE | PPL 基线 |
| 2 | Fixed EVQ τ=0.5 | 固定 | sweep 点 |
| 3 | Fixed EVQ τ=1.0 | 固定 | sweep 点 |
| 4 | Fixed EVQ τ=1.5 | 固定 | sweep 点 |
| 5 | Fixed EVQ τ=2.0 | 固定 | sweep 点 |
| 6 | **Learnable EVQ** τ_init=1.0 | lr_mult=10 | 核心实验 |
| 7 | **Learnable EVQ** τ_init=0.01 | 从 geometric 开始 | 鲁棒性验证 |

### Phase 2: 验证扩展 (如果 Phase 1 成功)
| 序号 | 任务 | 目的 |
|------|------|------|
| 8 | 100M TinyStories Learnable EVQ | 跨数据集 τ 差异 |
| 9 | 125M FineWeb-Edu Learnable EVQ (seed 42) | 规模不变性 |
| 10 | 125M FineWeb-Edu Learnable EVQ (seed 137) | 复现性 |

**总计**: ~12-15 hr (Phase 1) + ~5 hr (Phase 2)

---

## 六、论文叙事升级路径 (v3: Learnable τ)

**v1**: theory → EVQ formula → sweep 找到好的 τ → improvement
**v2**: theory → D(Δ) measurement → τ prediction → sweep validates
**v3 (当前)**: theory → EVQ 函数族 → **τ 可学习** → 自动收敛到最优 → 理论预测验证

从 "understanding paper" → "predictive theory paper" → **"method paper with theory"**

审稿人看到的不是"他们调了一个超参数赢了"，
而是"他们从变分理论推导出参数化族，τ 在训练中自动收敛到理论预测值"。

**核心卖点变化**：
- 理论的价值 = 把 N/2 维搜索空间压缩到 1D → 函数形式保证
- 工程的价值 = 零手动调参，drop-in replacement
- 三重验证 = Algorithm 1 预测 ≈ sweep 最优 ≈ 学习收敛值

详细设计: `docs/paperdraft/LEARNABLE_TAU_DESIGN.md`
实现代码: `rope/learnable_evq.py`
Claude Code prompt: `docs/paperdraft/PROMPT_CLAUDE_CODE_LEARNABLE_TAU.md`
