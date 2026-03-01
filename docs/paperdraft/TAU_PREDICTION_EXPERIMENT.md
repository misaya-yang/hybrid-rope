# τ 先验预测实验设计

> **创建日期**: 2026-03-01
> **核心 idea**: τ = √(β/α) 依赖数据的距离先验 D(Δ)，不依赖模型大小。
> 因此可以在 50M/125M 上低成本验证"从数据直接预测 τ"的完整闭环。

---

## 1. 理论链条

```
训练数据 → 测量 D(Δ) → 计算 E_diag(φ) → 拟合 α,β → τ* = √(β/α) → 与 τ_emp 对比
```

### 1.1 为什么 τ 与模型大小无关

τ = √(β/α) 中的 α, β 来自干涉核 K(φ₁,φ₂) 的 broadband 分解。
K 只依赖三个量：
- **b** (RoPE base) — 模型超参数，通常固定（如 10000）
- **L** (训练上下文长度) — 通常固定（如 2048）
- **D(Δ)** (距离先验) — 数据的性质

模型参数量（50M vs 125M vs 500M）不影响 K 的分解，因此不影响 τ。

**推论**：同一数据集 + 同样的 b 和 L → 最优 τ 应该相同。
- TinyStories: τ_emp = 1.5（50M 和 125M 一致 ✅）
- FineWeb-Edu: τ_emp = ?（理论预测应与 50M sweep 结果一致）

### 1.2 为什么不同数据集的 τ 会不同

TinyStories 是儿童故事：句子短、结构简单 → D(Δ) 衰减快（陡峭 power-law）
FineWeb-Edu 是 web 教育文本：段落长、引用多 → D(Δ) 衰减慢（平缓 power-law）

D(Δ) 越平缓 → E_diag(φ) 的斜率越小 → 需要的高频偏置越少 → τ* 越小。

**预测**: FineWeb-Edu 的 τ_emp < 1.5（可能在 0.5-1.2 之间）。

---

## 2. 实验计划

### 2.1 Phase A: 50M τ-sweep on FineWeb-Edu（~3-5小时, 5090）

| Run | τ | 训练量 | 评估 |
|-----|---|--------|------|
| 1 | 0.0 (geometric) | 50M tokens | PPL@2K/4K/8K/16K |
| 2 | 0.5 | 50M tokens | 同上 |
| 3 | 1.0 | 50M tokens | 同上 |
| 4 | 1.5 | 50M tokens | 同上 |
| 5 | 2.0 | 50M tokens | 同上 |

**产出**: FineWeb-Edu 上的 τ_emp（PPL@16K 最低的那个 τ）

### 2.2 Phase B: D(Δ) 测量 + 理论预测 τ*

**方法 1（简单版）**: Token 距离统计
- 从 FineWeb-Edu 训练集取 1000 个样本
- 对每个样本，统计所有 token pair 的距离分布
- 拟合 D(Δ) ∝ Δ^{-γ}，估计 power-law 指数 γ
- 对 TinyStories 做同样的事，得到 γ_tiny
- 比较 γ_fineweb vs γ_tiny

**方法 2（更精确）**: Attention 距离分布
- 用 geometric baseline 的 50M checkpoint
- Forward 100 个 batch
- 记录每层每头的 attention weight 在各距离上的分布
- 这就是模型"真正看到"的 D(Δ)

**方法 3（直接计算）**: 从 D(Δ) 计算 τ*
- 用测量的 D(Δ) 计算 E_diag(φ) = ½(1 + D̂_b(2b^{-φ}))
- 对 E_diag 做对角+非对角分解，估计 α, β
- 计算 τ* = √(β/α)
- 与 Phase A 的 τ_emp 对比

### 2.3 Phase C: 125M 双 seed 确认（~4-6小时, 5090）

| Run | τ | Seed | 训练量 |
|-----|---|------|--------|
| 1 | 0.0 | 42 | 100M tokens |
| 2 | τ_emp | 42 | 100M tokens |
| 3 | 0.0 | 137 | 100M tokens |
| 4 | τ_emp | 137 | 100M tokens |

---

## 3. 论文中的呈现

### 3.1 新增内容

**Table: Cross-dataset τ-sweep comparison**

| Dataset | γ (power-law) | τ* (predicted) | τ_emp (sweep) | PPL Δ@16K |
|---------|---------------|----------------|---------------|-----------|
| TinyStories | γ_tiny | τ*_tiny | 1.5 | -10.9% |
| FineWeb-Edu | γ_fw | τ*_fw | ? | ? |

**这张表是论文的 killer evidence**: 理论从数据先验预测了最优 τ，sweep 验证了预测。

### 3.2 叙事升级

原来的叙事：theory → EVQ formula → sweep finds good τ → improvement
升级后：theory → D(Δ) measurement → τ prediction → sweep validates prediction → improvement

这把论文从 "understanding paper" 升级成了 **"predictive theory paper"**。

---

## 4. 成本估算

| Phase | GPU 时间 (5090) | 备注 |
|-------|----------------|------|
| A: 50M sweep (5 runs) | ~3-5 小时 | 每 run ~40分钟 |
| B: D(Δ) 测量 | ~30 分钟 | 推理 only |
| C: 125M confirm (4 runs) | ~4-6 小时 | 每 run ~1小时 |
| **总计** | **~8-12 小时** | **一天完成** |

---

## 5. 是否还需要 500M？

**如果 τ 预测实验成功**（τ* ≈ τ_emp），500M 变成可选的 nice-to-have：
- 50M/125M 在两个数据集上的 τ 预测 + 验证已经是完整的 story
- 500M 只是额外的 scaling 点，锦上添花
- 可以在 review 阶段被要求时再补

**如果 τ 预测偏差较大**（τ* ≠ τ_emp 但方向一致），仍然有价值：
- 证明理论给出了正确的方向（定性预测）
- broadband 近似的有限 base 残差可以解释定量偏差
- 500M 可以帮助验证偏差是否随规模稳定

---

## 6. 与原来 500M 方案的对比

| 维度 | 原方案 (500M FineWeb-Edu) | 新方案 (τ prediction) |
|------|-------------------------|---------------------|
| 成本 | ~2天 RTX PRO 6000 | ~1天 5090 |
| 证明了什么 | "EVQ 在大模型+好数据集上也赢" | "理论能预测最优 τ" |
| 论文 impact | 堵住攻击面 | **新的正面贡献** |
| 审稿人反应 | "OK, 意料之中" | "哇，理论真的能预测" |
| 风险 | 低（一定赢） | 中（预测精度未知） |

**新方案的 risk-reward 明显更优。**
