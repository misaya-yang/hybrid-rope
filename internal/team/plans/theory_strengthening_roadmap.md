# EVQ 理论加强路线图

> **目的**：梳理当前理论中"方向对但不够 solid"的结论，列出加强实验和理论补丁的优先级。
> **创建时间**：2026-03-11
> **背景**：Phase 18 发现 EVQ 跨 50× base 范围 PPL@4K 几乎一致（192.39 vs 191.90），暗示频率优化可以部分替代 base tuning。这如果完全证明，论文 level 直接上升一个层级。

---

## 1. 核心待证命题：EVQ 部分补偿 base tuning

### 1.1 当前证据

| Base | Geo PPL@4K | EVQ PPL@4K | Δ |
|------|-----------|-----------|---|
| 10K | 246.12 | 192.39 | -21.8% |
| 500K | 284.87 | 191.90 | -32.6% |

- EVQ 跨 50× base：PPL@4K 差异仅 0.25%（192.39 vs 191.90）
- Geo 跨 50× base：PPL@4K 差异 15.7%（246.12 vs 284.87）
- Collision 维度缩减：base=10K 50%→25%，base=500K 66%→38%

### 1.2 当前弱点

1. **单 seed**：只有 seed=42，可能是运气
2. **只有两个 base 点**：10K 和 500K，中间和更极端的 base 未测
3. **只有 L=512**：不知道 L=1024/2048 下是否保持
4. **没有理论推导**：为什么 EVQ 会补偿 base？需要从变分解出发推导

### 1.3 加强实验（按优先级排序）

#### P0: Multi-seed 验证（base=10K + 500K）
- **成本**：~2h on M4 Max（4 runs × ~30min）
- **做法**：seed=137, 256 重跑 base=10K 和 base=500K
- **判定标准**：3-seed mean 是否保持 EVQ@10K ≈ EVQ@500K（PPL@4K 差距 <5%）
- **如果通过**：从 B+ 升级到 A-

#### P1: 补充中间 base 点（100K）
- **成本**：~1h on M4 Max（2 runs）
- **做法**：base=100K, seed=42, Geo vs EVQ
- **目的**：验证 EVQ PPL@4K 是否在 10K-100K-500K 三点上都 ≈192
- **如果通过**：强化"EVQ 创造了一条 base-invariant 性能底线"的叙事

#### P2: 极端 base 验证（1M, 10M）
- **成本**：~2h
- **做法**：base=1M 和 10M
- **目的**：验证 EVQ 补偿是否有上界——极高 base 下所有通道都在低频区，EVQ 重分配空间缩小
- **预期**：base=10M 时 c=0.38，大量通道已在有效区间，EVQ 增益收窄

---

## 2. Collision 维度缩减的理论推导

### 2.1 当前状态
Phase 18 的 collision 分析是经验观察：EVQ 把碰撞维度从 N 降到 ~N/2。但没有从变分解出发的推导。

### 2.2 推导思路

**核心论点**：EVQ-cosh 的频率重分配 φ_k(τ) 将一部分原本位于碰撞区（λ_k > L_train）的通道推入有效区（λ_k ≤ L_train）。

**数学化**：
- 碰撞条件：通道 k 碰撞 ↔ ω_k = base^{-φ_k} 使得 2π/ω_k > L_train
- 等价：φ_k > ln(L_train/2π)/ln(base) = 1 - c + ln(2π)/ln(base)（取 leading term 约为 1-c）
- Geometric：φ_k = k/(d/2)，碰撞通道数 = (d/2)(1 - (1-c)) = (d/2)c
- EVQ：φ_k(τ) < φ_k(0) = k/(d/2)（因为 τ>0 把所有通道向高频推移），碰撞数减少
- **减少量**：Δn_collision = Σ_{k: φ_k(0) > 1-c, φ_k(τ) ≤ 1-c} 1

**需要做的推导**：
1. 找到 EVQ 碰撞通道数的闭式表达
2. 证明 Δn_collision 单调增加 with τ（至少在 0 < τ < τ_max 范围内）
3. 如果可能，给出 EVQ 碰撞分数 c_EVQ(τ) = c_Geo - f(τ, c_Geo) 的近似表达

### 2.3 预期成果
- 一个定理或定量命题：EVQ-cosh(τ) 将碰撞分数从 c 降低到 c - Δc(τ)
- Δc 的 leading term 表达式
- 实验验证：用 Phase 18 的 base=10K/500K 数据点检验

---

## 3. Base-Allocation 正交性的理论深化

### 3.1 当前叙事
"base 控制频带宽度，EVQ 控制频带内分配，两者独立"

### 3.2 需要升级的叙事
"base 和 allocation 不完全独立——EVQ 的频率重分配隐式缩小了有效 collision fraction，等价于一部分 base 增大效应。但 EVQ 的机制不同于 base 增大：base 增大同时稀释所有通道的分辨率（等比缩小间距），而 EVQ 选择性地只重分配低频区间距（保留高频分辨率）。"

### 3.3 加强方式

#### 理论：分解 EVQ 效应为 base-equivalent + residual
- EVQ 在 base=10K 的效果 ≈ Geo 在 base=?K 的效果
- 用 PPL 匹配法求"等效 base"：找到使 Geo PPL@4K ≈ EVQ@base=10K PPL@4K 的 base 值
- 如果 EVQ@10K ≈ Geo@~50K，说明 EVQ 提供了"等效 5× base 增大"
- **剩余的增益**（EVQ@10K 比 Geo@50K 还好的部分）是 allocation 优化的"纯增益"

#### 实验：细粒度 base sweep（需要更多计算资源）
- base = {10K, 30K, 50K, 100K, 300K, 500K}，all Geo
- 找到 Geo PPL@4K ≈ 192 对应的 base 值 → 这就是 EVQ 的"等效 base"
- **如果这个实验做出来**：可以写成论文的 Figure，x 轴 base，y 轴 PPL@4K，两条线（Geo 和 EVQ），EVQ 是一条几乎水平的线

---

## 4. 碰撞块理论的完整验证矩阵

### 4.1 当前验证点

| L | Base | c | EVQ 赢？ | 实验 |
|---|------|---|----------|------|
| 4096 | 10K | 0.90 | ❌ 全败 | 350M 旧实验 |
| 512 | 10K | 0.68 | ✅ -21.8% | Phase 18 |
| 512 | 500K | 0.48 | ✅ -32.6% | Phase 18 |
| 2048 | 500K | 0.58 | ✅ -13.3% | 350M 3-seed |

### 4.2 缺失的关键验证点

| L | Base | c | 预测 | 优先级 |
|---|------|---|------|--------|
| 1024 | 10K | 0.75 | EVQ 赢但增益小于 c=0.68 | P1 |
| 2048 | 10K | 0.83 | EVQ 可能微赢或持平 | P2 |
| 4096 | 100K | 0.79 | EVQ 赢 | P2 |
| 512 | 100K | 0.54 | EVQ 赢，增益大于 500K | P1 |

### 4.3 最高价值实验
**EVQ 增益 vs c 的定量关系图**：
- x 轴：collision fraction c
- y 轴：EVQ PPL improvement (%)
- 当前 4 个点，补到 6-8 个点后可以拟合 Δ PPL = f(c)
- 如果 f(c) 与理论预测的 (1-c)/ln(b) 吻合 → 理论完美闭合

---

## 5. "EVQ+YaRN 超线性协同 + EVQ 替代 base"的统一叙事

### 5.1 当前两个独立发现
1. EVQ+YaRN 在 progressive training 后超线性放大（Phase 17 系列）
2. EVQ 在不同 base 下性能几乎一致（Phase 18）

### 5.2 统一叙事草案

> **RoPE long-context optimization has three knobs: base (bandwidth), allocation (within-band distribution), and inference-time scaling (YaRN). Industry has only tuned base and YaRN, leaving allocation at its degenerate default (Geometric). We show:**
>
> 1. **Allocation optimization (EVQ) partially substitutes for base tuning** — EVQ achieves near-identical extrapolation PPL across a 50× base range, reducing collision dimensions by ~50%.
>
> 2. **Allocation optimization superlinearly amplifies progressive training** — three-stage training increases EVQ's advantage from 34.6% to 81.2%.
>
> 3. **Optimal allocation provides the ideal foundation for YaRN** — EVQ+YaRN extends to 24× training length with PPL ≤ 3.3 and 100% retrieval.
>
> **Together: EVQ is the missing optimization that makes all other RoPE improvements work better.**

### 5.3 如果完全证明后的论文 level 提升

当前叙事："EVQ 优化了被忽视的 allocation 维度" → **poster-level insight**

升级叙事："EVQ 是 RoPE 优化链的关键缺失环节，它同时增强 base tuning、progressive training 和 YaRN" → **spotlight-level system contribution**

完全证明需要：
- ✅ Progressive 放大（Phase 17 系列，已有）
- ✅ YaRN 协同（Phase 17c，已有）
- ⬜ Base 补偿（Phase 18，单 seed，需 multi-seed）
- ⬜ 定量理论（collision 维度缩减公式）
- ⬜ 等效 base 分析（EVQ@10K ≈ Geo@?K）

---

## 6. 执行优先级总结

| 优先级 | 任务 | 预计耗时 | 平台 | 预期产出 |
|--------|------|----------|------|----------|
| **P0** | Phase 18 multi-seed（base=10K+500K, seed=137,256） | ~2h | M4 Max | 升级 Phase 18 到 A- |
| **P0** | Collision 维度缩减理论推导 | ~2h（理论工作） | 纸笔/导师 | 新定理/命题 |
| **P1** | 补充 base=100K 点 | ~1h | M4 Max | 3 点 base-invariance |
| **P1** | 等效 base 分析（细粒度 Geo sweep） | ~3h | M4 Max | 论文级 Figure |
| **P2** | 极端 base（1M, 10M） | ~2h | M4 Max | 完整 5-base sweep |
| **P2** | L×base 交叉验证（c=0.75, 0.83 点） | ~3h | M4 Max | EVQ gain vs c 拟合 |

**建议执行顺序**：P0 multi-seed → P0 理论推导（可与导师讨论） → P1 base=100K → P1 等效 base → P2

---

## 7. 给导师的汇报要点

> 1. Phase 18 发现 EVQ 在 base=10K 也赢（PPL@4K -21.8%），翻转了之前的 dead zone 结论（之前是 L=4096/c=0.90，现在 L=512/c=0.68）
> 2. 更重要的发现：EVQ@base=10K ≈ EVQ@base=500K（PPL@4K: 192.39 vs 191.90），说明 EVQ 的频率优化可以部分替代 base tuning
> 3. Collision 维度分析：EVQ 把碰撞维度减少约一半，这是机制解释
> 4. 如果 multi-seed 确认 + 理论推导完成，论文叙事可以升级为"EVQ 是 RoPE 三维优化链的关键缺失环节"
> 5. 需要导师协助的：collision 维度缩减的形式化推导（从变分解出发）
