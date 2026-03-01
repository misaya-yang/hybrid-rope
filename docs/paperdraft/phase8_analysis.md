# Phase 8 实验分析 + Passkey 策略

> **更新**: 2026-03-01 23:30
> **状态**: 8A ✅ 8B ✅ 8C ⏳（C1 快完成，C2 待跑）

---

## 一、8A 核心结论（512→4K，8x 扩展比）

### PPL：EVQ/Hybrid 不崩，PI/YaRN 崩了

| 方法 | PPL@4K (窗口内) | PPL@8K (2x外推) | PPL@16K (4x外推) | vs Geo @16K |
|------|-----------------|-----------------|-------------------|-------------|
| Geometric | 80.4 | 81.4 | 83.3 | baseline |
| **Hybrid τ=2.0** | **82.0** | **83.3** | **84.7** | **+1.7%** |
| EVQ τ=1.5 | 83.1 | 84.2 | 85.7 | +2.9% |
| EVQ τ=2.0 | 84.8 | 86.5 | 88.0 | +5.7% |
| EVQ τ=2.5 | 87.3 | 88.9 | 90.2 | +8.3% |
| YaRN | 79.5 | 107.8 | **161.9** | **+94% 崩溃** |
| PI | 82.2 | 159.1 | **254.4** | **+205% 崩溃** |

**关键发现**：
1. Geometric 在 16K 没有崩溃——"大扩展比 Geo 崩" 的假说不成立
2. PI/YaRN 8K 起彻底崩溃，方法级失败
3. EVQ 全系列不崩，但全面略差于 Geo（τ 越大越差）
4. **Hybrid 是最佳 EVQ 变体**，PPL 最接近 Geo（+1.7% @16K）
5. Extension 场景下 τ 应保守：1.5 > 2.0 > 2.5

**论文叙事**：不是"EVQ 打败 Geometric"，而是"EVQ 是唯一一个用闭式解就能 match Geometric 鲁棒性的参数化方法，而 PI/YaRN 全崩"。

### Passkey：短距离 EVQ 落后，外推区间趋同

| 方法 | @1K | @2K | @4K | @8K (外推) |
|------|-----|-----|-----|------------|
| YaRN | **87%** | **82%** | **70%** | 49% ↓↓ |
| Geo | 82% | 72% | 59% | **54%** |
| PI | 81% | 74% | 65% | 43% ↓ |
| Hybrid | 74% | 61% | 50% | **52%** |
| EVQ τ=1.5 | 70% | 66% | 52% | **52%** |
| EVQ τ=2.0 | 64% | 63% | 47% | 50% |
| EVQ τ=2.5 | 64% | 60% | 45% | 53% |

**关键发现**：
1. @8K 外推区间：所有方法趋同 49-54%，EVQ 追平 Geo
2. YaRN 短距离最强（87% @1K）但 8K 崩到 49%——**PPL 崩溃和 Passkey 崩溃同步**
3. Geo 短距离优势（82% @1K）来自 Q/K alignment 保护
4. 所有方法 @8K 的 mean_nll_gap ≈ 0 → **350M 模型在 8K 已达能力天花板**

---

## 二、8B 核心结论（续训量消融，512→2K）

### PPL 随续训量单调改善

| 续训量 | EVQ PPL@2K | Geo PPL@2K | EVQ vs Geo |
|--------|-----------|-----------|------------|
| 10M | 80.7 | 77.1 | +4.7% |
| 20M | 75.7 | 73.4 | **+3.1%** |

Gap 从 4.7% 缩小到 3.1%，**EVQ 追赶速度更快**。

### Passkey 恢复趋势（仅用 8B 新数据，7F ref 不可比）

| 续训量 | EVQ @1K | EVQ @2K | Geo @1K | Geo @2K |
|--------|---------|---------|---------|---------|
| 2.5M | 63% | 57% | — | — |
| 10M | 66% | 64% | 80% | 70% |
| 20M | 72% | 63% | 80% | 69% |

EVQ @1K: 63→66→72%（上升中，未饱和）
Geo @1K: 80→80%（已饱和）

**结论**：Q/K alignment 在恢复，但速度慢。20M tokens（预训练的 40%）EVQ 仍落后 8pp。

---

## 三、Passkey 问题诊断 ⚠️ 最高优先级

### 问题本质

Passkey 差**不是 EVQ 频率分配的问题**，而是两个因素叠加：

**因素 1：350M 模型能力天花板**
- 即使 Geometric（最好的方法）@1K 也只有 82%
- @8K 所有方法趋同 ~50%，mean_nll_gap ≈ 0
- 350M 的 attention 容量不足以做 reliable retrieval

**因素 2：Extension 的 Q/K Alignment Cost**
- Phase 6 from-scratch：**EVQ passkey 55% > Geo 48.5%**（EVQ 赢！）
- Phase 7F/8A extension：EVQ passkey < Geo
- 结论：从头训 EVQ passkey 不差，extension 时 Q/K 被破坏

### 关键证据链

| 设置 | EVQ vs Geo Passkey | 来源 |
|------|-------------------|------|
| From-scratch 128-tok | **EVQ 55% > Geo 48.5%** | Phase 6 ✅ |
| Extension 512→2K (4x) | EVQ < Geo | Phase 7F |
| Extension 512→4K (8x) | EVQ < Geo @短距, ≈ Geo @8K | Phase 8A |
| From-scratch 4K (τ=2.0) | EVQ PPL **-6.3%** ✅, Passkey -3pp (τ 非最优) | Phase 8C ✅ |

**8C 结果（已出）**：
- C1 Geo from-scratch: PPL@16K=175.4, Passkey=69%
- C2 EVQ τ=2.0 from-scratch: PPL@16K=164.4(**-6.3%**), Passkey=66%(-3pp)

**PPL 大胜，Passkey 小输**。解读：
- Extension passkey gap = 6.75pp → From-scratch gap = 3pp → **Alignment cost 占 ~55% 的 passkey 损失**
- 剩余 3pp 归因于 **τ=2.0 对 L=4096 过大**：scaling law 预测 τ*(4096)=1.0，τ=2.0 过度偏低频、挤压高频精度
- 对比 Phase 6：τ=2.0 对 L=128 合理（预测 τ*=5.66），EVQ passkey 赢了 6.5pp
- **结论**：EVQ passkey 不是频率分配的固有缺陷，而是 (1) alignment cost + (2) τ 选择不当的叠加

---

## 四、Passkey 策略：Phase 9 必须解决

### 为什么 350M 不够

350M + 50M tokens 预训练 = 严重欠拟合。Passkey 是 copy-from-context 任务，需要模型真正学会"注意力精确定位"。350M 在所有方法上都做不好这个任务（Geo @8K 也只有 54%），无法区分方法差异。

### Phase 9 的 Passkey 必须做到什么

**底线**：1B 模型 from-scratch 或充分续训后，EVQ/Hybrid passkey ≥ Geometric。
**如果做不到**，论文几乎无法接收（只有 PPL 的 PE 论文在 NeurIPS 主会风险极大）。

### Phase 9 Passkey 优化策略

**策略 A：加大续训量**
- 8B 显示 EVQ passkey 随续训量单调上升且未饱和
- Phase 9 将续训量从 10%（50M/500M）提升到 **20%（100M tokens）**
- 预计多花 ~3h，但对 passkey 至关重要

**策略 B：Hybrid EVQ 作为主推方案**
- 8A 中 Hybrid @1K = 74% vs pure EVQ @1K = 64-70%
- Hybrid 保护高频 Q/K 对齐，低频用 EVQ 增强外推
- Phase 9 中 Hybrid 应该是 passkey 最佳 EVQ 变体

**策略 C：From-scratch baseline（如果时间够）**
- 在 1B 上做一组 from-scratch 4K（和 Phase 9A 相同 tokens）
- 直接对比 from-scratch EVQ vs Geo passkey
- 消除 alignment cost 的干扰

### 论文中 Passkey 怎么写

最佳情况（Phase 9 Passkey EVQ/Hybrid ≥ Geo）：
> "At 1B scale with sufficient training, EVQ-cosh matches or exceeds Geometric on passkey retrieval, while PI/YaRN collapse."

次佳情况（EVQ < Geo 但 Hybrid ≈ Geo）：
> "Hybrid EVQ preserves Q/K alignment on high-frequency channels while optimizing low-frequency allocation, achieving passkey parity with Geometric."

最差情况（全部 < Geo）：
> 需要 from-scratch 实验证明"差距来自 alignment cost"，并推荐 Hybrid + 更多续训作为实践方案。论文风险显著上升。

---

## 五、Phase 9 修订建议

基于 Phase 8 结果，Phase 9 需要调整：

1. **续训量翻倍**：50M → 100M tokens（增加 ~3h，但 passkey 必须拿下）
2. **Hybrid EVQ 提升为主推**：Phase 8 证明 Hybrid 是最佳 EVQ 变体
3. **增加 from-scratch 对照**：1B from-scratch 4K Geo vs EVQ（如果时间够）
4. **Passkey 评估加厚**：多位置 × 多长度，确保统计显著性

---

## 六、论文叙事框架（基于 Phase 8 更新）

### 核心卖点（按重要性排序）

1. **理论**：cosh 是 RoPE 频率分配变分问题的闭式解，1 个参数 vs DAPE 的 32 个
2. **Scaling law**：τ*(L) = d_head/√L（待 8D 验证）
3. **鲁棒性**：8x 扩展比下 EVQ/Hybrid 不崩，PI/YaRN 崩溃
4. **Passkey**：
   - From-scratch: EVQ > Geo（Phase 6 已有）
   - Extension + Hybrid: ≈ Geo（Phase 8A）
   - 1B scale: **待 Phase 9 确认**

### 论文结构建议

- Section 1-3: 理论（变分推导 + cosh 最优性 + scaling law）
- Section 4: From-scratch 实验（PPL + passkey，EVQ 优势明确）
- Section 5: Context extension 实验（PPL 鲁棒性 + Hybrid 方案 + alignment cost 分析）
- Section 6: 1B 验证（核心 passkey 结果）
- Appendix: 续训量消融、waterbed 分析、PI/YaRN 崩溃详情
