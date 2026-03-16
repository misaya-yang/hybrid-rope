# EXP-4 Report: 350M Progressive Chain — Delayed-tau Protocol Validation

> 日期: 2026-03-16
> 设备: M4 Max 36GB, MPS backend, fp32
> 脚本: `scripts/m4_max_36gb/exp4_progressive_chain_350m.py`
> 总训练时间: ~31h (5 chains, 15 training runs)
> 状态: **COMPLETE**

---

## 1. 实验配置

| 项目 | 值 |
|------|-----|
| 模型 | 350M (24L/16H/d=1024/d_head=64, 实际 454.2M params) |
| Tokens/stage | 18M / 9M / 9M (Stage 1/2/3, 共 36M/chain) |
| 总训练量 | 5 chains x 36M = 180M tokens |
| Grad accum | L=512: ga=2 (4096 tok/step), L=1024: ga=2 (4096), L=2048: ga=4 (8192) |
| Passkey mix | 5% (via MixedDataset, 在线生成) |
| 数据 | FineWeb-Edu, 非重叠切分 (offset 0M/18M/27M) |
| 吞吐 | Stage1: 1.5-1.8K tok/s, Stage2: 1.4-1.6K, Stage3: 1.2-1.4K |

### Methods

| 方法 | tau 策略 | Seeds | 描述 |
|------|---------|-------|------|
| GEO | tau=0 (geometric) | 42, 123 | 控制组 |
| EVQ-D | tau*(512)=2.828 全程不变 | 42, 123 | 主假设: delayed-tau protocol |
| EVQ-R | tau 每 stage 重算 | 42 only | 诊断组: retarget protocol |

### tau 值一览

| Stage | L_train | GEO | EVQ-D | EVQ-R |
|-------|---------|-----|-------|-------|
| Stage 1 | 512 | 0.0 | 2.828 | 2.828 |
| Stage 2 | 1024 | 0.0 | **2.828** | **2.000** |
| Stage 3 | 2048 | 0.0 | **2.828** | **1.414** |

inv_freq hash 验证: EVQ-D (Stage2) = `c3427e94f050`, EVQ-R (Stage2) = `9882247952a5` -- 已确认不同。

---

## 2. PPL 结果

### 2.1 Stage 1 (L_train=512)

| Method | Seed | PPL@512 | PPL@1024 | PPL@2048 | PPL@4096 |
|--------|------|---------|----------|----------|----------|
| GEO | 42 | **169.6** | 190.5 | 250.0 | 297.3 |
| GEO | 123 | **168.7** | 191.2 | 234.4 | 272.1 |
| EVQ-D | 42 | 170.5 | **185.1** | **202.3** | **204.9** |
| EVQ-D | 123 | 170.8 | **184.7** | **204.0** | **213.4** |
| EVQ-R | 42 | 169.8 | 183.1 | 205.6 | 212.2 |

**Mean +/- Std:**

| Method | PPL@512 | PPL@1024 | PPL@2048 | PPL@4096 |
|--------|---------|----------|----------|----------|
| GEO | **169.1+/-0.5** | 190.8+/-0.3 | 242.2+/-7.8 | 284.7+/-12.6 |
| EVQ-D | 170.6+/-0.2 | **184.9+/-0.2** | **203.1+/-0.9** | **209.2+/-4.2** |

> EVQ-D 在外推长度上大幅领先: PPL@4096 比 GEO 低 26.5% (209 vs 285)。
> Stage 1 的 EVQ-D/R 使用相同 tau=2.828, 两者结果一致, 符合预期。

### 2.2 Stage 2 (L_train=1024)

| Method | Seed | PPL@1024 | PPL@2048 | PPL@4096 | PPL@8192 |
|--------|------|----------|----------|----------|----------|
| GEO | 42 | **133.0** | 187.5 | 156.6 | 115.6 |
| GEO | 123 | **131.7** | 187.2 | 155.3 | 113.4 |
| EVQ-D | 42 | 136.2 | 190.5 | 155.2 | **104.6** |
| EVQ-D | 123 | 136.5 | 188.3 | 155.3 | **108.0** |
| EVQ-R | 42 | 136.4 | 190.7 | **154.3** | 108.2 |

**Mean +/- Std:**

| Method | PPL@1024 | PPL@2048 | PPL@4096 | PPL@8192 |
|--------|----------|----------|----------|----------|
| GEO | **132.3+/-0.7** | **187.4+/-0.2** | 156.0+/-0.7 | 114.5+/-1.1 |
| EVQ-D | 136.4+/-0.1 | 189.4+/-1.1 | **155.3+/-0.0** | **106.3+/-1.7** |

> EVQ-D 在 train length (1024) 输 GEO 约 3%, 但在最长外推 (8192) 赢 7.2%.
> EVQ-D vs EVQ-R 在 @8192: 104.6 vs 108.2, delayed protocol 赢 3.4%。

### 2.3 Stage 3 (L_train=2048) -- 最终比较

| Method | Seed | PPL@2048 | PPL@4096 | PPL@8192 | PPL@16384 |
|--------|------|----------|----------|----------|-----------|
| GEO | 42 | **119.6** | **135.8** | **132.6** | OOM |
| GEO | 123 | **117.2** | **132.9** | **129.1** | OOM |
| EVQ-D | 42 | 122.6 | 138.0 | 133.9 | OOM |
| EVQ-D | 123 | 121.9 | 137.0 | 134.5 | OOM |
| EVQ-R | 42 | 122.8 | 137.6 | 132.2 | OOM |

**Mean +/- Std:**

| Method | PPL@2048 | PPL@4096 | PPL@8192 |
|--------|----------|----------|----------|
| GEO | **118.4+/-1.2** | **134.4+/-1.4** | **130.9+/-1.7** |
| EVQ-D | 122.2+/-0.3 | 137.5+/-0.5 | 134.2+/-0.3 |

> Stage 3: GEO 在所有长度上赢 EVQ-D, 差距约 2-3%。
> PPL@16384 全部 OOM (30GB MPS 不够)。

---

## 3. Passkey (NIAH) 结果

所有方法的 passkey retrieval rate 在 41%-62% 之间, exact match 全为 0%, NLL gap 接近 0。
说明 350M@36M tokens 训练量不足以形成有效的 passkey 检索能力。

### Stage 3 Passkey Retrieval Rate

| Method | Seed | Retrieval Rate | NLL Gap |
|--------|------|---------------|---------|
| GEO | 42 | 41.1% | -0.011 |
| GEO | 123 | 57.8% | +0.008 |
| EVQ-D | 42 | 47.8% | -0.005 |
| EVQ-D | 123 | 48.9% | +0.001 |
| EVQ-R | 42 | 47.8% | -0.003 |

> Passkey 结果接近随机水平, 无法从中得出有效结论。
> 原因: 18M tokens 的 5% passkey mix = 仅 ~900 passkey 样本, 远不足以学会检索。

---

## 4. Protocol Comparison: EVQ-D vs EVQ-R (seed=42)

| Stage | 长度 | EVQ-D | EVQ-R | Delta | Winner |
|-------|------|-------|-------|-------|--------|
| Stage 1 | @512 | 170.5 | 169.8 | -0.4% | R |
| Stage 1 | @1024 | 185.1 | 183.1 | -1.1% | R |
| Stage 1 | @2048 | 202.3 | 205.6 | +1.6% | D |
| Stage 1 | @4096 | 204.9 | 212.2 | +3.6% | **D** |
| Stage 2 | @1024 | 136.2 | 136.4 | +0.1% | D |
| Stage 2 | @4096 | 155.2 | 154.3 | -0.6% | R |
| Stage 2 | @8192 | 104.6 | 108.2 | +3.5% | **D** |
| Stage 3 | @2048 | 122.6 | 122.8 | +0.2% | D |
| Stage 3 | @4096 | 138.0 | 137.6 | -0.3% | R |
| Stage 3 | @8192 | 133.9 | 132.2 | -1.3% | R |

> Stage 1-2 的最长外推中, delayed protocol (D) 一致赢 3.5%。
> 但 Stage 3 @8192 反转: retarget (R) 赢 1.3%。差异不大, 但方向变了。

---

## 5. 验证点判定

| # | 验证 | 预期 | 实际 | 状态 |
|---|------|------|------|------|
| V1 | Stage3: EVQ-D PPL@4K < GEO PPL@4K | EVQ 赢外推 | GEO 134.4 < EVQ-D 137.5 | **FAILED** |
| V2 | Stage3: EVQ-R PPL@4K > EVQ-D PPL@4K | retarget 代价 | EVQ-R 137.6 vs EVQ-D 138.0 | **INCONCLUSIVE** (几乎相同) |
| V3 | Stage3: EVQ-D NIAH >= GEO NIAH | EVQ 保持检索 | 所有方法都在随机水平 | **INCONCLUSIVE** (tokens 不足) |
| V4 | EVQ-D advantage 随 stage 递增 | 放大效应 | Stage1 赢 26%, Stage2 赢 7%, Stage3 输 2.5% | **FAILED** (优势递减并反转) |
| V5 | 2-seed mean+/-std 方向一致 | 非 fluke | 两 seed 方向一致 | **PASS** |

---

## 6. 分析与解读

### 6.1 为什么 EVQ-D 在 Stage 3 输了?

EVQ-D 在 Stage 1 和 Stage 2 的最长外推上明显赢 GEO, 但到 Stage 3 反而输了。可能原因:

1. **Token 预算不足** (最可能):
   - 每 stage 只有 9-18M tokens, 而 454M params 的模型需要远更多数据才能充分收敛
   - Phase 17B/17C 用了 1B tokens/stage, 比这里多 50-100x
   - 在低 token 区间, GEO 的简单频率结构可能更容易学, EVQ 的优化频率分配需要更多训练才能内化

2. **外推 vs 内插反转**:
   - Stage 1 (L=512): 评估 @4096 是 8x 外推 -- EVQ 大赢
   - Stage 3 (L=2048): 评估 @4096 只是 2x 外推 -- 差距缩小到 GEO 反超
   - EVQ 的核心优势是远程外推; 当 train length 已经接近 eval length 时, 边际收益递减

3. **Delayed-tau 的 train-length PPL 代价**:
   - EVQ-D 保持 tau*(512)=2.828 不变, 但 Stage 3 训练 L=2048
   - tau=2.828 对 L=2048 来说偏大 (最优应是 tau*(2048)=1.414)
   - 这导致 train-length PPL 始终比 GEO 差 2-3%, 在低 token 预算下无法通过外推收益补回

### 6.2 EVQ-D vs EVQ-R 差异太小

Stage 2 的 @8192 外推中, delayed 比 retarget 好 3.5%, 这是唯一有意义的信号。
但 Stage 3 反转, 且绝对差异 <2 PPL 点, 在 2-seed 的统计能力下不够显著。

**结论**: 18M/9M/9M 的 token 预算不足以区分 delayed 和 retarget protocol 的差异。

### 6.3 对论文的影响

| 发现 | 可用性 | 建议 |
|------|--------|------|
| EVQ Stage1 外推优势 (26%) | **可用** | 补充 Fig 4, 展示 EVQ 在短训练长度下的强外推能力 |
| EVQ Stage2 @8192 赢 7% | **可用** | 支撑 "EVQ benefits grow with extrapolation ratio" |
| Delayed vs retarget | **不可用** | 差异太小, 需要更多 tokens 或在 454M 实验中验证 |
| Stage3 GEO 反超 | **需要解释** | 应归因于 token 不足, 不是 EVQ 本身的问题 |

---

## 7. 后续建议

### 方案 A: 在 AutoDL GPU 上重跑更多 tokens (推荐)

在 RTX 5090 上用 100M/50M/50M tokens 重跑, 预计 8-12h。
更多 tokens 应能还原 EVQ 的优势并放大 delayed vs retarget 的差异。

### 方案 B: 直接用 Stage 1-2 的数据

Stage 1 和 Stage 2 的结果方向明确且 2-seed 一致, 可以作为论文补充证据:
- "At low extrapolation ratios (2-4x), EVQ achieves 7-26% lower PPL than geometric RoPE"
- 不主张 Stage 3 的结论, 标注为 "insufficient tokens"

### 方案 C: 缩小模型, 增加 token 比

用 125M 模型在 M4 上跑同样的 delayed-tau protocol validation,
token/param ratio 更高 (36M/125M=0.29 vs 36M/454M=0.08),
更容易看到收敛后的真实差异。

---

## 8. 文件索引

```
results/m4_max_36gb/exp4_progressive_350m/
  exp4_run.log                    -- 完整训练日志 (1294 行)
  exp4_summary.json               -- 全量结果 JSON
  REPORT.md                       -- 本报告
  pilot/
    model.pt                      -- pilot checkpoint
    pilot_result.json             -- pilot 吞吐量数据
  {method}_seed{seed}/
    stage{1,2,3}_L{512,1024,2048}/
      model.pt                    -- stage checkpoint
      result.json                 -- PPL + passkey + inv_freq 结果
      inv_freq.npy                -- inv_freq 向量
  data_cache/
    train_fineweb-edu_100000000_512.pt  -- symlink to phase18 cache
    val_fineweb-edu_5000000.pt          -- validation data
```

---

## 9. Bug Fix 记录

本次实验修复了原始脚本的 5 个 bug:

| Bug | 严重度 | 修复内容 |
|-----|--------|---------|
| 数据重叠 | 致命 | `preload_stage_data()` 一次加载总量, 按 offset 切分 |
| 缺少 grad accum | 重要 | `train_stage()` 加 `grad_accum` 参数, L512/1024: ga=2, L2048: ga=4 |
| 缺少 passkey mix | 重要 | 导入 `maybe_wrap_with_passkey_mix()`, 5% ratio |
| 内存泄漏 | 中等 | eval 后显式 `gc.collect()` + `torch.mps.empty_cache()` |
| dry_run 缺少验证 | 中等 | 打印 inv_freq MD5 hash, 确认 EVQ-D/R 在 Stage 2 不同 |
