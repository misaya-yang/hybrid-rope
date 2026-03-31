# Phase 19: Progressive Chain 512→1024→2048 (125M, 3-seed, M4 Max)

> 状态: **PLANNED**
> 设备: M4 Max (~20GB), MPS backend
> 依赖: Phase 18 完成后执行（避免同时占用 MPS）
> 脚本: `scripts/mac_train/exp1_progressive_chain.py`

---

## 1. 动机

Phase 17b (454M, single-seed) 发现了三个重大结论：

1. **YaRN 相变**：512→1024 continuation 后，EVQ raw 反超 EVQ+YaRN（PPL@16K: 11.2 vs 16.8）
2. **Training-inference 等价**：evq_512+yarn ≈ evq_1024_cont raw（@16K: 11.6 vs 11.2）
3. **Progressive 放大**：EVQ vs Geo 优势从 34.6%→83.1%@16K

**问题**：全部是 454M 单 seed，reviewer 一定质疑 reproducibility。

Phase 19 用 125M × 3-seed 复现 + 延伸到 2048，一次实验填三个 gap：

| Gap | Phase 19 怎么填 |
|-----|----------------|
| YaRN 相变单 seed | 125M × 3 seeds 验证 |
| 只有 2 个 progressive 数据点 | 加 2048 stage → 3 个点 |
| 只在 454M 验证 | 125M cross-scale 验证 |

---

## 2. 实验设计

### 2.1 Progressive Chain

| Stage | L_train | Tokens | 来源 |
|-------|---------|--------|------|
| Stage 0 | 512 | 50M | From scratch |
| Stage 1 | 1024 | 25M | Continue from Stage 0 |
| Stage 2 | 2048 | 25M | Continue from Stage 1 |

### 2.2 配置

- 模型: 125M (hidden=768, 12L, 12H, d_head=64)
- Base: 500K
- 方法: Geo (τ=0) vs EVQ (τ*=d/√L_current)
  - Stage 0: τ* = 64/√512 = 2.83
  - Stage 1: τ* = 64/√1024 = 2.0
  - Stage 2: τ* = 64/√2048 = 1.41
- Seeds: 42, 137, 256

### 2.3 评估

每个 stage 完成后评估：
- **Raw PPL**: @{512, 1K, 2K, 4K, 8K, 16K, 32K}
- **+YaRN PPL**: 同上（NTK-aware scaling）
- 不跑 passkey（125M 太小，passkey 噪声大）

### 2.4 Run 数量

- 2 methods × 3 seeds × 3 stages = **18 training runs**
- 每个 stage ~10-15 min → **总计约 3-4.5 小时**

---

## 3. 关键验证点（按重要性）

### 3.1 YaRN 相变是否复现（P0）

在 Stage 1 (L=1024 continuation) 之后：
- [ ] EVQ raw PPL@16K < EVQ+YaRN PPL@16K?（3/3 seeds?）
- [ ] 相变点是否一致：在 1024 就发生，还是需要更长训练？

454M 参考值：EVQ raw 11.2 < EVQ+YaRN 16.8 ✓

### 3.2 Progressive 放大是否延续（P0）

EVQ vs Geo raw advantage 随 stage 的变化：
- [ ] Stage 0 → Stage 1 → Stage 2: advantage 单调递增?

454M 参考值：34.6% → 83.1%（两点，单调）

### 3.3 2048 后趋势是否继续（P1）

Stage 2 (L=2048) 之后：
- [ ] EVQ raw 是否继续优于 EVQ+YaRN?
- [ ] EVQ vs Geo 优势是否进一步扩大?

如果 2048 继续扩大 → 趋势线有 3 个点，可以拟合。
如果 2048 开始收敛 → 也有意义，说明存在 saturation point。

### 3.4 Training-inference 等价（P1）

- [ ] evq_stage0+yarn ≈ evq_stage1 raw? (类比 454M 的 11.6 vs 11.2)
- [ ] evq_stage1+yarn ≈ evq_stage2 raw? (延伸验证)

---

## 4. 预期结果与论文贡献

### 4.1 Best case（全部复现 + 延伸）

论文新增一张图 / 一个表：

```
Figure X: Progressive training amplifies EVQ advantage and renders YaRN unnecessary.
(a) EVQ vs Geo raw PPL@16K across 3 progressive stages (125M, 3-seed mean ± std)
(b) EVQ raw vs EVQ+YaRN: phase transition from synergy to competition
```

可以说：
- "The YaRN phase transition is reproducible across scales (125M, 454M)"
- "Progressive training monotonically amplifies EVQ's raw advantage"
- "After 2 stages of progressive training, EVQ raw alone matches or exceeds EVQ+YaRN from the first stage"

### 4.2 Worst case（125M 不复现）

如果 125M 不出现 YaRN 相变：
- 可能是 125M 容量不够内化频率分配
- 叙事调整为 "需要足够模型容量才能实现 internalization"
- 不影响 454M 结论，但降低 generality claim 的强度

### 4.3 论文位置

- 如果复现：主文 §5 或 §6（Training Curriculum Ablation）
- 如果不复现：Appendix（scale-dependent observation）

---

## 5. 执行

### 5.1 前置条件

- [ ] Phase 18 完成（避免同时占用 MPS）
- [ ] 数据缓存就绪（首次运行会自动下载）

### 5.2 命令

```bash
cd ~/neurIPS-2026/hybrid-rope
conda activate aidemo

# Pilot（seed=42 only，验证能跑）
python scripts/mac_train/exp1_progressive_chain.py --pilot

# 全量（3 seeds）
python scripts/mac_train/exp1_progressive_chain.py

# 汇总
python scripts/mac_train/exp1_progressive_chain.py --summary
```

### 5.3 输出

```
results/mac_train/exp1_progressive_chain/
├── geo_seed42/
│   ├── stage0_L512/   {result.json, model.pt}
│   ├── stage1_L1024/  {result.json, model.pt}
│   └── stage2_L2048/  {result.json, model.pt}
├── evq_seed42/
│   ├── ...
├── geo_seed137/
├── evq_seed137/
├── geo_seed256/
├── evq_seed256/
├── data_cache/
└── exp1_summary.json
```

### 5.4 断点续跑

脚本检查每个 stage 的 result.json，已完成的自动跳过。中途断了直接重新执行。

---

## 6. Claude Code 提示词

```
你是一个ML实验工程师。请运行 Phase 19 progressive chain 实验。

环境：M4 Max, MPS backend, conda activate aidemo

步骤：
1. cd ~/neurIPS-2026/hybrid-rope
2. 先 pilot 验证：
   python scripts/mac_train/exp1_progressive_chain.py --pilot
3. pilot 成功后全量：
   python scripts/mac_train/exp1_progressive_chain.py
4. 完成后生成汇总：
   python scripts/mac_train/exp1_progressive_chain.py --summary

脚本支持断点续跑。每个 stage 完成后报告 PPL 结果。
重点关注：EVQ raw 是否在 Stage 1 之后反超 EVQ+YaRN。
```
