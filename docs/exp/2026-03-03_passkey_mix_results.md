# Passkey Mix 实验结果（2026-03-03）

> **状态**：PENDING（10% seed=42 完成，5% seed=42,123 完成；等待多 seed 确认）
> **服务器**：5090 32GB, bf16, SDPA
> **模型**：350M (454.2M params), 24层, head_dim=64, base=500K
> **训练**：100M tokens FineWeb-Edu, seq_len=2048, lr=2e-4, cosine schedule
> **总 token 量相同**：10% 和 5% 实验均为 100M tokens，仅 passkey 信号浓度不同

---

## 1. 核心数据表

### 1.1 全量结果（当前可用 runs）

| Mix | Method | Seed | PK@2K | PK@4K | PK@8K | PK_Global | PPL@2K | PPL@4K | PPL@8K | PPL@16K |
|-----|--------|------|-------|-------|-------|-----------|--------|--------|--------|---------|
| 10% | Geo | 42 | 100% | **42%** | 46% | 63% | 67.4 | 94.9 | 156.5 | 251.9 |
| 10% | EVQ | 42 | 100% | **82%** | 60% | 81% | 68.0 | 95.3 | 152.5 | 240.8 |
| 5% | Geo | 42 | 100% | 64% | 56% | 73% | 63.7 | 95.3 | 152.8 | 247.5 |
| 5% | EVQ | 42 | 100% | 60% | 56% | 72% | 65.1 | 88.2 | 138.8 | 220.5 |
| 5% | Geo | 123 | 100% | 66% | 42% | 69% | 65.0 | 96.8 | 169.5 | 276.0 |

> 10% PI baseline (推理时 Position Interpolation): PK_Global=51%, PPL@2K=191.7 — 几乎退化到随机

### 1.2 EVQ vs Geo Delta（同 mix ratio 下）

| Mix | PK@2K | PK@4K | PK@8K | PK_Global | PPL@2K | PPL@4K | PPL@8K | PPL@16K |
|-----|-------|-------|-------|-----------|--------|--------|--------|---------|
| **10%** | +0pp | **+40pp** | +14pp | +18pp | +0.8% | +0.5% | -2.6% | -4.4% |
| **5%** | +0pp | -4pp | +0pp | -1pp | +2.3% | -7.5% | -9.2% | -10.9% |

### 1.3 10% vs 5% Delta（同 method 下，more passkey data 的效应）

| Method | PK@2K | PK@4K | PK@8K | PK_Global | PPL@2K | PPL@4K | PPL@8K | PPL@16K |
|--------|-------|-------|-------|-----------|--------|--------|--------|---------|
| **Geo** | +0pp | **-22pp** | -10pp | -11pp | +5.8% | -0.5% | +2.4% | +1.8% |
| **EVQ** | +0pp | **+22pp** | +4pp | +9pp | +4.3% | +8.1% | +9.9% | +9.3% |

---

## 2. 核心发现

### Finding 1: EVQ 在 4K 外推检索上 +40pp（10% mix, seed=42）

- 2K（训练长度内）：两者都是 100%，说明检索技能已充分学会
- 4K（2x 外推）：Geo 42% vs EVQ 82%，差距 +40pp
- 8K（4x 外推）：Geo 46% vs EVQ 60%，差距 +14pp
- **解读**：EVQ 的频率分配让训练长度内学到的检索技能能更好地泛化到更长上下文

### Finding 2: Geo 和 EVQ 对 passkey 信号的 scaling behavior 截然相反

这是最强的发现。相同总 token 量（100M），仅 passkey 浓度从 5%→10%：

- **Geo 4K 检索退化 -22pp**（64%→42%）：更多 passkey 训练让 Geo 过拟合到 2K 长度的检索模式，position encoding 的固有局限使泛化能力反而下降
- **EVQ 4K 检索提升 +22pp**（60%→82%）：EVQ 的频率分配让模型能将更多检索训练信号转化为外推泛化能力

**对称性**：Geo 每多看 1% passkey 数据，4K 检索约 -4.4pp；EVQ 每多看 1% passkey 数据，4K 检索约 +4.4pp。方向完全相反。

### Finding 3: Waterbed 不等式成立

10% mix 下（最清晰的信号）：
- 短端（2K）：PPL +0.8%（微涨，符合 waterbed）
- 长端（16K）：PPL -4.4%（改善）
- Waterbed score: ✓

5% mix 下 EVQ 的 PPL 优势更大（-10.9%@16K），但这部分来自 passkey 污染更少。

### Finding 4: 5% mix 下两者差异消失

5% 的 passkey 信号不够强，EVQ 和 Geo 的检索外推差异仅 -4pp（不显著）。
这说明需要足够的训练信号浓度（≥10%），EVQ 的优势才能体现。

---

## 3. 实验严谨性

### 3.1 公平性
- 总 token 量相同（100M），模型大小相同（454.2M params）
- 同一 tokenizer（GPT-NeoX-20B），同一数据源（FineWeb-Edu）
- 同一训练超参（lr=2e-4, cosine decay, warmup 2%, gradient clip 1.0）
- 唯一差异：RoPE inv_freq（Geometric vs EVQ-Cosh with τ=1.5）

### 3.2 待确认
- [ ] 10% 多 seed（123, 7）确认 +40pp 非偶然 → 已排队，预计 4h
- [ ] 5% 多 seed（123 EVQ + seed 7 全部）确认 5% 下差异不显著 → 进行中
- [ ] PE baselines（PI/YaRN/NTK-aware）对比 → 已排队

### 3.3 潜在 confound
- Passkey 评估使用 NLL gap 方法（teacher-forcing），10 trials per (length, depth)，可能方差较大
- 单 seed 结论不稳定（5% Geo seed=123 的 8K 是 42%，比 seed=42 的 56% 低 14pp）
- 需要 ≥3 seeds 的均值 ± std 才能写进论文

---

## 4. 论文论点（待多 seed 确认后定稿）

> **Claim**: EVQ's frequency allocation enables sample-efficient generalization of learned retrieval skills beyond the training context length. Under identical training budgets (100M tokens), increasing passkey supervision from 5% to 10% yields opposite effects:
> - **Geometric RoPE degrades** at 4K extrapolation (-22pp), overfitting to training-length retrieval patterns
> - **EVQ-Cosh improves** at 4K extrapolation (+22pp), converting additional supervision into generalization
>
> This asymmetric scaling behavior demonstrates that frequency allocation quality — not data quantity — is the bottleneck for length generalization.

---

## 5. 后续实验队列（5090 上自动执行）

| 序号 | 任务 | 状态 | 预计耗时 |
|------|------|------|----------|
| 0 | 5% × 3 seeds (tau=0,1.5) | 运行中 (4/6 done, ~2h remaining) | 6.5h total |
| 1 | PE baselines (Geo/PI/YaRN/NTK/DynNTK) | 排队 | ~15min |
| 2 | 10% × 2 seeds (123, 7) replication | 排队 | ~4h |

队列脚本 `run_queue.sh` 自动监控并依次执行。

---

## 6. 原始数据位置

- 10% results: `/root/autodl-tmp/evq_passkey_mix_10pct/results_final.json`
- 5% results: `/root/autodl-tmp/evq_passkey_mix_5pct/results_checkpoint.json`
- 10% model checkpoints: `/root/autodl-tmp/evq_passkey_mix_10pct/350m_tau{X}_seed{Y}/model.pt`
- 5% model checkpoints: `/root/autodl-tmp/evq_passkey_mix_5pct/350m_tau{X}_seed{Y}/model.pt`
- PE baselines script: `/root/autodl-tmp/hybrid-rope/scripts/m4_evq_sweep/eval_pe_baselines.py`
