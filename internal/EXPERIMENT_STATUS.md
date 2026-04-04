# EVQ-Cosh 实验状态总览
> **最后更新**: 2026-03-22
> **目的**: 一目了然看清所有实验的状态、数据完整度、以及在论文中的角色

---

## 铁证 (Iron-clad, multi-seed)

| # | 实验 | 模型 | Seeds | 状态 | 核心结果 | 论文角色 |
|---|------|------|-------|------|---------|---------|
| 1 | 350M from-scratch L=2048 | 350M MHA | 3 (42/137/256) | ✅ 完成 | PPL@16K -13.3% | §5.2 Cross-scale |
| 2 | Passkey mix 10% | 350M MHA | 6 (3+3) | ✅ 完成 | EVQ+YaRN@8K 100% (zero var) | §5.3 Capability |
| 3 | τ* sweep 99-run | 125M MHA | 3 per config × 9 | ✅ 完成 | τ*=d/√L, worst <1% gap | §Claim 2 |
| 4 | MLA-32 standalone | 432M MLA | 3 (42/43/88) | ✅ 完成 | PPL@16K -31.1%, EVQ > GEO+YaRN(s=4) | §6.6 MLA |
| 5 | 454M Stage 1 (L=512) | 454M MHA | 2 (43/44) | ✅ 完成 | PPL@4K -16.5%, NIAH +26pp | §Claim 4 |

## 强证据 (Strong, single seed but compelling)

| # | 实验 | 模型 | Seeds | 状态 | 核心结果 | 论文角色 |
|---|------|------|-------|------|---------|---------|
| 6 | Progressive 512→1024→2048 | 454M MHA | 1 (42) | ✅ 完成 | superlinear 34.6→52→81.2%, EVQ+YaRN@48K=2.63 | §Claim 3,4 |
| 7 | 750M 2K→4K continue | 750M MHA | 1 (42) | ✅ 完成 | PPL@16K -45.9%, AR exact 0%→77.5% | §6.1 |
| 8 | Phase 21a downstream NLL | 750M MHA | 1 | ✅ 完成 | +4.4%/-4.4% waterbed reversal, QA -16.8% | §Claim 5 |
| 9 | QuALITY full eval n=2086 | 454M MHA | 1 | ✅ 完成 | Gold NLL -30%@8K, acc +2.2pp (p≈0.02) | §Claim 5 |
| 10 | Base generalization (10K/500K) | 454M MHA | 1 | ✅ 完成 | EVQ leads at both, PPL@4K -21.8%/-32.6% | §5.6 |
| 11 | Video DiT head-to-head | 129.6M DiT | 1 | ✅ 完成 | τ=1.5: train -21%, extrap -35% | §6.4 |
| 12 | Video temporal AR | — | 2 (42/137) | ✅ 完成 | EVQ+YaRN@128f -47% | §6.3 |

## Phase 18: MLA YaRN FT 组合性 (核心新发现)

| # | 子实验 | 状态 | 核心结果 |
|---|--------|------|---------|
| 18a | 8K model, 3-seed standalone | ✅ 完成 | EVQ -31.1%@16K (= §6.6 MLA) |
| 18a | 8K model, +YaRN inference | ✅ 完成 | EVQ+YaRN(s=4) -39.7%@16K |
| 18a | 8K model → 16K YaRN FT | ⏳ 待做 | 预期: EVQ+YaRN+FT@16K 大幅领先 |
| 18b | 4K model 1B, baseline | ✅ 完成 | EVQ raw +11.1%@8K (GEO wins!) |
| 18b | 4K model 1B, YaRN+FT s=2 →8K | ✅ 完成 | **EVQ+YaRN+FT -2.5%@8K** (13.6pp swing!) |
| 18b | 4K model 1B, YaRN+FT s=4 →16K | ✅ 完成 | **EVQ+YaRN+FT -1.7%@16K** |
| 18b | 50% ckpt (500M), baseline | ✅ 完成 | EVQ+YaRN inference -3.1%@8K (不需FT就赢) |
| 18b | 50% ckpt, EVQ+YaRN+FT | ⏳ 待做 | — |
| 18b | 75% ckpt, all evals | ⏳ 部分 | GEO baseline done, EVQ pending |

## Phase 19 及其他

| # | 实验 | 状态 | 备注 |
|---|------|------|------|
| 19 | τ=1 vs GEO | ✅ 完成 | 见 PHASE19 report |
| — | 125M MHA @ 256-tok, τ*=4.0 | ⏳ 待做 | 验证 τ* formula at L=256 |
| — | 454M +250M continue | 💡 计划 | 测试更多训练量对 EVQ/GEO 影响 |
| — | 8K model 3-seed YaRN FT → 16K | ⏳ 待做 | 8K baseline EVQ极强(-31.1%), FT后预期碾压 |

---

## 关键叙事线: Superlinear Composability (Phase 18核心发现)

### 数据汇总

| Training Config | EVQ vs GEO raw@2× | EVQ vs GEO +YaRN+FT@target | Swing (pp) |
|-----------------|:------------------:|:---------------------------:|:----------:|
| 8K, 500M (undertrained) | **-31.1%** | **-39.7%** (YaRN s=4) | 8.6 |
| 4K, 1B (fully trained) | +11.1% | **-2.5%** (YaRN s=2) | **13.6** |
| 4K, 50% trained (500M) | +5.7% | pending | — |
| 4K, 75% trained (750M) | pending | pending | — |

### 论文主线

**一句话**: EVQ provides a better frequency starting point for YaRN — even when standalone EVQ loses 11% to GEO, EVQ+YaRN+FT wins 2.5%, a 13.6pp structural reversal that cannot be explained by YaRN alone.

**为什么这很重要**:
1. 回答了 "EVQ在充分训练后还有用吗?" → 有用, 通过组合性
2. 回答了 "这个组合性是 additive 还是 multiplicative?" → superlinear (13.6pp swing > YaRN inference-only 差距)
3. 与 Phase 17 的 -86% composition 效果方向一致, 但现在在充分训练的 regime 里也验证了

### 需要的 figures

1. **Figure: Training Amount vs EVQ Advantage** — 两条线:
   - 蓝线 (raw extrap): 从 -31.1% (undertrained) 到 +11.1% (fully trained) — 下降
   - 红线 (+YaRN+FT): 始终 < 0 — 不管训练量多少, EVQ+YaRN 组合都赢
   - X轴: 训练 tokens/参数 ratio 或训练百分比
   - 这张图是 "killer figure", 一眼说明 composability 的价值

2. **Table: 4K model all checkpoints** (等 50%/75% FT 数据后填入)

---

## 当前阻塞项

1. ⏳ 75% checkpoint full eval (GEO + EVQ baselines + YaRN)
2. ⏳ 50% checkpoint EVQ+YaRN+FT
3. ⏳ 8K model YaRN FT → 16K (最有价值的待做实验: baseline EVQ 极强)
4. ⏳ 125M@256 τ*=4.0 验证

## 可以立即做的 (不依赖新实验)

1. ✅ 把 Phase 18 composition 数据写入 mainstory.md
2. ✅ 准备 Figure 1 的数据脚本
3. ✅ 整理 undertrained hypothesis 的完整论述
4. ✅ 更新 §Claim 3 加入 MLA composition evidence
