# M4 Max 36GB 本地实验中心

> 设备：Apple M4 Max, 36GB 统一内存 (**实际可用 ~25GB**), MPS backend
> 环境：`conda activate aidemo`
> 定位：**125M 专精**——系统性验证与论文 gap 填充，单 run ~10-15min
> 创建日期：2026-03-10

---

## 0. 硬件约束（实测）

| 模型规模 | 参数 (fp16) | AdamW 状态 | seq=512 总估算 | seq=2048 总估算 | 可行？ |
|---------|------------|-----------|---------------|----------------|-------|
| **125M** | ~250MB | ~1GB | **~4-6GB** | **~10-12GB** | **主力** |
| 350M | ~700MB | ~2.8GB | ~8-10GB | ~18-22GB | batch 极小，不实用 |
| 454M+ | ~900MB+ | ~3.6GB+ | — | — | 不可行 |

**实际限制**：
- 统一内存 36GB，系统占 ~10GB → ML 可用 **~25GB**
- 125M 是唯一能以合理 batch 跑完全部 seq_len 的规模
- 长序列 eval (≥16K) 需要逐 chunk，可能 OOM 需要 fallback

---

## 1. 实验总览

### 论文核心 Gap 分析（2026-03-10）

| Gap | 当前状态 | 影响 | 本机能填？ |
|-----|---------|------|-----------|
| Phase 18: Base 泛化 sweep | **正在跑** | P0: 5个 base 的 collision-block 定量验证 | **是** |
| Phase 19: Progressive chain 125M × 3-seed | 脚本已写, 未跑 | P0: 验证 YaRN 相变 + progressive 放大 | **是** |
| τ landscape 平坦性系统验证 | Phase 16 已做 9 configs | P1: 补充 d_head=64 重点验证 | **是** |
| Training-inference 等价性 125M 验证 | 无 | P1: evq_512+yarn ≈ evq_1024_cont raw | **是 (EXP-1 副产品)** |
| Waterbed 精确量化 | 散落各 phase | P2: 系统性 short/long tradeoff 表格 | **是 (各实验汇总)** |

### 执行计划

| 顺序 | 实验 | Phase | 脚本 | 预计时间 | 状态 |
|------|------|-------|------|---------|------|
| 0 | Base Generalization Sweep | Phase 18 | `core_text_phases/phase18_*` | ~8-11h | **正在跑** |
| 1 | Progressive Chain 125M × 3-seed | Phase 19 | `exp1_progressive_chain_125m.py` | ~3-4.5h | ⏳ 等 Phase 18 |
| 2 | τ Landscape Flatness (d_head=64) | Phase 21 | `exp2_tau_landscape.py` | ~3h | ⏳ |
| 3 | Training-Inference Equivalence | Phase 22 | `exp3_train_infer_equiv.py` | ~15min | ⏳ 依赖 EXP-1 |

---

## 2. EXP-0: Base Generalization Sweep (Phase 18) — 正在执行

> **优先级**: P0 — 封堵 "只在 base=500K 测了" 的 reviewer 攻击
> **脚本**: `core_text_phases/phase18_base_generalization_sweep.py`（已有）

### 2.1 目标

验证 EVQ gain 跨 base 稳定性 + collision-block 理论定量预测：
- gain 随 base 单调递增
- base=10K 进入 dead zone

### 2.2 设计

| Base | lnb | 碰撞块占比 (L=512) | 理论预期 EVQ gain |
|------|-----|-------------------|------------------|
| 10K | 9.21 | 32.3% | ~0% (dead zone) |
| 50K | 10.82 | 27.5% | 低 |
| 100K | 11.51 | 26.0% | 中 |
| 500K | 13.12 | 22.8% | 高 |
| 10M | 16.12 | 18.5% | 最高 |

- 模型: 125M, d_head=64, L_train=512
- τ*=64/√512=2.83
- Seeds: 42, 137, 256

### 2.3 关键验证点

- [ ] base=10K EVQ ≤ Geo? (dead zone 复现)
- [ ] gain vs lnb 单调递增?
- [ ] collision-block 理论 (1-c)/lnb 与实际 gain 的拟合 R²

---

## 3. EXP-1: Progressive Chain 125M × 3-seed (Phase 19)

> **优先级**: P0 — 论文最缺的多 seed progressive 证据
> **来源**: 从 `mac_train/exp1_progressive_chain.py` 迁移
> **前提**: Phase 18 完成（避免同时占用 MPS）

### 3.1 目标

验证 Phase 17b (454M 单 seed) 的三大发现在 125M × 3-seed 上复现：
1. **YaRN 相变**: progressive training 后 EVQ raw > EVQ+YaRN
2. **Progressive 放大**: EVQ 优势随 stage 单调递增
3. **延伸到 2048**: 第三个数据点确认趋势

### 3.2 设计

| Stage | L_train | Tokens | τ* (EVQ) |
|-------|---------|--------|----------|
| Stage 0 | 512 | 50M | 2.83 |
| Stage 1 | 1024 | 25M | 2.0 |
| Stage 2 | 2048 | 25M | 1.41 |

- 模型: 125M (hidden=768, 12L, 12H, d_head=64)
- Base: 500K
- 方法: Geo (τ=0) vs EVQ (τ*=d/√L_current)
- Seeds: 42, 137, 256
- 评估: raw + YaRN @ {512, 1K, 2K, 4K, 8K, 16K, 32K}

### 3.3 超参（适配 25GB）

- `micro_batch_size`: 8（保守，Stage 2 seq=2048 时内存紧张）
- `grad_accum`: 2
- 有效 batch: 8 × 2 × seq_len tokens/step
- Stage 2 (seq=2048) 预计占 ~10-12GB，25GB 足够

### 3.4 关键验证点

- [ ] Stage 1 后: EVQ raw PPL@16K < EVQ+YaRN PPL@16K? (3/3 seeds?)
- [ ] Stage 0→1→2: EVQ vs Geo advantage 单调递增?
- [ ] `evq_stage0+yarn ≈ evq_stage1 raw`? (training-inference 等价)

### 3.5 命令

```bash
cd ~/neurIPS-2026/hybrid-rope
conda activate aidemo
python scripts/m4_max_36gb/exp1_progressive_chain_125m.py --pilot   # seed=42 验证
python scripts/m4_max_36gb/exp1_progressive_chain_125m.py           # 全量 3-seed
python scripts/m4_max_36gb/exp1_progressive_chain_125m.py --summary # 汇总
```

### 3.6 预期结果 & 论文贡献

**Best case（全部复现）**：

```
Figure X: Progressive training amplifies EVQ advantage (125M, 3-seed mean ± std)
(a) EVQ vs Geo raw PPL@16K across stages — advantage monotonically increases
(b) EVQ raw vs EVQ+YaRN — phase transition at Stage 1
```

论文可以写：
- "The YaRN phase transition is reproducible across scales (125M × 3-seed, 454M × 1-seed)"
- "Progressive training monotonically amplifies EVQ's raw advantage"

**Worst case（125M 不复现 YaRN 相变）**：
- 可能 125M 容量不够内化频率分配
- 叙事调整为 "需要足够模型容量才能 internalize"
- 不影响 454M 结论

---

## 4. EXP-2: τ Landscape Flatness (Phase 21)

> **优先级**: P1 — 加强 "practitioners can use τ*=d/√L without grid search" 的可视化
> **来源**: Phase 16 的延伸，专注 d_head=64 (MLA-realistic)

### 4.1 目标

在 d_head=64 的 3 个 L 值上做细粒度 τ sweep，画出 loss landscape curve：
- 证明 τ* 附近是浅盆地（shallow basin）
- 量化 PPL gap：τ=0 (Geo) → τ* (EVQ) → 2τ* (过度)

### 4.2 设计

| L_train | τ* (理论) | τ sweep 值 |
|---------|----------|-----------|
| 256 | 4.0 | {0, 1, 2, 3, **4**, 5, 6, 8} |
| 512 | 2.83 | {0, 1, 1.5, 2, **2.83**, 3.5, 4, 5, 6} |
| 1024 | 2.0 | {0, 0.5, 1, 1.5, **2**, 2.5, 3, 4} |

- 模型: 125M, d_head=64, base=500K
- Seed: 42 (pilot), +137 (confirm)
- 每 run: 25M tokens, ~5-8min
- 评估: raw PPL @ {L_train, 2×, 4×, 8×}
- 总 runs: 25 (pilot) / 50 (full)

### 4.3 预期输出

```
Figure: Loss landscape around τ*=d/√L (125M, d_head=64, base=500K)
  (a) L=256: PPL vs τ — broad minimum around τ=4
  (b) L=512: PPL vs τ — broad minimum around τ=2.83
  (c) L=1024: PPL vs τ — broad minimum around τ=2
  → Worst-case PPL gap between τ* and empirical optimum < 1%
```

### 4.4 命令

```bash
python scripts/m4_max_36gb/exp2_tau_landscape.py --pilot   # seed=42, L=512 only
python scripts/m4_max_36gb/exp2_tau_landscape.py           # 全量
python scripts/m4_max_36gb/exp2_tau_landscape.py --plot     # 画图
```

---

## 5. EXP-3: Training-Inference Equivalence Table (Phase 22)

> **优先级**: P1 — 验证 "progressive 一步替代 YaRN" 在 125M 上成立
> **前提**: EXP-1 完成（直接复用其结果，不需要额外训练）

### 5.1 目标

从 EXP-1 结果中提取等价性表格：

```
evq_stage0 + YaRN(factor=2) ≈? evq_stage1_raw     (512→1024)
evq_stage1 + YaRN(factor=2) ≈? evq_stage2_raw     (1024→2048)
```

454M 参考值：`evq_512+yarn PPL@16K=11.6 ≈ evq_1024_cont raw PPL@16K=11.2`（偏差 <4%）

### 5.2 输出

| 对照组 | PPL@4K | PPL@8K | PPL@16K | 等价偏差 |
|--------|--------|--------|---------|---------|
| evq_stage0 + YaRN | ? | ? | ? | baseline |
| evq_stage1 raw | ? | ? | ? | δ₁ |
| evq_stage1 + YaRN | ? | ? | ? | baseline |
| evq_stage2 raw | ? | ? | ? | δ₂ |

脚本只做汇总分析，**不需要额外训练**。

### 5.3 命令

```bash
python scripts/m4_max_36gb/exp3_train_infer_equiv.py  # 从 EXP-1 results 提取
```

---

## 6. 设计原则

### 6.1 通用原则

- 所有脚本支持 `--pilot`（单 seed）和 `--dry-run`（打印计划）
- 断点续跑：检查 `result.json` 是否存在
- 复用 `core_text_phases/run_evq_sweep.py` 的模型和数据加载
- 结果存放在 `results/m4_max_36gb/` 下

### 6.2 MPS / 内存注意事项

- MPS 不支持 flash-attention，使用 SDPA
- 定期 `torch.mps.empty_cache()` 防止内存泄漏
- 长序列 eval (≥16K) 逐 chunk 评估，OOM 时 skip
- fp16 + autocast 优先；125M 在 25GB 内应该不紧张
- **同一时间只跑一个实验**（避免 MPS 竞争）

### 6.3 与 mac_train 的关系

`mac_train/` 是早期基于 ~20GB 假设的实验目录，只含 Phase 19 脚本。
本目录 (`m4_max_36gb/`) 基于实际 ~25GB 可用内存重新规划：
- 明确 **125M 专精** 定位，不幻想跑 350M+
- 新增 τ landscape 和 training-inference 等价性验证
- Phase 18 继续使用 `core_text_phases/` 现有脚本
- EXP-1 从 mac_train 迁移

### 6.4 结果目录结构

```
results/m4_max_36gb/
├── exp1_progressive_125m/
│   ├── geo_seed42/stage0_L512/{result.json, model.pt}
│   ├── evq_seed42/stage0_L512/{result.json, model.pt}
│   └── ...
├── exp2_tau_landscape/
│   ├── L256_tau0.0_seed42/{result.json}
│   ├── L512_tau2.83_seed42/{result.json}
│   └── ...
└── exp3_train_infer_equiv/
    └── equivalence_table.json
```

---

## 7. 时间预算

| 实验 | 单 pilot | 全量 | 累计 |
|------|---------|------|------|
| EXP-0: Phase 18 Base Sweep | — | ~8-11h | **正在跑** |
| EXP-1: Progressive 125M | ~45min | ~3.5h | 3.5h |
| EXP-2: τ Landscape | ~40min | ~3h | 6.5h |
| EXP-3: Equiv Table | — | ~15min | 6.75h |

**Phase 18 之后的新实验总计约 7 小时**。
- pilot 约 1.5h，可快速验证所有设计
- 全量可在一个 overnight session 完成

---

## 8. 论文贡献映射

| 实验 | 论文位置 | 填补的 claim |
|------|---------|-------------|
| EXP-0 (Phase 18) | §4 / Appendix (Base Ablation) | collision-block 定量验证，封堵 "只测了 500K" |
| EXP-1 (Phase 19) | §5 Progressive Training (main text) | YaRN 相变跨规模复现 + 3-seed |
| EXP-2 (Phase 21) | §4 Figure (τ* landscape) | "flat basin, no grid search needed" 可视化 |
| EXP-3 (Phase 22) | §5 Table (equivalence) | training-inference 等价性跨规模证据 |
