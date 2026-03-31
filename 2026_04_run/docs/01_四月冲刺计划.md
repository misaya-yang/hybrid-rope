# April 2026 实验冲刺计划

> 目标：将论文从 Borderline Accept (~45%) 推到 Weak Accept (~65-75%)
> 截止日期：NeurIPS 2026 deadline 前
> 设备：AutoDL GPU (RTX 5090 / A100 等)

---

## 关键问题：Progressive Training 的 tau 策略

**论文 Phase 17C 使用的是 Retarget 协议（tau 随 stage 变化）：**

| Stage | L_train | tau = d_head/√L |
|-------|---------|-----------------|
| Stage 1 | 512 | 2.828 |
| Stage 2 | 1024 | 2.000 |
| Stage 3 | 2048 | 1.414 |

**EXP-4（M4 低 token 实验）结论：**
- 低 token 预算 (36M) 下 Delayed (tau 不变) 在 Stage 1-2 赢 3.5%，Stage 3 两者接近
- 高 token 预算 (500M-1B/stage, Phase 17C) 下 Retarget 协议是论文的实际配置
- **结论：3-seed 复现必须用 Retarget 协议，与 seed=42 保持一致**

---

## 实验优先级

### P0: Progressive Training 3-seed [最高优先级] ⭐⭐⭐

**审稿人原话**: "Can you provide a 3-seed replication of the progressive-training result, especially the Stage-3 48K claim?"

**预估中稿概率提升**: +10-15%

**当前状态**:
- seed=42: ✅ 完整三阶段 + eval，论文主结果
- seed=43: Stage 1 ✅ (PPL@4K -16.5%), Stage 2-3 ❌
- seed=44: Stage 1 ✅ (同上), Stage 2-3 ❌

**需要做的 (6 次训练)**:

| Run | Seed | Stage | L_train | tau | 从哪个 ckpt 继续 | Tokens | 预估时间 |
|-----|------|-------|---------|-----|------------------|--------|---------|
| 1 | 43 | 2 | 1024 | 2.000 | Stage 1 seed43 ckpt | 500M | ~6h A100 |
| 2 | 44 | 2 | 1024 | 2.000 | Stage 1 seed44 ckpt | 500M | ~6h |
| 3 | 43 | 3 | 2048 | 1.414 | Run 1 的 ckpt | 500M | ~8h |
| 4 | 44 | 3 | 2048 | 1.414 | Run 2 的 ckpt | 500M | ~8h |
| 5 | 43 | eval | — | — | Run 3 的 ckpt | — | ~1h |
| 6 | 44 | eval | — | — | Run 4 的 ckpt | — | ~1h |

**总计**: ~30h GPU 时间（可 2 卡并行 → ~15h）

**Eval 必须包含**:
- PPL @ 2K/4K/8K/16K/32K/48K (raw + YaRN)
- Passkey retrieval @ 2K/4K/8K/16K (raw + YaRN)
- 关键指标：EVQ+YaRN@48K PPL, EVQ+YaRN@16K PPL, Passkey@8K+YaRN

**验收标准**:
- 3 seed 的 EVQ+YaRN@48K PPL 方向一致（都 <4.0, seed42=2.63）
- 3 seed 的 Geo+YaRN@16K 都 collapse（>10, seed42=13.17）
- Passkey@8K+YaRN: 3 seed 都 ≥90%

**成功后论文改动**:
- Abstract: "(single seed)" → "(3 seeds)"
- Intro contribution #4: 同上
- Table 5: 扩展为 3-seed mean±std
- §4 progressive paragraph: 去掉 "single seed" 标注

**脚本**: 复用 `scripts/core_text_phases/phase17c_454m_1024_to_2048_continue.py`
环境变量：
```bash
# Stage 2 seed=43
PHASE17C_SEQ_LEN=1024 PHASE17C_SEED=43 PHASE17C_TOKENS=500000000 \
  PHASE17C_GEO_INIT_CKPT=<stage1_geo_seed43_ckpt> \
  PHASE17C_EVQ_INIT_CKPT=<stage1_evq_seed43_ckpt> \
  python scripts/core_text_phases/phase17b_454m_512_to_1024_continue_ckpt_eval.py

# Stage 3 seed=43
PHASE17C_SEQ_LEN=2048 PHASE17C_SEED=43 PHASE17C_TOKENS=500000000 \
  PHASE17C_GEO_INIT_CKPT=<stage2_geo_seed43_ckpt> \
  PHASE17C_EVQ_INIT_CKPT=<stage2_evq_seed43_ckpt> \
  python scripts/core_text_phases/phase17c_454m_1024_to_2048_continue.py
```

---

### P1: QuALITY Downstream 多种子 [高优先级] ⭐⭐

**审稿人 Weakness 3**: "downstream signal is still single-seed"

**预估中稿概率提升**: +5-8%

**当前状态**:
- seed=42: Gold NLL -30%@8K, accuracy +2.2pp (p≈0.02), n=2086

**需要做的**:
1. 454M continue@4K 训练, seed=43, seed=44 (如果 ckpt 不存在)
2. QuALITY full eval (n=2086) on 3 seeds

**预估成本**: 2 次 454M continue 训练 (~4h each) + 3 次 eval (~2h each) = ~14h

**验收标准**:
- 3 seed Gold NLL@8K 方向一致（都是负值）
- Mean Gold NLL@8K 至少 -20%

**脚本**: 复用现有 QuALITY eval 脚本（检查 `scripts/` 下的 eval 相关文件）

---

### P2: τ* 细粒度 Sweep [中优先级] ⭐

**审稿人 Question 2**: "How sensitive is the deployed τ* recipe to the calibration of λ?"

**预估中稿概率提升**: +3-5%

**当前状态**: 99-run sweep, 27 configurations, 但步长可能较粗

**需要做的**:
- 在 τ*±50% 范围内用 0.2 步长做 fine sweep
- 配置: 125M, L=512, base=500K (快速)
- τ range: [1.0, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.4, 3.8, 4.2]
  (τ* ≈ 2.83 for this config)
- Eval: PPL@4K/8K/16K raw + YaRN

**预估成本**: ~13 runs × ~1h = ~13h (125M 很快)

**产出**: PPL vs τ 碗形曲线图，量化 <1% PPL 变化的 τ 宽度

**验收标准**: 碗底 ±20% τ 范围内 PPL 变化 <1%

---

### P3: LongBench 覆盖 [低优先级，可选] ⭐

**审稿人 Question 4**: "what do the authors expect on fuller LongBench/RULER coverage?"

**预估中稿概率提升**: +2-3%（锦上添花）

**当前状态**: 750M 有 13-task NLL (Phase 21a), 但 accuracy-based eval 不完整

**需要做的**:
- 750M continue@4K ckpt 上跑 LongBench 全集 (17 subtask)
- 需要较大显存 (≥40GB for 750M + long context)

**成本**: ~8-12h (取决于 GPU)

---

## 优先级排序 & 时间估算

| 实验 | 优先级 | GPU 时间 | 概率提升 | ROI |
|------|--------|---------|---------|-----|
| P0: Progressive 3-seed | ⭐⭐⭐ | ~30h | +10-15% | **最高** |
| P1: QuALITY 3-seed | ⭐⭐ | ~14h | +5-8% | 高 |
| P2: τ sweep 细化 | ⭐ | ~13h | +3-5% | 中 |
| P3: LongBench | ⭐ | ~12h | +2-3% | 低 |

**最优顺序**: P0 → P1 → P2 → P3（如果时间有限，只做 P0+P1 就够了）

**如果只有 48h GPU**: 做 P0 + P1 = ~44h, 概率从 ~45% → ~60-68%
**如果有 72h GPU**: 做 P0 + P1 + P2 = ~57h, 概率从 ~45% → ~65-75%

---

## Checkpoint 依赖图

```
Stage 1 ckpt (seed 43/44) ─── 已有 ✅
        │
        ▼
Stage 2 训练 (P0 Run 1/2) ─── 需要跑
        │
        ├──► Stage 2 eval (验证 Stage 2 结果)
        │
        ▼
Stage 3 训练 (P0 Run 3/4) ─── 需要跑
        │
        ├──► Stage 3 eval + YaRN eval (P0 Run 5/6)
        │
        ▼
        Done → 更新论文

454M continue@4K ckpt (seed 43/44) ─── 可能需要跑
        │
        ▼
QuALITY eval (P1) ─── 需要跑
```

---

## 论文更新 Checklist（实验完成后）

- [ ] Abstract: "(single seed)" → "(3 seeds)" (P0 完成后)
- [ ] Intro contribution #4: 同上
- [ ] Table 5 (progressive): 扩展为 3-seed mean±std
- [ ] §4 progressive paragraph: 去掉 "single seed", 加 "3-seed mean ± std"
- [ ] §4.4 downstream: 加 QuALITY 3-seed 数据 (P1 完成后)
- [ ] §3.7: 加 τ sweep 碗形曲线引用 (P2 完成后)
- [ ] Limitations: 更新 single-seed 列表
- [ ] 重新编译验证 9 页
