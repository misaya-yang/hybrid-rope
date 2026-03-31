# EXP-4: 350M Progressive Chain — Delayed τ Protocol Validation

> 状态: **READY TO RUN**
> 设备: M4 Max 36GB (>30GB 可用), MPS backend
> 脚本: `scripts/m4_max_36gb/exp4_progressive_chain_350m.py`
> 预计时间: 40-50h (需 pilot 校准)
> 创建日期: 2026-03-13

---

## 1. 动机

Phase 17C (454M, seed 42-44) Stage 3 出现异常：EVQ 赢 NIAH@4096（100% vs 72-88%）但输 PPL@8192（45 vs 30）。

**根因诊断**：当前 pipeline 在每个 stage 重新计算 τ*(L_current) 并替换 inv_freq，迫使模型同时适应新频率 + 新长度（"dual adaptation"）。Phase 17B 意外保留了 τ*(512)=2.828（没有 strip inv_freq），反而效果更好。

**核心假设**：τ 应该在整个 progressive chain 中保持不变（delayed），仅在 eval 时通过 YaRN 处理长度外推。

## 2. 为什么是 350M

| 理由 | 说明 |
|------|------|
| **架构等同** | 350M = 454M 的 identical architecture (24L/16H/d=1024/d_head=64)，唯一区别是训练 tokens |
| **直接去风险** | 如果 350M 验证 delayed protocol，454M 的修复方案就有了双重保障 |
| **30GB 可行** | L=2048, batch=1 时约 18-22GB，30GB 充足 |
| **论文价值** | 把 454M 从 single-seed 升级为 350M×2-seed + 454M×1-seed 的多角度证据 |

## 3. 实验设计

### 3.1 三种方法

| 方法 | τ 策略 | 预期 | Seeds |
|------|--------|------|-------|
| **GEO** | τ=0 (geometric) | 控制组，无频率重分配 | 42, 123 |
| **EVQ-D** (delayed) | τ*(512)=2.828 全程不变 | 主假设：应全面赢 GEO | 42, 123 |
| **EVQ-R** (retarget) | τ 每 stage 重算 | 诊断组：应在 Stage 3 输 PPL | 42 only |

### 3.2 三阶段 Progressive Chain

| Stage | L_train | Tokens | Batch | LR | τ(GEO) | τ(EVQ-D) | τ(EVQ-R) |
|-------|---------|--------|-------|-----|--------|----------|----------|
| Stage 1 | 512 | 50M | 4 | 2e-4 | 0 | 2.828 | 2.828 |
| Stage 2 | 1024 | 25M | 2 | 5e-5 | 0 | **2.828** | **2.0** |
| Stage 3 | 2048 | 25M | 1 | 5e-5 | 0 | **2.828** | **1.414** |

> 关键对比：EVQ-D 在 Stage 2/3 保持 τ=2.828 不变，EVQ-R 按 τ*=d/√L 重算。

### 3.3 评估矩阵

| Stage | PPL 评估长度 | Passkey 评估长度 | Passkey depths |
|-------|------------|----------------|---------------|
| Stage 1 | 512, 1K, 2K, 4K | 512, 1K | 0.1, 0.5, 0.9 |
| Stage 2 | 1K, 2K, 4K, 8K | 1K, 2K, 4K | 0.1, 0.5, 0.9 |
| Stage 3 | 2K, 4K, 8K, 16K | 2K, 4K, 8K | 0.1, 0.5, 0.9 |

### 3.4 总运行量

| 类型 | 数量 |
|------|------|
| GEO chains (2 seeds) | 2 × 3 stages = 6 |
| EVQ-D chains (2 seeds) | 2 × 3 stages = 6 |
| EVQ-R chains (1 seed) | 1 × 3 stages = 3 |
| **Total training runs** | **15** |
| **Total tokens** | 15 × (50+25+25)M / 3 stages ≈ 500M |

---

## 4. 执行步骤

### Step 1: Pilot 校准 (~30min)

```bash
conda activate aidemo
cd ~/neurIPS-2026/hybrid-rope
python scripts/m4_max_36gb/exp4_progressive_chain_350m.py --pilot
```

Pilot 会：
- 跑 350M@L=512 的 5M tokens
- 测量 MPS 实际吞吐量 (tokens/sec)
- 投射完整实验时间
- 如果超过 50h，自动建议调整 token 数量

### Step 2: 根据 Pilot 调整 (如需)

```bash
# 如果 pilot 建议减少 tokens:
python scripts/m4_max_36gb/exp4_progressive_chain_350m.py \
    --stage1_tokens 30000000 --stage2_tokens 20000000 --stage3_tokens 20000000
```

### Step 3: 正式运行

```bash
# 全量运行 (支持断点续跑——每个 stage 完成后保存 result.json)
nohup python scripts/m4_max_36gb/exp4_progressive_chain_350m.py \
    > exp4_350m.log 2>&1 &

# 监控
tail -f exp4_350m.log
```

### Step 4: 生成汇总

```bash
python scripts/m4_max_36gb/exp4_progressive_chain_350m.py --summary
```

---

## 5. 关键验证点

| # | 验证 | 预期 | 论文影响 |
|---|------|------|---------|
| V1 | Stage 3: EVQ-D PPL@4K < GEO PPL@4K | EVQ 外推一定赢 | 支撑 C3 |
| V2 | Stage 3: EVQ-R PPL@4K > EVQ-D PPL@4K | retarget 的代价量化 | 新发现：protocol matters |
| V3 | Stage 3: EVQ-D NIAH@4K ≥ GEO NIAH@4K | 频率结构保持 | 支撑 C3 |
| V4 | EVQ-D advantage 随 stage 单调递增 | progressive 放大效应 | Fig 4 补充 |
| V5 | 2-seed mean ± std 方向一致 | 不是 single-seed fluke | 统计可信度 |

**最高价值验证**: V2 — 如果 EVQ-R 确实在 Stage 3 输 PPL 而 EVQ-D 赢，我们就有了 "delayed τ protocol" 的直接因果证据，可以写成论文的 key finding。

---

## 6. 预期结果 & 论文贡献

### Best case (全部验证通过)

```
新 Figure/Table:
  "Progressive training with delayed vs retarget τ protocol (350M, 2-seed)"
  (a) PPL@4K across stages: EVQ-D < GEO << EVQ-R (Stage 3)
  (b) NIAH@4K: EVQ-D > GEO > EVQ-R
```

论文叙事：
- "We identify that τ retargeting during progressive training incurs a dual-adaptation penalty"
- "The delayed protocol (keeping τ*(L₀) throughout) avoids this by treating length extension as a YaRN concern, not a frequency redistribution concern"
- 这是一个 **new contribution**，不只是验证已有 claim

### Worst case (EVQ-D 在 350M 不显著赢 GEO)

- 350M@50M tokens 可能训练不够充分（454M 用了 1B+ tokens）
- 叙事调整为 "需要足够训练量才能内化 EVQ 优势"
- EVQ-R vs EVQ-D 的对比仍然有信息量

---

## 7. 技术细节

### 内存估算 (fp32, MPS)

| Stage | Model | AdamW | Activations | Total Est. |
|-------|-------|-------|-------------|-----------|
| L=512, B=4 | 1.4GB | 2.8GB | ~4GB | ~8-10GB |
| L=1024, B=2 | 1.4GB | 2.8GB | ~8GB | ~12-14GB |
| L=2048, B=1 | 1.4GB | 2.8GB | ~16GB | ~20-22GB |

> 30GB 可用 → 所有 stage 都有余量。

### 断点续跑

脚本检查每个 `{method}_seed{seed}/{stage}_L{seq_len}/result.json` 是否存在。已完成的 stage 自动跳过。

### MPS 注意事项

- 不支持 flash-attention，使用 SDPA
- 每 100 steps 调用 `torch.mps.empty_cache()`
- 必须用 float32（MPS 不支持 bfloat16）
- Stage 3 (L=2048, B=1) 是最紧张的，如果 OOM 需要启用 gradient checkpointing

---

## 8. 与现有实验的关系

| 现有实验 | 本实验补充 |
|---------|----------|
| Phase 17B/17C (454M, 1-seed) | 同架构 2-seed 验证 + protocol A/B test |
| Phase 19 (125M, 3-seed) | 从 125M 扩展到 350M = 454M-equivalent |
| Phase 16 (99-run τ* validation) | 验证 τ* 在 progressive training 中的正确使用方式 |
