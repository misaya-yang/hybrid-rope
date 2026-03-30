# P0 Runbook: Progressive 3-seed 复现

> 这是论文中稿概率提升最大的单项实验 (+10-15%)

---

## 前置条件

- [ ] Stage 1 seed=43 checkpoint 存在 (geo + evq 各一个)
- [ ] Stage 1 seed=44 checkpoint 存在 (geo + evq 各一个)
- [ ] GPU 显存 ≥ 24GB (454M fp16/bf16)
- [ ] 数据: FineWeb-Edu 准备好

**检查 Stage 1 ckpt 位置**:
```bash
# 根据你的 ckpt 存储习惯，可能在:
ls results/evq_phase17*/  # 或
ls results/454m_staged_*/
# 找到 seed43/seed44 的 Stage 1 最终 checkpoint
```

---

## Step 1: Stage 2 训练 (L=1024)

### Seed 43
```bash
PHASE17C_SEQ_LEN=1024 \
PHASE17C_SEED=43 \
PHASE17C_TOKENS=500000000 \
PHASE17C_GEO_INIT_CKPT=<path_to_stage1_geo_seed43> \
PHASE17C_EVQ_INIT_CKPT=<path_to_stage1_evq_seed43> \
python scripts/core_text_phases/phase17b_454m_512_to_1024_continue_ckpt_eval.py
```

### Seed 44
```bash
PHASE17C_SEQ_LEN=1024 \
PHASE17C_SEED=44 \
PHASE17C_TOKENS=500000000 \
PHASE17C_GEO_INIT_CKPT=<path_to_stage1_geo_seed44> \
PHASE17C_EVQ_INIT_CKPT=<path_to_stage1_evq_seed44> \
python scripts/core_text_phases/phase17b_454m_512_to_1024_continue_ckpt_eval.py
```

**可并行**: 两个 seed 互不依赖，有 2 卡就同时跑。

### Stage 2 验收
- [ ] PPL@8K: EVQ < Geo (方向与 seed42 一致)
- [ ] 保存 ckpt 路径用于 Stage 3

---

## Step 2: Stage 3 训练 (L=2048)

### Seed 43
```bash
PHASE17C_SEQ_LEN=2048 \
PHASE17C_SEED=43 \
PHASE17C_TOKENS=500000000 \
PHASE17C_GEO_INIT_CKPT=<path_to_stage2_geo_seed43> \
PHASE17C_EVQ_INIT_CKPT=<path_to_stage2_evq_seed43> \
python scripts/core_text_phases/phase17c_454m_1024_to_2048_continue.py
```

### Seed 44
```bash
PHASE17C_SEQ_LEN=2048 \
PHASE17C_SEED=44 \
PHASE17C_TOKENS=500000000 \
PHASE17C_GEO_INIT_CKPT=<path_to_stage2_geo_seed44> \
PHASE17C_EVQ_INIT_CKPT=<path_to_stage2_evq_seed44> \
python scripts/core_text_phases/phase17c_454m_1024_to_2048_continue.py
```

### Stage 3 验收
- [ ] PPL@16K: EVQ+YaRN < Geo+YaRN (seed42: 2.48 vs 13.17)
- [ ] PPL@48K: EVQ+YaRN < 4.0 (seed42: 2.63)
- [ ] Progressive monotonicity: Stage 3 advantage > Stage 2 advantage

---

## Step 3: Extended Eval

如果 phase17c 脚本没有自动跑 YaRN eval 和 48K eval，单独跑:

```bash
# 用 phase17c_extended_eval.py
PHASE17C_SEED=43 \
PHASE17C_EVAL_CKPT=<path_to_stage3_ckpt_seed43> \
python scripts/core_text_phases/phase17c_extended_eval.py

PHASE17C_SEED=44 \
PHASE17C_EVAL_CKPT=<path_to_stage3_ckpt_seed44> \
python scripts/core_text_phases/phase17c_extended_eval.py
```

---

## Step 4: 汇总结果

填入以下表格:

### EVQ+YaRN PPL@16K (3-seed)

| Seed | Stage 1 (L=512) | Stage 2 (L=1024) | Stage 3 (L=2048) |
|------|-----------------|------------------|------------------|
| 42 | 3.80→2.48 (-34.6%) | 4.60→2.21 (-52.0%) | 13.17→2.48 (-81.2%) |
| 43 | ___ → ___ (___%) | ___ → ___ (___%) | ___ → ___ (___%) |
| 44 | ___ → ___ (___%) | ___ → ___ (___%) | ___ → ___ (___%) |
| **Mean±Std** | | | |

### EVQ+YaRN PPL@48K (Stage 3 only)

| Seed | Geo+YaRN | EVQ+YaRN | Δ |
|------|----------|----------|---|
| 42 | 14.22 | 2.63 | -82% |
| 43 | | | |
| 44 | | | |
| **Mean±Std** | | | |

### Passkey@8K+YaRN (Stage 3)

| Seed | Geo+YaRN | EVQ+YaRN |
|------|----------|----------|
| 42 | 61% | 100% |
| 43 | | |
| 44 | | |

---

## 失败处理

**如果某个 seed 方向不一致** (e.g., Geo+YaRN 没有 collapse):
- 首先检查 ckpt 加载是否正确 (inv_freq hash)
- 检查 tau 是否正确 (Stage 3 应该是 1.414)
- 如果确实不一致，报告 mean±std 并在论文中诚实标注

**如果 48K PPL > 4.0**:
- 只要方向一致 (EVQ < Geo)，仍然是正面结果
- 调整论文表述从 "PPL~2.63" 到 "PPL mean±std"
