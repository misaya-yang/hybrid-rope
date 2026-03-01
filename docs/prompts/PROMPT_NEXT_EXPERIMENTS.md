# 下一步实验执行指南 (Claude Code 专用)

> **用途**: 在 Claude Code 中说"请读 docs/paperdraft/PROMPT_NEXT_EXPERIMENTS.md，执行下一步实验"
> **前置条件**: Phase 0-3 已完成（128-tok PE Quality Test）
> **最后更新**: 2026-03-01

---

## 实验优先级排序

| 优先级 | 实验 | GPU 成本 | 价值 |
|:---:|------|---------|------|
| **P0** | Phase 4: Context Extension | ~4h | 堵住"不实用"攻击面 |
| **P1** | τ=1.5 跨数据集验证 | ~30min | 强化"默认值"claim |
| **P2** | 500M from-scratch | ~8h | 堵住"规模太小"攻击面 |

---

## P0: Phase 4 — Context Extension Experiment

### 目的
模拟真实 LLM "预训练→上下文扩展"工作流。证明 EVQ 在 context extension 场景中优于 PI/YaRN。

### 实验设计

**Step 1: 预训练基础模型 (1 run)**

```bash
python scripts/m4_evq_sweep/run_evq_sweep.py \
    --tier 125m \
    --method geometric \
    --tau 0.0 \
    --dataset fineweb-edu \
    --train_tokens 100000000 \
    --seq_len 2048 \
    --base 500000.0 \
    --seeds 42 \
    --work_dir /root/autodl-tmp/context_extension \
    --save_checkpoint
```

产出: `checkpoint_base.pt`（2K 训练的 geometric 模型）

**Step 2: 上下文扩展 (5 runs)**

从 `checkpoint_base.pt` 继续训练，序列长度扩展到 8192，训练 20M tokens。

| Run | Method | 关键修改 | 说明 |
|-----|--------|---------|------|
| D1 | Geometric (直接扩展) | 不改 freq，直接用 8K seq | baseline |
| D2 | PI (Position Interpolation) | freq /= 4 (8192/2048) | 经典方法 |
| D3 | YaRN | PI + 高频保护（参考 YaRN 论文公式） | 经典方法 |
| D4 | EVQ fixed τ=1.5 | 用 EVQ-Cosh 频率替换 | 理论方法 |
| D5 | EVQ learnable (init=1.0) | learnable τ + lr_mult=100 | 核心实验 |

**每个 run 的命令模板**（需要 Claude Code 实现 `--resume_from` 和 `--method` 的 context extension 逻辑）:

```bash
python scripts/m4_evq_sweep/run_evq_sweep.py \
    --tier 125m \
    --method [geometric|pi|yarn|evq_cosh|learnable] \
    --tau [0.0|1.5|learnable] \
    --dataset fineweb-edu \
    --train_tokens 20000000 \
    --seq_len 8192 \
    --base 500000.0 \
    --seeds 42 \
    --work_dir /root/autodl-tmp/context_extension \
    --resume_from /root/autodl-tmp/context_extension/checkpoint_base.pt \
    --eval_lengths 2048,4096,8192,16384,32768
```

### 需要实现的代码修改

1. **`--resume_from` flag**: 从 checkpoint 继续训练，加载模型权重但用新的 freq schedule
2. **PI 方法实现**: `inv_freq = standard_inv_freq / scale_factor` where `scale_factor = target_len / train_len`
3. **YaRN 方法实现**: 参考 YaRN 论文的 NTK-by-parts 公式
   ```python
   # YaRN: NTK-by-parts interpolation
   # 高频 (short wavelength): 不缩放
   # 低频 (long wavelength): 按 PI 缩放
   # 中频: 线性插值
   beta_fast = 32  # YaRN 论文默认值
   beta_slow = 1
   # ... (参考 transformers/models/llama/modeling_llama.py 中的 YaRN 实现)
   ```
4. **`--eval_lengths` 扩展**: 支持到 32768

### 评估指标

```
PPL@2048  — 训练内 (基础模型的训练长度)
PPL@4096  — 训练内 (context extension 的中间)
PPL@8192  — 训练内 (context extension 的目标长度)
PPL@16384 — 2× 外推
PPL@32768 — 4× 外推
```

### 预期结果

| Method | PPL@8K (训练内) | PPL@16K (外推) | PPL@32K (外推) |
|--------|----------------|----------------|----------------|
| Geometric | 高 (没有 adaptation) | 很高 | 爆炸 |
| PI | 中 (牺牲分辨率) | 中-高 | 高 |
| YaRN | 中-低 (保护高频) | 中 | 中-高 |
| **EVQ fixed** | **低** | **中-低** | **中** |
| **EVQ learnable** | **低** | **中-低** | **中** |

### 成功标准

1. EVQ (any τ) PPL@8K ≤ YaRN PPL@8K
2. EVQ PPL@16K < PI PPL@16K
3. Learnable τ 在 context extension 中 τ > 1.14（因为更长上下文需要更多高频）

---

## P1: τ=1.5 跨数据集验证

### 目的
验证 τ=1.5 作为"默认值"的普适性。在更多数据集上确认。

### 实验设计

用 128-tok 协议，只跑 Geometric + EVQ τ=1.5 两个 run，换数据集：

```bash
# OpenWebText (如果可用)
python scripts/m4_evq_sweep/run_evq_sweep.py \
    --tier 125m --method evq_cosh --tau 1.5 \
    --dataset openwebtext --train_tokens 15000000 \
    --seq_len 128 --base 500000.0 --seeds 42 \
    --work_dir /root/autodl-tmp/tau_universal

# TinyStories (128-tok regime, 对照已有 2K-tok 结果)
python scripts/m4_evq_sweep/run_evq_sweep.py \
    --tier 125m --method evq_cosh --tau 1.5 \
    --dataset tinystories --train_tokens 15000000 \
    --seq_len 128 --base 500000.0 --seeds 42 \
    --work_dir /root/autodl-tmp/tau_universal
```

### 成功标准

τ=1.5 在所有数据集上的 Δ@8K vs Geometric 都 > 10%。

---

## P2: 500M From-Scratch

### 目的
堵住"规模太小"攻击面 + 完善 scaling law 曲线。

### 实验设计

```bash
python scripts/m4_evq_sweep/run_evq_sweep.py \
    --tier 500m \
    --method evq_cosh --tau 1.5 \
    --dataset fineweb-edu \
    --train_tokens 200000000 \
    --seq_len 2048 \
    --base 500000.0 \
    --seeds 42 \
    --work_dir /root/autodl-tmp/500m_evq
```

需要先定义 500m tier 的配置（参考 125m tier 按比例放大）。

### 成功标准

EVQ τ=1.5 vs Geometric 在 PPL@16K 上改善 ≥ 10%。

---

## 通用注意事项

1. **所有实验结束后**，把结果 JSON/CSV 下载到本地 `results/paper_ready/` 目录
2. **τ trajectory JSON** 必须保存，论文 Figure 需要
3. **每个实验记录**: GPU 型号、总时间、peak memory
4. **随机 seed**: 核心实验用 seed=42，复现用 137 和 256
5. **base=500000**: 所有实验统一使用（对齐 Llama-3 配置）

---

## 代码修改 checklist (Phase 4 专用)

- [ ] `run_evq_sweep.py`: 添加 `--resume_from` 支持（加载 checkpoint，替换 freq schedule）
- [ ] `run_evq_sweep.py`: 添加 PI 方法 (`--method pi --scale_factor 4`)
- [ ] `run_evq_sweep.py`: 添加 YaRN 方法 (`--method yarn --scale_factor 4`)
- [ ] `run_evq_sweep.py`: 添加 `--eval_lengths` 自定义支持
- [ ] `run_evq_sweep.py`: `--seq_len` 支持 8192+ (检查 GPU 内存)
- [ ] 验证 125M 模型在 8192 seq_len 下能否 fit in 32GB VRAM（可能需要 gradient checkpointing 或减小 batch_size）

---

*指南创建: 2026-03-01*
