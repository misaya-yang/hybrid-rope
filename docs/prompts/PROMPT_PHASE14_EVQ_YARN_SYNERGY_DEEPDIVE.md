# Phase 14: EVQ + YaRN 超线性协同 — 冲击 Oral 的核心武器

> **核心命题**: 训练时频率优化 (EVQ) 与推理时位置缩放 (YaRN) 是正交的优化维度。
> 两者组合产生超线性增益: EVQ+YaRN >> EVQ + YaRN（独立效果之和）。
> 如果这个命题在多规模、多长度、多 seed 下成立, 就是改变工业级长上下文外推格局的结果。

> **硬件**: 5090 32GB
> **预计 GPU 时间**: 14A ~2h, 14B ~4h, 14C ~30min (理论分析不需要 GPU)
> **前置**: Phase 9F 750M checkpoints (Geo + Hybrid), 350M 3-seed checkpoints

---

## 为什么这是 Oral 级别的发现

当前工业界长上下文方案:
- **训练时**: 用标准 Geometric RoPE, 靠砸 tokens 续训 (LLaMA-3: 800B→续训 128K)
- **推理时**: YaRN/NTK/Dynamic Scaling 做 inference-time 外推

**问题**: 这两步是割裂的。没有人研究过训练时频率分配对推理时缩放效果的影响。

**我们的发现**:
- Geo + YaRN@8K: ~65% retrieval (3 seed)
- **EVQ + YaRN@8K: 100% retrieval (3 seed, zero variance)**

这不是增量改进, 是质变: EVQ 给 YaRN 提供了更好的"原材料"——频率间距更均匀、碰撞更少的频谱, 使得 YaRN 的平滑插值能完美工作。

**工业影响**: 如果训练时换 1 行 inv_freq + 推理时标准 YaRN 就能从 65% 跳到 100%, 这意味着所有 LLM 的长上下文能力都被训练时的 Geometric 频率分配"卡"住了。EVQ 解锁了这个瓶颈。

---

## Phase 14A: 多规模多 Seed 验证 (~2h)

### 目标
在已有的 checkpoints 上系统验证 EVQ+YaRN 推理时协同, 覆盖:
- 350M (3 seed: 42, 123, 7) → 统计显著性
- 750M (seed 42) → 规模一致性

### 方法

对每个 checkpoint, 评测 4 种组合:
1. **Geo (原始)** — baseline
2. **Geo + YaRN** — 推理时加 YaRN scaling
3. **Hybrid/EVQ (原始)** — 训练时优化
4. **Hybrid/EVQ + YaRN** — 训练时 + 推理时

YaRN 推理时应用方式:
```python
# 在 eval 时替换 inv_freq, 不改模型权重
# YaRN scaling: 对于 scale_factor s = L_eval / L_train
# 低频通道 (wavelength > L_train): freq_new = freq / s
# 高频通道 (wavelength < L_train): freq_new = freq (不变)
# 中间区域: 线性插值过渡

def apply_yarn_scaling(inv_freq, scale_factor, original_max_len=2048):
    """
    标准 YaRN progressive scaling.
    inv_freq: 模型训练好的 inv_freq (可以是 Geo 或 EVQ)
    scale_factor: L_eval / L_train (例如 8192/2048 = 4)
    """
    # 计算 wavelength
    wavelength = 2 * math.pi / inv_freq

    # YaRN 分区
    low_freq_factor = 1.0   # wavelength > beta_slow * L_train 的不缩放
    high_freq_factor = 4.0  # wavelength < beta_fast * L_train 的全缩放
    beta_fast = 32  # 标准 YaRN 参数
    beta_slow = 1   # 标准 YaRN 参数

    low_freq_wavelen = original_max_len / low_freq_factor
    high_freq_wavelen = original_max_len / high_freq_factor

    # 计算 smooth factor
    smooth = (wavelength - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
    smooth = smooth.clamp(0, 1)

    # 混合
    scaled_freq = inv_freq / scale_factor
    new_freq = (1 - smooth) * scaled_freq + smooth * inv_freq

    return new_freq
```

### 评测指标

对每种组合, 在以下长度评测:
- **PPL**: L = 2048, 4096, 8192, 16384
- **Passkey Retrieval**: L = 2048, 4096, 8192 (teacher-forcing NLL gap)
- **Passkey AR Exact Match**: L = 2048, 4096, 8192

### 执行命令

```bash
# === 需要新写的脚本或修改现有脚本 ===
# 核心: 在 eval 阶段对模型 inv_freq 施加 YaRN scaling
# 可以基于 eval_passkey_teacher_forcing.py 修改, 加入 --yarn_scale_factor 参数

# === 14A-1: 350M 3-seed, 4 种组合 ===
for SEED in 42 123 7; do
    for ROPE in geo hybrid; do
        for YARN in none 4; do  # 4 = scale_factor for 8K eval on 2K train
            python scripts/m4_evq_sweep/eval_yarn_synergy.py \
                --checkpoint_dir results/350m_3seed_fineweb/seed${SEED}/${ROPE}/ \
                --tier 350m \
                --rope_type ${ROPE} \
                --yarn_scale_factor ${YARN} \
                --eval_lengths 2048 4096 8192 16384 \
                --passkey_trials 40 \
                --output_dir results/phase14a_yarn_synergy/350m_seed${SEED}_${ROPE}_yarn${YARN}/ \
                --dtype bfloat16
        done
    done
done

# === 14A-2: 750M, 同样 4 种组合 ===
for ROPE in geo hybrid; do
    for YARN in none 4; do
        python scripts/m4_evq_sweep/eval_yarn_synergy.py \
            --checkpoint_dir results/phase9f_750m_2k_1b/${ROPE}/ \
            --tier 750m \
            --rope_type ${ROPE} \
            --yarn_scale_factor ${YARN} \
            --eval_lengths 2048 4096 8192 16384 \
            --passkey_trials 40 \
            --output_dir results/phase14a_yarn_synergy/750m_${ROPE}_yarn${YARN}/ \
            --dtype bfloat16
    done
done
```

### 预期结果

| 组合 | Passkey@8K | PPL@8K | 预期 |
|------|-----------|--------|------|
| Geo | ~50-60% | 115.0 | baseline |
| Geo + YaRN | ~65% | 改善 | YaRN 单独加成 |
| Hybrid/EVQ | ~80% | 121.6 | EVQ 训练时优势 |
| **Hybrid/EVQ + YaRN** | **~100%** | **大幅改善** | **超线性协同** |

**关键判断**: 如果 (EVQ+YaRN) - EVQ > (Geo+YaRN) - Geo, 即 YaRN 在 EVQ 上的增益 > YaRN 在 Geo 上的增益, 就是超线性协同。

### 统计显著性

350M 3-seed × 40 trials = 120 次观测。如果 EVQ+YaRN 全部 100% 而 Geo+YaRN 在 60-70%, Fisher exact test p < 0.001。

---

## Phase 14B: 更长上下文探测 (~4h)

### 目标

推到 16K、32K (16x训练长度), 看 EVQ+YaRN 的协同在多大外推比下仍然有效。

### 为什么重要

- 8K = 4x 训练长度, 已经验证
- **16K = 8x**: 工业级外推标准 (LLaMA-3 从 8K→128K = 16x)
- **32K = 16x**: 如果这也能工作, 就是真正的 "solve extrapolation"

### 执行

```bash
# === 14B-1: 750M, 极长上下文 PPL + Passkey ===
# 注意: 32K 可能 OOM, 先试 bfloat16, 不行用 gradient_checkpointing 或降到 24K
for ROPE in geo hybrid; do
    for YARN in none 4 8 16; do  # 4=8K, 8=16K, 16=32K
        python scripts/m4_evq_sweep/eval_yarn_synergy.py \
            --checkpoint_dir results/phase9f_750m_2k_1b/${ROPE}/ \
            --tier 750m \
            --rope_type ${ROPE} \
            --yarn_scale_factor ${YARN} \
            --eval_lengths 2048 4096 8192 16384 32768 \
            --passkey_trials 20 \
            --output_dir results/phase14b_long_context/750m_${ROPE}_yarn${YARN}/ \
            --dtype bfloat16
    done
done

# === 14B-2 (可选): 350M 单 seed, 极长上下文确认趋势 ===
for ROPE in geo hybrid; do
    for YARN in none 8 16; do
        python scripts/m4_evq_sweep/eval_yarn_synergy.py \
            --checkpoint_dir results/350m_3seed_fineweb/seed42/${ROPE}/ \
            --tier 350m \
            --rope_type ${ROPE} \
            --yarn_scale_factor ${YARN} \
            --eval_lengths 4096 8192 16384 32768 \
            --passkey_trials 20 \
            --output_dir results/phase14b_long_context/350m_${ROPE}_yarn${YARN}/ \
            --dtype bfloat16
    done
done
```

### 预期结果

| 外推比 | Geo+YaRN Passkey | EVQ+YaRN Passkey | 差距 |
|--------|-----------------|-----------------|------|
| 4x (8K) | ~65% | ~100% | +35pp |
| 8x (16K) | ~30-40% | ~80-90%? | 差距扩大 |
| 16x (32K) | ~10-20% | ~50-70%? | 如果仍有效 = 论文核心图 |

### 论文 Figure 构想

**Figure X: EVQ+YaRN 超线性协同**
- X 轴: 上下文长度 (log scale, 2K→32K)
- Y 轴: Passkey Retrieval (%)
- 4 条线: Geo, Geo+YaRN, EVQ, EVQ+YaRN
- EVQ+YaRN 线应该在所有长度上显著高于其他三条

---

## Phase 14C: 理论解释 (不需要 GPU)

### 为什么 EVQ+YaRN 是超线性的?

**直觉**:
- YaRN 的平滑插值假设相邻频率通道之间有足够间距
- Geometric RoPE 的低频通道间距 ∝ b^{-2k/(d-2)} 指数递减 → 低频区高度拥挤 → 碰撞严重
- YaRN 缩放后, 拥挤的低频通道进一步压缩 → 碰撞灾难
- **EVQ 的核心就是扩大低频间距 (cosh 分配)**
- EVQ 低频间距更大 → YaRN 缩放后仍有足够分辨率 → 无碰撞

**形式化**:

设 YaRN 的缩放因子为 s, 频率通道间距为 Δf_k = f_{k+1} - f_k。

Geometric: Δf_k^{geo} = f_k(b^{2/(d-2)} - 1) → 低频区 Δf 指数衰减
EVQ-Cosh: Δf_k^{evq} ∝ cosh(τφ_k) → 低频区 Δf 被 cosh 抬升

YaRN 缩放后的有效间距:
- Geo: Δf_k^{geo}/s → 快速趋零 → 碰撞
- EVQ: Δf_k^{evq}/s → cosh 抬升保持正值 → 无碰撞

**定理构想 (Theorem 5)**:
对于 YaRN 缩放因子 s ≤ s_max(τ), EVQ-cosh 频率分配满足 anti-collision condition:
  min_k Δf_k^{evq}/s ≥ ε_critical
而 Geometric 在 s ≥ s_geo 时必然违反:
  ∃k: Δf_k^{geo}/s < ε_critical

其中 s_max(τ) ∝ cosh(τ) → τ 越大, 允许的外推比越大。

### 写入位置

理论解释写入 CORE_THEORY.md 新增 §12:
"Orthogonal Optimization: Training-Time Frequency Design × Inference-Time Position Scaling"

---

## Phase 14D: 多推理时缩放方法对比 (可选, ~1h)

### 目标

不只测 YaRN, 测所有常见推理时缩放:
- **PI** (Position Interpolation): 全频率均匀缩放
- **NTK-aware**: 动态缩放
- **Dynamic NTK**: 根据实际长度动态调整
- **YaRN**: 渐进式缩放 (重点)

在 Geo 和 EVQ 上分别测, 看 EVQ 是否对所有推理时方法都有增益。

### 预期

| 推理时方法 | Geo + 方法 | EVQ + 方法 | EVQ 增益 |
|-----------|-----------|-----------|---------|
| PI | 极差 (已验证) | 可能稍好 | 小 |
| NTK-aware | 中等 | 好 | 中 |
| Dynamic NTK | 中等 | 好 | 中 |
| **YaRN** | 65% | **100%** | **最大** |

如果 EVQ 对 YaRN 增益最大 → 说明 EVQ 的低频间距改善恰好补上了 YaRN 最需要的地方。

---

## 执行优先级

1. **14A 先跑** (~2h): 350M 3-seed + 750M, 4 种组合 × 标准长度
2. 14A 确认超线性 → **14B 推长** (~4h): 16K, 32K
3. 14A/B 有结果 → **14C 理论**: 写形式化解释, 构造 Theorem 5
4. 14D 可选: 多推理时方法对比 (强化 generality)

## 成功标准

- **Green (Oral 级)**: 350M 3-seed EVQ+YaRN 100%@8K (zero variance), 750M 一致, 16K 仍领先 30pp+ → 核心 Figure + Theorem
- **Yellow (Spotlight 级)**: EVQ+YaRN 显著优于 Geo+YaRN 但非 100% → 依然写入论文, 但不是主打
- **Red**: 差距 <10pp 或不一致 → 检查 YaRN 实现是否正确 (尤其 beta_fast/beta_slow 参数)

## 脚本需求

### 需要新写: `eval_yarn_synergy.py`

核心功能:
1. 加载 checkpoint (支持 Geo 和 Hybrid 两种 rope_type)
2. 可选: 对 inv_freq 施加推理时 YaRN scaling (--yarn_scale_factor)
3. 评测 PPL (多长度) + Passkey (teacher-forcing NLL gap + AR exact match)
4. 输出 JSON 结果

参考:
- `eval_passkey_teacher_forcing.py` 的 passkey 评测逻辑
- `run_evq_sweep.py` 的模型加载和 tier 配置
- Phase 14A 中的 `apply_yarn_scaling()` 函数

### 或者: 修改现有脚本

在 `eval_passkey_teacher_forcing.py` 中加入 `--yarn_scale_factor` 参数:
- 默认 None = 不缩放 (现有行为)
- 给定值 = 在 eval 前对 model.inv_freq 施加 YaRN progressive scaling

这比新写脚本更省事, 也更不容易出 bug。

---

## Claude Code 启动提示

```
读 docs/prompts/PROMPT_PHASE14_EVQ_YARN_SYNERGY_DEEPDIVE.md。

核心任务: 验证 EVQ + 推理时 YaRN 的超线性协同效应。

第一步: 修改 eval_passkey_teacher_forcing.py, 加入 --yarn_scale_factor 参数,
在 eval 时对 model 的 inv_freq 施加 YaRN progressive scaling。

第二步: 在 350M 和 750M checkpoints 上跑 Phase 14A 的 4 种组合评测。

第三步: 汇总结果, 判断是否存在超线性协同 (EVQ+YaRN 增益 > Geo+YaRN 增益)。

注意:
- YaRN 只在 eval 时施加, 不改模型权重
- 确保 Hybrid 模型的 inv_freq 正确加载 (hybrid_evq_inv_freq)
- passkey_trials 至少 40 保证统计功效
```

---

## 论文价值

如果 14A+14B 成功:

**新增叙事层**:
- 现有: "EVQ 是 training-time 最优频率分配"
- 新增: "EVQ 还是 inference-time scaling 的最佳基座"
- 升华: "训练时频率设计 × 推理时位置缩放 = 正交优化空间, EVQ 同时优化了两个维度"

**Reviewer 防御**:
- "EVQ 只在训练时有用" → ❌ 推理时 YaRN 效果翻倍
- "为什么不直接用 YaRN" → ❌ Geo+YaRN 只有 65%, 瓶颈在训练时频率
- "工业上怎么用" → 改 1 行 inv_freq + 标准 YaRN, 零额外成本

这是从 spotlight → oral 的关键证据。"EVQ fundamentally solves the training-time bottleneck for long-context extrapolation" — 这句话如果有 3-seed, 多规模, 多长度的全面支撑, 就是 oral 级别的 claim。
