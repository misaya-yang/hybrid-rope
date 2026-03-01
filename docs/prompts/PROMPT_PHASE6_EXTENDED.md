# Phase 6: Extended τ Sweep + Baselines + Context Extension

> **目标**: 补齐论文所有缺失实验，找到真正的 τ peak，加上 SOTA baselines，并做 context extension 核心实验
> **硬件分配**:
> - **Part A (RTX 5090 32GB)**: 6A/6B/6C/6D — 128-tok 轻量实验，~2h
> - **Part B (RTX Pro 6000 96GB)**: 6E/6F — 1024-tok 和 2K→8K context extension，~6-10h
> **前置**: Phase 1-3 + mini-sweep 已完成，数据在 `/root/autodl-tmp/evq_128tok/` 和 `/root/autodl-tmp/evq_minisweep*/`

---

## 背景：核心发现与未解问题

### 已确认的发现

上一轮 mini-sweep（128-tok, 125M, FineWeb + TinyStories）：

| τ | FineWeb PPL@8K | TinyStories PPL@8K |
|---|----------------|---------------------|
| 0.0 | 513.7 | 30.95 |
| 0.5 | 524.8 (+2.1%) | 33.75 (+9.1%) |
| 1.0 | 477.5 (-7.1%) | 25.69 (-17.0%) |
| 1.5 | 419.7 (-18.3%) | 22.29 (-28.0%) |
| 2.0 | 406.1 (-20.9%) | 19.67 (-36.4%) |
| 2.5 | 383.3 (-25.4%) | 17.47 (-43.6%) |

### ⚠️ 关键矛盾：τ* 随训练 regime 变化

50M TinyStories 从零训练（**2K tokens**, 500K steps）：

| τ | 50M PPL@16K (2K-tok 训练) | Δ vs Geo |
|---|--------------------------|----------|
| 0.0 | 33.316 | — |
| 1.5 | **29.697** | **-10.9%** |
| 2.0 | 35.646 | **+7.0%** (更差！) |

**128-tok regime**: τ=2.5 还在改善，τ*≈2.7
**2K-tok regime**: τ=1.5 最优，τ=2.0 已经变差

**核心洞察**：RoPE 一共 d/2 个频率通道（head_dim=64 → 32 个），这是硬约束。τ 在这 32 个通道里重新分配资源。但模型的 Q/K 投影矩阵也在这同一个 32 维空间里做"软分配"——模型越强（越大/训练越充分），自己能做的分配就越好，τ 的边际收益递减，过大反而干扰模型自己找到的最优分配。

但模型权重**做不到**的是改变每个通道本身的频率值。**当序列超出训练长度时**（context extension），低频通道从未见过完整周期，此时 PE 的频率分配重新成为主导因素。

这意味着：
- **训练窗口内（model-dominant）**: τ 的收益小，过大有害
- **外推/context extension（PE-dominant）**: τ 的收益大，正是 EVQ 发力的地方

### 本轮必须回答的问题

1. 128-tok regime 下 τ 的真正 peak 在哪？
2. 高 τ 时 PPL@128（in-distribution）是否恶化？（waterbed）
3. YaRN 在同一 128-tok 协议下表现如何？
4. **1024-tok regime 下 τ* 是否介于 1.5 和 2.7 之间？（regime 敏感性）**
5. **2K 预训练 → 8K 续训场景下，EVQ 对比 Geo/PI/YaRN 如何？（context extension 核心实验）**
6. Passkey retrieval 上 EVQ 是否也赢？

---

# Part A: RTX 5090 32GB 轻量实验

> 这些实验全部使用 128-tok 训练的 125M 模型，在 RTX 5090 32GB 上跑
> 预计总时间：~2h

## 实验 6A: Extended τ Sweep (τ = 3.0, 3.5, 4.0, 5.0)

**目的**: 找到 128-tok regime 下 PPL@8K 的真正 peak
**优先级**: P0（必须）

### 配置

复用上一轮完全相同的训练设置：

```python
MODEL_CONFIG = dict(hidden=768, layers=12, heads=12, head_dim=64)  # 125M
TRAIN_SEQ_LEN = 128
TRAIN_TOKENS = 15_000_000  # 15M
ROPE_BASE = 500_000
BATCH_SIZE = 64
DATASET = "HuggingFaceFW/fineweb-edu-score-2"  # sample-10BT
EVAL_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]
```

### 要跑的 runs

| Run ID | Method | τ | Dataset | 预计时间 |
|--------|--------|---|---------|---------|
| D1 | Fixed EVQ | 3.0 | FineWeb | ~5 min |
| D2 | Fixed EVQ | 3.5 | FineWeb | ~5 min |
| D3 | Fixed EVQ | 4.0 | FineWeb | ~5 min |
| D4 | Fixed EVQ | 5.0 | FineWeb | ~5 min |
| D5 | Fixed EVQ | 3.0 | TinyStories | ~5 min |
| D6 | Fixed EVQ | 3.5 | TinyStories | ~5 min |
| D7 | Fixed EVQ | 4.0 | TinyStories | ~5 min |
| D8 | Fixed EVQ | 5.0 | TinyStories | ~5 min |

**预计时间**: 8 × 5 min = ~40 min

### 关键输出

1. 合并之前的 sweep 数据，画出完整的 τ vs PPL@8K 曲线（τ = 0.0 到 5.0）
2. 标注 peak 位置（如果在 [2.5, 5.0] 内找到）
3. 如果 τ=5.0 仍在下降 → **停！** 检查 PPL@128 是否已经恶化（>5%），如果是则 peak 在 waterbed 约束下；如果否则需要继续外推
4. 重新用所有点做二次/三次拟合，报告 τ* 和 R²
5. **重要**: 记录每个 τ 的 PPL@128，用于 waterbed 分析

### 结果记录格式

```json
{
  "experiment": "6A_extended_sweep",
  "runs": [
    {"tau": 3.0, "dataset": "fineweb", "ppl": {"128": X, "256": X, "..": X, "8192": X}},
    "..."
  ],
  "quadratic_fit": {"fineweb": {"a": X, "b": X, "tau_star": X, "r2": X}, "tinystories": {"...": "..."}},
  "peak_found": true,
  "peak_tau": X
}
```

---

## 实验 6B: YaRN Baseline

**目的**: 与当前 SOTA 的 RoPE 扩展方法对比
**优先级**: P1（必须）

### YaRN 实现

YaRN 的核心是在 RoPE 频率上做分段缩放:

```python
import torch
import math

def yarn_inv_freq(dim, base, original_max_position=128, target_max_position=8192, beta_fast=32, beta_slow=1):
    """
    YaRN frequency scaling.
    Scale factor s = target_max_position / original_max_position
    """
    s = target_max_position / original_max_position  # = 64
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    low = math.floor(dim * math.log(original_max_position / (beta_fast * 2 * math.pi)) / (2 * math.log(base)))
    high = math.ceil(dim * math.log(original_max_position / (beta_slow * 2 * math.pi)) / (2 * math.log(base)))
    low = max(low, 0)
    high = min(high, dim // 2 - 1)

    scaled_inv_freq = inv_freq.clone()
    for i in range(dim // 2):
        if i < low:
            scaled_inv_freq[i] = inv_freq[i]
        elif i > high:
            scaled_inv_freq[i] = inv_freq[i] / s
        else:
            t = (i - low) / (high - low)
            gamma = (1 - t)
            scaled_inv_freq[i] = inv_freq[i] / (1 + (s - 1) * (1 - gamma))

    return scaled_inv_freq
```

### 要跑的 runs

| Run ID | Method | 描述 | 预计时间 |
|--------|--------|------|---------|
| E1 | YaRN-train | YaRN 频率分配从头训练 128 tok，eval 到 8K | ~5 min |
| E2 | YaRN-infer | Geometric 训练 128 tok，推理时用 YaRN 扩展 eval | ~1 min (复用 A1 checkpoint) |

**对于 E2**: 直接加载 A1 (Geometric) 的 checkpoint，在 eval 时替换 inv_freq 为 YaRN 频率即可。

**FineWeb 和 TinyStories 都要跑。**

**预计时间**: ~12 min

---

## 实验 6C: 50M Model Scaling Point

**目的**: 验证 128-tok regime 下 τ* 是否随 model size 变化
**优先级**: P3（建议）

### 配置

```python
MODEL_CONFIG_50M = dict(
    hidden=512,
    layers=8,
    heads=8,
    head_dim=64  # 保持 head_dim=64，与 125M 一致
)
# 其他完全相同: seq_len=128, tokens=15M, base=500000
```

### 要跑的 runs

| Run ID | Model | Method | τ | Dataset |
|--------|-------|--------|---|---------|
| F1 | 50M | Geometric | 0.0 | FineWeb |
| F2 | 50M | Fixed EVQ | 1.5 | FineWeb |
| F3 | 50M | Fixed EVQ | 2.5 | FineWeb |
| F4 | 50M | Fixed EVQ | τ* from 6A | FineWeb |
| F5 | 50M | Learnable EVQ | init=1.0 | FineWeb |

**预计时间**: 5 × 3 min = ~15 min

---

## 实验 6D: Passkey Retrieval

**目的**: 用非 PPL 指标验证 PE quality
**优先级**: P2（强烈建议）

### 要跑的 runs

对每个方法，在不同序列长度和 passkey 位置测试：

| 序列长度 | Passkey 位置 | 试验次数 |
|---------|-------------|---------|
| 2048 | 0.5 | 100 次 |
| 8192 | 0.5 | 100 次 |

方法列表（复用已有 checkpoint）：

| Method | Checkpoint |
|--------|-----------|
| Geometric (τ=0) | A1 |
| EVQ τ=1.5 | A3 |
| EVQ τ=2.5 | mini-sweep |
| EVQ τ=τ* (from 6A) | D1-D8 中最优 |
| DAPE (lr×100) | B2 |
| YaRN-infer | A1 + YaRN freq |

**预计时间**: ~10 min

**注意**: 如果 125M 模型太小无法做 passkey retrieval（causal LM 可能学不到这种能力），用 PPL@needle_position 作为替代指标。

---

# Part B: RTX Pro 6000 96GB 重量实验

> 这些实验需要更大显存和更长时间
> 预计总时间：~6-10h

## 实验 6E: 1024-tok Regime Sensitivity

**目的**: 验证 τ* 随训练长度递减的假设
**优先级**: P1.5（必须）
**硬件**: RTX Pro 6000 96GB（1024-tok 训练 125M 需要更大显存）

### 配置

```python
MODEL_CONFIG = dict(hidden=768, layers=12, heads=12, head_dim=64)  # 125M
TRAIN_SEQ_LEN = 1024  # 与 128-tok 实验的唯一区别
TRAIN_TOKENS = 15_000_000  # 15M（保持一致）
ROPE_BASE = 500_000
BATCH_SIZE = 16  # 可能需要减小，根据显存调整
DATASET = "HuggingFaceFW/fineweb-edu-score-2"
EVAL_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
```

### 要跑的 runs

| Run ID | Method | τ | Train Len | Dataset | 预计时间 |
|--------|--------|---|-----------|---------|---------|
| G1 | Geometric | 0.0 | 1024 | FineWeb | ~15 min |
| G2 | Fixed EVQ | 1.0 | 1024 | FineWeb | ~15 min |
| G3 | Fixed EVQ | 1.5 | 1024 | FineWeb | ~15 min |
| G4 | Fixed EVQ | 2.0 | 1024 | FineWeb | ~15 min |
| G5 | Fixed EVQ | 2.5 | 1024 | FineWeb | ~15 min |
| G6 | Learnable EVQ | init=1.0 | 1024 | FineWeb | ~15 min |

**预计时间**: 6 × 15 min = ~1.5h

### 关键验证

如果 1024-tok 的 τ* 介于 1.5（2K-tok 结果）和 2.7（128-tok 结果）之间，就确认了：
- **τ* 随训练长度单调递减**
- 实际部署建议：训练长度越长 → 用越小的 τ
- 128-tok 的 τ*≈2.7 是 PE-dominant 极限，不是实际最优

---

## 实验 6F: Context Extension（论文核心实验）

**目的**: 模拟工业界的 context extension 流程，验证 EVQ 在最接近实际部署场景下的效果
**优先级**: ★★★ P0.5（与 6A 同等重要——这是论文 story 的核心）
**硬件**: RTX Pro 6000 96GB（2K 预训练 + 8K 续训需要大显存）

### 为什么这是核心实验

工业界做长上下文的标准流程是：
1. 短上下文预训练（如 4K/8K）——用标准 geometric RoPE
2. 长上下文续训（context extension）——换频率方案（NTK/YaRN/PI），在长文本上续训
3. SFT/RLHF

**EVQ 的实际价值在步骤 2**。之前我们所有实验都是从零训练（不是 context extension），这在审稿人看来不够接近实际应用。这个实验直接补上这一块。

### 阶段 1：2K 预训练（Geometric RoPE）

```python
MODEL_CONFIG = dict(hidden=768, layers=12, heads=12, head_dim=64)  # 125M
TRAIN_SEQ_LEN = 2048
TRAIN_TOKENS = 100_000_000  # 100M tokens（比 128-tok 的 15M 多，确保模型充分训练）
ROPE_BASE = 500_000
BATCH_SIZE = 8  # 2K seq_len 需要更小 batch
DATASET = "HuggingFaceFW/fineweb-edu-score-2"
```

训练完得到一个 "2K pretrained" 的 baseline checkpoint。

**预计时间**: ~2-3h

### 阶段 2：8K 续训（Context Extension）

从 2K checkpoint 出发，在 8K 长文本上续训，分别用不同的频率方案：

```python
TRAIN_SEQ_LEN = 8192
TRAIN_TOKENS = 10_000_000  # 10M tokens（续训量相对少，模拟工业实践）
# 其余同上
```

| Run ID | Method | 频率方案 | 预计时间 |
|--------|--------|---------|---------|
| H1 | Geometric | 原始 geometric，不做任何修改 | ~1h |
| H2 | PI | 所有频率 ÷ (8192/2048) = ÷4 | ~1h |
| H3 | YaRN | 分段缩放（高频不动、低频 ÷4） | ~1h |
| H4 | NTK-aware | base × 4（等比放大） | ~1h |
| H5 | EVQ τ=1.5 | cosh 频率分配（安全默认） | ~1h |
| H6 | EVQ τ=τ* | τ 用 mini-sweep 从 6A 确定 | ~1h |

### 评估

每个方法在续训后评估：

```python
EVAL_LENGTHS = [2048, 4096, 8192, 16384, 32768]
# 2K = 原始训练窗口（不应退化）
# 8K = 续训窗口（应该最好）
# 16K/32K = 外推（EVQ 应该在这里赢）
```

### 关键验证

1. **PPL@2K 不退化**：续训后 PPL@2K 应该 ≤ 预训练时的 PPL@2K（不能变差）
2. **PPL@8K 最好**：这是续训目标，所有方法都应该显著改善
3. **PPL@16K/32K 外推**：EVQ 应该比 PI/YaRN/NTK 更好，因为 cosh 分配在外推区间有理论保证
4. **如果 EVQ 在外推区间赢了但在 8K 内输了** → 也是好结果，因为这直接验证了 "PE-dominant vs model-dominant" 的理论

**预计时间**: 预训练 2-3h + 续训 6 × 1h = ~8-9h

### 简化方案（如果时间紧）

只跑 4 个最关键的方法：H1(Geo), H3(YaRN), H5(EVQ τ=1.5), H6(EVQ τ*)。~6h。

---

## 优先级排序（综合）

### RTX 5090 上立即跑（~2h）

| 优先级 | 实验 | 必要性 | 论文意义 |
|--------|------|--------|---------|
| **P0** | 6A: Extended τ sweep | 必须 | 没有 peak 就不知道 τ* 到底是多少 |
| **P1** | 6B: YaRN baseline | 必须 | 审稿人会问 "为什么不和 YaRN 比" |
| **P2** | 6D: Passkey retrieval | 强烈建议 | 证明 EVQ 不只是 PPL trick |
| **P3** | 6C: 50M scaling | 建议 | 证明 τ* 不依赖 model size |

### RTX Pro 6000 上排队跑（~8-10h）

| 优先级 | 实验 | 必要性 | 论文意义 |
|--------|------|--------|---------|
| **P0.5** | 6F: Context extension (2K→8K) | ★核心★ | 直接模拟工业实践，论文 story 的基石 |
| **P1.5** | 6E: 1024-tok regime sensitivity | 必须 | 验证 τ* 随训练长度递减 |

---

## 代码要求

### 目录结构

```
/root/autodl-tmp/evq_phase6/
├── extended_sweep/          # 6A results (5090)
├── yarn_baseline/           # 6B results (5090)
├── scaling_50m/             # 6C results (5090)
├── passkey_retrieval/       # 6D results (5090)
├── train_1024tok/           # 6E results (Pro 6000)
├── context_extension/       # 6F results (Pro 6000)
│   ├── pretrain_2k/         # 阶段 1: 2K 预训练 checkpoint
│   ├── extend_geo/          # H1
│   ├── extend_pi/           # H2
│   ├── extend_yarn/         # H3
│   ├── extend_ntk/          # H4
│   ├── extend_evq_1.5/      # H5
│   └── extend_evq_star/     # H6
├── results_phase6.json      # 汇总所有结果
└── phase6_report.md         # 自动生成的报告
```

### 汇总 JSON 格式

```json
{
  "phase": 6,
  "date": "2026-03-01",
  "hardware": {"part_a": "RTX 5090 32GB", "part_b": "RTX Pro 6000 96GB"},
  "experiments": {
    "6A_extended_sweep": {
      "complete_tau_curve": {
        "fineweb": [
          {"tau": 0.0, "ppl_128": "X", "ppl_8192": "X"},
          {"tau": 0.5, "ppl_128": "X", "ppl_8192": "X"},
          "...",
          {"tau": 5.0, "ppl_128": "X", "ppl_8192": "X"}
        ],
        "tinystories": ["..."]
      },
      "peak_tau": {"fineweb": "X", "tinystories": "X"},
      "fit": {"fineweb": {"type": "quadratic/cubic", "tau_star": "X", "r2": "X"}}
    },
    "6B_yarn": {
      "yarn_train": {"fineweb": {"ppl_8k": "X"}, "tinystories": {"ppl_8k": "X"}},
      "yarn_infer": {"fineweb": {"ppl_8k": "X"}, "tinystories": {"ppl_8k": "X"}}
    },
    "6E_1024tok": {
      "tau_curve": [{"tau": "X", "ppl_8k": "X"}],
      "tau_star_1024": "X",
      "regime_comparison": {
        "128tok_tau_star": "~2.7",
        "1024tok_tau_star": "X",
        "2ktok_tau_star": "~1.5",
        "monotone_decreasing": "true/false"
      }
    },
    "6F_context_extension": {
      "pretrain_2k_ppl": "X",
      "methods": {
        "geometric": {"ppl_2k": "X", "ppl_8k": "X", "ppl_16k": "X", "ppl_32k": "X"},
        "pi": {"ppl_2k": "X", "ppl_8k": "X", "ppl_16k": "X", "ppl_32k": "X"},
        "yarn": {"ppl_2k": "X", "ppl_8k": "X", "ppl_16k": "X", "ppl_32k": "X"},
        "ntk": {"ppl_2k": "X", "ppl_8k": "X", "ppl_16k": "X", "ppl_32k": "X"},
        "evq_1.5": {"ppl_2k": "X", "ppl_8k": "X", "ppl_16k": "X", "ppl_32k": "X"},
        "evq_star": {"ppl_2k": "X", "ppl_8k": "X", "ppl_16k": "X", "ppl_32k": "X"}
      },
      "best_at_8k": "method_name",
      "best_at_16k": "method_name",
      "best_at_32k": "method_name"
    }
  }
}
```

---

## 注意事项

1. **复用已有代码**: 训练脚本在 `/root/autodl-tmp/evq_128tok/` 下应该已有。不要重写，直接修改参数复用。
2. **复用已有 checkpoint**: Passkey 和 YaRN-infer 不需要重新训练，直接加载已有 checkpoint。
3. **先跑 P0 再跑其他**: 如果 6A 发现 τ=3.0 就开始恶化，后续实验的 τ* 就确定了，可以立即开始 6B-6D。
4. **TinyStories 用 `roneneldan/TinyStories`**: 与上一轮一致。
5. **每跑完一个实验就保存 checkpoint JSON**: 防止中途断连丢失数据。
6. **YaRN 实现**: 如果项目代码里没有 YaRN，请按照上面的伪代码实现。注意 128-tok 实验中 `original_max_position=128`, `target_max_position=8192`。
7. **Passkey**: 如果 125M 模型太小无法做 passkey retrieval，用 PPL@needle_position 作为替代指标。
8. **Context extension (6F) 的 YaRN/PI 参数**: `original_max_position=2048`（预训练长度），`target_max_position=8192`（续训目标长度），scale factor = 4。
9. **6F 预训练必须用标准 geometric RoPE**: 不要在预训练阶段引入任何频率修改。所有方法的差异只在续训阶段体现。
10. **6F 续训时 optimizer 状态**: 可以从头初始化（不需要继承预训练的 optimizer state），只继承 model weights。
