# Phase 7: 补充验证 + 350M Context Extension

> **目标**: (1) 堵住审稿人漏洞（YaRN 验证、multi-seed、τ lr 消融）；(2) 做 350M 的小步 context extension（512→2K）
> **硬件**: 全部在 RTX 5090 32GB 上跑
> **预计 GPU 时间**: ~4-5h
> **前置**: Phase 6 全部完成，数据在 `data/evq_128tok_results/`

---

## 背景

Phase 6 实验结果非常好，但有两个审稿人风险必须立刻堵上：

1. **YaRN 崩得太狠（2-8x worse）**：审稿人会怀疑实现 bug 或 hyperparameter 不匹配
2. **Extended τ sweep 只有 1 seed**：τ=3.0-5.0 全是 seed=42，统计显著性不足

同时，context extension 实验不要一步跳 2K→8K（太激进 + 需要 Pro 6000）。**先在 5090 上做 350M 的 512→2K 小步扩展**，验证 EVQ 在 context extension 场景下确实有效，数据稳了再上 Pro 6000 做大规模。

---

## 实验 7A: YaRN 实现验证 + 消融

**目的**: 排除 "搞坏 baseline" 的审稿人质疑
**优先级**: ★★★（论文可信度）
**预计时间**: ~15 min

### 7A-1: 实现正确性验证

```python
import torch
from transformers.models.llama.modeling_llama import LlamaLinearScalingRotaryEmbedding

# 我们的 YaRN 实现
our_inv_freq = yarn_inv_freq(dim=64, base=500000,
                              original_max_position=128,
                              target_max_position=8192)

# 打印 32 个频率值
print("Our YaRN inv_freq:", our_inv_freq)

# 同时打印 geometric 和 EVQ 的 inv_freq 做对比
geo_inv_freq = 1.0 / (500000 ** (torch.arange(0, 64, 2).float() / 64))
print("Geometric inv_freq:", geo_inv_freq)

# 验证 YaRN 的低频通道确实被压缩了
print("YaRN / Geo ratio:", our_inv_freq / geo_inv_freq)
# 预期：高频通道 ratio ≈ 1.0，低频通道 ratio ≈ 1/64
```

**必须记录**:
- 32 个频率通道的完整值
- 与 geometric 的 ratio
- 高频/中频/低频的分界点 (low, high)

### 7A-2: YaRN Hyperparameter 消融

YaRN 有两个关键参数 beta_fast 和 beta_slow，控制三段分界：

| Run ID | beta_fast | beta_slow | 描述 |
|--------|-----------|-----------|------|
| Y1 | 32 | 1 | 默认（Phase 6 用的） |
| Y2 | 16 | 1 | 更多通道不缩放 |
| Y3 | 64 | 2 | 更少通道不缩放 |
| Y4 | 32 | 1 | NTK-aware variant: base *= scale^(dim/(dim-2)) |

**每个只需 eval**，不需重新训练。直接生成不同的 inv_freq，用 geometric 的 checkpoint (A1) 做 eval。

**如果 Y2/Y3/Y4 也全崩** → 论文可以写 "YaRN 在所有超参配置下均崩溃于 128-tok from-scratch regime，确认这是方法层面的不兼容而非超参选择问题"。

### 7A-3: 论文中的解释段落

YaRN 崩溃的物理原因（必须写清楚）：

> YaRN's low-frequency scaling divides frequencies by factor s = L_target / L_train. With L_train=128 and L_target=8192, s=64. Channels originally with period T > 128 are compressed to period T/64 < 2 tokens — well below the Nyquist limit for any meaningful positional signal. During 128-token training, these compressed channels cycle through dozens of complete periods, learning only high-frequency noise rather than the long-range structure they were designed to encode. This is not a hyperparameter issue but a fundamental design mismatch: YaRN assumes the model has already learned long-range patterns during pretraining, which is impossible when training length is 128 tokens.

---

## 实验 7B: Multi-Seed 验证（τ=2.5 和 τ=5.0）

**目的**: 为 extended τ sweep 提供统计显著性
**优先级**: ★★★（论文可信度）
**预计时间**: ~30 min

### 要跑的 runs

全部用 128-tok, 125M, FineWeb，只换 seed：

| Run ID | τ | Seed | 预计时间 |
|--------|---|------|---------|
| S1 | 2.5 | 137 | ~5 min |
| S2 | 2.5 | 256 | ~5 min |
| S3 | 5.0 | 137 | ~5 min |
| S4 | 5.0 | 256 | ~5 min |

结合 Phase 6 已有的 seed=42 数据，每个 τ 值有 3 个 seed。

### 关键输出

```
τ=2.5: PPL@8K = [seed42, seed137, seed256] → mean ± std
τ=5.0: PPL@8K = [seed42, seed137, seed256] → mean ± std
```

**验证标准**: 如果 3-seed 的 std < 5% of mean，统计显著性足够。参考 Phase 3 的 learnable τ 结果（std/mean = 2.8%），τ sweep 的方差应该类似。

---

## 实验 7C: τ 学习率敏感性消融

**目的**: 回应 "learnable τ 的收敛值是否依赖 lr" 的审稿人质疑
**优先级**: ★★（消融实验）
**预计时间**: ~25 min

### 要跑的 runs

128-tok, 125M, FineWeb, seed=42：

| Run ID | τ lr_mult | τ 预期收敛值 | 预计时间 |
|--------|-----------|-------------|---------|
| L1 | 1 | ~1.0（梯度太小，不动） | ~5 min |
| L2 | 5 | ~1.1 | ~5 min |
| L3 | 10 | 1.14（已有数据） | 不需要跑 |
| L4 | 20 | ~1.14（如果 landscape 真的平，lr 大也无所谓） | ~5 min |
| L5 | 50 | ~1.14 or 不稳定 | ~5 min |
| L6 | 100 | 可能震荡 | ~5 min |

### 关键验证

如果 lr_mult=5/10/20/50 都收敛到 τ ≈ 1.14 → 论文可以写 "learnable τ 对学习率不敏感，在 10x 范围内收敛到相同值"。

如果 lr_mult=100 震荡 → 说明 loss landscape 虽然平但不是完全平的，有一个浅谷在 τ≈1.14。

---

## 实验 7D: 补全缺失的 PPL@128 数据

**目的**: 完整的 waterbed 分析需要每个 τ 值的 PPL@128
**优先级**: ★（数据完整性）
**预计时间**: ~10 min

Phase 6A 中 τ=0.5, 2.0, 2.5 的 PPL@128 在 JSON 里是 null。直接加载对应的 checkpoint，在 128 token 上 eval 即可。

---

## 实验 7E: NTK-Aware Baseline（128-tok）

**目的**: 补一个审稿人可能要求的 baseline
**优先级**: ★★
**预计时间**: ~10 min

NTK-aware 就是放大 base：base_new = base * s^(dim/(dim-2))，其中 s = L_target / L_train。

两种模式（同 YaRN）：
| Run ID | Method | 描述 |
|--------|--------|------|
| N1 | NTK-train | NTK 频率从头训练 128 tok |
| N2 | NTK-infer | Geometric 训练，推理时用 NTK 频率 |

FineWeb 和 TinyStories 各一组。

---

## 实验 7F: 350M Context Extension（512→2K）

**目的**: 在接近实际部署的场景下验证 EVQ，模型规模够大，扩展比适中
**优先级**: ★★★（论文核心实验）
**预计时间**: ~3h

### 为什么是 350M + 512→2K

1. **350M 够大**：24 层 16 头，模型有足够容量做有意义的语言建模
2. **512→2K 扩展比 = 4x**：和 DeepSeek 的 4K→16K 或 Llama 的 8K→32K 一样的倍数
3. **5090 32GB 完全够**：训练 512 tok ~5GB，eval 到 8K ~1.6GB
4. **时间可控**：预训练 ~1.5h，续训 ~1.5h

### 阶段 1：512-tok 预训练（Geometric RoPE）

```python
MODEL_CONFIG = dict(hidden=1024, layers=24, heads=16, head_dim=64)  # 350M
TRAIN_SEQ_LEN = 512
TRAIN_TOKENS = 50_000_000  # 50M tokens
ROPE_BASE = 500_000
BATCH_SIZE = 16
DATASET = "HuggingFaceFW/fineweb-edu-score-2"
```

**预计时间**: ~1.5h
**产出**: `pretrain_350m_512tok/checkpoint.pt`

### 阶段 2：2K 续训（Context Extension）

从 512-tok checkpoint 出发，在 2K 上续训：

```python
TRAIN_SEQ_LEN = 2048
TRAIN_TOKENS = 5_000_000  # 5M tokens（续训量 = 预训练的 10%）
BATCH_SIZE = 4
# 其余同上
```

| Run ID | Method | τ / 配置 | 预计时间 |
|--------|--------|---------|---------|
| H1 | Geometric | 不改频率 | ~15 min |
| H2 | PI | 所有频率 ÷ 4 (2048/512) | ~15 min |
| H3 | YaRN | 分段缩放，s=4 | ~15 min |
| H4 | EVQ τ=1.5 | cosh 频率分配 | ~15 min |
| H5 | EVQ τ=2.0 | cosh 频率分配（1024-tok 的 τ*） | ~15 min |

**预计续训时间**: 5 × 15 min = ~1.25h

### 评估

```python
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192]
# 512 = 原始训练窗口（不应退化）
# 2048 = 续训窗口（应该最好）
# 4096/8192 = 外推（EVQ 应该赢）
```

### 关键验证

1. **PPL@512 不退化**: 续训后不应比预训练时更差
2. **PPL@2K 显著改善**: 所有方法都应在 2K 上好于不续训的 baseline
3. **PPL@4K/8K 外推**: EVQ 应该比 PI/YaRN/Geo 更好
4. **对比**:
   - 如果 EVQ τ=2.0 > EVQ τ=1.5 > YaRN > PI > Geo → 完美叙事
   - 如果 YaRN 在 context extension 中表现正常 → 更好！这证明 YaRN 只在 from-scratch 崩，在 extension 中有效，而 EVQ 两者都有效

### Passkey 评估（如果 350M 够大的话）

续训完的 350M 模型尝试做 passkey retrieval（AR exact match）：
- 长度 1K, 2K, 4K
- 每个长度 100 trials
- 如果 350M 能做 passkey → 这是非常强的 EVQ 证据
- 如果不能 → 用 NLL-gap 方法（和 Phase 6D 一样）

---

## 优先级排序

| 优先级 | 实验 | 时间 | 意义 |
|--------|------|------|------|
| ★★★ P0 | 7B: Multi-seed (τ=2.5, 5.0) | 30 min | 统计显著性，堵审稿人 |
| ★★★ P0 | 7A: YaRN 验证+消融 | 15 min | 排除 "搞坏 baseline" |
| ★★★ P1 | 7F: 350M 512→2K context extension | 3h | 论文核心实验 |
| ★★ P2 | 7C: τ lr 敏感性消融 | 25 min | 消融实验 |
| ★★ P2 | 7E: NTK baseline | 10 min | 补 baseline |
| ★ P3 | 7D: 补 PPL@128 数据 | 10 min | 数据完整性 |

**推荐执行顺序**: 7B + 7A（并行，~30 min）→ 7F 开始预训练（~1.5h）→ 在预训练跑的同时做 7C + 7D + 7E（~30 min）→ 7F 续训（~1.25h）→ 7F passkey eval

**总 GPU 时间**: ~4-5h（可以在一天内完成）

---

## 代码要求

### 目录结构

```
/root/autodl-tmp/evq_phase7/
├── yarn_verification/       # 7A
│   ├── inv_freq_comparison.json
│   └── yarn_ablation/
├── multiseed/               # 7B
│   ├── tau2.5_seed137/
│   ├── tau2.5_seed256/
│   ├── tau5.0_seed137/
│   └── tau5.0_seed256/
├── tau_lr_sensitivity/      # 7C
├── ppl128_completion/       # 7D
├── ntk_baseline/            # 7E
├── context_extension_350m/  # 7F
│   ├── pretrain_512tok/
│   ├── extend_geo/
│   ├── extend_pi/
│   ├── extend_yarn/
│   ├── extend_evq_1.5/
│   ├── extend_evq_2.0/
│   └── passkey_eval/
├── results_phase7.json
└── phase7_report.md
```

### 注意事项

1. **7A 的 YaRN 验证最关键**: 一定要打印完整的 inv_freq 和参数。如果有 bug 就修，如果没 bug 就是方法不兼容。
2. **7B 的 multi-seed**: 必须和 Phase 6A 用完全相同的代码和配置，只换 seed。
3. **7F 预训练必须用标准 geometric**: 不要在预训练阶段引入任何频率修改。
4. **7F 续训的 YaRN**: 注意此时 s = 2048/512 = 4（不是 64！），YaRN 的压缩没那么极端，应该能正常工作。如果 YaRN 在 s=4 的 context extension 中表现良好，而在 s=64 的 from-scratch 中崩溃，这本身就是一个很好的发现。
5. **每跑完一个实验立刻保存 JSON**: 防止中途断连。
