# Phase 8: 大扩展比 Context Extension + Passkey 恢复实验

> **目标**: (1) 在 5090 32GB 上把扩展比从 4x 提升到 8x（512→4K），验证 Geometric 在大扩展比下崩溃、EVQ 反超的假说；(2) 通过增加续训量测试 passkey 是否能恢复；(3) 补充 from-scratch 4K 对照组
> **硬件**: RTX 5090 32GB（峰值实测 ~20.9GB，余量 ~12GB）
> **预计 GPU 时间**: ~5-6h
> **前置**: Phase 7F 完成。350M pretrain checkpoint 在 `/root/autodl-tmp/evq_phase7/context_extension_350m/pretrain_512tok/checkpoint.pt`

---

## 背景：Phase 7F 结果与核心矛盾

### 7F 数据回顾（512→2K，4x 扩展比）

| Method | PPL@2K | PPL@8K | Passkey@1K | Passkey@2K | Passkey@4K |
|--------|--------|--------|------------|------------|------------|
| Geometric | 73.2 | 98.0 | 90% | 78% | 52% |
| PI | 139.8 | 246.1 | — | — | — |
| YaRN | 108.3 | 174.5 | — | — | — |
| EVQ τ=1.5 | 73.8 | 99.4 | 78% | 70% | 48% |
| EVQ τ=2.0 | 73.5 | 99.1 | 82% | 72% | 40% |

**PPL**: EVQ ≈ Geometric >> PI/YaRN — 好消息
**Passkey**: EVQ 全面落后 Geometric（1K: -12pp, 2K: -8pp, 4K: -4pp）— 坏消息

### 为什么 Passkey 输了？

**假说: Q/K 对齐破坏**

350M 在 512-tok 预训练了 5000 万 token，Q/K 矩阵和 geometric 频率形成了深度耦合。续训时改变频率分配（EVQ）相当于在 32 维频率空间中做了非线性变换。

- **PPL 能恢复**：PPL 是全局平均指标，5M token 续训（预训练的 10%）足够让 Q/K 大致适应新频率空间
- **Passkey 恢复不了**：Passkey 要求在精确位置上做正确的注意力分配，对 Q/K 对齐的精度要求高一个数量级

**验证方式**：增加续训量（10%→20%→50%），如果 passkey 随续训量恢复 → 确认是 alignment gap 问题。

### 为什么要更大扩展比？

4x 扩展比对 Geometric 太温和。Geometric 的低频通道 θ_min ≈ 1/500000^(62/64) ≈ 3.3e-6，周期 T = 2π/θ ≈ 1.9M tokens。在 4K eval 长度上，低频通道只走了 0.2% 的周期——完全不构成压力。

但 EVQ-cosh 的优势恰恰在于重新分配低频端。**只有当 Geometric 的低频端真正"不够用"时（更长序列），EVQ 才会反超。** 8x 扩展比（512→4K）+ eval 到 16K 可以加大这个压力。

---

## 实验 8A: 512→4K Context Extension（8x 扩展比）

**目的**: 扩展比翻倍，验证 EVQ 在更大扩展比下反超 Geometric
**优先级**: ★★★（核心实验）
**预计时间**: ~2h

### 显存估算

350M 模型在 4K seq_len 训练：
- 参数: ~1.3GB (fp32)
- 激活 (4K, batch=2): ~4-5GB (gradient checkpointing)
- 优化器: ~2.6GB
- 总训练: ~10-12GB
- Eval@16K (batch=1): ~3-4GB
- **峰值 ~12-14GB，5090 32GB 完全安全**

### 配置

```python
# 复用 7F 的 pretrain checkpoint
PRETRAIN_CKPT = "/root/autodl-tmp/evq_phase7/context_extension_350m/pretrain_512tok/checkpoint.pt"

MODEL_CONFIG = dict(hidden=1024, layers=24, heads=16, head_dim=64)  # 350M
TRAIN_SEQ_LEN = 4096  # 8x expansion from 512
TRAIN_TOKENS = 10_000_000  # 10M tokens（= 预训练的 20%，比 7F 的 5M 翻倍）
ROPE_BASE = 500_000
BATCH_SIZE = 2  # 4K seq_len 需要小 batch
LR = 3e-5  # 续训 lr，比预训练低
DATASET = "HuggingFaceFW/fineweb-edu-score-2"
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
```

### 要跑的 runs

| Run ID | Method | 频率方案 | 预计时间 |
|--------|--------|---------|---------|
| A1 | Geometric | 原始频率不变 | ~20 min |
| A2 | PI | 所有频率 ÷ 8 (4096/512) | ~20 min |
| A3 | YaRN | 分段缩放, s=8, original_max=512, target_max=4096 | ~20 min |
| A4 | EVQ τ=1.5 | cosh 分配, target=4096 | ~20 min |
| A5 | EVQ τ=2.0 | cosh 分配, target=4096 | ~20 min |
| A6 | EVQ τ=2.5 | cosh 分配, target=4096（试探更高 τ） | ~20 min |
| A7 | Hybrid EVQ τ=2.0 | 高频 8 通道保持 Geometric，低频 24 通道用 EVQ-cosh τ=2.0 | ~20 min |

**预计续训时间**: 7 × 20 min = ~2.3h

### EVQ 频率计算方式

Context extension 场景下的 EVQ 频率生成（关键！）：

```python
import torch
import math

def evq_cosh_inv_freq(dim, base, tau, original_max=512, target_max=4096):
    """
    EVQ-cosh 频率分配 for context extension.

    核心: 在 [θ_min, θ_max] 区间用 cosh 分配替代 geometric 等比分配
    θ_max = 1.0 (最高频)
    θ_min = 1/base^((dim-2)/dim)  (最低频)

    u_k = k / (dim/2 - 1)  for k = 0, ..., dim/2-1
    φ(u; τ) = 1 - (1/τ) * arcsinh((1-u) * sinh(τ))
    θ_k = θ_min^φ(u_k; τ) * θ_max^(1-φ(u_k; τ))
    """
    n = dim // 2
    theta_max = 1.0
    theta_min = 1.0 / (base ** ((dim - 2) / dim))

    u = torch.arange(n, dtype=torch.float64) / (n - 1)  # [0, 1]

    if abs(tau) < 1e-8:
        phi = 1.0 - u  # geometric
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))

    inv_freq = (theta_min ** phi) * (theta_max ** (1.0 - phi))
    return inv_freq.float()
```

**注意**：EVQ 频率不需要知道 original_max 或 target_max，因为 cosh 分配是在频率空间上操作的，不依赖 position index 的缩放。这和 PI/YaRN 不同——PI/YaRN 根据 scale factor 缩放频率，EVQ 直接给出最优频率分配。

### Hybrid EVQ 频率计算方式（A7 专用）

**设计动机**：Phase 7F 中 EVQ passkey 落后 Geometric 的核心原因是 EVQ 修改了**全部 32 个通道**的频率，包括高频端。而 passkey retrieval 靠的是高频通道做精确位置区分——预训练时 Q/K 和高频 geometric 频率形成的对齐被破坏了。

**Hybrid 方案**：高频通道保持 Geometric 不变（保护 Q/K 对齐），只对低频通道用 EVQ-cosh（增强外推能力）。

```python
def hybrid_evq_inv_freq(dim, base, tau, n_geometric_high=8):
    """
    Hybrid EVQ: 高频通道保持 Geometric，低频通道用 EVQ-cosh。

    Args:
        dim: head_dim (64)
        base: RoPE base (500000)
        tau: cosh 参数，只作用于低频通道
        n_geometric_high: 保持 Geometric 的高频通道数（默认 8，即前 25%）

    Returns:
        inv_freq: shape (dim//2,) = (32,) 的频率向量
    """
    n = dim // 2  # 32

    # 标准 geometric 频率（全部 32 个）
    geo_inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))

    # 高频部分（前 n_geometric_high 个通道）：直接用 geometric
    # 低频部分（后 n - n_geometric_high 个通道）：用 EVQ-cosh 在剩余频率范围内重新分配

    n_evq = n - n_geometric_high  # 24 个低频通道

    # 低频通道的频率范围
    theta_max_low = geo_inv_freq[n_geometric_high].item()  # 第 8 个通道的频率（分界点）
    theta_min_low = geo_inv_freq[-1].item()  # 最低频

    # 在 [theta_min_low, theta_max_low] 区间做 cosh 分配
    u = torch.arange(n_evq, dtype=torch.float64) / (n_evq - 1)  # [0, 1]

    if abs(tau) < 1e-8:
        phi = 1.0 - u
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))

    evq_part = (theta_min_low ** phi) * (theta_max_low ** (1.0 - phi))

    # 拼接
    inv_freq = torch.cat([geo_inv_freq[:n_geometric_high].double(), evq_part])
    return inv_freq.float()
```

**为什么 n_geometric_high=8（25%）**：
- 前 8 个通道是最高频的，周期 < 100 tokens，主要负责 local position discrimination
- 后 24 个通道是中低频，周期 100~1.9M tokens，负责 long-range structure
- Passkey 主要依赖高频通道的精确性，EVQ 的优势主要在低频端的重新分配
- 这个分界和 YaRN 的 high/low boundary 思路类似，但低频端用 cosh 替代线性插值

### 评估

```python
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
# 512 = 原始预训练窗口（应该不退化）
# 4096 = 续训窗口（应该最好）
# 8192 = 2x 外推（关键战场）
# 16384 = 4x 外推（EVQ 应该反超的地方）
```

### Passkey 评估

```python
PASSKEY_LENGTHS = [1024, 2048, 4096, 8192]
# 1K/2K = 训练窗口内
# 4K = 续训窗口边界
# 8K = 外推
TRIALS_PER_LENGTH = 100
PASSKEY_POSITION = 0.5  # 放在中间位置
```

### 关键验证

1. **Geometric 是否在 8K-16K 开始崩？** 如果 PPL@16K 显著差于 EVQ → 论文核心 story 成立
2. **EVQ τ=2.0 或 τ=2.5 是否在大扩展比下反超 τ=1.5？** 在 from-scratch 中 τ 越大越好（128-tok），在 extension 中 τ=1.5 最安全，但 8x 扩展比可能改变平衡点
3. **PI/YaRN 是否比 4x 时更好或更差？** s=8 时 YaRN 的压缩更激进
4. **Passkey 是否因为续训量翻倍（10M vs 7F 的 5M）而改善？**

---

## 实验 8B: 续训量消融（Passkey 恢复测试）

**目的**: 验证 "5M token 续训不够恢复 Q/K 对齐" 的假说
**优先级**: ★★★（解释 passkey 失败的关键实验）
**预计时间**: ~1.5h

### 设计

复用 7F 的 512→2K 设置（4x 扩展比），只变续训 token 数。重点关注 EVQ τ=2.0（7F 中 passkey 最好的 EVQ 配置）。

| Run ID | Method | 续训量 | 占预训练比例 | 预计时间 |
|--------|--------|--------|------------|---------|
| B1 | EVQ τ=2.0 | 2.5M | 5% | ~8 min |
| B2 | EVQ τ=2.0 | 5M | 10%（= 7F 已有） | 不需要跑 |
| B3 | EVQ τ=2.0 | 10M | 20% | ~15 min |
| B4 | EVQ τ=2.0 | 20M | 40% | ~30 min |
| B5 | Geometric | 10M | 20%（对照） | ~15 min |
| B6 | Geometric | 20M | 40%（对照） | ~30 min |

**配置同 7F**:
```python
TRAIN_SEQ_LEN = 2048
BATCH_SIZE = 4
LR = 3e-5
# 只变 TRAIN_TOKENS
```

### 评估

每个 run 都做完整的 PPL + Passkey 评估：

```python
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192]
PASSKEY_LENGTHS = [1024, 2048, 4096]
TRIALS_PER_LENGTH = 100
```

### 关键验证

1. **画出 "续训量 vs Passkey@2K" 曲线**：
   - 如果 EVQ passkey 随续训量单调上升 → Q/K alignment recovery 确认
   - 如果存在交叉点（EVQ 在某个续训量后追上 Geometric）→ 论文可以写 "EVQ 需要约 X% 续训量恢复 Q/K 对齐"
   - 如果始终不交叉但差距缩小 → 也是有价值的发现，说明 alignment cost 是可量化的
2. **Geometric 续训更多是否也改善？** 如果 Geometric 已经饱和而 EVQ 还在上升 → 非常好的 story

---

## 实验 8C: From-Scratch 4K 对照组

**目的**: 对比 "context extension 到 4K" vs "从头训 4K"，量化 extension 的 alignment cost
**优先级**: ★★（消融）
**预计时间**: ~2h

### 设计

从零训练 350M 在 4K seq_len 上：

```python
MODEL_CONFIG = dict(hidden=1024, layers=24, heads=16, head_dim=64)  # 350M
TRAIN_SEQ_LEN = 4096
TRAIN_TOKENS = 50_000_000  # 50M tokens（与 512-tok 预训练相同量）
ROPE_BASE = 500_000
BATCH_SIZE = 2
LR = 6e-4  # 从头训的 lr
DATASET = "HuggingFaceFW/fineweb-edu-score-2"
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
```

| Run ID | Method | τ | 预计时间 |
|--------|--------|---|---------|
| C1 | Geometric | 0.0 | ~2h |
| C2 | EVQ τ=2.0 | 2.0 | ~2h |

**显存估算**：350M + 4K + batch=2 ≈ 12-14GB，5090 安全。

**注意**：这个实验时间最长（~2h per run），可以和 8A 的 eval 阶段并行，或者放在最后跑。如果时间不够，**8C 优先级最低，可以不跑**。

### 关键验证

对比 8A（extension 到 4K）和 8C（从头训 4K）：
- 如果 from-scratch EVQ >> extension EVQ on passkey → alignment cost 很大
- 如果差距不大 → alignment cost 在足够续训后可忽略

---

## 优先级排序

| 优先级 | 实验 | 时间 | 核心问题 |
|--------|------|------|---------|
| ★★★ P0 | 8A: 512→4K (8x 扩展比, 含 Hybrid EVQ) | ~2.3h | Geometric 崩不崩？EVQ/Hybrid 能否反超？ |
| ★★★ P0 | 8B: 续训量消融 | ~1.5h | Passkey 能否随续训量恢复？ |
| ★★ P1 | 8C: From-scratch 4K 对照 | ~2h | Alignment cost 有多大？ |

**推荐执行顺序**:

1. **先跑 8A 全部 7 个 run**（A1-A7，含 Hybrid EVQ）
2. 8A 跑完后看结果：
   - 如果 Hybrid EVQ passkey 明显好于纯 EVQ → Hybrid 路线确认，8B 可以加 Hybrid 消融
   - 如果 8A 中纯 EVQ passkey 已经因为续训量翻倍而改善 → 8B 优先级降低
3. 跑 8B（续训量消融）
4. 8C 最后跑（如果时间够的话）

**总 GPU 时间**:
- 最小方案（只跑 8A）: ~2.3h
- 推荐方案（8A + 8B）: ~3.8h
- 完整方案（8A + 8B + 8C）: ~5.8h

---

## 代码要求

### 目录结构

```
/root/autodl-tmp/evq_phase8/
├── ext_4k/                      # 8A: 512→4K extension
│   ├── extend_geo/
│   ├── extend_pi/
│   ├── extend_yarn/
│   ├── extend_evq_1.5/
│   ├── extend_evq_2.0/
│   ├── extend_evq_2.5/
│   ├── extend_hybrid_2.0/
│   └── passkey_eval/
├── finetune_ablation/           # 8B: 续训量消融
│   ├── evq2.0_2.5M/
│   ├── evq2.0_10M/
│   ├── evq2.0_20M/
│   ├── geo_10M/
│   └── geo_20M/
├── from_scratch_4k/             # 8C: 从头训 4K
│   ├── geo_4k/
│   └── evq2.0_4k/
├── results_phase8.json
└── phase8_report.md
```

### 汇总 JSON 格式

```json
{
  "phase": 8,
  "date": "2026-03-XX",
  "hardware": "RTX 5090 32GB",
  "experiments": {
    "8A_ext_4k": {
      "expansion_ratio": "8x (512->4K)",
      "continuation_tokens": "10M",
      "methods": {
        "geometric": {
          "ppl": {"512": null, "1024": null, "2048": null, "4096": null, "8192": null, "16384": null},
          "passkey": {"1024": null, "2048": null, "4096": null, "8192": null}
        },
        "pi": {"ppl": {}, "passkey": {}},
        "yarn": {"ppl": {}, "passkey": {}},
        "evq_1.5": {"ppl": {}, "passkey": {}},
        "evq_2.0": {"ppl": {}, "passkey": {}},
        "evq_2.5": {"ppl": {}, "passkey": {}},
        "hybrid_evq_2.0": {"ppl": {}, "passkey": {}}
      }
    },
    "8B_finetune_ablation": {
      "expansion_ratio": "4x (512->2K)",
      "methods": {
        "evq_2.0": {
          "2.5M": {"ppl": {}, "passkey": {}},
          "5M": "reference to 7F data",
          "10M": {"ppl": {}, "passkey": {}},
          "20M": {"ppl": {}, "passkey": {}}
        },
        "geometric": {
          "5M": "reference to 7F data",
          "10M": {"ppl": {}, "passkey": {}},
          "20M": {"ppl": {}, "passkey": {}}
        }
      }
    },
    "8C_from_scratch_4k": {
      "methods": {
        "geometric": {"ppl": {}, "passkey": {}},
        "evq_2.0": {"ppl": {}, "passkey": {}}
      }
    }
  }
}
```

### 结果输出格式

phase8_report.md 必须包含：

1. **8A 结果表**：7 个方法 × 6 个 eval 长度的 PPL + passkey（含 Hybrid EVQ）
2. **8A 关键图**：PPL@eval_length 曲线（7 条线），标注训练窗口和外推边界
2.5. **Hybrid vs Pure EVQ 对比**：重点对比 A5 (EVQ τ=2.0) vs A7 (Hybrid τ=2.0) 在 passkey 上的差异
3. **7F vs 8A 对比**：4x 扩展比 vs 8x 扩展比下各方法的排名变化
4. **8B 续训量曲线**：续训 token 数 vs passkey accuracy + PPL
5. **8C 对比**（如果跑了）：extension vs from-scratch 在 PPL 和 passkey 上的差距

---

## 注意事项

1. **复用 7F 的 pretrain checkpoint**: 所有 extension 实验共享同一个 512-tok pretrain checkpoint，不要重新预训练
2. **8A 的 YaRN 参数**: `original_max_position=512`, `target_max_position=4096`, `s=8`。注意和 7F 不同（7F 是 s=4）
3. **8A 的 PI 参数**: 所有频率 ÷ 8
4. **8B 需要 7F 的 5M 数据作为参照**: 直接引用 7F 结果即可，不需要重跑
5. **8A 的 batch_size=2**: 如果 OOM，降到 1 并开 gradient accumulation 2 步
6. **EVQ τ=2.5 在 extension 中是新的**: 之前只试过 1.5 和 2.0，8x 扩展比下 τ=2.5 可能更合适（因为外推区间更长，PE-dominant 更强）
7. **Passkey 实现**: 完全复用 7F 的 passkey 评估代码，不要修改
8. **每跑完一个 run 立刻保存 JSON**: 防止中途断连
9. **如果 8A 中 Geometric PPL@16K 没有崩** → 说明 350M 32 通道在 16K 还够用，可能需要 eval 到 32K 或用更大扩展比。在 report 中说明
10. **8C 如果时间不够可以跳过**: 8A 和 8B 是核心，8C 是锦上添花
11. **Hybrid EVQ (A7) 的 n_geometric_high=8**: 前 8 个通道（k=0..7）直接用 geometric inv_freq，后 24 个通道用 EVQ-cosh τ=2.0 在剩余频率范围内重新分配。文档中有完整的 `hybrid_evq_inv_freq` 函数，直接复用即可
12. **Hybrid 是独立的频率方案**: 不要把 Hybrid 理解为 YaRN+EVQ 的组合。Hybrid 的高频端 = Geometric（完全不变），低频端 = EVQ-cosh（完全替换）。没有中间的线性插值过渡区

---

## 实验 8D: τ* Scaling Law 验证（8A-8C 完成后补做）

**目的**: 验证 τ*(L_train) = d_head / √L_train = 64/√L 这一 scaling law 猜想
**优先级**: ★★（理论验证，8A-8C 之后再跑）
**预计时间**: ~1.5h
**依据**: Gemini 3.1 Pro 推导的变分论证 + 3 个已有数据点拟合（L=128→τ*>5.0, L=1024→τ*≈2.0, L=2048→τ*≈1.5）

### 背景

已有数据点（全部 from-scratch 125M/350M，head_dim=64）：

| L_train | 预测 τ* = 64/√L | 实测 τ* | 来源 |
|---------|-----------------|--------|------|
| 128 | 5.66 | >5.0（单调下降无peak） | Phase 6 |
| 1024 | 2.0 | ≈2.0 | Phase 6 1024-tok |
| 2048 | 1.414 | ≈1.5 | Phase 7F ext |

需要两个新验证点：L=256 和 L=512。

### 设计

两组 from-scratch 训练，每组做 τ sweep 找最优 τ*。模型用 125M（和 Phase 6 一致，保证可比性）。

```python
MODEL_CONFIG = dict(hidden=768, layers=12, heads=12, head_dim=64)  # 125M
ROPE_BASE = 500_000
DATASET = "HuggingFaceFW/fineweb-edu-score-2"
LR = 6e-4
BATCH_SIZE = 8  # L=256/512 比较短，可以用大 batch
```

**Run D1: L_train=256**

```python
TRAIN_SEQ_LEN = 256
TRAIN_TOKENS = 50_000_000  # 50M tokens
EVAL_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]
```

| Run ID | τ | 预测 | 预计时间 |
|--------|---|------|---------|
| D1a | 0.0 | Geometric baseline | ~8 min |
| D1b | 2.0 | 低于预测 | ~8 min |
| D1c | 3.0 | 接近但偏低 | ~8 min |
| D1d | 4.0 | **预测最优点** τ*=64/√256=4.0 | ~8 min |
| D1e | 5.0 | 偏高 | ~8 min |

**Run D2: L_train=512**

```python
TRAIN_SEQ_LEN = 512
TRAIN_TOKENS = 50_000_000  # 50M tokens
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192]
```

| Run ID | τ | 预测 | 预计时间 |
|--------|---|------|---------|
| D2a | 0.0 | Geometric baseline | ~12 min |
| D2b | 1.5 | 偏低 | ~12 min |
| D2c | 2.0 | 接近但偏低 | ~12 min |
| D2d | 2.83 | **预测最优点** τ*=64/√512≈2.83 | ~12 min |
| D2e | 3.5 | 偏高 | ~12 min |
| D2f | 4.0 | 明显偏高 | ~12 min |

### 评估指标

**核心指标**: PPL@(8×L_train)，即 D1 看 PPL@2048，D2 看 PPL@4096。因为 τ* 优化的是外推性能。

**辅助指标**: PPL@L_train（检查 waterbed），PPL@(2×L_train)，PPL@(4×L_train)

### 关键验证

1. **D1**: PPL@2048 的最优 τ 是否在 3.5-4.5 范围内（预测 4.0）？
2. **D2**: PPL@4096 的最优 τ 是否在 2.5-3.2 范围内（预测 2.83）？
3. **两组结合已有 5 个数据点**，做 τ* vs 1/√L 的线性回归。如果 R² > 0.95 → scaling law 强支撑
4. **PPL@L_train 是否对 τ 不敏感**（waterbed check）？Phase 6/7 已确认，但新 L 值需要复查

### 汇总 JSON（追加到 results_phase8.json）

```json
{
  "8D_scaling_law_verification": {
    "purpose": "Verify tau*(L) = 64/sqrt(L) conjecture",
    "model": "125M (head_dim=64)",
    "D1_L256": {
      "predicted_tau_star": 4.0,
      "results": {
        "tau_0.0": {"ppl_256": null, "ppl_2048": null, "ppl_8192": null},
        "tau_2.0": {"ppl_256": null, "ppl_2048": null, "ppl_8192": null},
        "tau_3.0": {"ppl_256": null, "ppl_2048": null, "ppl_8192": null},
        "tau_4.0": {"ppl_256": null, "ppl_2048": null, "ppl_8192": null},
        "tau_5.0": {"ppl_256": null, "ppl_2048": null, "ppl_8192": null}
      },
      "observed_tau_star": null,
      "error_vs_prediction": null
    },
    "D2_L512": {
      "predicted_tau_star": 2.83,
      "results": {
        "tau_0.0": {"ppl_512": null, "ppl_4096": null, "ppl_8192": null},
        "tau_1.5": {"ppl_512": null, "ppl_4096": null, "ppl_8192": null},
        "tau_2.0": {"ppl_512": null, "ppl_4096": null, "ppl_8192": null},
        "tau_2.83": {"ppl_512": null, "ppl_4096": null, "ppl_8192": null},
        "tau_3.5": {"ppl_512": null, "ppl_4096": null, "ppl_8192": null},
        "tau_4.0": {"ppl_512": null, "ppl_4096": null, "ppl_8192": null}
      },
      "observed_tau_star": null,
      "error_vs_prediction": null
    },
    "scaling_law_fit": {
      "data_points": [
        {"L": 128, "predicted": 5.66, "observed": ">5.0"},
        {"L": 256, "predicted": 4.0, "observed": null},
        {"L": 512, "predicted": 2.83, "observed": null},
        {"L": 1024, "predicted": 2.0, "observed": "~2.0"},
        {"L": 2048, "predicted": 1.414, "observed": "~1.5"}
      ],
      "linear_fit_R2": null,
      "note": "Fit tau* vs 1/sqrt(L), expect slope ~64"
    }
  }
}
```

### 目录结构（追加）

```
/root/autodl-tmp/evq_phase8/
├── ...（8A/8B/8C 已有）
└── scaling_law/                    # 8D
    ├── L256_tau0.0/
    ├── L256_tau2.0/
    ├── L256_tau3.0/
    ├── L256_tau4.0/
    ├── L256_tau5.0/
    ├── L512_tau0.0/
    ├── L512_tau1.5/
    ├── L512_tau2.0/
    ├── L512_tau2.83/
    ├── L512_tau3.5/
    └── L512_tau4.0/
```

### 注意事项

1. **用 125M 模型**（不是 350M）：和 Phase 6 保持一致，且 scaling law 预测 τ* 与模型大小无关，用小模型跑更快
2. **50M tokens 和 Phase 6 一致**：控制变量，只变 L_train
3. **τ=2.83 可以用 math.sqrt(8) 精确计算**: `tau = 64.0 / math.sqrt(512)`
4. **如果 D1 的最优 τ 不是 4.0 但在 3.5-4.5 之间**，仍然算 scaling law 成立（±15% 误差在 3 个已有数据点的范围内）
5. **report 中画 τ* vs L 的 log-log 图**：如果 scaling law 成立，log(τ*) = log(64) - 0.5·log(L) 应该是一条斜率 -0.5 的直线

---

## 实验 8E: From-Scratch 4K τ=1.0 验证（8D 完成后自动执行）

**目的**: Phase 8C 用 τ=2.0，passkey 输 Geo 3pp。Scaling law 预测 τ*(4096)=1.0。本实验验证用正确的 τ 是否能追平或反超 Geo passkey。
**优先级**: ★★★（Passkey 是论文生死线）
**预计时间**: ~25 min（1 个 run）+ ~25 min（Hybrid，如果时间够）
**前置**: 8D 完成后立即执行

### 设计

复用 8C 完全相同的配置，只改 τ：

```python
MODEL_CONFIG = dict(hidden=1024, layers=24, heads=16, head_dim=64)  # 350M
TRAIN_SEQ_LEN = 4096
TRAIN_TOKENS = 50_000_000  # 50M tokens
ROPE_BASE = 500_000
BATCH_SIZE = 2
LR = 6e-4  # from-scratch lr
DATASET = "HuggingFaceFW/fineweb-edu-score-2"
```

| Run ID | Method | τ | 目录 | 预计时间 |
|--------|--------|---|------|---------|
| E1 | EVQ τ=1.0 | 1.0 | `from_scratch_4k/evq1.0_4k/` | ~25 min |
| E2 | Hybrid τ=1.0 | 1.0 (n_geometric_high=8) | `from_scratch_4k/hybrid1.0_4k/` | ~25 min |

### 评估

和 8C 完全一致：

```python
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
# Passkey 也和 C1/C2 相同设置
PASSKEY_LENGTHS = [1024, 2048, 4096, 8192]
TRIALS_PER_LENGTH = 100
```

### 关键对比

| Run | Method | τ | 期望 PPL@16K | 期望 Passkey |
|-----|--------|---|-------------|-------------|
| C1 | Geo (已有) | — | 175.4 | 69% |
| C2 | EVQ τ=2.0 (已有) | 2.0 | 164.4 | 66% |
| **E1** | **EVQ τ=1.0** | **1.0** | **< 175?** | **≥ 69%?** |
| E2 | Hybrid τ=1.0 | 1.0 | — | bonus |

**成功标准**: E1 passkey ≥ C1 Geo passkey (69%) → 证明 τ=1.0 是正确选择，EVQ passkey 问题完全解决

### 注意事项

1. **从零训练**，不是 extension，不需要 pretrain checkpoint
2. **和 C1/C2 用完全相同的代码和 eval 设置**，只改频率生成函数的 τ 参数
3. **E2 (Hybrid) 优先级低于 E1**：如果时间不够只跑 E1
4. 结果追加到 `results_phase8.json`
5. **如果 8D 的实测 τ*(4096) ≠ 1.0**：用 8D 的实测值替换 E1 的 τ
