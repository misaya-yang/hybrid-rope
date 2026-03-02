# Phase 8H: Base=10K 系统化 τ 扫描（50M tokens，定版实验）

> **目标**: Phase 8G 初步结果显示 τ=1.1/1.2 在 50M tokens base=10K 下完败 Geometric。本实验系统化扫描 τ∈[0.2, 1.0]，找到 base=10K 下的真正 τ*，验证/证伪双变量 Scaling Law。
> **硬件**: RTX 5090 32GB
> **预计 GPU 时间**: ~7h（Phase A ~4h + Phase B ~3h）
> **优先级**: 🔴🔴🔴 论文核心数据——决定 Scaling Law 的最终形式
> **前置**: Phase 8G 的 geo_4k（50M）已完成，可复用

---

## 为什么需要这个实验

### 8G 已有数据（50M tokens, base=10K, seed=42）

| Method | τ | retrieval | PPL@16K | vs Geo |
|--------|---|-----------|---------|--------|
| **Geometric** | — | **0.680** | **274.2** | baseline |
| EVQ | 1.1 | 0.568 | 282.4 | ret -16.5%, PPL +3.0% |
| EVQ | 1.2 | 0.568 | 317.0 | ret -16.5%, PPL +15.6% |

**结论**: τ=1.1-1.2 大幅失败。但 10M 粗筛（欠拟合条件下）显示 τ=0.2-0.7 区间有优势。需要在 50M 下系统扫描低 τ 区间。

### 两套理论的对决

| 理论来源 | 预测 τ*(10K, L=4096) | 基于 |
|----------|---------------------|------|
| **简单公式** (Q6+Phase 8D) | 1.19 | τ* = d_head/√L · √(lnb_ref/lnb) |
| **Gemini 严格推导** | 0.68 | α* ∝ b/(L·(lnb)²) + Galerkin投影 |
| **8G 实验已排除** | ≠1.1, ≠1.2 | τ=1.1/1.2 均完败 |

**本实验将决定采用哪套理论。**

---

## 实验设计

### 固定配置（与 8G 完全一致）

```python
MODEL_CONFIG = dict(hidden=1024, layers=24, heads=16, head_dim=64)  # 350M
TRAIN_SEQ_LEN = 4096
TRAIN_TOKENS = 50_000_000  # 50M tokens
ROPE_BASE = 10_000
BATCH_SIZE = 2
LR = 6e-4
DATASET = "HuggingFaceFW/fineweb-edu-score-2"
SEED = 42
```

### Phase A: EVQ 粗扫（7 个 τ 值）

**目标**: 覆盖 τ∈[0.2, 1.0]，定位最优区间。

| Run ID | Method | τ | 目录 | 预计时间 | 备注 |
|--------|--------|---|------|---------|------|
| H-A0 | Geo | — | `base10k_8h/geo_4k/` | — | **复用 8G 已有结果** |
| H-A1 | EVQ | 0.2 | `base10k_8h/evq0.2_4k/` | ~25 min | 10M 下 retrieval 最优 |
| H-A2 | EVQ | 0.4 | `base10k_8h/evq0.4_4k/` | ~25 min | |
| H-A3 | EVQ | 0.5 | `base10k_8h/evq0.5_4k/` | ~25 min | |
| H-A4 | EVQ | 0.6 | `base10k_8h/evq0.6_4k/` | ~25 min | |
| H-A5 | EVQ | 0.7 | `base10k_8h/evq0.7_4k/` | ~25 min | Gemini 预测区间 |
| H-A6 | EVQ | 0.8 | `base10k_8h/evq0.8_4k/` | ~25 min | |
| H-A7 | EVQ | 1.0 | `base10k_8h/evq1.0_4k/` | ~25 min | 旧理论预测 |

**Phase A 总计**: 7 个新训练 × ~25 min = **~3h**（可并行跑 2 个以减半时间）

### Phase B: 精细扫 + Hybrid（基于 Phase A 结果）

**在 Phase A 找到最优 τ_peak 后**，在 τ_peak ± 0.15 范围内做精细扫 + Hybrid 对比。

**⚠️ 重要**: Hybrid 的 split point r 必须针对 base=10K 调整！

```python
# base=10K 下的理论最优 split
# r* = (d_head / (2·lnb)) · ln(L_train / 2π)
# r* = (64 / (2·9.21)) · ln(4096 / 6.28) = 3.474 · 6.48 = 22.5
# → r = 22 或 r = 23（而非 base=500K 下的 r=16！）

HYBRID_R_OPTIONS = [16, 22, 23]  # 测试 r=16（旧）和 r=22/23（理论新）
```

| Run ID | Method | τ | r | 目录 | 备注 |
|--------|--------|---|---|------|------|
| H-B1 | EVQ | τ_peak-0.1 | — | 自定 | 精细扫 |
| H-B2 | EVQ | τ_peak+0.1 | — | 自定 | 精细扫 |
| H-B3 | EVQ | τ_peak-0.05 | — | 自定 | 最精细 |
| H-B4 | EVQ | τ_peak+0.05 | — | 自定 | 最精细 |
| H-B5 | Hybrid | τ_peak | 16 | 自定 | 旧 split 对比 |
| H-B6 | Hybrid | τ_peak | 22 | 自定 | 理论新 split |
| H-B7 | Hybrid | τ_peak | 23 | 自定 | 理论新 split |

**Phase B 总计**: 7 个训练 × ~25 min = **~3h**

---

## EVQ 频率生成代码

```python
import torch
import math

def evq_cosh_inv_freq(d_head: int, tau: float, base: float) -> torch.Tensor:
    """EVQ-cosh frequency allocation with explicit τ."""
    K = d_head // 2
    u = torch.linspace(0.5/K, 1 - 0.5/K, K)
    phi = 1 - (1/tau) * torch.arcsinh((1 - u) * math.sinh(tau))
    return base ** (-phi)

def hybrid_evq_inv_freq(d_head: int, tau: float, base: float, r: int) -> torch.Tensor:
    """Hybrid: first r channels geometric, rest EVQ."""
    K = d_head // 2
    inv_freq = torch.zeros(K)
    # Geometric channels (0 to r-1)
    for i in range(r):
        inv_freq[i] = 1.0 / (base ** (2*i / d_head))
    # EVQ channels (r to K-1)
    K_evq = K - r
    u = torch.linspace(0.5/K_evq, 1 - 0.5/K_evq, K_evq)
    phi = 1 - (1/tau) * torch.arcsinh((1 - u) * math.sinh(tau))
    # Map to the EVQ portion of the frequency range
    phi_start = 2 * r / d_head  # φ at the split point
    phi_evq = phi_start + phi * (1 - phi_start)  # scale to [phi_start, 1]
    inv_freq[r:] = base ** (-phi_evq)
    return inv_freq

# 示例：base=10K, τ=0.7, d_head=64
inv_freq_evq = evq_cosh_inv_freq(64, 0.7, 10000.0)
inv_freq_hybrid = hybrid_evq_inv_freq(64, 0.7, 10000.0, r=22)
```

---

## 评估

```python
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
PASSKEY_LENGTHS = [1024, 2048, 4096, 8192]
TRIALS_PER_LENGTH = 100  # 4 长度 × 100 = 400 trials total
```

**评估指标**:
1. `retrieval`: passkey retrieval 全局平均（primary metric）
2. `PPL@16K`: perplexity at 16K context（secondary metric）
3. `mean_nll_gap`: 外推 NLL 增长率

---

## 判定标准

### Phase A 判定

1. **如果存在 τ 使 EVQ retrieval > Geo 0.68**:
   - 记录 τ_peak = argmax retrieval
   - 进入 Phase B 精细扫
   - **论文叙事**: "EVQ at τ* dominates Geometric at base=10K"

2. **如果所有 τ 的 retrieval 都 < Geo 0.68，但 PPL@16K 有显著赢**:
   - 记录 τ_ppl = argmin PPL@16K
   - 进入 Phase B 但 primary metric 改为 PPL
   - **论文叙事**: "EVQ trades retrieval for PPL improvement at base=10K"

3. **如果所有 τ 在 retrieval 和 PPL 上都不赢 Geo**:
   - **停止实验**，不进入 Phase B
   - **论文叙事需要重构**: 可能 base=10K 下模型行为和 base=500K 质的不同
   - 考虑 base=100K 作为替代验证点

### Scaling Law 判定

| 实测 τ_peak | 结论 |
|-------------|------|
| 0.6-0.8 | Gemini 严格推导正确 (τ*≈0.68)，简单公式需修正 |
| 0.9-1.3 | 简单公式正确 (τ*≈1.19)，Gemini 的 α* 投影有偏 |
| 0.2-0.5 | 两套理论都不对，可能存在未建模的 base=10K 特殊物理 |

---

## Phase A 的 Decision Tree（给执行者看）

```
Phase A 完成后：
│
├─ 找到 τ_peak（EVQ 赢 Geo）？
│   ├─ YES → 进入 Phase B，τ_peak 为中心扫描
│   │         同时用 r=16/22/23 测 Hybrid
│   │
│   └─ NO → 检查 PPL 是否有赢
│       ├─ PPL 有赢 → Phase B 只做 PPL 最优 τ 附近的精细扫
│       └─ 全面不赢 → 停止。改跑 base=100K 实验
│
Phase B 完成后：
│
├─ Hybrid r=22/23 > Hybrid r=16 ?
│   ├─ YES → r* 公式验证通过，论文又一个 prediction→experiment 闭环
│   └─ NO → r* 公式需要修正（可能 base=10K 下 Hybrid 本身不适用）
│
├─ 记录最终结果，更新 CORE_THEORY
└─ 设计 multi-seed 验证（Phase 8I）如果效应显著
```

---

## 目录结构

```
/root/autodl-tmp/evq_phase8/
├── ...（已有 8A-8G）
└── base10k_8h/                    # 8H
    ├── geo_4k/                    # 复用 8G
    ├── evq0.2_4k/                 # Phase A
    ├── evq0.4_4k/
    ├── evq0.5_4k/
    ├── evq0.6_4k/
    ├── evq0.7_4k/
    ├── evq0.8_4k/
    ├── evq1.0_4k/
    ├── evq{τ_peak±0.1}_4k/       # Phase B (精细扫)
    ├── hybrid{τ_peak}_r16_4k/    # Phase B (Hybrid 对比)
    ├── hybrid{τ_peak}_r22_4k/
    └── hybrid{τ_peak}_r23_4k/
```

## 汇总 JSON 格式

每个 run 完成后自动追加到 `results_8h_base10k_tau_sweep.json`:

```json
{
  "phase": "8H",
  "purpose": "Base=10K systematic tau sweep (50M tokens)",
  "config": {
    "model": "350M, from-scratch 4K",
    "tokens": 50000000,
    "rope_base": 10000,
    "seed": 42
  },
  "results": {
    "geo_4k": {
      "tau": null,
      "retrieval": 0.680,
      "mean_nll_gap": 0.1268,
      "ppl_16k": 274.246,
      "source": "reused from 8G"
    },
    "evq0.2_4k": { "tau": 0.2, "retrieval": null, "ppl_16k": null },
    "evq0.4_4k": { "tau": 0.4, "retrieval": null, "ppl_16k": null },
    "...": "..."
  },
  "analysis": {
    "tau_peak_retrieval": null,
    "tau_peak_ppl": null,
    "best_vs_geo_retrieval": null,
    "best_vs_geo_ppl": null,
    "scaling_law_verdict": null
  }
}
```

---

## ⚠️ 注意事项

1. **只改 τ 和 method，其他全部不动**: base=10000, seed=42, 50M tokens, 350M 模型
2. **Geo 不需要重跑**: 8G 的 geo_4k 结果直接复用
3. **τ=1.1 和 τ=1.2 不需要重跑**: 8G 已有数据
4. **Hybrid 的 r 值很关键**: base=10K 下 r*=22.5，必须用 r=22 或 23，r=16 只作为 ablation
5. **Phase B 的 τ 值需要人工决定**: Phase A 完成后看数据决定 τ_peak，然后手动设定 Phase B 的精确 τ 值
6. **如果 Phase A 完全没有 EVQ 赢 Geo 的情况**: 不要进入 Phase B，直接汇报结果
7. **评估一定包含 PPL@16K 和 passkey retrieval 两个指标**: 不能只看一个
