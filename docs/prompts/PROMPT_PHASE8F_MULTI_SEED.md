# Phase 8F: Multi-Seed 统计验证（8E 核心结果确认）

> **目标**: Phase 8E 的 headline 结果（EVQ τ=1.0 passkey 72% > Geo 69%，Hybrid 双赢）margin 只有 1.5-3pp，在 400 trials 下约 1σ。必须用 multi-seed 确认这不是噪音。
> **硬件**: RTX 5090 32GB
> **预计 GPU 时间**: ~4-5h（12 个 runs × ~25 min）
> **前置**: Phase 8E 完成
> **重要性**: ★★★ 论文生死线——如果 multi-seed 翻车，headline 不成立

---

## 统计分析：为什么需要 multi-seed

Phase 8E 单 seed 结果（400 trials passkey）：

| Method | PK Global | 95% CI (Wilson) | vs Geo |
|--------|-----------|-----------------|--------|
| Geo | 69.0% | [64.3%, 73.4%] | — |
| EVQ τ=1.0 | 72.0% | [67.4%, 76.3%] | +3pp |
| Hybrid τ=1.0 | 70.5% | [65.8%, 74.8%] | +1.5pp |

**CI 严重重叠**。单 seed 结论不可靠。3 seeds × 400 trials = 1200 trials 后，CI 宽度缩小到 ~±2.5pp，3pp 差异可达 ~2σ。

---

## 实验设计

### 核心验证：From-scratch 4K（复现 8C/8E）

每个 seed 完全从零训练 350M@4K，50M tokens，然后做完整 PPL + Passkey eval。

```python
MODEL_CONFIG = dict(hidden=1024, layers=24, heads=16, head_dim=64)  # 350M
TRAIN_SEQ_LEN = 4096
TRAIN_TOKENS = 50_000_000
ROPE_BASE = 500_000
BATCH_SIZE = 2
LR = 6e-4
DATASET = "HuggingFaceFW/fineweb-edu-score-2"
SEEDS = [42, 137, 256, 314]  # 4 seeds（含 8E 原始 seed 作为参照）
```

### 要跑的 runs

**每个 seed 跑 3 个配置**（Geo / EVQ τ=1.0 / Hybrid τ=1.0）：

| Seed | Run ID | Method | τ | 预计时间 |
|------|--------|--------|---|---------|
| 42 | F1a | Geometric | — | ~25 min |
| 42 | F1b | EVQ τ=1.0 | 1.0 | ~25 min |
| 42 | F1c | Hybrid τ=1.0 | 1.0 | ~25 min |
| 137 | F2a | Geometric | — | ~25 min |
| 137 | F2b | EVQ τ=1.0 | 1.0 | ~25 min |
| 137 | F2c | Hybrid τ=1.0 | 1.0 | ~25 min |
| 256 | F3a | Geometric | — | ~25 min |
| 256 | F3b | EVQ τ=1.0 | 1.0 | ~25 min |
| 256 | F3c | Hybrid τ=1.0 | 1.0 | ~25 min |
| 314 | F4a | Geometric | — | ~25 min |
| 314 | F4b | EVQ τ=1.0 | 1.0 | ~25 min |
| 314 | F4c | Hybrid τ=1.0 | 1.0 | ~25 min |

**总计**: 12 runs × ~25 min = ~5h

**注意**: 如果 8E 原始用的 seed 是上述之一（很可能是 42），那个 seed 的 3 个 run 可以直接引用 8C/8E 已有结果，不需要重跑。实际只需跑 9 个新 run（~3.75h）。

### Seed 设置

seed 同时控制：
1. 模型初始化权重
2. 数据 shuffle 顺序
3. dropout（如果有）

```python
import torch
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
```

---

## 评估

和 8C/8E 完全一致：

```python
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
PASSKEY_LENGTHS = [1024, 2048, 4096, 8192]
TRIALS_PER_LENGTH = 100  # 每 seed 400 trials
```

---

## 统计分析要求

### 1. PPL@16K 的 Mean ± Std

```
对每个方法，报告 4 seeds 的 PPL@16K mean ± std：
- Geo: mean ± std
- EVQ τ=1.0: mean ± std
- Hybrid τ=1.0: mean ± std

计算 paired difference:
- Δ(EVQ - Geo): mean ± std → t-test p-value
- Δ(Hybrid - Geo): mean ± std → t-test p-value
```

### 2. Passkey 的 Mean ± Std

```
对每个方法，报告 4 seeds 的 PK Global mean ± std：
- Geo: mean ± std
- EVQ τ=1.0: mean ± std
- Hybrid τ=1.0: mean ± std

Pooled passkey (4 seeds × 400 trials = 1600 trials):
- 用 Fisher's exact test 或 chi-squared 检验 EVQ vs Geo 的 passkey difference
- 报告 p-value 和 effect size (Cohen's h)
```

### 3. 按长度分解

```
对每个 eval length (1K/2K/4K/8K)，报告 4 seeds 的 passkey mean ± std：
重点关注 @1K 和 @8K（8E 中 Hybrid @1K=93% 是最强信号）
```

### 4. 成功标准

**强确认**:
- EVQ τ=1.0 mean PK > Geo mean PK，且 p < 0.05
- 或 Hybrid τ=1.0 mean PPL@16K < Geo mean PPL@16K 且 mean PK > Geo mean PK

**弱确认**:
- EVQ τ=1.0 mean PK > Geo mean PK（即使 p > 0.05），且 4/4 seeds 方向一致

**翻车**:
- EVQ τ=1.0 mean PK ≤ Geo mean PK
- 需要重新评估 Phase 9 策略

---

## 目录结构

```
/root/autodl-tmp/evq_phase8/
├── ...（已有 8A-8E）
└── multi_seed/                     # 8F
    ├── seed42/
    │   ├── geo_4k/
    │   ├── evq1.0_4k/
    │   └── hybrid1.0_4k/
    ├── seed137/
    │   ├── geo_4k/
    │   ├── evq1.0_4k/
    │   └── hybrid1.0_4k/
    ├── seed256/
    │   ├── geo_4k/
    │   ├── evq1.0_4k/
    │   └── hybrid1.0_4k/
    └── seed314/
        ├── geo_4k/
        ├── evq1.0_4k/
        └── hybrid1.0_4k/
```

---

## 汇总 JSON 格式

```json
{
  "8F_multi_seed_verification": {
    "purpose": "Statistical verification of 8E headline results",
    "model": "350M (head_dim=64)",
    "train_config": "from-scratch 4K, 50M tokens",
    "seeds": [42, 137, 256, 314],
    "per_seed_results": {
      "seed_42": {
        "geometric": {"ppl_16k": null, "passkey_global": null, "passkey_by_length": {}},
        "evq_1.0": {"ppl_16k": null, "passkey_global": null, "passkey_by_length": {}},
        "hybrid_1.0": {"ppl_16k": null, "passkey_global": null, "passkey_by_length": {}}
      },
      "seed_137": {},
      "seed_256": {},
      "seed_314": {}
    },
    "aggregated": {
      "geometric": {"ppl_16k_mean": null, "ppl_16k_std": null, "pk_mean": null, "pk_std": null},
      "evq_1.0": {"ppl_16k_mean": null, "ppl_16k_std": null, "pk_mean": null, "pk_std": null},
      "hybrid_1.0": {"ppl_16k_mean": null, "ppl_16k_std": null, "pk_mean": null, "pk_std": null}
    },
    "statistical_tests": {
      "evq_vs_geo_pk": {"pooled_evq": null, "pooled_geo": null, "p_value": null, "cohens_h": null},
      "hybrid_vs_geo_ppl": {"mean_diff": null, "p_value": null},
      "hybrid_vs_geo_pk": {"pooled_hybrid": null, "pooled_geo": null, "p_value": null}
    },
    "verdict": null
  }
}
```

---

## 结果输出格式

phase8_report.md 追加 8F 部分，必须包含：

1. **4 seeds × 3 methods 的完整 PPL 表**（@4K, @8K, @16K）
2. **4 seeds × 3 methods 的完整 Passkey 表**（@1K, @2K, @4K, @8K, Global）
3. **Mean ± Std 汇总表**
4. **Statistical tests 结果**（p-values, effect sizes）
5. **明确的 verdict**：强确认 / 弱确认 / 翻车

---

## 注意事项

1. **每个 seed 必须从零初始化模型**: 不能共享 checkpoint
2. **seed 必须同时控制模型初始化和数据顺序**: 确保完全独立
3. **Passkey eval 的 seed 也要固定**: 但和训练 seed 不同，使用固定的 passkey_eval_seed=999 确保评估一致性
4. **如果某个 seed 的 Geo passkey 异常高/低**（>75% 或 <60%），在 report 中标注但不剔除
5. **优先跑完所有 Geo runs，再跑 EVQ，最后 Hybrid**: 这样可以尽早看到 Geo baseline 的方差
6. **如果 8E 的原始 seed 可以复用**: 直接引用 8C/8E 数据，减少 3 个 runs
7. **每跑完一个 seed 的 3 个 runs，立刻计算该 seed 的 EVQ-Geo diff**: 可以提前判断趋势
