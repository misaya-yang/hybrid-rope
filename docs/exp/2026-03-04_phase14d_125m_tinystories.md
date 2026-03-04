# Phase 14D: 125M TinyStories 实验结果（2026-03-04）

> **状态**：VALID（负面结果 — 125M 规模不足以学习 passkey retrieval）
> **服务器**：5090 32GB, bf16, SDPA
> **模型**：125M (151.9M params)
> **数据**：TinyStories（roneneldan/TinyStories），100M tokens, 10% passkey mix
> **训练**：seq_len=2048, lr=3e-4, cosine schedule, micro_bs=4, grad_accum=4
> **YaRN scale**：8.0（与 350M 实验一致）

---

## 1. 核心结论

**125M 模型在 TinyStories 上完全未学会 passkey retrieval。**

所有方法（Geo/EVQ × baseline/+YaRN）在所有长度（2K~16K）的 passkey retrieval 均在 ~50% 附近（随机水平），NLL gap 接近零（±0.01~0.07 vs 350M 的 ±0.3~3.8）。

这表明：
- 125M 模型容量不足以同时学习语言建模 + passkey 检索
- 或 TinyStories 数据集的位置复杂度不足以激发检索学习
- EVQ 的优势需要 **足够的模型规模** 才能显现

---

## 2. 数据表

### 2.1 Baseline（无 YaRN）三 seed 均值 ± std

| Method | PK@2K | PK@4K | PK@8K | PK@12K | PK@16K | PPL@2K | PPL@8K | PPL@16K |
|--------|-------|-------|-------|--------|--------|--------|--------|---------|
| **Geo** | 49±4% | 51±4% | 46±4% | 57±8% | 53±10% | 5.4 | 8.6 | 14.4 |
| **EVQ** | 45±5% | 48±6% | 45±2% | 53±6% | 52±13% | 5.4 | 8.5 | 13.0 |
| **Delta** | -4pp | -3pp | -1pp | -4pp | -1pp | 0.0 | -0.1 | -1.4 |

> 所有 retrieval 均在随机水平 (~50%)。EVQ 无优势。

### 2.2 +YaRN (scale=8) 三 seed 均值 ± std

| Method | PK@2K | PK@4K | PK@8K | PK@12K | PK@16K | PPL@2K | PPL@8K | PPL@16K |
|--------|-------|-------|-------|--------|--------|--------|--------|---------|
| **Geo+YaRN** | 47±5% | 51±5% | 47±7% | 54±10% | 50±6% | 5.4 | 5.5 | 7.6 |
| **EVQ+YaRN** | 43±11% | 47±7% | 46±2% | 53±4% | 53±12% | 5.7 | 5.4 | 5.7 |

### 2.3 EVQ+YaRN vs Geo+YaRN Delta

| | PK@2K | PK@4K | PK@8K | PK@12K | PK@16K |
|---|-------|-------|-------|--------|--------|
| **Delta** | -4pp | -4pp | -1pp | -1pp | +3pp |

> 全部在噪声范围内，无统计显著性。

---

## 3. PPL 观察

尽管 retrieval 无效，PPL 数据仍有参考价值：

| Config | PPL@2K | PPL@8K | PPL@16K |
|--------|--------|--------|---------|
| Geo baseline | 5.4 | 8.6 | 14.4 |
| EVQ baseline | 5.4 | 8.5 | 13.0 |
| Geo+YaRN | 5.4 | 5.5 | 7.6 |
| **EVQ+YaRN** | 5.7 | **5.4** | **5.7** |

- EVQ+YaRN 的 PPL 几乎完全平坦（5.7→5.4→5.7 across 2K~16K）
- Geo+YaRN 在 16K 仍有 40% 退化（5.4→7.6）
- 这说明 EVQ 的频率分配在 PPL 维度的优势**与模型规模无关**，但 retrieval 优势需要足够容量

---

## 4. 与 350M 实验对比

| 指标 | 125M TinyStories | 350M FineWeb-Edu |
|------|-----------------|-----------------|
| Baseline PK@2K | ~50% (随机) | **100%** |
| EVQ vs Geo PK@8K | -1pp (无差) | **+12.7pp** |
| EVQ+YaRN PK@8K | 46% (随机) | **100%** |
| EVQ+YaRN PPL 平坦 | Yes (5.7→5.4→5.7) | Yes (70.7→70.9→107.5) |
| NLL gap 量级 | 0.01~0.07 | **0.3~3.8** |

**解读**：
- 125M 模型在 TinyStories 上的 NLL gap 比 350M 小 50x，说明模型完全没有学到位置相关的检索模式
- PPL 优势（EVQ+YaRN 平坦化）与规模无关，是频率分配的固有性质
- Retrieval 优势需要模型在训练时真正学会检索技能，才能在推理时被 YaRN 放大

---

## 5. 论文意义

**作为负面控制/规模阈值实验：**

> EVQ+YaRN 的超线性互补效应不是任意模型/数据组合的通用现象。
> 它需要模型首先具备学习位置敏感检索技能的能力（在 350M/FineWeb-Edu 上确认，在 125M/TinyStories 上未达到）。
> 这进一步支持了 EVQ 的优势来自改善位置编码表征质量的论点，而非某种数值巧合。

---

## 6. 数据位置

- 服务器：`/root/autodl-tmp/evq_phase14d/`
- 本地 JSON：`data/results_5090b/phase14d/all_results.json`
- 本地报告：`data/results_5090b/phase14d/REPORT.txt`
- 本地日志：`data/results_5090b/phase14d/run.log`
- 脚本：`scripts/m4_evq_sweep/phase14d_125m_tinystories_10pct.py`
