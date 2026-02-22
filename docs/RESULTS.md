# 实验结果汇总 (Paper Metrics & Results)

> 最后更新：2026-02-22
> ⚠️ **所有下表数字必须可在 `EXPERIMENT_REGISTRY.md` 指向的真实脚本日志和 JSON 中查实。**

## 1. 规模化扩展训练 (Scaling from Scratch)

在 50M、100M、350M 上进行标准从零预训练（基于 TinyStories 数据流）。结果标明：在相同的 $\theta$ 倍率上限环境里，通过改变重分布曲线并固定一部分极高频，泛化 PPL 获得坚实的下降。

### 1.1 主表：Across Scales的 16K 长程 PPL

| Model Size | PPL@16K (Geo) | PPL@16K (Hybrid) | Relative Gain | Protocol Source | Data Source JSON |
|------------|--------------|----------------|--------------|-----------------|------------------|
| **50M** (n=3 seed) | 18.21 $\pm$ 0.77 | **17.32 $\pm$ 0.36** | **-4.9%** | `EXP_50M_3SEED` | `results/evidence_chain_50m_3cfg3seed/results.json` |
| **100M** (seed=42) | 10.89 | **9.42** | **-13.5%** | `EXP_100M_FINAL`| `artifacts/a100_2026-02-13/data/100m_scaling/` |
| **350M** (seed=42) | 14.65 | **12.65** | **-13.7%** | `EXP_350M_FINAL`| `artifacts/a100_2026-02-13/data/350m_final/results.json` |

## 2. 渐进式基线对比 (Against YaRN Baseline)

以 YaRN（基于温度调节和渐进波段）作为基准进行比较，验证静态形状调节在原生阶段相比动态插值策略的优势。

### 2.1 50M 等级对比 PPL

| Context Length | Geo (Standard) | YaRN (Progressive) | Anchored Hybrid |
|----------------|----------------|--------------------|-----------------|
| **2048** (Train) | 6.84 | 6.84 | **6.67** |
| **4096** | 7.05 | 8.64 | **6.75** |
| **8192** | 8.83 | 16.90 | **8.69** |
| **16384** | 17.97 | 39.48 | **16.86** |

**JSON 溯源**: `results/50m_yarn_compare_v2/results.json` (`EXP_50M_YARN`)

## 3. 长程崩溃率机理验证 (Phase 4 Evaluator)

针对 `~124M` 参数级别的连续流动态评估：当逼近 32K 上下文时，传统均匀拉伸策略引发 D分布剧烈崩塌。

| 方法 (Freq Strategy) | 2K PPL | 8K PPL | 16K PPL | 32K PPL | Collapse Multiple (32K/2K) |
|---|---|---|---|---|---|
| **Standard (Geo)** | 24.2 | 41.0 | 56.1 | 412.8 | 17.0x |
| **Sigmoid** | **20.9** | **24.7** | **19.0** | **147.5** | **7.0x** |

**CSV 溯源**: `sigmoid_rope_experiments/data/ppl_vs_length.csv` (`EXP_PHASE4_124M`)

## 4. 8B 大规模微调模型比较

> 🚧 **状态：In-Progress** (待本夜 Overnight 跑数完成)。当前的所有占位均依赖公平协议架构 `EXP_8B_FAIR_LORA`。以前使用的 YaRN 与旧版 Hybrid 由于非公平面临 `rope_scaling` vs Monkey Patch 的争议，已列入禁止直接引用的黑名单档（详情见 `TERMS_AND_PROTOCOLS.md`）。

*（预期放置：不同方法的 Training Loss Curve 对比、NIAH 热力图矩阵与 LongBench 打分表现）。*
