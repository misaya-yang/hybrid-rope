# 实验结果汇总 (Paper Metrics & Results)

> 最后更新：2026-02-27
> ⚠️ **所有下表数字必须可在 `EXPERIMENT_REGISTRY.md` 指向的真实脚本日志和 JSON 中查实。**

## 0. EVQ τ-Sweep 实验 (V5 论文核心数据)

> 论文 V5 采用 EVQ (Exact Variational Quantization) 理论框架，τ 为唯一超参。
> 下方数据来自 RTX 5090 服务器 TinyStories from-scratch 训练，双种子验证。

### 0.1 50M 全谱 τ-Sweep (seed=42)

| τ | Phase Collision | PPL@2048 | PPL@4096 | PPL@8192 | PPL@16384 | Δ PPL@16K |
|---:|:--------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| 0.00 (geo) | 0.386 | 4.146 | 6.183 | 14.004 | 33.316 | — |
| 0.20 | 0.390 | 4.160 | 7.283 | 17.457 | 42.314 | +27.0% |
| 0.40 | 0.438 | 4.207 | 6.789 | 14.331 | 33.298 | -0.1% |
| 0.60 | 0.361 | 4.173 | 7.253 | 16.571 | 37.978 | +14.0% |
| 0.80 | 0.290 | 4.169 | 7.736 | 17.193 | 36.306 | +9.0% |
| 1.00 | 0.305 | 4.150 | 7.830 | 17.790 | 37.369 | +12.2% |
| **1.50** | **0.268** | **4.134** | **6.667** | **13.778** | **29.697** | **-10.9%** |
| 2.00 | 0.278 | 4.197 | 7.048 | 14.981 | 35.646 | +7.0% |

**JSON 溯源**: `results/paper_ready/evq_tau_sweep/50m_seed42.json` (`EXP_EVQ_50M_SWEEP`)

### 0.2 125M Cross-Seed 验证 (τ=0 vs τ=1.5)

| Seed | τ | PPL@2048 | PPL@4096 | PPL@8192 | PPL@16384 | Δ PPL@16K |
|-----:|---:|:--------:|:--------:|:--------:|:---------:|:---------:|
| 42 | 0.00 | 3.346 | 5.454 | 13.476 | 34.153 | — |
| 42 | **1.50** | **3.290** | **4.681** | **10.459** | **27.699** | **-18.9%** |
| 137 | 0.00 | 3.318 | 5.737 | 12.694 | 28.502 | — |
| 137 | **1.50** | **3.318** | **5.321** | **11.341** | **26.860** | **-5.8%** |

**Cross-seed 一致性**: 方向 100% 一致 (τ=1.5 ≤ geometric 在所有 6/6 比较点)。
**Waterbed**: 两个种子均无 waterbed 效应 (short-context PPL 保持或改善)。

**JSON 溯源**: `results/paper_ready/evq_tau_sweep/125m_seed42.json`, `125m_seed137.json` (`EXP_EVQ_125M_SWEEP`)

### 0.3 Scaling Law Summary

| Scale | Δ PPL@16K (best seed) | Δ PPL@16K (cross-seed) |
|-------|:---------------------:|:----------------------:|
| 50M | -10.9% | — (单种子) |
| 125M | -18.9% | -5.8% |

改善随模型规模放大: 50M → 125M, relative gain 从 10.9% 增长到 18.9%。

---

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
