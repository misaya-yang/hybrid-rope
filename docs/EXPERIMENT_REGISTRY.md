# 实验事实表 (Experiment Registry)

> 最后更新：2026-02-27
> 目的：作为唯一权威来源，为论文提供 100% 可追溯的实验数据支持。**任何在此表中标记为 "Deprecated" 或找不到路径的数据，禁止放入论文**。
>
> ⚠️ **重要提醒 (2026-02-27)**: 论文已从 V4 (anchored-sigmoid) 改写为 V5 (EVQ/τ)。
> 下方 Tier 1 实验使用旧方法（anchored_sigmoid/hybrid），与 V5 理论框架不完全匹配。
> 新的 EVQ τ-sweep 实验正在进行中（见 Tier 0），完成后将替代 Tier 1 数据。

## 0. EVQ τ-sweep 实验 (Tier 0: V5 论文核心 — 进行中)

| Experiment ID | Hypothesis / Purpose | Model | τ values | Seeds | Entry Script | Output | Status |
|---------------|----------------------|-------|----------|-------|--------------|--------|--------|
| `EXP_EVQ_50M_SWEEP` | 验证 EVQ τ 对 PPL 的影响 + waterbed | 50M | 0.0,0.2,0.4,0.6,0.8,1.0,1.5,2.0 | 42 | `scripts/m4_evq_sweep/run_evq_sweep.py --tier 50m` | 服务器 5090 | ✅ **完成** τ=1.5 最优，PPL@16K -10.9% |
| `EXP_EVQ_125M_SWEEP` | 验证 τ-sweep scaling 到 125M | 125M | 0.0, 0.2, 1.5 (部分) | 42 | `scripts/m4_evq_sweep/run_evq_sweep.py --tier 125m` | 服务器 5090 | ✅ **完成** τ=1.5 PPL@16K -18.9%, 全长度改善 |
| `EXP_EVQ_8B_LONGINST` | EVQ τ=1.5 在 8B LoRA long-instruction | Llama-3-8B | τ=1.5 (primary), 1.2/1.8 (bracket) | 42,1337 | `scripts/isolated/longinst/` | TBD | 🔴 **下一步** |

## 1. 从零训练主线 (Tier 1: 绝对核心证据)

| Experiment ID | Hypothesis / Purpose | Model | Dataset & Protocol | Baselines | Method Params | Seeds | Entry Script | Output Files (JSON/CSV) | Key Numbers (PPL@16K) | Status |
|---------------|----------------------|-------|--------------------|-----------|---------------|-------|--------------|-------------------------|------------------------|--------|
| `EXP_50M_3SEED` | 验证 Hybrid 优于 Geo 及其对噪声的稳健性 | 50M | TinyStories, 500k steps, 2k/16k PPL | Geo_500k, AnchPoly | a=0.2, t=100k | `[42,123,7]` | `archives/a100/unified_search_3cfg_3seed.py` | `results/evidence_chain_50m_3cfg3seed/results.json` | Geo: 18.2±0.8; Hybrid: **17.3±0.4** | ✅ Paper-ready |
| `EXP_50M_YARN` | 对比原生 Hybrid 与渐进式 YaRN | 50M | TinyStories, 500k steps, 2k-16k PPL | Geo, YaRN | none | 42 | `artifacts/a100_2026-02-13/scripts/run_50m_yarn_compare.py` / `scripts/plot_yarn_compare.py` | `results/50m_yarn_compare_v2/results.json` | Geo: 17.97; YaRN: 39.48; Hybrid: **16.86** | ✅ Paper-ready |
| `EXP_100M_FINAL` | 验证 100M 参数规模改善 | 100M | TinyStories, 500k steps | Geo | a=0.2, t=100k | 42 | (散列/详见统一脚本) | `artifacts/a100_2026-02-13/data/100m_scaling/` | Geo: 10.88; Hybrid: **9.41** (-13.5%) | ✅ Paper-ready |
| `EXP_350M_FINAL` | 验证改善随规模放大(至350M)稳定性 | 350M | TinyStories, 500M tokens, chunk eval | Geo_500k | a=0.2, t=100k | 42 | `archives/a100/run_350m_final.py` | `artifacts/a100_2026-02-13/data/350m_final/results.json` | Geo: 14.65; Hybrid: **12.65** (-13.7%) | ✅ Paper-ready |

## 2. Phase 4 训练期与机理验证 (Tier 1.5 - 2: 高可信支撑)

| Experiment ID | Hypothesis / Purpose | Model | Dataset & Protocol | Baselines | Method Params | Seeds | Entry Script | Output Files (JSON/CSV/PDF) | Key Numbers (PPL@16K/32K) | Status |
|---------------|----------------------|-------|--------------------|-----------|---------------|-------|--------------|-----------------------------|---------------------------|--------|
| `EXP_PHASE4_124M` | 验证 Sigmoid 频率设计在大上下文的动态 | ~124M | TinyStories, Phase4, 512-32K PPL | Standard, Anchored | Sigmoid, Anchored-20 | 42 | `sigmoid_rope_experiments/run_phase4_corrected.py` | `sigmoid_rope_experiments/data/ppl_vs_length.csv` | **16K** Std: 56.1, Sig: 19.0; **32K** Std: 412.8; Sig: 147.5 | ✅ Paper-ready |
| `EXP_COLLISION_D` | 极小 theta 崩溃由距离分布 D 决定 | Llama(类) | - | Phase Collision D | Geo_10k | geo=10k, Sigmoid=100k | - | `results/anchored_sigmoid_v3_followup/` | Collapse ratio: 22x -> 1.08x | ✅ Paper-ready |
| `EXP_ATTN_D_DIST` | 注意力距离分布近似幂律 D(Δ)∝Δ^(-γ) | Llama | - | 注意力探针 | - | - | (Attention probing script) | `results/attention_distribution/` | L2 γ=1.31, avg γ=0.72 | ✅ Paper-ready |

## 3. 长程 8B 及以上实验 (Tier 3 -> Tier 1: 决胜证据)

| Experiment ID | Hypothesis / Purpose | Model | Dataset & Protocol | Baselines | Method Params | Seeds | Entry Script | Output Files | Key Notes | Status |
|---------------|----------------------|-------|--------------------|-----------|---------------|-------|--------------|--------------|-----------|--------|
| `EXP_8B_FAIR_LORA` | 完全公平条件下对比长程方法 (消除实现差异) | Llama-3-8B | 600 steps, L=16K, PPL+NIAH | PI, YaRN | inv_freq.copy_() buffer 覆写 | - | `scripts/run_llama8b_fair_suite.py` / `archives/2026-02-22/scripts/run_overnight_8h.py` | `results/overnight_8h/summary/` | 4方法严格控制变量，统一使用 inv_freq 操作不使用 HF rope_scaling | 🔄 In-progress |

## 4. 已知问题与废弃实验 (⚠️ 禁止在正面结果中引用)

| Experiment ID | Content / Issue | Model | Reason for Deprecation | Corrective Action | Status |
|---------------|-----------------|-------|------------------------|-------------------|--------|
| `BAD_8B_LORA_OLD` | 旧版 YaRN / PI / Hybrid 结果对比 | Llama-8B | 协议不公平：YaRN/PI 使用官方内置 `rope_scaling` API，而 Hybrid 使用自定义模型 forward monkey patch。且 Hybrid 的超参并未针对 8B 做调优，缺乏 rigid_j0 高频保护，中频信息饿死。 | 设计并迁移至 `EXP_8B_FAIR_LORA`，采用 `inv_freq.copy_()` | ⚠️ **Do Not Cite** |
| `BAD_50M_BASE_300K` | 尝试用 base=300k 做 Sigmoid 消融 | 50M | Base 选择导致的局部失效 (Standard PPL 优于 Sigmoid，-9.1%)。 | 将其移入 Limitations / Failure Modes 讨论区，证明 Base 选择极大影响最终结果。 | ⚠️ Failure Mode (For Discussion Only) |
| `BAD_ZERO_SHOT_SWAP`| 未经训练直接替换推理频率验证 | 多种模型 | 彻底崩溃 | 认定频率重划分必须配合训练以激活 phase 适应。写入论文 Limitations。 | ⚠️ Discussion Only |
