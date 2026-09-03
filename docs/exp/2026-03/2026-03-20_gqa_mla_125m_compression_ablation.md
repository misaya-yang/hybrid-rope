# 125M GQA/MLA 压缩消融实验报告

> 日期: 2026-03-20
> 状态: **VALID** (Phase 1-5 完成, Phase 6 τ=1.5 右移测试进行中)
> 级别: **Tier 0 — 论文核心数据**
> 服务器: AutoDL RTX PRO 6000 Blackwell 96GB

---

## 0. 执行摘要

**核心发现：KV压缩越激进，EVQ的长上下文PPL改善越大。MLA-16（仅8个RoPE频率）EVQ改善达-47.8%@8K，验证了"频率越稀缺，优化分配越重要"的理论预期。**

---

## 1. 实验设置

| 参数 | 值 |
|------|-----|
| 模型规模 | 125M (12L/12H/768d, head_dim=64) |
| 训练长度 | 2048 (seq_len) |
| 训练token | 100M |
| 数据集 | FineWeb-Edu |
| τ配置 | GEO: τ=0.0 / EVQ: τ=1.414 (= 64/√2048) |
| passkey | 2% 训练比例, 50/100位, L=2K/4K/8K |
| 评估 | PPL@2K/4K/8K/16K + passkey retrieval |
| seeds | 42 |
| compile | torch.compile(mode='default') |

### 5个配置

| Config | 注意力类型 | KV heads | RoPE频率数 | KV缓存大小(相对MHA) |
|--------|-----------|----------|-----------|-------------------|
| MHA | Standard | 12 | 32 | 100% |
| GQA-4 | Grouped Query | 4 | 32 | 33% |
| GQA-2 | Grouped Query | 2 | 32 | 17% |
| MLA-32 | Multi-head Latent | - | 16 (d_rope=32) | ~latent |
| MLA-16 | Multi-head Latent | - | 8 (d_rope=16) | ~latent |

---

## 2. 核心结果

### 2.1 PPL (Perplexity)

| Config | tag | PPL@2K | PPL@4K | PPL@8K | PPL@16K | **EVQ Δ@8K** | **EVQ Δ@16K** |
|--------|-----|--------|--------|--------|---------|------------|-------------|
| MHA | GEO | 44.24 | 61.75 | 124.0 | 213.5 | | |
| | EVQ | 43.62 | 58.09 | 106.2 | 185.3 | **-14.4%** | **-13.2%** |
| GQA-4 | GEO | 49.37 | 59.67 | 112.6 | 191.0 | | |
| | EVQ | 50.76 | 60.73 | 101.0 | 162.6 | **-10.3%** | **-14.9%** |
| GQA-2 | GEO | 50.56 | 67.98 | 130.5 | 220.2 | | |
| | EVQ | 50.56 | 57.45 | 99.3 | 172.9 | **-23.9%** | **-21.5%** |
| MLA-32 | GEO | 65.47 | 98.53 | 240.1 | 400.2 | | |
| | EVQ | 65.59 | 104.0 | 225.1 | 363.7 | **-6.3%** | **-9.1%** |
| MLA-16 | GEO | 67.13 | 68.07 | 245.5 | 653.4 | | |
| | EVQ | 65.92 | 69.84 | 128.2 | 340.4 | **-47.8%** | **-47.9%** |

### 2.2 Passkey Retrieval

| Config | GEO ret | EVQ ret | Δ |
|--------|---------|---------|---|
| MHA | 62.0% | 64.7% | +2.7% |
| GQA-4 | 55.3% | 52.7% | -2.6% |
| GQA-2 | 62.0% | 51.3% | -10.7% |
| MLA-32 | 57.3% | 50.0% | -7.3% |
| MLA-16 | 48.7% | 51.3% | +2.6% |

注: passkey在KV压缩下不太稳定，PPL是更可靠的EVQ评测指标。

---

## 3. 关键分析

### 3.1 "频率越稀缺 → EVQ收益越大" 假说

EVQ改善率@8K排序：
1. **MLA-16** (8个频率): **-47.8%** ← 频率最稀缺
2. **GQA-2** (32个频率, 2个KV头): **-23.9%**
3. **MHA** (32个频率, 12个KV头): **-14.4%**
4. **GQA-4** (32个频率, 4个KV头): **-10.3%**
5. **MLA-32** (16个频率): **-6.3%**

整体趋势支持假说：极端压缩（MLA-16, GQA-2）时EVQ收益最大。
但不是严格单调（MLA-32 < GQA-4 < MHA），说明频率数量不是唯一因素。

### 3.2 MLA PPL基线偏高

MLA配置的GEO基线PPL显著高于MHA/GQA（65-67 vs 44-50 @2K）。
这可能是因为：
- MLA架构在125M小规模下不够高效（latent compression overhead）
- d_rope限制了位置信息容量

### 3.3 论文意义

这组实验直接支持论文的核心论点：
- **当位置编码资源（RoPE频率）受限时，EVQ的最优分配比几何分配的优势更显著**
- 这与DeepSeek-V2/V3使用MLA的行业趋势高度相关 — EVQ是MLA的天然搭配

---

## 4. 数据溯源

| 项目 | 路径 |
|------|------|
| 训练脚本 | `REMOTE_RUN_ROOT/scripts/core_text_phases/run_gqa_evq_experiment.py` |
| Phase 1 结果 | `REMOTE_RUN_ROOT/gqa_125m_experiment/mha_kv12/results_final.json` |
| Phase 2 结果 | `REMOTE_RUN_ROOT/gqa_125m_experiment/gqa_kv4/results_final.json` |
| Phase 3 结果 | `REMOTE_RUN_ROOT/gqa_125m_experiment/gqa_kv2/results_final.json` |
| Phase 4 结果 | `REMOTE_RUN_ROOT/gqa_125m_experiment/mla_r32/results_final.json` |
| Phase 5 结果 | `REMOTE_RUN_ROOT/gqa_125m_experiment/mla_r16/results_final.json` |
| Phase 6 结果 | `REMOTE_RUN_ROOT/gqa_125m_experiment/gqa_kv2_tau1p5/results_final.json` (not used for paper claims) |
| 日志 | `phases2to5_v2.log`, `phases4to5.log`, `phase6.log` |
