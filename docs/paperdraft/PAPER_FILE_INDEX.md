# 论文文件总索引 (Paper File Master Index)

> **最后更新**: 2026-03-01
> **用途**: 写论文时的快速导航，给 AI 助手的上下文入口

---

## 一、必读文档（写论文前按顺序读）

| 优先级 | 文件 | 内容 | 状态 |
|:---:|------|------|:---:|
| 1 | `docs/paperdraft/THEORY_IRONCLAD.md` | 理论体系权威参考：推导链、定理、证明、审稿防御 | ✅ 已更新 |
| 2 | `docs/paperdraft/EXPERIMENT_RESULTS_128TOK.md` | 128-tok 实验结果总结：核心数据表、claim 清单 | ✅ 新建 |
| 3 | `docs/paperdraft/LATEX_SNIPPETS.md` | 可直接粘贴的 LaTeX 段落（含实验数据） | ✅ v2 |
| 4 | `docs/paperdraft/FINAL_ACTION_PLAN.md` | 行动方案 v4：实验清单、论文叙事路径 | ✅ v4 |
| 5 | `docs/paperdraft/LEARNABLE_TAU_DESIGN.md` | Learnable τ 完整设计 + 实验结果 | ✅ v2 |
| 6 | `docs/paperdraft/DAPE_REFERENCE.md` | DAPE (NeurIPS 2024) 对标分析 | ✅ |
| 7 | `docs/paperdraft/EXPERIMENT_AUDIT_V4.md` | 实验困境诊断与破局方案 | ✅ |

---

## 二、核心实验数据

### 2.1 128-Token PE Quality Test（最新，核心）

**实验配置**: 125M, 128 tokens, 15M train, FineWeb-Edu, base=500000
**远程服务器数据路径**: `/root/autodl-tmp/evq_128tok/`

| 数据 | 远程路径 | 关键数值 |
|------|---------|---------|
| 完整结果 JSON | `results_checkpoint.json` | 所有 Phase 数据 |
| τ 轨迹 (seed42) | `125m_learnable_init1.00_seed42/tau_trajectory.json` | τ→1.139 |
| τ 轨迹 (seed137) | `125m_learnable_init1.00_seed137/tau_trajectory.json` | τ→1.144 |
| τ 轨迹 (seed256) | `125m_learnable_init1.00_seed256/tau_trajectory.json` | τ→1.138 |
| DAPE 学习频率 | `125m_dape_lr100_seed42/dape_learned_inv_freq.npy` | 32 个频率 |
| Algorithm 1 | `algorithm1_prediction.json` | τ*=40.96 (失败) |

**论文核心数据表**:

```
EVQ fixed τ=1.5:     PPL@8K = 419.7  (vs Geo 513.7, Δ = -18.3%)
EVQ learnable:        PPL@8K = 441.4  (vs Geo 513.7, Δ = -14.1%)
DAPE best (32 params): PPL@8K = 455.3  (vs Geo 513.7, Δ = -11.4%)
Learnable τ (3-seed): 1.141 ± 0.003
```

### 2.2 TinyStories From-Scratch Scaling（已有）

**本地路径**: `results/paper_ready/evq_tau_sweep/evq_sweep_paper_table.csv`

| Model | τ | PPL@16K | Δ vs Geo |
|-------|---|---------|----------|
| 50M | 1.5 | 29.697 | -10.9% |
| 125M (seed42) | 1.5 | 27.699 | -18.9% |
| 125M (seed137) | 1.5 | 26.860 | -5.8% |

### 2.3 LoRA 下游任务 Waterbed 验证（已有）

**本地路径**: `results/paper_ready/llama8b_fair_lora_suite_20260214/`

关键数据: Retrieval +2.50, Single-doc QA +0.84, Multi-hop QA -2.69, Code -1.42
(方向在双 seed、双模型族间完全复现)

### 2.4 其他已有结果

| 实验 | 路径 | 用途 |
|------|------|------|
| 350M from-scratch | `results/paper_ready/night_run_anchored_x20_9h/` | Scaling law |
| Passkey retrieval | `results/paper_ready/passkey_long/` | PE 质量评测 |
| NIAH heatmap | `results/paper_ready/niah_llama3_base_full/` | PE 质量评测 |
| Qwen cross-model | `results/paper_ready/qwen_plugandplay_wikitext_v1/` | 跨模型验证 |

---

## 三、论文 LaTeX 源文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 主 tex | `submission/paper/hybrid_rope_neurips.tex` | 当前提交版 |
| v5 导出 | `paper_exports/neurips_v5/hybrid_rope_neurips_v5.tex` | 带图版本 |
| 提交代码 | `submission/code/` | run_evq_sweep.py, train.py 等 |
| 提交结果 | `submission/results/` | table2_evq_sweep.csv 等 |

---

## 四、核心代码

| 文件 | 路径 | 说明 |
|------|------|------|
| EVQ 频率调度 | `rope/schedules.py` | 固定 EVQ 实现（~10 行核心） |
| Learnable EVQ | `rope/learnable_evq.py` | LearnableEVQRoPE + Algorithm 1 + 工具 |
| 训练脚本 | `scripts/m4_evq_sweep/run_evq_sweep.py` | 完整训练+评估 pipeline |
| 128-tok launcher | `experiments/run_125m_learnable.py` | 7-run 实验矩阵 |
| τ 轨迹可视化 | `experiments/plot_tau_trajectory.py` | Figure 生成 |

---

## 五、知识库（历史上下文）

| 文件 | 内容 | 写论文时是否需读 |
|------|------|:---:|
| `knowledge_base/ALL_IN_ONE.md` | 综合参考（可能过时） | 仅需要历史上下文时 |
| `knowledge_base/02_论文故事线与主张.md` | 早期叙事（v1/v2） | 参考但以 v4 为准 |
| `knowledge_base/09_unified_theory_crlb.md` | 早期理论（措辞过度） | 以 THEORY_IRONCLAD.md 为准 |
| `knowledge_base/11_waterbed_strict_proof.md` | Waterbed 证明 | 写 Appendix 时需要 |

---

## 六、待完成实验

### 5090 上（优先，便宜）

| 实验 | 指令文件 | 预计成本 | 优先级 | 价值 |
|------|---------|---------|:---:|------|
| **Phase 5: 1024-tok + Passkey + SOTA** | `PROMPT_PHASE5_1024TOK.md` | ~4h | **P0** | Passkey + YaRN/PI 对比 |
| Phase 4: Context Extension (2K→8K) | `PROMPT_NEXT_EXPERIMENTS.md` | ~4h | P1 | 实用场景验证 |
| 128-tok 50M scaling 点 | `PROMPT_PHASE5_1024TOK.md` §B | ~30min | P2 | Scaling 曲线 |

### 主力机器上（后续）

| 实验 | 预计成本 | 优先级 | 价值 |
|------|---------|:---:|------|
| 500M from-scratch | ~8h | P3 | 堵"规模太小" |
| Passkey/NIAH @longer context | ~2h | P3 | 更强检索证据 |

---

## 七、关键数值速查

| 数值 | 值 | 来源 |
|------|-----|------|
| EVQ vs Geometric @8K (128-tok) | **-18.3%** | Phase 1 A3 |
| EVQ vs DAPE @8K (128-tok) | **-7.8%** (fixed) / **-3.1%** (learnable) | Phase 2 |
| τ_learned (128-tok, 3-seed) | **1.141 ± 0.003** | Phase 3 |
| τ_sweep optimal | **1.5** (跨协议/跨数据集一致) | Phase 1 + TinyStories |
| EVQ vs Geometric @16K (TinyStories 125M) | **-18.9%** | evq_sweep_paper_table.csv |
| Waterbed: Retrieval vs Multi-hop | **+2.50 vs -2.69** | LoRA 8B 下游 |
| Algorithm 1 residual | **35-49%** | Phase 0 (失败) |
| Broadband R² (mid-region) | **0.994** | 论文 Table 1 |

---

*索引创建: 2026-03-01*
