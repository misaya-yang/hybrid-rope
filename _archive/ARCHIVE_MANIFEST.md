# 归档清单

> **归档日期**: 2026-03-01
> **原因**: 聚焦 NeurIPS 2026 投稿，清理非必要文件，防止后续 AI 或自己再走弯路

---

## 归档规则

以下内容将移入 `_archive/`，**不会删除**，但不再属于活跃开发路径。

### 应该归档的（与论文主线无关或已被取代）

| 目录/文件 | 归档原因 |
|----------|---------|
| `sigmoid_rope_experiments/` | Phase 1-4 的旧实验，已被 EVQ 框架取代 |
| `tmp_phase4_compare/` | 临时对比目录 |
| `batch_report_2026-02-23_downstream_eval/` | 旧 8B LoRA 评估报告（协议不统一） |
| `batch_report_2026-02-24_new_server_scan/` | 服务器扫描临时文件 |
| `neurips_plan/` | 早期规划文档，已被 paperdraft/ 取代 |
| `experiments/` | 早期实验配置 |
| `archives/` | 已有归档 |
| `results/` 中大量子目录 | 见下方详细分类 |
| `scripts/` 中 70+ 脚本 | 大部分是一次性脚本，见下方分类 |

### 必须保留的（论文主线）

| 目录/文件 | 保留原因 |
|----------|---------|
| `submission/` | 📦 干净的投稿代码包 |
| `docs/paperdraft/` | 📋 铁桶理论文档 + 项目决策 |
| `paper_exports/neurips_v5_fig/` | 📄 当前 LaTeX 源 |
| `rope/schedules.py` | 🔧 核心代码 |
| `train.py` | 🔧 训练入口 |
| `scripts/m4_evq_sweep/` | 🔧 EVQ 主实验脚本 |
| `scripts/eval_longbench.py` | 🔧 评估脚本 |
| `scripts/eval_passkey_teacher_forcing.py` | 🔧 Passkey 评估 |
| `results/paper_ready/evq_tau_sweep/` | 📊 论文 Table 2 数据 |
| `results/paper_ready/evidence_chain_50m_3cfg3seed/` | 📊 50M 3-seed 数据 |
| `results/350m_final/` | 📊 350M 数据 |
| `artifacts/reviewer_2026-02-25/` | 📊 Qwen 双 seed 数据（Table 4-5） |
| `knowledge_base/` | 📖 理论知识库（参考用，以 paperdraft 为准） |
| `docs/PAPER_DRAFT_STATUS.md` | 📋 论文进度 |
| `docs/exp/prompt_500m_experiment.md` | 📋 500M 实验 prompt |

### results/ 详细分类

**保留**（论文直接引用）:
- `paper_ready/evq_tau_sweep/` — Table 2
- `paper_ready/evidence_chain_50m_3cfg3seed/` — 50M baseline
- `350m_final/` — 350M 数据
- `50m_yarn_compare_v2/` — YaRN 对比

**归档**（Legacy/已被取代/一次性）:
- `advisor_package_2026-02-15/` — 旧顾问汇报包
- `anchored_sigmoid_v3_followup/` — 旧 sigmoid 实验
- `archive_low_priority/` — 已归档
- `attention_distribution/` — 探索性分析
- `baseline_passkey/` — 旧 passkey
- `comprehensive_theta/` — theta 搜索
- `cross_model_wikitext_v1/` — 旧跨模型
- `eval_700m/` — 700M 评估（未完成）
- `frequency_range_analysis/` — 分析脚本输出
- `gamma_search/` — gamma 搜索
- `hybrid_comparison/`, `hybrid_comparison_v2/` — 旧对比
- `llama13b_triangle/` — 13B 实验（已放弃）
- `llama8b_fair_lora_suite_*` — 旧 8B LoRA
- `llama8b_post_eval_*` — 旧评估
- `llama_shape_theta_min/` — theta 搜索
- `llama_theta_matched_shape_control/` — theta 控制
- `niah_llama3_base_full/` — 旧 NIAH
- `night_run_anchored_x20_9h/` — 夜间运行
- `optimal_base_search/` — base 搜索
- `our_method_comparison/` — 旧对比
- `passkey_long/` — 旧 passkey
- `phase4_passkey_sanity*/` — Phase 4 检查
- `phase_collision_comparison*/` — 碰撞对比
- `phase_transition/` — 相变分析
- `qwen_*/` — 旧 Qwen 实验（非双 seed 版本）
- `rope_scaling_v2/` — 旧缩放
- `theoretical_validation/`, `theory_validation/` — 旧理论验证
- `theory_2026-02-22/` — 旧理论数据
- `train_freq_comparison/` — 训练频率对比
- `unified_search*/` — 统一搜索
