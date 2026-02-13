# 实验总索引（2026-02-13）

本文件用于快速定位：**实验配置 -> 脚本 -> 结果文件 -> 图表/总结**。

## 1. A100（50M/100M/350M 主线）

| 实验 | 入口脚本 | 结果文件 | 说明文档/图表 |
|---|---|---|---|
| 50M 因子公平对照（theta × shape） | `artifacts/a100_2026-02-13/scripts/run_50m_theta_shape_factorial.py` | `artifacts/a100_2026-02-13/data/50m_theta_factorial/results.json` | `artifacts/a100_2026-02-13/A100_RESULTS_SUMMARY_FOR_ADVISOR_2026-02-13.md` |
| 50M YaRN 对照 v2 | `artifacts/a100_2026-02-13/scripts/run_50m_yarn_eval_v2.py` | `artifacts/a100_2026-02-13/data/50m_yarn_compare_v2/results.json` | `results/figures/50m_yarn_compare_v2.svg` |
| 100M scaling 冲刺 | `artifacts/a100_2026-02-13/scripts/run_100m_scaling.py` | `artifacts/a100_2026-02-13/data/100m_scaling/results.json` | `artifacts/a100_2026-02-13/README.md` |
| 350M final 对照 | `artifacts/a100_2026-02-13/scripts/run_350m_final.py` | `artifacts/a100_2026-02-13/data/350m_final/results.json` | `artifacts/a100_2026-02-13/figures/350m_geo_vs_hybrid.png` |
| unified search（A split） | `artifacts/a100_2026-02-13/scripts/unified_search.py` | `artifacts/a100_2026-02-13/data/unified_search/results_A.json` | `artifacts/a100_2026-02-13/data/unified_search/log_A.txt` |
| 3cfg×3seed 稳健性 | `artifacts/a100_2026-02-13/scripts/unified_search_3cfg_3seed.py` | `artifacts/a100_2026-02-13/data/unified_search_3cfg_3seed/results.json` | `results/unified_search_3cfg_3seed/results.json` |

## 2. A800（Llama-3-8B Hybrid-LoRA）

| 实验 | 入口脚本 | 结果文件 | 说明文档 |
|---|---|---|---|
| LoRA 训练主线 | `artifacts/a800_2026-02-13/run_llama3_hybrid_lora_v3.py` | `artifacts/a800_2026-02-13/results/llama3_hybrid_lora/summary.json` | `artifacts/a800_2026-02-13/A800_LLAMA3_HYBRID_LORA_EVAL_SUMMARY_2026-02-13.md` |
| base vs hybrid 评测 | `artifacts/a800_2026-02-13/scripts/eval_hybrid_lora_vs_base.py` | `artifacts/a800_2026-02-13/results/llama3_hybrid_lora_eval/results.json` | `artifacts/a800_2026-02-13/results/llama3_hybrid_lora_eval/SANITY_SUMMARY.md` |
| h800 并行频率搜索 | `server_artifacts_2026-02-13/a800/scripts/run_h800_parallel_rope.py` | `artifacts/a800_2026-02-13/results/h800_parallel/results_h800.json` | `artifacts/a800_2026-02-13/README.md` |

## 3. R6000（Qwen2.5-7B Hybrid-LoRA）

| 实验 | 入口脚本 | 结果文件 | 状态 |
|---|---|---|---|
| Qwen Hybrid-LoRA 训练 | `artifacts/r6000_2026-02-13/scripts/run_qwen_hybrid_lora_train.py` | `artifacts/r6000_2026-02-13/results/`（训练结束后写入） | 进行中（见 live_sync） |
| 自动评测套件（PPL + Passkey + KV） | `artifacts/r6000_2026-02-13/scripts/run_qwen_eval_suite.py` | `/opt/dfrope/results/qwen_hybrid_lora/eval_suite.json`（服务器） | 训练结束自动触发 |

## 4. H100 计划包（尚未执行）

| 内容 | 位置 |
|---|---|
| 1.5B 计划、运行手册、租用决策 | `h100_advanced_experiments/docs/` |
| 实验矩阵与结果 schema | `h100_advanced_experiments/configs/` |
| 执行脚本与作图脚本 | `h100_advanced_experiments/scripts/` |

## 5. 快速阅读顺序

1. `README.md`
2. `docs/RESULTS.md`
3. `docs/METHODOLOGY.md`
4. `docs/EXPERIMENT_INDEX_CN.md`
5. 进入对应 `artifacts/<machine>_2026-02-13/`
