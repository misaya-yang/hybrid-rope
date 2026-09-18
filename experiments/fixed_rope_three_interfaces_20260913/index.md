# 固定表实验流水线

当前方法、数据、对照与停止条件以[TailSpline统一评测合同](../../docs/research/next_stage_20260912/TAILSPLINE_ROPE_METHOD_AND_UNIFIED_EVAL_20260914.md)为准；研究优先级见[研究索引](../../docs/research/next_stage_20260912/index.md)。
本目录同时保留旧实验代码，脚本存在不表示它仍在执行队列中。

| 复用用途或历史路径 | 实现 |
|---|---|
| 构造静态表与记录身份 | [tables.py](tables.py) |
| 核验TailSpline数学与边界 | [tailspline_verification.py](tailspline_verification.py)：CPU证明不代表任务性能 |
| 核验YaRN–MrPro等剂量后移对照 | [dose_matched_yarn_mrpro_verification.py](dose_matched_yarn_mrpro_verification.py)：只定义未来理论实验，不进入当前GPU队列 |
| Llama现有72行8K/32K诊断 | [run_tailspline_llama_s4_core6.sh](run_tailspline_llama_s4_core6.sh)：不可称8/16/32K |
| 无重复补齐正确Core-6三长度 | [run_tailspline_llama_s4_complete108.sh](run_tailspline_llama_s4_complete108.sh)：只新增16K的36行/臂 |
| 准备Full-13与PPL46资产 | [prepare_tailspline_llama_classic_assets.sh](prepare_tailspline_llama_classic_assets.sh)：CPU-only，32 ProofPile + 14 PG19 |
| Llama经典评测 | [run_tailspline_llama_s4_classic.sh](run_tailspline_llama_s4_classic.sh)、[tailspline_llama_classic_report.py](tailspline_llama_classic_report.py)：两臂已完成，结果owner见[经典结果](../../docs/research/next_stage_20260912/TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md) |
| OLMo跨模型确认 | [prepare_tailspline_olmo_classic_assets.sh](prepare_tailspline_olmo_classic_assets.sh)、[run_tailspline_olmo_s4_classic.sh](run_tailspline_olmo_s4_classic.sh)、[tailspline_olmo_classic_report.py](tailspline_olmo_classic_report.py)：两臂已完成；结果owner见[OLMo经典结果](../../docs/research/next_stage_20260912/TAILSPLINE_OLMO_CLASSIC_RESULT_20260914.md) |
| 查原Qwen转Llama衔接逻辑 | [chain_qwen_tailspline_to_llama.sh](chain_qwen_tailspline_to_llama.sh)：查回执时读取，不据此重放已完成衔接 |
| 驻留评测与断点恢复 | [resident_eval.py](resident_eval.py)、[pipeline.py](pipeline.py) |
| 同prompt结果汇总 | [matched_generation_report.py](matched_generation_report.py) |

结果以目标运行目录的完整raw、完成回执和配对报告为准。72行及补齐后的108行Core-6均为诊断，不替代最终PPL、NIAH/passkey与Full-13交付。

按需追溯固定表三接口、fixed-u、mix075、局部修复与旧启动命令，见[历史实现和结果目录](CATALOG_20260914.md)。保留的历史证据不并入精确TailSpline的统一主比较。
