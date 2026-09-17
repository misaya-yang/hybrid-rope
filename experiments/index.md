# 当前实验入口

当前目标、方法与停止条件见[研究索引](../docs/research/next_stage_20260912/index.md)。
本页导航不表示远端任务已经启动或完成。

| 用途 | 入口 |
|---|---|
| 当前实验缺口与下一版整合准备（非执行命令） | [统一准备清单](../docs/research/next_stage_20260912/PAPER_NEXT_REVISION_PREPARATION_20260916.md) |
| 当前结果、负结果与运行中任务总览 | [关键实验罗盘](../docs/research/next_stage_20260912/KEY_EXPERIMENT_COMPASS_20260914.md) |
| 精确TailSpline与Llama统一评测 | [方法与评测合同](../docs/research/next_stage_20260912/TAILSPLINE_ROPE_METHOD_AND_UNIFIED_EVAL_20260914.md) |
| 实现、运行脚本与报告工具 | [固定表流水线](fixed_rope_three_interfaces_20260913/index.md) |
| 当前服务器与显存分层 | [任务分层](iclr2027_three_track_sprint_20260915/SERVER_TASK_LAYERS.md)：运行中、32GB停放、48GB+、CPU与历史证据 |
| 当前冲刺代码和便携报告 | [2026-09-15执行目录](iclr2027_three_track_sprint_20260915/README.md)；[紧凑报告索引](iclr2027_three_track_sprint_20260915/reports/README.md) |
| 后续强证据实验包装器 | [2026-09-15 strong-evidence入口](iclr2027_strong_evidence_20260915/README.md)：clean跨模型、自然长文、C对照与OLMo QA→RULER-200顺序 |
| Llama-3-70B NF4尺度迁移 | [70B执行与报告入口](llama70b_scale_20260916/README.md)：S4/32K完整结果、S16/128K PPL及未完成NIAH边界 |
| 原生增强、理论先行与评测提速 | [Native/oral实验与结果](native_enhancement_oral_20260915/index.md)：NCP Native-4K NLL、Full-13、Natural-QA及机制干预结果 |
| CA-NCP载波对齐原生实验 | [OLMo代码与运行合同](ca_ncp_native_20260917/README.md)：无标签Native Q/K统计、rank-2换基、五臂Full-13×10；[Llama迁移准备](ca_ncp_llama_native_20260917/README.md)：仅在OLMo gate支持后执行，不代表Llama GPU结果 |
| CA-NCP安全约束诊断 | [三臂后续合同](ca_ncp_safe_followup_20260917/README.md)：算子预算测地线与跨来源坐标共识；复用同一Full-13×10，属于失败后的开发诊断 |
| Native后续注意力算子 | [A/B/C代码入口](native_followup_five_20260917/README.md)：偶奇核、距离质量投影与置信排序；默认PLAN_ONLY，不代表已运行或已有结果 |
| Phi-3-mini-4K S32门控 | [16K→32K执行合同](phi3_s32_20260917/README.md)：TailSpline S32、Full-13×10、80%固定阈值与授权关机条件 |
| 核实已进入论文的结果 | [论文证据](../paper-2027/research/evidence/index.md) |

[完整实验目录](CATALOG_20260913.md)用于复用旧代码或追溯历史结果；其中队列和服务器状态不定义当前优先级。
