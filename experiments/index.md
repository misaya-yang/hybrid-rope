# 实验入口

研究结论与优先级见[当前研究索引](../docs/research/next_stage_20260912/index.md)，论文主张追溯见[证据索引](../paper-2027/research/evidence/index.md)。本页只回答三件事：正式结果在哪里、代码从哪里复用、哪些目录只是历史或准备状态。

2026-09-18当天不再启动新GPU实验。完成状态只由下列结果owner和紧凑报告确定；旧launcher、队列文档或服务器目录不代表任务仍在运行。

新实验的准备／执行／报告方法见[实验工作流程](../docs/research/protocols/EXPERIMENT_WORKFLOW.md)。查全部实验家族时用[目录分层表](DIRECTORY_MAP.md)，而不是依次打开每个历史README。

## 核心完成结果

| 科学问题 | 唯一结果入口 |
|---|---|
| Llama S4多长度、clean Full-13、Natural-QA与原生参照 | [Llama结果owner](../docs/research/next_stage_20260912/TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md) |
| OLMo S4跨模型确认、clean 16K与Natural-QA | [OLMo结果owner](../docs/research/next_stage_20260912/TAILSPLINE_OLMO_CLASSIC_RESULT_20260914.md) · [强证据报告索引](iclr2027_strong_evidence_20260915/reports/README.md) |
| Qwen/GLM S4 128K三臂、自然长书与高倍率边界 | [Pro6000结果owner](../docs/research/next_stage_20260912/PRO6000_EXTREME_NATURAL_QA_RESULTS_20260916.md) |
| Llama/OLMo官方静态YaRN与Llama-3-70B尺度迁移 | [双服务器结果owner](../docs/research/next_stage_20260912/DUAL_SERVER_YARN_AND_70B_RESULTS_20260917.md) · [70B目录](llama70b_scale_20260916/README.md) |
| Kanana官方runtime YaRN：64K Full-13与128K完整上下文QA | [Kanana结果owner](kanana_yarn_tailspline_64k_20260918/RESULT.md) |
| OLMo原生窗口NCP：NLL、Full-13、Natural-QA和机制拆分 | [Native/NCP结果owner](native_enhancement_oral_20260915/index.md) |
| 固定支持、432M MLA、750M继续训练与Cosh适配证据 | [论文资产A01／A09–A12／A19](../paper-2027/research/evidence/index.md) · [当前学习协议](../paper-2027/appendix/compact_f_learning.tex) |
| Native后续的成立边界与负结果 | [NTS2四模型结果](native_tailspline_s2_midgain_20260917/RESULT.md) · [CA-NCP否证](ca_ncp_native_20260917/RESULT.md) · [安全约束后续](ca_ncp_safe_followup_20260917/RESULT.md) |

跨结果的简明科学罗盘见[关键实验罗盘](../docs/research/next_stage_20260912/KEY_EXPERIMENT_COMPASS_20260914.md)。上表不复制逐任务数字；正式数值、比较合同、输出健康和证据边界只在对应owner维护。

## 可复用实现

| 用途 | 代码入口 |
|---|---|
| 静态TailSpline/MrPro/YaRN构表、安装与通用runner | [固定表流水线](fixed_rope_three_interfaces_20260913/index.md) |
| clean RULER、自然QA、官方YaRN和便携报告 | [strong-evidence流水线](iclr2027_strong_evidence_20260915/README.md) |
| 9月15日冲刺工具与历史服务器回执 | [冲刺目录](iclr2027_three_track_sprint_20260915/README.md) |
| Native/NCP评测与机制干预 | [Native/oral目录](native_enhancement_oral_20260915/index.md) |
| Kanana 64K/128K运行与报告复算 | [Kanana代码入口](kanana_yarn_tailspline_64k_20260918/README.md) |

数据准备、tokenization缓存和大raw继续留在实验服务器；Git只收纳可复算代码、紧凑报告和必要身份。不得为了目录整洁移动仍被脚本或registry引用的来源文件。

## 历史与准备状态

- [服务器任务分层](iclr2027_three_track_sprint_20260915/SERVER_TASK_LAYERS.md)是带时间戳的执行回执，不再是当前队列。
- `ca_ncp_llama_native_20260917`、`native_followup_five_20260917`和`phi3_s32_20260917`保留代码与失败/未执行边界，不列为当前任务。
- [完整实验目录](CATALOG_20260913.md)只用于追溯旧代码、旁线和历史结果，不作为默认上下文。
