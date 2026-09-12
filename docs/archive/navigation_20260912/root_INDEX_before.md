# INDEX

文件导航。当前工作见 [HANDOFF](paper-2027/HANDOFF.md)；具体结论以对应记录及其后续纠正为准。

## 项目与代码

| 入口 | 内容 |
| --- | --- |
| [README](README.md) | 项目简介与目录 |
| [AGENTS](AGENTS.md) | 通用工作原则 |
| [scripts](scripts/) | 实验、分析、训练和评测代码 |
| [tests](tests/) | 测试 |
| [环境资料](docs/overview/) | 计算环境、资产及复现资料；按记录日期使用 |

## 论文

| 入口 | 内容 |
| --- | --- |
| [main.tex](paper-2027/main.tex) | 活动论文入口 |
| [修订说明](paper-2027/REVISION_BRIEF.md) | 论文修订背景与要求 |
| [compile.sh](paper-2027/compile.sh) | 论文构建脚本 |

## 研究记录

历史记录用于查证，不构成当前计划或执行指令。

| 入口 | 内容 |
| --- | --- |
| [研究方案与结果](docs/research/) | 按主题和日期保存的协议、结果与分析 |
| [频率分配理论核心](docs/research/ROPE_ALLOCATION_THEORY_CORE_20260910.md) | 非均匀傅里叶频率分配的当前问题、已有数学工具与下一轮KKT推导起点 |
| [30代理归档与已有进展](docs/research/ROPE_ALLOCATION_PROGRESS_20260910.md) | 28份原报告、2份回传整理、代码、证据状态与全部材料入口 |
| [本会话统一推导工作台](analysis/unify_20260910/) | KKT问题陈述（NEXT_DERIVATION_KKT_PROBLEM.md）、权威起点（STARTING_POINT_YARN_VS_MRPRO.md）、32+30代理摘要（digests/、digests_codex/）、地面真值表与四问推导（tables/、answers/）、汇总（INTEGRATION_20260910.md） |
| [失败复盘](docs/research/ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md) | 历史失败、后续纠正及原始证据入口 |
| [用户提示词与阶段纠正](docs/research/USER_INTENT_GUIDE_20260909.md) | 四个任务的95条用户消息、持续原则、后续修正及09_09分支速览 |
| [论文修订交接总档](paper-2027/research/PAPER_REVISION_HANDOFF_20260911.md) | 改稿代理单一入口：四阶段（EVQ/NeurIPS、z-分配主线、MrRoPE 战役、Llama-3）实验与理论资产、旁线与被否决史、术语守卫、数字溯源索引、修订 do/don't |
| [有限窗慢频塌缩 Receipt](paper-2027/research/evidence/FINITE_WINDOW_SLOW_RANK_RECEIPT_20260911.md) | b=256 核心实验 min ωW=1.1892 缺口的精确闭合：最慢 8 对 r₂=2.1147、全退化曲线、谱证据与复算脚本；03_theory.tex 已引用 |
| [论文研究材料](paper-2027/research/) | 理论、实验依据和审查材料 |
| [外部原文](paper-2027/research/external-reviews/) | 作者提供的报告与外部分析 |
| [早期实验](docs/exp/) | 按月份保存的历史报告 |

精简前材料位于只读分支 `main_0726`，可用 `git show main_0726:<path>` 查阅。
