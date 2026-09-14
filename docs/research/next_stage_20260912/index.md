# 当前研究入口

论文主线：z发现 → EVQ → full-z → 基于z变化超过YaRN/MrRoPE的实验现象 → 更优理论。
三个设计问题是高频保持边界、低频缩放终点与中频过渡；全窗口质量用于评价。

**2026-09-14：精确TailSpline是唯一新候选，后续统一评测使用Llama-3-8B。**
方法、对照、交付与停止条件以[TailSpline方法与统一评测合同](TAILSPLINE_ROPE_METHOD_AND_UNIFIED_EVAL_20260914.md)为准。
合同记录CPU核验完成，尚无精确TailSpline性能结论；当前Llama 72行块只作8K/32K
配对诊断，主判决仍是PPL、完整NIAH/passkey和Full-13；实时进度需查实际运行回执。
既有零训练与冻结checkpoint边界见合同，执行沿用目标任务的有效授权。

| 任务 | 入口 |
|---|---|
| 当前实现与结果定位 | [实验流水线](../../../experiments/fixed_rope_three_interfaces_20260913/index.md) |
| Llama经典资产审计与正确主合同 | [资产审计](LLAMA_CLASSIC_ASSET_AUDIT_20260914.md) |
| 核实论文已有证据 | [证据索引](../../../paper-2027/research/evidence/index.md) |
| Web Pro理论与论文组织讨论 | [十个研究问题与自包含背景](WEB_PRO_TEN_RESEARCH_QUESTIONS_20260914.md) |
| 外部AI分析或公司PC交接 | [独立问题清单](OPEN_QUESTIONS_FOR_EXTERNAL_AI_AND_PC_20260914.md) |

fixed-u已退出；旧mix075、局部修复和Qwen后继队列不再执行。旧结果仍按原证据范围有效，不能充作精确TailSpline统一比较。
追溯此前推导、迁移结果或旧计划时使用[历史研究目录](CATALOG_20260914.md)，无需作为当前任务前置阅读。
