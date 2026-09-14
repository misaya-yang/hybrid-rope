# 当前研究入口

论文主线：z发现 → EVQ → full-z → 基于z变化超过YaRN/MrRoPE的实验现象 → 更优理论。
三个设计问题是高频保持边界、低频缩放终点与中频过渡；全窗口质量用于评价。

理论目标：从RoPE、注意力结构或非均匀傅里叶建模出发，提出不依赖权重/激活的通用构造；公开base、K、L、S可用于计算。模型前向用于验证，不用于拟合选表。详见[理论问题与完成标准](THEOREM_FIRST_ROPE_DESIGN_20260913.md)。

**2026-09-14：精确TailSpline是唯一新候选；Llama-3-8B与OLMo-2-1B的统一两臂判决
均已完成，并在三个预注册family endpoint上胜MrPro。停止新曲线搜索。**
方法、对照、交付与停止条件以[TailSpline方法与统一评测合同](TAILSPLINE_ROPE_METHOD_AND_UNIFIED_EVAL_20260914.md)为准。
Llama结果见[经典两臂结果](TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md)：Full-13 AUC差
`+3.20pp`且95%区间为`[+0.65,+5.79]pp`。OLMo前瞻确认见
[跨模型结果](TAILSPLINE_OLMO_CLASSIC_RESULT_20260914.md)：Full-13差`+49.23pp`，13任务
AUC差全部为正；两模型均3/3方向通过。YaRN/BM与机制实验后置，不自动启动。
既有零训练与冻结checkpoint边界见合同，执行沿用目标任务的有效授权。

| 任务 | 入口 |
|---|---|
| 当前实现与结果定位 | [实验流水线](../../../experiments/fixed_rope_three_interfaces_20260913/index.md) |
| Llama数据、样本量与执行实现 | [资产审计](LLAMA_CLASSIC_ASSET_AUDIT_20260914.md) |
| Llama TailSpline–MrPro主结果 | [经典两臂结果](TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md)：Full-13/NIAH/PPL 3/3方向通过，局部反转与证据边界完整保留 |
| OLMo TailSpline–MrPro跨模型确认 | [经典两臂结果](TAILSPLINE_OLMO_CLASSIC_RESULT_20260914.md)：Full-13/NIAH/PPL 3/3方向通过，13任务AUC差全部为正 |
| YaRN–MrPro理论对照 | [等剂量单交叉后移审计](MRROPE_YARN_EQUAL_DOSE_PRINCIPLE_AUDIT_20260914.md)：CPU闭式已核验，YaRN按作者要求后置 |
| 核实论文已有证据 | [证据索引](../../../paper-2027/research/evidence/index.md) |
| Web Pro理论与论文组织讨论 | [十个研究问题与自包含背景](WEB_PRO_TEN_RESEARCH_QUESTIONS_20260914.md) |
| Web Pro终审后下一问 | [自包含长提示词模板](WEB_PRO_POST_TAILSPLINE_MRPRO_PROMPT_TEMPLATE_20260914.md)：完成后只替换真实结果区，不依赖GitHub完整检索 |
| 外部AI分析或公司PC交接 | [独立问题清单](OPEN_QUESTIONS_FOR_EXTERNAL_AI_AND_PC_20260914.md) |

fixed-u已退出；旧mix075、局部修复和Qwen后继队列不再执行。旧结果仍按原证据范围有效，不能充作精确TailSpline统一比较。
追溯此前推导、迁移结果或旧计划时使用[历史研究目录](CATALOG_20260914.md)，无需作为当前任务前置阅读。
