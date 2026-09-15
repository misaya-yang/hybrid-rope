# 当前研究入口

上位研究对象是RoPE内部频率分布z。Cosh、TailSpline等是不同构造；原生窗口增强、
外推及窗口内外联合改善是作用形式。现稿的具体构造重心不定义整个研究的范围。

现稿研究频率覆盖范围确定之后，内部配置怎样改变有限窗口的位置结构及模型对频率的使用。
受控干预、完整旋转对几何、解析构造和模型验证构成论证链。exact TailSpline是主要冻结部署
构造，Cosh保留辅助外推及配对学习证据；同一静态表在2L/4L的质量提升是核心实证贡献。
BM、YaRN、旧mix075和Native-Z探索各保留其证据角色。

模型前向只用于验证已经冻结的规则，不用于从权重、激活、Q/K/V、梯度或校准分数中拟合
通用TailSpline构造。Native-Z5是单独标记的checkpoint-calibrated反事实，不改变这一边界。

## 唯一读取顺序

| 需要回答的问题 | Canonical owner |
|---|---|
| 当前有哪些成立、失败、运行中或仅准备好的实验？ | [关键实验罗盘](KEY_EXPERIMENT_COMPASS_20260914.md) |
| TailSpline的定义、控制变量和统一评测合同是什么？ | [方法与评测合同](TAILSPLINE_ROPE_METHOD_AND_UNIFIED_EVAL_20260914.md) |
| Llama classic、clean 16K/32K、Native与Natural-QA的完整结果是什么？ | [Llama结果owner](TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md) |
| OLMo跨模型确认是什么？ | [OLMo结果owner](TAILSPLINE_OLMO_CLASSIC_RESULT_20260914.md) |
| Qwen 32K/64K、NIAH小样本和ProofPile-only PPL说明什么？ | [辅助GPU结果owner](SECONDARY_GPU_RESULTS_20260915.md) |
| Native-Z5究竟成立了什么？ | [Native-Z5结果owner](NATIVE_Z5_EXPLORATION_RESULT_20260915.md) |
| 如何重新研究native增强？ | [给Web Pro的自包含分析提示词](WEB_PRO_NATIVE_Z_ENHANCEMENT_PROMPT_20260915.md) |
| 原生四臂完成后，如何设计零训练增强方法？ | [完整Web Pro提示词：证据与方法交付](WEB_PRO_NATIVE_ZERO_TRAIN_NEXT_STEP_20260915.md) · [接续迁移回复的纠偏提示词](WEB_PRO_NATIVE_METHOD_DELIVERY_CORRECTION_20260915.md) |
| CPU理论核验支持到哪一层？ | [理论深化CPU结果](THEORY_DEEPENING_CPU_VERIFICATION_20260915.md) |
| Web Pro的有限窗口、换基与边界理论是否值得采用？ | [独立CPU核查与采用判断](WEB_PRO_FINITE_WINDOW_AUDIT_20260915.md) |
| 三段式改进做过什么，怎样向Pro追问下一步？ | [详细研究总结](THREE_BAND_RESEARCH_SYNTHESIS_FOR_PRO_20260915.md) · [可直接发送的提示词](WEB_PRO_THREE_BAND_FOLLOWUP_PROMPT_20260915.md) |
| YaRN→MrPro→TailSpline的中频变化有什么可验证的解释？ | [独立理论分析、CPU图表与Pro对比](INDEPENDENT_MIDBAND_THEORY_ANALYSIS_20260915.md) |
| 本轮理论如何形成论文修改？ | [已应用的理论整合与精确增量](../../../paper-2027/research/theory_revision_proposal_20260915/README.md) · [R08审稿](../../../paper-2027/research/pdf-review-rounds/20260915_theory_integration_r08/README.md) |
| 论文已登记证据及来源在哪里？ | [论文证据索引](../../../paper-2027/research/evidence/index.md) |
| 服务器上哪些任务在跑、能跑或需要48GB以上？ | [服务器任务分层](../../../experiments/iclr2027_three_track_sprint_20260915/SERVER_TASK_LAYERS.md) |

## 下一轮强实验计划（尚未执行）

[面向强接收与突出研究评价的实验计划](STRONG_EXPERIMENT_PLAN_20260915.md)：
统一clean跨模型矩阵、自然长文、等位移配置对照与部署取舍；列明已有入口和待实现包装器。
这是后续执行规格，不自动改变当前队列或YaRN停放状态。

[Oral目标的研究判断](../../../paper-2027/research/ICLR2027_ORAL_RESEARCH_STRATEGY_20260915.md)
保留具体实验增强建议；顶部已按作者纠正撤回TailSpline中心定位。操作仍沿用上述计划和执行owner。

## 当前执行边界

- Llama NIAH Full20正式报告已完成，见[720条每臂配对报告](../../../experiments/iclr2027_three_track_sprint_20260915/reports/niah_full20_tailspline_vs_mrpro.json)；后续执行以服务器任务分层为准。
- Llama S16 128K gate的CPU资产已冻结，但GPU尚未执行，且入口要求至少45,000 MiB显存。
- YaRN代码处于停放状态；没有明确推进决定时不自动运行。
- Native-Z5的V1、consensus和all-50 refit均已结束；现有结果不支持继续复用同一确认集调表。
- fixed-u、proxy选表、曲线系数/band/gain追调及旧队列均已退出当前路线。

计划、CPU恒等式、开发proxy和真实模型结果必须分别标记；代码准备不等于GPU完成，报告摘要
不等于raw-row复核。当前数值只在各结果owner中维护，其他文档通过链接引用，避免多处复制后
发生漂移。

追溯旧计划、历史候选或外部模型讨论时使用[历史研究目录](CATALOG_20260914.md)，不把它作为
默认上下文。论文修改、编译与评审另走[论文入口](../../../paper-2027/index.md)。
