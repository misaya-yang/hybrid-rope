# 当前研究入口

上位对象为RoPE内部配置z：固定实际范围与旋转预算后，内部位置仍可影响原生与扩展窗口质量。
TailSpline、NCP、Cosh分别承担冻结扩展、原生增强和学习/外推支持，三者不冒充同一个任务最优解。
构造只使用各自声明的公开参数；不从模型权重、激活、Q/K/V、梯度或输出中拟合通用规则。

**当前阶段：**[下一版论文准备](PAPER_NEXT_REVISION_PREPARATION_20260916.md)。
本轮更新概念、证据和导航，等待更多实验后统一改稿；现稿9/29页暂不修改。
[Pro采用判断](../reviews/PRO_REASSESSMENT_DISPOSITION_20260916.md)和
[十篇审稿经验](../reviews/TEN_PAPER_REVIEW_LESSONS_20260916.md)是编辑依据，不是自动执行指令。

已入稿的Llama/OLMo、等位移、NCP和学习证据见[论文索引](../../../paper-2027/index.md)。
新完成的Qwen/GLM S4三臂Full-13、自然QA、Qwen S8单针及Llama S16结果见
[本批结果owner](PRO6000_EXTREME_NATURAL_QA_RESULTS_20260916.md)，尚待统一入稿。
执行完成状态见[带时间戳执行快照](../../../experiments/iclr2027_three_track_sprint_20260915/SERVER_TASK_LAYERS.md)。

## 唯一读取顺序

**2026-09-15：clean Llama 32K Full RULER-13×200确认完成。** TailSpline/MrPro为
`68.27/56.54%`，差`+11.72pp`，95%区间`[+10.32,+13.11]pp`；12/13任务与四个
family为正，输出健康更好。Natural-QA631用于真实输出迁移，Cosh保留学习期证据。
具体协议、局部反转与分数边界以结果owner为准。

Native窗口探索是单独的checkpoint-calibrated反事实：冻结成熟OLMo-2-1B权重与Native频率
support，仅用五个有效自由度校准interior `z`，不作为目标无关的解析曲线，也不改变
TailSpline主线优先级。

| 需要回答的问题 | Canonical owner |
|---|---|
| 下一版怎样吸收Pro、新结果和审稿经验？ | [下一版准备](PAPER_NEXT_REVISION_PREPARATION_20260916.md) |
| 当前有哪些成立、失败、运行中或仅准备好的实验？ | [关键实验罗盘](KEY_EXPERIMENT_COMPASS_20260914.md) |
| TailSpline定义、控制变量和统一评测合同是什么？ | [方法与评测合同](TAILSPLINE_ROPE_METHOD_AND_UNIFIED_EVAL_20260914.md) |
| Llama classic、clean 16K/32K与Natural-QA的完整结果是什么？ | [Llama结果owner](TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md) |
| OLMo跨模型确认是什么？ | [OLMo结果owner](TAILSPLINE_OLMO_CLASSIC_RESULT_20260914.md) |
| Qwen、NIAH小样本与PPL补充结果说明什么？ | [辅助GPU结果owner](SECONDARY_GPU_RESULTS_20260915.md) |
| Native-Z5结果支持到哪一步？ | [Native-Z5结果owner](NATIVE_Z5_EXPLORATION_RESULT_20260915.md)；[预注册](NATIVE_Z5_ENHANCEMENT_PREREG_20260914.md) |
| Qwen 32K/64K、NIAH小样本和ProofPile-only PPL说明什么？ | [辅助GPU结果owner](SECONDARY_GPU_RESULTS_20260915.md) |
| Pro6000上的128K/256K与自然长文结果是什么？ | [极限长度与自然长文结果](PRO6000_EXTREME_NATURAL_QA_RESULTS_20260916.md) |
| Native-Z5究竟成立了什么？ | [Native-Z5结果owner](NATIVE_Z5_EXPLORATION_RESULT_20260915.md) |
| 如何重新研究native增强？ | [给Web Pro的自包含分析提示词](WEB_PRO_NATIVE_Z_ENHANCEMENT_PROMPT_20260915.md) |
| 原生四臂完成后，如何设计零训练增强方法？ | [完整Web Pro提示词：证据与方法交付](WEB_PRO_NATIVE_ZERO_TRAIN_NEXT_STEP_20260915.md) · [接续迁移回复的纠偏提示词](WEB_PRO_NATIVE_METHOD_DELIVERY_CORRECTION_20260915.md) |
| base如何改变z的作用，怎样向Pro追问？ | [自包含分析提示词](WEB_PRO_BASE_ALLOCATION_PROMPT_20260916.md) |
| CPU理论核验支持到哪一层？ | [理论深化CPU结果](THEORY_DEEPENING_CPU_VERIFICATION_20260915.md) |
| Web Pro的有限窗口、换基与边界理论是否值得采用？ | [独立CPU核查与采用判断](WEB_PRO_FINITE_WINDOW_AUDIT_20260915.md) |
| 三段式改进做过什么，怎样向Pro追问下一步？ | [详细研究总结](THREE_BAND_RESEARCH_SYNTHESIS_FOR_PRO_20260915.md) · [可直接发送的提示词](WEB_PRO_THREE_BAND_FOLLOWUP_PROMPT_20260915.md) |
| YaRN→MrPro→TailSpline的中频变化有什么可验证的解释？ | [独立理论分析、CPU图表与Pro对比](INDEPENDENT_MIDBAND_THEORY_ANALYSIS_20260915.md) |
| 本轮理论如何形成论文修改？ | [已应用的理论整合与精确增量](../../../paper-2027/research/theory_revision_proposal_20260915/README.md) · [R08审稿](../../../paper-2027/research/pdf-review-rounds/20260915_theory_integration_r08/README.md) |
| 服务器上哪些任务在跑、能跑或需要48GB以上？ | [服务器任务分层](../../../experiments/iclr2027_three_track_sprint_20260915/SERVER_TASK_LAYERS.md) |
| 论文已登记证据及来源在哪里？ | [论文证据索引](../../../paper-2027/research/evidence/index.md) |
| YaRN–MrPro理论对照的决定是什么？ | [等剂量单交叉审计](MRROPE_YARN_EQUAL_DOSE_PRINCIPLE_AUDIT_20260914.md) |
| 外部AI分析或公司PC交接材料在哪里？ | [独立问题清单](OPEN_QUESTIONS_FOR_EXTERNAL_AI_AND_PC_20260914.md) |
| 当前实现与Llama资产定位在哪里？ | [实验流水线](../../../experiments/fixed_rope_three_interfaces_20260913/index.md)；[资产审计](LLAMA_CLASSIC_ASSET_AUDIT_20260914.md) |

## 准备材料与历史规格

当前优先级由[下一版准备](PAPER_NEXT_REVISION_PREPARATION_20260916.md)统领：回收已在执行的
GLM和直接基线报告，准备NCP独立确认及必要的YaRN单臂，机制面板作为有明确问题的增强。

- [Native/oral准备包](NATIVE_ORAL_PREPARATION_PLAN_20260915.md)保留通用构表、288题反事实、
  参考数学和吞吐资产；其中旧“尚无NCP结果”的时间状态已更新。
- [强实验规格](STRONG_EXPERIMENT_PLAN_20260915.md)保留X1–X8实施细节，不能把其旧待跑列表当现状。
- [方法谱系与native审查](../reviews/NATIVE_BENEFIT_AND_METHOD_LINEAGE_REVIEW_20260915.md)保留历史比较与gain控制。
- [Oral研究判断](../../../paper-2027/research/ICLR2027_ORAL_RESEARCH_STRATEGY_20260915.md)为历史策略参考，
  不排除NCP的核心地位，也不自动要求重跑已有实验。

## 当前执行边界

- Llama NIAH Full20正式报告已完成，见[720条每臂配对报告](../../../experiments/iclr2027_three_track_sprint_20260915/reports/niah_full20_tailspline_vs_mrpro.json)；后续执行以服务器任务分层为准。
- Llama S16 128K gate、自然长文压力测试、Qwen 256K健康检查与Qwen S4/128K En.QA均已完成，见[结果owner](PRO6000_EXTREME_NATURAL_QA_RESULTS_20260916.md)。
- Qwen与GLM的128K三臂Full-13和长书QA已完成；Llama/OLMo的NIAH-8×200、
  PPL46、Natural-QA631与Full-13×10官方静态YaRN三臂也已完成，见
  [便携报告索引](../../../experiments/iclr2027_strong_evidence_20260915/reports/README.md)。
- Native-Z5的V1、consensus和all-50 refit均已结束；现有结果不支持继续复用同一确认集调表。
- fixed-u、proxy选表、曲线系数/band/gain追调及旧队列均已退出当前路线。

计划、CPU恒等式、开发proxy和真实模型结果必须分别标记；代码准备不等于GPU完成，报告摘要
不等于raw-row复核。当前数值只在各结果owner中维护，其他文档通过链接引用，避免多处复制后
发生漂移。

追溯旧计划、历史候选或外部模型讨论时使用[历史研究目录](CATALOG_20260914.md)，不把它作为
默认上下文。论文修改、编译与评审另走[论文入口](../../../paper-2027/index.md)。
