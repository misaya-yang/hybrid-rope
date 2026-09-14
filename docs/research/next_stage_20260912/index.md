# 当前研究：边界、低频终点与中频过渡

**当前状态（2026-09-14）：精确TailSpline是唯一新候选；Qwen2.5-3B统一S2判决运行中。**
当前执行以[TailSpline方法与统一评测合同](TAILSPLINE_ROPE_METHOD_AND_UNIFIED_EVAL_20260914.md)
为准。fixed-u已失败退出；旧mix075、Llama局部修复和非统一S4后继队列不再执行。
冻结checkpoint、零训练、持续GPU接管和禁止无依据参数搜索的边界不变。
交给外部AI分析或转公司PC时使用[独立问题清单](OPEN_QUESTIONS_FOR_EXTERNAL_AI_AND_PC_20260914.md)。
下列旧计划仍不是自动执行队列；本轮授权只覆盖决定性闭环中的零训练固定表实验。

更新：2026-09-13，按作者最新纠正。实验分别回答三个设计问题：高频保持区与可调整区边界如何确定；低频最终缩放到ω/s是否最优；两个端点确定后中频如何过渡。每项需要数学解释、可计算方案和最小验证。问题可分别提出，干预仍需说明交互。

论文主线为 z 发现→EVQ→full-z→基于z变化超过YaRN/MrRoPE的实验现象→更优理论。全窗口质量作为评价方式，不取代三个设计问题。以下早期方向文档和固定表确认队列按此最新目标解释，不要求先做Llama大模型确认再研究变量。继续实验服从目标任务的最新用户指令。

## 按当前任务读取

| 任务 | 入口 |
|---|---|
| 当前唯一方法与GPU主线 | [TailSpline方法与统一评测](TAILSPLINE_ROPE_METHOD_AND_UNIFIED_EVAL_20260914.md)：精确有限网格闭式；先Qwen3B 324行/臂统一对照，再进入PPL、NIAH/passkey、full-13 RULER |
| fixed-u倍率迁移判决 | [OLMo结果owner](OLMO_S8_FIXED_U_TRANSPORT_RESULT_20260914.md)：相对fixed-m的AUC差为负且区间不跨零；终止该分支，不终止整体研究 |
| 历史mix075完整臂 | Qwen2.5-3B S2的mix075已324/324完成并保留raw；因band/gain不同，只作近似开发prior，不与当前统一TailSpline主比较合并 |
| 上一轮最终实验结论与停止点 | [理论与实验阶段报告](THEORY_AND_EXPERIMENT_PAUSE_REPORT_20260914.md)：Qwen累计只确认对BM的AUC胜出并保持Native；MrPro/C42仍未决；当时GPU实验停止 |
| 最新推导：从计算约束到具体构造 | [功能区间、最小改动与跨倍率z迁移](ROPE_FUNCTIONAL_CONSTRAINTS_AND_Z_TRANSPORT_20260914.md)：逐槽排序、条件边界/终点解、唯一保持band内分配的倍率迁移；CPU验证，未跑新模型 |
| 上一轮8×与模型迁移 | [倍率与模型计划](SCALE_MODEL_TRANSFER_PLAN_20260913.md)，历史执行计划；当前以TailSpline统一合同为准 |
| Llama S8冻结规则迁移 | [结果owner](LLAMA_S8_SCALE_TRANSFER_RESULT_20260913.md)：共同32K桥接与64K三对照、Native 8K已完成；未形成匹配区间AUC |
| OLMo S8冻结allocation迁移 | [结果owner](OLMO_S8_SCALE_TRANSFER_RESULT_20260913.md)：相对BM/MrPro的采样网格AUC正结果；父gain在两个测试点中较好；Native 4K未测 |
| Qwen1.5B S2冻结transition迁移 | [完成结果owner](QWEN_S2_MIX075_RANGE_RESULT_20260913.md)：累计18确认AUC超过BM并保持Native；相对MrPro/C42未决 |
| 当前理论缺口与作者纠正 | [已有证明尚未导出当前候选](THEOREM_FIRST_ROPE_DESIGN_20260913.md)：旧margin执行顺序已退出默认计划；完成当前确认，不追加猜表 |
| 早期方向与背景 | [作者方向历史记录](PAPER_INTERVAL_DIRECTION_20260913.md)，以本页最新目标为准 |
| 将强开发候选推进为可信比较 | [固定表确认流水线](../../../experiments/fixed_rope_three_interfaces_20260913/index.md) |
| 本轮同口径结果与负例 | [Llama S=4区间确认、Native反例与三接口判决](LLAMA_S4_RANGE_CONFIRM_AND_INTERFACE_RESULT_20260913.md) |
| 选择评测面板与读出协议 | [分层面板合同](RULER_TIERED_PANEL_CONTRACT_20260913.md) |
| 核实当前区间开发结果 | [固定表区间结果](4080_FIXED_TABLE_RANGE_RESULT_20260913.md)、[band结果](BAND_NEXT_STAGE_RESULT_20260913.md) |
| 核实已进入论文的证据 | [证据索引](../../../paper-2027/research/evidence/index.md) |

已有强开发信号不等于独立确认或全窗口赢家；逐任务、逐长度保留局部代价。早期流水线的“实现就绪/运行未验证”已是历史状态；Sol任务已报告OLMo S=4运行与结果，最新进度以该任务实际回执和当前会话授权为准。

## 按需研究材料

只有构造/比较预测器时才读[区间研究自查](RANGE_OPTIMAL_FIXED_ROPE_SELF_AUDIT_20260913.md)和[replay实现](../../../experiments/checkpoint_attention_replay_20260913/index.md)。它们不阻止已有候选的任务质量确认。

[原计划与历史状态](PLAN_HISTORY_20260913.md)完整保留S1–S4、LoRA优先队列、Agent-range、外部方案与当时回执，仅用于追溯或明确指定的任务。它们不是默认研究清单，也不因离开本页而变成无效结果。当前作者要求优先于旧时点计划。
