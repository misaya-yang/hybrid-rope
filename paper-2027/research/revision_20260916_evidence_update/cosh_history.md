# Cosh历史回溯：研究本体与归因控制

独立历史代理与主代理分别核查Git、实验owner及旧稿，未运行模型。

## 核心结论

Cosh从研究开始就是标准RoPE算子下的闭式频率配置，学习后的外推是核心问题。
无新增可学习位置参数不等于模型没有训练。研究包括零训练安装、小模型训练、
更大模型同配方训练、成熟模型LoRA及随后改进监督的下游任务。
151.9M固定端点是同一Cosh系列的归因控制，不是新的FMRoPE方法比较或零训练项目。

## 原始审稿与直接回复

- 正式三分review：`7754486:rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`，
  reviewer27bE，rating3/confidence4。L291概括为“allocating spectral resources throughout optimization”；
  L312承认50M–750M训练；L316要求“keep the positional operator unchanged and vary only the fixed frequency schedules”。
- 直接答复：`bc59ff2:rebuttal/rebuttal_0723/paste/REVIEWER_27bE.md`，§4写
  “we ran the experiment you specified”，随后给出固定解析配置、native grid/RMS匹配、exact-range三层控制。
  该回复的Level3具体为50.9M factorial；151.9M固定端点在同期另一owner中，是同一归因路线的另一实验。
- FMRoPE novelty/retargeting问题还来自另一个reviewer与AC。不能因为历史目录名含fmrope，
  就把Cosh训练项目改写成FMRoPE比较。稿件保留实际base及端点条件，不写审稿历史。

## 证据链

| 阶段 | 已核实证据 | 当前作用 |
|---|---|---|
| 约50M早期训练 | `8616af4:docs/paperdraft/phase6_report.md:120–140`，8K Geo/Cosh PPL540.4/414.2 | 说明研究起点早于固定端点控制 |
| 50.9M factorial | 当前附录C.4，两base×两长度×三head，三seed | 强度、结构条件与替代配置 |
| 151.9M固定端点 | 当前附录C.2，三seed×约500M tokens；扩展NLL均改善 | 排除仅频率范围改变的解释 |
| 432M MLA | 16 rotary pairs，500M tokens，三seed，16K138.8→95.6 | 有限旋转预算的训练与外推 |
| 750M续训 | 16K45.1→24.4，8K answer-token exact0/40→31/40 | 共同起点继续学习与生成读出 |
| 1.485B OLMo | [完整owner](../../../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md)，同public step0、scientific recipe、数据前缀与2.097B tokens；16K182.73→159.64 | 更大规模训练的同类外推趋势 |
| Llama8B LoRA | `bc59ff2:rebuttal/rebuttal_0723/theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md`，300步匹配LoRA，16K108.958→24.068，32K991.475→127.911 | 成熟模型PPL收益 |
| OLMo后续监督 | 当前附录F：8K完整answer+EOS18→98%，16K0→60% | 改进监督后的真实生成收益 |
| OLMo选择性Q/K | `bc59ff2:rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md`，8K 2Wiki F10.07→21.48 | 独立适配配方下的任务推进 |

1.485B的完整结果位于MD owner，不能拿sibling native-only JSON否定已经完成的比较。
Llama和OLMo不同adapter按各自配置呈现，不把PPL和后来的任务结果拼成同一checkpoint。
同类native/extension取舍是反复观察到的趋势，不是所有配置必须遵循的性能守恒定律。

零训练Cosh也有直接owner：
[固定表安装](../attention-aware-retrofit/results/zero-training-deployment/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md)。
该候选同时有外推收益与native代价；不据此否定训练系列，也不写“Cosh从未做过零训练”。

## 本版修复

- 151.9M段落明确位于Cosh训练系列，去掉FMRoPE方法对比标签。
- 主文保留432M曲线和750M结果，补充1.485B、LlamaLoRA、OLMo任务推进的简短连接。
- 已有附录F恢复这些相关规模/适配结果，配方、评价和native结果清楚区分。
- 相关工作突出MrRoPE/YaRN和LeRoPE；所有公共参数构造都不依赖checkpoint观测。
