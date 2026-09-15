# Hybrid-RoPE关键实验罗盘

更新：2026-09-15。本文只回答三件事：哪些结果可以进入论文、哪些只提供支持或反例、
哪些尚在运行或仅准备好。计划、CPU恒等式、开发proxy和真实模型结果严格分开。

## 当前结论

论文一级贡献是**固定端点与支持下，RoPE内部allocation仍是可识别、可设计并参与学习的
变量**。零训练是核心落地，exact TailSpline是当前唯一重点展开的冻结部署构造；Cosh保留
学习期证据，不把Cosh、BM和TailSpline写成并列方法合集。

已经成立的最强方法结论是：TailSpline在匹配的Llama-3-8B S4合同下，于16K和32K均
显著胜MrPro；相同构造方向也在OLMo-2-1B S4经典合同上成立。尚未成立的是：自然QA
优势、Native增强、one-sided边界机制、YaRN胜负和128K外推。

## 一、论文主结果

这些是当前可直接承担方法结论的结果。同一owner承载多个匹配合同；本页只保留判决数字，
不复制任务breakdown、raw hash或运行日志。

| 结果 | 最少关键数字 | 论文价值与边界 | Canonical owner |
|---|---|---|---|
| **Llama clean 32K Full RULER-13×200** | TailSpline/MrPro `68.27/56.54%`，差**`+11.72pp`**，95%区间**`[+10.32,+13.11]pp`**；12/13任务和四family为正 | 当前hero：2,600个source-order、unpadded、同prompt配对样本证明32K整体胜MrPro；它是单长度端点，不称多长度AUC，也不识别机制 | [Llama结果owner](TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md#6-clean-32k大样本确认) |
| **Llama clean 16K Full RULER-13×50** | `86.09/82.71%`，差**`+3.39pp`**，95%区间`[+1.53,+5.34]pp`；QA `+8pp`，区间`[+1,+15]pp` | 650对样本确认同一S4构造在2L也有任务收益；与32K共同排除“只在极限端点有效”，但不代表连续全窗口 | [Llama结果owner](TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md#9-clean16k-intermediate-length-result-2026-09-15) |
| **Llama classic 8/16/32K** | Full-13 log-length AUC `0.7880/0.7560`，差`+3.20pp`，区间`[+0.65,+5.79]pp`；NIAH `+3.75pp`；PPL AUC差`−0.00449` | 完整多长度与PPL健康曲线；PPL收益由32K驱动，8/16K的PPL略差，保留局部负格；不再把它当hero | [Llama结果owner](TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md#2-三个主终点) |
| **OLMo classic 4/8/16K** | Full-13 AUC差**`+49.23pp`**，NIAH `+64.45pp`，PPL AUC差`−3.853`；13任务差全部为正 | 第二模型族前瞻确认，支持两个checkpoint family上的方法性；历史OLMo方向先验、较高cap率和总剂量混杂均保留 | [OLMo结果owner](TAILSPLINE_OLMO_CLASSIC_RESULT_20260914.md) |

## 二、支持证据

这些结果加强allocation论文骨架、检查副作用或提供真实任务背景，但不能单独升级成
TailSpline主胜负。

| 证据 | 最少关键数字/发现 | 正确用途 | Canonical owner |
|---|---|---|---|
| **Natural-QA631** | 五任务T/P `41.08/40.88%` F1，差`+0.20pp`，文档簇区间`[−1.53,+1.89]pp`；Native外315条差`−0.94pp`，区间跨0 | 真实输入上排序未决；说明clean RULER优势尚未迁移成确定QA优势，不写“等效”或“胜出” | [Llama结果owner](TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md#8-natural-qa631-v2完整结果2026-09-15) |
| **Native 8K任务与PPL参照** | RULER Native/T/P `91.88/89.74/87.50%`，Native−TailSpline区间跨0；PPL `5.2759/5.2953/5.2864`，TailSpline比Native高约`0.37%` | 支持“原生任务未见确定损伤”，同时量化真实PPL代价；不能写成无损或非劣 | [Llama结果owner](TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md#7-本轮论文整合与e1资格复核) |
| **ProofPile-only PPL32** | 32文档8/16/32K PPL-AUC T−P约`+0.00006`，区间`[−0.00181,+0.00192]` | 对齐MrRoPE所用语料类型时两法总体基本持平；与ProofPile+PG19主PPL口径分开 | [分析实现](../../../experiments/iclr2027_three_track_sprint_20260915/proofpile_ppl_parity.py)；服务器报告`tailspline_llama_s4_mrrope_niah_heatmap/reports/proofpile32_ppl_curve.json` |
| **A01 固定支持151.9M三seed** | 只改30个内点，三seed在2×/4×/8× tail NLL同向改善；256窗内约`+2.65%` PPL代价 | allocation在端点固定后仍有因果效应，并存在窗内—窗外交换；不证明成熟checkpoint或Cosh普适最优 | [A01证据owner](../../../paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) |
| **A09学习期、A12完整读出、A14自然QA反例** | 稀缺rotary budget下EVQ三seed优于GEO；OLMo完整答案+EOS约`18→98%`（8K）、`0→60%`（16K）；BM/MrPro自然QA `25.44/21.62%` | 分别支持学习期价值、读出可恢复和“MrPro并非处处最优”；三种协议互不拼分数 | [论文证据索引](../../../paper-2027/research/evidence/index.md) |
| **A16 C42/C42V24开发受控对** | 同支持、band、总位移与质心，350行开发RULER差`+10.73pp`，16文档NLL差`−0.1109` | 反驳“总剂量/质心足以描述allocation”；仍是开发发现，不是独立泛化确认 | [开发结果owner](../../../ds_workspace/recon_20260910/verdicts/HEADLINE_20260911.md) |

## 三、负结果与未关闭问题

负点估计、不显著结果和确认门失败必须保留，但不把“未确认”偷换成“已否决”。

| 路线/实验 | 已有结果 | 当前判决 | Owner/来源 |
|---|---|---|---|
| **Native-Z5原生增强** | V1在46文档4K NLL为`−0.002109`，区间`[−0.004207,−0.000148]`；RULER `+1.68pp`、Natural-QA `+1.98pp`但区间均跨0。consensus-plus的4K NLL为`−0.000631`且区间跨0；all50 refit为`−0.001543`且区间跨0，晋级门失败 | 只支持“一个checkpoint上存在post-hoc z-only NLL改进”的初步证据；**稳健原生增强与任务增强均未关闭**，后续两种改法没有超过V1 | [Native-Z5结果owner](NATIVE_Z5_EXPLORATION_RESULT_20260915.md)；服务器raw位于`olmo_native_z5_enhancement/` |
| **Llama NIAH深度小样本** | 4长度×9深度×3重复，T/P `88.89/92.59%`，差`−3.70pp`，区间`[−7.41,0]pp`；24K持平，32K均100%，额外失败集中8/16K | 不显著负点估计且长端天花板，既不证明MrPro胜，也不支持TailSpline胜；只用于定位可能的短端retrieval代价 | [报告实现](../../../experiments/iclr2027_three_track_sprint_20260915/mrrope_niah_heatmap_report.py)；服务器报告`tailspline_llama_s4_mrrope_niah_heatmap/reports/tailspline_vs_mrpro_niah_heatmap.json` |
| **C2紧凑profile** | Qwen 64/128K保留部分64维transport行为，但Native 32K由`0.82`降至`0.7125`，注册双门失败 | 低维描述可行，不是合格统一部署法；停止调C2 | [A15证据owner](../../../paper-2027/research/attention-aware-retrofit/evidence/LOW_DIM_COUPLING_GPU_RECEIPT_20260901.json) |
| **fixed-u倍率迁移** | OLMo S4→S8同prompt下显著差于fixed-m | 当前迁移规则已否决，不换模型/倍率继续救 | [A36结果owner](OLMO_S8_FIXED_U_TRANSPORT_RESULT_20260914.md) |
| **proxy选表与继续调曲线** | OOD/SEP、Fisher、局部margin、coherence及开发赢家多次与完整任务反转 | proxy仅可事后解释；不再据此调系数、band、gain、lambda或新曲线 | [当前理论完成标准](THEOREM_FIRST_ROPE_DESIGN_20260913.md) |

尚未关闭但当前不自动启动：TailSpline相对YaRN的实用胜负；TailSpline与MrPro之间
`sum(m)`、early transport和tail landing的机制解混；自然QA稳定收益；跨更多checkpoint的
普适性。

## 四、CPU数学与理论证据

CPU结果只验证定义、恒等式和条件唯一性，不作为GPU任务胜负。

| CPU结果 | 已核验内容 | 结论边界 | Canonical owner |
|---|---|---|---|
| **A37 TailSpline有限网格构造** | one-sided roughness的唯一解为TailSpline，symmetric版本唯一解为BM；exact表与mix075近似关系已核验 | 证明“给定边界目标后的解”，不能证明为什么任务应选择one-sided边界 | [方法与CPU合同](TAILSPLINE_ROPE_METHOD_AND_UNIFIED_EVAL_20260914.md) |
| **A38 YaRN–MrPro等剂量审计** | Llama `n=17`时唯一`S*=7.5132428221`、等总log位移且单交叉 | 给出可做的零剂量差对照；没有模型分数，Fast/Slow原四格也不是等剂量析因 | [CPU审计owner](MRROPE_YARN_EQUAL_DOSE_PRINCIPLE_AUDIT_20260914.md) |
| **A42理论深化** | `z→相位logit→key竞争→value读出`等15类算子/数值检查全部通过 | 支持条件计算路径、gain可吸收边界和T−C边际配对；不支持实证中介、任务排序或通用最优表 | [15类CPU核验owner](THEORY_DEEPENING_CPU_VERIFICATION_20260915.md) |
| **A03–A06几何与共适应** | 完整sin/cos位置基、有限窗方向、同谱置换与权重×表crossing成立 | 区分“几何供给”与“checkpoint学会使用”；不能用静态几何量直接筛赢家 | [论文证据索引](../../../paper-2027/research/evidence/index.md) |

## 五、正在运行与仅已准备

这一节是易变的执行状态，不是论文结果；完成后必须先生成正式报告，再移入前述类别。

| 状态 | 实验合同 | 现在能说什么 | 执行入口 |
|---|---|---|---|
| **RUNNING：Llama NIAH Full20** | 8/16/24/32K×9 depths×20全新重复=`720 prompts/arm`，TailSpline→MrPro，batch 1 | 资产已冻结、队列在运行；**当前没有可引用分数**，不能把中途正确率写入论文 | [Full20 launcher](../../../experiments/iclr2027_three_track_sprint_20260915/run_mrrope_niah_heatmap_full20.sh)；服务器根`tailspline_llama_s4_mrrope_niah_heatmap_full20/` |
| **READY ONLY：Llama S16 128K gate** | 128K Full-13×10=`130 prompts/arm`＋ProofPile10 PPL；TailSpline/MrPro，band `[18,35]`，gain `1.2772588722` | CPU资产与两张表已准备；`gpu_execution=false`，**没有128K模型结果**；入口要求至少48GB显存并在目标机先选prefill策略 | [48GB入口](../../../experiments/iclr2027_three_track_sprint_20260915/run_llama_s16_128k_gate_48gb.sh)；服务器`tailspline_llama_s16_128k_gate/assets/ready.json` |
| **PREPARED、未排队：YaRN** | Natural-QA与classic launcher存在 | 代码准备不等于基线结果；按作者决定后置 | [YaRN launcher](../../../experiments/iclr2027_three_track_sprint_20260915/run_naturalqa_yarn.sh) |

## 六、后续判决顺序

1. 完成Full20并判定小样本NIAH负点估计是否消失；只更新同一NIAH结果owner，不新建
   重复报告。
2. 需要48GB/96GB服务器时直接运行已冻结S16 gate；先报告128K Full-13与PPL两臂，
   不在揭盲后改band、gain或任务子集。
3. Native-Z5若继续，只能使用新确认数据回答稳健性；不能继续复用已经看过的PPL46、
   RULER130和Natural-QA99来宣称确认。
4. YaRN与机制实验不会因代码存在而自动成为下一项；必须由论文缺口和明确授权触发。

## 导航

- [当前研究入口](index.md)
- [实验执行目录](../../../experiments/iclr2027_three_track_sprint_20260915/README.md)
- [论文证据索引](../../../paper-2027/research/evidence/index.md)
- [TailSpline方法与统一评测合同](TAILSPLINE_ROPE_METHOD_AND_UNIFIED_EVAL_20260914.md)
