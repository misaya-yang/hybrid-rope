# 当前状态、文档修正与论文修改必要性

日期：2026-09-15。范围：当前工作区、R07最终稿、实验结果owner、证据登记和服务器只读快照。
授权：修正文档；论文仅分析。论文正文、图表、图源、PDF、标题摘要与源码包均未改写。

## 1. 结论

**仓库的分层方向正确，主要问题是修订后的语义同步；现稿也没有理由再整体重搭。**
本轮修正了旧队列指令、图表定位、证据ID、评价文件入口和完成状态用词。
当前最短阅读路径是[关键实验罗盘](../../../docs/research/next_stage_20260912/KEY_EXPERIMENT_COMPASS_20260914.md)
看结果，再由[证据索引](../evidence/index.md)的“五组”表按论文问题追溯资产。

我收回将上一份方案的整套叙事重排、补图和新增实验视作现稿必改项的判断。
其中一些只是理想写法，不能据此认定现稿存在缺陷。R07已经具备清楚的研究对象、受控识别、
位置结构分析、解析构造和模型验证。现有证据具有支撑有竞争力接收评价的实质内容；继续提升
评审判断，应聚焦读者能否理解新增知识及其证据，而不是累积审稿轮次或继续堆砌限制说明。

### 核查的版本

- [当前PDF](../../main.pdf)：63页，科学正文9页；SHA256
  `649fdb68e7b39d2bcbd986db5263c2fda4cc8c10326a3377d4600faf59a2058f`。
- 该PDF与上一份优化方案最后审读的R07最终PDF字节相同。本轮新增变化主要在仓库整理，
  不能把旧输入的评论重新当成此稿的新问题。
- [R07处置](../pdf-review-rounds/20260915_astra_sol_five_rounds/r07/disposition.md)
  记录了Astra四视角7/7/6/7、AC7，Sol四视角8/8/7/8、AC8。
  这是两个模型对同轮冻结输入的模拟评价，不是八位独立审稿人，也不是最终改后PDF的新评分。
- 本次检查覆盖当前主文相关源码、关键附录定位、已登记报告与索引；不是对全部历史生成流的
  重评分，也不声称完成了新的独立理论审稿。

## 2. 文档整理：哪些正确，哪些已修

### 保留的结构

根索引分到研究、论文、实验和证据；当前研究索引分到方法合同、Llama、OLMo、辅助GPU、
Native-Z5、CPU理论和服务器状态。这种“问题 → 结果owner → 报告/原始来源”的结构可以保留。
运行脚本、raw目录与旧回执具有路径依赖，保持原位置。历史计划本身不是当前失败或待执行任务。

### 已完成修正

| 问题 | 对阅读或执行的影响 | 本轮处理 |
|---|---|---|
| 根/研究/论文入口对主线的摘要不一致，部分仍写旧EVQ→full-z阶段路线 | 容易把历史研究过程当现稿结构 | 按现稿三个发现概括：配置与范围、位置结构与学得使用、解析构造的任务收益；不改实际研究队列 |
| sprint README顶部退役旧队列，后文却要求启动它们 | 操作者可能重开已完成或停放任务 | 改成当前Full20、S16、parked YaRN路由；原wrapper保留为历史 |
| “Natural-QA631在正文表2” | 读者打开表2看到另一实验 | 改为§6.1段落及附录H.9/表41；表2与Fig.4明确对应clean16K/32K |
| 当前claim map以A13支持同支持冻结识别 | A13实际是selective-QK自然QA | 改为A08；旧图表安放表明确标成历史 |
| A43–A46缺人工索引，A41散落页尾 | 最新控制、Natural-QA和16K确认不易发现 | 补齐A01–A46的表格入口，增加按科学问题分组的阅读表 |
| registry和人工表保留旧§5/§6及图号 | 找到文件却对不上现稿 | 对照当前LaTeX标签修正36项定位，包含TailSpline Fig.3、clean Fig.4、Cosh Fig.5 |
| MLA只指向ignored的results路径 | 容易误以为评价文件不可移植或需要重跑 | 加入已tracked的原评价JSON镜像，核对其与旧来源SHA一致；保留旧路径身份 |
| 人工索引仍残留MLA/slot历史材料缺失断言 | 超出实际核查结论 | 改为真实训练/评价配方和证据用途；不推定不存在的记录或污染问题 |
| 辅助NIAH/PPL入口直达分析代码 | 要读实现才能知道结果 | 改为辅助GPU结果owner，保留其中的报告路径与哈希 |
| classic和clean行数写法混淆 | 可能误读classic也有2600行 | 分别写清classic 390生成/138 LM每臂、clean32K 2600生成每臂 |
| 把总位移写成完整T/P配置比较的外部混杂 | 不恰当地削弱有效结果 | 与现稿一致：D是配置统计量；完整配置收益成立，细形状归因由额外对照回答 |
| Full20尚未完成却称pilot已被替代，目标写成负点估计是否消失 | 容易丢弃不利的已完成证据 | 保留pilot与独立Full20，完整后按差值及不确定性判读 |
| 只检查报告存在/PID存活的helper输出COMPLETE/running | 观测被误当成完整性验证 | 本地helper改成PRESENT及明确的marker/PID标签，执行判断仍按服务器owner；未部署远端 |
| 旧优化方案仍像一整套待办 | 可能无必要地再次大改论文 | 加入必要性复核说明，以本页取舍为准 |

所有数据值、原始路径身份和已有来源哈希保留。MLA新增的是既有评价JSON的导航，
不是新增训练数据、恢复运行日志或补造模型结果。

## 3. 核心实验能否一目了然

**可以，前提是先按论证职责读，再展开全部资产。** 下面是本次审计时的紧凑读数；
后续更新回到各结果owner，不将本页作为另一份动态数值总账。T/P分别表示TailSpline/MrPro。

| 论证职责 | 核心实验及结果 | 能回答什么 | 来源 |
|---|---|---|---|
| 识别内部配置作用 | 151.9M三seed，固定端点只改30内点；2×/4×/8× tail NLL差−0.281/−0.176/−0.146；native +0.026 | 内部配置在范围固定时仍能改变学习与外推；保留训练窗代价 | [A01](../evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) |
| 解释位置结构与学得使用 | 实际网格最慢8对占16坐标，有效rank 2.11；151.9M两seed权重×表crossing | 完整旋转对方向会重叠，模型对表的使用还依赖学习；不把rank当loss预测器 | [A03–A06](../evidence/index.md)；现稿Fig.1c、Fig.2 |
| 主要方法价值 | clean16K：650对，86.09/82.71%，+3.39pp [1.53,5.34]；clean32K：2600对，68.27/56.54%，+11.72pp [10.32,13.11] | 同一静态S4表在2L、4L具有明确任务收益；32K的12/13任务均值为正 | [Llama owner](../../../docs/research/next_stage_20260912/TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md) |
| 第二模型上的完整配置验证 | OLMo classic4/8/16K，Full-13 AUC +49.23pp，13任务差均正 | 相同公式在另一模型族/已记录合同上有效；与clean样本数和采样方式分开 | [OLMo owner](../../../docs/research/next_stage_20260912/TAILSPLINE_OLMO_CLASSIC_RESULT_20260914.md) |
| 辅助外推构造 | MLA432M三seed：16K PPL 138.8→95.6；750M共享起点单seed续训：16K PPL 45.1→24.4，8K答案token exact 0/40→31/40，4K PPL 22.0→22.3 | Cosh在配对学习下的外推价值；MLA未固定端点，750M不是完整答案加EOS指标 | [A09–A10](../evidence/index.md)；现稿§6.2/Fig.5 |
| 自然任务与native代价 | Llama Natural-QA631：+0.20pp [−1.53,1.89]；Native8K任务T−Native −2.14pp [−6.14,1.92]、PPL +0.37% | 自然QA排序未决；原生任务点估计及LM成本已量化 | [Llama owner](../../../docs/research/next_stage_20260912/TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md) |
| 细形状归因 | E1 T−C −0.41pp [−2.63,1.82]，跨batch，QUALIFIED_ONLY | 不能单独宣称已证明尾部平滑导致收益；不推翻完整T/P收益 | [A43审计](../../../experiments/iclr2027_three_track_sprint_20260915/reports/e1_matched_displacement_audit_v2.json) |

其余重要资产继续可见：BM自然QA是另一构造/模型的正面证据；Qwen小面板、三重复NIAH、
ProofPile-only近零AUC差在[辅助GPU结果](../../../docs/research/next_stage_20260912/SECONDARY_GPU_RESULTS_20260915.md)；
[Native-Z5](../../../docs/research/next_stage_20260912/NATIVE_Z5_EXPLORATION_RESULT_20260915.md)
已有checkpoint-calibrated的初步NLL改进，任务区间跨零、后续确认门未过。它们的存在既不应被遗漏，
也不自动升级为当前TailSpline论文的主胜负。

## 4. 先前意见是否还适用

| 先前问题/建议 | 对现稿的复核 | 当前必要性 |
|---|---|---|
| 主线只剩“提高窗口质量”或“allocation有用” | 引言开头仍较泛，但三条贡献、§3/§4/§5已经给出研究链 | **局部可优化，整套重写撤回** |
| TailSpline与Cosh/BM并列导致方法失焦 | §5.1 TailSpline、§5.2辅助Cosh；Fig.3只画TailSpline；BM保留对照 | **已解决，保留现稿** |
| 缺少与实参相连的几何展示 | 当前Fig.2展示b256网格的完整pair overlap及rank2.11 | **已解决，不补第二张重复主图** |
| 只在32K极限端点有效 | 已有clean16K；Table2和Fig.4同时展示2L/4L及全部任务差 | **已解决**；连续全窗口仍不是当前claim |
| 新颖性建立在更换坐标上 | 已承认radix product与cumulative displacement等价；贡献由识别、几何和构造承担 | **已解决，保持这一区分** |
| 总位移不等使TailSpline–MrPro结果无效 | 固定外部条件后D属于z的统计量，现稿§2/§5.1已明确 | **原要求过强，撤回**；C只回答更细归因 |
| 必须由理论推出普适任务最优边界 | TailSpline定理证明声明目标的唯一解，现稿将任务价值交给评测 | **不必要**；不以通用排序理论作为接收门槛 |
| Natural-QA与native负格没有处理 | 主文有完整Natural-QA总结、Native成本及其协议；E1资格也已报告 | **已解决**；无需继续加泛泛局限段落 |
| 必须重跑MLA才能使用现有结果 | 评价JSON已有同SHA的tracked镜像；现稿写清共享chunks与midpoint配方 | **不必要**；路径问题由文档修正解决 |
| 把已有750M正面证据拿回主文 | §6.2已有具体PPL与答案token exact，并保留监督与native成本 | **已解决** |
| 把BM自然QA抬到主文来增强故事 | 它是另一构造与模型的有效证据，已在附录；强行前移会扩大正文焦点 | **可选且非优先** |
| 必须补YaRN/native clean/更多模型才承认当前结果 | 可增强对比覆盖与推广性，但现稿没有宣称这些未测结论 | **增强实验，不是修复现有结果的前提** |
| 新z、native s=1、稀疏位置编码扩成新主线 | 会改变本次投稿对象；已有Z5是单独探索 | **本轮不纳入** |

### 同领域参照怎样用于这次判断

上一份[同领域比较](../ICLR2027_STABLE7_STRATEGY_20260915.md#3-同领域论文怎样把想要的效果变成科学问题)
已读MrRoPE、FoPE、Round and Round、Selective RoPE等本地论文。它们适合帮助检查
“问题—新知识—方法—验证”是否对应，不适合要求本稿复制它们的篇章形态或理论范围。
本稿与MrRoPE的合理差异是固定范围下的识别、完整pair有限窗结构、学得使用的区分和明确的
TailSpline构造及验证；不能把z记号本身写成表达空间的新增，也无需把MrRoPE简化成只追求长度。
现稿已经落实这些关键区别。

## 5. 仅针对论文的优化计划

### P1：可考虑的一处局部增强

**位置：Introduction第一段到第二段的连接。** 当前“how this allocation affects model quality”
可以更具体地提出未解决问题，随后直接进入现有受控干预。建议内容仅两层：

1. 给定频率范围仍没有确定内部配置提供的有限窗位置结构，学得使用也不由频率集合单独决定。
2. 本文通过可区分的干预与结构分析研究这两个方面，并给出可安装的解析构造。

可选的表达方向，而非本次改稿：

> A frequency range does not determine the positional structure supplied within a finite window, nor how a trained model uses that structure. We study these distinct effects of internal frequency allocation through controlled interventions, complete rotary-pair geometry, and explicit constructions.

此处只替换泛化目标句，不新增理论承诺，不改三条贡献、节顺序或图组。若作者认为现有首段配合
贡献列表已足够明确，也可以不改；这不是影响结论正确性的缺陷。

### P2：由新结果决定的实证增强

当前[强实验计划](../../../docs/research/next_stage_20260912/STRONG_EXPERIMENT_PLAN_20260915.md)
已经安排了跨模型clean、自然长文、clean等位移和部署取舍的执行规格，本审计不另建竞争计划。
对稿件的用途依次是：统一合同扩大外部适用证据、检验自然长文收益、分解细形状作用、补充更高倍率。
每个完整结果独立判断是否值得入稿，不要求全部完成才能保留现有主张。

若新增对照无优势，保留真实结果并相应限定更细结论；不从正负方向反推是否公开。既有2L/4L优势、
classic迁移和识别实验各自有成立的比较合同，不由一个新增指标统一推翻或升级。

### 本轮不建议做的改动

- 不改标题，不重新分配全文节次，不再次把Cosh扩成并列主方法。
- 不追加一套通用attention/value最优性理论来“补齐”现稿没有声称解决的问题。
- 不再添加相同含义的native限制、proxy限制和单模型式自我总结。
- 不因审稿意见数量增加就不断加实验；新增实验应回答现有主张之外的一项明确问题。

## 6. 执行状态与验证

只读服务器快照时间为`2026-09-15T08:17:36Z`：Full20的TailSpline 665/720、MrPro 0/720，
对应evaluator存活，GPU100%；完整标记与报告未生成。S16是已准备资产，YaRN停放。
此快照不提供任何中途性能结论。后续状态看[服务器owner](../../../experiments/iclr2027_three_track_sprint_20260915/SERVER_TASK_LAYERS.md)。
本次未启动GPU、下载数据、调整队列或部署远端脚本。

本轮验证：

- `python3 scripts/check_repository_docs.py --refresh-inventory`及最终只读检查通过；
  检查文档链接与46项资产的91处可移植来源哈希，未要求重建ignored raw镜像。
- A01–A46在人工资产表各出现一次：46/46，无缺项或重复。
- `bash -n experiments/iclr2027_three_track_sprint_20260915/server_task_status.sh`通过。
- 修改前后核对173个论文保护文件：正文、附录、表、图及图源、PDF、标题摘要、参考文献与
  投稿包均无变化。保留了工作区原本已有的论文修改。
- 对照源码包的90个当前科学内容文件，无差异；未重编译或重绘。
- 新增审计报告的文档链接已单独检查，并纳入维护清单的持续链接检查范围。

文档检查证明导航与来源身份一致，不替代数学审稿或模型能力验证。本报告中的剩余修改是
供作者选择的论文计划，本轮没有应用到稿件。
