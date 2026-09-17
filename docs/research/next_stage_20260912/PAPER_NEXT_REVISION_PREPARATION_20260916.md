# 下一版论文准备：概念、证据与实验取舍

**2026-09-17执行更新：**本页已完成项已进入[本轮稿件升级](../../../paper-2027/research/revision_20260917_evidence_update/README.md)。
当前稿件9/34页；直接基线、70B、GLM第二书池已纳入；NCP保留并新增同目标NLL。NTS2整组等待更完整QA确认。
原生部分按真实下游任务、预测质量、RULER诊断的顺序组织；NCP既有原生固定支持内容保留。
以下日期队列只保留历史准备语境，后续执行仍看各实验owner。


更新：2026-09-16。**作者已授权并实施一版已完成证据增量；本文件下方保留此前准备记录。**
当前状态见[论文修改记录](../../../paper-2027/research/revision_20260916_evidence_update/README.md)。
现稿已纳入四模型quick、GLM65对Full-13与Qwen/GLM35题三臂QA；其后的大样本YaRN、
GLM新书池及Llama-3-70B NF4结果现已完成并登记，下一版统一替换或追加。
本文件负责下一次论文整合取舍，不替代实验执行owner，不授权新GPU任务。
现稿保持主文9页、全稿29页；下一版目标总页数≤35，硬上限40。

## 1. 主线与方法职责

**给定实际频率覆盖范围与旋转预算后，内部配置z仍是改善模型质量的设计自由度。**
研究对象包括可受控的内点、与范围策略的交互、既有内容坐标对频率的使用。
“独立”指干预可分离，不意味着各种效果互不作用。

| 构造/证据 | 论证职责 | 下一版处理 |
|---|---|---|
| 固定支持151.9M三seed、成熟冻结控制、等位移C、坐标干预 | 识别z的作用及其与模型使用的联系 | 保留正文控制链，不因Cosh是支持方法而移走关键识别实验 |
| TailSpline | 不更新权重的扩展配置，保留标准旋转计算 | 主方法；跨长度、模型和自然任务结果直接呈现 |
| NCP | 固定端点、单位gain、公开参数下的原生窗口增强 | 正文独立地位；考虑将构造放方法、结果放实验，待统一排版 |
| Cosh | 密度搬运及学习/外推支持 | 解释目标与移动方向，控制篇幅；完整学习曲线可放附录 |
| 完整pair几何、内容竞争连接、turn归一化 | 明确结构怎样改变可用距离响应 | 条件解释，避免把几何代理升级为任务排序定律 |

## 2. 哪些已入稿，哪些等待整合

| 状态 | 证据 | 来源 |
|---|---|---|
| 已入现稿 | Llama clean8/16/32K、OLMo clean16K及自然QA、clean T/C、NCP、学习支持 | [现稿索引](../../../paper-2027/index.md)、[当前主张表](../../../paper-2027/research/EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md) |
| 已完成，待下一版统一整合 | Qwen S4/128K En.QA、S8/256K单针；Llama S16 RULER/PPL/自然任务；抽样稳定性 | [Pro6000正式结果owner](PRO6000_EXTREME_NATURAL_QA_RESULTS_20260916.md) |
| 已完成，待下一版统一整合 | Qwen S4官方静态YaRN quick三方法：NIAH T/P/Y为72.50/73.125/76.25%，固定面板由YaRN领先；PPL三者接近 | [配对报告](../../../experiments/iclr2027_strong_evidence_20260915/reports/pro6000_qwen3b_s4_128k_yarn.json) |
| 已完成，待下一版统一整合 | Llama与OLMo官方静态YaRN大样本三方法：NIAH-8×200、PPL46、Natural-QA631、Full-13×10 | [双服务器结果owner](DUAL_SERVER_YARN_AND_70B_RESULTS_20260917.md) |
| 已完成，待下一版统一整合 | Qwen/GLM Full-13×10三臂、GLM独立第二书池 | [Pro6000结果owner](PRO6000_EXTREME_NATURAL_QA_RESULTS_20260916.md) |
| 已完成，待下一版统一整合 | Llama-3-70B NF4 S4/32K Full-13、Natural-QA与PPL；S16/128K PPL | [双服务器结果owner](DUAL_SERVER_YARN_AND_70B_RESULTS_20260917.md) |
| 已完成，待下一版统一整合 | OLMo NCP Native-4K同目标NLL、新Full-13×10、Natural-QA99及机制拆分 | [NCP Native结果owner](../../../experiments/native_enhancement_oral_20260915/index.md) |
| 后置，不是当前论文缺口 | 70B S16/128K任务侧：TailSpline多针已失效、MrPro仅3/80；补齐约3 GPU小时 | [70B执行入口](../../../experiments/llama70b_scale_20260916/README.md) |

GLM及Qwen现已有10/task严格配对三臂报告；早期5/task报告保留为开发历史，不替换正式结果。
已完成报告、输入预算与实际长度、单针与Full-13、官方F1与accuracy分别维护。

### 2.1 新增三方法直接比较：Llama与OLMo

2026-09-16读取PRO6000正式报告，并从报告中的40条配对评分及5篇文档NLL独立复算宏均值和差值；
未重新评分原始生成文本。两份报告均为`COMPLETE`，合同为
`official-static-yarn-zero-training-quick-three-method-v1`。
远端owner在`root@connect.westd.seetacloud.com:51638`：
`/root/autodl-tmp/today_rope_plan_20260914/official_yarn_quick/`下的
`llama3_8b_s4_32k/report.json`与`olmo2_1b_s4_16k/report.json`。
这批比较使用官方静态YaRN的零训练安装，不代表经过微调的YaRN模型。

| 模型与长度 | NIAH-8×5：T / P / Y（%） | T−Y（pp），95%区间 | PPL-5：T / P / Y |
|---|---|---|---|
| Llama S4/32K | 85.00 / 75.00 / 75.00 | +10.00，[-0.625, 20.00] | 2.7698 / 2.7958 / 2.7889 |
| OLMo S4/16K | 61.875 / 8.125 / 7.50 | +54.375，[43.125, 66.25] | 5.6616 / 8.3218 / 7.9126 |

Llama的T−P NIAH为+10.00pp，区间[-0.625,20.625]；文档平均NLL的T−P/T−Y
为−0.009362/−0.006891，两项文档bootstrap区间均低于零。
OLMo的T−P NIAH为+53.75pp，区间[41.25,66.25]；文档平均NLL的T−P/T−Y
为−0.385183/−0.334756，区间分别为[-0.447080,-0.349714]、[-0.382658,-0.303669]。
PPL为同一5篇文档的token汇总指标，区间针对配对文档平均NLL差；不混用既有46篇结果。

**改稿价值：** OLMo在直接三方法比较中同时提高检索准确性并降低语言建模损失；
Llama两项指标同样由TailSpline取得正式点分领先。已有大样本T/P结果继续承担主证据，
quick补充直接YaRN基线；三者共享的T/P行只计作一次模型运行，不重复包装为独立复现。
这批结果强化配置设计的实际价值，不单独证明尾部平滑性是收益的唯一机制。

上述quick已由大样本NIAH/PPL、Natural-QA631和Full-13×10正式报告补齐；下一版直接使用
[双服务器结果owner](DUAL_SERVER_YARN_AND_70B_RESULTS_20260917.md)中的最终读数，quick只保留溯源。

## 3. 已确定的概念纠正

采用[Pro处理结论](../reviews/PRO_REASSESSMENT_DISPOSITION_20260916.md)：

- native增强已非纯未来方向：NCP已有原生任务与NLL正结果；TailSpline则以约`0.37%`
  原生PPL代价换取显著长端收益，两类构造分别陈述。
- 自然QA收益不再只指OLMo：Qwen S4完整长书QA增加另一模型族的实际证据。
- Qwen旧NIAH面板接近不等于大base压制z；同S4下任务不同也可显现配置差异。
- 不使用一般“水床定律”；固定频率数量/质量不推出任务质量守恒。
- Cosh抑制密度集中及共同慢端占用，锚定内点向较快端搬运，不增加旋转维数。
- 总位移是z的统计量；其与增量重心的关系不算两项独立控制。
- 局部固定Q/K状态的读取恒等式不等于冻结权重下所有层状态不变。
- GLM是partial RoPE/GQA，32个旋转对；不是MLA，也不是仅一个head在旋转。
- 现有原始结果不因文件散在分支或租赁机器而失效，不重开432M来源问题。

[十篇审稿经验](../reviews/TEN_PAPER_REVIEW_LESSONS_20260916.md)只筛选与以上主张有关的压力。

## 4. 结果回收之后的实验准备

### A. 已完成的当前队列

GLM、Qwen、Llama和OLMo的YaRN正式报告均已接收；70B冻结尺度迁移也已完成S4主套件。
下一版保留全部预定任务、正负结果和运行身份，不把GLM或70B得分当作单变量base/规模因果证据。

### B. 原生主张新增确认：已完成

作者将原建议的Full-13×100调整为新source-order Full-13×10快速正式面板，固定NCP与Native
两臂，不根据输出调表。130条/臂的正式task-equal差为`+3.2564pp`；同轮Native-window
Natural-QA99为`−0.8549pp`。另有103源文档、128窗口的同目标4K NLL差`−0.012786`
nat/token，约等于PPL降低`1.27%`。完整结果与机制分解由
[NCP Native结果owner](../../../experiments/native_enhancement_oral_20260915/index.md)维护。

### C. 官方静态YaRN的直接方法定位：已完成

Qwen、GLM、Llama和OLMo的同协议结果均已完成。Llama/OLMo复用既有T/P并只生成缺失YaRN臂；
Full-13×10、Natural-QA631、NIAH-8×200和PPL46均有正式报告。
官方默认band/rounding/gain的实用比较和人为匹配band的控制比较分别标记，不能冒充彼此。
不通过“前人胜过Y、我们胜过前人”传递得出T优于Y。

### D. 原生信息利用的行为检验：已完成

288条、按world组织的反事实面板已完成Native/H/NCP/V1四臂。NCP在1K/2K/4K的正式
exact差为`+2.083/0/−11.458pp`；固定末四层双向相位干预也已完成。它们排除了所测
合成面板与该固定层块作为NCP总体收益的简单中介解释，但不反向抹掉NLL与Full-13成绩。
更细Q/K/V signed-response没有生成紧凑正式报告，保持未完成身份，不作为改稿前置条件。

### 优先级如何选择

GLM/直接基线、70B尺度迁移、NCP确认与反事实行为均已回收；后续优先级转为统一论文整合。
70B 128K任务基线仅在预算允许时补，不是论文提交前的默认前置条件。
既有T/C、151.9M与432M不重跑；极限倍率扩样与更多单针热图按明确问题再决定。

## 5. 下一版改稿增量清单：尚未应用

| 位置 | 准备好的修改方向 | 等待什么 |
|---|---|---|
| 标题/摘要/引言 | 保持Beyond the Base与z中心；自然任务概括吸收Qwen及70B尺度迁移；不写数字进摘要 | 双服务器结果owner已给出准确边界 |
| §2–3 | 明确控制变量与训练/冻结干预；突出已有等位移证据 | 无需新实验；与全稿一次修改 |
| 理论段 | 用目标/干扰竞争短解释接现有结构；充分条件就地注明 | 取舍清单已完成；不增造主定理 |
| 方法段 | TailSpline→NCP→Cosh候选顺序；Cosh两项含义清楚 | 九页正文整体排版与作者审核 |
| 实验主表 | 跨模型RULER与自然QA并列，保留指标和任务覆盖；NCP单独Native对照 | 使用A59–A62正式报告；不使用70B 128K partial任务行 |
| 极限结果 | Llama S16与Qwen S8明确各自方法/任务画像 | 当前报告已可用，后续RULER块按身份追加 |
| 学习支持 | 151.9M识别保留；432M完整曲线可进附录，正文紧凑展示 | 仅叙事/排版，不动已有数据 |
| 附录 | 在现有A–F放置短推导/协议/任务表，历史旁支继续归档 | 完整新结果；不恢复大乱炖 |

真正改稿时再同步TeX、主张位置、图表、PDF、源码包和编译验证。
本轮只修文档概念、导航与准备状态；不将计划写入论文事实。

## 6. 阅读顺序与完成标准

先读本文；需要理由读[Pro取舍](../reviews/PRO_REASSESSMENT_DISPOSITION_20260916.md)，
需要审稿案例读[十篇经验](../reviews/TEN_PAPER_REVIEW_LESSONS_20260916.md)，
需要数值进入[结果罗盘](KEY_EXPERIMENT_COMPASS_20260914.md)的唯一owner链接。
源材料完整留档供核查，不作为每次任务的启动阅读栈。

下一版验收：主文独立讲清问题、控制、构造与结果；术语不跨对象偷换；关键结果有直接对照；
所有方法/数据身份清楚；无数字摘要和无图首页保持；主文≤9页，总体优先≤35页且≤40页。
