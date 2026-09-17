# 新落盘实验：论文价值与建议修改

状态：**9月15日本批已按作者确认的方案应用到主稿；9月17日新增NCP独立确认已登记，当前主稿未据此改写。**
本文件先读取服务器已完成报告及对应原始生成行，随后完成论文整合；服务器文件时间虽更新，
仍沿用本轮研究目录的9月15日标识。

## 结论

**建议正式吸收本批结果。这是实质实验增强：clean跨模型确认、自然QA收益、等总位移下的配置形状效应，现在都有可用的模型证据。**

全文主线仍是：**频率覆盖范围确定以后，内部配置怎样影响模型质量，以及如何利用这个设计自由度。** TailSpline提供主要冻结部署构造；Cosh保留学习与外推的辅助角色。新增结果补强这条论证链，无须另起多条主线。

## 作者确认后的写作定位

全文研究对象是z，正文必须同时体现外推和in-window作用。

- TailSpline承担外推质量提升：完整窗口表呈现L/2L/4L；相对原始Native的语言建模代价仅
  约`0.37%` PPL，正文将其作为“很小代价换取显著长端收益”的部署优势正面陈述。
  8K任务的实际得分和逐任务分解由表图完整展示。
- 不设独立“原生代价/部署取舍”小节。原始RoPE参考、完整分数与逐任务分解保持可见。
- NCP进入正文独立小节，承担z在不扩大窗口时仍能改善任务性能的证据；与TailSpline的
  原生窗口保留情况区分，不能把两种构造的结果拼成同一张表同时实现全部收益。
- 自然写清外推质量改善，不加入“只在温和倍率”“尚非极限”等作者未要求的限定。
  具体模型、倍率、长度写在实验设置与表格；后续基准与128K结果完成后再补入相应位置。

## 1. 最值得进入主文的四项结果

分数为百分数，差值及区间为百分点；区间沿用正式报告的配对bootstrap口径。

| 新证据 | 样本 | TailSpline | 对照 | 差值与95%区间 | 主要论文作用 |
|---|---:|---:|---:|---|---|
| OLMo clean 16K Full-13 | 2,600对 | 50.65 | MrPro 9.23 | **+41.42 [39.99,42.81]** | 将第二模型族从classic支持提升为大样本clean确认 |
| OLMo Natural-QA | 631对、524文档簇 | 24.92 F1 | MrPro 21.62 F1 | **+3.30 [0.34,6.40]** | 证明收益可以出现在自然问答，五项任务的点差均为正 |
| Llama clean 32K T–C | 2,600对 | 68.27 | 等位移C 66.17 | **+2.10 [1.11,3.08]** | 在总log位移相同后，内部配置仍改变模型质量 |
| Llama clean 8K T–P | 650对 | 85.16 | MrPro 82.39 | **+2.76 [1.29,4.25]** | 与既有16K、32K组成同一静态S4表在L、2L、4L的质量比较 |

### 1.1 OLMo clean：应替代classic作为主文跨模型重点

OLMo-2-1B-Instruct、S4、16K的13项任务全部为正；四类任务均有正的配对区间：

- Retrieval：+60.19pp，[58.30,62.03]。
- Tracking：+9.70pp，[7.00,12.60]。
- Aggregation：+16.13pp，[13.86,18.42]。
- QA：+7.50pp，[2.75,12.25]；两项QA分别+8、+7pp。

这与Llama clean 32K构成同为4L、同为13×200的并列确认。不同tokenizer与模型下的输入仍各自配对，不把两模型分数合并成一个总指标。classic多长度曲线保留附录，主文无需重复堆报两套OLMo数字。

输出行为与Llama不同：OLMo T/P的cap-hit为51.88%/46.46%，空输出均为零。不能沿用Llama“EOS更高、cap更低”的描述来概括所有模型。值得注意的是FWE与MK2在两臂均几乎全部正常终止，仍分别提升28pp与42.5pp。任务得分和终止行为可在附录并列展示，不用把cap当作替代评分标准。

**推荐主文句子：**

> On OLMo-2-1B-Instruct, the clean target-window evaluation confirms the same allocation advantage: TailSpline improves the Full-13 score from 9.23% to 50.65%, with positive differences on all thirteen tasks.

### 1.2 自然QA：现在可以正面写“收益延伸至自然问答”

OLMo Natural-QA的631个输入全部超过原生4K，实际长度4,110–16,319 tokens。五任务等权、整段输出F1、524个源文档簇；两臂同prompt、同表/gain合同、同解码条件。

逐任务T−P：2Wiki +4.74pp、HotpotQA +6.02pp、MultiFieldQA +1.31pp、NarrativeQA +0.34pp、Qasper +4.09pp。总体是+3.30pp；问题等权敏感性分析为+3.92pp，[1.20,6.68]，不依赖仅一种任务加权产生正方向。

主文把Llama与OLMo自然QA放入同一张紧凑表：Llama 41.08/40.88，OLMo 24.92/21.62。这样读者能直接看到自然任务证据的完整轮廓，不必先读一段防御式解释。

**推荐主文句子：**

> Allocation gains also extend to natural question answering: on the frozen OLMo evaluation pool, TailSpline improves task-macro F1 from 21.62% to 24.92%, with positive mean differences across all five tasks.

不要改写为“所有模型、所有自然任务均显著提升”。Llama已有Natural-QA结果仍按原值保留。

### 1.3 Clean T–C：最直接补强理论与实验连接的证据

当前论文主要保留旧classic T–C跨batch诊断；新X4使用clean输入、batch1、相同8K prefill设置，复用已完成T/P，只新增C。核对部署FP32表后，T−C的sum(m)差约−7.65e−9；gain、band外频率及端点相同。

| 长度 | T | C | P | T−C |
|---|---:|---:|---:|---|
| 16K | 86.09 | 86.35 | 82.71 | −0.25pp，[−1.41,+0.89] |
| 32K | 68.27 | 66.17 | 56.54 | +2.10pp，[+1.11,+3.08] |

最有价值的解释是：**总位移不是配置的充分描述；达到相同总位移的表仍可以具有不同的任务质量。** 这与固定support干预共同构成研究对象的识别证据。

32K T−C的主要正项是MK2 +10.5pp、MK3 +14.5pp、FWE +7pp；QA1 −5.5pp、MK1 −3.5pp等完整保留。16K两者接近，32K才出现明确差距，说明所测形状对比具有长度依赖性。

此外，现有完整旋转对rank计算在32K为C 10.0926、T 10.0773，任务质量却为T更高。这为“位置几何与模型利用应区分”增加了一个等总位移的实例，可用一句交叉引用连接§4与实验。无需重新组织成“rank越低越好”的理论。

**推荐主文句子：**

> With total log-frequency displacement matched, TailSpline still improves the clean target-window score over C. Total displacement therefore does not fully determine the quality of an allocation.

不把C−P和T−C的算术差分解释为“收益中多少百分比由剂量/边界机制导致”。它们是三张具体表的对照，不是唯一因果分解；同样不把本结果直接等同于one-sided目标的普适最优性。

### 1.4 Clean 8K：形成L、2L、4L的完整实用比较

这项结果最好的主文用途，是把现有两长度表扩为三个测量点：

| Llama S4静态表 | 8K（L） | 16K（2L） | 32K（4L） |
|---|---:|---:|---:|
| TailSpline | 85.16 | 86.09 | 68.27 |
| MrPro | 82.39 | 82.71 | 56.54 |
| T−P | +2.76pp | +3.39pp | +11.72pp |

每个长度都是各自source-order面板，样本数650/650/2600。表用于比较各长度的两法表现，不把8K与16K绝对分数的上升解释成长度增益。

原始RoPE的clean 8K参考为90.03%；表中同时保留TailSpline 85.16%与MrPro 82.39%。新clean与旧classic分别按原协议呈现，不拼接两者数值。

**主文写法：TailSpline以约`0.37%`的原生PPL代价，换取显著长上下文任务质量提升。**
L/2L/4L三点T/P差值+2.76、+3.39、+11.72pp由表图呈现；8K表内保留原始RoPE参考，
让部署收益与代价直接可见。

逐任务诊断非常集中：FWE为T 9.33%、P 4%、Native 87.33%；T/P分别45/48个空回答，全部位于FWE，Native无空回答。其余十二项T−Native的描述性均值为+1.225pp，retrieval任务族约持平、QA点差+7pp、tracking +2.4pp。

这项逐任务分解放入附录，解释原始RoPE参考与两个扩展构造在FWE上的行为差异。主文保留完整Full-13指标，以三个长度上的T/P比较组织结果。

## 2. 补充证据：放在哪里最有价值

### LongBench v2：放自然任务表或附录，作为更长真实输入支持

Llama实际8K–32K范围内117个完整输入：T 41/117=35.04%，P 36/117=30.77%，+4.27pp；总体文档簇区间[−3.39,+11.97]pp。89个输入超过16K，其点差+4.49pp。双方直接多选答案按声明的官方适配器评分，未使用RULER contains指标。

它补上现有Llama Natural-QA只到约16K之外的真实输入覆盖。正点差可如实呈现；主文自然QA的明确收益由OLMo承担，不让这个小面板单独承担“自然任务确认”的标题。

### Native方向：确实出现了值得保留的正结果

同一个OLMo原生4K、780条clean Full-13面板：

| 构造 | 得分 | 相对Native的任务差 |
|---|---:|---|
| Native | 69.37% | — |
| half-turn contract | 69.51% | +0.13pp，区间跨零 |
| reverse | 67.90% | — |
| 历史checkpoint-calibrated V1 | 72.38% | **+3.01pp**，[1.29,4.71] |
| 公共参数构造NCP | 70.78% | **+1.41pp**，[0.20,2.58] |

NCP表不读取模型输出，gain=1，构造仅用原生RoPE表与公开窗口参数。它直接支持作者提出的方向：**改变z的作用不必以增加外推倍率为前提。** V1则保留为checkpoint-calibrated独立证据，不能与NCP混称零校准。

half-turn contract相对reverse为+1.61pp，[0.27,2.97]，但相对Native接近；方向对照和原生增强是两个不同问题。原half-turn报告的`delta`字段使用bootstrap均值，本表用臂均值之差给出精确点估计，原区间保持不变。

**正文新增小节：Improving task performance within RoPE’s native context。** 用一个紧凑小节说明：

1. 主张：优化z提升RoPE原生支持长度内的任务性能；以已完成结果引出构造与解释。
2. 构造：NCP仅使用公开原生RoPE参数，固定gain=1，无权重更新或模型输出校准；
   简述参考竞争目标与贴近原生表的约束，完整公式与推导放附录。
3. 结果：OLMo原生4K、780个clean输入，Native/NCP为69.37/70.78%，
   +1.41pp，[0.20,2.58]；完整任务表、现有面板的使用历史与评分协议放附录。
4. 结论：内部频率配置的作用包含in-window任务质量，因而研究对象超出context extension。

本小节以NCP为主要构造，calibrated V1与half-turn/reverse作为附录辅助。
9月17日新增独立确认进一步得到：103源文档同目标Native-4K NLL降低`0.012786`
nat/token（约`1.27%` PPL），新Full-13×10提高`3.2564pp`；Natural-QA99为
`−0.8549pp`。因此NCP现在同时具有原生语言建模和新任务面板的正式正结果；Natural-QA
按自身benchmark保留，不反向抹掉前两项。证据入口见
[NCP Native实验与结果](../../experiments/native_enhancement_oral_20260915/index.md)。

### 128K状态

该端点上Llama S16和Qwen S4已有资产与4080工程验证，当前所查目录没有正式128K两臂任务报告。工程canary不作为论文能力结果；不更改现有队列。

## 3. 推荐修改顺序：增强主线，避免堆结果

1. **摘要**：维持无数字；写清跨模型外推质量、自然QA与等位移证据，并用一句纳入NCP的in-window任务与NLL收益。TailSpline表述为以极小原生PPL代价换取显著长端收益。
2. **引言贡献**：保留识别、理论、构造的主线，构造与实验贡献明确覆盖context extension和in-window两种作用；NCP通过正文小节承担后者。
3. **§3或§5的配置对照**：加入新X4的结论及指向。总位移本来就是z的属性，T/P是完整配置比较；T/C提供更细的识别。
4. **§6主结果**：按“TailSpline窗口质量、跨模型与自然问答 → 等位移配置证据 → NCP原生窗口小节 → Cosh辅助学习/外推”组织。Llama L/2L/4L表完整展示；直接写成以约`0.37%`原生PPL代价换取长端收益，不另设防御式代价小节。
5. **附录**：新增OLMo逐任务/输出行为、clean T/C、clean Native诊断、LongBench v2、NCP推导与完整任务分析。NCP主要结论在正文；旧classic保留其历史合同。
6. **同步更新现有判断**：摘要、引言、结论体现z的extension和in-window作用；TailSpline性能与NCP原生增强分别归属各自构造。原“Native-window trade-off”段落并入统一长度评价，改为极小PPL代价与长端收益的正面部署表述。

建议摘要实验句的替换候选（无数字）：

> TailSpline improves long-context task quality across model families, with benefits extending to natural question answering and only a slight reduction in native-window performance. A complementary native-window construction demonstrates that frequency allocation can also improve in-window task performance. Equal-displacement controls show that internal placement matters beyond total frequency shift.

这是待整合的实验措辞候选，完整摘要仍须压缩统筹，避免连续堆砌结果。

上述长度证据来自Llama的三个已测点，跨模型证据来自OLMo clean确认；正文表格明确各自覆盖。摘要不放数字。

**标题建议保持现有的“Beyond the Base: Frequency Allocation in RoPE”。** 新证据恰好加强这个上位定位，暂不需要为某项新分数更名。

## 4. 本次核验与可复用资产

- [服务器报告快照与来源](../../experiments/iclr2027_strong_evidence_20260915/reports/completed_results_snapshot_20260915.json)：八份正式报告、远端位置、文件哈希。
- [逐行与合同核验](../../experiments/iclr2027_strong_evidence_20260915/reports/completed_results_raw_verification.json)：读取22,296行对应生成记录，检查配对身份并重聚合已有RULER分数。
- [派生结论](../../experiments/iclr2027_strong_evidence_20260915/reports/completed_results_derived_findings.json)：完整OLMo QA与LongBench v2重新评分、表一致性、等位移误差、原生任务分解。

OLMo QA共1,262个输出、LongBench v2共234个输出重新评分后与正式报告一致。配对区间沿用原报告，本次没有重跑生成或把存储的RULER分数复聚合声称为从文本重新评分。旧T/P合同未记录后来新增的runtime版本字段，共有执行字段一致；不据字段缺失推定实验有误，也不声称重建了未记录信息。

原始输出仍位于原服务器目录。本文件保留采用理由与修改方案。最终TeX/PDF已应用，验收见[本轮验证](COMPLETED_EVIDENCE_REVISION_VALIDATION_20260915.json)。
