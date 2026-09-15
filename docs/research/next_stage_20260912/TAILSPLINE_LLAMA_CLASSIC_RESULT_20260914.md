# TailSpline–MrPro Llama经典两臂与clean 32K确认

更新：2026-09-15。状态：**经典多长度判决与clean 32K大样本确认均完成；TailSpline在
clean Full-13的2,600个严格配对prompt上高出MrPro `11.72pp`，95%区间
`[+10.32,+13.11]pp`。**

## 1. 冻结身份

- checkpoint：`Meta-Llama-3-8B-Instruct`，冻结权重，Native 8192；
- 方法：exact finite-grid TailSpline vs exact MrRoPE-Pro；
- 两臂共同`S=4`、canonical band `[18,35]`、gain `1.138629436111989`、
  prompt、greedy decoder、precision与scorer；
- Full-13：13任务×8/16/32K×10行，共390个严格配对prompt/臂；
- PPL：32篇ProofPile test与14篇PG19 test，共46文档×3长度=138行/臂；
- Passkey复用Full-13中的`niah_single_1`，不算独立数据。

服务器结果根目录：
`/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic`。
严格报告为`reports/tailspline_vs_mrpro_classic.json`，SHA256
`13ea5e79d119de499df1fceab600b11cf62433f92f44b151cd897e2b9f5200ac`。

## 2. 三个主终点

| endpoint | TailSpline | MrPro | TailSpline−MrPro | 配对区间 |
|---|---:|---:|---:|---:|
| Full-13 task-equal log-AUC ↑ | 0.788013 | 0.756026 | +0.031987 | 95% `[+0.006538,+0.057853]` |
| NIAH task-equal log-AUC ↑ | 0.917187 | 0.879687 | +0.037500 | family bootstrap 95% `[+0.005469,+0.070313]` |
| PPL log-length AUC ↓ | 4.616635 | 4.621120 | −0.004485 | 95% `[−0.008716,−0.001214]` |

Passkey AUC为`1.000`对`0.975`，差`+0.025`。Full-13配对bootstrap
`P(delta_auc>0)=0.9935`。预注册的2/3继续门实际为3/3同向通过。

## 3. 长度曲线与代价

Full-13在8/16/32K分别为：

- TailSpline：`0.897436 / 0.840513 / 0.573590`；
- MrPro：`0.875000 / 0.800897 / 0.547308`；
- 差值：`+0.022436 / +0.039615 / +0.026282`。

NIAH在8/16/32K差值为`−0.003125 / +0.059375 / +0.034375`，即Native端有极小
负格，但16K与32K改善。PPL分别为：

- TailSpline：`5.295283 / 4.530318 / 4.110622`；
- MrPro：`5.286429 / 4.518111 / 4.161831`。

因此PPL的总体AUC优势由32K改善驱动；8K和16K分别轻微变差`+0.008854`与
`+0.012207`，不能写成逐长度全面占优。

按任务族AUC，TailSpline相对MrPro为retrieval `+0.03750`、tracking `+0.06500`、
aggregation约`+0.03792`、QA `−0.01250`。最大正任务是`niah_multikey_2`
`+0.225`；`qa_1`和`niah_multikey_1`各为`−0.050`，局部反转保留。

完整输出审计为：TailSpline/MrPro空输出`30/37`、EOS终止`388/384`、cap-hit
`2/6`（各390行）。

## 4. 原始资产与复用

| 资产 | 行数 | SHA256 |
|---|---:|---|
| TailSpline `generations.jsonl` | 390 | `8423a36cc998f609146970227e671ff99777dfd839928cb446da6146f14d2bad` |
| TailSpline `lm_rows.jsonl` | 138 | `ce93310af8f0e5723845d6c5ad4c344eaf58bce112d57c6a21e6250a2b808524` |
| MrPro `generations.jsonl` | 390 | `5c1b4246923ad3164b22b82c937d6bb0da3bf016a5adf3e36d2ecbc7f73c5745` |
| MrPro `lm_rows.jsonl` | 138 | `6ade51ef4f866857b5ef08571d08a6b4dda620641811dc55a6c9e294acc3e182` |

MrPro已登记到`/root/autodl-tmp/mrrope_baselines/current/llama3_8b_s4_full13_ppl46_mrpro`，
`ready_for_score_reuse=true`。只有checkpoint、prompt/data、table/gain、decoder、precision
与scorer全部匹配时才能复用分数；其他实验只复用资产。

## 5. 允许与禁止的结论

本结果支持：在这一冻结Llama S4经典合同上，exact TailSpline整体优于exact MrPro，且
Full-13与NIAH的配对区间下界为正，PPL AUC的小幅优势区间也低于零。

本结果尚不证明跨checkpoint普适优势，不比较official YaRN，也不证明one-sided
roughness或tail landing是收益原因。TailSpline与MrPro仍同时改变总log位移和细形状；
现阶段不得由赢家倒推边界泛函正确。下一步只做作者已授权的OLMo两臂跨模型确认，
不启动YaRN/Fast/Slow四格或新的曲线搜索。

## 6. Clean 32K大样本确认

该确认使用同一Llama checkpoint、exact TailSpline/MrPro表、`S=4`、band
`[18,35]`、gain、greedy decoder、precision与scorer，只替换为上游RULER revision
`c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`生成的source-order输入。面板为13任务
各200行，共2,600个prompt/臂；不做depth筛选或内容padding，实际输入为
`28,270...32,606` tokens，运行固定batch 1与8,192-token prefill chunk。

主指标是单一32K端点的13-task-equal macro，不称多长度AUC：

| endpoint | TailSpline | MrPro | TailSpline−MrPro | 95%配对区间 |
|---|---:|---:|---:|---:|
| Full RULER-13 32K | 0.682660 | 0.565436 | **+0.117224** | **[+0.103231,+0.131148]** |
| retrieval | 0.804219 | 0.695625 | +0.108594 | [+0.091094,+0.126094] |
| tracking | 0.796000 | 0.556000 | +0.240000 | [+0.206000,+0.274000] |
| aggregation | 0.317417 | 0.199833 | +0.117583 | [+0.088000,+0.147667] |
| QA | 0.505000 | 0.415000 | +0.090000 | [+0.042500,+0.137500] |

TailSpline在12/13个任务上获胜，仅`niah_multivalue`下降`4.5pp`。删除任意一个任务后，
其余12项的task-equal差仍为`+9.53...+13.07pp`，不是单任务驱动。TailSpline/MrPro
的EOS率为`98.96%/96.81%`，cap-hit为`1.04%/3.19%`，优势不是以更差输出健康换取。

RULER每一行由上游生成器独立产生；统计在任务内配对重采样并固定任务权重。相同答案、
模板或生成器index不等于共享自然文档，不能据此伪造source cluster。Natural-QA另按真实
source document聚类。

便携报告为
[clean_ruler200_tailspline_vs_mrpro.json](../../../experiments/iclr2027_three_track_sprint_20260915/reports/clean_ruler200_tailspline_vs_mrpro.json)，
SHA256 `5c7e2fe7275029c4aad98520a420a05e170bbe5dcea354f1e01f190e4ee1fe4b`；
TailSpline/MrPro raw SHA256分别为
`7b7ca8951b5b7cce5d64ec8b38dc3ec37edf49037aa5a4d94d3b3f2d008b96ab`与
`05529c18cbec4fae8cdbfd13ca163b7fd6027cb3eac42d96d93ac4e543630e21`，各2,600行。

两个补充控制已经完成：39行batch-2相对batch-1的任务分数漂移为零，但6行生成文本
不同，因此只支持评分稳定诊断，不证明bitwise等价；Native 8K同一经典130行面板为
`0.918846`，TailSpline为`0.897436`，Native−TailSpline区间
`[-0.019231,+0.061410]`，没有确认两者存在任务差距。Native显著高于MrPro
`4.38pp`，区间`[+0.86,+7.73]pp`。

本结果把clean 32K比较提升为TailSpline冻结部署的hero experiment。它仍不识别总位移与
高阶shape的各自贡献，不比较YaRN，也不把synthetic RULER替代Natural-QA。下一项主证据
只允许使用已冻结的Natural-QA631 TailSpline/MrPro比较。

## 7. 本轮论文整合与E1资格复核

2026-09-15，本轮从服务器读取完整T/C的390生成及138 LM行、clean T/P各2600生成，
重新核对配对身份、官方评分及点估计。clean数字与上节owner一致，
[可移植score-only输入](../../../paper-2027/figs/field_gap_inputs.json)保留全部2600分数对，
[独立检查器](../../../paper-2027/figs/verify_field_gap.py)复核点估计、配对bootstrap并重建逐任务表。
原始输入/完整输出仍在既有服务器路径，不称随匿名包提供了全部raw。

[E1审计V2](../../../experiments/iclr2027_three_track_sprint_20260915/reports/e1_matched_displacement_audit_v2.json)
保留原始数字：T/C Full-13 AUC为78.8013/79.2115%，T−C −0.4103pp，原区间
[−2.6282,+1.8205]pp，任务4胜5负4平。数学约束PASS，raw完整；batch和顺序不同，
旧合同也未完整记录revision/backend等字段。因此`runtime_match=QUALIFIED_ONLY`、
`evidence_role=cross_runtime_diagnostic`，E0不能提升成完整纯shape确认；没有事前等效
阈值，也不作等效或tail-only中介结论。原区间保留原任务-长度行内bootstrap口径。

[真正Native-8K PPL](../../../experiments/iclr2027_three_track_sprint_20260915/reports/native8k_ppl_comparison.json)
是46文档、376832个目标token的whole-prefix汇总：Native/T/P=5.275919/5.295283/5.286429。
T−Native +0.019364，文档配对区间[+0.006602,+0.034861]。与classic PPL AUC采用不同
汇总，不直接混算。Native任务参考的跨零区间不证明无损或非劣。

论文与下一步的具体落点见[差距与决策映射](../../../paper-2027/research/COMPARATIVE_GAP_AND_DECISION_MAP_20260915.md)。
本轮只读取既有服务器产物并运行本地CPU；未更改远端进程或启动新GPU臂。
