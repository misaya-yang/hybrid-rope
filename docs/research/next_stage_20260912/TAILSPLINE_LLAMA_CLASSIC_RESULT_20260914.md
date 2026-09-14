# TailSpline–MrPro Llama经典两臂结果

更新：2026-09-14。状态：**冻结两臂主判决完成；TailSpline在三个预注册family endpoint上均优于MrPro，允许进入跨模型确认。**

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
