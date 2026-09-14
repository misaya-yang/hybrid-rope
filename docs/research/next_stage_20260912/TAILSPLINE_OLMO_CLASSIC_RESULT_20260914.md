# TailSpline–MrPro OLMo经典两臂结果

更新：2026-09-14。状态：**冻结两臂跨模型确认完成；TailSpline在Full-13、NIAH与PPL
三个预注册方向上均显著优于MrPro，13个任务AUC差全部为正。**

## 1. 冻结身份

- checkpoint：`OLMo-2-0425-1B-Instruct`，冻结权重，Native 4096；
- 方法：exact finite-grid TailSpline vs exact canonical MrRoPE-Pro；
- 两臂共同`S=4`、canonical band `[14,32]`、gain `1.138629436111989`、
  prompt、greedy decoder、precision与scorer；
- Full-13：13任务×4/8/16K×10行，共390个严格配对prompt/臂；
- PPL：与Llama实验相同来源的32篇ProofPile test与14篇PG19 test，按OLMo tokenizer
  冻结4/8/16K前缀，共138行/臂；
- 生成使用`batch_size=4`；方法和数据在读取任何本次OLMo输出前冻结。

服务器结果根目录：
`/root/autodl-tmp/today_rope_plan_20260914/tailspline_olmo_s4_classic`。
严格报告为`reports/tailspline_vs_mrpro_classic.json`，SHA256
`51db6850c2486eb3d1e2b8ffc6047f2a2f83fef0efc58d3dc22adc093db65a09`。

TailSpline/MrPro表SHA256分别为
`4e5f3fcee2c46f18e000d08b42333a644aa6d2741ddf5a7397d21a496fce51a3`与
`ed0120e3ba436199c71d2e25e923d7665a1e9c264fe18135f61f310296b94069`。
后者与服务器既有canonical OLMo S4 MrPro表逐位身份相同，排除了临时重建错表。

## 2. 三个主终点

| endpoint | TailSpline | MrPro | TailSpline−MrPro | 配对区间 |
|---|---:|---:|---:|---:|
| Full-13 task-equal log-AUC ↑ | 0.665962 | 0.173622 | +0.492340 | 95% `[+0.453365,+0.530449]` |
| NIAH task-equal log-AUC ↑ | 0.810937 | 0.166406 | +0.644531 | family bootstrap 95% `[+0.592188,+0.694531]` |
| PPL log-length AUC ↓ | 9.980871 | 13.833802 | −3.852932 | 95% `[−4.775042,−3.125459]` |

Passkey AUC为`0.950`对`0.375`，差`+0.575`。Full-13 bootstrap
`P(delta_auc>0)=1.0`。预注册的2/3继续门为3/3通过。

## 3. 长度、任务与输出审计

| 长度 | TailSpline Full-13 | MrPro Full-13 | 差值 | TailSpline NIAH | MrPro NIAH | 差值 |
|---:|---:|---:|---:|---:|---:|---:|
| 4K | 0.726667 | 0.325128 | +0.401538 | 0.887500 | 0.331250 | +0.556250 |
| 8K | 0.725513 | 0.148462 | +0.577051 | 0.878125 | 0.131250 | +0.746875 |
| 16K | 0.486154 | 0.072436 | +0.413718 | 0.600000 | 0.071875 | +0.528125 |

PPL在4/8/16K分别为TailSpline
`11.152472 / 10.117407 / 8.536196`，MrPro
`13.795244 / 14.356459 / 12.827048`；三个长度均更低。

13个任务的log-AUC差全部为正：最大为`niah_single_3 +0.875`，最小为
`cwe +0.0325`。任务族差为retrieval `+0.64453`、tracking `+0.37000`、
aggregation约`+0.16208`、QA约`+0.27563`，各自bootstrap区间下界均为正。

完整输出审计为：TailSpline/MrPro空输出`0/0`、EOS终止`257/248`、cap-hit
`133/142`（各390行）。两臂都有较高任务相关cap率，必须随主分报告；TailSpline的优势
不是由更高EOS率或单个任务独占造成。

GPU实测：TailSpline从09:11:46Z至09:19:06Z，MrPro从09:19:06Z至09:26:17Z；
两臂生成加PPL约14.5分钟，CPU配对报告约29秒。此前约9分钟为Full-13资产准备。

## 4. 原始资产与MrPro复用

| 资产 | 行数 | SHA256 |
|---|---:|---|
| TailSpline `generations.jsonl` | 390 | `9e985a9dc82d87dca8ed571e38871eeb9ae609bcdca709b83d3fee3f45f92ea3` |
| TailSpline `lm_rows.jsonl` | 138 | `8b061a37502eab47c3a176e7b36ee81b74a7c5f3303dba4a0c80b161110cfae9` |
| MrPro `generations.jsonl` | 390 | `78de95767a613fdf1f78f9d6ac7fdf81654213e9f48bb32e3b373bd9641a2644` |
| MrPro `lm_rows.jsonl` | 138 | `6a4b9b4e0720fd33e7d1b52f134dff938cf9d16ec463455b0032b6fbe01f2351` |

MrPro已登记到
`/root/autodl-tmp/mrrope_baselines/current/olmo2_1b_s4_full13_ppl46_mrpro`，
`ready_for_score_reuse=true`。只有checkpoint、prompt/data、table/gain、decoder、precision
与scorer完全匹配时才能复用分数。

## 5. 允许与禁止的结论

结合Llama A39，本次结果支持：exact TailSpline在两个不同模型家族、不同Native窗口和
不同canonical band的对应S4经典合同上，均整体胜过exact MrPro；OLMo的方向不是只由
一项任务产生。这使TailSpline达到可信方法候选，而不再只是Llama开发先验。

OLMo并非严格未接触的模型族：历史BM/front-loaded结果已经提供方向先验；准确表述应为
“本次exact TailSpline表未读取OLMo分数而冻结的前瞻确认”，不能称完全独立盲测。
不同tokenizer下的Llama/OLMo prompts也不能逐行跨模型配对。

更重要的是，OLMo上TailSpline与MrPro的`sum(m)`分别约`42.5676/37.6667`。当前结果
证明完整方法胜负，但不能区分总减速量、early transport和one-sided tail landing。
不得由巨大差值倒推roughness机制已经成立，也不得据此自动启动YaRN四格、调参或新曲线。
