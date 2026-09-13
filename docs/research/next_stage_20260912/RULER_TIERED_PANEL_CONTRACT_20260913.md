# 固定表 RULER low / mini / medium / high 四级评价合同

更新：2026-09-13。本页将候选筛选与统计确认分层，避免用每格4行的离散波动宣布
胜负，也避免每张开发候选直接支付完整 RULER 成本。四级面板严格嵌套；候选只能
向上确认，不能根据高层结果返回增加新 band。

## 满规模基准与三级规模

本合同把 `13 tasks × 5 lengths × 50 rows/cell = 3250 rows/arm` 定义为100%
full RULER 时间基准。任务集合严格嵌套：

- Core-6：`niah_single_2`、`niah_multikey_2`、`niah_multiquery`、`vt`、
  `fwe`、`qa_1`；
- Medium-9：Core-6 + `niah_multivalue`、`cwe`、`qa_2`；
- High-13：Medium-9 + `niah_single_1`、`niah_single_3`、
  `niah_multikey_1`、`niah_multikey_3`。

| 版本 | 任务 | 长度 | 每格行数 | 总行数/臂 | full成本比 |
|---|---:|---:|---:|---:|---:|
| low | 6 | 3 | 6 | 108 | 3.32% |
| mini | 6 | 3 | 18 | 324 | 9.97% |
| medium | 9 | 4 | 22 | 792 | 24.37% |
| high | 13；保留medium，新增29个cell | 5 | 旧cell 22；新增cell 18 | 1314 | 40.43% |

长度同样嵌套。跨模型、跨倍率机制研究统一报告

\[
x=\log_2(L/W),\qquad u=x/\log_2S,
\]

其中`x`表示相对Native窗口的doubling数，`u`表示到目标horizon的进度；同时保留
实际token长度。mini使用首、中、末三个`x`点，medium/high逐步补齐半doubling网格。
成熟历史网格优先复用；新模型可用

\[
L_j=W S^{j/4},\qquad j=0,1,2,3,4
\]

并量化到可生成的token bucket。AUC始终使用实际长度，不假装量化后仍等距。

| 模型/目标 | high五点 | mini | medium新增 | high最后新增 |
|---|---|---|---|---|
| OLMo W=4K, S=4 | 4/6/8/12/16K | 4/8/16K | 6K | 12K |
| Llama W=8K, S=8 | 8/16/32/48/64K | 8/32/64K | 16K | 48K |
| Qwen W=32K, S=2 | 32/40/48/56/64K | 32/48/64K | 40K | 56K |
| Qwen W=32K, S=4 | 32/48/64/96/128K | 32/64/128K | 48K | 96K |

greedy decoding没有可解释为独立重复的“模型seed”。数据每格使用两个冻结seed：
low为3+3，mini累计到9+9，medium累计到11+11；high新增cell为9+9。low行必须是
mini的严格前缀。先纳入已有冻结行，只为缺口生成第二seed，不为凑整重做完整基线。
缺失时固定数据seed为`137`与`20260913`；seed只决定数据抽样。

## 平均与统计口径

主分数不是逐行W/L，而是：

1. 每个任务×长度cell先对row取 official mean；
2. 每个长度对任务等权平均；
3. 在实际 `log(length)` 上做归一梯形AUC。

\[
A_m=
\frac{\sum_j\frac{M_j+M_{j+1}}2
\bigl(\log L_{j+1}-\log L_j\bigr)}
{\log L_{max}-\log L_{min}}.
\]

同一模型不同S应复用相同semantic case ID；跨模型尽量复用同一官方生成器seed与
任务参数，按semantic ID配对，而不是要求token IDs相同。每个候选与基线必须先按
row ID、prompt hash配对。置信区间使用20,000次paired
hierarchical bootstrap：

- benchmark任务集合固定，不把13个任务伪装成随机抽样；
- 在每个任务内按同一source case cluster重采样；
- 同一case跨长度、跨候选和基线保持绑定；
- 若长度间没有同源case，则在任务×长度cell内配对重采样；
- 每次重新计算完整任务等权曲线与log-AUC；
- 每次在draw内重新取`worst_length=min M(x)`，不能先选观察到的最差长度；
- 报告`delta AUC`区间、worst-length、逐长度差、任务族差、跨S/跨模型交互与
  bootstrap `P(delta>0)`。

逐行W/L/T只用于定位失败模式，不能替代平均分、AUC或区间。

## 逐级升级规则

每个候选形成连续结果向量：

```text
[delta log-AUC, delta Native, delta endpoint,
 worst task-family delta AUC, delta EOS rate, delta cap-exhaustion]
```

1. 只有实现、配对、表身份或评分协议无效才直接作废运行。
2. low后保留uncertainty-aware Pareto前沿；只有另一方案所有质量轴均不差，且
   至少一轴的配对区间明确更好，才判dominated。
3. `delta AUC`为正，或修复独立任务族且没有整体崩塌的方案进入mini；均值略负、
   区间跨零或有互补价值的方案不因low噪声删除。
4. Pareto点过多时，每个同构band/shape机制只升级一个代表；其余标为deferred，
   不写成失败。
5. 若根据low改了band或gain，必须生成新candidate ID；low成为开发证据，mini
   新增216行作为stage-only确认，同时另报累计324行均值。
6. medium后冻结table、gain和band再进入high；禁止看high新cell后继续调参。
7. high同时报告累计1314行AUC、high新增block的独立结果、相对BM/MrPro的
   Pareto关系和任务反转。

low只能排除身份错误、EOS/cap灾难和明显全局崩塌，并观察平均曲线方向与任务族
反转。它禁止用于宣称击败MrPro、用CI是否过零作生死门、按单cell删除方案或形成
论文结论。

最终“最佳”是high层非支配、平均AUC最强且没有明确Native/endpoint/内部长度灾难
的方案，不是某个单阈值或一个任务最大者。机制哨兵C42与Solver在完成连通矩阵前
不得因单个mini局部落后而删除；只有身份无效或表张量重复才可直接取消。

## 永久基线登记键

MrPro、BM、Native按benchmark身份只运行一次。缓存键至少包括：

```text
model revision
+ tokenizer / template hash
+ prompt collection / row hashes
+ task / length
+ decoding budget / mode
+ scorer / RULER commit
+ baseline table hash / band / gain
+ precision / arithmetic path
```

硬件只进入耗时receipt；换GPU不重跑科学分数。候选band/table不进入baseline键。
新增长度或prompt只补缺失cell；新checkpoint可复用任务、seed和统计代码，不能移植
旧模型分数。

## 当前资产、缺口与估时

- Llama已有三任务×五长度×4行，以及64K八任务×4行；mini主要缺任务广度和
  每格补到18行。
- OLMo已有16K七任务×50行；主要缺6/8/12K breadth、4K非core任务，以及16K
  core6补到22行。
- Qwen3B S=2已有core6的32K×2、64K×16；主要缺40/48/56K、非core7任务、
  32K补齐和64K core6补到22。
- Qwen S=4主要缺48/64/96K breadth，并需扩32/128K当前2/4行。

排除已缓存baseline后的单候选粗略时间：

| 版本 | OLMo S4 | Qwen3B S2 | Qwen3B S4 | Llama8B S8 |
|---|---:|---:|---:|---:|
| low | 约2--3分钟 | 约13--15分钟 | 约28--32分钟 | 约18--22分钟 |
| mini | 约7分钟 | 约40分钟 | 约1.5小时 | 约0.9小时 |
| medium | 约17分钟 | 约1.5小时 | 约3.5小时 | 约2.2小时 |
| high | 约28分钟 | 约2.8小时 | 约5.8小时 | 约4.1小时 |

估时允许约±20%，由长度、生成预算和EOS提前结束决定。比例在执行前用冻结基线
行的实测耗时加权校准，不能根据候选成绩改变面板。

## 当前机制研究关系

Band选择与跨模型、跨倍率机制实验由
[Band统一机制研究](BAND_NEXT_STAGE_RESULT_20260913.md)维护。low用于约100行快速趋势，
mini用于候选确认，medium用于冻结候选，high用于声明范围内的最终比较。PPL只作
健康检查和近似平局的次级诊断，不进入主要升级条件。
