# 面向强接收与突出研究评价的实验计划

更新：2026-09-15。**交付状态：执行规格已完成；本轮没有启动GPU、下载数据或修改远端。**
目标是增强论文的可推广方法价值和科学认识，不把模拟评分当作录用概率。

## 1. 核心决策

论文主体保持为 **z的作用、位置结构、模型使用关系与目标范围内的质量**。
TailSpline是主要冻结构造；Cosh保持辅助外推实例，不再重跑432M学习实验。

新增实验要交付四个清楚的答案：

1. **固定公式跨模型：**同一公开参数规则，能否在统一clean合同下取得跨模型收益？
2. **真实上下文质量：**扩展后的自然长文是否更好用，而非只增加一个可运行长度？
3. **配置内部的作用：**完整allocation收益中，等总位移后剩余形状差异是否具有实际作用？
4. **部署取舍：**同一张表在native、2L、4L及更高倍率上的表现如何？

**推荐主线：跨模型clean矩阵 + 自然长文 + clean等位移对照。**
S16/128K是并行的规模迁移检查，不以最大s统领论文。
新Native增强路线暂不纳入投稿关键路径；已有Native-Z5结果保留原有研究身份。

本计划提供后续执行范围，不覆盖现有运行队列。当前Full20状态、48GB入口及停放任务以
[服务器任务分层](../../../experiments/iclr2027_three_track_sprint_20260915/SERVER_TASK_LAYERS.md)为准。
YaRN仍后置；本计划列出其价值，不自动改变其停放状态。

## 2. 已有资产：直接复用

| 资产 | 已有结论 / 用途 | 后续动作 |
|---|---|---|
| Llama clean32K，13×200 | T/P 68.27/56.54%，+11.72pp | 固定为主锚点，复用全部T/P输出 |
| Llama clean16K，13×50 | +3.39pp；QA +8pp | 固定为2L锚点，复用T/P的两个任务分片 |
| OLMo classic4/8/16K | Full13 AUC +49.23pp | 保留；新增clean合同，不重新生成classic |
| Qwen2.5-3B S2 Core6 | 32K正点估计、64K负点估计，AUC接近零 | 保留；新增预先固定的S4 Full13，不从小面板选择任务或调表 |
| Natural-QA631 | 完整配对、总体F1接近 | 不扩充同一短输入池来追求显著性；新增不同长度与任务的自然评测 |
| Native8K任务、PPL46及ProofPile32 | 已量化任务取舍和PPL长度响应 | PPL直接复用；新clean native任务只补统一合同 |
| E1 classic T/C | −0.41pp，跨batch诊断 | 保留；不重启旧batch2包装脚本 |
| NIAH Full20 | 当前owner记录运行中 | 先让现有运行完成，再读取正式报告；不重复做heatmap网格 |
| S16/128K gate | CPU资产已就绪；GPU待执行 | 在48GB以上机器复用现成入口 |
| 151.9M、432M、750M | 配置识别和辅助外推证据 | 复用，不重训、不因文件位置变动重跑 |

结果以[关键实验罗盘](KEY_EXPERIMENT_COMPASS_20260914.md)及其链接的结果owner为准。
本表是规划基线，不声称检查过此刻的GPU状态。

## 3. 固定实验清单

以下样本数是执行前的固定规格，不能逐批增加直到某个p值过线。
所有计数为新增生成次数；已有T/P不计入新增成本。

| ID / 优先级 | 实验 | 新增工作量 | 代码状态 | 主要论文落点 |
|---|---|---:|---|---|
| X1 / P0 | OLMo S4，clean8K/16K，Full13×50，T/P | 2,600次生成 | 核心准备器/评估器可用，需通用包装器 | 第二模型的同合同确认 |
| X2 / P0 | Qwen2.5-3B S4，clean64K/128K，Full13×50，T/P | 2,600次生成 | 同上 | 更长native范围下的固定规则迁移 |
| X3 / P0 | Llama S4，LongBench-v2实际输入8K–32K的完整可容纳子集，T/P | 2N；N由CPU长度普查确定 | 需数据适配与官方评分接入 | 自然长文质量 |
| X4 / P1 | Llama S4，C在原clean16K/32K面板；复用T/P | 3,250次生成 | C构表和评估器可用，需新clean包装器 | 完整配置效应与残余形状的分解 |
| X5 / P1 | Llama8K clean Full13×50，Native/T/P | 1,950次生成 | 通用包装器复用X1实现 | 同一部署表的native取舍 |
| X6 / 并行 | 已冻结Llama S16/128K gate，T/P | 260次生成 + 20个128K LM文档前向 | **现成可执行**，48GB以上 | 倍率迁移与原论文长度场景 |
| X7 / 条件增强 | S16自然任务En.Dia/En.QA与独立Full13确认 | 见§8 | 部分复用X3；需新增适配 | 超长真实应用与高倍率确认 |
| X8 / 后置 | 官方YaRN在原clean32K面板单臂 | 2,600次生成 | 表和评估器可用，clean包装器待接入 | 基线横向定位 |

X1/X2/X4/X5/X6合计 **10,660次新增生成**，另加X3的2N次与S16的20个LM文档前向。
这不是要求先全部完成才更新论文；每个完整实验独立形成结果。

### 工作量与调度

按标称输入上限估算，X1约32M、X2约256M、X4约96M、X5约16M、X6约34M输入tokens；
真实长度通常更短。这些只是容量上界，**不等于相同GPU时间**：长序列注意力成本非线性。
利用现有日志或正式任务前32条的实际耗时更新ETA，继续原任务，不另开重复测速网格。

- 32GB队列建议：当前Full20 → X1 → X3 → X4 → X5。
- 48GB以上队列建议：X6现成gate → X2 → 符合§8条件时X7。
- X8保持最后，只有执行任务明确包含YaRN时启动。
- 每张GPU同一时间一个评估进程；CPU数据准备可与另一张GPU的评估并行。
- 预算只够一包时：优先X1+X3；再加X4。不要先耗在更多单针heatmap或重训Cosh上。

## 4. X1/X2：固定规则的跨模型clean矩阵

### 合同

| 模型 | 已有本地路径 | Native L | 固定s | 测试输入上限 |
|---|---|---:|---:|---|
| OLMo-2-0425-1B-Instruct | `/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct` | 4096 | 4 | 8192、16384 |
| Qwen2.5-3B-Instruct | `/root/autodl-tmp/rope_qwen_baseline_20260907/model` | 32768 | 4 | 65536、131072 |

构表使用`tables.py analytic`中的exact `tailspline`、`mrpro`；原生config决定band，
不通过新分数调band、gain或系数。两臂共用 `gain=1+0.1*ln(s)`，每个模型一张静态表服务两个长度。
模型路径按已有owner复用；若租赁磁盘路径改变，定位已有checkpoint即可，不默认重新下载。

数据使用固定RULER revision `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`，
全13任务，每任务每长度50条source-order记录，按各模型tokenizer重新生成目标长度输入。
T/P共享同一批prompt IDs，batch1，无内容填充。跨模型输入不假装是逐token相同的prompt。

**首要终点：**各模型4L的Full13任务等权差；2L及family/task分解是预先声明的次要终点。
每个长度报告绝对分数、配对差和95%区间。可附多模型描述性平均，但不把所有行混成一个
“跨模型样本量”来产生虚假的模型泛化精度。

**怎样形成强结果：**不仅报告一个总平均，而是展示同一固定公式在完整模型×长度矩阵的表现。
若某一模型接近或反转，照原合同完成两臂并保留，解释为已测条件下的差异；不换模型寻找赢家。
这会增加关于配置适用条件的认识，不自动否定已成立的Llama结果。

## 5. X3：真正的自然长文证据

使用官方 **LongBench-v2** 数据和direct-answer提示方式。它是多项选择题，可减少自由回答
F1与格式解释的歧义。官方入口：
[数据与评测说明](https://github.com/THUDM/LongBench)、
[任务与长度说明](https://longbench2.github.io/)。

### CPU阶段先完成数据规格

- 加载官方数据；优先使用现有本地缓存。没有缓存时，由执行任务的数据获取范围决定下载。
- 用Llama tokenizer计算**完整实际提示长度**，包括指令、问题和选项。
- 主池：输入大于8192 tokens，且输入加固定生成预算能装入32768；不依据官方的words分组代替tokens。
- 使用全部可容纳记录，按`_id`稳定排序；不人工截断长文、不补无关文本、不按输出选择题目。
- CPU普查输出N、六类任务实际覆盖、8K–16K/16K–32K分层数量和源上下文簇数；不存在的类别不填零分。
- 生成预算与官方direct-answer流程保持一致，并在两臂前固定。若本地适配需要显式预算，使用
  128新tokens并记录为本地direct-answer合同；不混称原榜单的所有解码设置。

### GPU与评分

Llama使用现有S4 T/P静态表、相同prompt、同runtime、greedy生成。保留所有回答，使用官方
选项解析与准确率。首要终点是整个可容纳池的配对accuracy差；长度/领域分层为次要结果。
重复源上下文按簇处理不确定性，单独报告实际N，不把多个相关问题当作独立文档。

Natural-QA631保持原结果。X3测试新的问题和长度条件；其成功不能覆盖旧结果，旧结果也不预先
决定X3成败。若新任务仍接近，保留“RULER改善、该自然任务近似持平的观测”，不编造推理瓶颈机制。

## 6. X4：等位移对照，检验更细的配置作用

**问题不是“总位移是否混杂了z”。** 固定native表和目标范围后，总位移是z的统计量。
T/P已经是完整allocation比较；T/C进一步问：相同总位移下，这两个明确形状的表现是否不同？

- C使用现有解析 `tailspline_dose_control`，不拟合新的曲线。
- 在原clean32K 2600行与clean16K 650行上仅生成C。
- 完整复用同prompt的T/P；16K的nonqa11和qa2分片保持各自顺序与相同batch1运行方式。
- 从现有T运行合同沿用backend、精度、prefill策略、gain、生成预算；新输出根，禁止覆盖旧classic C。
- 首要终点：32K T−C Full13差；16K、task/family及C−P为次要比较。
- 同时呈现T、C、P三者，不把C−P称为“纯剂量效应”，因为两张具体表仍可能具有其他形状差异。

| 结果形态 | 有价值的解释 |
|---|---|
| T明确优于C | 支持在该等位移比较中残余形状有贡献；不直接升级为普适平滑定理 |
| T/C接近且都优于P | 支持更广的有效transport形状；说明特定残余形状不是这组数据上可分辨的主要因素 |
| C优于T | 保留对照胜出，说明当前目标的任务选择仍有空间；不在同一确认集改TailSpline参数 |
| 区间很宽 | 该对照精度有限，不能把不显著当等价；不逐批追加直到得到偏好的结论 |

### 机制深化：先用已有输出，再决定是否前向

先做所有配对行的“双方都对 / T独对 / P独对 / 双方都错”、任务族和长度分解，输出健康作为
单独维度；全样本官方分数始终是主结果。按非空输出筛选的分数不代替主结果。

如果要增加token-level分析，新增 **gold-reference score-only** 路径：在预先固定的全部单答案
检索样本上评分原始参考答案及source生成的错误候选，复用同一prefix KV。只作分析，不据此选表。
不要复用`prepare_margin_targets.py`作为确认集选择器：它挑选已经成功的teacher输出，回答的是另一问题。
该诊断不属于X4的完成前提，也不把相关性报告成中介因果证明。

## 7. X5：用统一clean合同展示native取舍

Llama8K，Full13×50，Native/T/P各650行。T/P沿用S4静态表；Native使用原始频率和**原始gain=1**。
数据使用新source-order block。无长度自适应切表，无padding。

首要终点：T−Native在8K的配对任务差；P−Native和T−P帮助解释扩展方法的共同代价。
原有PPL46和ProofPile32结果直接复用，不为这张任务表重跑LM。

交付一张清楚的表：native成本、2L收益、4L收益，各自样本数和区间。不同面板不拼成同一个
联合置信区间。这里目标是把代价测清楚，不要求先证明无损，也不根据已见分数临时设置非劣阈值。

## 8. X6/X7：S16迁移与超长自然任务

### X6已经可执行

只在克隆数据盘的48GB以上服务器，进入`/root/autodl-tmp/hybrid-rope`后运行：

```bash
bash experiments/iclr2027_three_track_sprint_20260915/run_llama_s16_128k_gate_48gb.sh
```

现有入口会检查显存并在实际GPU上选择generation/LM的prefill策略，复用已准备的CPU资产。
预期每臂 `rows=130, lm_rows=10`，输出：
`tailspline_llama_s16_128k_gate/reports/tailspline_vs_mrpro_s16_128k_gate.json`。
这是Full13×10的gate，不把130条筛查写成大样本超长确认。

### X7的推进逻辑

X6完成后，无论方向如何都先生成报告。任务差正且配对区间支持正方向时，优先做新的独立
128K Full13×50确认（每臂650条）。如果gate未分出方向，就保留gate结果，回到X1–X5，
不边看分数边增加gate样本。PPL独立报告，不与任务分数平均，也不把PPL近零差当作任务失败。

独立128K确认成立后，再补S16的8/16/32/64K各Full13×20，形成相同S16表的完整区间响应。
旧S4分数不填入S16曲线。Native8K参考只复用合同一致的原始Native臂；S4的T/P不能冒充S16。

自然任务优先对齐MrRoPE的实际应用：
[InfiniteBench官方任务](https://github.com/OpenBMB/InfiniteBench/blob/main/README.md)中的
`longdialogue_qa_eng`（En.Dia）和`longbook_qa_eng`（En.QA）。每任务取source-order前100个
完整可容纳记录；不足100则用全部并报告N。按实际Llama token预算过滤，不截断后假装原任务。
先固定全部题目和官方prompt/scorer，再执行两臂。总计至多400次生成。
这个结果比再加一张单针热图更能增强超长应用价值。

## 9. X8：后置的强基线，不启动旧队列

YaRN的确能帮助说明实用比较位置，但当前作者路线将其后置。若后续执行范围明确包含X8：
只在现有clean32K 2600条上运行一个YaRN臂，复用全部T/P，不扩大基线网格。

构表选择 `tables.py analytic --method yarn`，明确其真实公式、gain及共享项；不能用历史
index-ramp冒充官方YaRN。若实际端点无法完全匹配，准确写出变化对象，不修改定义强行匹配。
使用两个预先声明的T/P、T/YaRN比较，并报告相应多重比较处理；现有单比较主结果仍保留。
`run_naturalqa_yarn.sh`是QA631入口，**不能拿它代替clean32K入口**。

## 10. Codex实现与执行清单

### 已有可调用核心

- `experiments.llama3_60dir_20260911.prepare_planb_panel`
- `experiments.fixed_rope_three_interfaces_20260913.prepare_tailspline_llama_32k_ruler200_clean`
- `experiments.fixed_rope_three_interfaces_20260913.tables`
- `experiments.olmo_recovery_20260912.recovery_v2_eval`
- `experiments.fixed_rope_three_interfaces_20260913.matched_generation_report`
- `experiments.fixed_rope_three_interfaces_20260913.matched_naturalqa_report`

### 必须先实现，不能伪称已有的入口

在`experiments/iclr2027_strong_evidence_20260915/`增加以下薄包装，不重写模型评估内核：

| 新文件 | 职责 | 实现完成判据 |
|---|---|---|
| `prepare_clean_transfer.py` | 调用既有source-order准备器与unpadded converter；显式model/caps/counts/seed/QA offset | 正确模型名与tokenizer身份；13×50完整；不产生LLAMA字样的Qwen/OLMo合同 |
| `run_clean_matrix.py` | X1/X2/X5逐臂执行、续跑、报告 | 默认打印计划；`--execute`才运行；单GPU锁；已有完成臂不重复 |
| `run_clean_c.py` | X4复用现有T/P，只跑C，支持16K分片 | 输出新根；不得调用旧batch2 C launcher；T/C精确配对 |
| `prepare_natural_long.py` | X3/X7数据普查、官方prompt适配、完整长度筛选 | 不读取模型输出；保存完整候选清单及可容纳原因；记录实际N |
| `run_natural_long.py` | 官方任务评分、两臂生成、配对报告 | 全输出保留；MC/En.Dia/En.QA各用官方对应评分，生成器的RULER contains字段不作为这些任务成绩 |
| `summarize_matrix.py` | 各完整实验汇总成模型×长度表 | 不混合clean/classic/S2/S4/S16，不将行bootstrap解释为跨模型不确定性 |

包装器参数固定为：`--model`、`--model-id`、`--data-root`、`--out`、`--scale`、
`--lengths`、`--rows-per-task`及`--execute`（准备器无execute）。运行器还显式接收
`--data-manifest`和`--python`，避免隐藏依赖某个模型的LM manifest。
默认模型前向策略是已有的BF16、batch1；prefill根据实际机器及现有runtime经验选择。

验收只测试容易错的实际行为：model身份、source-order无padding、两臂prompt对应、
续跑不重复、未知/空/截断回答的官方评分，以及报告只覆盖完成臂。数学构表直接复用已有检查。

### 现有核心命令示例（参数已对照当前源码）

下例准备一项OLMo任务；新包装器循环13个任务与两个长度。执行前将`OUT`设为新的实验根。

```bash
cd /root/autodl-tmp/hybrid-rope
export PYTHONPATH=.
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
UPSTREAM=/root/autodl-tmp/rope_qwen_baseline_20260907/ruler_upstream/RULER-c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
OUT=/root/autodl-tmp/today_rope_plan_20260914/strong_clean_olmo_s4

"$PY" -m experiments.llama3_60dir_20260911.prepare_planb_panel \
  --model "$MODEL" --model-contract generic --upstream "$UPSTREAM" \
  --out "$OUT/source_parts/8192/niah_single_1" --stage H --contract planb \
  --tasks niah_single_1 --caps 8192 --counts-by-cap 8192:50 \
  --selection-mode source-order --source-only --qa-base-offset 5600 --seed 20261101

"$PY" -m experiments.fixed_rope_three_interfaces_20260913.tables analytic \
  --config "$MODEL/config.json" --method tailspline --scale 4 \
  --candidate-id strong_olmo_s4_tailspline --model-id olmo2_1b \
  --role candidate --changed-variable internal_frequency_allocation \
  --out "$OUT/tables/tailspline.json"
```

QA offset5600与seed20261101是**新计划的预先指定值**，不是历史运行参数。原提议的7000
超过当前SQuAD源的5928行；5600可容纳单长度200行，也让两长度各50行使用5600–5699，
并与既有5800起始块分开。包装器先检查上游源数据能供给完整样本数；不足时在任何GPU输出
产生前形成完整新清单，不截断任务集合。
`prepare_planb_panel`支持`--model-contract generic`与`--source-only`；必须使用，避免继承Llama身份检查。
现有clean converter虽支持`--model`、`--length`、`--rows-per-task`，但manifest名称仍硬编码LLAMA，
因此先做上述小范围通用封装，不能直接把错误标签带入新模型报告。

单个clean生成臂的已有内核用法如下。`PANEL`是该模型新clean输入，`DATA`是该模型已存在的
数据manifest（`--skip-lm`时不新增LM评估），`TABLE`是该臂已冻结表，`RUN`是新臂目录：

```bash
"$PY" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
  --data "$DATA" --model "$MODEL" --arm Native \
  --extra-panel "$PANEL" --only-extra-panels --skip-lm \
  --length-cap "$LENGTH" --prefill-chunk-size 8192 --batch-size 1 \
  --static-table-json "$TABLE" --table-label "$LABEL" --out "$RUN" --execute

"$PY" -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
  --source "tailspline=$T_RAW" --source "mrpro=$P_RAW" \
  --candidate tailspline --baseline mrpro --length "$LENGTH" --out "$REPORT"
```

这些环境变量由包装器显式赋值，不是未填路径也能运行的完整队列。
报告器遇到已有输出会拒绝覆盖；续跑时复用完整报告，新分析写新文件。

## 11. 接手任务可直接粘贴的执行指令

> 以本计划为实验规格，先读取SERVER_TASK_LAYERS确认当前运行任务与已完成资产。
> 保持z研究主线、TailSpline固定规则和Cosh辅助定位。先完成§10列出的必要薄包装，
> 给出真实的plan输出与相应小范围测试，然后按本次明确授权的实验ID执行。
> 每个完整实验生成正式报告，复用现有T/P和学习实验，不按中途分数改方法或样本量。
> 48GB入口复用冻结S16资产，不启动旧original/clone队列。若提供的执行授权是X1–X7，
> YaRN继续停放；只有明确包含X8时才运行。新checkpoint、数据下载和租赁不从文档存在中推断授权。
> 交付完成矩阵、原始输出位置、配对报告、实际GPU耗时、论文可写的新认识及相应图表。
> 没有产生模型结果的入口只标记已实现/已准备，不写成实验完成。

## 12. 最终论文产物

1. **主表：**Llama/OLMo/Qwen的clean2L/4L，固定公式、不按模型调参。
2. **自然任务表：**X3及可执行时X7，与已有Natural-QA631并列保留。
3. **配置对照图：**T/P/C三张明确表、完整任务效果与同位移残余比较。
4. **部署取舍图：**同一静态表在native与扩展窗口的实测变化；S16独立成图。
5. **研究结论：**哪些收益随模型、范围和任务而变化；哪些是整体allocation效应，哪些细形状差异
   已被测量。只有实际完成的比较进入论文，不能把这份计划写成将会成功的结果。

这样的实验组合有机会把“强单项结果”提升为“可迁移的方法 + 可复查的结构认识 + 实际部署价值”。
优先级来自这些研究收益，而非为了让某个模拟审稿视角机械多打一分。
