# 扩大到跨文本分布的真实生成比较

用户要求扩大测试，避免同一重复背景生成器上的局部成功。该目录复用当前Qwen PM和NOSA PC2 reader；数据准备、评分和运行入口独立，未覆盖原实验。

## 已冻结面板

本地：`results/broad_position_eval_20260909/panel_v1`。

| 内容 | Qwen/PM | NOSA/PC2 | 识别的问题 |
|---|---:|---:|---|
| Qasper完整科学文档 | 8 DEV + 16 TEST | 8 + 16 | 是否覆盖科学文本中的问答需求 |
| MultiFieldQA英文完整文档 | 8 + 16 | 8 + 16 | 是否迁移到混合领域 |
| NarrativeQA完整叙事上下文 | 4 + 8 | 2 + 4 | 是否迁移到长叙事；同故事多问题不算独立文档 |
| 未用HotpotQA文档 | 8 + 16 | 3 + 5 | 是否保住新自然多跳需求 |
| 未用2WikiMQA文档 | 8 + 16 | 8 + 15 | 是否保住新跨文档关系 |
| 重复背景，16条竞争记录 | 8 + 16 | 8 + 16 | 旧背景风格下的多候选控制 |
| 自然prose背景，同16条记录 | 8 + 16 | 8 + 16 | 背景多样化后能否保持收益 |
| 自然prose背景，256条竞争记录 | 8 + 16 | 8 + 16 | 低频token很多、预算不足以保留所有记录时是否仍有效 |
| 新MRCR长对话，两种序数查询 | 8 + 16 | 不满足当前16K设置，未截断 | 是否保持不同发生次数的内容绑定 |
| 同MRCR保留全部8次竞争回答的compact对照 | 8 + 16 | 8 + 16 | 区分长历史干扰与基本发生次序读取能力 |

Qwen共228行，其中76 DEV行来自48个源文档/对话/材料家族，152 TEST行。NOSA共181行，61 DEV、120 TEST；覆盖差异由完整token长度决定，不能把两个模型的全表均值直接比较。

Qwen DEV完整输入范围：自然文档约2.7K–31.1K，控制检索约7.4K–7.9K，MRCR长对话19.7K–26.5K。NOSA保持现有16K评估设置，不新增RoPE频率/缩放。该设置不借由本脚本声称其配置文件原生窗口是16K。

自然文档不截断；控制检索显式使用prose片段，不冒称自然QA。MRCR来自公开的模型生成对话，采用新源对话及派生序数问题，不能称原版全量MRCR。所有旧PM DEV/TEST文档和已使用MRCR源对话均在选择前排除。新split在源文档/对话/背景家族级隔离。生成器只读来源及长度，不读模型结果。

## 执行入口

服务器独立目录：`/root/autodl-tmp/position_overnight_20260909/prepared/broad_panel_20260909`，下有`code/`与`data/`。

入口默认dry run；加`--execute`才运行。**入口自己持有既有`queue.lock`，不要再在外面包同一把flock，否则形成嵌套锁。** `--wait-for-lock`排在健康的现有作业后；不抢占、不停止当前锁定TEST。

```bash
BROAD=/root/autodl-tmp/position_overnight_20260909/prepared/broad_panel_20260909
TASKROOT=/root/autodl-tmp/position_overnight_20260909
TASKPY=/root/miniconda3/bin/python
cd "$BROAD/code"

# 1. 原锁定均衡候选，完整跨领域DEV。25%总prefix槽，rawV，H128，M256。
$TASKPY -m experiments.broad_position_eval.run --engine pm --sampling balanced --panel "$BROAD/data" --output "$TASKROOT/runs/broad_pm_balanced_dev_v1" --execute --wait-for-lock

# 2. 相同数据/预算/位置与value设置，仅换回均匀内容采样。
# 固定对手无需重复；本行只生成P/C/U，用于隔离采样的跨分布效果。
$TASKPY -m experiments.broad_position_eval.run --engine pm --sampling uniform --arms P C U --panel "$BROAD/data" --output "$TASKROOT/runs/broad_pm_uniform_dev_v1" --execute --wait-for-lock

# 3. NOSA：native、COBS适配、当前rank1候选、exact参考、dense参考。
$TASKPY -m experiments.broad_position_eval.run --engine pc2 --panel "$BROAD/data" --output "$TASKROOT/runs/broad_pc2_dev_v1" --execute --wait-for-lock

$TASKPY -m experiments.broad_position_eval.report --runs "$TASKROOT/runs/broad_pm_balanced_dev_v1" "$TASKROOT/runs/broad_pm_uniform_dev_v1" "$TASKROOT/runs/broad_pc2_dev_v1" --balanced "$TASKROOT/runs/broad_pm_balanced_dev_v1" --uniform "$TASKROOT/runs/broad_pm_uniform_dev_v1" --output "$TASKROOT/reports/broad_dev_v1"
```

先完成上述有区分力的跨领域DEV，不按单条结果换任务、删掉失败或挑seed。后续确认使用相同入口的`--split test`和新的输出目录；若DEV促成方法修改，先锁定该版本再打开新TEST。旧TEST已见结果，不再作为新独立确认。

GPU队列由接管任务统一安排；可在其外层顺序运行这些自带锁的入口。没有新的自动时间截止或关机指令。某个task没有该模型的完整输入时入口报错，不静默改长度；需要分任务调度时用`--tasks`指定manifest中的实际任务名。

## 判读与复用

- 自然QA沿用固定英文QA F1/EM；合成记录使用完整字符串加EOS；MRCR同时记录marker门控的SequenceMatcher、完整字符串加EOS及错误发生次数的整串命中。三类指标分别报告。
- PC2在预填充结束前尚未触发选块的输入单列`__short_prefill_control`。这种控制可以检查读取接口，不与真正启用稀疏路由的主比较混在同一任务均值中；生成很长时仍可能随后触发路由。
- P/C/U若共同改善，优先归因共享的内容/目标改变。均衡对均匀采样的对照在新来源上单独报告，不能只用对EA差值解释采样。
- exact_mass是访问所有raw K的诊断参考，不能改名为PC2或宣称25%常驻KV。dense与稀疏参考的计算支持不同，均保留原始CIS和实际reader语义。
- 每个原始输出包含代码/数据合同、任务分数、timings、KV bytes和配对执行峰值。PM总时间字段是现有runner的standalone估计，配对峰值不是部署单分支峰值；不由此直接宣布系统加速。
- 新数据没有旧输入可复用，必要的新基线需要计算一次。之后相同输入/预算/评分的固定基线自动复用；原始实验的基线缓存和结果不被改写。
- 报告按源文档/对话/材料家族做配对bootstrap，多个MRCR问题不冒充多个独立对话；来源只有几个的格子属于小样本证据。短板任务与全部退化都保留。

CPU验证涵盖两个真实tokenizer的query-blind边界、原文与token ID完整性、旧/新split隔离、256条记录完整映射、MRCR完整输出评分，以及实际tiny Qwen的缓存分支隔离。该验证不代表新面板已经取得模型效果。
