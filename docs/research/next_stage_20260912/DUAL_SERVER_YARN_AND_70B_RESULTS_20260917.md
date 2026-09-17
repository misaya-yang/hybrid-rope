# 双服务器结果更新：官方静态YaRN与Llama-3-70B

本页是下一版论文的增量结果owner，不改当前论文正文。它只汇总两批已经生成正式报告的结果：

- 32GB服务器：Llama-3-8B与OLMo-2-1B的TailSpline、MrRoPE-Pro、官方静态YaRN直接比较；
- RTX PRO 6000：预量化`Llama-3-70B-Instruct` NF4检查点上的冻结TailSpline/MrPro尺度迁移。

## 结论

1. **70B尺度迁移成立。** 在S4/32K的四个完整读数中，TailSpline全部胜过MrPro：
   Full-13 `78.67/60.95%`、NIAH-8 `86.25/70.63%`、Natural-QA631
   `49.28/48.39%`、PPL5 `2.495/2.508`。构造直接复用Llama-3-8B的冻结表，
   没有读取70B权重或结果重新选band、gain或曲线。
2. **直接YaRN比较总体支持TailSpline，但不是每个自然任务都获胜。** Llama-8B和OLMo上，
   TailSpline均在NIAH、PPL和Full-13领先YaRN；Natural-QA631中，OLMo仍由
   TailSpline领先，Llama则由YaRN以`+0.67pp`小幅领先TailSpline。
3. **70B的S16/128K只完成PPL。** PPL10为TailSpline `2.245`、MrPro `2.313`，
   NLL差`−0.02981`。NIAH的TailSpline完成80条、MrPro仅3/80；TailSpline单臂的
   multikey-1/2/3已为`40/0/0%`，说明多针在128K明显失效。补齐MrPro约需3 GPU小时，
   仅在后续预算允许时再做；它不是当前论文更新的必补项，也不能用partial行形成比较结论。

## 实验身份

70B检查点为`unsloth/llama-3-70b-Instruct-bnb-4bit`，NF4双量化、BF16计算；
与8B共享tokenizer、词表、chat template和BOS/EOS。S4与S16都使用canonical
band `[18,35]`，两臂共同gain分别为`1.1386294361`和`1.2772588722`。
评测为batch1，不使用padding；32K使用直接prefill。128K经Blackwell实测选择32K chunk，
单条完整生成峰值约89.8GB allocated、96.6GB reserved。因此论文只能写“70B NF4检查点”，
不能写成BF16/FP16 70B结果，也不能把70B与8B绝对分数差归因于模型规模。

### Pro6000：70B完整结果

| 条件 | 指标 | TailSpline | MrPro | TailSpline−MrPro |
|---|---:|---:|---:|---:|
| S4 / 32K | Full-13×10 | 78.67% | 60.95% | +17.72pp |
| S4 / 32K | NIAH-8（同一Full-13 raw的检索族视图） | 86.25% | 70.63% | +15.63pp |
| S4 / 32K | Natural-QA631 | 49.28% | 48.39% | +0.89pp |
| S4 / 32K | PPL5 | 2.4954 | 2.5075 | NLL −0.00484 |
| S16 / 128K | PPL10 | 2.2449 | 2.3129 | NLL −0.02981 |

Full-13中TailSpline胜10项、平1项、负2项；cap-hit宏均值为`0.77/8.46%`，
EOS率为`99.23/91.54%`。Natural-QA中TailSpline胜2WikiMQA、HotpotQA和
MultiFieldQA，MrPro胜NarrativeQA并在Qasper近乎持平。固定面板正式成绩是论文主读数；
报告内bootstrap只作为换样敏感性，不否决固定面板成绩。

### 4080：Llama/OLMo三方法完成结果

| 模型 | 指标 | TailSpline | MrPro | YaRN | 最优 |
|---|---|---:|---:|---:|---|
| Llama-3-8B S4/32K | NIAH-8×200 | 80.42% | 69.56% | 70.97% | TailSpline |
| Llama-3-8B S4/32K | PPL46 | 4.1106 | 4.1618 | 4.1495 | TailSpline |
| Llama-3-8B S4/32K | Full-13×10 | 70.38% | 58.35% | 55.88% | TailSpline |
| Llama-3-8B S4/32K | Natural-QA631 | 41.08% | 40.88% | 41.75% | YaRN |
| OLMo-2-1B S4/16K | NIAH-8×200 | 65.55% | 5.36% | 6.27% | TailSpline |
| OLMo-2-1B S4/16K | PPL46 | 8.5362 | 12.8270 | 12.1410 | TailSpline |
| OLMo-2-1B S4/16K | Full-13×10 | 52.24% | 10.60% | 9.81% | TailSpline |
| OLMo-2-1B S4/16K | Natural-QA631 | 24.92% | 21.62% | 22.27% | TailSpline |

Full-13×10的TailSpline/MrPro行来自既有大面板的相同前10条/task，YaRN只生成缺失臂；
它是严格配对三方法比较，但不替换Llama/OLMo已有200条/task的TailSpline–MrPro主结果。
“官方静态YaRN”指零训练安装，不等于用YaRN训练或微调过的公开checkpoint。

## 对下一版论文的直接修改建议

1. 在跨尺度结果表加入一行“Llama-3-70B-Instruct NF4，8K→32K”，并同时列
   Full-13、PPL和Natural-QA；正文只需一句说明冻结表从8B直接迁移到70B仍全面胜MrPro。
2. 将Llama/OLMo原来的YaRN quick替换为本页大样本读数；Qwen/GLM仍沿用各自128K正式报告。
3. 在附录只报告70B S16/128K PPL10。任务侧已有TailSpline多针失效且缺完整MrPro基线，
   约3 GPU小时的补齐工作后置；当前不放半成品，也不把它列为论文提交前缺口。
4. 显式保留两个边界：70B是NF4而非BF16；70B尚未运行YaRN，因此不能声称70B上胜YaRN。
5. 论文主张应写成“跨模型、跨规模的整体优势及条件性局部反转”，不写成每项任务普遍支配。

## 证据入口

- [70B实验说明与执行身份](../../../experiments/llama70b_scale_20260916/README.md)
- [70B S4/32K Full-13](../../../experiments/llama70b_scale_20260916/reports/ruler13x10.json)
- [70B S4/32K Natural-QA631](../../../experiments/llama70b_scale_20260916/reports/naturalqa631.json)
- [70B S4/32K PPL5](../../../experiments/llama70b_scale_20260916/reports/ppl5_32k.json)
- [70B S16/128K PPL10](../../../experiments/llama70b_scale_20260916/reports/ppl10_128k.json)
- [70B S16/128K TailSpline NIAH单臂摘要](../../../experiments/llama70b_scale_20260916/reports/niah128k_tailspline_summary.json)
- [70B S16/128K MrPro中断进度](../../../experiments/llama70b_scale_20260916/reports/niah128k_mrpro_partial_live.json)
- [70B Blackwell 128K prefill回执](../../../experiments/llama70b_scale_20260916/reports/prefill_128k.json)
- [4080三方法报告索引](../../../experiments/iclr2027_strong_evidence_20260915/reports/README.md)

服务器逐行raw继续留在数据盘；Git中的报告与服务器正式报告SHA256一致。
