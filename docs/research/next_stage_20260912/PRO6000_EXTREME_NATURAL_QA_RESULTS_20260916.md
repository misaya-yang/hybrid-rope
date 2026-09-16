# RTX PRO 6000：极限长度与自然长文结果

## 结论

本轮冻结比较均为零训练、同checkpoint、同prompt的exact TailSpline与exact MrPro配对。

1. **Qwen S4/128K的最终Full-13×10三臂结果为TailSpline胜出。** 130条/臂、
   13任务等权分数为TailSpline `64.40%`、官方静态YaRN `62.04%`、MrPro
   `59.87%`；TailSpline分别领先`+2.36pp`和`+4.53pp`。
2. **GLM S4/128K给出新的模型结构迁移证据。** 130条/臂、Full-13×10为
   TailSpline `39.65%`、MrPro `30.78%`、YaRN `27.85%`；相对两条基线分别
   领先`+8.87pp`和`+11.81pp`。首批65条的`+12.79pp`没有在扩样后反转。
3. **GLM独立换书后，长书QA仍由TailSpline胜出。** 新书池为77题、15个来源簇，
   与原7本书零重叠；TailSpline/YaRN/MrPro的官方token-F1为
   `26.87/23.50/22.86%`。原书池对应为`28.61/27.32/25.28%`，两批均保持
   TailSpline第一。区间分析作为跨书稳定性附录，不替换这两个固定面板的正式成绩。
4. **Qwen自然长文主区间仍然成立。** Qwen2.5-3B在Native 32K到128K（S4）的
   原35题完整长书En.QA上，TailSpline为`19.14%`，MrPro为`15.90%`，差
   `+3.24pp`；加入YaRN后同面板三臂为TailSpline `19.14%`、YaRN `18.45%`、
   MrPro `15.90%`。
5. **极限长度表现按能力分化。** Qwen2.5-3B在256K（S8）的三项single-NIAH
   宏平均为`93.33%/40.00%`，TailSpline领先`+53.33pp`
   （区间`[+33.33,+66.67]pp`）；五篇InfiniteBench LongBook的PPL点估计为
   `30.34/23.93`，TailSpline的NLL高`0.238`，但五文档区间跨零且很宽，不能写成
   已确认的语言建模退化。

Llama-3-8B的S16/128K是压力测试而非论文主赛道。Full-13端点TailSpline领先
`+20.77pp`，ProofPile NLL低`0.0321`；InfiniteBench En.Dia为`12%/8%`
（`+4pp`，区间跨零），En.QA为`13.21%/20.17%`（`−6.96pp`），该设置下TailSpline落后。
另有完整答案字符串匹配T `7/41`、P `11/41`的输出诊断；它不等于官方token-F1，
不据此将两臂称为同处地板。En.QA来源为8个书籍context。结果保留为**Llama S16的任务画像**；
模型、倍率和输入条件须分别解释，不凭跨模型比较断言具体失败原因。

## 实验身份

| 条件 | 模型与倍率 | 数据与比较单位 | 主要指标 |
|---|---|---|---|
| Llama 128K gate | Llama-3-8B-Instruct，8K→128K，S16 | Full-13×10/task；10篇PPL文档 | task-equal RULER；token NLL |
| Llama自然长文 | 同上 | En.Dia 100对；En.QA 41对/8簇；完整100K–131K上下文 | 官方accuracy / token-F1，来源簇bootstrap |
| Qwen 256K健康检查 | Qwen2.5-3B，32K→256K，S8 | single-NIAH三任务×5；5篇LongBook | task宏平均；token NLL |
| Qwen 128K自然QA | Qwen2.5-3B，32K→128K，S4 | En.QA 35对/7簇；完整100K–128K上下文 | 官方token-F1，来源簇bootstrap |
| Qwen 128K Full-13 | 同上 | 13任务×10，130个严格配对prompt/臂 | RULER官方分数，任务等权端点macro |
| GLM 128K Full-13 | GLM-4-9B-0414，32K→128K，S4；32个rotary pair | 13任务×10，130个严格配对prompt/臂 | RULER官方分数，任务等权端点macro |
| GLM第二书池 | 同上 | En.QA 77对/15簇；输入69,660–95,486 tokens | 官方token-F1，来源簇bootstrap |

全部生成使用BF16、PyTorch Flash-SDPA与KV cache；不是vLLM/FP8结果。候选与基线
使用同一运行时。Qwen 128K QA的逐题结果为TailSpline胜13、MrPro胜6、平16。

## 样本量与解释边界

Llama S4/32K已完成200条/task总体中，完整差值为`+11.72pp`；source-order前10条为
`+12.04pp`。随机无放回抽10条/task的有限总体95%采样区间为
`[+5.72,+17.90]pp`，20,000次均为正；只有`0.21%`达到S16/128K gate的
`+20.77pp`。S4/32K与S16/128K同时改变倍率和绝对长度，不能把两者差异归因为样本量。

## 可移植报告

- [Llama S16/128K gate](../../../experiments/iclr2027_strong_evidence_20260915/reports/pro6000_llama_s16_128k_gate.json)
- [Llama S16 En.Dia](../../../experiments/iclr2027_strong_evidence_20260915/reports/pro6000_llama_s16_infinite_en_dia.json)
- [Llama S16 En.QA](../../../experiments/iclr2027_strong_evidence_20260915/reports/pro6000_llama_s16_infinite_en_qa.json)
- [Qwen S8/256K PPL与single-NIAH](../../../experiments/iclr2027_strong_evidence_20260915/reports/pro6000_qwen3b_s8_256k_health.json)
- [Qwen S4/128K En.QA](../../../experiments/iclr2027_strong_evidence_20260915/reports/pro6000_qwen3b_s4_128k_en_qa.json)
- [Qwen S4/128K Full-13×10三臂](../../../experiments/iclr2027_strong_evidence_20260915/reports/pro6000_qwen_s4_128k_full13x10_triarm.json)
- [GLM S4/128K Full-13×10三臂](../../../experiments/iclr2027_strong_evidence_20260915/reports/pro6000_glm_s4_128k_full13x10_triarm.json)
- [GLM S4/128K第二书池三臂](../../../experiments/iclr2027_strong_evidence_20260915/reports/pro6000_glm_s4_128k_naturalqa_second_books_triarm.json)
- [RULER抽样稳定性](../../../experiments/iclr2027_strong_evidence_20260915/reports/pro6000_ruler_sampling_stability.json)

原始逐行生成保留在实验数据盘。confirm40、256K Natural-QA和Llama32K截断控制
没有作为本轮完成结果。后续32GB队列的Llama/OLMo大样本YaRN与Natural-QA已经完成，
单独登记在[便携报告索引](../../../experiments/iclr2027_strong_evidence_20260915/reports/README.md)，
不与本批Qwen/GLM 128K结果混写。下一版取舍见[准备清单](PAPER_NEXT_REVISION_PREPARATION_20260916.md)。
