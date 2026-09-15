# 一轮独立审稿后的处理

审稿原文见[review](review.md)，冻结输入及隔离条件见[identity](identity.json)。审稿人仅收到PDF，未读源码、既有意见或对话；内部评分6/10、置信度4/5。以下是作者侧对记录的核对与一轮修正，不是第二次审稿，不宣称修改后分数变化。

| 意见 | 处理 | 依据与剩余问题 |
|---|---|---|
| 1 同面板简单基线及Native参照 | 部分采纳：主结果标题明确为TailSpline相对MrPro的匹配面板收益 | 当前两臂数据不拼接其他面板的Uni/BM/YaRN分数。新增同面板评估未执行，比较覆盖缺口保留。 |
| 2 TailSpline自身等位移任务归因 | 采纳措辞修正：Fig5由“Boundary design improves”改为“TailSpline improves over MrPro on the matched panels” | 已有T/C代数与证明保留；没有把目标最优性写成任务中介证明。E1任务证据仍未接入。 |
| 3 四单元绝对NLL | 未补数字 | 当前注册owner及可移植输入只提供配对差与逐seed差；本地未定位完整四单元记录。不能从差值反推绝对值，也没有重放模型评估。 |
| 4 RULER输入改造 | 部分采纳：正文明确depth-selected、prefix-padded；附录给出filler原文、换行、无特殊token编码、token级重复及截断规则 | 来源为已打包的prepare_planb_panel.py。未找到对应classic逐行输入镜像供汇总padding比例/实际深度误差，不编造分布。 |
| 5 适配代价 | 采纳：正文报告同一独立Q/K-only RULER适配的4/8/16K差为−29.75/+29.61/+4.65pp | 原metrics.json确认；该协议与Table1 EOS适配、QA适配分开。不用另一adapter的负分否定Table1。 |
| 6 几何到设计目标 | 部分采纳：明确两个独立spacing prior编码设计偏好，模型评估检验实用价值 | 保留正面理论衔接，不额外重复换表失败或添加普遍风险最优性要求；公式无改动。 |
| 7 开发/确认标签冲突 | 采纳并修复正文、G.4与G.8 | 原报告区分development（20260909）与seed_replication（20260910）；Uni/YaRN为existing_controls，在同一72行确认输入上追加。准确称BM/MrPro确认及同输入follow-up controls，不把追加控制当新的独立确认，也不把全部72行称原开发面板。 |

## 核对来源

- [BM结果报告](../../../../../docs/research/ROPE_OLMO_BM_RESULT_20260908.md)及[JSON](../../../../../docs/research/ROPE_OLMO_BM_RESULT_20260908.json)：development、seed_replication、existing_controls和bm_vs_existing_controls各自身份。结论为报告支持，未声称本轮重算这些生成raw。
- [Q/K适配metrics](../../../../../rebuttal/rebuttal_0723/theory_results/olmo2_qk_phase_adaptation_20260729/metrics.json)：ruler_13_family.qk_phase_adapted；每个长度EVQ−Native乘100，与正文三项差一致。
- [面板准备代码](../../../../../experiments/llama3_60dir_20260911/prepare_planb_panel.py)：padding_unit与irrelevant_padding_tokens生成规则。
- [151.9M注册owner](../../../evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.json)：现有配对差，非四单元绝对NLL。

本轮不启动训练、推理或新曲线，不修改原始实验数字与协议。
