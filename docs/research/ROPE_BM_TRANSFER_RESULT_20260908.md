# MrRoPE 论文3B上的BM冒烟结果

**结论：没有复现OLMo的长端收益，不建议据此直接进入全量。** Qwen2.5-3B-Instruct上，BM改善32K，但128K低于MrPro 7.29个百分点。本轮仅判断固定BM在该小面板的迁移，不宣称所有BM变体无效，也不能单独归因于模型容量。

## Material Passport

2026-09-08；状态COMPLETE / NO_LONG_GAIN。用户最终指定MrRoPE论文使用的3B，先做类似冒烟，再视胜率考虑全量。本轮只执行既定36条/臂；没有启动全量。模型为Qwen/Qwen2.5-3B-Instruct，revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`，实际3,085,938,688参数。权重与metadata已对照缓存的固定官方元数据核对。

[MrRoPE原文Table2(b)与Appendix B](https://arxiv.org/html/2601.22181v1)明确使用Qwen2.5-3B-Instruct、32K→128K、S4、边界23/40。本轮使用同模型与RoPE设置，但仅六个官方RULER任务的固定子集，不能直接与论文完整13任务的53.2对照判胜负。

## 配对结果

| 长度上限 | MrPro | BM | BM−MrPro | 逐题胜/负/平 |
| --- | ---: | ---: | ---: | --- |
| 32K | 87.22% | 91.67% | +4.44 pp | 2/0/10 |
| 128K | 78.12% | 70.83% | -7.29 pp | 4/4/16 |

总计6胜4负26平，但长端4胜4负的分值变化不等，均分反而下降；不能以总胜场替代主终点。

| 128K任务 | MrPro | BM |
| --- | ---: | ---: |
| niah_single_2 | 100.00% | 100.00% |
| niah_multikey_2 | 75.00% | 50.00% |
| niah_multiquery | 93.75% | 100.00% |
| vt | 75.00% | 75.00% |
| fwe | 75.00% | 75.00% |
| qa_1 | 50.00% | 25.00% |

## 原始答案复核

- **密集多键真实误绑定：** 问`bizarre-inhabitant`，其真实值3954314；MrPro正确。BM输出9289114，该数确实属于另一条`bizarre-slime`记录，不是标点或格式评分造成的假负例。
- **追踪既有收益也有损失：** 两条分别20%→80%、80%→100%，另一条100%→20%，最终任务均分均为75%。退化条BM只输出起始变量ZOFPD，6个生成token即EOS，缺失其他4个变量，不是触及30-token上限才截断。
- **聚合相互抵消：** 一条多命中一个高频词，另一条少命中一个，任务均分均为75%。
- **QA答非所问：** 问P与co-NP通常被认为有什么关系，MrPro回答not equal；BM输出P⊆NP⊆PP⊆PSPACE，未回答所问关系。该下降不是只差标点或别名。
- **多查询有一条真实集合召回增益：** MrPro的一项数字为10034850，BM改为gold 9034850，官方集合召回从3/4到4/4。此评分不额外证明所有标签-数值配对语义正确。

## 输入、实现和成本

seed20260913；QA官方pre_samples256；每任务32K两条、128K四条，共36条，官方逐任务回答预算，greedy、repetition_penalty1、默认合法EOS、BF16、Flash SDPA、FP32频率。两臂完整使用相同prompt IDs、模型及解码，只有中段频率改变；gain均为1+.1ln4。实际输入29,740–131,039 tokens，总计3,501,674 input tokens/臂。

32K对应模型原生窗口长度，但两臂均以静态S4部署；不能把MrPro32K分数称Native原表参照。两个长度的QA可能复用问题，因此不视为双倍独立样本。每任务长端仅4题，结果用于本轮是否进入全量的判断，不作总体显著性声明。

两臂完整生成分别879.75/874.82秒，监督阶段1766.55秒（约29.44分钟），含加载；无OOM、无长度截断或backend回退。32项相关CPU测试通过，实际GPU参数量与准备的权重头一致。

1.5B最初准备并启动，随后按用户“直接使用官方3B”要求STOP中止：MrPro仅19/36条，没有完整对照，不能用其部分结果判断方法或容量。其原始输出留存，不恢复该队列。

## 复现与下一步边界

完整资产：本地`results/bm_transfer_20260908/`，远端`/root/autodl-tmp/bm_transfer_20260908/`。`code_executed/`是实际执行源码快照；准备manifest记录shard哈希、tokenizer、完整频率、解码、token IDs及顺序。最新源码不能代替旧manifest对应快照。

[结果JSON](ROPE_BM_TRANSFER_RESULT_20260908.json)由完成回执和原始JSONL重新计算，核对文件哈希、行顺序、prompt身份和官方评分。复算：

```bash
python -m scripts.analysis.summarize_bm_transfer --root results/bm_transfer_20260908 --out docs/research/ROPE_BM_TRANSFER_RESULT_20260908.json
```

本轮的“类似冒烟、再考虑全量”已完成。没有达到长端晋级条件，因此当前不扩展完整13任务。OLMo的局部正结果继续保留；3B结果说明其增益没有按同一闭式直接迁移。对于OLMo32K失败是否由容量造成，本次同时改变模型家族、原生窗口及相对扩展倍数，未隔离容量因素。
