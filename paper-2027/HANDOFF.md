# 当前交接

- **目标与授权：** 作者要求从MrPro自主实现零训练改进，运行、及时分析并实施后续方法；不设置例行确认门槛。
- **OLMo轮次已完成：** [完整结果](../docs/research/ROPE_OLMO_BM_RESULT_20260908.md)。7个GPU阶段、780次完整生成；全部正常结束，逐行重算与回执核对完成。
- **保留成果：** BM在OLMo-2-0425-1B-Instruct、静态S4的独立72条六任务复核中，16K为51.32%，同输入MrPro2.78%、MrUni32.12%、官方YaRN6.94%；4K BM81.81%。这是局部能力收益，不是完整RULER、跨模型或SOTA结论。
- **已实现：** [可复用BM函数](../scripts/lib/rope/boundary_matched.py)，与实际FP32实验数组逐位一致；29项相关测试通过。标准S4 BM作为当前已验证配置；新幅度分配两项均未超过它。
- **适用边界：** 静态S8在32K BM6.94%、MrPro0.69%，检索/追踪地板；同32K使用S4 BM为0%，S8频率+S4gain为5.56%，两个恢复方案均不晋级。Native短端也有逐任务取舍。不要重启这些固定负结果，或把短端恢复当32K收益。
- **服务器：** `ssh -p 27741 root@connect.westc.seetacloud.com`，RTX4080SUPER32GB；Python `/root/miniconda3/bin/python`。OLMo作业已完成；Qwen3B冒烟也已完成，无运行中的本轮作业。机器未关机。
- **资产：** 远端根目录`/root/autodl-tmp/olmo_fast_screen_20260908/`；本地`results/olmo_fast_screen_20260908/`保留完整输入、逐行生成、manifest及各阶段源码快照。归档快照与最新源码分开，旧manifest应配其对应快照。
- **后续研究边界：** 可以从已验证S4 BM继续做其他模型/自然任务的匹配检验，或提出能针对16K密集绑定/追踪残余失败的新干预；现有结果不支持继续调失败gain系数、盲增scale或无条件扩大32K矩阵。

- **最新任务已完成：** 作者最终指定MrRoPE论文的Qwen2.5-3B-Instruct，先做冒烟再考虑全量。[3B结果](../docs/research/ROPE_BM_TRANSFER_RESULT_20260908.md)：32K BM91.67%对MrPro87.22%；128K BM70.83%对MrPro78.13%，长端4胜4负16平、均分下降7.29pp，未晋级全量。72次生成全部正常完成，约29.44分钟；1.5B按用户要求在MrPro19/36条时中止，不作方法结论。
- **跨模型资产：** 远端`/root/autodl-tmp/bm_transfer_20260908/`；本地`results/bm_transfer_20260908/`，含原始输入/输出、metadata和执行代码快照。最新运行器支持分片权重与各模型原生短端长度；32项相关测试通过。不要重新启动被用户中止的1.5B或无条件展开3B全量。

- **用户最新要求已完成：** 研究128K错误并尽快收尾、提交推送，回家PC接续。[最小诊断](../docs/research/ROPE_BM_128K_DIAGNOSIS_20260908.md)两例已完成14次生成：C/P均满分；保留完整预填充KV的L多键仍0%、VT仅60%，O与原冒烟逐token一致。说明原长背景形成的状态参与退化，不能仅归因于128K坐标或单一竞争key；层/头/槽机制尚未识别。原始数据远端`bm_transfer_20260908/diagnosis_run_02`，紧凑证据随Git保存。全部GPU任务已结束，无自动队列。
