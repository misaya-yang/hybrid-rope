# 当前交接

- **目标与授权：** 作者要求从MrPro自主实现零训练改进，运行、及时分析并实施后续方法；不设置例行确认门槛。
- **本轮已完成：** [完整结果](../docs/research/ROPE_OLMO_BM_RESULT_20260908.md)。7个GPU阶段、780次完整生成；全部正常结束，逐行重算与回执核对完成。
- **保留成果：** BM在OLMo-2-0425-1B-Instruct、静态S4的独立72条六任务复核中，16K为51.32%，同输入MrPro2.78%、MrUni32.12%、官方YaRN6.94%；4K BM81.81%。这是局部能力收益，不是完整RULER、跨模型或SOTA结论。
- **已实现：** [可复用BM函数](../scripts/lib/rope/boundary_matched.py)，与实际FP32实验数组逐位一致；29项相关测试通过。标准S4 BM作为当前已验证配置；新幅度分配两项均未超过它。
- **适用边界：** 静态S8在32K BM6.94%、MrPro0.69%，检索/追踪地板；同32K使用S4 BM为0%，S8频率+S4gain为5.56%，两个恢复方案均不晋级。Native短端也有逐任务取舍。不要重启这些固定负结果，或把短端恢复当32K收益。
- **服务器：** `ssh -p 27741 root@connect.westc.seetacloud.com`，RTX4080SUPER32GB；Python `/root/miniconda3/bin/python`。本轮GPU作业已完成，没有未完成训练或候选队列。机器未关机。
- **资产：** 远端根目录`/root/autodl-tmp/olmo_fast_screen_20260908/`；本地`results/olmo_fast_screen_20260908/`保留完整输入、逐行生成、manifest及各阶段源码快照。归档快照与最新源码分开，旧manifest应配其对应快照。
- **后续研究边界：** 可以从已验证S4 BM继续做其他模型/自然任务的匹配检验，或提出能针对16K密集绑定/追踪残余失败的新干预；现有结果不支持继续调失败gain系数、盲增scale或无条件扩大32K矩阵。
