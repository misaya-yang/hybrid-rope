# RoPE 频率分配：已有进展与 30 代理材料归档

日期：2026-09-10。状态：文档／代码归档；不启动新研究、推导、训练或测评。

**当前问题以 [理论核心](ROPE_ALLOCATION_THEORY_CORE_20260910.md) 为准：研究有限 RoPE 非均匀傅里叶频率分配的最优条件，兼顾 native／近距损失和远距能力，覆盖 EVQ、MrRoPE、YaRN 等规则。搬运是可用理论之一，不是预设定义。**

## 1. 归档完整性

- 当轮共派出20个 Sol、10个 Astra；并发限制下分批执行。
- 保留 **28份原报告、27份原阅读回执、30份历史分工**；原文逐字复制，SHA256 见 [归档清单](rope_allocation_20260910/archive_manifest.json)。
- Astra10、Sol20 原报告没有完成落盘；两位的已回传贡献单独整理保存，明确标记，未伪造原稿。Sol18 有报告但无独立阅读回执。
- 保存5个 Python 文件、29个 CPU 生成的频率表、4条32K完整模型响应记录、输入清单和5份外部／Pro问题附件。
- 阅读回执是当时各代理的记录，不代表主任务重新验证了所有原始文件。30位代理共享初始分工中的若干假设；相同建议的出现次数不是独立实证支持。
- 原分工／报告中的运行计划、强判断和“必须使用某工具”等文字均为历史材料。它们不自动约束当前目标。

## 2. 全部代理材料

报告、回执、历史分工分别位于 agents/、coverage/、assignments/。缺项显式列出。

| 代理 | 保留的主要贡献 | 报告或回传整理 | 阅读回执 |
| --- | --- | --- | --- |
| sol01 | MrRoPE 累计 radix／离散索引；最优性与设计假设分开 | [原报告](rope_allocation_20260910/agents/sol01.md) | [回执](rope_allocation_20260910/coverage/sol01_coverage.json) |
| sol02 | EVQ Cosh、分位数、有限窗与 gap 梯度 | [原报告](rope_allocation_20260910/agents/sol02.md) | [回执](rope_allocation_20260910/coverage/sol02_coverage.json) |
| sol03 | Pro 第一性原理材料；跨度分配及生命周期 | [原报告](rope_allocation_20260910/agents/sol03.md) | [回执](rope_allocation_20260910/coverage/sol03_coverage.json) |
| sol04 | Pro 分配／兼容性框架与水位式候选 | [原报告](rope_allocation_20260910/agents/sol04.md) | [回执](rope_allocation_20260910/coverage/sol04_coverage.json) |
| sol05 | 6Pro 全行 KL、局部占优与块边界问题 | [原报告](rope_allocation_20260910/agents/sol05.md) | [回执](rope_allocation_20260910/coverage/sol05_coverage.json) |
| sol06 | 联合关系、Smooth 反例、受限二次候选 | [原报告](rope_allocation_20260910/agents/sol06.md) | [回执](rope_allocation_20260910/coverage/sol06_coverage.json) |
| sol07 | 实际代码／E7 精度与 prefix-read margin 分解 | [原报告](rope_allocation_20260910/agents/sol07.md) | [回执](rope_allocation_20260910/coverage/sol07_coverage.json) |
| sol08 | scratch／冻结失败审计及适配差异 | [原报告](rope_allocation_20260910/agents/sol08.md) | [回执](rope_allocation_20260910/coverage/sol08_coverage.json) |
| sol09 | 会话与失败分片1；有限有符号边际作用 | [原报告](rope_allocation_20260910/agents/sol09.md) | [回执](rope_allocation_20260910/coverage/sol09_coverage.json) |
| sol10 | 会话与失败分片2；间隔的累计影响 | [原报告](rope_allocation_20260910/agents/sol10.md) | [回执](rope_allocation_20260910/coverage/sol10_coverage.json) |
| sol11 | 会话与失败分片3；真实竞争与完整生成 | [原报告](rope_allocation_20260910/agents/sol11.md) | [回执](rope_allocation_20260910/coverage/sol11_coverage.json) |
| sol12 | 会话与失败分片4；指标、标签与原结果范围 | [原报告](rope_allocation_20260910/agents/sol12.md) | [回执](rope_allocation_20260910/coverage/sol12_coverage.json) |
| sol13 | 会话与失败分片5；案例账本与条件保证 | [原报告](rope_allocation_20260910/agents/sol13.md) | [回执](rope_allocation_20260910/coverage/sol13_coverage.json) |
| sol14 | 会话与失败分片6；未测统计量与结论边界 | [原报告](rope_allocation_20260910/agents/sol14.md) | [回执](rope_allocation_20260910/coverage/sol14_coverage.json) |
| sol15 | 论文／审稿材料；Cosh 现有论断范围 | [原报告](rope_allocation_20260910/agents/sol15.md) | [回执](rope_allocation_20260910/coverage/sol15_coverage.json) |
| sol16 | 可微 RoPE、16自由度、checkpointing 参考 | [原报告](rope_allocation_20260910/agents/sol16.md) | [回执](rope_allocation_20260910/coverage/sol16_coverage.json) |
| sol17 | 旧优化器、LeRoPE 类校准与新理论的区别 | [原报告](rope_allocation_20260910/agents/sol17.md) | [回执](rope_allocation_20260910/coverage/sol17_coverage.json) |
| sol18 | 记录定位、全模型目标、gain／配对／位置实现 | [原报告](rope_allocation_20260910/agents/sol18.md) | 未落盘 |
| sol19 | 历史全族关闭／Hessian／因果叙述的核对 | [原报告](rope_allocation_20260910/agents/sol19.md) | [回执](rope_allocation_20260910/coverage/sol19_coverage.json) |
| sol20 | OLMo 350条配对正收益及复杂绑定未解决 | [回传整理，非原稿](rope_allocation_20260910/recovered/sol20.md) | 未落盘 |
| astra01 | 信号／干扰模型、非负最优条件、有限 K 实现 | [原报告](rope_allocation_20260910/agents/astra01.md) | [回执](rope_allocation_20260910/coverage/astra01_coverage.json) |
| astra02 | 连续精确核原子最优、证书与 Galerkin 归一化 | [原报告](rope_allocation_20260910/agents/astra02.md) | [回执](rope_allocation_20260910/coverage/astra02_coverage.json) |
| astra03 | 有限相位变化界、条件 conic 求解、KL 投影 | [原报告](rope_allocation_20260910/agents/astra03.md) | [回执](rope_allocation_20260910/coverage/astra03_coverage.json) |
| astra04 | 完整归一化、附加干扰 KL 分解及块映射 | [原报告](rope_allocation_20260910/agents/astra04.md) | [回执](rope_allocation_20260910/coverage/astra04_coverage.json) |
| astra05 | 联合整数模式精确投影；29个 CPU 表 | [原报告](rope_allocation_20260910/agents/astra05.md) | [回执](rope_allocation_20260910/coverage/astra05_coverage.json) |
| astra06 | 特定噪声模型下的整数通道／带标签 DP | [原报告](rope_allocation_20260910/agents/astra06.md) | [回执](rope_allocation_20260910/coverage/astra06_coverage.json) |
| astra07 | 系数适配算子、密度／加载区别与条件投影 | [原报告](rope_allocation_20260910/agents/astra07.md) | [回执](rope_allocation_20260910/coverage/astra07_coverage.json) |
| astra08 | 记录角色、输出敏感度与有限 replay | [原报告](rope_allocation_20260910/agents/astra08.md) | [回执](rope_allocation_20260910/coverage/astra08_coverage.json) |
| astra09 | 精确有限 softmax、罕见 key 反例、参考代码 | [原报告](rope_allocation_20260910/agents/astra09.md) | [回执](rope_allocation_20260910/coverage/astra09_coverage.json) |
| astra10 | 交叉综合及新 YaRN／Mr 四格分析核对 | [回传整理，非原稿](rope_allocation_20260910/recovered/astra10.md) | 未落盘 |

## 3. 已有结果：保留正收益，也保留条件

下表复用已有 owner／结果，不是本次新增实验。不同模型、长度、gain、任务面板不能合并成一个排名。

| 证据 | 已有进展 | 使用范围 |
| --- | --- | --- |
| EVQ 固定 support、匹配训练 | 分配形状确实能够影响训练后表现；相关 OOD 对比有正收益 | 见论文 identification 记录；不自动成为冻结换表结论 |
| 权重×频率表交叉 | 已学权重与频率配置存在耦合 | 说明使用条件重要，不否定一般频谱设计 |
| OLMo BM 七任务扩展面板 | 16K，350条/臂：BM41.67%、MrPro7.09%，+34.59pp；156胜9负 | 固定 S4；不是统一全量官方 RULER；multikey_3双方仍0 |
| Qwen E1_s28_less | 128K开发面板约+5.21pp，两个改善样本 | 小样本开发信号；不声称独立泛化已确认 |
| Qwen E1_s29_more | 32K开发面板约+8.33pp；128K约−0.21pp | 长度／任务效应不同，不能推断组合一定更好 |
| Qwen FullLagP2 | 128K开发81.67%，MrPro78.13%；同gain MrPro75.35% | P2短端代价明显；默认gain与同gain对照分开 |
| P2 真实128K长文预检 | 四篇尾512-token PPL：P2 5.35755、同gain MrPro5.37052、默认MrPro5.50353 | 是完整128K前缀后的尾部PPL，不是全文PPL；对同gain两胜两负 |
| Smooth_MrBudget | 多种几何／QK加权指标更好，但128K开发得分更差 | 约束这些具体代理指标，不构成所有几何理论的否定 |
| E7 | 理想有限改动与一阶预测约7.94e−8；BF16实现约1.28e−5 | 精度差异和下游失败分开；不能说一阶非线性被低估160倍 |
| E1 的 prefix/read 四格 | MK正确答案margin −2.125、−1.125、−1、+0.25 | 两路近加性改善一起跨过阈值；二元成败不证明强非加性 |

主要入口：[近期完整 owner](NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md)、[OLMo七任务结果](ROPE_OLMO_BM_EXTRA_RULER_RESULT_20260908.md)、[逐行 JSON](ROPE_OLMO_BM_EXTRA_RULER_RESULT_20260908.json)、[历史失败综合](ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md)、[分配识别附录](../../paper-2027/appendix/a5_identification.tex)。

## 4. 本轮新产生的东西实际到了哪一步

| 项目 | 状态 | 不能据此声称 |
| --- | --- | --- |
| 连续核／整数 DP／投影／softmax 界 | 报告中完成推导与部分 CPU 例子 | 已推出实际模型的统一最优解 |
| 联合模式29表 | CPU生成，正值／排序／端点／gain及投影检查完成 | 29个模型实验已跑、已选出有效表 |
| 4条32K整网频率响应 | 已完成并有本地原始 JSONL | 生成了新表、模型权重训练、证明长程收益 |
| 2条真实128K响应测量 | 曾启动；本地未取得完成结果回执 | 完成／失败／当前仍在运行中的任何一种确定状态 |
| 新表128K PPL／passkey | 没有本轮完成回执 | 已验证新分配优于MrRoPE |
| 6Pro完整自然Q/K捕获 | 前期六篇32K原始全K/V已生成，源表为Native频率＋MrPro共同gain | 原生gain=1基线；带正确记录标签的数据；真实128K轨迹 |

128K响应的最后一次成功观察是进程运行中。之后状态查询遇到连接失败，没有拿到终态。本次归档未再次连接服务器；若后续需要使用这项结果，先收取原始回执，不能从进程启动或当前脚本推断完成。

32K记录使用的是 **MrPro频率表、原生窗口长度**，不是原始Native频率表。目标为 teacher-forced 正确答案 CE，包含完整前缀反传；只测梯度，没有 optimizer.step，也没有改变频率表或模型权重。

| 记录 | CE | 耗时（秒） |
| --- | ---: | ---: |
| niah_multikey_2_32768_0 | 0.013202 | 13.95 |
| niah_multikey_2_32768_1 | 0.110649 | 13.48 |
| niah_multiquery_32768_0 | 0.308742 | 13.44 |
| niah_multiquery_32768_1 | 0.474598 | 13.48 |

## 5. 代码与可复用附件

| 文件 | 内容／依赖 | 已有验证状态 |
| --- | --- | --- |
| [joint_mode_candidates.py](rope_allocation_20260910/code/joint_mode_candidates.py) | 从既有Native/Mr实际表生成联合模式投影；原输入路径见脚本 | 当轮CPU数学／数组核对；无模型成绩 |
| [joint_mode_candidates.json](rope_allocation_20260910/evidence/joint_mode_candidates.json) | 29表、gain、delta_log_period、累计压缩变化、相位变化 | 已生成；未选中、未部署、未测 |
| [astra07_lifecycle.py](rope_allocation_20260910/code/astra07_lifecycle.py) | NumPy有限模型的投影／适配演示 | 当轮CPU通过；非Transformer效果 |
| [astra09_rule.py](rope_allocation_20260910/code/astra09_rule.py) | 固定状态有限softmax参考与下降界 | 当轮CPU通过；未知全模型迁移 |
| [sol16_frequency_calibration_reference.py](rope_allocation_20260910/code/sol16_frequency_calibration_reference.py) | 17增量／16自由度可微参数化 | 当轮CPU梯度与边界检查通过 |
| [full_model_response.py](rope_allocation_20260910/code/full_model_response.py) | 冻结Qwen参数，替换no_grad rotary路径后测量CE频率响应；依赖既有Worker和服务器输入 | 32K四条完成；128K终态未取得 |
| [4条整网响应](rope_allocation_20260910/evidence/full_model_response_native.jsonl) | 正确答案token损失、64维导数、时间、显存 | 本地完成记录 |

参考代码保持原样；部分默认路径依赖原研究环境。归档不意味着所有脚本无需原输入即可独立运行。没有在本次整理中重跑CPU例子或模型实验。

## 6. 新 YaRN／MrRoPE 附件的接入

[全文](rope_allocation_20260910/source_inputs/mrrope_yarn_user_analysis.md)已复制。原附件的 sandbox 脚本／JSON链接是其作者会话内路径，实际文件没有随附件提供；本仓库没有假称收到它们。核心公式和给定数值已在上一轮回答中独立复算，公开代码已核对。

值得使用的内容：标准YaRN索引ramp与论文圈数表达的差别；Mr中前段少压缩／中后段多压缩；固定边界下的尺度响应；无额外曲线参数的max/min四格。

四格含义记为 F00=Y、F10=max、F01=min、F11=M。F01>F11只证明A侧替换在该B背景下有利；F11优于F10、F01不自动证明非加性协同。交互为 F11−F10−F01+F00，且依赖所用指标尺度。四格当前仍是建议，没有新增运行结果。

## 7. 下一次推导所需的最小材料

先读 [理论核心](ROPE_ALLOCATION_THEORY_CORE_20260910.md)，再按需要使用本页的数学工具、正负结果和原报告。优先把近距损失、远距收益、有限频率资源及其最优条件定义清楚；不因30位代理提出了许多统计／校准方案，就把那些方案变成必须完成的前置工作。

保留问题：高频冗余如何定义；中段是否从最优条件自然出现；低频资源如何同时服务范围与分辨；水床约束在什么意义下成立；不同频率规则是否是共同问题的特殊情形或近似。尚未证明某个具体新分配同时解决这些问题。

## 8. 输入覆盖与来源范围

原分片保存了38个会话快照，其中30个为此前项目会话、8个是当轮首批代理启动后的快照，并非38份独立历史实验。可见正文去重为1530条记录；独立工具输出总计约1.15亿字符。工具输出已归集不等于全部被模型阅读。

代理回执记录各自分配到的全文与补充材料；本轮归档复制回执及输入清单，没有声称所有原始结果二进制、每条工具输出或所有38GB结果都进入过上下文。大型原始输入仍在本地 .agents/rope_unification_20260910/corpus/，本次不把重复工具输出作为研究成果提交。

[归档清单与SHA256](rope_allocation_20260910/archive_manifest.json)是本次复制完整性的记录；其中历史team_plan另标为旧快照，当前归档状态以本页为准。
