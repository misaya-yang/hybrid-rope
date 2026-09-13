# OLMo：长上下文能力恢复

2026-09-12。5090 本 session 的最高优先级：联合改善长上下文建模、完整生成和原生能力保留。当前仅推进已有 OLMo-2 1.485B；Llama 在 OLMo 稳定后再考虑。

- [运行范围与证据](../../docs/research/next_stage_20260912/5090_recovery_preparation.md)：用户最新优先级、历史依据、数据及模型缺口、无卡准备状态。
- `make_plan.py`：由实际模型配置生成 Native/Cosh 配对合同，绑定数据身份；YaRN训练对照已取消。
- `download_sources.py`：服务器原始数据下载；PG19/LongAlign/Dolly/QASPER，复用RULER/NIAH，不分词或加载模型。
- `data_prepare.py` / `data_multiscale.py` / `data_audit.py`：真实8K/16K连续文本与完整长指令、独立短能力replay；来源分组及答案终止检查。
- `runtime.py` / `train.py` / `length_schedule.py`：真实8K/8K/16K轮转及token计数，Q/K/V/O 与 gate/up/down LoRA、分块输出头、分项梯度累积和可恢复训练。
- `probe.py` / `cpu_checks.py`：丢弃式显存资格入口和 CPU 实现检查。
- `readiness.py`：仅 metadata 的轻量清单，适用于 0.5 核无卡实例。
- `evaluate.py`：未适配 0-step 与已适配 checkpoint 的完整生成、Native 保留及分块 NLL。
- `data_score.py`：完整响应、normalized exact、literal exact 与终止分别记账。

默认不执行 GPU。缺少模型权重、实际训练数据或显卡资格时，不将 CPU 计划或小模型测试写成训练就绪/实验结果。具体命令和已验证状态见上面的运行记录。

- [LoRA v2设计与公开数据扩容](../../docs/research/next_stage_20260912/OLMO_LORA_V2_20260912.md)。
- `download_public_sft.py`：UltraChat train_sft、LongAlpaca、LongCite原始数据下载，支持并发续传，不做SHA扫描。
- `expand_public_sft.py`：取消小样本上限，按来源和真实长度构建新SFT池。
- `expand_pg19_sources.py`：扩PG19至512本train，复用原文档与heldout。

- `expand_longcite.py`：完整来源引用长QA的流式筛选，可用4个CPU worker。
- `recovery_v2_data.py`：五家族/八长度池组装，复用已完成数据阶段。
- `recovery_v2_runtime.py` / `recovery_v2_train.py`：全层QKVO64+FFN16、分组LR、短KL与固定schedule续训。
- `recovery_v2_probe.py` / `recovery_v2_eval.py`：必要工程检查与完整任务/NLL评价。
- `gradual_table_train.py`：8K 300步单变量安装路径对照；20步Native、180步log-frequency过渡、100步固定Cosh，最终表与骤换Cosh相同。
- `prepare_fixed_table_interval.py` / `score_fixed_table_interval.py`：Llama Native 8K
  到64K的单固定g8表面板及log-length AUC、内部最弱点、Native/endpoint regret汇总；
  48K作为显式内部压力点，不动态换表。
- `test_fixed_table_interval.py`：gamma3、BM/MrPro等距表与BM range-mean gain的
  固定表身份和AUC算术检查。
- `prepare_range_solver_data.py` / `prepare_range_source_cf.py`：冻结OLMo的
  4/6/8/10/12/14/16K任务与4K/16K来源反事实8/4/4拆分；目标含完整答案和EOS。
- `native_relative_allocation.py` / `solve_range_table.py`：固定S=4、band[14,32]
  的17个有效increment自由度与一个gain自由度；冻结模型权重，以全fit真实值和
  轮转小批梯度做字典序顺序线性trust-region求解。
- `measure_range_baselines.py`：Native、BM、MrPro、C42V24在同一fit任务和
  source-counterfactual合同上的一次性teacher-forced基线。
- `recovery_v2_eval.py` / `summarize_range_generation.py`：加载冻结求解器tensor，
  在独立select/confirm上分别报告RULER official contains、完整答案+EOS与来源
  pair-follow，不把三种口径混成总分。
- `compare_broad_fixed_candidate.py`：将一个冻结候选与Git归档、row-matched的
  350行七任务16K C42/C42V24输出比较，只生成候选，不重跑历史基线。
- `compare_natural_fixed_candidate.py`：将同一冻结候选与391行自然QA历史原始行
  做row/prompt身份对齐后比较；只生成候选，BM/MrPro基线永久复用。
- `build_range_factorial_tables.py`：构造C42V24与SolverC42之间缺失的两个
  shape×gain单变量交叉格，复用两个已测端点格以分离allocation、gain及其交互。
- `bootstrap_task_equal_contrast.py`：对row-matched候选/基线做任务内配对重采样后
  再任务等权，报告敏感性区间；它不是总体置信保证，也不作为候选淘汰门禁。
- `transfer_range_profile.py`：按目标Native表与倍率重建频率；跨Native长度时将
  求解器形状从source transition band归一重映射到target band。literal-slot模式
  只作为显式机制消融，gain策略另行声明；支持Solver与精确C42V24来源以补齐归因
  2×2。
- `compare_llama_fixed_candidate.py`：将一个迁移候选与已归档Llama g8 BM/MrPro
  8/16/32/48/64K原始行对齐，汇总固定表AUC、最弱点和regret，不重跑基线。
- `compare_llama_runner_parity.py`：在8个相同64K任务行上比较新旧runner的原始
  token、文本与统一重算official，决定旧32行BM/MrPro能否永久复用。
- `compare_llama_64k8task.py`：用同一scorer从原始文本重算候选和runner-matched
  BM/MrPro的64K八任务official，报告任务向量与配对胜负。
- `winding_matched_table.py`：实现作者提出的逐槽最大合法整数绕圈闭式表，输出
  FP64/模拟FP32端点残差、频率顺序与m单调违例；不把构造断言当任务结果。
- `prepare_llama_minimal_band_screen.py`：一次准备Llama S=2的BM/MrPro与4个平移
  C42 band、两篇固定PG19文本；复用8/16K passkey/NIAH面板做分钟级最小漏斗。
- `summarize_llama_minimal_band_screen.py`：分列8/16K task-equal official与两篇
  PG19 PPL，四指标支配关系只作描述，不作为删除候选的门禁。
- `evaluate_static_tail_nll.py`：对任意冻结64槽表运行配对自然文本tail-NLL，保存
  逐文档token损失、表身份和分长度PPL；用于复用历史BM/MrPro而只测新band。
- `summarize_qwen_minimal_band_screen.py`：汇总Qwen1.5B S=2的Native/BM/MrPro与
  四种跨模型band，分列32/64K NIAH official、完整答案+EOS和两篇tail-512 PPL。

判定采用分层证据，不把开发集阈值当作淘汰器：实现或协议错误可以否决一次运行；
任务层明确受支配才淘汰候选；小样本未过线、代理冲突或单任务退化只记为Pareto/
未决，并保留静态表供后续宽面板、自然任务和独立seed复核。
