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
