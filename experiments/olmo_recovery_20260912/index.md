# OLMo：长上下文能力恢复

2026-09-12。5090 本 session 的最高优先级：联合改善长上下文建模、完整生成和原生能力保留。当前仅推进已有 OLMo-2 1.485B；Llama 在 OLMo 稳定后再考虑。

- [运行范围与证据](../../docs/research/next_stage_20260912/5090_recovery_preparation.md)：用户最新优先级、历史依据、数据及模型缺口、无卡准备状态。
- `make_plan.py`：由实际模型配置生成 Native/Cosh/OfficialYaRN 配对合同，绑定数据身份。
- `data_prepare.py` / `data_audit.py`：真实文档、长指令、短能力 replay；原 tokenizer 解码后用目标 tokenizer 重编码；来源分组及答案终止检查。
- `runtime.py` / `train.py`：Q/K/V/O 与 gate/up/down LoRA、分块输出头、分项梯度累积和可恢复训练。
- `probe.py` / `cpu_checks.py`：丢弃式显存资格入口和 CPU 实现检查。
- `readiness.py`：仅 metadata 的轻量清单，适用于 0.5 核无卡实例。
- `evaluate.py`：未适配 0-step 与已适配 checkpoint 的完整生成、Native 保留及分块 NLL。
- `data_score.py`：完整响应、normalized exact、literal exact 与终止分别记账。

默认不执行 GPU。缺少模型权重、实际训练数据或显卡资格时，不将 CPU 计划或小模型测试写成训练就绪/实验结果。具体命令和已验证状态见上面的运行记录。
