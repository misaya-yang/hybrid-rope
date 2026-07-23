# Standalone Experiments

`experiments/` 保存不适合放入主 phase chain 的独立模型实验包。

当前只有：

- `lora_evq_v2/`：LLaMA-3-8B LoRA / evaluator / provenance utilities；其中
  `eval_temporal_holdout_matched.py` 和 `eval_temporal_holdout_three_arm.py`
  用于冻结 2026 temporal holdout 上的 matched-prefix 与三臂比较。
- `rebuttal_2026/sft_distillation/`：程序持有 oracle、DeepSeek 仅做表面
  naturalization 的短上下文 SFT 数据流水线；Paper-Geo/EVQ 共用同一份
  messages 文件和顺序，pilot 受 100 条人工审计门禁约束。

该包属于 supporting 或 rebuttal-triggered evidence，不承担论文三个 primary anchors。历史 LoRA 与 fresh controls 的语料、runtime 和 evaluator 未形成 strict matched pair 时，不得计算 causal EVQ delta。

主论文实验 runner 仍应放在 `scripts/core_text_phases/`；可复用 RoPE 实现在 `scripts/lib/rope/`。

Temporal holdout 是 rebuttal-triggered evaluation path，不是论文已报告结果。数据由
`scripts/data_prep/prepare_temporal_holdout_2026.py` 生成；提交的仓库只保存生成器、
manifest 契约和测试，不保存下载语料、tokenized packs 或评估输出。
