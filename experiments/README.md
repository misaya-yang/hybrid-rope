# Standalone Experiments

`experiments/` 保存不适合放入主 phase chain 的独立模型实验包。

当前只有：

- `lora_evq_v2/`：LLaMA-3-8B LoRA / evaluator / provenance utilities。

该包属于 supporting 或 rebuttal-triggered evidence，不承担论文三个 primary anchors。历史 LoRA 与 fresh controls 的语料、runtime 和 evaluator 未形成 strict matched pair 时，不得计算 causal EVQ delta。

主论文实验 runner 仍应放在 `scripts/core_text_phases/`；可复用 RoPE 实现在 `scripts/lib/rope/`。
