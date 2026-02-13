# A800 Artifacts (2026-02-13)

本目录保存 A800 机器上 Llama-3-8B Hybrid-LoRA 相关实验产物（不含权重）。

## 目录

- `A800_LLAMA3_HYBRID_LORA_EVAL_SUMMARY_2026-02-13.md`
  - 给导师汇报的一页总结（训练设置 + PPL 对比 +关键结论）
- `llama3_hybrid_lora/`
  - `summary.json`: 训练摘要（时间、数据源、rope 配置、时长）
  - `run.log`: 微调训练日志
  - `adapter_config.json`: LoRA 结构配置（不含权重）
  - `final_lora_README.md`: PEFT 自动生成说明
- `llama3_hybrid_lora_eval/`
  - `results.json`: 评测结果（base_unfinetuned vs hybrid_lora）
  - `run.log`: 评测日志
- `scripts/`
  - `eval_hybrid_lora_vs_base.py`: 本次对比评测脚本

## 关键信号

- `PPL@16384`
  - `base_unfinetuned`: 190.566
  - `hybrid_lora`: 15.400
- 说明在 16K 长上下文下，Hybrid-LoRA 对未微调基线存在显著优势。

## 说明

- 本目录不包含 `adapter_model.safetensors` 等权重文件。
- `base_yarn_x2` 在当前环境评测失败（`'rope_type'`），详见 `llama3_hybrid_lora_eval/results.json`。
