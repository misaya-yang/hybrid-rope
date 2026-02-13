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
- `results/mechanism_p1/`
  - `2x2_factor_results.json`: M00/M10/M01/M11 在 2K~16K + 双 slicing 的因子结果
  - `loss_curve_per_model.json`: 16K token-wise raw/smoothed NLL 曲线
  - `attention_stats.json`: 分层分头 attention 熵/sink/long-range/距离统计
  - `phase_collision_index.json`: base_orig vs hybrid 的相位碰撞指标
  - `lora_weight_diff.json`: LoRA 在 Q/K 上的低中高频能量分布
  - `summary.md`: 机制验证阶段（P1）结论汇总
  - `figures/`: 上述指标对应可视化图（含 caption）
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
