# mechanism_p2_framework

机制分析第二阶段的统一框架输出（聚合 + 可视化）。

## 聚合来源

- `results/mechanism_p1/`：LLaMA 7B 机制指标（2x2、token-wise loss、attention、phase、LoRA）
- `results/h800_3h_followup/`：频谱主对照（sigmoid/geometric/hybrid）
- `results/h800_3h_poly_followup/`：poly 补充对照

## 输出

- `results.json`：统一结构化结果
- `summary.md`：结论摘要与下一步建议
- `figures/`：
  - `frequency_shape_16k_bar.png`
  - `frequency_shape_multi_length.png`
  - `factor_2x2_16k.png`
  - `tokenwise_loss_base_vs_hybrid.png`
  - `attention_metrics_base_vs_hybrid.png`
  - `phase_collision_vs_length.png`
  - `lora_weight_freq_heatmap.png`

## 说明

- 跨架构部分（Qwen）当前标记为 pending，等待 `eval_suite.json` 同步后可自动并入。
