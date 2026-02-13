# Cross-Model WikiText 结果摘要（R6000）

时间戳：2026-02-13_193301  
脚本：`scripts/run_cross_model_wikitext_eval.py`  
协议：WikiText-103 validation, random_start, lengths=[2048,16384], seeds=[42,123,777], windows_per_seed=10

## 主表

| Model | PPL@2K (mean±std) | PPL@16K (mean±std) | Collapse (16K/2K) |
|---|---:|---:|---:|
| llama_geo_10k | 549.853 ± 38.306 | 12111.057 ± 741.127 | 22.026 |
| llama_sigmoid_best_t100k | 11.673 ± 1.060 | 12.572 ± 0.285 | 1.077 |
| qwen_orig_theta | 8.463 ± 0.095 | 6.976 ± 0.153 | 0.824 |
| qwen_geo_100k | 8.581 ± 0.093 | 7.156 ± 0.153 | 0.834 |

## 快速结论

1. LLaMA 上 `geo_10k` 在 16K 出现灾难性退化；`sigmoid_best_t100k` 显著稳定。
2. Qwen 原始 theta 在该协议下本身稳定，且优于 Qwen 人为降 theta=100k。
3. 该结果强烈支持“模型结构 + theta + 频谱形状共同决定稳定边界”的机制叙事。

## 文件

- 原始结果：`results.json`
- 运行日志：`run.log`

