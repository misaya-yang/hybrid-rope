Mechanism P2 Framework Summary

## 1) 频谱形状与稳定性（A800, TinyStories from-scratch）

| rank | variant | ppl@16k(rand) | collapse_ratio(16k/2k) |
|---:|---|---:|---:|
| 1 | sigmoid_th100k_steep8_mid0.5_omf0.3 | 25.847 | 1.232 |
| 2 | sigmoid_th500k_steep8_mid0.5_omf0.3 | 26.116 | 1.258 |
| 3 | geo_500k | 27.217 | 1.282 |
| 4 | hybrid_basegeo500k_alpha0.2 | 27.487 | 1.332 |
| 5 | sigmoid_steep8_mid0.5_omf0.3 | 27.870 | 1.432 |
| 6 | poly_th500k_p3.9_omf0.3 | 31.231 | 1.524 |
| 7 | poly_th100k_p3.9_omf0.3 | 35.146 | 1.693 |
| 8 | geo_10k_baseline | 76.989 | 3.615 |

结论：`sigmoid_th100k` 最优，`poly` 明显优于 `geo_10k_baseline`，但弱于最优 sigmoid/high-theta 方案。

## 2) 2x2 因子消融（LLaMA-3-8B）

- M00@16K(seq): 190.566
- M11@16K(seq): 15.400
- 提升倍数 (M00/M11): 12.37x
- 结论：频谱+LoRA 耦合带来最大稳定化收益。

## 3) Token-wise Loss / Attention / Collision / LoRA 证据

- attention entropy: base=0.3550, hybrid=0.6440
- sink_mass: base=0.4881, hybrid=0.1714
- phase collision @16K: base=0.223710, hybrid=0.195933
- LoRA Q energy(low/mid/high): 2.9865/4.0625/4.5749
- LoRA K energy(low/mid/high): 2.2641/3.3639/5.6829
- 结论：指标整体一致支持“结构性外推失稳 + 频谱形状可调控稳定边界”。

## 4) 跨架构状态

- qwen_eval_suite_json_found: False
- status: pending_qwen_eval_output_sync
- 说明：当前仓库内仅有 Qwen 运行状态与脚本，缺少最终 eval_suite.json 同步。

## 5) 下一步执行建议

1. 同协议补齐 Qwen eval_suite.json，并复用本框架脚本生成跨架构对比图。
2. 对 top-2 频谱（sigmoid_th100k, sigmoid_th500k）做训练 seed 重训复现（>=3 seeds）。
3. 在 LLaMA 7B 上做 poly/sigmoid 的统一频谱 sweep，闭环验证 tiny->7B 一致性。
