# 50M：Hybrid vs Geo vs YaRN（修正版评测）

实验时间：2026-02-13  
结果文件：`results/50m_yarn_compare_v2/results.json`

## 评测口径（修正后）

- 训练长度 `2048` 内不做缩放（`scale=1`）。
- 仅在 `L>2048` 时做渐进缩放：`scale=min(8, L/2048)`。
- 目的是避免把训练域也压缩，导致不公平退化。

## 结果表（PPL，越低越好）

| Length | Hybrid (native) | Geo (native) | Geo + YaRN (progressive) |
|---|---:|---:|---:|
| 2048 | **6.672** | 6.839 | 6.839 |
| 4096 | **6.748** | 7.045 | 8.640 |
| 8192 | **8.688** | 8.833 | 16.899 |
| 12288 | **13.333** | 13.588 | 29.352 |
| 16384 | **16.861** | 17.966 | 39.479 |

## 图表

![50M YaRN Comparison](../results/figures/50m_yarn_compare_v2.svg)

## 结论（当前设置下）

- Hybrid native 在所有评测长度上都优于 Geo native。
- 在该 50M 配置下，Geo + YaRN progressive 明显劣于 native 基线。
- 因此当前可稳定报告的对比是：`Hybrid > Geo > Geo+YaRN(progressive)`（同一数据、同一模型尺寸、同一评测协议）。
