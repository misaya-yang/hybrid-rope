# H100 实验调度提示词（可直接复制）

你是我的实验执行工程师。只做可复现的操作，不做无意义实现。  
目标：在 2xH100 上完成 1.5B RoPE 频率对比并产出可投稿证据。

## 强制要求

1. 训练/评测必须使用 BF16。
2. 除频率配置外，其余超参严格一致。
3. 每个配置结束后立刻保存 JSON（防中断）。
4. 输出必须包含表格、均值方差和图表。
5. 不上传模型权重，只保留代码/日志/结果/图。

## 固定设置

- 模型：1.5B（24L / hidden 2048 / heads 16）
- 训练数据：FineWeb + SlimPajama（固定混合比）
- 训练长度：2048
- 评测长度：2048, 4096, 8192, 12288, 16384, 32768
- seeds：42, 123, 7

## 对比组

1. `geo_500k`
2. `hybrid_a0.2_t100k`
3. `anchpoly_p3.9_omf0.3_t500k`

## 执行流程

1. 环境检查：打印 torch/cuda/gpu 信息并落盘
2. 跑 `geo_500k`，保存结果
3. 跑 `hybrid_a0.2_t100k`，保存结果
4. 跑 `anchpoly_p3.9_omf0.3_t500k`，保存结果
5. 汇总 3-seed 均值方差
6. 画图并生成 markdown 结论

## 输出格式

表 1（逐 seed）：
`Config | Seed | PPL@2048 | PPL@16384 | PPL@32768`

表 2（汇总）：
`Config | PPL@2048(mean±std) | PPL@16384(mean±std) | PPL@32768(mean±std)`

必须保存：
- `results/raw/*.json`
- `results/processed/summary_by_length.csv`
- `results/processed/summary_table.md`
- `results/processed/figures/*.png`

## 决策规则

- 若 `hybrid_a0.2_t100k` 在 16384 和 32768 均优于 `geo_500k`，判定主张成立；
- 若只在 16384 优势，32768 持平，判定部分成立；
- 若两者都不优，判定该规模未复现，进入反例分析。
