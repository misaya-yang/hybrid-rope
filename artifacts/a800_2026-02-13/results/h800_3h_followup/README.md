# h800_3h_followup

3 小时窗口下的定向机制验证补充实验（训练 + 严格复评）。

## 设置

- 模型族：TinyStories from-scratch（与 `h800_parallel` 相同配置）
- 候选数：6（baseline + 高 theta/sigmoid/hybrid）
- 训练：同一数据、同一顺序、同一 seed（42）
- 评测长度：`2048, 4096, 8192, 12288, 14336, 16384`
- 切片策略：`sequential` + `random_start`
- random_start seeds：`42, 123, 777`

## 关键结果（random_start @16K）

1. `sigmoid_th100k_steep8_mid0.5_omf0.3`: `25.847`
2. `sigmoid_th500k_steep8_mid0.5_omf0.3`: `26.116`
3. `geo_500k`: `27.217`
4. `hybrid_basegeo500k_alpha0.2`: `27.487`
5. `sigmoid_steep8_mid0.5_omf0.3`: `27.870`
6. `geo_10k_baseline`: `76.989`

## 文件

- `results.json`: 全量结构化结果（训练信息 + 各长度各切片统计）
- `summary.md`: 人类可读总结与排行榜
- `run.log`: 总耗时信息
- `variants/*/result.json`: 单候选详细结果

## 说明

- 本目录不包含 checkpoint 权重文件（仅保留结果与日志）。
