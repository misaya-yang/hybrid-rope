# h800_3h_poly_followup

poly 频谱补充实验（A800，训练+稳健性复评）。

## 协议

- 数据与训练设置：与 `h800_3h_followup` 保持一致
- 候选：`poly_th100k_p3.9_omf0.3`, `poly_th500k_p3.9_omf0.3`
- 评测长度：`2048, 4096, 8192, 12288, 14336, 16384`
- 切片：`sequential + random_start`
- random_start seeds：`42, 123, 777`

## 关键结果（random_start @16K）

1. `poly_th500k_p3.9_omf0.3`: `31.231`
2. `poly_th100k_p3.9_omf0.3`: `35.146`

## 文件

- `results.json`: 全量结果
- `summary.md`: 摘要
- `run.log`: 总耗时
- `variants/*/result.json`: 单候选结果

## 说明

- 本目录不包含 checkpoint 权重文件。
