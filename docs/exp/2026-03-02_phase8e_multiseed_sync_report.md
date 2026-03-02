# Phase8E/多 Seed 实验回传报告（2026-03-02，最终版）

## 1) 已回传内容

本次从远端 `root@connect.bjb2.seetacloud.com:24215` 回传到本地目录：

- `artifacts/_server_sync/2026-03-02_phase8e_multiseed/raw/evq_phase8/results_phase8.json`
- `artifacts/_server_sync/2026-03-02_phase8e_multiseed/raw/evq_phase8/phase8e_log.txt`
- `artifacts/_server_sync/2026-03-02_phase8e_multiseed/raw/evq_phase8/phase8f_log.txt`
- `artifacts/_server_sync/2026-03-02_phase8e_multiseed/raw/evq_phase8/multi_seed/results_8f.json`
- `artifacts/_server_sync/2026-03-02_phase8e_multiseed/raw/evq_phase8/multi_seed/**/result.json`（12 个）

说明：

- 只同步了关键结果与日志，未拉取大文件 `model.pt`。
- 远端快照时间：`2026-03-02 10:39:59`（服务器时间，UTC+8）。

## 2) 多 Seed 完成度（8F）

目标组合：`4 seeds x 3 methods = 12` 项  
已完成：`12/12`（100%）

- `seed42`: `geo_4k` / `evq1.0_4k` / `hybrid1.0_4k` 完成
- `seed137`: `geo_4k` / `evq1.0_4k` / `hybrid1.0_4k` 完成
- `seed256`: `geo_4k` / `evq1.0_4k` / `hybrid1.0_4k` 完成
- `seed314`: `geo_4k` / `evq1.0_4k` / `hybrid1.0_4k` 完成

## 3) 关键指标明细（来自 multi_seed/*/result.json）

| seed | method | retrieval | mean_nll_gap | PPL@4K | PPL@8K | PPL@16K |
|---|---|---:|---:|---:|---:|---:|
| 42 | geo_4k | 0.6900 | 0.1814 | 91.063 | 115.560 | 175.437 |
| 42 | evq1.0_4k | 0.7200 | 0.2521 | 92.792 | 120.296 | 180.081 |
| 42 | hybrid1.0_4k | 0.7050 | 0.2195 | 92.995 | 117.272 | 172.609 |
| 137 | geo_4k | 0.8150 | 0.2884 | 87.893 | 115.132 | 194.515 |
| 137 | evq1.0_4k | 0.7000 | 0.2222 | 89.797 | 122.110 | 182.802 |
| 137 | hybrid1.0_4k | 0.7175 | 0.2326 | 89.334 | 118.030 | 187.300 |
| 256 | geo_4k | 0.7225 | 0.1898 | 86.561 | 107.162 | 162.664 |
| 256 | evq1.0_4k | 0.7150 | 0.1617 | 90.332 | 122.172 | 194.889 |
| 256 | hybrid1.0_4k | 0.7025 | 0.2003 | 87.482 | 113.815 | 177.244 |
| 314 | geo_4k | 0.7125 | 0.2545 | 89.055 | 113.749 | 170.281 |
| 314 | evq1.0_4k | 0.6900 | 0.1859 | 91.307 | 127.815 | 217.660 |
| 314 | hybrid1.0_4k | 0.7125 | 0.2092 | 91.035 | 119.235 | 170.747 |

## 4) 分方法汇总（最终样本）

| method | n | retrieval (mean±std) | mean_nll_gap (mean±std) | PPL@4K (mean±std) | PPL@8K (mean±std) | PPL@16K (mean±std) |
|---|---:|---:|---:|---:|---:|---:|
| geo_4k | 4 | 0.7350±0.0550 | 0.2285±0.0516 | 88.643±1.908 | 112.901±3.903 | 175.724±13.582 |
| evq1.0_4k | 4 | 0.7062±0.0138 | 0.2055±0.0398 | 91.057±1.315 | 123.098±3.263 | 193.858±17.123 |
| hybrid1.0_4k | 4 | 0.7094±0.0069 | 0.2154±0.0139 | 90.212±2.356 | 117.088±2.327 | 176.975±7.406 |

## 5) 8F 统计检验与结论

- 来自：`artifacts/_server_sync/2026-03-02_phase8e_multiseed/raw/evq_phase8/multi_seed/results_8f.json`
- 脚本 verdict：`FAIL: EVQ PK (70.6%) ≤ Geo PK (73.5%)`
- 主要检验结果：
  - `evq1.0_vs_geo_pk`: `p=0.419405`
  - `evq1.0_vs_geo_ppl16k`: `p=0.266402`
  - `hybrid1.0_vs_geo_pk`: `p=0.380923`
  - `hybrid1.0_vs_geo_ppl16k`: `p=0.807920`
  - pooled chi2: `evq p=0.076242 (h=-0.0641)`, `hybrid p=0.114418 (h=-0.0572)`

## 6) 当前运行状态

- `8F` 已结束（`phase8f_log` 末尾为 `DONE!`，`results_8f.json` 时间为 `2026-03-02 10:25:02`）。
- 未发现自动启动的新一轮 `8E`（无相关进程、无新增 `8E` 结果文件）。
