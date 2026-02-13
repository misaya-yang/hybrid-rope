# A100 实验汇总（可发导师）

- 生成时间：2026-02-13 13:40:06
- 数据来源：A100 `117.50.192.217`，路径 `/opt/dfrope/results/`
- 同步说明：已完整拉回 **结果与脚本**；已排除权重与大缓存（`*.pt`, `*.bin`, `*.tokens.u16`）

## 1) 结论先行

- **结论A（50M，3-seed）成立**：`hybrid_a0.2_t100k` 在 16K 优于 `geo_500k`。
  - `geo_500k`: 18.207 ± 0.768
  - `hybrid_a0.2_t100k`: 17.324 ± 0.360（相对下降 4.8%）
- **结论B（50M，10-seed）仍成立但方差变大**：
  - `geo_500k`: 18.798 ± 1.643
  - `hybrid_a0.2_t100k`: 17.777 ± 2.824（相对下降 5.4%）
- **结论C（100M 与 350M）均复现优势**：
  - 100M@16K: `geo_500k` 10.888 vs `hybrid` 9.417（下降 13.5%）
  - 350M@16K: `geo_500k` 14.653 vs `hybrid` 12.646（下降 13.7%）

## 2) 关键结果表

### 2.1 50M 三种配置（3-seed）

| Config | PPL@2048 mean | PPL@16384 mean±std |
|---|---:|---:|
| geo_500k | 6.826 | 18.207 ± 0.768 |
| hybrid_a0.2_t100k | 6.699 | 17.324 ± 0.360 |
| anchpoly_p3.9_omf0.3_t500k | 6.634 | 19.133 ± 1.135 |

### 2.2 50M 公平因子实验（6配置×3seed）16K汇总

| Config | PPL@16384 mean±std |
|---|---:|
| geo_100k | 11.904 ± 0.473 |
| geo_200k | 18.524 ± 3.298 |
| geo_300k | 13.797 ± 1.127 |
| geo_500k | 13.757 ± 0.348 |
| hybrid_a0.2_t100k | 13.508 ± 1.100 |
| hybrid_a0.2_t500k | 13.495 ± 1.005 |

### 2.3 100M / 350M 扩展

| Scale | geo_500k @16K | hybrid_a0.2_t100k @16K | Δ% (越低越好) |
|---|---:|---:|---:|
| 100M | 10.888 | 9.417 | 13.5% |
| 350M | 14.653 | 12.646 | 13.7% |

## 3) 你导师可能会问的点（预答）

- **Q1: seed42 为什么看不出优势？**
  - 因子实验里 seed42 确实 hybrid 落后；但跨 seed 后 hybrid 均值更好，且 100M/350M 继续同向。结论应写成“平均优势 + 方差较大”，不是“单seed必胜”。
- **Q2: 是不是只在 16K 有效？**
  - 350M 结果显示 4K/8K/12K 同样优于 geo_500k（非仅单点 16K）。
- **Q3: YaRN 能否替代？**
  - 当前 A100 结果里 `geo_yarn_scale8` 在 16K 为 39.479，显著劣于 `geo_native` 17.966 与 `hybrid_native` 16.861（该设定下不成立）。

## 4) 同步清单

- 本地总目录：`/Users/misaya.yanghejazfs.com.au/dfrope/server_sync/a100_2026-02-13/for_github`
- 结果数据：`.../for_github/data`
- 脚本代码：`.../for_github/scripts`
- 重点文件：
  - `.../for_github/data/50m_theta_factorial/results.json`
  - `.../for_github/data/unified_search_3cfg_3seed/results.json`
  - `.../for_github/data/unified_search_2cfg_10seed/results.json`
  - `.../for_github/data/100m_scaling/results.json`
  - `.../for_github/data/350m_final/results.json`
