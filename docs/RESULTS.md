# 实验结果（中文）

本文件汇总当前已完成的关键结果，并给出“论文口径”建议（哪些能下结论，哪些必须等更多 seed/更大模型验证）。

## 1. 50M：unified_search 小规模扫描（单 seed）

实验脚本：`a100/unified_search.py`  
数据/训练/评测定义见：`docs/METHODOLOGY.md`

### 1.1 A100（split A）最优结果（按 PPL@16384 升序）

来源：`results/unified_search/results_A.json`

| Config | PPL@2048 | PPL@16384 | 备注 |
|---|---:|---:|---|
| `anchpoly_p3.9_omf0.3_t500k` | 6.578 | 16.459 | 单次 run 看起来略优于 geo_500k |
| `geo_500k_ALIGN` | 6.804 | 16.991 | 高 theta baseline |

### 1.2 A800（split B）最优结果（按 PPL@16384 升序）

来源：`results/unified_search/results_B.json`（如果你本地暂时没同步到，可从远端拉取）

| Config | PPL@2048 | PPL@16384 | 备注 |
|---|---:|---:|---|
| `hybrid_a0.2_t100k` | 6.658 | 16.316 | 单次 run 显著优于 geo_500k |
| `geo_500k_ALIGN` | 6.852 | 17.947 | 高 theta baseline |

**结论（单 seed 级别）：**

- `hybrid_a0.2_t100k` 有很强信号：用 `theta=100k` 的 hybrid，可以打赢 `theta=500k` 的 geometric。
- `anchpoly_p3.9_omf0.3_t500k` 的优势较小，必须做多 seed 才能确认是不是“lucky seed”。

## 2. 50M：3 配置 × 3 seed 稳健性验证（论文级关键）

脚本：`a100/unified_search_3cfg_3seed.py`  
结果：`results/unified_search_3cfg_3seed/results.json`

Seeds：`[42, 123, 7]`

### 2.1 mean ± std（以 PPL@16384 为外推主指标）

| Config | PPL@2048 (mean±std) | PPL@16384 (mean±std) |
|---|---:|---:|
| `geo_500k` | 6.826 ± 0.048 | 18.207 ± 0.768 |
| `hybrid_a0.2_t100k` | 6.699 ± 0.168 | **17.324 ± 0.360** |
| `anchpoly_p3.9_omf0.3_t500k` | 6.634 ± 0.192 | 19.133 ± 1.135 |

### 2.2 论文口径（可直接写）

- **主结论（稳健）：** `hybrid_a0.2_t100k` 在 3 个 seed 下，`PPL@16384` 的 **均值优于** `geo_500k`，且 std 更小，说明“hybrid 替代极大 theta”不是偶然。
- **否定结论（同样有价值）：** `anchpoly_p3.9_omf0.3_t500k` 在多 seed 下 **均值更差且方差更大**，单次 run 的 16.459 更像是 lucky draw 或数据流噪声，不建议将其作为主结论。

## 3. 350M：终极规模验证（已完成，seed=42）

脚本：`a100/run_350m_final.py`  
远端日志：`/opt/dfrope/results/350m_final/run.log`

结果文件：
- `results/350m_final/results.json`
- `results/350m_final/run.log`

### 3.1 核心结果（PPL）

| Length | geo_500k | hybrid_a0.2_t100k |
|---|---:|---:|
| 2048 | 2.477 | 2.467 |
| 3072 | 2.511 | 2.489 |
| 4096 | 2.784 | 2.606 |
| 5120 | 3.213 | 2.808 |
| 6144 | 3.920 | 3.349 |
| 8192 | 5.452 | 4.726 |
| 12288 | 10.370 | 8.814 |
| 16384 | 14.653 | **12.646** |

### 3.2 结论

- 在 350M 上，`hybrid_a0.2_t100k` 仍然显著优于 `geo_500k`。
- 主指标 `PPL@16384`：
  - `geo_500k = 14.653`
  - `hybrid = 12.646`
  - 相对提升约 **13.7%**

这说明“hybrid 用中等 theta 替代极大 theta”不是仅限 50M 小模型的现象，在更大规模上依然成立（至少在当前 seed=42 和相同协议下成立）。

## 4. 50M：Hybrid vs Geo vs YaRN（修正版）

结果文件：`results/50m_yarn_compare_v2/results.json`  
完整说明：`docs/YARN_COMPARISON_2026-02-13.md`

| Length | Hybrid (native) | Geo (native) | Geo + YaRN (progressive) |
|---|---:|---:|---:|
| 2048 | **6.672** | 6.839 | 6.839 |
| 4096 | **6.748** | 7.045 | 8.640 |
| 8192 | **8.688** | 8.833 | 16.899 |
| 12288 | **13.333** | 13.588 | 29.352 |
| 16384 | **16.861** | 17.966 | 39.479 |

补充说明：该版本已修正 YaRN 评测口径（训练长度 2048 不缩放，仅在超出训练长度时渐进缩放）。
