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

## 3. 350M：终极规模验证（运行中）

脚本：`a100/run_350m_final.py`  
远端日志：`/opt/dfrope/results/350m_final/run.log`

注意：

- 350M 实验包含 500M token 的 streaming tokenization -> 磁盘 memmap cache，这一阶段 GPU 利用率可能为 0%，属于正常现象。
- 真正训练开始后，`nvidia-smi` 会显示显存占用与进程。

