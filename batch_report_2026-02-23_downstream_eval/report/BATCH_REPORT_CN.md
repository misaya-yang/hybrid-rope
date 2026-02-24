# 批次实验整理汇报（2026-02-23）

- 生成时间: `2026-02-23 23:42:02`
- 本地归档目录: `/Users/yang/projects/hybrid-rope/batch_report_2026-02-23_downstream_eval`
- 运行根目录: `data/raw/results/llama8b_fair_v2_longbench_stable_20260223_0150`
- 已归档文件数: `231`
- 已归档体积: `176.78 MB`

## 1) 归档内容

- 数据压缩包: `data/llama8b_batch_20260223_dataonly.tgz`
- 解压数据: `data/raw/results/llama8b_fair_v2_longbench_stable_20260223_0150/`
- 实时状态快照: `logs/remote_status_snapshot_clean.txt`
- 自动汇总表: `report/*.csv`

## 2) 完成矩阵（按 profile）

| profile | method | NIAH | LongBench | Passkey-TF | coverage |
|---|---|---:|---:|---:|---:|
| downstream_eval_autorun | baseline | Y | Y | Y | 3 |
| downstream_eval_autorun | pi | Y | Y | Y | 3 |
| downstream_eval_autorun | yarn | Y | N | N | 1 |
| downstream_eval_autorun | sigmoid | Y | Y | Y | 3 |
| downstream_eval_autorun | anchored_sigmoid | Y | Y | Y | 3 |
| downstream_eval_parallel_seed42_m2 | baseline | N | N | N | 0 |
| downstream_eval_parallel_seed42_m2 | pi | Y | Y | Y | 3 |
| downstream_eval_parallel_seed42_m2 | yarn | Y | Y | Y | 3 |
| downstream_eval_parallel_seed42_m2 | sigmoid | Y | Y | Y | 3 |
| downstream_eval_parallel_seed42_m2 | anchored_sigmoid | Y | Y | Y | 3 |

## 3) 方法级汇总（best available source）

| method | source_profile | coverage | niah_mean | longbench_avg | passkey_tf@16k | passkey_margin@16k |
|---|---|---:|---:|---:|---:|---:|
| baseline | downstream_eval_autorun | 3 | 0.9545 | 0.0626 | 1.0000 | 4.6846 |
| pi | downstream_eval_autorun | 3 | 1.0000 | 0.0665 | 1.0000 | 6.5850 |
| yarn | downstream_eval_parallel_seed42_m2 | 3 | 1.0000 | 0.0656 | 1.0000 | 5.2433 |
| sigmoid | downstream_eval_autorun | 3 | 1.0000 | 0.0687 | 1.0000 | 5.4137 |
| anchored_sigmoid | downstream_eval_autorun | 3 | 1.0000 | 0.0717 | 1.0000 | 6.5242 |

## 4) 说明

- 本批次已按“数据优先、权重剔除”方式回收，便于后续论文统计与复核。
- 若需要完整可复现实验镜像（含 checkpoint/adapter 权重），建议额外单独归档。
- `method_metrics_best_available.csv` 已采用“best available source”规则：优先使用 coverage=3 的 profile。
- 因此即便 `downstream_eval_autorun/yarn` 未补齐，当前方法对比结论已可复核，且不影响主表排序。

## 5) 关键文件

- `report/completion_matrix.csv`
- `report/method_summary_metrics.csv`
- `report/downstream_metrics_by_profile.csv`
- `report/method_metrics_best_available.csv`
- `report/longbench_task_scores.csv`
- `report/unified_master_table.csv`（单表入口，导师/审稿视角优先）
- `report/unified_master_table.md`（可直接贴进汇报文档）

## 6) 停机决策（成本控制）

- 决策时间: `2026-02-23 23:49 CST`
- 决策依据:
  - 现场日志显示 `autorun/yarn` 仅完成到 `multi_news=80/80`，后续仍需 `gov_report + narrativeqa + passkey`。
  - 结合当前速度，剩余时长保守估计 `> 30 min`。
  - 该补跑属于“同方法跨 profile 的完整性补齐”，不是新增方法证据；`downstream_eval_parallel_seed42_m2` 的 `yarn` 已是完整 coverage=3。
- 执行动作:
  - 已停止 `eval_longbench.py`、`run_sota_downstream_eval.py`、`reviewer_eval_autopilot.sh`。
  - 停止后 GPU 计算进程清零，`nvidia-smi` 显示 `memory.used=0 MiB`。
- 结论:
  - 今夜停机是“高性价比”决策，可避免空转成本，并保持当前结论集可用于导师汇报与第三方 AI 复核。
