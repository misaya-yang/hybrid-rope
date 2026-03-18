# Qwen Fast400 结果回收归档（2026-02-26）

## 1) 归档目标

将服务器上的 `seed42 + seed1337` 双 seed 评测结果完整回收至仓库，供后续“是否写入论文”决策使用。  
注意：本组实验当前定位为 `trade-off 证据`，不作为“主提升证据”。

## 2) 已回收目录（本地仓库）

- `artifacts/reviewer_2026-02-25/h1_baseline_gold_seed42/`（约 10MB）
- `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/`（约 21MB）
- `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/`（约 21MB）

## 3) 核心结果文件与 SHA256

- `artifacts/reviewer_2026-02-25/h1_baseline_gold_seed42/longbench_lb21_baseline_gold_seed42.json`
  - `7c14810213b4a60f15e8212344fced5ada75286058f623acca9a6c0643d50f79`
- `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/downstream_eval_fast400/longbench/baseline.json`
  - `0c0f9f6c6aa9365de72a167ce88d64f26dbf7d1cf194fca7b6fc7edfbb1a8386`
- `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/downstream_eval_fast400/longbench/anchored_sigmoid.json`
  - `e6199f8dad8363426e7d74d45b7ddf051b8bcac22756ba6abe0d186e14a71a16`
- `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/downstream_eval_fast400/longbench/baseline.json`
  - `d94e51d2d5ca0f622cb61b7f9e6b1b4b1740d889c37784996d0a59bde06999f4`
- `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/downstream_eval_fast400/longbench/anchored_sigmoid.json`
  - `d504ea45bc1c8dfede21a748540da48e7c101aa22a4e9dd534e71c81e8ec2de2`
- `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/significance_full21_fdr_qwen.json`
  - `b9986541f33fdecd163dee1ceaf910d907098ae9cd53850fe14b9b91ef7d9413`
- `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/significance_full21_fdr_qwen.csv`
  - `bed43263fe8a1fad1f956fdb241584fc84c814fc42743515f83d545fed192e95`
- `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/claim_policy_report.md`
  - `79bb4294202131fa683ee35957c6c101e19622fad5944d99cf4dc107d15de839`

## 4) 结果研判文档

- 主分析报告：
  - `docs/exp/QWEN_FAST400_DUAL_SEED_ANALYSIS_2026-02-26.md`

该报告已包含：
- 双 seed 全局均值对比；
- 分任务涨跌与任务族群分解；
- per-sample + FDR 统计结论；
- 是否可用于论文主证据的建议边界。

## 5) 当前结论标签（供论文决策）

- 标签：`Directional trade-off / not strong enough for main claim`
- 原因：
  - 双 seed 宏平均均为负差；
  - per-sample FDR 后为 `no_improvement`；
  - 结果更支持“检索-推理 trade-off”而非“全局提升”。

## 6) 与今日训练的关系

本归档已完成，后续重点转回今日训练主线。  
本次归档不干预正在运行的训练进程，仅做结果回收与文档固化。

