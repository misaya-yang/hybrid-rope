# AI Handoff (Read This First)

更新时间：2026-02-15 01:25（本地）

## 1. 项目目标

- 论文方向：Hybrid-RoPE（长上下文能力）。
- 当前核心任务：完成 8B 公平 LoRA 对比（YaRN / PI / Hybrid / PI-soft），并产出 NIAH + LongBench + PPL 证据链。

## 2. 关键环境约束（必须遵守）

- 本地仓库：`e:/rope/hybrid-rope`
- 远端仓库：`/root/autodl-tmp/dfrope/hybrid-rope`
- 用户无 HuggingFace 外网权限，不要依赖在线下载。
- 模型和数据优先使用本地已存在路径（Llama/Qwen/数据集已准备）。
- 同步结果时只拉数据，不拉权重。
- 不要删除用户已有实验，只做归档或隔离。

## 3. 当前已完成整理

- 已把历史 root 杂项远程脚本迁移到：`tools/remote_legacy/`
- 已做结果分层：
  - 主结果：`results/`
  - 低优先级归档：`results/archive_low_priority/`
  - 论文可读副本：`results/paper_ready/`
  - 权重隔离区：`results/_weights_quarantine/`
- 导师汇报包已生成：`results/advisor_package_2026-02-15/`
  - 入口：`results/advisor_package_2026-02-15/INDEX.md`
- 数据-only 同步工具已写：`tools/sync_results_data_only.ps1`

## 4. 当前训练状态（接管快照）

- 远端在跑：
  - `scripts/run_llama8b_fair_suite.py`
  - `scripts/train_llama8b_lora_variant.py --variant hybrid`
  - `scripts/run_8b_post_eval.py --wait_for_suite ...`
- 已完成：`yarn`、`pi`
- 正在进行：`hybrid`
- 最新日志（约 01:25）显示 `hybrid` 在 `~66/600`，loss 从 `5.13 -> 2.47`，显存约 `max_reserved_gb=81.66`，训练正常下降。

## 5. 最有价值证据文件（优先读）

- `results/advisor_package_2026-02-15/01_scaling_from_scratch/350m_final_results.json`
- `results/advisor_package_2026-02-15/01_scaling_from_scratch/unified_search_3cfg_3seed_results.json`
- `results/advisor_package_2026-02-15/02_llama_long_context/llama_shape_theta_min_results.json`
- `results/advisor_package_2026-02-15/03_llama8b_fair_lora/yarn_summary.json`
- `results/advisor_package_2026-02-15/03_llama8b_fair_lora/pi_summary.json`
- `results/advisor_package_2026-02-15/04_niah_and_retrieval/niah_results_base.json`
- `results/advisor_package_2026-02-15/05_qwen_and_cross_model/qwen_hybrid_lora_eval_suite.json`

## 6. 下一步该做什么（按优先级）

1. 等待 8B fair suite 完成 `hybrid` 和 `pi_soft`。
2. 确认 `run_8b_post_eval.py` 自动启动并产出：
   - NIAH（单针/多针热力图）
   - LongBench（qasper/hotpotqa/gov_report）对比 JSON
3. 将新增结果做 data-only 回传，刷新：
   - `results/llama8b_fair_lora_suite_20260214`
   - `results/llama8b_post_eval_20260214`
4. 更新 `results/advisor_package_2026-02-15/INDEX.md` 的最终数值。
5. 若要对外汇报，优先从 `results/advisor_package_2026-02-15/` 出图表和表格。

## 7. 常用接管命令

```powershell
# 看远端训练进程
C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "ps -eo pid,etimes,cmd | grep -E 'run_llama8b_fair_suite.py|train_llama8b_lora_variant.py|run_8b_post_eval.py' | grep -v grep"

# 看 fair suite 日志尾部
C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "tail -n 60 /root/autodl-tmp/dfrope/hybrid-rope/logs/fair_suite_8b_run.log"

# 看 post-eval 等待日志
C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "tail -n 60 /root/autodl-tmp/dfrope/hybrid-rope/logs/post_eval_8b_run.log"

# data-only 同步（不传权重）
powershell -ExecutionPolicy Bypass -File tools/sync_results_data_only.ps1 `
  -RemoteHost connect.bjb1.seetacloud.com `
  -RemotePort 42581 `
  -RemoteUser root `
  -RemotePassword "htG0sD63/yG0" `
  -RemoteRepoRoot "/root/autodl-tmp/dfrope/hybrid-rope" `
  -LocalRepoRoot "." `
  -ResultDirs @("llama8b_fair_lora_suite_20260214","llama8b_post_eval_20260214")
```

## 8. 注意事项

- `results/350m_final/` 当前有效结果文件名是 `results1.json`（不是 `results.json`）。
- `results/_weights_quarantine/` 是本地权重隔离区，不用于汇报。
- 汇报材料优先使用：`results/advisor_package_2026-02-15/`。

