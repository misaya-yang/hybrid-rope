# scripts 目录说明

本目录是仓库级通用脚本（不绑定单一实验）。

## 文件

- `plot_yarn_compare.py`：绘制 50M YaRN 对比图
- `pull_a100_350m_artifacts.sh`：从 A100 拉取 350M 相关产物
- `commit_and_push.sh`：快速提交并推送
- `prepare_longbench_local_data.py`：下载并固化 LongBench 子任务到本地 jsonl（离线评测，支持 `--source auto/hf/modelscope/dashscope`）
- `eval_passkey_teacher_forcing.py`：Passkey teacher-forcing 真/假候选对比评测
- `run_sota_downstream_eval.py`：统一运行 NIAH/LongBench/Passkey 并产出论文表格和图（支持断点续跑，默认跳过已完成输出）
- `reviewer_eval_autopilot.sh`：审稿人视角的自动化评测守护脚本（排队多 seed、自动续跑、防 GPU 空转）

## 使用建议

- 具体实验脚本优先看 `artifacts/<machine>_2026-02-13/scripts/`
- 本目录脚本更多是辅助同步/画图/维护
