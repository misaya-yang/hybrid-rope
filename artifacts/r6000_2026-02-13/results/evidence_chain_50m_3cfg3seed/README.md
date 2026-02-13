# 50M 证据链复验（3 configs x 3 seeds）

状态：RUNNING

## 远端任务
- 机器：R6000 (`connect.bjb1.seetacloud.com:42581`)
- 脚本：`/root/autodl-tmp/dfrope/hybrid-rope/scripts/run_50m_3cfg3seed_evidence.py`
- 输出目录：`/root/autodl-tmp/dfrope/hybrid-rope/results/evidence_chain_50m_3cfg3seed/`
- 进程 PID：`16844`

## 目标
- 在统一口径下复验 3 组配置：
  - `geo_500k`
  - `hybrid_a0.2_t100k`
  - `anchpoly_p3.9_omf0.3_t500k`
- seeds：`42/123/7`
- 输出：`results.json`（包含每个配置每个 seed 的 PPL）

## 本地归档计划
任务完成后同步以下文件：
- `run.log`
- `results.json`
- `summary.md`（自动或人工生成）

