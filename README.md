# RoPE 频谱优化实验（DF-RoPE Pivot）

这个目录是“可上传 GitHub 的整理包”，包含我们从 DF-RoPE pivot 到 **RoPE 频率分布优化** 之后的代码与结果（不含权重）。

## 目录结构

- `a100/`
  - `unified_search.py`: 50M 模型小规模候选扫描（unified search）
  - `unified_search_3cfg_3seed.py`: 50M 模型 3 配置 × 3 seed 稳健性验证
  - `run_50m_theta_shape_factorial.py`: 50M 因子分离公平对照（geo θ 扫描 + hybrid 对照）
  - `run_350m_final.py`: 350M 最终验证（运行中）

- `h100_advanced_experiments/`
  - 2xH100 / 1.5B 实验作战包（计划、runbook、配置、环境检查、自动汇总作图脚本）
  - 可直接迁移到 H100/H200 新环境执行

- `server_artifacts_2026-02-13/`
  - 双机（A100/A800）最新同步归档（不含权重）
  - 已按 `scripts/results/logs/meta` 分层，可直接用于复核与上传
  - 包含 A100 100M scaling/因子实验中间产物、A800 LoRA 运行日志
  - 详见 `server_artifacts_2026-02-13/README.md`

- `artifacts/a800_2026-02-13/`
  - A800 上 `Llama-3-8B Hybrid-LoRA` 的训练与对比评测入库（不含权重）
  - 包含原始 `run.log/results.json`、评测脚本与导师汇报摘要
  - 详见 `artifacts/a800_2026-02-13/README.md`

- `docs/`
  - `DFROPE_EXPERIMENTS_ROPE_FREQ_SUMMARY.md`: 全过程叙事总结（较长）
  - `METHODOLOGY.md`: 不允许变动的定义（频率函数/数据/评测 slicing）
  - `RESULTS.md`: 关键表格与“论文口径”结论
  - `REPRODUCE.md`: 复现指南（中文）
  - `SERVER_SYNC_2026-02-13.md`: 本次双机同步记录、排除规则与状态快照

- `results/`
  - `unified_search/`: unified 扫描的 JSON + log
  - `unified_search_3cfg_3seed/`: 3cfg×3seed 的 JSON + log
  - `350m_final/`: 350M 的 run.log / results.json（不含权重，不含 memmap cache）

## 约定与说明

- 不上传权重：体积大且不必要，我们关心可复现的对比结论。
- 350M 实验会把 streaming tokenization 写成磁盘 memmap cache（体积极大），该 cache 不同步到 GitHub。

## 推荐阅读顺序

1. `docs/RESULTS.md`
2. `docs/METHODOLOGY.md`
3. `docs/REPRODUCE.md`
4. `docs/DFROPE_EXPERIMENTS_ROPE_FREQ_SUMMARY.md`
