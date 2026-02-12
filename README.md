# RoPE 频谱优化实验（DF-RoPE Pivot）

这个目录是“可上传 GitHub 的整理包”，包含我们从 DF-RoPE pivot 到 **RoPE 频率分布优化** 之后的代码与结果（不含权重）。

## 目录结构

- `a100/`
  - `unified_search.py`: 50M 模型小规模候选扫描（unified search）
  - `unified_search_3cfg_3seed.py`: 50M 模型 3 配置 × 3 seed 稳健性验证
  - `run_350m_final.py`: 350M 最终验证（运行中）

- `docs/`
  - `DFROPE_EXPERIMENTS_ROPE_FREQ_SUMMARY.md`: 全过程叙事总结（较长）
  - `METHODOLOGY.md`: 不允许变动的定义（频率函数/数据/评测 slicing）
  - `RESULTS.md`: 关键表格与“论文口径”结论
  - `REPRODUCE.md`: 复现指南（中文）

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
