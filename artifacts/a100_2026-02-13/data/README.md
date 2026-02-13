# A100 data 目录说明

本目录保存 A100 实验原始结果与运行日志。

## 子目录

- `50m_theta_factorial/`：theta × shape 因子公平对照
- `50m_yarn_compare/`：早期 YaRN 对照
- `50m_yarn_compare_v2/`：修订版 YaRN 对照
- `100m_scaling/`：100M scaling 冲刺结果
- `350m_final/`：350M final 对照
- `350m_hybrid*/`：350M hybrid 系列探索
- `350m_validation/`：350M 验证实验
- `unified_search/`：统一搜索（A split）
- `unified_search_2cfg_10seed/`：2 配置 10 seed 稳健性
- `unified_search_3cfg_3seed/`：3 配置 3 seed 稳健性
- `a100_aligned/`：跨机对齐验证产物
- `round3_50m/`：round3 阶段产物

## 使用建议

先看每个子目录下 `results.json`，再看 `run.log`。
