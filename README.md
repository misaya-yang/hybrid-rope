# Hybrid RoPE 实验仓库

这是一个“**代码 + 结果 + 文档**”一体化实验仓库，覆盖：

- 50M/100M/350M from-scratch 训练与外推评测
- Llama-3-8B / Qwen2.5-7B 的 Hybrid-LoRA 实验
- 频率分布搜索（geometric / hybrid / sigmoid / anchored polynomial）
- H100 1.5B 规模化计划包

说明：仓库默认**不包含模型权重**，只保留可复核结论所需代码与产物。

## 5 分钟上手

1. 看结论：`docs/RESULTS.md`
2. 看方法口径：`docs/METHODOLOGY.md`
3. 看实验索引：`docs/EXPERIMENT_INDEX_CN.md`
4. 看机器产物：`artifacts/`
5. 需要复现时：`docs/REPRODUCE.md`

## 主目录说明

- `artifacts/`
  - 按机器归档的主入口（A100/A800/R6000），包含脚本、结果、日志、图表
- `docs/`
  - 方法、结果、复现文档，以及实验索引表
- `results/`
  - 聚合后的核心结果快照（便于快速查阅）
- `h100_advanced_experiments/`
  - H100/H200 执行包（计划、配置、runbook、脚本）
- `server_artifacts_2026-02-13/`
  - A100/A800 服务器原样镜像归档（复核用途）
- `a100/`
  - 早期历史兼容目录（与 `artifacts/a100_2026-02-13/` 有重叠）
- `scripts/`
  - 仓库级工具脚本（拉取、作图、提交辅助）

## 机器入口

- A100 主线：`artifacts/a100_2026-02-13/README.md`
- A800 LoRA：`artifacts/a800_2026-02-13/README.md`
- R6000 Qwen：`artifacts/r6000_2026-02-13/README.md`

## 关键约定

- 不上传权重文件（`*.pt`, `*.bin`, `*.safetensors` 等）
- 仅同步可复现和可审阅所需产物（脚本、JSON、日志、图表）
- 每个主要目录都有 `README.md` 说明用途和入口

## 推荐阅读顺序（对导师/审稿友好）

1. `docs/RESULTS.md`
2. `docs/METHODOLOGY.md`
3. `docs/EXPERIMENT_INDEX_CN.md`
4. `docs/REPRODUCE.md`
5. `artifacts/a100_2026-02-13/A100_RESULTS_SUMMARY_FOR_ADVISOR_2026-02-13.md`
