# 仓库结构总览（中文）

## 顶层结构

```text
github_bundle/
  README.md
  docs/                          # 方法/结果/复现/索引文档
  artifacts/                     # 按机器归档的主产物
    a100_2026-02-13/
    a800_2026-02-13/
    r6000_2026-02-13/
  results/                       # 聚合结果快照
  h100_advanced_experiments/     # H100 计划与执行包
  server_artifacts_2026-02-13/   # 服务器原样镜像归档
  scripts/                       # 仓库级工具脚本
  a100/                          # 历史兼容目录
```

## 查找路径建议

- 找实验脚本：
  - `artifacts/<machine>_2026-02-13/scripts/`
- 找关键 JSON：
  - `artifacts/<machine>_2026-02-13/data/` 或 `results/`
- 找导师汇报材料：
  - `artifacts/<machine>_2026-02-13/*SUMMARY*.md`
- 找实时状态（R6000）：
  - `artifacts/r6000_2026-02-13/live_sync/`

## 最少必看文件

1. `README.md`
2. `docs/RESULTS.md`
3. `docs/METHODOLOGY.md`
4. `docs/EXPERIMENT_INDEX_CN.md`
