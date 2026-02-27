# 仓库结构与文件管理器（中文）

目标：任何人 `git clone` 后 30 秒内知道
1) 该看哪些文档；2) 该跑哪些脚本；3) 新的实验产物该放哪；4) 什么证据能进论文。

## 1) 顶层结构（建议阅读顺序见 `README.md`）

```text
hybrid-rope/
  README.md                    # 总览（给新合作者）
  AI_HANDOFF.md                # 交接入口（给 AI/换机）
  docs/                        # 论文口径与协议（权威）
  scripts/                     # 可复现实验入口（训练/评测/审计）
  rope/                        # 核心实现：schedule + 注入
  results/                     # 结果（paper_ready / advisor_package 等）
  paper_exports/               # 小文件导出包（表/图/统计 json/csv）
  artifacts/                   # 机器/批次快照（小文件、manifest、审计）
  outputs/                     # 本地临时输出（默认忽略，不入 git）
  archives/                    # 历史快照/批处理报告（不作为“当前”入口）
  experiments/                 # 单次实验/侧向项目（每个子目录自带 README）
  sigmoid_rope_experiments/    # Sigmoid 子项目（含 data/results）
  tools/                       # 工具脚本（同步、实用脚本）
  data/                        # 数据（通常不入 git，按协议拉取）
```

## 2) 哪些文件是“权威证据”

唯一权威事实表：`docs/EXPERIMENT_REGISTRY.md`  
补充状态索引：`docs/exp/EXPERIMENT_INVENTORY.md`（VALID/PENDING/INVALID/DEPRECATED）

**论文可引用（VALID）证据必须具备**：
- 可追溯：`config + code hash + env freeze + data/source + model/adapter + inv_freq hash`
- 可复现：有明确 `scripts/...` 入口命令
- 可审计：产物包含 per-sample traces 或至少 task-level 明细（视实验类型）

## 3) 新文件放哪（文件管理规则）

| 你要新增的东西 | 放哪里 | 命名建议 | 是否进 git |
|---|---|---|---|
| 论文口径协议/方法/结果说明 | `docs/` 或 `docs/protocols/` | `TOPIC_YYYY-MM-DD.md` | ✅（小文件） |
| 实验计划/进度/对照矩阵 | `docs/exp/` | `update_plan.md` / `EXPERIMENT_INVENTORY.md` | ✅ |
| 审稿意见/问题清单 | `docs/review/` | `YYYY_review_notes.md` | ✅ |
| 可复现实验脚本（稳定入口） | `scripts/` | `verb_object.py` | ✅ |
| 一次性/临时实验脚本 | `experiments/<name>/scripts/` | `exp_*.py` | ✅（若可复现） |
| 大体积权重/ckpt/原始 traces | 不进 git；放服务器盘或 quarantine | `...` | ❌ |
| paper-ready 小产物（表/图/统计） | `paper_exports/YYYY-MM-DD_*/` | `table_*.csv`, `fig_*.png` | ✅（小文件） |
| 本地运行日志/PID/临时输出 | `outputs/` | `*/run_*/` | ❌ |
| 历史快照（server_artifacts/batch_report） | `archives/` | 保留原目录名 | ✅（仅追溯用） |

## 4) 常用“去哪找”

- 想复现核心数字：`docs/REPRODUCE.md`
- 想知道能不能进论文：`docs/EXPERIMENT_REGISTRY.md`
- 想看当前结果汇总：`docs/RESULTS.md`
- 想跑 LongBench：`scripts/eval_longbench.py`
- 想估计 attention distance prior：`scripts/run_attn_hist.py`
- 想要换机无缝：`AI_HANDOFF.md` + `handoff/2026-02-25/0_README.md`

## 5) 归档策略（避免仓库继续变乱）

- 新的“批量报告/服务器镜像”统一进 `archives/`，并在 `docs/EXPERIMENT_REGISTRY.md` 写明引用规则。
- 新的实验产物默认进 `artifacts/`（小文件）与 `paper_exports/`（可用于论文的表/图/统计）。
- `experiments/` 下每个子项目必须有 `README.md`，并写清：
  - 目的、入口命令、产物路径、是否可引用（VALID/PENDING）。
