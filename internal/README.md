# Internal Archive

`internal/` 保存历史工作快照、旧稿、恢复线索和本机归档。它不是当前 paper、rebuttal 或 reviewer supplement 的入口。

| 路径 | 说明 |
| --- | --- |
| `2026_03_run/`, `2026_04_run/` | 按月份保存的历史研究记录与 paper 快照 |
| `brief/` | 内部简报源文件与导出物 |
| `draft_scripts/` | 旧脚本和图形探索，不是 canonical implementation |
| `paper_plans/`, `pitfalls/` | 历史计划与风险记录 |
| `team/` | 仅保留少量 paper-era 理论分析；协作 handoff 与状态文件已清理 |
| `local_archive/`, `local_snapshots/` | 本机 ignored；可能含大数据、私有路径或恢复 artifact |

规则：

- canonical code 在 `scripts/` 和 `experiments/`；
- 当前实验 provenance 在 `docs/overview/RESULT_PROVENANCE_MANIFEST.md`；
- 当前 rebuttal 在 `rebuttal/`；
- 不从 internal 文档直接复制 reviewer-facing 数字；
- 不将整个 `internal/` 打包或公开。

旧协作 handoff、代理日志、聊天脚手架和非理论型重复审计壳已从
`main_0726` 清理；paper-era 理论推导与验证脚本保留。需要追溯被删材料时
使用 `main` 或备份分支。
