# internal/ — NeurIPS-era 工作归档

**状态：只读历史层。** 不是当前 paper、rebuttal 或 reviewer supplement 的入口。
当前入口是根级 [`AGENTS.md`](../AGENTS.md) → [`INDEX.md`](../INDEX.md) →
[`paper-2027/HANDOFF.md`](../paper-2027/HANDOFF.md)。

内容是 2026-03/04 冲刺期的研究记录、旧稿快照、计划与审计。完成本次重复清理后
保留约 6 MB、91 个 tracked 文件。

| 路径 | 内容 | 备注 |
| --- | --- | --- |
| `2026_03_run/` | 三月研究记录：数据清单、结果表、finetune 诊断、论文审计、技术债审计、AI handoff | `docs/` 编号 01–15，其中 07/08/09 三份 τ 文档已删（与 `docs/tau_algor/`、`docs/archive/` 逐字节重复），原文见那里 |
| `2026_04_run/` | 四月冲刺：计划、行文策略、六大理论问题、λ 闭合、τ 决策、实验手册、rebuttal 骨架、理论攻击 C1–C3 | 含 `paper_snapshot_0417_writing_fixed.pdf` 与 `audit_archives/`（Bessel 代换与理论一致性验证脚本） |
| `paper_plans/` | NeurIPS 投稿计划、`CORE_THEORY.md`、`SECONDARY_THEORY.md`、错误更正、LaTeX 片段 | 三份 τ 文档已删（与 `docs/tau_algor/` 重复） |
| `brief/` | 内部简报源文件与中英文 PDF 导出 | — |
| `draft_scripts/` | 旧脚本与图形探索 | **不是** canonical implementation |
| `pitfalls/` | `AI_HANDOFF_PITFALLS.md` | 历史风险记录 |
| `reviews/` | 代码评审与接收评估 docx | — |
| `team/` | 少量 paper-era 理论分析与 legacy handoff | 协作状态文件已清理 |
| `tools/` | `neurips-paper.skill`、`tau-theory-assistant.skill` 及其源目录 | skill 的仓库内历史备份 |

## 规则

- canonical code 在 `scripts/` 与 `experiments/`，不在这里；
- 当前实验 provenance 在 `docs/overview/RESULT_PROVENANCE_MANIFEST.md`；
- 当前理论权威见 [`INDEX.md`](../INDEX.md) §2；
- 不从 internal 文档直接复制 reviewer-facing 数字；
- 不将整个 `internal/` 打包或公开；
- 本目录不再接收新文件。

2026-08-24 清理：删除 8 个与 `docs/` 或不可变 `paper/` 逐字节重复的文件；
README 移除了两个已不存在的目录（`local_archive/`、`local_snapshots/`）。
更早的协作 handoff、代理日志与聊天脚手架在 `main_0726` 上已清理；需要追溯被删
材料时使用 `main` 或备份分支。
