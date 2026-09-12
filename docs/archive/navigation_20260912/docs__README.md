# docs/ — 研究记录与历史文档

默认入口是根级[`AGENTS.md`](../AGENTS.md)（规则）、[`README.md`](../README.md)
（项目背景）和[`paper-2027/HANDOFF.md`](../paper-2027/HANDOFF.md)（唯一实时状态）。
按[`INDEX.md`](../INDEX.md)定位具体问题的owner；文件夹名称不决定证据级别。

`research/`保存近期跨阶段研究方案、实验复盘、原始结果摘要及机器可读证据；
`exp/`保存按月份归档的早期实验。论文层面的理论和结果仍按
[`paper-2027/research/`](../paper-2027/research/README.md)分层。每份报告只在其
声明的协议和证据范围内拥有结论，历史方案不是待执行队列。

## 子目录

| 目录 | 内容 | 状态 |
| --- | --- | --- |
| `research/` | 近期研究协议、实验复盘、失败谱系、数组与证据JSON | 按INDEX和各owner的日期/状态读取；运行授权只看HANDOFF |
| `overview/` | provenance manifest、claims map、复现、数据准备、术语、Blackwell profile | **仍在维护**：provenance 与复现的 owner |
| `theory/`、`tau_algor/`、`archive/` | pre-slim时期的理论与退役材料 | 归档位置通过INDEX或`git show main_0726:<path>`查阅，不恢复为当前队列 |
| `exp/` | 历史实验报告，按 `YYYY-MM/YYYY-MM-DD_slug.md` 分层 | 保留原数字；后续纠正放在原owner可见位置 |

## 这里仍然拥有的东西

| 需求 | 文件 |
| --- | --- |
| 结果是否 reviewer-safe / 哈希 | `overview/RESULT_PROVENANCE_MANIFEST.md` |
| 从 Figure/Table 找生成脚本 | `overview/PAPER_CLAIMS_MAP.md` |
| 复现核心结果 | `overview/REPRODUCE.md` |
| 数据来源与准备 | `overview/DATA_PREPARATION.md` |
| 指标与协议词汇 | `overview/TERMS_AND_PROTOCOLS.md` |
| RTX 5090 / Blackwell 运行时 | `overview/RTX5090_BLACKWELL_PROFILE.md` |

## 维护规则

- 近期跨阶段研究记录放在`research/`，早期实验归档按`exp/YYYY-MM/`组织；
  更新或新增owner时，同步根INDEX的对应条目，不依赖易变化的章节编号。
- 综合报告负责结论和证据边界，配套JSON负责来源/数组/回执身份，协议负责
  问题、预测和条件；实时进程、预算与下一行动只在HANDOFF维护。
- **不要**在 `docs/` 下新建 README、索引或 handoff。索引只有 `INDEX.md` 一份。
- 缺 raw artifact 时只能写 report-backed / missing-artifact，不能用叙述文档升级
  证据。
