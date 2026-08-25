# docs/ — 历史文档层

**这个目录不是当前权威。** 当前入口是根级
[`AGENTS.md`](../AGENTS.md)（规则）→ [`INDEX.md`](../INDEX.md)（索引）→
[`paper-2027/HANDOFF.md`](../paper-2027/HANDOFF.md)（状态）。

`docs/` 保存 NeurIPS-era 的 provenance、复现路径、历史实验报告和被取代的理论
推导。它有用，但**不覆盖**当前 ICLR 路由。任何冲突以 `INDEX.md` §3 指向的
canonical owner 为准。

## 子目录

| 目录 | 内容 | 状态 |
| --- | --- | --- |
| `overview/` | provenance manifest、claims map、复现、数据准备、术语、Blackwell profile | **仍在维护**：provenance 与复现的 owner |
| `exp/` | 历史实验报告，`YYYY-MM-DD_slug.md` | 归档层；claim 归属见 `INDEX.md` §3 |
| `theory/` | 早期理论推导与数值验证 | 只读，已被 `INDEX.md` §2.1 取代 |
| `tau_algor/` | τ scaling / habitable zone / softmax transport 原始推导（2026-03） | 只读；取代关系见 `INDEX.md` §2.3。由 `tau-theory-assistant` skill 使用，保留原位 |
| `archive/` | 明确退役的 τ 理论文档 | 只读 |

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

- 新实验报告用 `YYYY-MM-DD_slug.md` 放进 `exp/`，并在 `INDEX.md` §3 登记 owner。
- **不要**在 `docs/` 下新建 README、索引或 handoff。索引只有 `INDEX.md` 一份。
- 缺 raw artifact 时只能写 report-backed / missing-artifact，不能用叙述文档升级
  证据。
