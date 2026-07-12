# EVQ-Cosh Rebuttal Control Room

创建：2026-06-10；最后更新：2026-07-12

- Preparation：`triage_ready`
- Response package：`needs_author_input`
- 当前模式：`triage-only`。截至 2026-07-12，实际 NeurIPS reviews 尚未收到。

本目录是作者内部 rebuttal control room，不是论文、公开补充材料或 reviewer supplement，也不得作为仓库根目录打包。唯一中心主张是：**RoPE 的有限频率表也是 finite spectral budget；EVQ-Cosh 把 training-time frequency allocation 作为 operator design 与 inference-time range scaling 之外的第三个 PE 设计轴。** 这不是 universal long-context SOTA，也不是 YaRN、LongRoPE、DAPE、FIRE 或 learned PE 的替代主张。

## 1. 一个统一策略入口 + 五份分轨材料 + 一个原文索引

| 文件 | 唯一职责 |
| --- | --- |
| `rebuttal_playbook.md` | **统一策略入口**：response-only 原则、P0/P1/P2、理论边界与最小行动清单 |
| `README.md` | 全局状态、导航与目录安全边界 |
| `REVIEWER_TRIAGE_PLAYBOOK.md` | 真实 reviews 到来后的最多五项 score-driving concern 分流 |
| `REBUTTAL_MASTER_QUESTION_LEDGER_20260711.md` | 完整问题、证据、状态与 decision gate 总账 |
| `THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md` | 理论数学权威：exact / conditional proxy / empirical 三层边界 |
| `LORA_GEO_CONTROL_RESULT_AUDIT_20260711.md` | fresh Geo 结果、LongAlign/LongAlpaca provenance 与 clean-pair 因果边界 |
| `raw_sources/00_INDEX.md` | tracked 控制索引；指向 local-only 原文并规定模拟材料隔离规则 |

已删除的旧文档仅可从 Git 历史追溯，不是入口，不得覆盖以上控制文件或最新 provenance。

## 2. 权威来源必须分轨

| 需要判断什么 | 权威来源 | 不得误用 |
| --- | --- | --- |
| submitted/current source 写了什么 | 提交件与 `paper/` 当前源码，使用时明确版本 | 只能证明某版本的文字、图表和声明；不能覆盖数学审计，也不能单独证明实验 provenance |
| 实验数字与 provenance | `docs/overview/RESULT_PROVENANCE_MANIFEST.md` 及其指向的 raw-backed artifacts | 已移出根目录的 2026-06-14 snapshot 只保存在 ignored local snapshot 中，不得恢复为权威 |
| 理论正确性 | `THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md` | submitted/current paper source 只证明“写了什么”，不裁决数学是否正确 |
| LoRA 因果与数据来源 | `LORA_GEO_CONTROL_RESULT_AUDIT_20260711.md` | fresh Geo 使用 official LongAlign；historical EVQ 指向 LongAlpaca。没有 same-data fresh EVQ pair 前不得作 causal attribution |
| 当前决策与可发送范围 | `REBUTTAL_MASTER_QUESTION_LEDGER_20260711.md`；真实评论的快速映射用 `REVIEWER_TRIAGE_PLAYBOOK.md` | 旧草稿、旧 action board 或单次内部报告不得反向升级 claim |
| 模拟审稿原文 | `simulated_reviews/` | 只用于内部压力测试；不是真实 NeurIPS reviews，其科学判断已被 2026-07-11 audits 与总账取代 |

## 3. 2026-07-22 真实 review 工作流

真实 reviews 到来前只做 triage 准备，不创建或预填 author response。

1. local-only 保存每条真实评论的逐字版本，并保留 reviewer / AC 身份标签。
2. 按 `REVIEWER_TRIAGE_PLAYBOOK.md` 分配稳定 ID；ID 一旦分配，不因排序变化而重编号。
3. 对每个被触发 concern 填写 `correction / concession / evidence / boundary`，再映射到总账和分轨权威来源。
4. 需要作者选择、未匿名化新证据或 provenance gate 的条目保持 `needs_author_input`，不得自行补全。
5. 仅在真实评论到达后创建唯一回复文件：`AUTHOR_RESPONSE_20260722.md`。不要恢复多路径草稿入口。
6. 发送前逐项执行总账与 triage send gate；只回答真实 reviewer 触发的 3–5 个 score-driving concerns。

## 4. 当前五个 P0/P1 边界

| 优先级 | 边界 | 必须保留的事实 |
| --- | --- | --- |
| P0 | Trust / provenance | submitted、current source、raw-backed result 与 future revision 必须分开；实验 provenance 以 `docs/overview/RESULT_PROVENANCE_MANIFEST.md` 为准 |
| P0 | KL / shape–scale correctness | ordinary baseline KL 一阶变分为零，从 `O(τ^4)` 开始；保留 exact cosh surrogate theorem、conditional diffuse probability-transport proxy、empirical finite-`τ` basin 三层身份 |
| P0 | Exact-kernel / surrogate identity | exact kernel 只测 content-independent phase redundancy；cosh 是 stated surrogate 的 optimizer，不是 exact kernel、attention 或 LM objective 的闭式最优解 |
| P1 | Baseline fairness / replication | Primary I 是 fixed-scale repo-defined progressive overlay；Primary II 的旧 “DAPE” 行实际是 32-parameter learnable-frequency control；Primary III 是 3-seed MLA，`d_eff=d_head` 只是架构 operating convention |
| P1 | Metric / capability | PK 是 teacher-forced NLL-gap。8K AR exact：Geo+YaRN 0/0/0，EVQ+YaRN 58/18/98（mean 58%）；同时保留 4K Geo+YaRN 100% 对 EVQ+YaRN 77.3% 的反向边界 |

## 5. 禁止措辞

- “EVQ is universal long-context SOTA.”
- “EVQ replaces YaRN / LongRoPE / DAPE / FIRE / learned PE.”
- “ordinary KL gives an `O(τ²)` gain”或“ordinary KL derives the deployed optimum.”
- “`τ=d_eff/√L` is globally optimal”或“MLA `d_eff=d_head` is a theorem.”
- “EVQ beats tuned Geo/YaRN”或“the matched-scale result proves tuned dominance.”
- “Primary II is fully replicated”或把 seed-42 diagnostic 升级成广义 learned-PE dominance。
- “PK means autoregressive exact retrieval.”
- 用 historical LongAlpaca EVQ 与 fresh LongAlign Geo 计算 matched EVQ effect，或写成 controlled causal comparison。
- 把 2B/4B 历史 trace、无 raw JSON 的 phase label 或旧 MLA `τ` 标签写成 reviewer-grade completion。
- 把模拟审稿问题称为 reviewer 原话，或声称真实 NeurIPS reviews 已收到。

## 6. 目录安全

- `rebuttal/` 整体不进入 supplement；只将经过匿名化、provenance 核验且被真实 review 触发的 reviewer-grade artifact 单独移入受控 tracked 位置。
- `raw_sources/00_INDEX.md` 是 tracked 控制索引；其列出的 `*_verbatim.md` payload 是 local-only，不提交、不打包、不公开引用。已移除的 reasoning attachment 不得恢复。
- `simulated_reviews/` 仅保存字节不变的内部模拟原文，不进入 supplement，也不得伪装为真实 review。
- 不复制身份、私有机器路径、凭据、内部推理或未匿名化 artifact 到 paper、public docs 或 response。
- 本轮不改论文、实验数字或数据。仅允许为退役旧目录而解耦 evidence maintenance 脚本与测试；这类维护不得改变任何科学数值或 claim tier。

## 7. Consolidation note

2026-07-12 起，`rebuttal_playbook.md` 成为 rebuttal 统一策略入口，本文件只保留导航、状态和安全边界。导航收敛到上面的分轨材料和原文索引。已被 2026-07-11 总账覆盖的旧策略、旧 Path A/B、旧 response、旧 action/runbook 与阶段性 `rebuttal_7/` 文档在本轮删除；历史仍可由 Git 追溯，但不再保留为活跃文件。两份 2026-07-10 模拟审稿已字节不变迁移到 `simulated_reviews/`；它们只提供压力测试原文，不提供真实 reviewer 身份或最新科学裁决。本次 consolidation 不改变任何实验数字。
