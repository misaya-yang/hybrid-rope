# EVQ-Cosh Rebuttal Control Room

创建：2026-06-10；最后更新：2026-07-13

- Preparation：`triage_ready`
- Decision：`unclear / high-risk trust repair`
- Response package：`needs_real_reviews + needs_author_input`；当前不可发送
- 当前模式：`triage-only`。截至 2026-07-13，实际 NeurIPS reviews 尚未收到。

本目录是作者内部 rebuttal control room，不是论文、公开补充材料或 reviewer supplement，也不得作为仓库根目录打包。唯一中心主张是：**RoPE 的有限频率表也是 finite spectral budget；EVQ-Cosh 把 training-time frequency allocation 作为 operator design 与 inference-time range scaling 之外的第三个 PE 设计轴。** 这不是 universal long-context SOTA，也不是 YaRN、LongRoPE、DAPE、FIRE 或 learned PE 的替代主张。

## 1. 统一策略入口、分轨材料与原文索引

| 文件 | 唯一职责 |
| --- | --- |
| `rebuttal_playbook.md` | **唯一操作入口**：claim disposition、P0/P1/P2、理论边界、实验 gate、作者决策门与发送 QA |
| `FULL_PAPER_INTEGRITY_AUDIT_20260713.md` | **最新事实权威**：DAPE、YaRN、midpoint-Geo、KL、`c_coll`、Phase16、协议与 provenance 的全论文审计；与旧材料冲突时以此为准 |
| `REBUTTAL_VIABILITY_AND_VENUE_PLAN_20260713.md` | **政策与决策背景**：NeurIPS 可行性、诚实披露、官方政策、公开案例与后续 venue 修复路线 |
| `frequency_adaptation_8b/` | **定向机制实验**：检验 8B checkpoint 是否能在连续改变 RoPE 频率分配时获得足够任务梯度；它是独立新协议，不替代 LongAlpaca clean pair，也不自动进入 rebuttal |
| `README.md` | 全局状态、导航与目录安全边界 |
| `REVIEWER_TRIAGE_PLAYBOOK.md` | 真实 reviews 到来后的 verbatim comment、稳定 ID 与最多五项 score-driving concern 分流 |
| `REBUTTAL_MASTER_QUESTION_LEDGER_20260711.md` | **archival risk inventory**：仅用于检索历史攻击面；不再裁决当前事实、实验优先级或发送范围 |
| `THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md` | 理论长推导：exact / conditional proxy / empirical 三层边界；方法身份、`c_coll` 与 Phase16 以 2026-07-13 full audit 为准 |
| `LORA_GEO_CONTROL_RESULT_AUDIT_20260711.md` | fresh Geo 结果、LongAlign/LongAlpaca provenance 与 clean-pair 因果边界 |
| `LORA_LONGALPACA_TEMPORAL_NLL_20260712.md` | fresh LongAlpaca seed-42 Geo+LoRA/EVQ+LoRA matched-training-pipeline temporal NLL；native-Geo与midpoint-EVQ非同quantizer，仅作 supporting evidence |
| `raw_sources/00_INDEX.md` | tracked 控制索引；指向 local-only 原文并规定模拟材料隔离规则 |

已删除的旧文档仅可从 Git 历史追溯，不是入口，不得覆盖以上控制文件或最新 provenance。

## 2. 权威来源必须分轨

| 需要判断什么 | 权威来源 | 不得误用 |
| --- | --- | --- |
| submitted/current source 写了什么 | 提交件与 `paper/` 当前源码，使用时明确版本 | 只能证明某版本的文字、图表和声明；不能覆盖数学审计，也不能单独证明实验 provenance |
| 实验数字与 provenance | `docs/overview/RESULT_PROVENANCE_MANIFEST.md` 及其指向的 raw-backed artifacts | 已移出根目录的 2026-06-14 snapshot 只保存在 ignored local snapshot 中，不得恢复为权威 |
| 全论文事实与方法身份 | `FULL_PAPER_INTEGRITY_AUDIT_20260713.md` | 当前 paper label、旧报告和 class 名不能覆盖实际 forward path 与官方定义 |
| 理论正确性 | `FULL_PAPER_INTEGRITY_AUDIT_20260713.md`；长推导见 `THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md` | submitted/current paper source 只证明“写了什么”，不裁决数学是否正确；旧 theory note 不得覆盖 2026-07-13 的 `c_coll`/Phase16 复核 |
| LoRA 因果与数据来源 | `LORA_GEO_CONTROL_RESULT_AUDIT_20260711.md`；fresh protocol见 `LORA_LONGALPACA_TEMPORAL_NLL_20260712.md` | 旧 LongAlign/LongAlpaca contrast不可作因果比较；新 LongAlpaca pair虽匹配训练流水线，但native-Geo与midpoint-EVQ非同quantizer、单seed且为teacher-forced NLL，不得称纯shape control或静默替换旧表 |
| 当前决策与可发送范围 | `rebuttal_playbook.md`；政策/venue判断见 `REBUTTAL_VIABILITY_AND_VENUE_PLAN_20260713.md`；真实评论映射用 `REVIEWER_TRIAGE_PLAYBOOK.md` | master ledger、旧草稿、旧 action board 或单次内部报告不得反向升级 claim |
| 模拟审稿原文 | `simulated_reviews/` | 只用于内部压力测试；不是真实 NeurIPS reviews，其科学判断已被 2026-07-13 full audit 与当前 playbook 取代 |

## 3. 2026-07-22 真实 review 工作流

真实 reviews 到来前只做事实、短答组件和作者决策准备，不创建或预填 author response。新实验不是默认动作；只有能直接回答真实 reviewer 的 score-changing question 且满足 `rebuttal_playbook.md` §8 gate 时才启动。

1. local-only 保存每条真实评论的逐字版本，并保留 reviewer / AC 身份标签。
2. 按 `REVIEWER_TRIAGE_PLAYBOOK.md` 分配稳定 ID；ID 一旦分配，不因排序变化而重编号。
3. 对每个 concern 填写 category、severity、action、readiness、evidence、boundary 与 author decision，再映射到分轨权威来源。
4. 区分 **review-triggered response** 与 **untriggered material-integrity disclosure candidate**。普通回复只回答真实评论；若重大方法身份/理论错误未被点名但会污染 accepted record，由作者决定是否向 AC 作一条合并 disclosure。该建议是诚信判断，不是 Handbook 明文规定的专用流程。
5. 新实验只在匹配控制、任务端点、provenance 和负结果边界完整时成为候选证据；此前保持内部研究状态，不得自行写入 response。
6. 需要作者选择、未匿名化新证据或 provenance gate 的条目保持 `needs_author_input`，不得自行补全。
7. 仅在真实评论到达后创建唯一回复文件：`AUTHOR_RESPONSE_20260722.md`。不要恢复多路径草稿入口。
8. 每份 review 最多 10,000 characters；response 不放链接、不上传 revised paper/supplement，并检查 OpenReview readers 与双盲信息。最多聚焦 3–5 个 score-driving concerns，另加一条作者批准的合并 integrity disclosure（如确有必要）。

## 4. 当前核心 P0/P1 边界

| 优先级 | 边界 | 必须保留的事实 |
| --- | --- | --- |
| P0 | Trust / provenance | submitted、current source、raw-backed result 与 future revision 必须分开；实验 provenance 以 `docs/overview/RESULT_PROVENANCE_MANIFEST.md` 为准 |
| P0 | KL / shape–scale correctness | ordinary baseline KL 一阶变分为零，从 `O(τ^4)` 开始；保留 exact cosh surrogate theorem、conditional diffuse probability-transport proxy、empirical finite-`τ` basin 三层身份 |
| P0 | Exact-kernel / surrogate identity | exact kernel 只测 content-independent phase redundancy；cosh 是 stated surrogate 的 optimizer，不是 exact kernel、attention 或 LM objective 的闭式最优解 |
| P0 | Method identity | Primary I 是 repo-defined fixed-ramp scaler，不是官方 YaRN；Primary II 的旧 “DAPE” 是 shared learnable inverse frequencies，不是 DAPE；不同 MLA scaler也不得共用 YaRN身份 |
| P0 | Geo identity | 核心 Geo 是 midpoint-discretized geometric grid，不是 native endpoint RoPE；matched shape contrast可保留，standard-RoPE dominance不可保留 |
| P0 | Collision / Phase16 | `c_coll=1.171` verification未做优化；Phase16是99 runs/9 configs/selected confirmation，不是27-config全3-seed且全部PPL差小于1% |
| P1 | Baseline fairness / replication | Primary II 是约151.9M、seed-42 headline；Primary III三 seed batch不一致，actual head_dim=64，而`tau=1.414`对应的`d_eff=128`只是ad-hoc convention |
| P1 | Metric / capability | PK 是 teacher-forced NLL-gap。按真实方法重标后，8K AR exact：Midpoint-Geo+fixed-ramp 0/0/0，EVQ+fixed-ramp 58/18/98（mean 58%）；同时保留 4K 的100%对77.3%反向边界 |

## 5. 禁止措辞

- “EVQ is universal long-context SOTA.”
- “EVQ replaces YaRN / LongRoPE / DAPE / FIRE / learned PE.”
- 把 repo fixed-ramp scaler称为 official YaRN，或把 shared learnable frequencies称为 DAPE。
- 把 midpoint-Geo称为 native/standard RoPE control。
- “ordinary KL gives an `O(τ²)` gain”或“ordinary KL derives the deployed optimum.”
- 用 `c_coll=1.171`、27 configurations或“all <1% PPL”证明公式闭环。
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
- 当前定向实验可以新增 spec、代码和内部运行产物，但不得覆盖论文数字、历史 artifacts 或 claim tier。只有真实 reviewer trigger、作者明确决定且实验通过 `rebuttal_playbook.md` §8 gate 后，结果才可进入 response 候选。

## 7. Consolidation note

2026-07-13 起，`FULL_PAPER_INTEGRITY_AUDIT_20260713.md` 是方法身份、理论与协议事实的最高入口，`rebuttal_playbook.md` 是唯一操作入口，`REBUTTAL_VIABILITY_AND_VENUE_PLAN_20260713.md` 提供政策与venue决策背景。本文件只保留导航、状态和安全边界；master ledger 已降为 archival risk inventory。两份 2026-07-10 模拟审稿只提供压力测试原文，不提供真实 reviewer 身份或最新科学裁决。本次更新不改变任何实验数字。
