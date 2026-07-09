# EVQ-Cosh Rebuttal Evidence And Action Board

日期：2026-06-10

用途：把当前 rebuttal 准备材料转成最终 response 前的行动决策板。核心问题不是“还能补什么”，而是：

1. 哪些内容现在已经 response-ready；
2. 哪些内容必须等 exact numbers 或日志；
3. 哪些实验/措辞会把 rebuttal 变成二次投稿，或者直接反噬；
4. 如果只能做少数几件事，先做哪几件。

本文依赖：

- 原文归档：local-only `rebuttal/raw_sources/00_INDEX.md`
- 总准备文档：`rebuttal/REBUTTAL_PREPARATION.md`
- 论文问题审计：`rebuttal/PAPER_ISSUE_AUDIT.md`
- 图表审计：`rebuttal/FIGURE_TABLE_AUDIT.md`
- LoRA 表模板：`rebuttal/TABLE23_LORA_WORKSHEET.md`
- 逐 reviewer 模板：`rebuttal/REVIEWER_RESPONSE_SKELETON.md`
- 逐 claim 准入账本：`rebuttal/REBUTTAL_CLAIM_LEDGER.md`
- 最终 author response 双路径写作包：`rebuttal/AUTHOR_RESPONSE_PACKET.md`
- 当前默认 Path B 无占位符草稿：`rebuttal/AUTHOR_RESPONSE_PATH_B_READY_DRAFT.md`
- 当前默认 Path B 紧凑提交版：`rebuttal/AUTHOR_RESPONSE_PATH_B_COMPACT.md`
- 不依赖 Geo+LoRA 的 paper-only Path B 主线：`rebuttal/PATH_B_PAPER_ONLY_BRIEF.md`
- 最小补实验和停止规则：`rebuttal/MINIMAL_EXPERIMENT_RUNBOOK.md`
- 完成度与条件项审计：`rebuttal/COMPLETION_AUDIT.md`

## 0. Rebuttal 守门规则

### 0.1 能写进 rebuttal 的内容

一段 response 只有在满足以下任一条件时才应该写：

- 直接回答 reviewer 的问题；
- 澄清 reviewer 对当前论文 claim/metric/scope 的误读；
- 报告一个已经完成、能给出 exact number/source 的最小补充证据；
- 承认真实 limitation，并明确会如何改写论文。

### 0.2 不能写进 rebuttal 的内容

以下内容即使听起来有帮助，也不该进正式 rebuttal：

- 没有 exact numbers 的新实验结论；
- 以计划代替结果的强 claim；
- 大规模 benchmark 计划；
- “future work” 包装成当前证据；
- 把 supporting/exploratory row 升级成 primary evidence；
- 把 reviewer 合理质疑说成误解。

### 0.3 最危险的混淆

| 混淆 | 为什么危险 | 正确处理 |
| --- | --- | --- |
| Geo+LoRA 用户新证据 vs 当前 workspace 无 exact numbers | 这是最改变局面的证据，但没有数字就不能写最终表 | 先用 `TABLE23_LORA_WORKSHEET.md` 收数；response 只写 verified table |
| Figure/Table mismatch vs reviewer misread | 已验证且已修复，不是 reviewer 误读 | 主动承认旧图 stale/mislabeled，并说明 working PDF 已改为 NLL 图 |
| 1B row vs saturation robustness | 旧 label 会自我反噬，现已改为 schedule-sensitivity check | response 仍要把 1B row 写成 limitation |
| PK diagnostic vs AR exact retrieval | teacher-forced NLL-gap 不能当 exact generation | 反复定义，AR exact 单独标 |
| matched-scale YaRN vs tuned baseline | fixed `s=8` 不能代表 best Geo+YaRN | 能 sweep 就报，不能 sweep 就 scope down |

## 1. Response-Ready Without New Experiments

这些内容现在就可以写进 rebuttal，因为证据来自当前论文源文件或已经完成的审计。注意：能写不等于要写很多，最终要按 reviewer 问题取舍。

| Item | 可写结论 | 证据 | 推荐语气 | 不能写 |
| --- | --- | --- | --- | --- |
| Scope narrowing | EVQ-Cosh 是 training-time RoPE frequency allocation 机制研究，不是 universal long-context SOTA | `REBUTTAL_PREPARATION.md` 和 AGENTS claim rule | 主动收缩 claim | EVQ replaces YaRN/LongRoPE |
| PK metric definition | PK 是 teacher-forced NLL-gap retrieval，除非明确标 AR exact | `paper/sections/05_experiments.tex:7`; `PAPER_ISSUE_AUDIT.md:I8-I9` | 澄清指标 | PK means exact retrieval |
| 1B row relabel | 1B MLA row 不能叫 robustness to training saturation；working tree 已改 | `paper/tables/table_evidence_tier.tex:20`; `PAPER_ISSUE_AUDIT.md:I3-I4` | 承认并保持 limitation | Reversal is noise |
| LoRA wording scope | 当前两行 LoRA 表还没有 Geo+LoRA exact row；正文已改为 post-hoc observation，并声明 attribution requires matched Geo+LoRA | `paper/appendix/a4_supporting_experiments.tex`; `PAPER_ISSUE_AUDIT.md:I1-I2` | 承认旧 confound；等 exact numbers 再写 control claim | Base -> EVQ-LoRA proves EVQ-specific attribution |
| QuALITY figure correction | Figure 8/Table 21 mismatch 已修复并编译验证 | `FIGURE_TABLE_AUDIT.md` | 感谢 reviewer，承认旧图 stale/mislabeled，说明已 corrected | Reviewer misread |
| Primary II seed scope | Geo/DAPE/EVQ 是 seed-42 diagnostic，不能当 broad PE dominance | `paper/sections/05_experiments.tex:41`; `paper/tables/table4_pe_dominant.tex:2` | scope down | single seed proves dominance |
| NTK non-universality | EVQ 不保证与每个 scaler 都组合变好 | `paper/tables/table5_phase11_leverage.tex` | 主动限定为 EVQ+YaRN matched-scale evidence | EVQ helps any scaler |
| MLA production scope | MLA 是 scarce-channel production-relevant stress test，不是 production-identical DeepSeek | `paper/appendix/a3_supporting_results.tex:6-10` | 诚实 scope | This is the DeepSeek config |
| Downstream scope | QuALITY accuracy near random；NLL 是 supporting probability-space check | `paper/appendix/a3_supporting_results.tex:71-87` | 不把 benchmark 当主 claim | QuALITY proves downstream win |

## 2. Response-Ready Only After Exact Data

这些是最有价值的补证据，但只有在 exact numbers、seed scope、结果文件或日志路径齐全后才能进入最终 rebuttal。

| Item | 需要什么证据 | 如果结果好，怎么写 | 如果结果弱，怎么写 | 反噬句 |
| --- | --- | --- | --- | --- |
| Base / Geo+LoRA / EVQ-LoRA | 8K/16K/32K PPL、seed scope、rank、steps、data、checkpoint | LoRA/LongAlign control isolates EVQ frequency injection | LoRA row becomes supporting/cautionary，不能当主防线 | LoRA proves industrial-scale training |
| Geo + Dynamic NTK/YaRN eval-only for LoRA | 16K/32K PPL，zero-training protocol | EVQ-LoRA not trivially replaced by default training-free scaler | If scaler close/wins, scope as competitive or supporting only | Raw Geo baseline is enough |
| Primary I Geo+YaRN scale sweep | scale set、best Geo+YaRN、EVQ+YaRN comparison | fixed-scale result not due to obviously mistuned Geo | If best Geo catches up, claim becomes matched-scale diagnostic only | EVQ beats tuned YaRN |
| Primary I AR exact | generation exact rate, trial count, length, seeds | Report AR exact separately from TF PK | If weak, preserve PK as diagnostic and avoid generation claim | TF PK equals retrieval |
| Primary I/II provenance | token budget、seq_len、seed list、result source | Done: `PRIMARY_PROVENANCE_NOTE.md` and appendix token table make Primary I/II auditable | If challenged, report only traceable Table 4 protocol | mixed 15M/128 with 100M/256 |
| Learned tau trajectory | tau logs/checkpoint metadata | negative result supports myopic in-range loss explanation | If noisy/flat, call it weak signal/flat basin | learnable tau validates closed form |
| MLA tau sanity | tau=d_rope/sqrt(L) or alternate d_eff result | strengthens systems claim | if it wins, revise MLA convention story | d_eff=d_head is a theorem |

## 3. Do Not Lead With These

这些方向不是完全没价值，但在 rebuttal 窗口优先级低，容易变成二次投稿或引入新风险。

| Direction | Why not lead | 只有什么时候做 |
| --- | --- | --- |
| LongBench/RULER 大规模补榜 | 454M capacity 容易贴地板；不直接回应核心 reviewer veto | 已经有稳定结果且 reviewer 明确要求 |
| 1B multi-seed from scratch | 成本高，结果风险高，rebuttal 窗口不现实 | 已经在跑且可快速收敛 |
| 新理论 theorem | 可能暴露更多假设，且不能补实验信任问题 | 有已经写好的严谨 proof |
| 大量 method comparison | 像二次投稿，稀释主线 | 只加一个 reviewer 指名 baseline |
| VideoRoPE 对比叙事 | 审稿文化分析有用，但正式 rebuttal 里像抱怨 | 只用于内部判断，不进 response |

## 4. Reviewer-Specific Action Routing

### 4.1 R1 Theory

| Concern | 当前状态 | 最小 response | 最小补证据 |
| --- | --- | --- | --- |
| shape/scale separation | 真实 limitation，但可 scope | shape derived under surrogate, scale is operating default | A.15 measure-then-allocate 或 learned tau trajectory |
| learnable tau negative | 可转为方法动机 | training loss cannot see OOD benefit | tau trajectory |
| NTK composition reversal | 已有证据显示非 universal | composition claim limited to EVQ+YaRN | 不需要新实验，除非 reviewer 强追 |

R1 最容易被反噬的句子：

- “We derive the globally optimal tau.”
- “The learned tau result validates EVQ.”
- “The surrogate is the exact trained attention objective.”

### 4.2 R2 Empirical

| Concern | 当前状态 | 最小 response | 最小补证据 |
| --- | --- | --- | --- |
| LoRA confound | 当前 paper 真实硬伤；用户新证据可能关闭 | old table confounded; new table only with exact numbers | Base/Geo+LoRA/EVQ-LoRA |
| undertraining | 不能用 overtraining 反驳 | progression + 750M + controlled LoRA, scoped | token/provenance table now done; Geo+LoRA exact numbers still needed |
| YaRN tuned baseline | fixed scale limitation | matched-scale comparison, not tuned leaderboard | Geo+YaRN sweep |
| TF PK vs AR exact | metric clarification needed | PK is diagnostic NLL-gap | AR exact if feasible |
| Primary II single seed | 真实 scope issue | seed-scoped diagnostic | extra seeds or provenance note |
| 1B reversal | 真实 hard issue | schedule-sensitive limitation | fixed-L continuation only if feasible |
| Figure/Table mismatch | 已验证 trust issue | correct stale/mislabeled figure | corrected plot/caption |

R2 最容易被反噬的句子：

- “9B tokens is overtraining.”
- “1B proves saturation robustness.”
- “EVQ beats tuned YaRN.”
- “PK is retrieval.”
- “Geo+LoRA proves EVQ scales industrially.”

### 4.3 R3 Systems

| Concern | 当前状态 | 最小 response | 最小补证据 |
| --- | --- | --- | --- |
| scale/practicality | production-scale未完成 | zero-parameter schedule + diagnostic utility | controlled LoRA |
| MLA relevance | production-relevant但非production-identical | sparse-channel stress test | tau sanity if possible |
| downstream weak | accuracy weak, NLL supporting | diagnostics primary; downstream non-regression/supporting | do not chase broad benchmark |

R3 最容易被反噬的句子：

- “Production ready.”
- “Industrial validation complete.”
- “Benchmarks are irrelevant.”
- “This is the DeepSeek production config.”

### 4.4 AC

AC 需要的是一眼能看懂的诚实边界：

| AC question | 最好回答 |
| --- | --- |
| Is the main claim overstated? | We narrowed it to a mechanism/design-axis claim. |
| Did authors address the biggest confound? | Yes, if Geo+LoRA exact table is present; otherwise we concede LoRA remains supporting. |
| Are metrics honest? | PK is TF NLL-gap; AR exact only when labeled. |
| Did authors hide failures? | 1B row relabeled as schedule-sensitivity limitation; Figure/Table bug acknowledged and corrected. |
| Is there still value? | Primary I matched-scale EVQ+YaRN, Primary III MLA scarce-channel, dead-channel audit, controlled LoRA if verified. |

## 5. If We Must Choose Only Three Actions

### 5.1 First: Table 23 Geo+LoRA exact numbers

Why:

- It is the only new evidence that can simultaneously answer LoRA confound and undertraining/industrial-checkpoint concern.
- It changes reviewer prior more than another small diagnostic.

Required before use:

- exact Base / Geo+LoRA / EVQ-LoRA numbers;
- same checkpoint/data/rank/steps;
- seed scope;
- whether 8K cost decomposes into adaptation cost vs EVQ incremental cost.

### 5.2 Second: Figure 8/Table 21 fix

Why:

- It is not a mechanism failure, but it damages trust across all numbers.
- Current PDF mismatch is verified.

Current status:

- Done: NLL plot regenerated via `scripts/figures/build_fig5_downstream_qa.sh` from the tracked TikZ source.
- Done: `paper/main.pdf` recompiled with Tectonic.
- Done: page 36 visually checked.

### 5.3 Third: Provenance + 1B relabel package

Why:

- R2 can weaponize missing token budgets and the current “robustness” label.
- This is mostly no-GPU work.

Current status:

- Done: Primary I token/seed row from curated JSON captured in `PRIMARY_PROVENANCE_NOTE.md`.
- Done: Primary II traceable protocol separated from Phase 11B in `PRIMARY_PROVENANCE_NOTE.md`.
- Done: appendix reproducibility table now includes token budgets.
- Done: evidence-tier row renamed from “robustness to training saturation” to schedule-sensitivity limitation.
- Done: LoRA appendix wording now scopes the two-row table as post-hoc adaptation evidence and requires matched Geo+LoRA for attribution.
- Done: unsupported text `base sweep` wording removed from the evidence-tier robustness row; base tuning remains a conditional/practitioner concern, not a ready claim.

## 6. If We Can Do Five Actions

After the top three:

4. Primary I Geo+YaRN scale sweep or LoRA Geo+Dynamic-NTK/YaRN eval-only, depending on which checkpoints are immediately available.
5. Primary I AR exact or learned tau trajectory, depending on whether R2 metric attack or R1 theory attack is stronger in the real reviews.

Decision rule:

| If real reviews emphasize... | Choose |
| --- | --- |
| “YaRN not tuned” / “raw baseline weak” | Geo+YaRN sweep |
| “LoRA vs training-free scaler” | LoRA Geo+Dynamic-NTK/YaRN eval |
| “Passkey is not retrieval” | AR exact |
| “tau is heuristic” / “learned tau negative” | learned tau trajectory |
| “MLA tau arbitrary” | MLA tau sanity |

## 7. Final Response Inclusion Gate

Before any paragraph enters final rebuttal, answer these questions:

1. Does this answer a reviewer concern?
2. Is every number traceable to a file/log/table?
3. Is this a completed result, not a plan?
4. Does it preserve the mechanism-scoped claim?
5. Does it avoid upgrading supporting rows to primary claims?
6. If the result is weak, does the paragraph honestly scope it?
7. Would a hostile reviewer be able to quote this sentence against us?

If the answer to 2 or 3 is no, the paragraph must be rewritten as scope/limitation or removed.

## 8. Current Completion State

| Requirement from active goal | Current status | Evidence |
| --- | --- | --- |
| 全部材料 MD 化 | Done locally | local-only `raw_sources/00_INDEX.md`; 5 verbatim files are not required for the pushed rebuttal strategy |
| 详细阅读并综合 | Done enough for strategy, still expandable | `REBUTTAL_PREPARATION.md`; `PAPER_ISSUE_AUDIT.md` |
| 先认真正改变局面的新证据 | Done, but exact numbers missing | Geo+LoRA elevated to P0; `TABLE23_LORA_WORKSHEET.md` |
| 制定 rebuttal 计划 | Done | `REBUTTAL_PREPARATION.md`; `REVIEWER_RESPONSE_SKELETON.md`; this board |
| 生成 evidence-scoped rebuttal draft | Done | `REBUTTAL_DRAFT_EVIDENCE_SCOPED.md` |
| 建立逐 claim 准入账本 | Done | `REBUTTAL_CLAIM_LEDGER.md` |
| 生成最终 author response 双路径包 | Done | `AUTHOR_RESPONSE_PACKET.md`; Path A uses Geo+LoRA exact numbers, Path B concedes LoRA remains supporting |
| 生成当前证据 Path B 无占位符草稿 | Done | `AUTHOR_RESPONSE_PATH_B_READY_DRAFT.md`; safe default if Geo+LoRA exact numbers remain unavailable |
| 生成当前证据 Path B 紧凑提交版 | Done | `AUTHOR_RESPONSE_PATH_B_COMPACT.md`; shorter default for response budget pressure |
| 可能实验 | Done as prioritized queue | Sections 2, 5, 6 |
| 论文误解 vs 真实问题 | Done | `PAPER_ISSUE_AUDIT.md`; Section 4 |
| 反噬措辞 | Done | `REBUTTAL_PREPARATION.md`; `REVIEWER_RESPONSE_SKELETON.md`; Section 4 |
| 最终可发送 rebuttal | Path B ready as paper-only strategy | `PATH_B_PAPER_ONLY_BRIEF.md` and `AUTHOR_RESPONSE_PATH_B_COMPACT.md`; Path A remains an optional future upgrade if Geo+LoRA exact numbers arrive |

Therefore the current prep is strong enough to write a draft, but not enough to mark the whole rebuttal work complete.
