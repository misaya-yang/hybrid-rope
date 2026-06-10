# EVQ-Cosh Rebuttal Control Room

日期：2026-06-10

这不是论文补充材料，也不是第二版投稿。这个文件夹是 rebuttal 作战室：保存原文、证据状态、最小实验队列、危险措辞、Path A/Path B author response。

## 0. Current Decision

当前默认走 **Path B**：基于已经完成的论文修复和现有 primary evidence 写 rebuttal，不等待尚未完成的 Geo+LoRA。

原因：当前 workspace 没有可核验的 Base / Geo+LoRA / EVQ-LoRA exact table。没有这张表，就不能把 LoRA 写成“关闭 LoRA confound 和欠训练质疑”的强控制证据。

| Path | 何时使用 | 用哪个草稿 | LoRA 怎么写 |
| --- | --- | --- | --- |
| Path A | exact Base / Geo+LoRA / EVQ-LoRA 8K/16K/32K 数字、seed scope、same checkpoint/data/rank/steps 都齐全 | `AUTHOR_RESPONSE_PACKET.md` 的 Path A 段落 | matched control; only Geo+LoRA -> EVQ-LoRA gap attributed to EVQ |
| Path B | 当前状态，或 Geo+LoRA 数字不可追溯 | `AUTHOR_RESPONSE_PATH_B_COMPACT.md` 或 `AUTHOR_RESPONSE_PATH_B_READY_DRAFT.md` | supporting/post-hoc only; concede attribution requires Geo+LoRA |

快速规则：

- 没有 exact Geo+LoRA table：不要写 Path A，但这不是 Path B 的阻塞项。
- 有 exact Geo+LoRA table：先填 `TABLE23_LORA_WORKSHEET.md`，再升级 `AUTHOR_RESPONSE_PACKET.md`。
- 不要让 optional P1/P2 实验拖住已经完成的 trust/scope 修复。

## 1. Start Here

| 你要做什么 | 打开 |
| --- | --- |
| 查原文是否逐字保存 | local-only `raw_sources/00_INDEX.md` |
| 看完整策略和逐条分析 | `REBUTTAL_PREPARATION.md` |
| 看当前是否满足原始目标 | `COMPLETION_AUDIT.md` |
| 看每句话能不能写 | `REBUTTAL_CLAIM_LEDGER.md` |
| 看下一步行动取舍 | `REBUTTAL_ACTION_BOARD.md` |
| 看最小实验怎么跑、怎么停 | `MINIMAL_EXPERIMENT_RUNBOOK.md` |
| 不依赖 Geo+LoRA、只基于当前论文怎么写 | `PATH_B_PAPER_ONLY_BRIEF.md` |
| 写最终 response | `AUTHOR_RESPONSE_PACKET.md` |
| 没有 Geo+LoRA 数字时直接用 | `AUTHOR_RESPONSE_PATH_B_COMPACT.md` |

## 2. What Is Already Done

| Done item | Evidence |
| --- | --- |
| 原文 MD 化 | local-only `raw_sources/*.md`; three attachment files checked byte-for-byte with `cmp=0` |
| 全面 rebuttal plan | `REBUTTAL_PREPARATION.md` |
| claim 准入账本 | `REBUTTAL_CLAIM_LEDGER.md` |
| 最小实验 runbook | `MINIMAL_EXPERIMENT_RUNBOOK.md` |
| Path B author response | `AUTHOR_RESPONSE_PATH_B_COMPACT.md`, `AUTHOR_RESPONSE_PATH_B_READY_DRAFT.md` |
| Paper-only Path B strategy | `PATH_B_PAPER_ONLY_BRIEF.md` |
| Figure 8/Table 21 trust fix | `FIGURE_TABLE_AUDIT.md`; regenerated NLL figure |
| Primary token/protocol reconciliation | `PRIMARY_PROVENANCE_NOTE.md`; paper appendix token table |
| 1B MLA relabel | paper wording changed to schedule-sensitivity |
| LoRA overclaim prevention | appendix wording changed to post-hoc/supporting unless matched Geo+LoRA exists |

## 3. What Is Still Conditional

| Missing or optional evidence | Needed for | Current fallback |
| --- | --- | --- |
| Base / Geo+LoRA / EVQ-LoRA exact table | Path A LoRA control | Path B concession; not required for paper-only response |
| LoRA Geo+YaRN or Dynamic NTK eval-only | answer “raw baseline too weak” | do not claim training-free scaler dominance |
| Primary I Geo+YaRN scale sweep | answer fixed-scale/tuned-baseline attack | call Table 2 matched-scale diagnostic |
| Primary I AR exact | answer PK metric attack | define PK as teacher-forced NLL-gap |
| learned tau trajectory | strengthen R1 | use myopic-loss explanation cautiously |
| MLA tau sanity | strengthen systems/MLA convention | call d_eff an operating convention |

## 4. The Rebuttal Posture

Defend:

- EVQ-Cosh as a training-time RoPE frequency-allocation mechanism.
- finite spectral budget / active-band framing.
- EVQ+YaRN matched-scale substrate/range complementarity.
- MLA scarce-channel stress test as production-relevant, not production-identical.
- dead-channel audit and diagnostic value.

Concede or scope:

- not universal long-context SOTA;
- not a YaRN/LongRoPE/DAPE/FIRE replacement;
- Primary II is seed-scoped diagnostic;
- PK is teacher-forced NLL-gap unless AR exact is explicitly marked;
- 1B MLA row is schedule-sensitivity limitation;
- LoRA is supporting unless matched Geo+LoRA numbers are filled.

## 5. Do Not Write

These phrases are rebuttal traps:

- “9B tokens is overtraining.”
- “The 1B row proves robustness to training saturation.”
- “EVQ beats tuned YaRN.”
- “PK means exact retrieval.”
- “Geo+LoRA proves EVQ scales industrially.”
- “EVQ-LoRA solves long-context LLaMA.”
- “+30% cost is modest.”
- “tau is globally optimal.”
- “Figure 8 was a reviewer misunderstanding.”
- “MLA is production-identical DeepSeek.”

Use `REBUTTAL_CLAIM_LEDGER.md` for safe alternatives.

## 6. If Geo+LoRA Numbers Arrive Later

Do this in order:

1. Fill `TABLE23_LORA_WORKSHEET.md`.
2. Verify same checkpoint, data, rank, steps, seed scope.
3. Compute Base -> Geo+LoRA adaptation cost.
4. Compute Geo+LoRA -> EVQ-LoRA EVQ-incremental difference.
5. Update `AUTHOR_RESPONSE_PACKET.md` Path A table.
6. Update `REBUTTAL_CLAIM_LEDGER.md` conditional C-01/C-02 status.
7. Recheck no Path B concession language remains mixed into the Path A paragraph.

If numbers are weak or mixed:

- do not force Path A;
- keep LoRA as supporting/cautionary;
- report the result only if it directly answers a reviewer question.

## 7. Minimal Response Order

Use this order for final response:

1. Thank reviewers and state narrow scope.
2. List trust repairs: Figure NLL fix, token/seed provenance, PK definition, 1B relabel.
3. Address R2 first:
   - LoRA Path A or Path B;
   - training budget without “overtraining”;
   - YaRN/tuned scaler scope;
   - PK vs AR exact;
   - Primary II seed scope;
   - 1B schedule sensitivity.
4. Address R1:
   - shape vs scale;
   - learned tau;
   - NTK-aware limitation.
5. Address R3:
   - zero-parameter schedule;
   - MLA scarce-channel relevance;
   - downstream benchmark scope.
6. Close with concrete paper edits, not new grand claims.

## 8. Send Gate

Before sending:

- [ ] Choose Path A or Path B explicitly.
- [ ] If Path A, every LoRA number is traceable in `TABLE23_LORA_WORKSHEET.md`.
- [ ] If Path B, remove all controlled-LoRA upgrade language.
- [ ] PK is defined as teacher-forced NLL-gap.
- [ ] 1B row is called schedule-sensitivity, not robustness.
- [ ] Primary II seed scope is explicit.
- [ ] Figure/Table correction is acknowledged as our stale/mislabeled figure.
- [ ] No forbidden sentence from Section 5 appears.
- [ ] Final response answers reviewer questions, not a new paper.
