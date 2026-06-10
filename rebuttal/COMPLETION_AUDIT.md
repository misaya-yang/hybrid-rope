# Rebuttal Preparation Completion Audit

日期：2026-06-10

用途：对照用户原始目标，逐项审计当前 `rebuttal/` 包是否已经能支撑一次严肃 rebuttal 准备。本文不是新的 response 草稿，而是“验收矩阵”：证明哪些事情已经完成，哪些只能条件完成，哪些不得在正式 rebuttal 里伪装成完成。

当前总裁决：

- 在 **Path B**（不依赖 Geo+LoRA exact numbers）下，rebuttal 准备包已经可用于写一版基于现有论文证据、证据受限、诚实收缩的 author response。
- 在 **Path A**（把 Geo+LoRA 作为最强新控制证据）下，仍需未来补入 Base / Geo+LoRA / EVQ-LoRA 的 exact 8K/16K/32K 数字、seed scope、rank/steps/data/checkpoint 证据。
- 因此：Geo+LoRA 不是当前 Path B 的阻塞项，而是未来升级项。当前完成的是 rebuttal strategy、claim ledger、Path B response、Path A intake/runbook 和论文 trust/scope 修复。

## 0. Audited Objective

用户目标可拆成以下要求：

1. 详细阅读 `rebuttal/fable相关资料原文.md`。
2. 把收到的材料 MD 化，最好保留原文，而不是只做短摘要。
3. 为 rebuttal 制定完整计划。
4. 制定可能的补实验，但不能无限加实验。
5. 梳理论文被 reviewer 误解的地方。
6. 梳理论文真实硬伤和 limitation。
7. 逐条认清新证据里真正改变局面的内容。
8. 标出哪些 rebuttal 措辞会反噬。
9. 牢记 rebuttal 不是二次提交论文，只能回应 reviewer 的问题。
10. 形成可直接用于最终 author response 的材料。
11. 多角度研究论文实际问题，确保 response 最针对 reviewer veto。

## 1. Evidence Inventory

| Artifact | Purpose | Current evidence |
| --- | --- | --- |
| `rebuttal/README.md` | 文件夹总入口与 Path A/B 操作门 | control-room status, send gate, file map |
| `rebuttal/PATH_B_PAPER_ONLY_BRIEF.md` | 不依赖 Geo+LoRA 的 paper-only rebuttal 主线 | Path B paper/current-evidence strategy |
| `rebuttal/fable相关资料原文.md` | 用户原始合集 | 277 lines |
| local-only `rebuttal/raw_sources/00_INDEX.md` | 原文索引与校验入口 | lists sources, line counts, SHA256, usage rules |
| local-only `rebuttal/raw_sources/01_review_panel_a_verbatim.md` | 第一份 attachment 原文 | byte-for-byte `cmp=0` checked against attachment |
| local-only `rebuttal/raw_sources/02_review_panel_b_verbatim.md` | 第二份 attachment 原文 | byte-for-byte `cmp=0` checked against attachment |
| local-only `rebuttal/raw_sources/03_reasoning_attachment_verbatim.md` | 模型推理附件原文 | byte-for-byte `cmp=0`; flagged as not public material |
| local-only `rebuttal/raw_sources/04_user_five_point_note_verbatim.md` | 用户五条判断原文 | hand verbatim extracted from合集 |
| local-only `rebuttal/raw_sources/05_probability_calibration_verbatim.md` | 概率校准原文 | hand verbatim extracted from合集 |
| `rebuttal/REBUTTAL_PREPARATION.md` | 综合准备主文档 | 1385 lines; source summary, issue ledger, runbook draft, response snippets |
| `rebuttal/PAPER_ISSUE_AUDIT.md` | 论文问题审计 | paper-facing risks and fixes |
| `rebuttal/REBUTTAL_CLAIM_LEDGER.md` | claim准入账本 | ready / conditional / forbidden claims |
| `rebuttal/REBUTTAL_ACTION_BOARD.md` | 行动决策板 | response-ready, exact-data-needed, do-not-lead routing |
| `rebuttal/MINIMAL_EXPERIMENT_RUNBOOK.md` | 最小实验 runbook | P0/P1/P2, stop rules, script entrypoints |
| `rebuttal/AUTHOR_RESPONSE_PACKET.md` | 最终 response 写作包 | Path A / Path B switch, reviewer snippets |
| `rebuttal/AUTHOR_RESPONSE_PATH_B_READY_DRAFT.md` | Path B 完整草稿 | no-placeholder safe default when Geo+LoRA absent |
| `rebuttal/AUTHOR_RESPONSE_PATH_B_COMPACT.md` | Path B 紧凑提交版 | short response-budget version |
| `rebuttal/TABLE23_LORA_WORKSHEET.md` | Geo+LoRA 数据接收表 | exact-number intake template |
| `rebuttal/PRIMARY_PROVENANCE_NOTE.md` | Primary token/seed/protocol | reconciles Primary I/II/III and Phase 11B |
| `rebuttal/FIGURE_TABLE_AUDIT.md` | QuALITY 图表审计 | documents Figure 8/Table 21 mismatch and fix |
| `rebuttal/REVIEWER_RESPONSE_SKELETON.md` | R1/R2/R3/AC skeleton | reviewer-by-reviewer response template |

## 2. Requirement-by-Requirement Audit

### 2.1 Detailed Read Of `fable相关资料原文.md`

Status: **Proved complete for preparation purposes.**

Evidence:

- `REBUTTAL_PREPARATION.md` indexes the exact source ranges from `fable相关资料原文.md:1-79`, `84-158`, `161-189`, `192-206`, and `208-278`.
- `REBUTTAL_PREPARATION.md` section 9 decomposes all four source groups into reviewer stance, key questions, response implications, and action items.
- `REBUTTAL_PREPARATION.md` section 10 turns those into a master issue ledger.

Residual caveat:

- The reading is a strategic extraction, not a public quote pack. Exact raw wording is kept local-only in `raw_sources/`.

### 2.2 MD化 And Verbatim Preservation

Status: **Proved complete.**

Evidence:

- `raw_sources/01_review_panel_a_verbatim.md`, `02_review_panel_b_verbatim.md`, and `03_reasoning_attachment_verbatim.md` were checked byte-for-byte against the three attachment files.
- `raw_sources/04_user_five_point_note_verbatim.md` and `05_probability_calibration_verbatim.md` preserve the user-supplied message sections separately.
- `raw_sources/00_INDEX.md` records file purpose, line counts, hashes, and usage rules.

Important boundary:

- `03_reasoning_attachment_verbatim.md` contains identity/thought-process material and must not be copied into public rebuttal, paper, supplement, anonymous material, or pushed commits.

### 2.3 Rebuttal Plan

Status: **Proved complete as a working plan.**

Evidence:

- `REBUTTAL_PREPARATION.md` sections 2-4, 7, 12, and 15 define reviewer matrix, experiment priority, execution checklist, runbook, and decision tree.
- `REBUTTAL_ACTION_BOARD.md` condenses the plan into response-ready vs exact-data-needed vs do-not-lead buckets.
- `AUTHOR_RESPONSE_PACKET.md` gives the final response structure and decision switch.

What the plan decides:

- Lead with scope and trust repairs.
- Put R2 empirical vetoes first.
- Use Path A only if exact Geo+LoRA data exists.
- Otherwise use Path B and do not pretend the LoRA confound is closed.

### 2.4 Possible Experiments Without Infinite Expansion

Status: **Proved complete as a prioritized experiment plan; experiments themselves are not all complete.**

Evidence:

- `MINIMAL_EXPERIMENT_RUNBOOK.md` defines:
  - P0: no-GPU/result assembly work;
  - P1: eval-only/low-risk checks;
  - P2: training-class experiments;
  - do-not-run list;
  - global stop rules;
  - response upgrade matrix.
- `REBUTTAL_CLAIM_LEDGER.md` section 5 repeats the queue with stop rules.

Most important experiment decisions:

| Priority | Experiment | Current status | Why |
| --- | --- | --- | --- |
| P0 | Base / Geo+LoRA / EVQ-LoRA exact table | missing | only evidence that upgrades LoRA to Path A |
| P1 | LoRA Geo+YaRN/Dynamic NTK eval-only | missing | blocks “raw baseline too weak” shift |
| P1 | Primary I Geo+YaRN scale sweep | missing | blocks “fixed YaRN scale mistuned” attack |
| P1 | PK AR exact | missing | blocks TF PK inflation attack |
| P1 | learned tau trajectory | missing | helps R1, optional |
| P2 | MLA tau sanity | missing | helps MLA convention story, training cost/risk |

Key stop rule:

- Do not run broad LongBench/RULER fishing, new theorem work, huge 1B multi-seed reruns, or VideoRoPE fairness discourse as rebuttal work.

### 2.5 Reviewer Misunderstandings

Status: **Proved complete.**

Evidence:

- `REBUTTAL_PREPARATION.md` section 11 separates reviewer overreach/misunderstanding from real hard issues.
- `REBUTTAL_CLAIM_LEDGER.md` sections 1 and 4 route safe clarifications by reviewer.

Core misunderstandings/overextensions:

| Reviewer interpretation | Correct response |
| --- | --- |
| EVQ is claimed as universal long-context SOTA | No, it is a training-time RoPE frequency allocation mechanism/design axis |
| EVQ replaces YaRN/LongRoPE/DAPE/FIRE | No, it changes the substrate on which scaling can act |
| PK means autoregressive exact retrieval | No, PK is teacher-forced NLL-gap unless AR exact is explicitly marked |
| Primary I is tuned YaRN leaderboard | No, it is matched-scale substrate/range complementarity |
| MLA setup is production-identical DeepSeek | No, it is production-relevant scarce-channel stress test |

### 2.6 Real Hard Issues / Limitations

Status: **Proved complete.**

Evidence:

- `PAPER_ISSUE_AUDIT.md` enumerates paper-facing risks and patch status.
- `REBUTTAL_CLAIM_LEDGER.md` section 2 lists conditional claims that cannot be written without evidence.
- `AUTHOR_RESPONSE_PACKET.md` gives concession wording for Path B.

Real hard issues:

| Issue | Current stance |
| --- | --- |
| Geo+LoRA exact numbers absent | cannot close LoRA confound in final response |
| 1B MLA reversal | relabel as schedule-sensitivity limitation, not saturation robustness |
| Primary II seed scope | seed-42 diagnostic, not broad PE dominance |
| fixed YaRN scale | matched-scale claim only, not tuned-baseline dominance |
| TF PK vs AR exact | metric clarification required |
| MLA tau convention | operating convention, not theorem |
| QuALITY downstream signal | NLL diagnostic only; accuracy weak |

### 2.7 New Evidence That Truly Changes The Situation

Status: **Strategically identified; exact evidence not yet verified in workspace.**

Evidence:

- `REBUTTAL_PREPARATION.md` section 1.1 treats Geo+LoRA as the only currently game-changing new evidence.
- `REBUTTAL_ACTION_BOARD.md` section 5 names Table 23 Geo+LoRA exact numbers as the first action if only three actions are possible.
- `TABLE23_LORA_WORKSHEET.md` is prepared for exact intake.

Conclusion:

- Geo+LoRA is the only evidence that can simultaneously answer:
  - LoRA/LongAlign confound;
  - undertraining-only concern on an industrial checkpoint.
- But without exact numbers, this remains conditional and must not be used as a final claim.

### 2.8 Backfire / 反噬 Wording

Status: **Proved complete.**

Evidence:

- `REBUTTAL_PREPARATION.md` section 5 lists the main backfire blacklist.
- `REBUTTAL_CLAIM_LEDGER.md` section 3 lists forbidden sentences and safe alternatives.
- `AUTHOR_RESPONSE_PACKET.md` repeats do-not-write lines under each reviewer issue.

Highest-risk forbidden sentences:

| Forbidden | Safe replacement |
| --- | --- |
| “9B tokens is overtraining.” | Do not use overtraining; use progression/750M/provenance chain |
| “The 1B row proves robustness to training saturation.” | It is schedule-sensitivity limitation |
| “EVQ beats tuned YaRN.” | EVQ+YaRN is matched-scale substrate/range evidence |
| “PK is retrieval accuracy.” | PK is teacher-forced NLL-gap diagnostic |
| “Geo+LoRA proves EVQ scales industrially.” | Controlled LoRA is an industrial-checkpoint adaptation anchor |
| “Figure 8 was reviewer misunderstanding.” | The figure was stale/mislabeled and has been fixed |
| “tau is globally optimal.” | tau is operating default / basin selector |

### 2.9 Rebuttal Is Not A Second Submission

Status: **Proved complete in the planning docs.**

Evidence:

- `REBUTTAL_PREPARATION.md` core constraints state that rebuttal is not a second submission and must not invent results.
- `MINIMAL_EXPERIMENT_RUNBOOK.md` has global stop rules, do-not-run list, and response upgrade matrix.
- `AUTHOR_RESPONSE_PACKET.md` structures response as reviewer answers, not a new paper.

Practical consequences:

- Only exact, reviewer-targeted results can enter response.
- New broad experiments are not recommended.
- Failed or mixed new results must scope claims down rather than be hidden.

### 2.10 Author Response Material

Status: **Proved complete for Path B; conditional upgrade for Path A.**

Evidence:

- `AUTHOR_RESPONSE_PACKET.md` contains Path A/Path B switch and reviewer-specific snippets.
- `AUTHOR_RESPONSE_PATH_B_READY_DRAFT.md` is a no-placeholder longer Path B draft.
- `AUTHOR_RESPONSE_PATH_B_COMPACT.md` is a no-placeholder compact Path B draft.

Current default:

- Use Path B unless Geo+LoRA exact numbers arrive.
- For the current instruction, ignore not-yet-done Geo+LoRA and use the paper-only Path B brief.

What Path A still needs:

- Exact Base / Geo+LoRA / EVQ-LoRA PPL at 8K/16K/32K.
- Seed scope.
- Same checkpoint/data/rank/steps proof.
- Optional training-free scaler reference if fast.

### 2.11 Multi-Angle Paper Problem Research

Status: **Proved complete.**

Evidence:

- Theory angle: `REBUTTAL_CLAIM_LEDGER.md` R1 section, `AUTHOR_RESPONSE_PACKET.md` section 4.
- Empirical angle: `REBUTTAL_ACTION_BOARD.md` sections 1-6 and `MINIMAL_EXPERIMENT_RUNBOOK.md`.
- Systems angle: `REBUTTAL_CLAIM_LEDGER.md` R3 section, `AUTHOR_RESPONSE_PACKET.md` section 5.
- AC/champion angle: `REBUTTAL_ACTION_BOARD.md` section 4.4 and `REBUTTAL_PREPARATION.md` probability calibration section.
- Paper-source angle: `PAPER_ISSUE_AUDIT.md`, `FIGURE_TABLE_AUDIT.md`, `PRIMARY_PROVENANCE_NOTE.md`.

## 3. Paper/Artifact Changes Already Made

These are not just strategy docs; several paper-facing fixes have been applied.

| File | Change | Rebuttal reason |
| --- | --- | --- |
| `paper/tables/table_evidence_tier.tex` | 1B row relabeled to schedule-sensitivity; unsupported base-sweep wording removed | prevent R2 reversal attack |
| `paper/sections/05_experiments.tex` | MLA 1B wording narrowed | avoid saturation robustness overclaim |
| `paper/appendix/a4_supporting_experiments.tex` | LoRA wording scoped as post-hoc; attribution requires matched Geo+LoRA | avoid false LoRA attribution |
| `paper/appendix/a2_experiment_details.tex` | token/protocol reproducibility table added | answer training budget/provenance concern |
| `scripts/figures/fig5_downstream_qa_nll.py` | new NLL figure generator | fix QuALITY figure/table mismatch |
| `paper/figs/fig5_downstream_qa.pdf/png` | regenerated NLL figure | align Figure 8 with Table 21 |
| `paper/main.pdf` | recompiled after fixes | working PDF reflects trust/scope corrections |

Verification already performed earlier in this work:

- Tectonic compile succeeded after paper edits.
- PDF gate found 41 pages, no Type 3 fonts, and text containing the new scope/fix terms.
- `git diff --check` passed for the touched rebuttal/paper/script files.

Current audit note:

- If any paper `.tex` is edited again, recompile and rerun PDF gates before claiming final paper readiness.

## 4. Current Blocking / Conditional Items

These are not failures of preparation; they are evidence dependencies that determine which response path is allowed.

| Dependency | Needed for | Current fallback |
| --- | --- | --- |
| Base / Geo+LoRA / EVQ-LoRA exact table | Path A LoRA control | Path B concession; not required for current paper-only response |
| LoRA Geo+YaRN or Dynamic NTK eval-only | stronger answer to training-free scaler shift | matched-scale scope wording |
| Primary I tuned Geo+YaRN sweep | stronger answer to fixed YaRN scale concern | say Table 2 is matched-scale, not tuned leaderboard |
| Primary I AR exact | stronger answer to PK metric attack | define PK as teacher-forced NLL-gap |
| learned tau trajectory | stronger R1 shape/scale response | theoretical myopic-loss explanation only |
| MLA tau sanity | stronger MLA convention response | call d_eff an operating convention |

## 5. Final Response Decision Procedure

Use this procedure before sending any author response.

1. Open `TABLE23_LORA_WORKSHEET.md`.
2. If exact Geo+LoRA data are filled and traceable:
   - choose Path A in `AUTHOR_RESPONSE_PACKET.md`;
   - replace placeholders in Path A table;
   - check `REBUTTAL_CLAIM_LEDGER.md` conditional C-01/C-02.
3. If exact Geo+LoRA data are absent:
   - choose Path B;
   - use `AUTHOR_RESPONSE_PATH_B_COMPACT.md` if word budget is tight;
   - do not include controlled-LoRA upgrade language.
4. Regardless of path:
   - lead with scope and trust repairs;
   - define PK;
   - relabel 1B as schedule-sensitivity;
   - keep Primary II seed scope explicit;
   - do not claim tuned-scaler dominance.

## 6. Completion Verdict

| Requirement | Verdict |
| --- | --- |
| Read and structure fable materials | Complete |
| Preserve raw materials in MD | Complete |
| Build rebuttal plan | Complete |
| Build constrained experiment plan | Complete |
| Identify reviewer misunderstandings | Complete |
| Identify real hard issues | Complete |
| Recognize game-changing evidence | Complete strategically; exact data missing |
| Identify backfire wording | Complete |
| Enforce “not a second submission” | Complete in docs/runbook |
| Prepare final response material | Complete for Path B; conditional for Path A |
| Final send-ready rebuttal | Path B ready as paper-only strategy; Path A remains optional upgrade |

Operational conclusion:

- If the team wants to submit now with no more numbers, use Path B and `PATH_B_PAPER_ONLY_BRIEF.md`.
- If the team wants the strongest possible rebuttal, fill Geo+LoRA exact numbers first, then upgrade to Path A.
- Do not let optional P1/P2 experiments delay the already-ready trust/scope response unless the real reviewer text specifically makes them decisive.
