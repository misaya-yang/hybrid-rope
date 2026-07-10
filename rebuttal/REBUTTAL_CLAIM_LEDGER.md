# EVQ-Cosh Rebuttal Claim Ledger

日期：2026-06-10

用途：把 `fable相关资料原文.md`、三份模拟评审、用户五点新证据和当前论文源码，压成最终 author response 的准入账本。它不是新 rebuttal 草稿，也不是二次投稿计划；它只回答：

1. 哪些 claim 现在可以写；
2. 哪些 claim 只有拿到 exact numbers 后才能写；
3. 哪些措辞会直接反噬；
4. reviewer 问题对应的最小回应和最小实验是什么。

## 0. 总裁决

当前 rebuttal 可以进入 **Path B paper-only author response** 阶段。若未来补入 Geo+LoRA exact table，可升级到 Path A；但 Geo+LoRA 不是当前 paper-only rebuttal 的前置条件。

原因很简单：

- 已经完成的高价值修复：原文 MD 化、Figure 8/Table 21 修复、Primary token/provenance 补报、1B MLA relabel、LoRA 两行表降级为 post-hoc observation、删除 unsupported text base-sweep wording。
- 仍缺 Path A 最改变局面的 exact evidence：Base / Geo+LoRA / EVQ-LoRA 的 8K/16K/32K 数字、seed scope、rank/steps/data/checkpoint。
- 如果没有 Geo+LoRA exact table，LoRA 只能作为 supporting/cautionary row，不能承担“工业 checkpoint 上排除欠训练和 LoRA 混杂”的主防线。
- 依据当前用户指示，正式策略应先基于现有论文证据和已完成修复走 Path B。

## 1. Response-Ready Claims

这些 claim 已经能写进 rebuttal，但仍要短、准、只回应 reviewer 问题。

| ID | 可写 claim | 当前证据 | 推荐写法 | 禁止延展 |
| --- | --- | --- | --- | --- |
| R-01 | 论文主 claim 是机制/设计轴，不是 universal long-context SOTA | `paper/sections/05_experiments.tex:7`; `paper/sections/06_limitations.tex:6` | EVQ-Cosh studies training-time RoPE frequency allocation as a finite-spectral-budget design axis complementary to inference-time scaling. | EVQ is SOTA / production recipe / YaRN replacement |
| R-02 | PK 是 teacher-forced NLL-gap diagnostic，不是默认 AR exact | `paper/sections/05_experiments.tex:7`; `paper/tables/table6_750m_continue_supporting.tex:15-17` | PK denotes teacher-forced NLL-gap retrieval unless explicitly marked AR exact; AR exact is reported separately where available. | PK means exact retrieval |
| R-03 | Primary I 是 matched-scale YaRN substrate test，不是 tuned YaRN leaderboard | `paper/sections/05_experiments.tex:22`; `paper/tables/table2_evq_yarn_main.tex:1-14` | Under the same fixed YaRN scale, EVQ gives YaRN higher leverage than Geo. | EVQ beats best tuned Geo+YaRN |
| R-04 | NTK-aware reversal is real and limits universality | `paper/tables/table5_phase11_leverage.tex:1-17` | Composition evidence is method-specific; the strongest claim is EVQ+YaRN under matched scale. | EVQ composes with any inference-time scaler |
| R-05 | Primary II is seed-scoped diagnostic | `paper/sections/05_experiments.tex:41`; `paper/tables/table_evidence_tier.tex:12-14` | The DAPE-style 128->8K result is a seed-42 PE-dominant diagnostic; learnable-PE row is 3-seed. | Broad dominance over DAPE or learned PE |
| R-06 | Token/provenance visibility has been improved | `paper/appendix/a2_experiment_details.tex:20-31`; `rebuttal/PRIMARY_PROVENANCE_NOTE.md` | We added a compact reproducibility snapshot and separated Table 4's 128-token/15M protocol from Phase 11B's 256-token/100M protocol. | Mixing 15M/128 and 100M/256 as one protocol |
| R-07 | 1B MLA row is schedule-sensitivity limitation | `paper/tables/table_evidence_tier.tex:18-20`; `paper/appendix/a4_supporting_experiments.tex:29-30` | The 1B row is a single-seed schedule-sensitivity stress check, not saturation robustness. | The 1B row proves robustness to training saturation |
| R-08 | MLA uses an empirical operating convention, not a theorem from d_rope | `paper/sections/05_experiments.tex:52`; `paper/appendix/a3_supporting_results.tex:6-10` | d_rot determines channel count; the deployed d_eff is an architecture-specific operating convention and direct tau ablations are natural checks. | d_eff=d_head is theoretically derived for MLA |
| R-09 | QuALITY should be NLL/probability-space supporting evidence, not downstream accuracy win | `data/curated/quality_454m_full_eval.json`; `rebuttal/FIGURE_TABLE_AUDIT.md` | Accuracy is near random at 454M; the n=2086 Gold-answer NLL aggregate is the rebuttal source of truth. | QuALITY proves downstream task improvement |
| R-10 | LoRA current table is scoped correctly as post-hoc observation | `paper/appendix/a4_supporting_experiments.tex:32-51` | The current two-row table cannot attribute the entire gain to EVQ; matched Geo+LoRA is required for causal attribution. | Base->EVQ-LoRA proves EVQ-specific gain |
| R-11 | 750M continuation helps rebut “TF metric only” but remains single-seed supporting evidence | `paper/tables/table6_750m_continue_supporting.tex:1-19` | The 750M row shows TF can saturate while AR exact separates; it motivates honest metric separation. | 750M single seed proves full scaling durability |
| R-12 | Figure/Table trust issue is acknowledged and quantified | `rebuttal/FIGURE_TABLE_AUDIT.md` | Submitted Figure 8/9 are inconsistent; use aggregate/table sources and commit to revision. | Reviewer misread the figure; current PDF is already fixed |

## 2. Conditional Claims

这些 claim 只有拿到具体结果后才能写。没有 exact numbers 时，必须改成 limitation 或删除。

| ID | 条件 claim | 必须先有的证据 | 如果证据强，怎么写 | 如果证据弱/缺失，怎么写 |
| --- | --- | --- | --- | --- |
| C-01 | Geo+LoRA control closes LoRA confound | Base / Geo+LoRA / EVQ-LoRA PPL@8K/16K/32K; seed scope; same checkpoint/data/rank/steps | Geo+LoRA isolates LongAlign/LoRA adaptation; only Geo+LoRA->EVQ-LoRA difference is attributed to EVQ frequency injection. | The original LoRA row remains supporting only and cannot isolate EVQ from LoRA adaptation. |
| C-02 | LoRA also rebuts undertraining-only story | Same C-01 evidence plus pretrained checkpoint identity | The effect appears on a heavily pretrained checkpoint under a matched adaptation control. | Do not use LoRA for undertraining defense. Use progression/750M/token provenance only. |
| C-03 | EVQ-LoRA beats default training-free Llama extension | Geo + Dynamic NTK or Geo + YaRN eval-only at 16K/32K on same checkpoint | EVQ-LoRA is not trivially replaced by zero-training scaling. | If zero-training scaler is close/better, scope LoRA as competitive or cautionary. |
| C-04 | Primary I survives tuned Geo+YaRN scale sweep | Geo+YaRN sweep over reviewer-relevant scales, same checkpoint/eval | Fixed-scale result is not due to an obvious mistuned Geo baseline. | Claim remains matched-scale diagnostic only. |
| C-05 | Primary I AR exact supports generation retrieval | AR exact rate, trial count, length, seeds, evaluator protocol | Report AR exact separately and strengthen metric story. | Keep PK as diagnostic NLL-gap; do not imply generation success. |
| C-06 | Learned tau trajectory supports myopic-loss explanation | Learned tau logs/checkpoints showing drift toward small tau or instability | Training loss cannot see extrapolation benefit, explaining why learned tau underperforms closed form. | If logs are flat/noisy, call it a weak signal or omit. |
| C-07 | MLA tau convention is safe | tau=d_rope/sqrt(L), tau=d_head/sqrt(L), deployed tau comparisons | Deployed d_eff convention is empirically validated against natural alternatives. | If alternate tau wins, revise MLA story and demote convention. |
| C-08 | Base tuning is not an easy replacement | Geo best-b vs EVQ@b=500K or comparable text base sweep | EVQ is not only exploiting a poor base choice. | If not run, concede fixed-base scope and do not answer practitioner base-tuning fully. |
| C-09 | Primary II multi-seed stabilizes DAPE contrast | Extra Geo/DAPE/EVQ seeds under the same 128-token protocol | Report mean/std and remove seed-42 fragility. | Keep as seed-scoped diagnostic and do not lean on it for final decision. |
| C-10 | Measure-then-allocate closes shape/scale bridge | A.15 L_eff^J and empirical D(Delta) measurement from existing checkpoint | Shape and scale can be calibrated from the same measured function. | Keep theory as shape variational + scale calibrated, not unified theorem. |

## 3. Forbidden / Backfire Sentences

这些句子不应进入正式 rebuttal。部分可以出现在内部文档里作为“反噬黑名单”，但不能原样出现在 author response。

| 禁句 | 为什么反噬 | 安全替代 |
| --- | --- | --- |
| 9B tokens is overtraining. | Chinchilla is compute-optimal, not an overtraining threshold; industrial models often exceed it. | Chinchilla-style budgets are not overtraining thresholds; our narrower evidence argues against the simplest undertraining-only explanation. |
| The 1B row proves robustness to training saturation. | The raw EVQ advantage reverses and the row is single-seed/schedule-confounded. | The 1B row is a schedule-sensitivity limitation. |
| EVQ beats tuned YaRN. | Current Primary I is matched fixed scale, not a tuned sweep. | EVQ gives YaRN higher leverage under the matched scale we tested. |
| PK is retrieval accuracy. | PK is teacher-forced NLL-gap unless AR exact is explicitly marked. | PK is a diagnostic; AR exact is separately reported where available. |
| Geo+LoRA proves EVQ scales industrially. | Even a clean Geo+LoRA table is post-hoc adaptation, not from-scratch industrial pretraining. | The controlled LoRA row is an industrial-checkpoint adaptation anchor. |
| EVQ-LoRA solves long-context LLaMA. | RULER does not improve and 8K PPL cost is nontrivial. | EVQ-LoRA is a post-hoc frequency-adaptation observation with explicit cost. |
| +30% cost is modest. | A reviewer can quote it as dismissing a real cost. | Report +30% explicitly and decompose Base->Geo+LoRA vs Geo+LoRA->EVQ-LoRA if the control table exists. |
| tau=d_eff/sqrt(L) is globally optimal. | AGENTS and paper scope say operating default / basin selector only. | tau is an operating default supported by scaling structure and empirical basin evidence. |
| EVQ replaces YaRN/LongRoPE/DAPE/FIRE/learned PE. | It contradicts the paper identity. | EVQ changes the training-time frequency substrate and is complementary to those methods. |
| Figure 8 was a reviewer misunderstanding. | The mismatch was real. | The submitted figure was stale/mislabeled; it has been replaced with a Gold-NLL figure. |
| Downstream benchmarks prove EVQ. | QuALITY accuracy is near random and RULER does not improve. | Diagnostics show PE-layer/probability-space effects; downstream accuracy is capacity-limited at this scale. |
| MLA is production-identical DeepSeek. | Current setup differs in d_rope and base. | MLA is a production-relevant scarce-rotary-channel stress test. |

## 4. Reviewer-Issue Routing

### 4.1 R1 Theory

| Concern | Truth status | Current response | Optional minimal evidence |
| --- | --- | --- | --- |
| Shape/scale are two derivations | True limitation | Admit two layers: cosh shape from surrogate, tau as calibrated operating default. | A.15 measure-then-allocate if logs/checkpoints are available. |
| Learned tau worse than fixed tau | True, can become positive | Training objective is myopic within L_train and cannot see extrapolation benefit. | Learned tau trajectory. |
| Bessel/cosh shape concern | Partly true | Cosh is exact for the stated broadband surrogate; functional validation tests the deployed direction. | Offline Bessel-shape collision/PPL ablation only if cheap. |
| Undefined internal terms | Possible paper polish issue | Remove or define in revision; not central to response unless reviewer cites it. | Text cleanup only. |
| MLA d_eff convention | True limitation | Treat as empirical operating convention. | MLA tau sanity ablation. |

### 4.2 R2 Empirical

| Concern | Truth status | Current response | Required/optional evidence |
| --- | --- | --- | --- |
| LoRA confound | True hard issue | Current table scoped; final defense requires Geo+LoRA exact table. | P0: Base/Geo+LoRA/EVQ-LoRA exact values. |
| Undertraining / 1B reversal | True hard issue | Do not say overtraining. Use token provenance, progression, 750M, and controlled LoRA only if verified. | Fixed-L continuation is strongest but likely expensive. |
| YaRN fixed scale | Valid limitation | Matched-scale substrate test, not tuned baseline dominance. | Geo+YaRN scale sweep. |
| TF PK not AR exact | Valid limitation | Define PK and separate AR exact. | Primary I AR exact if feasible. |
| Primary II single seed | Valid scope issue | Seed-42 diagnostic, not broad dominance. | Extra seeds if cheap and same protocol. |
| Figure 8/Table 21 mismatch | Real trust issue; rebuttal response ready | Acknowledge stale/mislabeled figure, 26.6%→24.6% erratum, and n=2086 source of truth. | PDF correction is deferred to revision. |
| MLA tau=d_rope missing | Valid limitation | Empirical d_eff convention, not theorem. | tau=d_rope/sqrt(L) sanity check. |
| Base-tuning baseline | Valid practitioner concern | Do not claim best practical schedule. | Text Geo best-b comparison if available. |

### 4.3 R3 Systems / Practicality

| Concern | Truth status | Current response | Best positioning |
| --- | --- | --- | --- |
| Production scale missing | True | Present as mechanism study and production-relevant stress tests, not deployment recipe. | Zero-parameter schedule + MLA scarce-channel + dead-channel audit. |
| LoRA retrofit cost | True | Report +30%; do not call modest; control needed for decomposition. | If Geo+LoRA exists, separate adaptation cost from EVQ incremental cost. |
| Downstream utility weak | True | Accuracy floor at 454M; NLL/PPL/PK are diagnostics. | Do not chase broad benchmark unless reviewer explicitly demands it and data are ready. |
| Base and modality calibration | True | Acknowledge operating defaults and modality/base sensitivity. | Avoid universal deployment language. |

### 4.4 AC

AC wants three things: trust, scope, and decision-changing controls.

The response should open with corrections/provenance, not with a new theoretical sales pitch:

1. Corrected Figure 8/Table 21 mismatch.
2. Added primary token/seed provenance.
3. Relabeled 1B row as schedule sensitivity.
4. Clarified PK/AR exact.
5. If and only if exact numbers are available: added Geo+LoRA control.

## 5. Minimal Experiment Queue With Stop Rules

This queue is intentionally narrow. Anything outside it risks turning rebuttal into a second submission.

| Priority | Action | Why it matters | Stop rule |
| --- | --- | --- | --- |
| P0 | Assemble Geo+LoRA exact table | Only new evidence that can close both LoRA confound and pretrained-checkpoint undertraining attack | Stop if numbers/seeds/protocol are not traceable; use concession paragraph instead. |
| P0 | Keep Figure 8/Table 21 correction in final PDF | Trust repair | Done unless later edits break the figure. |
| P0 | Keep token/provenance table | Blocks undertraining/provenance attack | Done unless protocol uncertainty is discovered. |
| P1 | Geo+YaRN scale sweep for Primary I | Stops “mistuned Geo” attack | If best Geo catches up, scope claim to matched-scale only. |
| P1 | Primary I AR exact | Stops TF PK inflation attack | If weak, report it honestly and preserve PK as diagnostic. |
| P1 | Geo + Dynamic NTK/YaRN eval-only for LoRA checkpoint | Stops “training-free scaler baseline” shift | If scaler wins, demote LoRA. |
| P1 | Learned tau trajectory | Helps R1 theory response | If logs are missing/noisy, omit rather than speculate. |
| P2 | MLA tau=d_rope/sqrt(L) sanity | Protects strongest systems result | If alternate tau wins, revise MLA convention story. |
| P2 | Primary II extra seeds | Reduces seed-42 fragility | Must use same 128-token protocol; do not mix with Phase 11B. |
| P2 | Fixed-L continuation gap-vs-tokens | Cleanest answer to 1B reversal | Only if checkpoints/compute already exist; otherwise too large for rebuttal. |

Do not lead with:

- LongBench/RULER fishing at 454M;
- broad new comparisons;
- new theorem work;
- multi-seed 1B from scratch unless already finished;
- VideoRoPE fairness discourse in formal response.

## 6. Final Response Assembly Order

Use this order unless the real reviewer text forces a different order:

1. **Opening scope**: mechanism/design-axis, not universal SOTA.
2. **Trust repairs**: Figure 8 NLL correction, token/seed provenance, PK definition.
3. **R2 empirical vetoes**:
   - LoRA confound: exact Geo+LoRA table if available, otherwise concession.
   - 1B reversal: schedule-sensitivity limitation.
   - YaRN scale: matched-scale scope or sweep.
   - AR exact: report if available, otherwise metric scope.
4. **R1 theory**:
   - shape derived under surrogate;
   - scale as operating default;
   - learnable tau/myopic loss explanation if evidence exists.
5. **R3 systems**:
   - zero-parameter schedule;
   - MLA scarce-channel relevance;
   - dead-channel audit;
   - LoRA only if controlled.
6. **Closing**: acknowledge remaining production-scale validation as future work; do not promise unlimited new experiments.

## 7. Send / Do-Not-Send Gate

Before final author response, every paragraph must pass:

1. Does it answer a reviewer concern?
2. Is every number traceable to a current table/log/file?
3. Is it a completed result, not a plan?
4. Does it preserve the mechanism-scoped claim?
5. Does it avoid upgrading supporting rows to primary evidence?
6. If it mentions LoRA, does it distinguish Base->Geo+LoRA from Geo+LoRA->EVQ-LoRA?
7. If it mentions 1B, does it say schedule-sensitivity limitation rather than saturation robustness?
8. If it mentions PK, does it say teacher-forced NLL-gap unless AR exact is actually reported?
9. If it mentions YaRN, does it say matched-scale unless a tuned sweep is actually available?
10. Would a hostile reviewer be able to quote the sentence against us?

If any answer fails, rewrite or delete the paragraph.

## 8. Current Goal Completion Audit

| Objective requirement | Current evidence | Status |
| --- | --- | --- |
| 详细阅读 `fable相关资料原文.md` | Source is 277 lines; issues from all sections are represented in this ledger and `PAPER_ISSUE_AUDIT.md` | Satisfied for strategy drafting |
| 全部材料 MD 化 | `rebuttal/raw_sources/00_INDEX.md` survives, but the indexed verbatim files are unavailable and prior byte checks are not reproducible here | Not independently satisfied |
| 认下真正改变局面的新证据 | Geo+LoRA is P0 in this ledger and `REBUTTAL_PREPARATION.md` | Satisfied conceptually; exact numbers missing |
| 制定 rebuttal 计划 | `REBUTTAL_PREPARATION.md`, `REBUTTAL_ACTION_BOARD.md`, `REBUTTAL_DRAFT_EVIDENCE_SCOPED.md`, this ledger | Satisfied |
| 制定可能实验 | Section 5 above plus existing action board | Satisfied as prioritized queue |
| 识别论文误解 | Sections 1 and 4 above; `PAPER_ISSUE_AUDIT.md` | Satisfied |
| 标出反噬措辞 | Section 3 above | Satisfied |
| 不把 rebuttal 变二次投稿 | Section 5 stop rules and Section 7 gate | Satisfied as process guard |
| 最终可发送 rebuttal | Path B uses explicit concession path; Path A needs exact Geo+LoRA numbers | Path B satisfied as paper-only strategy |

Conclusion: preparation is strong enough to draft Path B from current paper evidence. Geo+LoRA exact numbers remain a Path A upgrade, not a blocker for the paper-only rebuttal.
