# EVQ-Cosh Rebuttal Response Skeleton

日期：2026-06-10

用途：把 `raw_sources/`、`REBUTTAL_PREPARATION.md`、`PAPER_ISSUE_AUDIT.md`、`FIGURE_TABLE_AUDIT.md` 和 `REBUTTAL_ACTION_BOARD.md` 转成逐 reviewer 可执行的 response skeleton。

这不是最终 rebuttal 文案。最终写作前必须先填入真实数字、seed scope、token budget、以及哪些补充实验已经完成。本文所有带 `[PLACEHOLDER]` 的内容都不能直接发送。

## 0. 使用原则

### 0.1 Rebuttal 的边界

Rebuttal 不是二次投稿。每个 response paragraph 必须满足至少一个条件：

- 回答 reviewer 明确问到的问题；
- 修正 reviewer 基于论文文本产生的合理误解；
- 报告一个最小补充实验，用来关闭一个具体攻击面；
- 承认一个真实 limitation，并说明论文会如何 scope down。

不要为了显得更强而堆新实验。新实验只有在关闭以下 veto 点时才值得写：

- LoRA confound；
- tuned/training-free scaler baseline；
- teacher-forced PK vs autoregressive exact；
- Primary seed/token provenance；
- 1B schedule sensitivity；
- MLA tau convention；
- Figure/Table trust issue。

### 0.2 全局 opening stance

推荐开头：

> We thank the reviewers for the detailed comments. We agree that several rows should be read as mechanism evidence rather than production-scale validation. Our central claim is narrower: EVQ-Cosh changes the training-time RoPE frequency substrate as a finite-spectral-budget allocation, and inference-time range scaling can act differently on that substrate. We have clarified this scope, added controls for the main confounds where possible, and will relabel supporting rows that were too broadly described.

不要开头就说：

- “EVQ is SOTA for long context.”
- “EVQ replaces YaRN/LongRoPE.”
- “The reviewers misunderstood our contribution.”
- “All concerns are addressed by new experiments.”

### 0.3 证据状态词

| 状态 | response 里能怎么写 | 不能怎么写 |
| --- | --- | --- |
| 已核实论文内证据 | “The paper already defines/reports...” | “This fully proves...” |
| 用户称已有新结果但 workspace 无数字 | “We can report this once exact numbers are inserted” | 直接写具体数值或 claim |
| eval-only 可补 | “We will add / have added, if completed” | 把 plan 写成已完成 |
| 真实硬伤 | “We agree and will relabel/correct” | “This is only a presentation issue” |
| reviewer 误解 | “We clarify that...” | “The reviewer is wrong” |

## 1. Global Change Log For Response

最终 rebuttal 应该先给一个短 change log，降低 AC 的阅读成本。

可用结构：

1. Scope clarified: mechanism/design-axis claim, not universal SOTA.
2. LoRA confound addressed: Base / Geo+LoRA / EVQ-LoRA table, if exact numbers are available.
3. Metric clarified: PK is teacher-forced NLL-gap; AR exact is only named when measured.
4. Supporting-row relabel: 1B MLA row is schedule-sensitivity limitation, not robustness to training saturation.
5. Figure/Table correction: QuALITY Figure 8/Table 21 stale/mislabeled figure has been corrected in the working PDF.
6. Provenance added: total tokens and seed scope for primary rows, using `PRIMARY_PROVENANCE_NOTE.md` and only where traceable.

必须避免：

- 把 supporting rows 写成 new main results；
- 在没有 exact numbers 前声称 Geo+LoRA “proves”；
- 用 “overtraining” 描述 9B tokens 或 Chinchilla token counts；
- 把 QuALITY accuracy 当成下游性能主证据。

## 2. R1: Theory / Positional Encoding Reviewer

### 2.1 Likely Concern: Shape and scale are not jointly derived

Reviewer concern:

> The cosh shape may be derived under a surrogate, but the scale `tau=d_eff/sqrt(L)` appears heuristic or empirically chosen.

Current evidence:

- Theory is scoped as a surrogate: `paper/sections/03_theory.tex:15`.
- Tau is already described as an operating-point/basin selector: `paper/sections/03_theory.tex:93-115`.
- AGENTS rule: do not describe `tau=d_eff/sqrt(L)` as globally optimal.

Response posture:

- Defend the shape derivation.
- Concede that tau is not a global optimum theorem.
- Reframe tau as operating default / basin selector with empirical support.

Draft:

> We agree that the paper should separate the two levels more explicitly. The cosh density is derived for the stated broadband surrogate, while `tau` is used as an operating-point selector rather than a theorem of global optimality for trained attention. We will revise the text to avoid implying a single unified optimum and to state that the contribution is the shape-plus-calibrated-scale allocation rule.

If measure-then-allocate is completed:

> We also added a measurement-based check following the protocol in A.15: `[MEASUREMENT_SUMMARY]`. This ties the empirical active band estimate to the selected scale without treating `tau` as a learned parameter.

反噬句：

- “We derive the optimal tau.”
- “The surrogate is the trained-transformer objective.”
- “The empirical tau sweep is part of the theorem.”

### 2.2 Likely Concern: Learnable tau underperforms fixed EVQ

Reviewer concern:

> If the allocation is useful, why does learning tau not work?

Current evidence:

- Table 4 reports learnable tau worse than EVQ: `paper/tables/table4_pe_dominant.tex:12-15`.
- Existing prep explains training loss inside `L_train` is myopic for OOD extrapolation.

Response posture:

- Treat this as evidence that in-training allocation is hard to learn from in-range loss.
- If logs exist, report learned tau trajectory.
- Do not claim learnable tau validates the closed form.

Draft:

> This is an important negative result. The training objective only observes the in-range loss, while the extrapolation benefit is out of range and the in-range waterbed cost is immediate. Therefore gradient-based tau learning is biased toward the in-range basin and does not reliably discover the extrapolation allocation. We will add this interpretation and, if logs are included, the learned-tau trajectory.

If trajectory available:

> In our runs, learned tau `[DRIFTED_TO_SMALLER / OSCILLATED / STAYED_FLAT]`, consistent with the training-loss signal being weak or myopic for extrapolation allocation.

反噬句：

- “Learnable tau failure is irrelevant.”
- “The learned tau result supports the closed form automatically.”
- “The learned row is just undertrained.”

### 2.3 Likely Concern: NTK-aware composition can reverse

Reviewer concern:

> EVQ does not compose universally with range scalers; NTK-aware can be worse.

Current evidence:

- Table 5 reports NTK-aware at 32x: Geo 198.1, EVQ2 143.3, EVQ4 331.4: `paper/tables/table5_phase11_leverage.tex:1-17`.

Response posture:

- Concede universal scaler composition is not claimed.
- Defend Primary I only as matched-scale EVQ x YaRN complementarity.

Draft:

> We agree and will sharpen the wording. The primary composition claim is matched-scale EVQ+YaRN substrate/range complementarity, not universal monotonic compatibility with every inference-time rescaler. The NTK-aware row is useful precisely because it shows that composition depends on the downstream scaler.

反噬句：

- “EVQ helps any scaler.”
- “NTK-aware is a bad baseline.”
- “The NTK reversal does not matter.”

## 3. R2: Empirical / Rigorous Reviewer

R2 is the most dangerous reviewer. The response should not begin with theory; it should begin by closing confounds.

Recommended R2 order:

1. LoRA confound control.
2. Training budget / undertraining story.
3. YaRN tuned-scale scope.
4. PK vs AR exact metric.
5. Primary II seed scope.
6. 1B raw reversal as limitation.
7. Figure/Table correction.

### 3.1 LoRA confound

Reviewer concern:

> The LLaMA-3 LoRA gains may come from LoRA/LongAlign adaptation, not EVQ frequency injection.

Current evidence:

- Current paper table only has Base vs EVQ-LoRA: `paper/appendix/a4_supporting_experiments.tex:35-51`.
- User-provided new evidence says Geo+LoRA is the strongest update, but exact numbers are not yet in current workspace.
- Data intake file: `rebuttal/TABLE23_LORA_WORKSHEET.md`.

Response if exact numbers are available:

> The reviewer is right that the original two-row table could not isolate frequency injection from LoRA/LongAlign adaptation. We added a matched Geo+LoRA control using the same pretrained checkpoint, data, rank, and 300-step schedule. We now report Base, Geo+LoRA, and EVQ-LoRA side by side. The Base -> Geo+LoRA change measures adaptation cost; only the Geo+LoRA -> EVQ-LoRA difference is attributed to EVQ frequency injection.

Table skeleton:

| Model | Adaptation | 8K PPL | 16K PPL | 32K PPL | Seed scope |
| --- | --- | ---: | ---: | ---: | --- |
| Base | none | `[BASE_8K]` | `[BASE_16K]` | `[BASE_32K]` | `[BASE_SCOPE]` |
| Geo+LoRA | same LoRA/LongAlign | `[GEO_LORA_8K]` | `[GEO_LORA_16K]` | `[GEO_LORA_32K]` | `[GEO_SCOPE]` |
| EVQ-LoRA | same LoRA/LongAlign + EVQ | `[EVQ_LORA_8K]` | `[EVQ_LORA_16K]` | `[EVQ_LORA_32K]` | `[EVQ_SCOPE]` |

If zero-training scaler eval is available:

> We also include a training-free Geo+[Dynamic NTK/YaRN] reference at 16K/32K to separate EVQ-LoRA from the default eval-only context-extension baseline.

If exact numbers are not available:

> We should not use this paragraph in final rebuttal. Without exact Geo+LoRA numbers, the safe response is to concede the old table is supporting and confounded.

反噬句：

- “LoRA proves industrial-scale training.”
- “EVQ-LoRA solves LLaMA-3 long context.”
- “The +30% 8K cost is negligible.”
- “The whole Base -> EVQ-LoRA difference is EVQ.”

### 3.2 Training budget / undertraining

Reviewer concern:

> EVQ might only work because the baseline is undertrained.

Current evidence:

- Primary I curated JSON has 100M tokens, train length 2048, seeds `[42,123,7]`: `data/curated/table2_evq_yarn_454m_passkey_10pct.json:8-18`.
- MLA progression grows to `-31.1%` while in-range cost shrinks: `paper/appendix/a3_supporting_results.tex:21-31`.
- 750M continuation has larger 16K gap despite lower in-range PPL: `paper/tables/table6_750m_continue_supporting.tex:10-17`.
- LoRA is an industrial checkpoint anchor only if Geo+LoRA control is clean.

Response:

> We added total token budgets and seed scope next to the primary tables. We also avoid calling Chinchilla-style counts “overtraining.” The narrower point is that the EVQ signal is not explained by a simple undertraining-only story: in the MLA progression, the long-range gap grows while the in-range cost shrinks; the 750M continuation row shows a large 16K gap despite low in-range PPL; and the controlled LoRA experiment, if included, tests a heavily pretrained checkpoint.

Primary II provenance:

> For the DAPE-style diagnostic, we report only the traceable seed/token scope and avoid mixing it with later Phase 11B scripts: Table 4 is the 128-token / 15M-token protocol, while Phase 11B is a separate 256-token / 100M-token supporting protocol.

反噬句:

- “9B tokens is overtraining.”
- “Industrial models are trained less than this.”
- “Training longer cannot remove EVQ.”
- “The 1B row proves robustness.”

### 3.3 YaRN tuned-scale / training-free scaler baseline

Reviewer concern:

> Fixed `s=8` Geo+YaRN is not a tuned scaler baseline.

Current evidence:

- Table 2 is matched-scale `s=8`: `paper/tables/table2_evq_yarn_main.tex:1-2`.
- Main text says not dominance over every tuned-scale baseline: `paper/sections/05_experiments.tex:22`.

Response if sweep completed:

> We added a Geo+YaRN scale sweep over `[SCALE_SET]`. The purpose is not to claim EVQ dominates all rescalers, but to test whether the matched-scale result is explained by an obviously mistuned Geo baseline. The best Geo+YaRN result is `[BEST_GEO_RESULT]`, while EVQ+YaRN is `[EVQ_RESULT]`.

Response if no sweep:

> We agree that Table 2 is a matched-scale substrate comparison, not a tuned-scaler leaderboard. We will make this scope explicit and avoid claiming dominance over best-tuned Geo+YaRN, Dynamic NTK, LongRoPE, or LongRoPE2.

反噬句:

- “EVQ beats tuned YaRN.”
- “Training-free scaling is irrelevant.”
- “Fixed scale is the default production setting.”

### 3.4 PK vs autoregressive exact

Reviewer concern:

> Passkey may be teacher-forced and not actual retrieval.

Current evidence:

- Main text defines PK as teacher-forced NLL-gap unless AR exact is explicitly marked: `paper/sections/05_experiments.tex:7`.
- 750M table shows teacher-forced passkey and AR exact can diverge: `paper/tables/table6_750m_continue_supporting.tex:10-17`.

Response if AR exact completed:

> We now report AR exact separately from teacher-forced PK. The original PK endpoint is an NLL-gap diagnostic, not a claim of exact generation.

Response if AR exact not completed:

> We clarify the metric definition throughout: PK denotes teacher-forced NLL-gap retrieval unless explicitly labeled AR exact. We do not use PK alone as evidence that the model can generate the key.

反噬句:

- “PK means exact retrieval.”
- “Teacher-forced metrics are equivalent to generation.”
- “AR exact is unnecessary.”

### 3.5 Primary II seed scope

Reviewer concern:

> Primary II is single seed and may be cherry-picked.

Current evidence:

- Body says retained seed-42 Geo/DAPE/EVQ: `paper/sections/05_experiments.tex:41`.
- Table 4 says Geo/DAPE/EVQ seed 42; learnable tau is 3-seed: `paper/tables/table4_pe_dominant.tex:2`.

Response:

> We will make the seed scope explicit. The DAPE-style row is a PE-dominant diagnostic stress test, not the sole statistical anchor of the paper. Geo/DAPE/EVQ are reported under the retained seed-42 protocol, while the learnable-tau row is multi-seed. We will not present this as comprehensive DAPE or learned-PE dominance.

If extra seeds completed:

> We added seeds `[SEEDS]`; the resulting mean/std is `[RESULT]`.

反噬句:

- “Single seed is enough.”
- “Retained seed-42 is equivalent to three-seed evidence.”
- “Primary II proves broad PE dominance.”

### 3.6 1B raw reversal

Reviewer concern:

> The 1B row reverses raw EVQ and undermines the claim that training saturation does not matter.

Current evidence:

- Current appendix acknowledges raw EVQ reversal and EVQ+YaRN+FT small win: `paper/appendix/a4_supporting_experiments.tex:29-30`.
- Evidence-tier table has been relabeled from the harmful “robustness to training saturation” wording to a schedule-sensitivity check: `paper/tables/table_evidence_tier.tex:20`.
- A.19-like analysis supports channel scarcity amplification: `paper/appendix/a1_proofs.tex:628-630`.

Response:

> We agree that the 1B MLA row should not be described as robustness to training saturation. It is a single-seed schedule-sensitivity stress check in a scarce-channel MLA regime, not a same-configuration token-scaling ablation. We have relabeled it and will discuss it as a limitation motivating fixed-length continuation or stage-wise re-warp/adaptation.

Optional mechanism:

> This also explains why the progressive MHA row need not contradict the MLA reversal: the MLA setup has far fewer rotary channels, and the K-dependent distortion terms make allocation mismatch more severe in scarce-channel regimes.

反噬句:

- “The reversal is noise.”
- “EVQ is robust to saturation.”
- “Progressive MHA proves schedule mismatch is harmless.”
- “The K=16 explanation quantitatively predicts the 1B PPL.”

### 3.7 QuALITY Figure/Table mismatch

Reviewer concern:

> Figure 8 and Table 21 disagree; the figure plots accuracy but the caption says NLL.

Current evidence:

- Verified in `rebuttal/FIGURE_TABLE_AUDIT.md`.
- In the stale version, Table 21 was NLL-coherent while Figure 8 was an accuracy figure under an NLL caption. The working PDF now replaces Figure 8 with a Gold-NLL plot.

Response:

> We thank the reviewer for catching the stale/mislabeled QuALITY figure. The table values and text use gold-answer NLL; the figure panel was an older accuracy visualization and should not have been captioned as NLL. We have replaced the figure with a Gold-NLL plot consistent with Table 21, and we do not use QuALITY accuracy as a primary claim.

反噬句:

- “Reviewer misread the figure.”
- “Accuracy deltas and NLL deltas are equivalent.”
- “This is only an appendix issue.”

## 4. R3: Systems / Practical Reviewer

### 4.1 Likely Concern: Scale and practicality

Reviewer concern:

> Evidence is mostly diagnostic/small scale; why should systems readers care?

Current evidence:

- Main claim is zero learned parameters and training-time frequency schedule.
- MLA scarce-channel test is architecturally relevant but not production-identical.
- Dead-channel audit is reusable diagnostic evidence.
- Controlled LoRA can be industrial-checkpoint relevant if exact numbers are inserted.

Response:

> We agree that production-scale from-scratch validation remains future work. The systems relevance is narrower: EVQ-Cosh is a zero-learned-parameter schedule change, the MLA experiment tests a scarce-rotary-channel regime that is relevant to compressed-attention designs, and the dead-channel audit exposes a diagnostic failure mode that can be applied independently of EVQ adoption.

If LoRA table is clean:

> The controlled LoRA experiment adds an industrial-checkpoint anchor: under the same LoRA/LongAlign budget, the Geo control does not reproduce the long-range gain, while EVQ-LoRA does.

反噬句:

- “Production ready.”
- “Industrial scale proven.”
- “Downstream benchmark gaps do not matter.”

### 4.2 Likely Concern: MLA tau convention and production mismatch

Reviewer concern:

> The MLA tau choice uses `d_eff=d_head`, and the setup differs from DeepSeek production settings.

Current evidence:

- Appendix says `d_eff=d_head` is empirical operating convention and tau ablations are natural: `paper/appendix/a3_supporting_results.tex:8-10`.
- Appendix says paper uses `d_rope=32`, base 500K; production DeepSeek uses different base/channel settings: `paper/appendix/a3_supporting_results.tex:6`.

Response:

> We agree and will clarify the wording. The MLA experiment is a production-relevant scarce-channel stress test, not a production-identical DeepSeek validation. The `d_eff=d_head` rule is an empirical operating convention for this architecture; we will either add the `tau=d_rope/sqrt(L)` sanity ablation or mark it as a limitation.

反噬句:

- “This is the DeepSeek configuration.”
- “`d_eff=d_head` is derived by the theory.”
- “No tau sanity check is needed.”

### 4.3 Likely Concern: Downstream benchmarks are weak

Reviewer concern:

> QuALITY/RULER/LongBench signals are not strong enough.

Current evidence:

- QuALITY accuracy is near random; NLL separates: `paper/appendix/a3_supporting_results.tex:71-87`.
- AGENTS scope: supporting evidence remains supporting.

Response:

> We do not use QuALITY accuracy as a primary benchmark claim. At 454M parameters, accuracy is capacity-limited and near random; we report it as supporting/non-regression context while the probability-space NLL shows the directional effect. The primary evidence remains diagnostic mechanism tests rather than downstream leaderboard performance.

反噬句:

- “Benchmarks are irrelevant.”
- “Diagnostics are better than benchmarks.”
- “QuALITY proves downstream improvement.”

## 5. AC Response Strategy

The AC likely cares less about every technical detail than about whether the authors are honest, scoped, and responsive.

### 5.1 AC summary paragraph

Recommended:

> Across the reviews, the common concern is not whether frequency allocation is interesting, but whether the current evidence overstates its scope. We agree with that distinction. We therefore narrow the claim to training-time RoPE frequency allocation as a finite-spectral-budget design axis, add controls for the main empirical confounds, and relabel supporting stress checks that were too broadly described. The revised paper will not claim universal long-context SOTA or production-scale validation.

### 5.2 AC-visible concessions

List these explicitly if space allows:

- 1B MLA row: relabeled as schedule-sensitivity limitation.
- LoRA: old row was confounded; new table only if Geo+LoRA exact numbers are available.
- Primary II: seed-scoped diagnostic.
- PK: teacher-forced NLL-gap unless AR exact.
- QuALITY figure: stale/mislabeled and will be corrected.
- MLA: sparse-channel stress test, not production-identical DeepSeek config.

### 5.3 AC-visible positive anchors

Use only scoped anchors:

- Primary I: three-seed matched-scale EVQ+YaRN substrate/range complementarity.
- Primary III: three-seed MLA scarce-channel stress test.
- Dead-channel audit: reusable diagnostic.
- Controlled LoRA: industrial checkpoint relevance, if exact numbers and Geo control are clean.

Do not anchor on:

- QuALITY accuracy;
- broad RULER/LongBench claims;
- 1B raw reversal;
- learnable tau as positive result;
- universal scaler composition.

## 6. If We Can Only Fit Three New Items In Rebuttal

### Item 1: Geo+LoRA Control

Use only with exact table.

One-sentence version:

> A matched Geo+LoRA control isolates LoRA/LongAlign adaptation from EVQ frequency injection; the table now reports Base, Geo+LoRA, and EVQ-LoRA under the same checkpoint, data, rank, and 300-step schedule.

### Item 2: Relabel 1B + Provenance

One-sentence version:

> We relabel the 1B MLA row as schedule-sensitivity evidence, not saturation robustness, and add token/seed scope to the primary tables.

### Item 3: Metric/Figure Trust Fix

One-sentence version:

> We clarify PK as teacher-forced NLL-gap unless explicitly AR exact, and we correct the stale QuALITY figure so that the NLL caption matches the plotted data.

## 7. Final Pre-Send Checklist

Do not send final rebuttal until every checked item is true:

- [ ] Every new number has a file/log/source path.
- [ ] Geo+LoRA exact table is filled or the LoRA control paragraph is removed.
- [ ] If training-free scaler eval is mentioned, its exact result is filled.
- [ ] Primary I token budget and seed scope are traceable.
- [x] Primary II protocol is not mixed with Phase11B.
- [ ] 1B row is not called robustness to training saturation.
- [ ] PK is not described as AR exact unless AR exact is reported.
- [ ] Figure 8/Table 21 correction is acknowledged or fixed.
- [ ] MLA result is called production-relevant scarce-channel, not production-identical DeepSeek.
- [ ] No response claims EVQ replaces YaRN, LongRoPE, DAPE, FIRE, or learned PE.
