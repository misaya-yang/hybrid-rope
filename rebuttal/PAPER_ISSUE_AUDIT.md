# EVQ-Cosh Paper-Facing Rebuttal Issue Audit

日期：2026-06-10

用途：把 `rebuttal/raw_sources/` 和 `rebuttal/fable相关资料原文.md` 里的 reviewer 攻击面，逐项对照当前论文源文件。本文不是 rebuttal 正文，也不是二次投稿计划；它只回答四个问题：

1. reviewer 的质疑在当前论文里有没有事实基础；
2. 这是误解、真实硬伤，还是 mixed issue；
3. rebuttal 里应该 defend、concede、scope down，还是补一个最小实验；
4. 哪些措辞会反噬。

## 0. 当前结论

现在最应该保护的主线是窄 claim：

> EVQ-Cosh studies training-time RoPE frequency allocation as a finite-spectral-budget design axis, complementary to inference-time range scaling.

当前材料能 defend 三根主柱：

| Pillar | 当前证据 | 能说 | 不能说 |
| --- | --- | --- | --- |
| Primary I EVQ x YaRN | `paper/sections/05_experiments.tex:19-23`; `paper/tables/table2_evq_yarn_main.tex:1-17`; `data/curated/table2_evq_yarn_454m_passkey_10pct.json:8-18` | matched-scale substrate/range complementarity | tuned Geo+YaRN/LongRoPE dominance |
| Primary II PE-dominant diagnostic | `paper/sections/05_experiments.tex:38-43`; `paper/tables/table4_pe_dominant.tex:1-18`; `data/curated/fig3_extreme_128.json:7-34` | seed-scoped PE-dominant stress test | broad learned-PE/DAPE dominance |
| Primary III MLA scarce-channel | `paper/sections/05_experiments.tex:49-52`; `paper/appendix/a3_supporting_results.tex:6-31`; `paper/appendix/a1_proofs.tex:628-630` | scarce rotary channels amplify allocation effects | production-identical DeepSeek validation or derived MLA tau theorem |

最危险的四个 trust sinks：

1. `paper/tables/table_evidence_tier.tex:20` has now been changed from `MLA 1B-token (robustness to training saturation)` to `MLA 1B-token schedule-sensitivity check`. This removes a direct rebuttal self-own.
2. `paper/appendix/a4_supporting_experiments.tex:35-51` still has only Base vs EVQ-LoRA, no Geo+LoRA result row, despite the new user-provided evidence saying Geo+LoRA is the strongest rebuttal update. The wording is now scoped as a post-hoc adaptation observation and explicitly says attribution requires a matched Geo+LoRA control.
3. Primary II provenance is now reconciled for rebuttal: Table 4 uses the 128-token / 15M-token DAPE-style protocol, while Phase 11B is a separate 256-token / 100M-token supporting protocol. Do not mix these. See `rebuttal/PRIMARY_PROVENANCE_NOTE.md`.
4. Figure/Table mismatch was verified in the submitted artifact. This rebuttal-only pass does not modify `paper/main.pdf`; the response acknowledges Figure 8/9 errors and the Table 21 erratum. See `rebuttal/FIGURE_TABLE_AUDIT.md`.

## 1. Source-Grounded Issue Matrix

| ID | Reviewer attack | Current paper evidence | Judgment | Rebuttal posture | Minimal action |
| --- | --- | --- | --- | --- | --- |
| I1 | LoRA row is confounded with LongAlign/LoRA adaptation | Existing appendix table has only Base and EVQ-LoRA (`paper/appendix/a4_supporting_experiments.tex:35-49`); revised text now explicitly says attribution requires matched Geo+LoRA control (`paper/appendix/a4_supporting_experiments.tex:51`) | True hard issue in current paper; wording risk reduced, but exact control row still needed | Concede old confound, then present new Geo+LoRA control if numbers are verified | Rebuild Table 23 as Base / Geo+LoRA / EVQ-LoRA; include seed scope and exact metrics |
| I2 | LoRA cost is too high, not modest | Existing table reports +30% at 8K (`paper/appendix/a4_supporting_experiments.tex:44-46`); revised text no longer calls this modest and states the cost explicitly (`paper/appendix/a4_supporting_experiments.tex:51`) | Wording risk now mitigated; attribution risk remains | Scope and decompose cost | Attribute Base -> Geo+LoRA as adaptation cost, Geo+LoRA -> EVQ-LoRA as EVQ incremental cost |
| I3 | 1B MLA raw reversal undermines saturation robustness | Current appendix acknowledges raw EVQ reversal and EVQ+YaRN+FT small win (`paper/appendix/a4_supporting_experiments.tex:29-30`) | True hard issue | Concede and relabel as schedule-sensitivity limitation | Label is now changed in `table_evidence_tier.tex:20`; still avoid using row as support |
| I4 | 1B row is being used as support | Evidence tier says Supporting, and label now says schedule-sensitivity check (`paper/tables/table_evidence_tier.tex:18-20`) | Tier is right; old wording fixed | Defend tiering, keep limitation framing | Call it single-seed schedule-sensitivity stress check |
| I5 | Primary II is single seed but called primary | Body says retained seed-42 Geo/DAPE/EVQ (`paper/sections/05_experiments.tex:41`); table says Geo/DAPE/EVQ seed 42 and Learnable tau 3-seed (`paper/tables/table4_pe_dominant.tex:2`) | True, but already partly disclosed | Scope as diagnostic, not broad statistical anchor | If no seeds, say seed-scoped; if seeds added, report all |
| I6 | Primary I token budget not visible enough | Curated JSON has 100M tokens (`data/curated/table2_evq_yarn_454m_passkey_10pct.json:8-18`); appendix reproducibility table now includes token budgets (`paper/appendix/a2_experiment_details.tex:20-29`) | Transparency gap, now addressed in working tree | Add token budget / seed scope | Use traceable JSON for Primary I; do not infer beyond curated source |
| I7 | Primary II token budget/provenance is inconsistent | Curated protocol says `L_train=128` (`data/curated/fig3_extreme_128.json:7`); historical report says 15M tokens (`docs/exp/2026-02-24_128tok_baseline_report.md:14-23`); phase11b script is `L=256`, 100M (`scripts/core_text_phases/phase11b_125m_dape.py:1-13`, `36-41`); reconciliation note created | True provenance risk, now documented | Report Table 4 as 128-token/15M and Phase11B as separate 256-token/100M supporting curves | Use `rebuttal/PRIMARY_PROVENANCE_NOTE.md`; do not mix protocols |
| I8 | Table 2 PK is not real retrieval | Main text defines PK as teacher-forced NLL-gap unless AR exact (`paper/sections/05_experiments.tex:7`); table caption uses PK but not AR exact (`paper/tables/table2_evq_yarn_main.tex:1-17`) | Mostly misunderstanding, but response should repeat definition | Clarify, do not overclaim | Add AR exact if available; otherwise keep PK diagnostic |
| I9 | 750M shows TF can saturate while AR differs | Table 6 has passkey 100/100 but AR exact 0% vs 77.5% (`paper/tables/table6_750m_continue_supporting.tex:10-17`) | True; useful for honest metric framing | Use as caution, not main proof | Mention PK diagnostic and AR exact separately |
| I10 | Geo+YaRN fixed scale is not tuned baseline | Table 2 caption says same fixed scale `s=8` (`paper/tables/table2_evq_yarn_main.tex:1-2`); body says not dominance over every tuned-scale baseline (`paper/sections/05_experiments.tex:22`) | Valid limitation already scoped | Defend matched-scale question only | If possible, add Geo+YaRN scale sweep; otherwise explicitly concede tuned-scaler gap |
| I11 | NTK-aware can reverse with EVQ | Appendix table notes NTK-aware at 32x: Geo 198.1, EVQ4 331.4 (`paper/tables/table5_phase11_leverage.tex:1-17`) | True, important | Scope EVQ+YaRN as YaRN-specific matched-scale evidence | Do not claim arbitrary scaler complementarity |
| I12 | MLA `d_eff=d_head` convention is underived | Appendix explicitly says empirical operating convention and direct tau ablations are natural sanity checks (`paper/appendix/a3_supporting_results.tex:8-10`) | True but already disclosed | Concede convention, defend result as empirical stress test | Run or promise `tau=d_rope/sqrt(L)` and code-head-dim ablations |
| I13 | MLA production mismatch | Appendix says paper uses `d_rope=32`, base 500K; DeepSeek uses `d_rope=64`, base 10K (`paper/appendix/a3_supporting_results.tex:6`) | True | Scope as compressed-RoPE family, not production-identical | Use “production-relevant sparse-channel regime,” not “DeepSeek configuration” |
| I14 | Downstream QuALITY is weak | Appendix says accuracy near 25% random baseline; NLL separates (`paper/appendix/a3_supporting_results.tex:71-87`) | True but not fatal | Scope downstream as non-regression / NLL check | Do not claim downstream SOTA or benchmark win |
| I15 | QuALITY figure/table mismatch | Submitted Figure 8 uses the superseded n=200 accuracy pilot under an NLL caption; n=2086 aggregate also proves the 26.6%→24.6% erratum (`rebuttal/FIGURE_TABLE_AUDIT.md`) | True P0 trust issue; response prepared, PDF unchanged | Concede and say it will be corrected in a revision | Keep QuALITY as supporting probability-space evidence, not a downstream accuracy claim |
| I16 | Theory is surrogate + scale splice | Theory says analytic claim is conditional on surrogate and tau is operating-point selector (`paper/sections/03_theory.tex:15`, `93-115`) | Mostly reviewer over-framing; paper already honest | Defend with scope | Avoid “derive optimal tau” language |
| I17 | Learnable tau underperforms fixed rule | Table 4: learnable tau PPL 437.9 vs EVQ 333.7 (`paper/tables/table4_pe_dominant.tex:12-15`); historical report shows learned tau converges around 1.14 and in-training objective is myopic (`docs/exp/2026-02-24_128tok_baseline_report.md:98-115`, `133-139`) | True, can become positive argument | Explain training loss cannot see OOD extrapolation | Add learned tau trajectory if logs are available |
| I18 | LoRA phenomenology is circular | A.13 says PPL 77.1 is reproduced after calibrating on same observation (`paper/appendix/a1_proofs.tex:405-416`) | True; already honest | Do not use as predictive proof | Use qualitative mechanism only |
| I19 | Channel-scarcity theory supports MLA mismatch explanation | A.19-like bound says K=16 vs K=64 amplifies K^-1/K^-2 components (`paper/appendix/a1_proofs.tex:628-630`) | Useful defense | Use to explain why progressive MHA and MLA schedule mismatch can differ | Do not oversell as quantitative PPL prediction |

## 2. Misunderstanding vs Real Hard Issue

### 2.1 Reviewer Misunderstandings / Over-Extensions

These can be corrected without new experiments, but should still be repeated clearly.

| Misunderstanding | Current evidence | Safe correction |
| --- | --- | --- |
| Paper claims universal long-context SOTA | Main text says PE mechanism study, not rescaler or learned PE (`paper/sections/05_experiments.tex:7`); limitations repeat this (`paper/sections/06_limitations.tex:6`) | “This is a mechanism study, not a deployment recipe or SOTA claim.” |
| PK was hidden as autoregressive retrieval | Main text defines PK as teacher-forced NLL-gap unless AR exact (`paper/sections/05_experiments.tex:7`) | “PK is a diagnostic endpoint; AR exact is separately named.” |
| 1B row is primary evidence | Evidence tier lists it as Supporting (`paper/tables/table_evidence_tier.tex:18-20`) | “It is supporting and should be relabeled as limitation/schedule sensitivity.” |
| QuALITY accuracy is the main downstream claim | Appendix says accuracy is near random and NLL is the discriminator (`paper/appendix/a3_supporting_results.tex:71-87`) | “Downstream accuracy is capacity-limited; the claim rests on diagnostics.” |
| MLA result claims production-identical DeepSeek | Appendix explicitly distinguishes settings (`paper/appendix/a3_supporting_results.tex:6`) | “Compressed-RoPE family relevance, not production identity.” |

### 2.2 True Hard Issues

These cannot be waved away; response must concede, scope, or add evidence.

| Hard issue | Why it matters | Minimum credible response |
| --- | --- | --- |
| Geo+LoRA missing in current paper | Without control, 8x/19x LoRA gains are confounded | Use new Base / Geo+LoRA / EVQ-LoRA table |
| 1B raw reversal | It directly attacks training saturation robustness | Relabel as limitation, explain schedule/channel mismatch, do not call support |
| Primary II seed scope | Seed-42 primary diagnostic can look cherry-picked | Report exact seed history and avoid broad DAPE dominance |
| Primary I tuned scaler gap | Fixed-scale YaRN is not default tuned Llama context extension | Add sweep or concede matched-scale scope |
| AR exact missing for Primary I | TF PK can overstate generation ability | Add AR exact or explicitly keep metric diagnostic |
| MLA tau convention | Strongest systems result has underived tau choice | Add tau ablation or limitation |
| Token budget visibility | Reviewers can frame all primaries as undertrained | Add traceable token budgets only where confirmed |
| Figure/Table consistency | Any mismatch creates global trust loss | Working PDF now fixed; acknowledge if reviewer saw stale version |

## 3. Minimal Experiment / Artifact Plan

Priority is based on rebuttal ROI and whether the action answers an actual reviewer question. This is not an open-ended experiment wishlist.

### P0: No-GPU Or Result-Assembly Gates

| Task | Goal | Completion evidence | Backfire if skipped |
| --- | --- | --- | --- |
| Submitted Figure/Table audit | Resolve rebuttal posture for Figure 8/9 and Table 20/21 | `rebuttal/FIGURE_TABLE_AUDIT.md` records both errors, source-of-truth values, and revision commitment | Reviewer distrusts all numeric presentation |
| Geo+LoRA table assembly | Close LoRA confound with new evidence | One table with Base / Geo+LoRA / EVQ-LoRA; same checkpoint, data, rank, steps, seed scope | R2 moves from “interesting” to “confounded supporting row” |
| LoRA wording scope fix | Prevent the current two-row appendix from over-claiming before exact Geo+LoRA numbers arrive | Done: appendix now calls Table 23 a post-hoc adaptation observation, reports +30% cost explicitly, and says attribution requires matched Geo+LoRA | Reviewer quotes “modest” or treats Base -> EVQ-LoRA as claimed causal attribution |
| Primary token/provenance note | Prevent undertraining/provenance attack | Done: token counts added to appendix table and `rebuttal/PRIMARY_PROVENANCE_NOTE.md` reconciles Primary II vs Phase11B | Reviewer controls the narrative with “missing token budget” |
| Rename 1B evidence label | Stop self-own | Done: `robustness to training saturation` removed | Rebuttal contradicts paper’s own evidence-tier label |
| Remove unsupported text base-sweep wording | Stop R2 from asking for a robustness table that is not visible in the paper | Done: evidence-tier robustness row now says multi-scale extrapolation instead of base sweep | Reviewer quotes the tier table and asks where the text base-sweep results are |

LoRA-specific artifact status: current public workspace has scripts and plans for Geo+LoRA (`scripts/2026-04/README.md:7-21`, `experiments/lora_evq_v2/EXPERIMENT_PLAN.md:27-70`) but no JSON/CSV result file under `experiments/lora_evq_v2` or `scripts/2026-04`. Use `rebuttal/TABLE23_LORA_WORKSHEET.md` as the data intake template before writing exact LoRA claims.

### P1: Eval-Only Or Log-Only Checks

| Task | Goal | Why it is small enough for rebuttal |
| --- | --- | --- |
| Geo + Dynamic NTK / YaRN zero-training reference for LoRA checkpoint | Stop R2 from replacing LoRA confound with “default training-free scaler baseline” | Eval-only on existing checkpoint |
| Primary I Geo+YaRN scale sweep | Show fixed matched scale is not hiding an easy Geo win, or scope if it is | Eval-only if checkpoints exist |
| Primary I AR exact | Separate diagnostic PK from actual generation | Existing evaluator supports AR exact path according to prep notes; run only if checkpoints available |
| Learned tau trajectory | Turn negative result into mechanism argument | Log extraction only if training logs/checkpoints exist |
| QuALITY source/data audit | Verify current source vs submitted figure | No training |

### P2: Training Runs Only If Budget Allows

| Task | Goal | Risk |
| --- | --- | --- |
| MLA `tau=d_rope/sqrt(L)` sanity | Protect strongest systems claim | If it wins, must revise MLA convention story |
| Primary II extra seeds | Reduce seed-42 attack | Must not create provenance drift with Phase11B |
| Fixed-L MLA continuation gap-vs-tokens | Directly answer saturation vs schedule mismatch | More expensive; bad result forces scope down |

### P3: Do Not Lead With These

| Task | Reason |
| --- | --- |
| Big LongBench/RULER push | 454M capacity may be at floor; likely adds noise |
| 1B multi-seed from scratch in rebuttal window | Too expensive and high risk |
| Broad new method comparisons | Turns rebuttal into second submission |
| New theoretical theorem for tau | Risky; current best posture is operating default + basin selector |

## 4. Response Postures By Reviewer Type

### R1 Theory

Defend:

- Cosh shape is exact for stated broadband surrogate.
- Paper already distinguishes shape and scale layers (`paper/sections/03_theory.tex:93-115`).
- Tau is operating default / basin selector, not global optimum.

Concede:

- Shape and scale are not unified into one full-attention theorem.
- `L_eff^J` measurement is a registered falsifiable test, not completed (`paper/sections/03_theory.tex:113`; `paper/appendix/a1_proofs.tex:504`).
- Learnable tau underperformance requires explanation.

Best extra evidence:

- learned tau trajectory;
- measure-then-allocate proxy on existing checkpoints;
- cite Table 1 / epistemic map rather than pretending everything is proven.

Never say:

- “We derive the globally optimal tau.”
- “Learnable tau validates the closed form.”
- “The surrogate is the exact trained-attention objective.”

### R2 Empirical

Defend:

- Primary I has 3 seeds and a curated JSON with 100M tokens, seed list, and PK metric (`data/curated/table2_evq_yarn_454m_passkey_10pct.json:8-18`).
- Primary III is 3-seed MLA at 500M tokens (`paper/appendix/a3_supporting_results.tex:8`, `21-31`).
- Metrics are scoped; PK is teacher-forced.

Concede:

- Primary II Geo/DAPE/EVQ is seed 42.
- Fixed-scale YaRN is not tuned scaler dominance.
- 1B raw reversal is real and cannot support saturation robustness.
- Current LoRA row is confounded unless new Geo+LoRA numbers are shown.

Best extra evidence:

- Geo+LoRA table;
- Geo+YaRN scale sweep;
- AR exact for Primary I;
- exact token/provenance table;
- Figure/Table audit.

Never say:

- “9B tokens is overtraining.”
- “1B row proves robustness.”
- “PK means the model can generate the key.”
- “EVQ beats tuned YaRN/LongRoPE.”

### R3 Systems / Practical

Defend:

- Zero learned parameters and one schedule change.
- MLA scarce-channel stress test is architecturally relevant.
- Dead-channel audit is useful to video/temporal RoPE community.
- New controlled LoRA result is industrial-checkpoint relevant, if table is clean.

Concede:

- Production-scale from-scratch validation is not done.
- LoRA post-hoc retrofit has nontrivial cost.
- RULER/LongBench/downstream are not the main value.
- MLA production config differs.

Best extra evidence:

- controlled LoRA table with incremental cost;
- avoid weak benchmark overreach;
- emphasize immediate diagnostic utility of dead-channel audit and MLA channel scarcity.

Never say:

- “Production ready.”
- “Industrial scale proven.”
- “Downstream benchmark gap is irrelevant.”

## 5. Response Snippet Inventory

These are safe snippets only after the corresponding evidence exists.

### Scope Opening

> We agree that several rows should be read as mechanism evidence rather than production-scale validation. We therefore separate primary stress tests from supporting rows. The main claim is narrower: EVQ changes the training-time RoPE frequency substrate, and matched range scaling can act differently on that substrate.

### LoRA Control

> The reviewer is right that the original LoRA row could not isolate frequency injection from LongAlign/LoRA adaptation. We added a matched Geo+LoRA control using the same pretrained checkpoint, data, rank, and 300-step schedule. We will report Base, Geo+LoRA, and EVQ-LoRA side by side, and attribute only the Geo+LoRA -> EVQ-LoRA difference to EVQ frequency injection.

Do not use this until exact Geo+LoRA numbers are in hand.

### 1B Reversal

> We agree that the 1B MLA row should not be described as robustness to training saturation. It is a single-seed schedule-sensitivity stress check in a scarce-channel MLA regime, not a same-configuration token-scaling ablation. We will relabel it and discuss it as a limitation motivating fixed-length continuation or stage-wise re-warp/adaptation.

### Training Amount

> We added total token budgets and seed scope next to the primary tables. We also avoid describing Chinchilla-style token counts as overtraining. The narrower empirical point is that the EVQ signal is not explained by a simple undertraining-only story: the 8K/500M MLA progression grows to the final gap while in-range cost shrinks, the 750M continue@4K row shows a large long-range gap despite low in-range PPL, and the controlled LoRA experiment tests a heavily pretrained checkpoint.

### Primary II

> The DAPE-style comparison is a PE-dominant diagnostic. Geo, DAPE, and EVQ are seed-scoped in the reported contrast; the learnable-tau row is multi-seed. We will make this seed scope and token provenance explicit and avoid presenting the row as comprehensive learned-PE dominance.

### PK Metric

> PK denotes teacher-forced NLL-gap retrieval in the primary tables, not autoregressive exact match. We use it as a positional-encoding diagnostic; AR exact, when reported, is explicitly named separately.

### MLA Tau

> In MLA, `d_rope` counts quantized rotary channels, while the deployed `d_eff` is an empirical operating convention for the latent attention pathway. We agree this is not a theorem from `d_rope` alone and will either add the direct `tau=d_rope/sqrt(L)` sanity ablation or list it as a limitation.

## 6. Patch Map For Later

Do not patch paper numbers until final evidence is in hand. These are likely later edits:

| File | Change | Why |
| --- | --- | --- |
| `paper/tables/table_evidence_tier.tex` | Rename “MLA 1B-token (robustness to training saturation)” | Prevent direct contradiction with rebuttal |
| `paper/appendix/a4_supporting_experiments.tex` | Replace LoRA table with Base / Geo+LoRA / EVQ-LoRA if numbers are verified | Close confound |
| `paper/sections/05_experiments.tex` | Add or point to token budgets and seed scope | Answer R2 transparency |
| `paper/appendix/a2_experiment_details.tex` | Add confirmed token/provenance rows for Primary I/II | Reproducibility |
| QuALITY figure/caption/table | Fix verified Figure 8/Table 21 mismatch | Trust |
| `paper/sections/06_limitations.tex` | Add sharper 1B schedule-sensitivity limitation and tuned-scaler gap | Avoid overclaim |

## 7. Completion State Against Current Goal

Goal requirement | Current state | Evidence | Status
--- | --- | --- | ---
Read fable original in detail | Original is archived and indexed | `rebuttal/raw_sources/00_INDEX.md`, `rebuttal/fable相关资料原文.md` | Done
Store all source material as MD | Verbatim per-source MD files exist | `rebuttal/raw_sources/*.md` | Done
Recognize new evidence that changes the situation | Geo+LoRA is elevated to P0 and old LoRA confound is marked obsolete if numbers are verified | `REBUTTAL_PREPARATION.md`, this audit I1-I2 | Partly done; exact numbers still needed
Prepare rebuttal plan | Plan exists in `REBUTTAL_PREPARATION.md`; this file adds source-grounded matrix | `rebuttal/REBUTTAL_PREPARATION.md`, this file | Done for strategy
Prepare possible experiments | P0/P1/P2 plan included | Section 3 | Done for prioritization
Identify misunderstandings of paper | Misunderstanding vs hard issue section included | Section 2 | Done
Identify rebuttal wording that will backfire | Per-reviewer never-say and response posture included | Sections 4-6 | Done
Multi-angle research of actual paper issues | Current paper source lines audited for core attacks | Sections 0-6 | In progress; submitted PDF figure/table audit and exact LoRA numbers remain missing

This is enough to start drafting a reviewer-by-reviewer response, but not enough to mark the full rebuttal preparation goal complete because the most decision-changing new evidence, Geo+LoRA, still lacks exact numeric table values in this workspace.
