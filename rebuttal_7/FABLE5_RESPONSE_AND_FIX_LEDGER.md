# Fable5 response and fix ledger

## Scope and response posture

This ledger triages every potential rebuttal question in the authoritative Fable5 packet. It is an internal preparation artifact, not a claim that the simulated questions are the final NeurIPS reviews. Status meanings:

- `DONE`: resolved in tracked paper/code or by an auditable local calculation.
- `PARTIAL`: the honest textual/provenance part is resolved, but a requested artifact or experiment is still unavailable.
- `PENDING-EXP`: requires training, checkpoint access, or a new empirical analysis.

Overall readiness is `draft_with_placeholders`. The response should lead with the QuALITY correction, concede remaining evidence gaps directly, and avoid promising results before they exist.

## Acceptance-critical order

1. Reporting integrity: Q1–Q2.
2. Primary II seeds: Q3.
3. Nearest-neighbor tuned-base controls: Q5–Q6.
4. MLA convention ablation: Q7.
5. Recovered autoregressive exact match and metric naming: Q9.
6. Trained-attention validation: Q8.

The proposed 7B-class fine-tuning experiment addresses Q18 and deployment scope, but it is not ahead of the AC-critical items above. Because the submitted supporting row is LLaMA-3-8B-Instruct, the cleanest rebuttal repair is a matched 8B Geo/EVQ rerun; introducing a different 7B family first would add a new confound.

## Rebuttal-ready core response

Package readiness: `draft_with_placeholders`. The first item below is ready; the remaining items must retain their visible placeholders until the named artifacts exist.

Portable provenance update: `IGNORED_ASSET_RECONCILIATION.md` records the complete local ignored-asset audit. Primary III, the L=256 Phase11 archive, QuALITY n=2,086 aggregate, and the base=10K/500K pilot are raw-JSON-backed in `data/curated/`; the 99-run sweep has a sanitized portable manifest. Learnable-tau and the MLA channel-count pilot remain report-backed.

### Reporting integrity (F5-Q1–Q2; ready)

We audited the QuALITY chain from the recovered full-evaluation aggregate through Table 21, Figure 8(a), and the signal-gradient paragraph. Table 21 is the retained result: a seed-42, `n=2086` evaluation of checkpoints initialized at 2K, continued and fine-tuned at 4K, and evaluated at 4K/8K/16K. The former Figure 8(a) instead visualized an obsolete `n=200` accuracy-only pilot, included a 32K point outside the retained full-evaluation table, and had an NLL caption. We replaced it with the four gold-answer-NLL rows from Table 21, explicitly excluded the pilot, and clarified why 4K is in-distribution for this downstream checkpoint. The tracked snapshot is raw-JSON-backed by the recovered aggregate SHA256 while excluding machine and checkpoint fields.

The review packet's `+0.2pp` summary appears to be a transcription/arithmetic error: the printed 8K-raw entries are `26.8 - 24.6 = +2.2pp`. The manuscript nevertheless used that one near-floor row too strongly. We removed it as a downstream endpoint, now list all four rounded accuracy deltas (`+0.7/+2.2/+0.1/-0.4pp`), and state that they have no stable direction. The retained probability-level observation is the 8K-raw gold-answer-NLL change (`3.202→2.239`, `-30.1%`), while QuALITY accuracy is treated as inconclusive.

### Acceptance-critical empirical gaps (F5-Q3, Q5–Q8; placeholders)

- **Primary II seeds (Q3):** the submitted Geo/DAPE/EVQ rows remain a seed-42 diagnostic. `[Insert exact 128→8K seeds 137/256, per-seed values, mean/std, and paired deltas.]` The recovered L=256 raw/YaRN three-seed records are portable in `data/curated/phase11_l256_3seed_recovered.json`, but are a different protocol and contain no DAPE row; they cannot substitute for this replication.
- **Tuned base and b=10K (Q5–Q6):** the recovered raw-backed 151.9M/L=512 seed-42 pilot supports only that the direction appears at base 10K as well as 500K. The acceptance-critical closure still uses the 125M, `L_train=128` anchor with identical data/tokens/optimizer/seeds: `[Insert Geo b∈{10K,100K,500K,2M}, EVQ b=500K, and at b=10K bare-rule versus c_pred results.]`
- **MLA convention (Q7):** the current three-seed run is now portable and raw-JSON-backed, but it tests only `tau=1.414`; it does not identify the optimal `d_eff` convention. `[Insert tau=0.354, 0.707, and 1.414 screen, then replicated relevant comparison.]` The 125M channel-count pilot is qualitative support, not this ablation.
- **Measured effective length (Q8):** `1/L` remains a falsifiable diffuse-attention approximation. `[Insert estimator definition, sampled layers/heads/tokens, kappa_att, L_eff^J, and uncertainty from existing checkpoints.]`
- **Autoregressive passkey (Q9):** the paper now labels PK as teacher-forced NLL-gap retrieval. Using recovered raw logs, we compare Teacher-Forced (TF) retrieval and Autoregressive (AR) exact match rates side-by-side. At 8K, Geo+YaRN reaches 61.3% TF but drops to 0.0% AR exact match, while EVQ+YaRN reaches 100.0% TF and maintains a significant 58.0% AR exact match (seed 42: 58%, seed 123: 18%, seed 7: 98%).

## Reporting correction record

The corrected downstream QA artifact now plots the four gold-answer-NLL rows reported in Table 21, rather than the obsolete 200-sample accuracy pilot. The paper states that the evaluated models were initialized at 2K, continued and fine-tuned at 4K, making 4K in-distribution for this downstream protocol. The full table is the `n=2086`, seed-42 evaluation preserved in `data/curated/quality_454m_full_eval.json`; its source aggregate SHA256 is recorded, while machine and checkpoint fields are omitted. The old `phase21b_quality_454m_report.json` is the obsolete `n=200` accuracy-only pilot and is excluded from the corrected table and figure.

No experimental value was changed beyond correcting the contradictory presentation. The downstream accuracy deltas are `+0.7`, `+2.2`, `+0.1`, and `-0.4` percentage points from the rounded table entries; they are treated as unstable/capacity-limited, not positive evidence. The signal-gradient paragraph now uses the supported gold-answer-NLL change and lists all accuracy deltas instead of cherry-picking `+2.2pp`.

## Local paired-delta audit for Q15

The calculations below use paired seeds and report effect sizes without significance claims.

**Primary I, PK@8K, EVQ+YaRN minus Geo+YaRN:** seed 42 `+38pp`, seed 123 `+42pp`, seed 7 `+36pp`. Source: `data/curated/table2_evq_yarn_454m_passkey_10pct.json`. The minimum paired advantage is `+36pp`; EVQ+YaRN is `100%` for all three seeds, while Geo+YaRN ranges from `58%` to `64%`.

**Primary III, relative PPL change, EVQ versus Geo:**

| Length | Seed 42 | Seed 43 | Seed 88 | Paired mean | Range |
|---|---:|---:|---:|---:|---:|
| 8K | +1.61% | +0.74% | +0.47% | +0.94% | +0.47% to +1.61% |
| 16K | -33.59% | -30.02% | -29.75% | -31.12% | -33.59% to -29.75% |
| 24K | -18.17% | -18.35% | -9.05% | -15.19% | -18.35% to -9.05% |
| 32K | -13.94% | -11.43% | -4.52% | -9.96% | -13.94% to -4.52% |

**Primary III, relative PPL change, EVQ+YaRN(s=4) versus Geo+YaRN(s=4):**

| Length | Seed 42 | Seed 43 | Seed 88 | Paired mean | Range |
|---|---:|---:|---:|---:|---:|
| 8K | +1.63% | +0.75% | +0.56% | +0.98% | +0.56% to +1.63% |
| 16K | -40.95% | -39.54% | -38.47% | -39.65% | -40.95% to -38.47% |
| 24K | -28.02% | -28.25% | -19.51% | -25.26% | -28.25% to -19.51% |
| 32K | -19.13% | -16.49% | -9.57% | -15.06% | -19.13% to -9.57% |

Source: `data/curated/table18_mla_3seed_aggregate.json`, promoted from the exact ignored source with SHA256 `1e44d30bb880e4b7427ae55bd7034782989152bd2afca9217495f9b8ece30953`; formula is `100 * (EVQ - Geo) / Geo` within each seed. We do not attach a p-value to `n=3` and do not make significance claims for one- or two-seed supporting rows.

## Question-by-question disposition and draft response

### F5-Q1 — QuALITY figure/table inconsistency

- Status: `DONE` locally; provenance caveat retained.
- Fix: replaced the obsolete accuracy visualization with a vector gold-answer-NLL figure generated by `scripts/figures/fig5_downstream_qa_nll.tex`; synchronized caption, protocol, checkpoint history, `n=2086`, and the exclusion of the old `n=200` pilot.
- Draft response: “The inconsistency was a real figure-provenance error. The old panel came from an earlier 200-sample accuracy-only pilot, whereas Table 21 summarizes the later full `n=2086` evaluation. We replaced the panel with Table 21's four gold-answer-NLL rows and now state the checkpoint path explicitly: 2K initialization, 4K continuation and QuALITY fine-tuning, then 4K/8K/16K evaluation. The pilot is excluded. The recovered aggregate reproduces every retained row and is recorded by source SHA256; accuracy remains near random and is not used as positive evidence.”
- 中文核对：主动承认图错和来源混用；不能把缺失的全量 JSON 说成已归档。

### F5-Q2 — Mischaracterized `+2.2pp` signal-gradient statement

- Status: `DONE`.
- Fix: verified the arithmetic, identified the review packet's apparent transcription, and removed the implication of a stable QA-accuracy gain. The paper reports gold-answer NLL as the probability-level signal and all four rounded accuracy deltas.
- Draft response: “The printed 8K-raw entries do derive `+2.2pp`: `26.8-24.6=2.2`; the review summary's `+0.2pp` appears to be a transcription error. The manuscript's real problem was interpretive: it singled out this one near-random row as the downstream endpoint. We corrected the paragraph to list all four deltas (`+0.7/+2.2/+0.1/-0.4pp`) and state that they have no stable direction. The retained probability-level observation is gold-answer NLL (`3.202→2.239`, `-30.1%` at 8K raw), not accuracy.”
- 中文核对：这是纠错，不包装为新结果。

### F5-Q3 — Primary II seeds

- Status: `PENDING-EXP`.
- Current boundary: Table 4 deliberately retains seed 42 for Geo/DAPE/EVQ; only learnable-tau is three-seed. The recovered L=256 raw/YaRN payload contains nine runs per evaluator but no DAPE row and does not silently upgrade the distinct `128→8K`, tau=5 protocol.
- Draft response placeholder: “We agree that the `128→8K` comparison is under-seeded. The submitted value remains a seed-42 diagnostic, and we do not infer variance from the separate L=256 sweep. [Insert seeds 137/256 with mean/std and paired deltas.] If replication is unavailable by rebuttal, we will re-tier this row as supporting and remove primary/headline language.”
- 中文核对：必须二选一：补齐同协议种子，或降级；不能拿相邻实验冒充复现。

### F5-Q4 — Learnable-tau failure and trajectory

- Status: `PARTIAL`.
- Fix: the paper now explains the advance-prediction/calibration boundary and waterbed mechanism. The tracked reports preserve final tau values `1.1391/1.1445/1.1383` for seeds `42/137/256`, but no per-step trajectory was recovered.
- Draft response placeholder: “Across the three reported seeds, learned tau ends at `1.1391/1.1445/1.1383`, while the fixed extrapolation-oriented allocation uses larger tau. This is consistent with weak extrapolation-directed signal from a flat in-range objective. We do not claim optimizer impossibility or trajectory-level convergence without the missing logs. [Insert the trajectory only if recovered or rerun.]”
- 中文核对：没有轨迹就不能写“实验证明优化器找不到”。

### F5-Q5 — Tuned geometric base

- Status: `PENDING-EXP`.
- Existing support: `data/curated/text_base_10k_500k_pilot.json` is raw-JSON-backed and shows the single-seed direction at both base 10K and 500K. It is supporting evidence only and does not replace a tuned Geo grid or the requested `c_pred` comparison.
- Draft response placeholder: “We agree that tuned training-time base is the nearest one-knob control. We will compare Geo at `b∈{10K,100K,2M}` against EVQ at `b=500K` under the same model/data/token budget and report both in-range and extrapolation metrics. The current collision and video base sweeps are complementary mechanism evidence, not substitutes for this control.”
- 中文核对：不要声称 Table 5 已经回答训练后的 base tuning。

### F5-Q6 — b=10K external validity

- Status: `PENDING-EXP`; decision rule documented.
- Fix: Appendix now includes a practitioner guide: use the bare rule only in the tested regime; at smaller base/larger length compute `c_pred(L,b)` and apply the forcing-branch diagnostic. The recovered raw-backed pilot is still single-seed and does not compare the bare rule against `c_pred`.
- Draft response placeholder: “We agree that all trained-text anchors use `b=500K`. The revision now marks the bare rule as regime-conditional and gives the explicit `c_pred(L,b)` fallback. [Insert matched `b=10K` bare versus corrected results.] Until this run exists, we make no trained-text generalization to the LLaMA-default base.”
- 中文核对：指南已解决误导风险，但外部有效性仍需实验。

### F5-Q7 — MLA d_eff convention

- Status: `PENDING-EXP`; claim boundary fixed.
- Fix: main text and appendix distinguish quantized `d_rope` from the calibrated transport convention `d_eff=d_head` and state that current results do not identify the optimal convention. The current three-seed values are portable in `data/curated/table18_mla_3seed_aggregate.json`; the report-backed d_rope=32/16 pilot is not a tau convention test.
- Draft response placeholder: “We agree this is the key sanity check for Primary III. The reported run tests allocation sensitivity at `tau=1.414`; it does not derive or identify the optimal MLA convention. [Insert `tau=0.354` and intermediate result, ideally replicated after the one-seed screen.] We have revised the claim accordingly.”
- 中文核对：先承认 convention 未消融，再给结果。

### F5-Q8 — Measure L_eff^J

- Status: `PENDING-EXP` (checkpoint-dependent evaluation, no retraining).
- Fix: limitations now make the missing measurement and the falsifiable `1/L` approximation prominent.
- Draft response placeholder: “We agree that this is a direct test of the transport approximation. The revision no longer lets the registered test read as evidence. [Insert kappa_att and L_eff^J from the existing checkpoints, including estimator details.] Until measured, `1/L` remains a falsifiable diffuse baseline.”
- 中文核对：这是理论链的缺口，不用“future work”轻描淡写。

### F5-Q9 — Autoregressive exact match and metric name

- Status: `DONE`.
- Fix: abstract, experiments, Figure 2, and Table 2 prominently call PK “teacher-forced NLL-gap retrieval.” We recovered the AR exact-match results from the primary run payload.
- Draft response: “We agree that the prior shorthand was too easy to read as generation accuracy. We now name the metric as teacher-forced NLL-gap retrieval everywhere it is prominent. Across the three seeds at 8K, EVQ+YaRN reaches 100.0% TF retrieval and 58.0% AR exact match (seed 42: 58.0%, seed 123: 18.0%, seed 7: 98.0%), whereas Geo+YaRN reaches 61.3% TF retrieval but drops to 0.0% AR exact match on all seeds. At 4K, EVQ+YaRN reaches 77.3% AR exact match and Geo+YaRN reaches 100.0% AR exact match. We report both metrics side-by-side to ensure full transparency while retaining NLL-gap retrieval as the mechanism-sensitive diagnostic.”
- 中文核对：改名已完成，真实AR数据已列出并对比，未将NLL-gap描述为生成精度。


### F5-Q10 — NTK anti-composition

- Status: `DONE` for the supported claim.
- Fix: limitations promote the concrete `32×` counterexample (EVQ4 331.4 vs Geo 198.1) and state that composition is scaler-specific. The mechanism is presented as a plausible interaction, not a proved taxonomy.
- Draft response: “We agree and have promoted this caveat. Our positive composition claim is specifically matched-scale YaRN. NTK-aware scaling re-warps the frequency table and can compound a strong cosh warp; at `L=256`, `32×`, EVQ4+NTK is worse than Geo+NTK. We therefore do not claim scaler-agnostic complementarity, and recommend validating any rescaler jointly with the training-time substrate.”
- 中文核对：不能给出没有消融支持的“兼容缩放器白名单”。

### F5-Q11 — 1B-token MLA reversal

- Status: `PENDING-EXP`; scope correction `DONE`.
- Fix: the evidence-tier table calls this a single-seed schedule-sensitivity check; the text no longer calls 500M progression evidence “durability.”
- Draft response placeholder: “We agree that the single-seed 1B row cannot establish durability. The 500M three-seed result is the primary claim; the 1B reversal is now labeled schedule sensitivity. [Insert replicated 1B curves.] We will not infer compute-optimal behavior from one seed, even though the composed row remains slightly favorable.”
- 中文核对：不把 `-2.5%` 包装成稳健优势。

### F5-Q12 — Realistic distance prior

- Status: `PENDING-EXP/ANALYSIS`; limitation `DONE`.
- Fix: limitations explicitly state that exact-kernel validation uses a uniform prior and that heavy-tailed/trained-model priors are unmeasured.
- Draft response placeholder: “The theorem is exact only for the stated surrogate, and our current functional validation uses a uniform distance prior. We have made this dependency explicit. [Insert power-law and empirical-prior refits/collision results.] We will describe any deviation from the cosh family rather than assume robustness.”
- 中文核对：理论条件要保留，不能称 realistic-D 鲁棒。

### F5-Q13 — Exponent 0.465 versus 0.500

- Status: `PARTIAL`.
- Fix: theory now separates advance prediction from calibration and states that the deployable claim is basin membership, not exact exponent identity; the new rank figure makes `3/9` exact, `8/9` top-3 legible. The underlying 99 sanitized run rows are now portable in `data/curated/phase16_99run_manifest.csv`, but they do not evaluate exponent 0.465 directly.
- Draft response placeholder: “The `-1/2` exponent is a structural operating rule, not an exact identification result; the Pearson-chi-square sensitivity calculation gives 0.465. Our empirical claim is that the resulting point lands in a flat basin, not that 0.500 is uniquely derived. [Insert matched evaluation of the 0.465 schedule if checkpoint access permits.] We have sharpened this distinction and made the rank counts readable.”
- 中文核对：不要用 top-3 替代审稿人要求的 0.465 成本；该数字仍待评估。

### F5-Q14 — Global versus per-head allocation

- Status: `PARTIAL`; conceptual scope `DONE`, pilot `PENDING-EXP`.
- Fix: limitations state that EVQ is one shared training-time initialization and that composition with learned per-head methods is untested. Related work already distinguishes CARoPE's learned per-head stage.
- Draft response placeholder: “EVQ addresses the initialization substrate; per-head specialization can still emerge during training and may be complementary. We do not claim a shared schedule is optimal. [Insert per-head jitter/CARoPE pilot if run.] The current attention-distance scatter supports head differentiation after shared initialization but is not a composition experiment.”
- 中文核对：只能论证 stage-distinctness，不能声称已组合。

### F5-Q15 — Paired deltas/statistics

- Status: `DONE` for existing three-seed primaries.
- Fix: paired deltas and ranges are tabulated above, and the Primary III seedwise source is now tracked as a raw-backed portable JSON. No formal significance claim is made at `n=3`; one- and two-seed rows stay supporting/exploratory.
- Draft response: “We agree that paired effects are more informative than an unqualified significance claim at `n=3`. For Primary I, the three PK@8K paired gains are `+38/+42/+36pp`. For Primary III, EVQ's paired 16K PPL changes are `-33.59/-30.02/-29.75%`, and EVQ+YaRN versus Geo+YaRN gives `-40.95/-39.54/-38.47%`. We report ranges and per-seed values and do not claim significance for smaller-n supporting rows.”
- 中文核对：最终 rebuttal 需视字数压缩，但数字和基线定义不能丢。

### F5-Q16 — Terminology, checkpoint, figures, practitioner guidance

- Status: `DONE`.
- Fix: removed “Habitable Zone” and internal phase labels from paper-facing text; clarified the QuALITY checkpoint and 4K status; replaced the unreadable rank panel; added a worked EVQ quantile/frequency example and a conditional practitioner guide.
- Draft response: “We completed the editorial pass: internal phase terminology was removed, the downstream checkpoint path and 4K protocol are explicit, the tau-rank figure was redrawn at print size, and the appendix now includes both a numerical schedule example and a decision guide for the bare rule, `c_pred`, MLA convention, and forcing diagnostic.”
- 中文核对：这是可直接声称完成的编辑修复。

### F5-Q17 — Video correction factor

- Status: `DONE` for scoping; derivation remains future work.
- Fix: the paper treats the 0.53 factor as a directional decomposition, not a predictive law, and bases transfer evidence on the base controls that do not depend on the constant.
- Draft response: “We agree that 0.53 should not be read as an independently predicted universal correction. We now frame Eq. 59 as a directional, post-hoc decomposition and leave RF-schedule derivation open. The modality-transfer evidence is the controlled base=1,000 and base-sweep behavior, which does not depend on fitting 0.53.”
- 中文核对：明确承认 post-hoc，避免‘future derivation’被误写为已有理论。

### F5-Q18 — LoRA threshold generality and matched Geo control

- Status: `PARTIAL`; hypothesis downgrade and code repair `DONE`, empirical control `PENDING-EXP`.
- Fix: the appendix calls `r≈d_head/2` a single-model calibration hypothesis, not general rank guidance. The LoRA stack now creates true native-geometric controls; infers head dimension/base from model config; handles old/new Transformers training-argument APIs; requires method-matched frequency artifacts; verifies every rotary module; records path-safe SHA-256 provenance; rejects stale checkpoints before reuse; and uses variant-safe RULER filenames.
- Draft response placeholder: “We agree that the threshold is calibrated on one LLaMA-3-8B setting and is not general guidance; the revision labels it as a testable hypothesis. We also found and fixed a control-path issue: the old ‘Geo’ wrapper used the EVQ midpoint construction at tau zero rather than native geometric endpoints. [Insert a fresh matched 8B Geo/EVQ+LongAlign rerun from the corrected code; use a 7B-family run as additional cross-model evidence, not as a replacement.] We will not reuse affected legacy comparisons.”
- 中文核对：这是重要代码问题，旧 Geo 结果不能继续当严格 matched control。

## Final response checks

- Every numeric insert must link to a tracked artifact or a newly packaged rebuttal artifact.
- Keep QuALITY as a correction and negative/limited downstream result; do not sell accuracy.
- Do not call the seed-42 Primary II contrast replicated until the exact protocol is rerun.
- Do not call NLL-gap PK autoregressive retrieval.
- Do not upgrade the MLA convention, 0.465 exponent, realistic prior, per-head, video correction, or LoRA threshold beyond the status above.
