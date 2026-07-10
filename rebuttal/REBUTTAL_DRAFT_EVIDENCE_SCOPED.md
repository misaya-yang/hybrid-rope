# EVQ-Cosh Evidence-Scoped Rebuttal Draft

日期：2026-06-10

用途：把当前已经核实和已修正的内容组织成 NeurIPS rebuttal 可用草稿。本文不是最终提交版；它是“可发送段落 + 条件插入段落 + 禁用段落”的组合包。

Current status:

- Ready now: scope narrowing, PK metric clarification, token/seed provenance, 1B relabel, Figure 8/Table 21 correction, Primary II seed scope, MLA/DeepSeek scope.
- Conditional: Geo+LoRA control paragraph and Table 23 rewrite require exact numbers.
- Optional: Geo+YaRN scale sweep, Primary I AR exact, learned tau trajectory, MLA tau sanity.

Do not send any paragraph containing `[FILL]`, `[IF AVAILABLE]`, or table placeholders.

## 0. Global Opening

> We thank the reviewers for the detailed comments. We agree that several rows should be read as mechanism evidence rather than production-scale validation. The central claim is narrower than universal long-context SOTA: EVQ-Cosh changes the training-time RoPE frequency substrate as a finite-spectral-budget allocation, and inference-time range scaling can act differently on that substrate. We have clarified this scope, corrected presentation issues that could reduce trust, added token/seed provenance, and relabeled supporting rows that were too broadly described.

Use if space allows:

> In particular, PK denotes teacher-forced NLL-gap retrieval unless explicitly marked as autoregressive exact match; downstream accuracy at 454M is reported as a non-regression/supporting check rather than a primary benchmark claim.

Do not write:

- “EVQ is a universal long-context SOTA method.”
- “EVQ replaces YaRN, LongRoPE, DAPE, FIRE, or learned PE.”
- “All reviewer concerns are resolved by new experiments.”

## 1. Change Summary For AC

Ready-to-use bullet list:

> We make four concrete clarifications/corrections in this response. First, we report total token budgets and seed scope for the primary anchors: Primary I uses 100M tokens at \(L_{\mathrm{train}}=2048\) with seeds 42/123/7; Primary II Table 4 is the 128-token, 15M-token seed-scoped DAPE-style diagnostic; Phase 11B is a separate 256-token, 100M-token supporting protocol. Second, we treat the 1B MLA row as a schedule-sensitivity limitation rather than robustness to training saturation. Third, we acknowledge the stale Figure 8 and the Table 21 erratum and will correct them in a revision. Fourth, PK is teacher-forced NLL-gap retrieval unless separately marked AR exact.

Conditional LoRA bullet:

> [IF GEO+LORA EXACT NUMBERS ARE AVAILABLE] Fourth/Fifth, we added a matched Geo+LoRA control under the same checkpoint, data, rank, and adaptation schedule, so only the Geo+LoRA -> EVQ-LoRA difference is attributed to EVQ frequency injection.

If Geo+LoRA numbers are not available, do not include the conditional bullet.

## 2. R2 / Empirical Response

### 2.1 LoRA Confound

If exact Geo+LoRA numbers are NOT available:

> We agree that the original LoRA row is supporting and cannot by itself isolate EVQ frequency injection from LoRA/LongAlign adaptation. We therefore will not use the two-row Base vs EVQ-LoRA table as primary evidence. The row remains a post-hoc adaptation observation, and a matched Geo+LoRA control is the required attribution test.

If exact Geo+LoRA numbers ARE available:

> The reviewer is right that the original two-row table could not isolate frequency injection from LoRA/LongAlign adaptation. We added a matched Geo+LoRA control using the same pretrained checkpoint, data, rank, and 300-step schedule. We now report Base, Geo+LoRA, and EVQ-LoRA side by side. The Base -> Geo+LoRA change measures adaptation cost; only the Geo+LoRA -> EVQ-LoRA difference is attributed to EVQ frequency injection.

Conditional table:

| Model | Adaptation | 8K PPL | 16K PPL | 32K PPL | Seed scope |
| --- | --- | ---: | ---: | ---: | --- |
| Base | none | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |
| Geo+LoRA | same LoRA/LongAlign | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |
| EVQ-LoRA | same LoRA/LongAlign + EVQ | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |

Do not write:

- “LoRA proves EVQ scales to industrial pretraining.”
- “The full +30% 8K cost is EVQ’s cost.”
- “Geo+LoRA has no effect” unless the table proves it.
- “EVQ-LoRA beats default LLaMA context extension” unless a zero-training Geo scaler reference is included.

### 2.2 Training Budget / Undertraining

Ready-to-use:

> We added total token budgets and seed scope next to the primary anchors. Primary I uses 100M training tokens at \(L_{\mathrm{train}}=2048\) with seeds 42/123/7. Primary II Table 4 is a 128-token, 15M-token PE-dominant diagnostic; the later Phase 11B curves are a separate \(L_{\mathrm{train}}=256\), 100M-token supporting protocol and are not mixed with Table 4. Primary III MLA uses 500M tokens at \(L_{\mathrm{train}}=8192\) with three seeds.

Ready-to-use:

> We also avoid describing Chinchilla-style token counts as “overtraining.” The narrower empirical point is that the EVQ signal is not explained by a simple undertraining-only story: in the MLA progression, the long-range gap grows while the in-range cost shrinks; the 750M continuation row shows a large 16K gap despite low in-range PPL; and, if the controlled LoRA row is included, it tests a heavily pretrained checkpoint. These do not establish trillion-token from-scratch durability, but they rule out the simplest undertraining-only explanation.

Do not write:

- “9B tokens is overtraining.”
- “Training longer cannot remove EVQ.”
- “The 1B row proves robustness.”

### 2.3 YaRN Tuned-Scale Baseline

If no scale sweep is available:

> We agree that Table 2 is a matched-scale substrate comparison, not a tuned-scaler leaderboard. The question is whether the same range-scaling operation has different leverage on a Geo-trained versus EVQ-trained frequency substrate. We will make this scope explicit and avoid claiming dominance over best-tuned Geo+YaRN, Dynamic NTK, LongRoPE, or LongRoPE2.

If scale sweep is completed:

> [IF AVAILABLE] We added a Geo+YaRN scale sweep over `[FILL SCALE SET]`. The best Geo+YaRN result is `[FILL]`, while EVQ+YaRN is `[FILL]`. This tests whether the matched-scale result is explained by an obviously mistuned Geo baseline, without turning the paper into a tuned-scaler leaderboard.

Do not write:

- “EVQ beats tuned YaRN.”
- “Training-free scaling is irrelevant.”

### 2.4 PK vs Autoregressive Exact

If AR exact is not available:

> We clarify the metric definition throughout: PK denotes teacher-forced NLL-gap retrieval unless explicitly labeled AR exact. We use PK as a positional-encoding diagnostic endpoint and do not use it alone as evidence that the model can generate the key.

If AR exact is available:

> [IF AVAILABLE] We now report AR exact separately from teacher-forced PK. The original PK endpoint remains an NLL-gap diagnostic, not a claim of exact generation.

Do not write:

- “PK means exact retrieval.”
- “Teacher-forced metrics are equivalent to generation.”

### 2.5 Primary II Seed Scope

Ready-to-use:

> We will make the seed scope explicit. The DAPE-style comparison is a PE-dominant diagnostic stress test, not the sole statistical anchor of the paper. Geo, DAPE, and EVQ are reported under the retained seed-42 protocol, while the learnable-tau row reports a 3-seed mean/std over 42/137/256. We do not present this row as comprehensive DAPE or learned-PE dominance.

Do not write:

- “Primary II is fully 3-seed for all rows.”
- “The row proves broad learned-PE dominance.”

### 2.6 1B MLA Reversal

Ready-to-use:

> We agree that the 1B MLA row should not be described as robustness to training saturation. It is a single-seed schedule-sensitivity stress check in a scarce-channel MLA regime, not a same-configuration token-scaling ablation. We have relabeled it accordingly and discuss it as a limitation motivating fixed-length continuation or stage-wise re-warp/adaptation. The primary MLA evidence remains the 8K/500M, three-seed scarce-channel stress test.

Optional mechanism sentence:

> This also explains why the progressive MHA row need not contradict the MLA reversal: the MLA setup has far fewer rotary channels, and the K-dependent distortion terms make allocation mismatch more severe in scarce-channel regimes.

Do not write:

- “The reversal is noise.”
- “EVQ is robust to saturation.”
- “The K=16 argument quantitatively predicts the 1B PPL.”

### 2.7 Figure 8 / Table 21

Ready-to-use:

> We thank the reviewer for catching the stale/mislabeled QuALITY figure. Figure 8 used the superseded n=200 accuracy pilot under a Gold-NLL caption. The surviving n=2086 aggregate is the source of truth and also shows that the submitted Table 21 8K-raw Geo accuracy should be 24.6% (513/2086), not 26.6%. The Gold-NLL values and conclusions are unchanged. We do not use QuALITY accuracy as a primary claim and will correct the figure and table entry in a revision.

Do not write:

- “The reviewer misread the figure.”
- “Accuracy deltas and NLL deltas are equivalent.”
- “This is only an appendix issue.”

## 3. R1 / Theory Response

### 3.1 Shape vs Scale

Ready-to-use:

> We agree that the paper should separate the two levels more explicitly. The cosh density is derived for the stated broadband surrogate, while \(\tau\) is used as an operating-point selector rather than a theorem of global optimality for trained attention. We will revise the text to avoid implying a single unified optimum and to state that the contribution is the shape-plus-calibrated-scale allocation rule.

If measurement is available:

> [IF AVAILABLE] We also added a measurement-based check following the A.15 protocol: `[FILL]`. This ties the empirical active-band estimate to the selected scale without treating \(\tau\) as a learned parameter.

Do not write:

- “We derive the globally optimal tau.”
- “The surrogate is the trained-transformer objective.”

### 3.2 Learnable Tau Negative Result

Ready-to-use:

> This is an important negative result. The training objective only observes the in-range loss, while the extrapolation benefit is out of range and the in-range waterbed cost is immediate. Therefore gradient-based tau learning is biased toward the in-range basin and does not reliably discover the extrapolation allocation. We will add this interpretation and, if logs are included, the learned-tau trajectory.

If trajectory is available:

> [IF AVAILABLE] In our runs, learned tau `[FILL DRIFT/OSCILLATION/FLAT]`, consistent with the training-loss signal being weak or myopic for extrapolation allocation.

Do not write:

- “Learnable tau failure is irrelevant.”
- “The learned tau result validates the closed form automatically.”

### 3.3 NTK-Aware Composition

Ready-to-use:

> We agree and will sharpen the wording. The primary composition claim is matched-scale EVQ+YaRN substrate/range complementarity, not universal monotonic compatibility with every inference-time rescaler. The NTK-aware row is useful precisely because it shows that composition depends on the downstream scaler.

Do not write:

- “EVQ helps any scaler.”
- “NTK-aware is a bad baseline.”

## 4. R3 / Systems Response

### 4.1 Practical Relevance

Ready-to-use:

> We agree that production-scale from-scratch validation remains future work. The systems relevance is narrower: EVQ-Cosh is a zero-learned-parameter schedule change, the MLA experiment tests a scarce-rotary-channel regime relevant to compressed-attention designs, and the dead-channel audit exposes a diagnostic failure mode that can be applied independently of EVQ adoption.

Conditional LoRA sentence:

> [IF GEO+LORA TABLE IS FILLED] The controlled LoRA experiment adds an industrial-checkpoint anchor: under the same LoRA/LongAlign budget, the Geo control does not reproduce the long-range gain, while EVQ-LoRA does.

Do not write:

- “Production ready.”
- “Industrial scale proven.”

### 4.2 MLA Tau Convention / DeepSeek Scope

Ready-to-use:

> We agree and will clarify the wording. The MLA experiment is a production-relevant scarce-channel stress test, not a production-identical DeepSeek validation. The \(d_{\mathrm{eff}}=d_{\mathrm{head}}\) rule is an empirical operating convention for this architecture; we will either add the \(\tau=d_{\mathrm{rope}}/\sqrt{L}\) sanity ablation or mark it as a limitation.

Do not write:

- “This is the DeepSeek configuration.”
- “\(d_{\mathrm{eff}}=d_{\mathrm{head}}\) is derived by the theory.”

### 4.3 Downstream Benchmarks

Ready-to-use:

> We do not use QuALITY accuracy as a primary benchmark claim. At 454M parameters, accuracy is capacity-limited and near random; we report it as supporting/non-regression context while the probability-space NLL shows the directional effect. The primary evidence remains diagnostic mechanism tests rather than downstream leaderboard performance.

Do not write:

- “Benchmarks are irrelevant.”
- “QuALITY proves downstream improvement.”

## 5. AC Closing

Ready-to-use:

> Across the reviews, the common concern is not whether frequency allocation is interesting, but whether the current evidence overstates its scope. We agree with that distinction. We therefore narrow the claim to training-time RoPE frequency allocation as a finite-spectral-budget design axis, add controls or provenance for the main empirical confounds where available, and relabel supporting stress checks that were too broadly described. The revised paper will not claim universal long-context SOTA or production-scale validation.

If Geo+LoRA exact numbers are available:

> [IF AVAILABLE] The matched Geo+LoRA control is the main new attribution evidence: it separates LongAlign/LoRA adaptation from EVQ frequency injection under the same adaptation budget.

If Geo+LoRA exact numbers are not available:

> Without the matched Geo+LoRA table, the LoRA row remains supporting only and should not carry the main rebuttal.

## 6. Final Send Gate

Before sending:

- [ ] Remove every `[FILL]` and `[IF AVAILABLE]` block that lacks data.
- [ ] If Geo+LoRA exact table is absent, use the concession paragraph, not the control paragraph.
- [ ] If no AR exact result is available, keep PK as teacher-forced diagnostic only.
- [ ] If no Geo+YaRN scale sweep is available, keep the matched-scale scope paragraph.
- [ ] Keep 1B as schedule-sensitivity limitation.
- [ ] Keep QuALITY as NLL/supporting, not accuracy benchmark.
- [ ] Do not call EVQ a YaRN/LongRoPE/DAPE replacement.
