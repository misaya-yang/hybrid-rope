# EVQ-Cosh Author Response Draft: Path B

日期：2026-06-10

状态：当前证据可用版。此版本假设没有可引用的 Base / Geo+LoRA / EVQ-LoRA exact table，因此主动降级 LoRA。若之后补到 Geo+LoRA exact numbers，应改用 `AUTHOR_RESPONSE_PACKET.md` 的 Path A。

## Opening / AC Summary

We thank the reviewers for the detailed comments. We agree that several rows should be read as mechanism evidence rather than production-scale validation. The central claim is narrower than universal long-context SOTA: EVQ-Cosh changes the training-time RoPE frequency substrate as a finite-spectral-budget allocation, and inference-time range scaling can act differently on that substrate. We have clarified this scope, corrected presentation issues that could reduce trust, added token/seed provenance, and relabeled supporting rows that were too broadly described.

We made four concrete corrections/clarifications. First, we added total token budgets and seed scope for the primary anchors: Primary I uses 100M tokens at \(L_{\mathrm{train}}=2048\) with seeds 42/123/7; Primary II Table 4 is the 128-token, 15M-token seed-scoped DAPE-style diagnostic; Phase 11B is a separate 256-token, 100M-token supporting protocol. Second, we relabeled the 1B MLA row as a schedule-sensitivity check rather than robustness to training saturation. Third, we corrected the stale QuALITY figure so Figure 8 now plots Gold-answer NLL, consistent with Table 21. Fourth, we clarified that PK is teacher-forced NLL-gap retrieval unless separately marked AR exact.

## R2: Empirical Concerns

### LoRA Confound

We agree that the original LoRA row is supporting and cannot by itself isolate EVQ frequency injection from LoRA/LongAlign adaptation. We therefore will not use the two-row Base vs EVQ-LoRA table as primary evidence. The row remains a post-hoc adaptation observation with explicit 8K cost, and a matched Geo+LoRA control is the required attribution test.

This also changes how we discuss cost: the current table reports the full Base to EVQ-LoRA change, so it should not be described as EVQ-specific cost without a matched Geo+LoRA row. We will report the existing LoRA result only as supporting adaptation evidence and avoid using it to claim production-scale validation.

### Training Budget / Undertraining

We added total token budgets and seed scope next to the primary anchors. Primary I uses 100M training tokens at \(L_{\mathrm{train}}=2048\) with seeds 42/123/7. Primary II Table 4 is a 128-token, 15M-token PE-dominant diagnostic; the later Phase 11B curves are a separate \(L_{\mathrm{train}}=256\), 100M-token supporting protocol and are not mixed with Table 4. Primary III MLA uses 500M tokens at \(L_{\mathrm{train}}=8192\) with three seeds.

We also avoid describing Chinchilla-style token counts as overtraining. The narrower empirical point is that the EVQ signal is not explained by a simple undertraining-only story: in the MLA progression, the long-range gap grows while the in-range cost shrinks, and the 750M continuation row shows a large 16K gap despite low in-range PPL. These do not establish trillion-token from-scratch durability, but they rule out the simplest undertraining-only explanation.

### YaRN / Training-Free Scalers

We agree that Table 2 is a matched-scale substrate comparison, not a tuned-scaler leaderboard. The question is whether the same range-scaling operation has different leverage on a Geo-trained versus EVQ-trained frequency substrate. We will make this scope explicit and avoid claiming dominance over best-tuned Geo+YaRN, Dynamic NTK, LongRoPE, or LongRoPE2.

The NTK-aware row is also useful for this scope: it shows that composition is scaler-dependent. The primary composition claim is EVQ+YaRN under the matched scale we tested, not universal monotonic compatibility with every inference-time rescaler.

### PK vs Autoregressive Exact

We clarify the metric definition throughout: PK denotes teacher-forced NLL-gap retrieval unless explicitly labeled AR exact. We use PK as a positional-encoding diagnostic endpoint and do not use it alone as evidence that the model can generate the key. Where AR exact is available, we report it separately from teacher-forced PK.

### Primary II Seed Scope

We will make the seed scope explicit. The DAPE-style comparison is a PE-dominant diagnostic stress test, not the sole statistical anchor of the paper. Geo, DAPE, and EVQ are reported under the retained seed-42 protocol, while the learnable-tau row reports a 3-seed mean/std. We do not present this row as comprehensive DAPE or learned-PE dominance.

### 1B MLA Reversal

We agree that the 1B MLA row should not be described as robustness to training saturation. It is a single-seed schedule-sensitivity stress check in a scarce-channel MLA regime, not a same-configuration token-scaling ablation. We have relabeled it accordingly and discuss it as a limitation motivating fixed-length continuation or stage-wise re-warp/adaptation. The primary MLA evidence remains the 8K/500M, three-seed scarce-channel stress test.

This also explains why the progressive MHA row need not contradict the MLA reversal: the MLA setup has far fewer rotary channels, and K-dependent distortion terms make allocation mismatch more severe in scarce-channel regimes.

### Figure 8 / Table 21

We thank the reviewer for catching the stale/mislabeled QuALITY figure. The table values and text use gold-answer NLL; the figure panel was an older accuracy visualization and should not have been captioned as NLL. We have replaced the figure with a Gold-NLL plot consistent with Table 21, and we do not use QuALITY accuracy as a primary claim.

## R1: Theory Concerns

### Shape vs Scale

We agree that the paper should separate the two levels more explicitly. The cosh density is derived for the stated broadband surrogate, while \(\tau\) is used as an operating-point selector rather than a theorem of global optimality for trained attention. We will revise the text to avoid implying a single unified optimum and to state that the contribution is the shape-plus-calibrated-scale allocation rule.

### Learnable Tau

The learnable-tau result is an important negative result. The training objective only observes the in-range loss, while the extrapolation benefit is out of range and the in-range waterbed cost is immediate. Therefore gradient-based tau learning is biased toward the in-range basin and does not reliably discover the extrapolation allocation. We will add this interpretation and report the learned-tau trajectory if the logs are included.

### NTK-Aware Composition

We agree and will sharpen the wording. The primary composition claim is matched-scale EVQ+YaRN substrate/range complementarity, not universal monotonic compatibility with every inference-time rescaler. The NTK-aware row is useful precisely because it shows that composition depends on the downstream scaler.

## R3: Systems / Practicality Concerns

### Practical Relevance

We agree that production-scale from-scratch validation remains future work. The systems relevance is narrower: EVQ-Cosh is a zero-learned-parameter schedule change, the MLA experiment tests a scarce-rotary-channel regime relevant to compressed-attention designs, and the dead-channel audit exposes a diagnostic failure mode that can be applied independently of EVQ adoption.

### MLA Tau Convention / DeepSeek Scope

We agree and will clarify the wording. The MLA experiment is a production-relevant scarce-channel stress test, not a production-identical DeepSeek validation. The \(d_{\mathrm{eff}}=d_{\mathrm{head}}\) rule is an empirical operating convention for this architecture; we will either add the \(\tau=d_{\mathrm{rope}}/\sqrt{L}\) sanity ablation or mark it as a limitation.

### Downstream Benchmarks

We do not use QuALITY accuracy as a primary benchmark claim. At 454M parameters, accuracy is capacity-limited and near random; we report it as supporting/non-regression context while the probability-space NLL shows the directional effect. The primary evidence remains diagnostic mechanism tests rather than downstream leaderboard performance.

## Closing

Across the reviews, the common concern is not whether frequency allocation is interesting, but whether the current evidence overstates its scope. We agree with that distinction. We therefore narrow the claim to training-time RoPE frequency allocation as a finite-spectral-budget design axis, add controls or provenance for the main empirical confounds where available, and relabel supporting stress checks that were too broadly described. The revised paper will not claim universal long-context SOTA or production-scale validation.

Without the matched Geo+LoRA table, the LoRA row remains supporting only and should not carry the main rebuttal.

## Do Not Add To This Draft Unless New Evidence Exists

- Matched Geo+LoRA attribution claim.
- Geo+YaRN tuned-scale sweep result.
- Primary I AR exact result.
- Primary II additional-seed mean/std.
- MLA \(\tau=d_{\mathrm{rope}}/\sqrt{L}\) ablation result.
- Learned-tau trajectory description.
- A.15 measurement result.

## Final Language Guard

This draft intentionally avoids:

- EVQ is SOTA.
- EVQ replaces YaRN or LongRoPE.
- LoRA proves industrial-scale training.
- The 1B row proves saturation robustness.
- 9B tokens is overtraining.
- PK means exact retrieval.
- EVQ beats tuned YaRN.
- The QuALITY figure mismatch was a reviewer misunderstanding.
