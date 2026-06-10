# EVQ-Cosh Compact Author Response: Path B

日期：2026-06-10

状态：当前默认可用短版。假设没有 Base / Geo+LoRA / EVQ-LoRA exact table，因此 LoRA 不作为主防线。

## Compact Draft

We thank the reviewers for the detailed comments. We agree that several rows should be read as mechanism evidence rather than production-scale validation. The central claim is narrower than universal long-context SOTA: EVQ-Cosh changes the training-time RoPE frequency substrate as a finite-spectral-budget allocation, and inference-time range scaling can act differently on that substrate. We will keep the paper framed as a PE mechanism study, not a deployment recipe or a replacement for YaRN/LongRoPE-style range scaling.

We made four concrete corrections/clarifications. First, we added total token budgets and seed scope for the primary anchors: Primary I uses 100M tokens at \(L_{\mathrm{train}}=2048\) with seeds 42/123/7; Primary II Table 4 is the 128-token, 15M-token seed-scoped DAPE-style diagnostic; Phase 11B is a separate 256-token, 100M-token supporting protocol; Primary III MLA uses 500M tokens with three seeds. Second, we relabeled the 1B MLA row as a schedule-sensitivity check, not evidence of robustness to training saturation. Third, we corrected the stale/mislabeled QuALITY figure so Figure 8 now plots Gold-answer NLL, consistent with Table 21. Fourth, we clarified that PK denotes teacher-forced NLL-gap retrieval unless explicitly marked AR exact.

On the LoRA row, we agree with the reviewer. The original Base vs EVQ-LoRA table is supporting and cannot by itself isolate EVQ frequency injection from LoRA/LongAlign adaptation. We therefore will not use it as primary evidence. It remains a post-hoc adaptation observation with explicit 8K cost; a matched Geo+LoRA control is the required attribution test. Without that matched table, the LoRA row should not carry the main rebuttal.

On training budget, we do not use Chinchilla-style token counts as overtraining thresholds. The narrower empirical point is that the EVQ signal is not explained by the simplest undertraining-only story: in the MLA progression, the long-range gap grows while the in-range cost shrinks, and the 750M continuation row shows a large 16K gap despite low in-range PPL. These results do not establish trillion-token from-scratch durability, but they motivate the scoped mechanism claim and the schedule-sensitivity limitation.

On YaRN and scalers, Table 2 is a matched-scale substrate comparison, not a tuned-scaler leaderboard. The question is whether the same range-scaling operation has different leverage on a Geo-trained versus EVQ-trained substrate. We will avoid claiming dominance over best-tuned Geo+YaRN, Dynamic NTK, LongRoPE, or LongRoPE2. The NTK-aware row is useful precisely because it shows that composition is scaler-dependent.

On Primary II, we will make the seed scope explicit. The DAPE-style comparison is a PE-dominant diagnostic stress test, not the sole statistical anchor of the paper. Geo, DAPE, and EVQ are reported under the retained seed-42 protocol, while the learnable-tau row reports a 3-seed mean/std. We do not present this as comprehensive DAPE or learned-PE dominance.

On the 1B MLA reversal, we agree that the row should not be described as saturation robustness. It is a single-seed schedule-sensitivity stress check in a scarce-channel MLA regime, not a same-configuration token-scaling ablation. The primary MLA evidence remains the 8K/500M, three-seed scarce-channel stress test. We will discuss the 1B row as a limitation motivating fixed-length continuation or stage-wise re-warp/adaptation.

On theory, we agree that shape and scale should be separated more explicitly. The cosh density is derived for the stated broadband surrogate, while \(\tau\) is an operating-point selector rather than a theorem of global optimality for trained attention. We will state the contribution as a shape-plus-calibrated-scale allocation rule. The learnable-\(\tau\) result is also informative: because the training objective only sees in-range loss, while extrapolation benefit is out of range and the in-range waterbed cost is immediate, gradient-based tau learning need not discover the extrapolation allocation.

On systems relevance, production-scale validation remains future work. The practical contribution is narrower: EVQ-Cosh is a zero-learned-parameter schedule change, the MLA experiment tests a scarce-rotary-channel regime relevant to compressed-attention designs, and the dead-channel audit exposes a diagnostic failure mode that can be applied independently of EVQ adoption. We do not use QuALITY accuracy as a primary benchmark claim; at 454M, accuracy is capacity-limited and near random, while Gold-answer NLL preserves the probability-space signal.

Across the reviews, the common concern is not whether frequency allocation is interesting, but whether the current evidence overstates its scope. We agree with that distinction. We therefore narrow the claim to training-time RoPE frequency allocation as a finite-spectral-budget design axis, add provenance/corrections for the main empirical confounds, and relabel supporting stress checks that were too broadly described.

## If More Space Exists

Add only if space permits:

> The progressive MHA row need not contradict the 1B MLA schedule-sensitivity result: the MLA setup has far fewer rotary channels, so K-dependent distortion terms make allocation mismatch more severe in the scarce-channel regime.

## Do Not Add Without New Evidence

- Matched Geo+LoRA attribution claim.
- Geo+YaRN tuned-scale sweep result.
- Primary I AR exact result.
- Primary II additional-seed mean/std.
- MLA \(\tau=d_{\mathrm{rope}}/\sqrt{L}\) ablation result.
- Learned-tau trajectory description.
