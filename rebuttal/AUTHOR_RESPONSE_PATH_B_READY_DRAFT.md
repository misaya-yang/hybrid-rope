# EVQ-Cosh Author Response — Evidence-Safe Draft

日期：2026-07-10

状态：rebuttal-only 版本。当前没有修改已提交 PDF 或论文源码；文中的 “we will revise/correct” 指后续修订，不表示 reviewer-facing PDF 已被替换。本版本使用仓库现有代码、配置、curated aggregates 和可重跑入口，不依赖新增训练实验。

## Opening / AC Summary

We thank the reviewers for the detailed and constructive assessment. We agree that the current evidence supports a narrower claim than a universal long-context or deployment result. EVQ-Cosh identifies training-time RoPE frequency allocation as a finite-spectral-budget design axis. The cosh family is derived for the stated broadband surrogate, whereas the deployed \(\tau\) is a semi-analytic operating-point rule supported by an empirical basin, not a globally optimal parameter for trained attention. Likewise, the YaRN experiment is a matched-scale substrate comparison rather than a tuned-scaler leaderboard, and the MLA result is a scarce-rotary-channel stress test rather than a production-identical DeepSeek validation.

We also acknowledge two presentation errors in the submitted appendix. Figure 8 used a superseded \(n=200\) accuracy pilot under a Gold-NLL caption. The full-evaluation aggregate is \(n=2086\); it also shows that the submitted Table 21 8K-raw Geo accuracy should be 24.6% (513/2086), not 26.6%, while the reported Gold-NLL values and their conclusions are unchanged. Figure 9 mixed a different progressive-training result with the nominal cross-scale comparison. The \(n=2086\) aggregate and Table 20 are the corresponding sources of truth. Correcting the figures changes the stale visual values, not the Gold-NLL conclusion or Table 20’s directional observation. We will correct both figures in a revision.

## R1 — Theory and Foundations

### Shape versus scale

We agree with the reviewer’s central distinction, which is already explicit in Sec. 3. The variational argument derives the cosh allocation shape conditional on the broadband surrogate; the \(d_{\mathrm{eff}}/\sqrt{L}\) rule comes from a separate stiffness–utility model and empirical basin selection. The contribution is therefore a surrogate-derived allocation family with a semi-analytic operating rule, not a globally optimal transformer schedule. This distinction narrows the epistemic status of the scale; it does not change the exact conditional minimizer or the implemented inverse-CDF family.

### Surrogate grounding and Proposition 1

The constant-diagonal Green-kernel surrogate is a tractable approximation, not the exact RoPE collision kernel. The submission therefore validates it functionally: the derived allocation reduces the exact-kernel collision score by 24–92% over 12 configurations, followed by trained PPL/PK tests. Proposition 1 establishes the \(L^{-1/2}\) structure within the diffuse-baseline/Pearson-stiffness model; the practical claim is basin membership rather than exponent uniqueness. A 99-run sweep across nine \((L,H,d_{\mathrm{head}})\) settings places the rule exactly first in 3/9, top-2 in 6/9, top-3 in 8/9, with every empirical optimum within 1.5× of the prediction.

### Pure-tether regime and the waterbed interpretation

The experiment-specific \(R_F\) diagnostic and a trained forced-branch ablation are not reported, so we do not claim that the perturbative branch reduction is quantitatively exact at \(\tau=4\). Importantly, the experiments evaluate the exact cosh inverse-CDF schedule directly; they do not use a Taylor-truncated frequency table. This leaves a mechanism diagnostic open without invalidating the measured allocation result.

The “waterbed” inequality is likewise an entropy/divergence lower bound for moving away from uniform frequency allocation. It does not by itself prove a short-range-PPL versus long-range-PPL tradeoff. Any PPL interpretation is empirical and will be labeled as such.

### MLA operating convention

The reviewer is correct that \(d_{\mathrm{eff}}=d_{\mathrm{head}}\) is an empirical architecture convention in the reported MLA experiment, whereas only \(d_{\mathrm{rope}}\) dimensions are rotary. We do not claim this convention is derived by the theory, and the direct \(\tau=d_{\mathrm{rope}}/\sqrt L\) test remains open. However, the repository already contains a single-seed within-MLA channel-count pilot: at \(d_{\mathrm{rope}}=32\), EVQ changes 8K/16K PPL by -6.3%/-9.1%; at \(d_{\mathrm{rope}}=16\), by -47.8%/-47.9%, with the latter treated cautiously because its Geo baseline is weak. This is qualitative mechanism support; the 432M three-seed result remains the primary anchor.

## R2 — Experiments and Reproducibility

### Tuned-base and tuned-scaler controls

The current experiments do not claim superiority over a best-tuned Geo+YaRN configuration; Primary I asks the factorial matched-scale question of whether the same YaRN operation has different leverage on Geo-trained and EVQ-trained substrates. The repository also contains a text base pilot that directly addresses the stronger “only at 500K” concern: on a 151.9M, \(L=512\), 50M-token seed-42 run, EVQ at base 10K improves PPL by -22.28% at 2K and -21.83% at 4K (and at base 500K by -28.67%/-32.64%). This is supporting evidence rather than a complete tuned-base sweep, but it shows the direction survives at the LLaMA-style 10K base.

### Primary II seed scope

Geo, DAPE, and fixed EVQ in the printed Table 4 contrast use the retained seed-42 protocol; this is stated in the caption and the claim remains the result in that tested protocol, not comprehensive DAPE dominance. Replicated EVQ evidence already exists: fixed \(\tau=5\) has a three-seed PPL@8K aggregate of \(335.7\pm1.7\), learned \(\tau\) converges to \(1.1406\pm0.0034\) across three seeds, and a separate \(L=256\) three-seed sweep favors the predicted operating basin. Matched Geo/DAPE seeds would complete the uncertainty comparison, but the fixed-EVQ effect is not supported by seed 42 alone.

### Why learnable tau differs from fixed EVQ

The existing training record explains this asymmetry. Across seeds 42/137/256, learned tau converges tightly to \(1.1406\pm0.0034\), so the result is not optimizer noise. In-range PPL@128 varies by only about 1.6% over the fixed-tau sweep, while 8K PPL continues improving at larger tau. The training objective therefore identifies an in-range operating point but cannot observe the out-of-range utility that selects the stronger fixed allocation. This is the intended distinction between learning from in-range gradients and selecting a training-time substrate for extrapolation.

### Figure 8, Figure 9, and Table 21

The reviewer’s inconsistency finding is correct, not a misunderstanding. Figure 8 used the superseded \(n=200\) accuracy pilot under an NLL caption. The surviving \(n=2086\) aggregate reports 8K-raw Geo accuracy \(513/2086=24.59\%\), which rounds to 24.6%; the submitted 26.6% entry is an erratum. The Gold-NLL values are unchanged, and QuALITY accuracy is not used as a primary claim.

Figure 9 used the approximately \(-81\%\) value from a different progressive-training setting, whereas the Table 20 three-seed 454M FineWeb-Edu row reports \(-13.3\%\). The correction changes that plotted magnitude, not Table 20’s qualitative observation: every individually scoped row reports a long-range improvement. We treat the table as heterogeneous supporting consistency across tested settings, not as a controlled scaling law.

### Reproducibility

The repository provides the canonical schedule implementation, locked environment, public-data preparation, model configurations, training/evaluation entrypoints, curated expected aggregates, and focused tests. In particular, it contains the Primary I protocol and seedwise values, a complete 99-run manifest, the \(n=2086\) QuALITY aggregate, and the MLA three-seed aggregate, together with scripts that rerun the corresponding protocols from public data. This is a reproducible path, not merely a claim that historical checkpoints exist: curated outputs are audit targets, while the code and configuration are the route for independent reruns.

### PK metric, 1B reversal, and LoRA scope

PK denotes teacher-forced NLL-gap retrieval unless explicitly labeled autoregressive exact match. We use it as a positional diagnostic and do not equate it with exact generation.

The 1B MLA row is not a token-only continuation of the primary anchor: it changes \(L_{\mathrm{train}}\) from 8K to 4K as well as data/schedule and is single seed. It therefore cannot establish that the 500M/8K three-seed effect vanishes with more training. We report the raw reversal as schedule sensitivity, while noting that EVQ+YaRN+FT remains -2.5%; the primary 432M result and its three-seed conclusion are unchanged.

The Base-versus-EVQ-LoRA comparison cannot isolate frequency injection from LoRA/LongAlign adaptation and carries a nontrivial 8K cost. Without a matched Geo+LoRA row, it remains supporting post-hoc evidence and is not used for an industrial-scale or causal attribution claim.

## R3 — Applications, Clarity, and Impact

### Application evidence and when to use EVQ

The paper does not use QuALITY accuracy, RULER, or teacher-forced PK as a deployment leaderboard. Its practical evidence is the zero-learned-parameter initializer, the 3-seed matched-scale YaRN interaction, and the 3-seed MLA anchor. A useful operating boundary also emerges: EVQ is most relevant when the geometric allocation leaves scarce or inactive channels. The video base sweep and text base-10K pilot show both that the mechanism is diagnosable and that the text effect is not confined to base 500K.

### Exposition and systems impact

We agree that the mechanism is simpler than the current exposition. A revision will consolidate overlapping dimension notation and separate exact results, model-conditional arguments, and empirical heuristics more visibly. The practical systems claim is implementation simplicity and zero learned parameters; the paper does not report a measured FLOP, memory, latency, or wall-clock reduction.

## Closing

Across the reviews, the common issue is scope and reporting trust, not the existence of the cosh solution or the direction of the three primary results. The repository assets let us answer several concerns more strongly than the submitted presentation did: a 99-run operating-basin validation, a base-10K text pilot, tightly replicated learnable-tau behavior, a within-MLA channel-count pilot, and Primary-I paired seed evidence. We therefore retain the core conclusions at their stated scope, correct the two stale visualizations and Table 21 erratum, and reserve tuned-scaler/full-seed upgrades for reproducible follow-up runs.

## Do Not Claim Without New Evidence

- EVQ beats a tuned geometric base or best-tuned Geo+YaRN.
- Primary II is multi-seed for Geo/DAPE/EVQ.
- MLA validates \(d_{\mathrm{eff}}=d_{\mathrm{head}}\) theoretically.
- The 1B row proves saturation robustness.
- PK is autoregressive exact retrieval.
- LoRA isolates EVQ or proves production-scale efficacy.
- QuALITY proves downstream task improvement.
- Figure 9 establishes a controlled scaling law.
- EVQ reduces measured FLOPs, memory, latency, or wall-clock time.

## Repository Assets Used In This Response

Available and safe to cite:

- `data/curated/quality_454m_full_eval.json`
- `data/curated/phase16_99run_manifest.csv`
- `data/curated/table18_mla_3seed_aggregate.json` (aggregate only)
- `data/curated/table2_evq_yarn_454m_passkey_10pct.json`
- `data/curated/text_base_10k_500k_pilot.json`
- `data/curated/mla_channel_count_125m_pilot.json`
- `data/curated/learnable_tau_128tok_evidence.json`
- `docs/overview/REPRODUCE.md` and `docs/overview/DATA_PREPARATION.md`
