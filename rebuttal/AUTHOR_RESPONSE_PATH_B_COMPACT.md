# EVQ-Cosh Compact Author Response — Asset-Grounded

日期：2026-07-10

状态：可交给 Fable5 压缩/复核的 rebuttal-only 短版。未修改已提交 PDF；“will correct” 指后续修订。

We thank the reviewers for the careful assessment. We retain the paper’s core conclusion at its stated scope: training-time RoPE frequency allocation is a substantive finite-spectral-budget design axis. The cosh family is the exact minimizer of the stated convex broadband surrogate; tau is a separate semi-analytic operating-point selector supported by an empirical basin, not a globally optimal transformer parameter. Primary I is a matched-scale substrate/range test, Primary II is a seed-scoped PE-dominant diagnostic, and Primary III is a three-seed scarce-channel MLA stress test.

On theory, the submission already separates shape and scale. The surrogate is not presented as the exact oscillatory RoPE kernel; it is validated functionally by 24–92% exact-kernel collision-score reductions over 12 configurations and then by trained PPL/PK outcomes. The practical tau claim is basin membership rather than exponent uniqueness: a 99-run sweep across nine (L,H,d_head) settings ranks the rule exactly first in 3/9, top-2 in 6/9, top-3 in 8/9, with every empirical optimum within 1.5x. The missing trained R_F and L_eff^J measurements remain mechanism tests, but the experiments use the exact cosh inverse-CDF schedule directly rather than a Taylor-truncated approximation.

On baseline fairness, Primary I asks whether the same YaRN operation has different leverage on two training-time substrates, not whether EVQ beats the best-tuned Geo+YaRN. Its 8K interaction is consistent in every seed: EVQ+YaRN exceeds Geo+YaRN by +38/+42/+36 pp. We also found an existing 151.9M text pilot at the LLaMA-style base 10K: EVQ improves PPL by -22.28% at 2K and -21.83% at 4K (versus -28.67%/-32.64% at base 500K). This supporting pilot does not replace a complete tuned-base sweep, but it rules out the stronger claim that the effect exists only at base 500K.

Primary II’s printed Geo/DAPE/fixed-EVQ contrast is seed 42, so the conclusion is the comparison in that tested protocol, not comprehensive DAPE dominance. The EVQ result is not a one-seed anomaly: fixed tau=5 has a separate three-seed PPL@8K aggregate of 335.7±1.7, learned tau converges to 1.1406±0.0034 across three seeds, and an L=256 three-seed sweep favors the predicted operating basin. Geo/DAPE still need matched seeds for a complete uncertainty comparison. The learned-versus-fixed gap is explained by objective mismatch, not optimizer noise: in-range PPL@128 is nearly flat across tau, while 8K PPL continues improving at larger fixed tau, so in-range gradients cannot observe the extrapolation utility.

For MLA, d_eff=d_head is an empirical operating convention rather than a theorem. The direct tau=d_rope/sqrt(L) test remains useful, but scarce-channel support is not limited to a cross-architecture comparison. An existing within-MLA seed-42 pilot changes d_rope from 32 to 16: EVQ changes 8K/16K PPL by -6.3%/-9.1% at d_rope=32 and -47.8%/-47.9% at d_rope=16. We treat the latter cautiously because the Geo baseline is weak; the 432M three-seed MLA aggregate remains the primary evidence.

The reviewers are correct about two submitted figures. Figure 8 used a superseded n=200 accuracy pilot under a Gold-NLL caption. The full n=2086 aggregate is the source of truth and shows that Table 21’s 8K-raw Geo accuracy should be 24.6% (513/2086), not 26.6%; the Gold-NLL values and conclusions are unchanged (-30.1% at 8K raw and -21.4% at 16K raw). Figure 9 used an approximately -81.2% value from a separate progressive-training setting, while the corresponding three-seed 454M FineWeb-Edu Table 20 row is -13.3%. Correcting that point changes the plotted magnitude, not Table 20’s directional observation that every individually scoped row shows a long-range improvement. We treat Table 20 as heterogeneous supporting consistency, not a controlled scaling law, and will correct both figures in a revision.

The 1B MLA reversal also does not constitute a token-only saturation test: relative to the primary 500M/8K three-seed anchor, it changes L_train to 4K as well as data/schedule and is single seed. We report it as schedule sensitivity, not robustness, while noting that EVQ+YaRN+FT remains -2.5%. PK is teacher-forced NLL-gap retrieval unless explicitly labeled AR exact; the paper does not equate it with exact generation. The LoRA row remains supporting because Base versus EVQ-LoRA does not isolate LongAlign/LoRA adaptation without a matched Geo+LoRA control.

Finally, the experiments are reproducible from the current repository. It provides a locked environment, public-data preparation, exact model configurations, canonical schedule code, training/evaluation entrypoints, curated expected aggregates, and focused tests. Historical checkpoints are useful provenance artifacts, but independent reproduction is defined by rerunning the public protocol, not by possessing the original machine.

In short, the audits identify real reporting and scope issues, but they do not overturn the three primary conclusions. The response should correct the stale figures and erratum, make the epistemic layers explicit, and use the existing 99-run, base-10K, learnable-tau, MLA-channel, and seedwise Primary-I assets rather than treating them as missing experiments.

## Fable5 final checks

1. Do not write that partial historical artifact loss makes the experiments non-reproducible.
2. Do not withdraw the three core conclusions; preserve their matched-scale/seed/protocol scope.
3. Keep Figure 8/9 as our reporting errors and 26.6%→24.6% as a disclosed erratum.
4. Do not upgrade the base/MLA pilots into full tuned or multi-seed primary controls.
5. Do not claim measured compute, memory, or latency savings.
