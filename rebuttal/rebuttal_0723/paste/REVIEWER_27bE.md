Thank you for separating the exact statement from the finite-τ operating regime. You identified three confounds — allocation shape against parameterization, against optimization effort, and against the operating-point rule — and we built experiments that take them apart one at a time. We answer your five requests in order:

1. We separate the exact statement from the operating rule into four explicit links.
2. Across 21 configurations in two studies the empirically selected τ never leaves a 0.75×–1.5× window around the formula value, and the allocation effect survives a deformation-matched non-Cosh schedule.
3. It transfers untuned to a held-out base configuration, and across bases 500K/1M and head dimensions 32/64/128.
4. It survives a three-level control ladder fixing parameter count, tuning budget, reference grid, deformation magnitude and spectral range.
5. The larger pre-specified run is 1.485B.

**1. The approximation chain and finite τ.** You are right that the small-τ analysis does not determine a finite optimum, and that the submission let the two rationales sit too close together. We now separate four links: (i) Cosh is the stationary density of C_app — exact given that surrogate; (ii) the quadratic surrogate, the discrete grid and the pure-tether branch are modeling choices; (iii) the analysis yields the *scaling form* τ ∝ d_eff/√L_train; (iv) the constant is not supplied by the analysis and is set to one as a zero-search convention.

Link (iii)/(iv) is where your τ ≈ 4 observation lands. The local stationarity condition for the deployment objective gives θ* = [−g/h]₊, where g and h depend on the data's relative-distance structure, head feature usage, the base, K, the span convention, deployment lengths and the objective. The expansion fixes how θ* scales with d_eff and L_train but does not evaluate those functionals, so the best multiplier may move with the configuration while the scaling form stays stable.

On the surrogate: α and β are fitted from the exact kernel on the deployed channel grid, not chosen for convenience, and the submitted validation is functional — allocations from C_app cut the exact-kernel collision score by **24–92% across 12 configurations**, and a collision-only oracle search on the exact kernel reaches comparable reductions.

**2. Independently tuned τ and matched non-Cosh schedules.** Two studies test this directly.

The main study is a 99-run sweep over nine configurations — L_train ∈ {256,512,1024} crossed with d_head ∈ {32,64,128}, five τ values each at pilot, then three seeds on the retained arms. Against midpoint-Geo the formula wins 7/9 configuration means and 18/27 paired runs, mean weighted extrapolation NLL improvement 0.0133. Against the pilot-selected neighboring τ, on the two seeds not used in selection, the formula wins 3/9 — but the selected neighbor lies within 0.75×–1.5× of the formula value in all nine. The rule is not a per-configuration point optimum; it consistently places the operating point in the relevant basin.

The exact-range factorial you requested extends this under a stricter control: three seeds over two bases (500K/1M), two training lengths (256/1024) and three head dimensions (32/64/128), giving 12 configurations in which every schedule shares the same sampled extrema and log span. The spectral range is identical by construction; only interior spacing differs.

On tuned τ, the best pre-specified multiplier is 0.75× in 2 configurations, 1.00× in 4 and 1.25× in 6; the two more widely swept boundary cases land at 1.5× and 0.75×. Across both studies, the selected τ never leaves a **0.75×–1.5×** window around the formula value. On alternative schedules, a deformation-matched exponential is statistically indistinguishable from Cosh (+0.0007 NLL, exact sign-flip p = 0.836). What survives both ablations is the allocation axis: pre-specified 1.25× Cosh and the matched exponential improve weighted OOD NLL over Geo by 0.0121 and 0.0106, favoring non-geometric allocation in 10/12 and 9/12 configurations.

Two positive conclusions survive both ablations: c = 1 is a zero-search default that reliably lands in the relevant basin, and the allocation axis stays identifiable across independently tuned Cosh and matched non-Cosh schedules. The revision will state it that way.

**3. Base, head dimension, and lineage.** Two suites answer this.

A three-seed suite at base 1M with d_head = 128, held out from the submitted rule's calibration, gives mean tail NLL deltas of **−0.802/−0.664/−0.433/−0.287/−0.212** at 1K/2K/4K/8K/16K, all three seeds agreeing and paired 95% intervals excluding zero; the in-domain cost at 512 is +0.069. The exact-range factorial of (2) also crosses bases 500K/1M and head dimensions 32/64/128 under the pinned-range construction. The allocation effect is therefore not confined to base 500K or d_head = 64; on the held-out configuration the formula value was applied untuned, so that gain cannot be attributed to the calibration settings. These are mechanistic controls at 50.9M–151.9M; scale is answered in (5).

One clarification on seeds, since it bears on this concern: the 432M MLA study is three-seed — its table caption reads "432M, d_rope = 32, 3-seed mean ± std", and entries carry three-seed deviations (GEO 138.8±5.5 vs EVQ 95.6±4.1 at 16K). The seed-42 designation applies to the PE-dominant diagnostic, not the MLA result.

**4. Allocation shape versus parameterization and optimization effort.** This is the confound you identified most sharply, and we ran the experiment you specified: keep the positional operator unchanged and vary only fixed frequency schedules — geometric, EVQ and alternative analytic allocations — at three levels of control, each removing one more confound.

Level 1 removes learned capacity and tuning budget: same operator, same protocol, every arm a fixed zero-parameter analytic schedule, so parameter count and optimization effort are identical by construction. Across three seeds, EVQ-Cosh minus Paper-Geo mean tail NLL is **−0.256/−0.305/−0.223/−0.238** at 1K/2K/4K/8K, all three seeds agreeing. Span-matched uniform and RMS-deformation-matched power and exponential arms are included; EVQ also beats the uniform arm.

Level 2 removes the reference-grid and deviation-magnitude questions: all arms use the native Std-RoPE endpoint grid, share the same span, and every non-Geo arm has the same RMS deformation from native Geo (0.2557), so no arm is simply farther from geometric. Native EVQ minus native Std-RoPE is −0.113/−0.149/−0.207/−0.190/−0.099 NLL at 512/1K/2K/4K/8K, with every paired 95% interval excluding zero and all three seeds agreeing at every length.

Level 3 removes any implicit base/range change: the exact-range factorial of (2).

Together these give the attribution test you requested: with operator, protocol, parameter count, tuning budget, reference grid, deviation magnitude and spectral range all fixed, changing only the interior allocation still changes trained NLL. Allocation is a separately identifiable design axis. Alternative analytic schedules can be competitive at some lengths, so Cosh's role is the closed-form inverse-CDF operating point derived from the surrogate, not a universal optimum.

On the row you raise, we accept your conclusion rather than defend it: a comparison against a 32-parameter learned baseline cannot separate allocation shape from parameterization and optimization effort, whatever tuning budget it received. We no longer rest the allocation claim on that row. The three levels above are the direct test you specified — positional operator unchanged, only fixed schedules varying, parameter count and tuning budget identical by construction — and that is where the attribution now sits.

**5. A larger pre-specified run.** We trained OLMo-2 (1,484,916,736 parameters, d_head = 128, base 500K) from the public step-0 initialization for exactly 2,097,152,000 counted tokens, under a protocol fixed in advance and sharing the pinned recipe, data-order prefix and evaluation rows with the released geometric checkpoint.

Geo/EVQ PPL on the same 128 document-disjoint PG-19 documents is 161.19/167.45 at 4K, 163.88/156.87 at 8K and 182.73/159.64 at 16K. Paired document-level bootstrap 95% intervals exclude zero at every length: the EVQ−Geo NLL delta is +0.0381 [+0.0332,+0.0428] at 4K, −0.0437 [−0.0494,−0.0380] at 8K and −0.1351 [−0.1420,−0.1281] at 16K, with 122/128 and 126/128 documents favoring EVQ at 8K/16K. These intervals measure held-out document sampling, not training-seed uncertainty.

The 4K regression is the in-window cost of reallocating a finite channel budget toward long range. We call this a same-initialization, same-recipe comparison: the released geometric checkpoint came from the upstream distributed trainer and our EVQ branch from a single-GPU loop on the same recipe. It is one trajectory rather than a multi-seed estimate.

**Revision plan.** (1) §3.3 and Table 1 will separate the four links and present c = 1 as a zero-search default. (2) A new appendix will report the exact-range factorial with the tuned-τ and matched-exponential comparisons and the 0.75×–1.5× containment. (3) A second will report the three-level ladder and the base/head factorial. (4) The PE-dominant table will no longer support the allocation-shape claim, which rests on the ladder. (5) A third will report the 1.485B result with its single-trajectory and trainer boundaries in line. (6) §6 will state that the evidence supports the allocation axis and a closed-form zero-search operating point, not universal optimality of Cosh or of the τ default.

**What your review changed.** The three confounds you named are the reason the attribution now rests on a fixed-schedule ladder instead of a comparison against a learned baseline, and the reason we present the τ rule as a basin default instead of an optimum. Both are more precise statements than the submission made, and both hold under the controls you specified.
