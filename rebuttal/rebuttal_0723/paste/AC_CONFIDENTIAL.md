We are writing privately because the two weaknesses identified in the metareview — limited novelty and insufficient empirical validation — both trace to Reviewer zWsa's review, and three of that review's premises can be checked directly against the submission. We are not asking that any review be discounted. We are asking that the novelty determination rest on facts the committee can verify.

**1. The mechanism attributed to FMRoPE is not the mechanism FMRoPE uses.**

The review states that Oka et al. proposed "a method that enables long-context extrapolation without additional parameters by modifying the allocation of frequency bands", and concludes from this that our findings (1) and (2) substantially overlap.

FMRoPE sets θ_train = L_train and θ_infer = L_target, so that ω_i(θ) = θ^(-2i/d). As that paper describes it, this shifts the frequency band toward lower frequencies. It changes where the band sits. It does not change how the K channels are distributed inside it: the exponents remain uniform in i, and the normalized geometric order is preserved exactly.

EVQ-Cosh holds the nominal base fixed and replaces the exponent map itself, u_i → φ_τ(u_i), where φ_τ is the closed-form inverse CDF of the stationary density of a stated variational objective. Moving a band and redistributing channels within a band are different operations on the frequency table, and only the second is what we claim.

This is not a matter of interpretation, and it is testable. If our effect were reducible to where the band sits, pinning the band would remove it. We pinned the highest sampled frequency, the lowest sampled frequency and the log span — together with initialization, token order, optimizer, budget and all 32 evaluation anchors — and changed only the 30 interior frequencies. Out-of-distribution NLL improves by 0.478/0.205/0.113 at 512/1K/2K, winning 32/32, 27/32 and 22/32 anchors. A three-seed factorial reproduces the direction across bases 500K/1M, training lengths 256/1024 and head dimensions 32/64/128, with the pre-specified 1.25× Cosh and deformation-matched exponential arms favoring the non-geometric allocation in 10/12 and 9/12 structural configurations. Every quantity a base or range choice can set is held constant, so the effect is not attributable to band placement.

We would also note that being parameter-free does not identify a mechanism: position interpolation and NTK-aware scaling are parameter-free as well. The property is shared across most of this literature.

**2. The scale premise does not match the submitted version.**

The review states that "the model sizes evaluated in this paper are too small" and asks for validation "on models of at least approximately 1B to 7B parameters". The submitted version reports an 8B LLaMA-3 LoRA evaluation in Appendix D, Table 23, a 750M continuation with strict autoregressive retrieval in Table 12, a 432M scarce-channel MLA study in Table 18, and 129M/382M bidirectional 3D-RoPE video-DiT experiments in Table 14.

The metareview's "concentrated on relatively small models … diverse architectures" language appears to follow from this premise. The response additionally provides matched Native/EVQ comparisons at 1.485B and 8B: 0/100 versus 69/100 strict autoregressive first-number exact at 8K on OLMo-2, with a second independently trained EVQ seed at 67/100, and 16K RULER macro of 0.295% versus 14.03% on LLaMA-3-8B.

**3. The dead-channel and related-work premises.**

The review states that the existence of dead channels "has already been demonstrated" by Oka et al. and that the related work "appears insufficiently surveyed". We have never claimed the dead-channel observation as a contribution: §2 of the submitted version credits channel inequality to prior work, citing Barbero et al. (2025) on high/low-frequency specialization and Resonance RoPE on critical frequencies, alongside HoPE, FoPE, CARoPE, Clipped RoPE and the multimodal RoPE line.

**What we accept.** The missing Oka et al. citation is real and it is ours. It should have been cited and directly compared, and the revision does both. That omission is the legitimate core of this review, and we do not minimize it.

**What we ask.** The metareview offers three conditions. On technical novelty, the distinction is a formula-level difference between moving a band and redistributing within one. On the direct controlled comparison, the exact-range control isolates precisely that variable and is positive. On stronger evaluation, the metareview names matched analytic-schedule ablations, sensitivity to the allocation parameter and base frequency, and results on a stronger benchmark or larger model; the response contains a three-level zero-parameter schedule ladder, two τ studies spanning 21 configurations, a 500K/1M exact-range factorial, and matched 1.485B and 8B results on strict generation and 13-family RULER.

We ask that the novelty determination rest on whether the training-time exponent allocation is a distinct and consequential design variable — which the exact-range control was built to test — rather than on a characterization of FMRoPE's mechanism that the source does not support.
