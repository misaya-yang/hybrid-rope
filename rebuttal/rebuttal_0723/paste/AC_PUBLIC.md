The metareview names three conditions that could change the recommendation; we take them in turn.

**1. Technical novelty over FMRoPE.** The RoPE table \\(\\omega_i = b^{-u_i}\\), \\(u_i = 2i/d\\), admits three points of intervention: the realized vector (PI, YaRN, LongRoPE), the base \\(b\\) (NTK-aware, ABF, FMRoPE), and the exponent grid \\(u_i\\). Every method here modifies that table; what differs is which quantity is the variable. To our knowledge, EVQ-Cosh is the first closed-form, zero-parameter training-time rule for that grid, derived from a stated variational objective.

FMRoPE is a base-selection rule: the base is set to \\(L_{\\rm train}\\) at training and \\(L_{\\rm target}\\) at inference; its abstract describes the effect as "shifting the band toward lower frequencies", with exponents uniform in i. Oka et al. call that dependence a practical limitation in their §6.3 and list adaptive schemes as future work.

We should have cited Oka et al.; the revision will. We do not claim the dead-channel observation — §2 credits it to Barbero et al. (2025), the source Oka et al. cite.

**2. Direct controlled comparison.** With both sampled extrema, log span, initialization, token order, optimizer, budget and 32 anchors fixed, and only the 30 interior frequencies varied, Cosh allocation improves fixed-range OOD NLL by **0.478/0.205/0.113** at 512/1K/2K, winning **32/32, 27/32 and 22/32** anchors. A separate three-seed exact-range factorial supports the same allocation-axis conclusion across bases 500K/1M and head dimensions 32/64/128, favoring non-geometric allocation in 10/12 cases. The endpoints and log span — what a base or range choice sets — are fixed, so the effect is not attributable to either.

We implement the published §6.1 rule; under retargeting it obtains lower NLL, as a target-aware range method should, and ours is positive where it isolates our variable. On YaRN, submitted Table 3 gives **61±3% vs 100±0%** NLL-gap retrieval at 8K on Geo vs EVQ substrates, one fixed transform, three seeds.

**3. Stronger evaluation.** *Larger models.* EVQ is used two ways: as the frequency table when training from scratch, or swapped into a released checkpoint and recovered with a short LoRA fine-tune. The submission showed the second at 8B: LLaMA-3 LoRA PPL goes **176.3 → 21.5** at 16K — unusable to usable — and 1942.5 → 104.3 at 32K. On OLMo-2 (1.485B), matched arms took the same LoRA chain, backward passes capped at 4,096 tokens; by exact answer-string equality plus terminal EOS, Native/EVQ is 95/100 vs 100/100 at 4K, **18/100 vs 98/100** at 8K, **0/100 vs 60/100** at 16K — same checkpoint, only the EVQ table replaced.

*Stronger benchmarks.* Over 13 RULER families under identical 4K supervision, OLMo-2 Native/EVQ macro is 82.16/37.51% at 4K, 0.08/21.29% at 8K, 0/6.13% at 16K; at 8B under 8K supervision, 16K macro goes **0.295 → 14.03%**.

*Real downstream tasks.* With the LoRA restricted to Q/K, Native/EVQ exact match on 200 held-out 2WikiMultiHop prompts is 22.0/21.5% at 4K, **0/17.5%** at 8K, **0/4.0%** at 16K, from 0% parents — near-identical in-window, only EVQ carrying past it. Under that restriction EVQ's RULER macro is **42.44/31.63%** at 4K/8K.

*Base frequencies.* A three-seed suite at base 1M, d_head = 128, held out from the rule's calibration and untuned, gives tail-NLL deltas **−0.802/−0.433/−0.212** at 1K/4K/16K, seeds agreeing, intervals excluding zero.

*Diverse architectures.* A three-seed 432M MLA study (PPL@16K **138.8 → 95.6**) and video-DiT.

**Attribution controls.** We ran the matched analytic-schedule ablation the metareview names: operator, protocol and evaluation fixed, every arm zero-parameter, so parameter count and tuning budget are identical by construction. EVQ-Cosh − Geo mean tail-NLL deltas are **−0.256/−0.305/−0.223/−0.238** at 1K/2K/4K/8K over three agreeing seeds, with span-matched uniform and deformation-matched arms included. Cosh follows exactly from the surrogate, whose coefficients are fitted from the exact kernel; allocations from it cut the exact-kernel collision score by **24–92% over 12 configurations**. On τ, the derivation gives the scaling form τ ∝ d_eff/√L_train but not the constant; over 21 configurations the selected τ stays within **0.75×–1.5×** of it.

**What we ask.** Two boundaries constrain the claim: under retargeting FMRoPE obtains lower NLL, limiting us to a fixed-range advantage; and EVQ trails in-window on RULER macro while matching or leading on the generation and 2Wiki endpoints above. Oka et al. report the same ordering for base selection, FMRoPE underperforming conventional RoPE in short contexts, and the Q/K-only ablation shows adaptation scope contributes to the in-window gap. Neither changes the question the metareview poses: whether training-time exponent allocation is a distinct, independently identifiable design variable whose effect persists across base, head dimension, architecture and scale. On the evidence above, we believe it is.
