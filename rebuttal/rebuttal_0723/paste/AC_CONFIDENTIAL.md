We are writing privately because four premises underlying Reviewer zWsa's assessment can be checked against the submission and the source that review cites. We ask only that the determination rest on facts the committee can verify.

In one line each: (1) the mechanism the review attributes to FMRoPE is not the one FMRoPE's own abstract and §6.1 describe; (2) the substrate-dependence finding in submitted Table 3 is neither claimed nor tested there; (3) "the model sizes are too small" does not match Tables 12, 14, 18 and 23; (4) the dead-channel observation the review credits to Oka et al. is credited by them to Barbero et al. (2025), which we cite. The missing FMRoPE citation is real, and it is ours.

**1. The mechanism attributed to FMRoPE.** The review describes Oka et al. as proposing "a method that enables long-context extrapolation without additional parameters by modifying the allocation of frequency bands", and concludes that findings (1) and (2) overlap.

Every method in this literature modifies the frequency table — that is what makes it a RoPE method. What separates them is which quantity in \\(\\omega_i = b^{-u_i}\\) is treated as the design variable.

FMRoPE is a choice of base frequency. Its abstract states that "setting θ to the training length shifts the band toward lower frequencies and improves extrapolation", and §6.1 reads "In FMRoPE, we set the RoPE base equal to the training context length: \\(\\theta = L_{\\rm train}\\)".

The band moves. What does not move is how the channels sit inside it: the exponents stay uniform in i and the normalized geometric order is preserved exactly. EVQ-Cosh holds the base fixed and replaces the exponent map, \\(u_i \\to \\varphi_\\tau(u_i)\\), the closed-form inverse CDF of the stationary density of a stated variational objective. Moving a band and redistributing channels within one are different operations.

If the effect were reducible to where the band sits, pinning the band would remove it. We pinned both sampled extrema and the log span — the quantities a base or range choice sets — together with initialization, token order, optimizer, budget and all 32 anchors, and varied only the 30 interior frequencies. OOD NLL improves by **0.478/0.205/0.113** at 512/1K/2K, winning **32/32, 27/32 and 22/32** anchors, and a three-seed factorial reproduces it across twelve configurations.

The rules also differ in what they require: FMRoPE is defined by \\(\\theta_{\\rm train} = L_{\\rm train}\\) and \\(\\theta_{\\rm infer} = L_{\\rm target}\\), selecting a range for a declared deployment length, while EVQ-Cosh sets the training grid and introduces no target. Oka et al. note this dependence: their §6.3 calls needing the target length at inference "a practical limitation" and names adaptive schemes as future work. Parameter-free is not mechanism-specific: PI and NTK-aware scaling are too.

**2. The cited overlap does not cover our Table 3 finding.** The review lists finding (3) — that EVQ-Cosh is not in conflict with inference-time scaling such as YaRN — among the claims overlapping Oka et al. That paper does run YaRN experiments, but we found no claim or experiment there on our question: whether changing the training-time table changes what the same fixed inference-time transform can recover. That is what Table 3 measures.

**3. The scale premise.** The review states that "the model sizes evaluated in this paper are too small" and asks for validation on "at least approximately 1B to 7B parameters". The submission reports an 8B LLaMA-3 LoRA evaluation (Appendix D, Table 23) where extrapolated PPL goes from **176.3 to 21.5** at 16K and **1942.5 to 104.3** at 32K, alongside a 750M continuation, a three-seed 432M MLA study and 129M/382M video-DiT results. These were available at review time; the response adds matched comparisons at 1.485B and 8B.

**4. The dead-channel attribution.** The review states that dead channels have "already been demonstrated" by Oka et al. We never claimed that observation: §2 credits channel inequality to prior work, citing Barbero et al. (2025) and Resonance RoPE. Oka et al. credit it to the same source — their §3 opens by investigating "the frequency band identified by Barbero et al. (2025)".

Oka et al. also distinguish overlapping observations from distinct scientific questions: "While our visual observations overlap with Barbero et al. (2025), the core scientific questions and contributions differ substantially." That is the standard we ask the committee to apply here.

**What we accept.** The missing Oka et al. citation is real and it is ours. It should have been cited and directly compared; that omission is the legitimate core of this review.

**What we ask.** That the novelty determination rest on whether training-time exponent allocation is a distinct and consequential design variable — which the exact-range control was built to test — rather than on a characterization of FMRoPE's mechanism that the source does not support.
