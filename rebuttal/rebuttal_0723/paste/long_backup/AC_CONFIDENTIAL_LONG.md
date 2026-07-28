We are writing privately because three premises underlying Reviewer zWsa's novelty and scale assessment can be checked directly against the submission and against the source that review cites. We are not asking that any review be discounted. We are asking that the novelty determination rest on facts the committee can verify.

In one line each: (1) the mechanism the review attributes to FMRoPE is not the one FMRoPE's own title, abstract, Figure 1 caption and §6.1 describe; (2) the statement that our models are too small does not match Tables 12, 14, 18 and 23 of the submitted version; (3) the dead-channel observation the review credits to Oka et al. is credited by Oka et al. to Barbero et al. (2025), which the submission cites. The missing FMRoPE citation itself is real, and it is ours.

**1. The mechanism attributed to FMRoPE is not the mechanism FMRoPE uses.**

The review states that Oka et al. proposed "a method that enables long-context extrapolation without additional parameters by modifying the allocation of frequency bands", and concludes from this that our findings (1) and (2) substantially overlap.

FMRoPE is defined as a choice of base frequency, and that paper says so in its own words in four places: the title ("Base Frequency and Context Length Shape the Interpolation–Extrapolation Trade-off"); the abstract ("setting θ to the training length shifts the band toward lower frequencies and improves extrapolation"); the Figure 1 caption ("FMRoPE sets the maximum base frequency to match the maximum sequence length in pre-training"); and §6.1 ("In FMRoPE, we set the RoPE base equal to the training context length: θ = L_train"). Section 6 introduces the method as answering "What is the impact on model performance when the frequency band is shifted toward lower frequencies during pretraining?"

The band moves. What does not move is how the K channels are distributed inside it: the exponents remain uniform in i, and the normalized geometric order is preserved exactly.

EVQ-Cosh holds the nominal base fixed and replaces the exponent map itself, u_i → φ_τ(u_i), where φ_τ is the closed-form inverse CDF of the stationary density of a stated variational objective. Moving a band and redistributing channels within a band are different operations on the frequency table, and only the second is what we claim.

The two rules also differ in what the practitioner must supply. FMRoPE's rule is defined by two declared quantities, θ_train = L_train and θ_infer = L_target: it selects a range for a deployment length that has to be known and fixed in advance. EVQ-Cosh sets the training-time exponent grid and introduces no deployment target at all. That is the practical content of the level distinction — a range method is target-aware by construction and an allocation method is not — and it is why the two are not two settings of one method. It is also why we report, rather than contest, the fact that FMRoPE obtains lower NLL once it is given its target.

That asymmetry is not our characterization of their method; it is theirs. §6.3 states: "While FMRoPE demonstrates strong extrapolation, the requirement of knowing the target sequence length at inference time poses practical limitations. Future work should explore dynamic or adaptive schemes for adjusting θ based on observed context." A training-time allocation that fixes the grid before deployment and introduces no target at all is one answer to the question they leave open. We would read that as the two works being complementary rather than overlapping.

This is not a matter of interpretation, and it is testable. If our effect were reducible to where the band sits, pinning the band would remove it. We pinned the highest sampled frequency, the lowest sampled frequency and the log span — together with initialization, token order, optimizer, budget and all 32 evaluation anchors — and changed only the 30 interior frequencies. Out-of-distribution NLL improves by 0.478/0.205/0.113 at 512/1K/2K, winning 32/32, 27/32 and 22/32 anchors. A three-seed factorial reproduces the direction under the same pinned-range construction across twelve structural configurations. Every quantity a base or range choice can set is held constant, so the effect is not attributable to band placement.

We would also note that being parameter-free does not identify a mechanism: position interpolation and NTK-aware scaling are parameter-free as well. The property is shared across most of this literature.

**2. The scale premise does not match the submitted version.**

The review states that "the model sizes evaluated in this paper are too small" and asks for validation "on models of at least approximately 1B to 7B parameters". The submitted version reports an 8B LLaMA-3 LoRA evaluation in Appendix D, Table 23, in which extrapolated PPL goes from 176.3 to 21.5 at 16K and from 1942.5 to 104.3 at 32K — an 8B model moving from unusable to usable at 2× and 4× the training length. It also reports a 750M continuation with strict autoregressive retrieval in Table 12, a 432M scarce-channel MLA study in Table 18, and 129M/382M bidirectional 3D-RoPE video-DiT experiments in Table 14. These were available at review time.

The response additionally provides matched Native/EVQ comparisons at 1.485B and 8B. On OLMo-2, under an identical downstream chain and judged by literal equality of the complete answer string followed by a terminal EOS token, Native/EVQ is 18/100 versus 98/100 at 8K and 0/100 versus 60/100 at 16K. A separate strict first-number retrieval experiment on the same model gives two independently trained EVQ seeds at 69/100 and 67/100 against Native 0/100. Under identical physical-4K supervision over all 13 RULER families, 8K official macro is 0.08% versus 21.29% and 16K is 0% versus 6.13%. On LLaMA-3-8B, 16K RULER macro is 0.295% versus 14.03%.

**3. The dead-channel and related-work premises.**

The review states that the existence of dead channels "has already been demonstrated" by Oka et al. and that the related work "appears insufficiently surveyed". We have never claimed the dead-channel observation as a contribution: §2 of the submitted version credits channel inequality to prior work, citing Barbero et al. (2025) on high/low-frequency specialization and Resonance RoPE on critical frequencies, alongside HoPE, FoPE, CARoPE, Clipped RoPE and the multimodal RoPE line.

Oka et al. attribute that observation to the same source we do. Their §2 states that "Barbero et al. (2025) revealed that there are 'frequency bands' … they also revealed that pretraining while replacing the low-frequency dimension RoPE with NoPE does not change performance", and their §3 opens "We first investigate the frequency band identified by Barbero et al. (2025)." On this premise we cited the work they cite.

One further observation. Oka et al. themselves distinguish overlapping observations from distinct scientific questions, in these terms: "While our visual observations overlap with Barbero et al. (2025), the core scientific questions and contributions differ substantially." That is the same standard we are asking the committee to apply here, and under it FMRoPE and EVQ-Cosh remain distinct contributions.

**On the RULER premise.** The review asks whether the method improves RULER. It does beyond the training length and costs in-window, and we report both. That ordering is the established behaviour of this design space, not an artefact of our method: the Section 6 takeaway of Oka et al. reads "Matching θ to the training length … improves extrapolation but hurts interpolation, and this trade-off persists under position interpolation such as YaRN."

**What we accept.** The missing Oka et al. citation is real and it is ours. It should have been cited and directly compared. That omission is the legitimate core of this review, and we do not minimize it.

**What we ask.** We ask that the novelty determination rest on whether the training-time exponent allocation is a distinct and consequential design variable — which the exact-range control was built to test — rather than on a characterization of FMRoPE's mechanism that the source does not support.
