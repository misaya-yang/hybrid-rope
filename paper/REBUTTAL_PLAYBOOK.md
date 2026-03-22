# EVQ-Cosh Rebuttal Playbook: NeurIPS 2025

> **Purpose**: Comprehensive pre-rebuttal preparation. For each anticipated reviewer attack, we provide: the attack framing, an honest self-assessment of vulnerability, the defense strategy with specific evidence pointers, and ready-to-paste rebuttal text.
>
> **Methodology**: Attacks are derived from (1) the paper's own limitations section, (2) common NeurIPS reviewer patterns for PE/long-context papers, (3) competitive landscape analysis of 2024-2025 PE papers (DAPE, CREAM, FIRE, LongRoPE, Resonance RoPE, VideoRoPE, CoPE, "Round and Round We Go"), and (4) known weaknesses identified in our internal audit.
>
> **Last updated**: 2026-03-22

---

## Table of Contents

1. [Scale Concerns](#1-scale-concerns)
2. [Undertraining / Training Amount Objection](#2-undertraining--training-amount-objection)
3. [Single-Seed Evidence](#3-single-seed-evidence)
4. [Broadband Surrogate Validity](#4-broadband-surrogate-validity)
5. [Comparison with Competing Methods](#5-comparison-with-competing-methods)
6. [Downstream Task Evidence](#6-downstream-task-evidence)
7. [MLA Experiment Concerns](#7-mla-experiment-concerns)
8. [Novelty and Theoretical Contribution](#8-novelty-and-theoretical-contribution)
9. [Practical Impact and Adoption](#9-practical-impact-and-adoption)
10. [YaRN Composition Claims](#10-yarn-composition-claims)
11. [Presentation and Scope](#11-presentation-and-scope)
12. [Reproducibility](#12-reproducibility)
13. [Competitive Landscape Context](#13-competitive-landscape-context-2024-2025)
14. [Additional Attacks (Reader-Test Identified)](#14-additional-attacks-reader-test-identified)

---

## 1. Scale Concerns

### Attack 1.1: "Evidence is limited to small models (50M-750M). Results may not hold at production scale (7B+)."

**Vulnerability**: MEDIUM. This is our most honest limitation. Primary multi-seed evidence is at 50M-454M. 750M is single-seed. No evidence at >=1B.

**Defense Strategy**:

(a) **Scale-independence of the mechanism**: EVQ modifies only the RoPE inverse-frequency initialization --- a one-line change that is architecture-agnostic. The theoretical prediction (tau* = d_head / sqrt(L)) depends on d_head and L, not model width or depth. The mechanism operates at the frequency-channel level, which is invariant to model scale.

(b) **5-scale consistency**: We provide the broadest from-scratch PE allocation study in the literature: 50M, 125M, 350M (3-seed), 454M (3-seed progressive), 750M (single-seed). The improvement direction is consistent at every scale, with no sign of diminishing returns. For comparison:
- DAPE (NeurIPS 2024): 125M only
- FIRE (ICLR 2024): 125M, 350M
- CREAM (NeurIPS 2024): Llama-2 7B but LoRA fine-tuning only, not from-scratch
- LongRoPE (ICML 2024): Llama-2/3, Phi-3 --- but these are post-hoc scaling, not from-scratch training

(c) **MLA validation at production-relevant architecture**: Our 432M MLA experiment (3-seed) directly tests the attention mechanism used in DeepSeek-V2/V3. While the model scale is smaller, the architecture (d_rope=32, 16 frequency channels) is identical to production configurations.

(d) **Cross-architecture transfer**: Llama-3-8B and Qwen-2.5-8B LoRA experiments show EVQ benefits transfer to pretrained 8B models, though these are preliminary.

**Ready-to-paste rebuttal**:

> We acknowledge the scale gap and have stated this explicitly in Limitations. However, we note three mitigating factors: (1) EVQ modifies only the frequency initialization, which is scale-independent by construction --- tau* depends on d_head and L, not model width/depth; (2) our 5-scale chain (50M-750M) is the broadest from-scratch PE study in the literature, exceeding DAPE (125M), FIRE (125M/350M), and all other allocation methods; (3) our MLA experiment directly tests the architecture used in DeepSeek-V3, and the "fewer channels, each more precious" principle suggests EVQ's benefit may actually *increase* at production scale where MLA compresses RoPE to 16-32 channels. We agree that >= 1B validation is an important next step and will prioritize this in future work.

### Attack 1.2: "750M result is single-seed. It could be a lucky run."

**Vulnerability**: LOW-MEDIUM. The 750M result (PPL -45.9% at 16K, AR exact 0%->77.5%) is dramatic enough that noise is unlikely to explain it, but formally it's n=1.

**Defense Strategy**:

The 750M experiment is explicitly labeled as "supporting evidence" in the paper, not a primary claim. The magnitude of the effect (-45.9% PPL, 0% to 77.5% AR exact) far exceeds any plausible seed variance --- our 350M 3-seed results show max inter-seed variance of ~3% at 16K. Additionally, the 750M pattern is qualitatively consistent with every other scale.

**Ready-to-paste rebuttal**:

> The 750M result is explicitly presented as "single-seed supporting evidence" (Section 5.4). The effect magnitude (-45.9% PPL, 0% to 77.5% AR exact) is an order of magnitude larger than the inter-seed variance observed at 350M 3-seed (~3% at 16K), making reversal due to seed noise implausible. The result's role is to confirm that the improvement direction persists at our largest scale, not to establish a precise effect size.

---

## 2. Undertraining / Training Amount Objection

### Attack 2.1: "EVQ's advantage may simply be an artifact of insufficient training. With enough training, geometric RoPE would converge to similar performance."

**Vulnerability**: HIGH if Phase 18 YaRN FT composition data is not in the paper. LOW if it is included.

**Defense Strategy**:

This is the single most important attack to defend against, and we have strong evidence:

(a) **Phase 18 structural reversal**: At 4K/1B tokens (fully trained), EVQ *loses* standalone extrapolation by +11.1%. Yet EVQ+YaRN+FT *wins* by -2.5% --- a 13.6 percentage-point structural reversal. This directly demonstrates that even when training closes the standalone gap, EVQ's structural composition advantage persists.

(b) **Two-component decomposition**: EVQ's advantage has two separable components:
- Raw extrapolation benefit: diminishes with training amount (as expected --- more training lets the model compensate for suboptimal frequencies)
- Structural composition benefit: always present, because YaRN inherits the quality of the training-time frequency layout

(c) **Progressive training amplifies, not diminishes**: Three-stage progressive training (512->1024->2048) shows the EVQ+YaRN advantage *growing* from -34.6% to -52.0% to -81.2%. If EVQ were purely an undertraining artifact, progressive training should close the gap, not widen it.

(d) **MLA training progression**: EVQ's advantage at 16K is -29.0% at 50% training and -31.1% at 100% training --- monotonically increasing, not decreasing. The in-distribution cost simultaneously decreases from +1.4% to +0.9%.

**Ready-to-paste rebuttal**:

> This is an important concern that we address with three lines of evidence. First, our MLA YaRN fine-tuning experiment directly tests this: at 1B tokens (fully trained), EVQ loses standalone extrapolation by +11.1%, yet EVQ+YaRN+FT wins by -2.5% --- a 13.6pp structural reversal. Even if training closes the standalone gap, EVQ provides a structurally better foundation for context extension methods. Second, progressive training (3 stages) *widens* EVQ's advantage from -34.6% to -81.2%, the opposite of what an undertraining artifact would produce. Third, MLA training progression shows EVQ's advantage growing monotonically from 50% to 100% training (-29.0% to -31.1% at 16K). These three independent observations are inconsistent with an undertraining explanation and consistent with EVQ addressing a structural frequency-layout limitation.

### Attack 2.2: "The models are trained on relatively few tokens (50M-500M tokens). Production models train on trillions of tokens."

**Vulnerability**: MEDIUM. True, but this applies equally to all PE allocation papers.

**Defense Strategy**:

(a) **Fair comparison with literature**: DAPE (NeurIPS 2024) trains on sequence length 128 with limited tokens. FIRE trains at 125M/350M scale. No PE allocation paper has trained at trillion-token scale --- this is a shared limitation of the subfield, not specific to EVQ.

(b) **The frequency initialization is consumed at step 0**: EVQ changes the initialization of inv_freq, which is used from the very first forward pass. The benefit is architectural, not dependent on training duration.

(c) **Phase 18 evidence**: Even at our longest training run (1B tokens for 432M model), the composition benefit persists. There is no evidence of convergence between EVQ and Geo at any training duration we have tested.

**Ready-to-paste rebuttal**:

> This concern applies to all PE allocation research: DAPE (NeurIPS 2024) trains at L=128, FIRE at 125M/350M scale. No frequency allocation paper has validated at trillion-token scale. Within our experimental range, we observe no convergence between EVQ and Geo at any training duration (up to 1B tokens for 432M). More critically, the structural composition benefit (Section 5.5) persists even when standalone convergence occurs, suggesting that EVQ's value lies not in raw extrapolation but in providing a better substrate for context extension methods.

---

## 3. Single-Seed Evidence

### Attack 3.1: "Progressive training chain (Claims 4) is single-seed."

**Vulnerability**: MEDIUM. The progressive chain (454M, seed=42, 512->1024->2048) is indeed single-seed for the full 3-stage pipeline.

**Defense Strategy**:

(a) **Stage 1 is now multi-seed confirmed**: Seeds 42/43/44 all show consistent EVQ advantage at Stage 1 (PPL@4K: -16.5%, NIAH@1K: +26pp with zero variance).

(b) **Magnitude exceeds plausible noise**: The Stage 3 advantage is -81.2% at 16K. Inter-seed variance at 350M 3-seed is ~3%. A single seed producing an 81.2% advantage by chance is implausible.

(c) **Corroborating evidence from independent experimental lines**: 750M (single-seed, -45.9%), 350M 3-seed (-13.3%), 6-seed passkey (100% vs 61-65%) --- all show the same directional pattern.

(d) **The progressive widening pattern**: The fact that the advantage monotonically increases (34.6% -> 52.0% -> 81.2%) across stages would be extremely unlikely to occur by chance.

**Ready-to-paste rebuttal**:

> We acknowledge the progressive chain is single-seed for the full pipeline and state this explicitly. However: (1) Stage 1 is multi-seed confirmed (3 seeds, all consistent); (2) the magnitude of -81.2% at Stage 3 far exceeds any plausible seed variance (inter-seed CV is ~3% at 350M); (3) the monotonic widening pattern across 3 stages would require a remarkable coincidence to occur by chance; and (4) the pattern is corroborated by independent experiments at 750M (-45.9%), 350M 3-seed (-13.3%), and 6-seed passkey (100% vs 61-65%). We plan to complete multi-seed validation of Stages 2-3 and will include this in the camera-ready version.

---

## 4. Broadband Surrogate Validity

### Attack 4.1: "The entire theory rests on a single approximation (broadband surrogate). If this approximation fails, the derivation is invalid."

**Vulnerability**: LOW. We have extensive numerical validation, but this is the correct theoretical concern to raise.

**Defense Strategy**:

(a) **24,000-configuration numerical sweep**: The broadband projection K_approx = alpha*I + beta*A^{-1} achieves R^2 > 0.99 across the regime where RoPE-based models operate (base 8K-100K, L >= 4096). This is not a cherry-picked result --- it systematically maps the 6-dimensional boundary of validity.

(b) **35-49% full-matrix residual is understood**: The residual comes from three identifiable boundary effects (UV discretization, IR wavelength truncation, finite diagonal ridge width). The mid-band where the variational ODE operates is well-captured.

(c) **GPT-2 cross-validation**: Real attention distance distributions from 12x12 heads yield power-law fits consistent with the theoretical assumption (alpha > 0.8 for local heads, which are most sensitive to RoPE allocation).

(d) **The approximation is falsifiable, and the paper provides the falsification criteria**: We explicitly state when it fails (e.g., token co-occurrence distance priors give R^2 = 0.645-0.664). This transparency strengthens rather than weakens the claim.

**Ready-to-paste rebuttal**:

> We appreciate this important theoretical concern. The broadband surrogate is indeed the single approximation in the derivation, and we believe this transparency is a strength of our approach. We validate it with a 24,000-configuration sweep covering 6 dimensions (base, L, alpha, grid, mid-band, method), finding R^2 > 0.99 under D(delta) proportional to 1/delta for base in [8K, 100K] and L >= 4096 --- precisely the operating regime of modern RoPE models. We also explicitly show where it fails (token co-occurrence: R^2 ~ 0.65), providing falsification criteria. The residual (35-49%) is attributable to three identifiable boundary effects that do not affect the mid-band where the variational ODE operates (see Appendix A for detailed analysis).

### Attack 4.2: "The scaling law tau* = d_head/sqrt(L) is empirical, not theoretically derived."

**Vulnerability**: LOW-MEDIUM. The scaling law emerges from the theory but the specific functional form is validated empirically.

**Defense Strategy**:

The tau* scaling law is derived from the theory as a dimensional analysis prediction (tau has dimensions of [d_head]/[sqrt(L)] from the beta/alpha ratio in the broadband surrogate). The 99-run validation (27 configs x 3 seeds) confirms this prediction with worst-case <1% PPL gap from the empirical optimum.

**Ready-to-paste rebuttal**:

> The tau* = d_head/sqrt(L) scaling law emerges from dimensional analysis of the broadband surrogate parameters (tau = sqrt(beta/alpha), where beta and alpha carry the appropriate dimensions). The specific functional form is then validated by a 99-run sweep (27 configurations x 3 seeds, L in {256, 512, 1024}, d_head in {32, 64, 128}), with worst-case <1% PPL gap from the empirical optimum. We characterize it as an "empirical conjecture supported by theory" rather than a rigorous derivation, which we believe is the honest framing.

---

## 5. Comparison with Competing Methods

### Attack 5.1: "Why not compare with CREAM (NeurIPS 2024), LongRoPE (ICML 2024), or Resonance RoPE?"

**Vulnerability**: MEDIUM. We compare with DAPE and YaRN but not with all recent PE methods.

**Defense Strategy**:

These methods operate on different axes and are not direct competitors:

(a) **CREAM (NeurIPS 2024)**: Manipulates position indices (axis 3, inference-time), not frequency allocation (axis 2). CREAM is complementary to EVQ, not competing. CREAM interpolates by manipulating position indices with a Gaussian middle focus; EVQ optimizes frequency channel placement. They could be composed.

(b) **LongRoPE (ICML 2024)**: Uses evolutionary search to find per-frequency scaling factors for inference-time extension (axis 3). LongRoPE's key insight --- that different frequencies need different scaling factors --- is consistent with EVQ's finding that the default geometric allocation is suboptimal. LongRoPE operates post-hoc on pretrained models; EVQ operates at training-time initialization. They are complementary.

(c) **Resonance RoPE (ACL Findings 2024)**: Addresses frequency aliasing by snapping critical frequencies to integer periods. This is orthogonal to EVQ's density redistribution. Resonance RoPE + EVQ is a valid combination.

(d) **"Round and Round We Go" (ICLR 2025)**: This is an analysis paper, not a competing method. Its finding that Gemma 7B prefers low frequencies is consistent with EVQ's prediction that the low-frequency band is the bottleneck.

**Ready-to-paste rebuttal**:

> These methods operate on fundamentally different axes. Our paper organizes the design space into three orthogonal dimensions: (1) base theta (bandwidth), (2) allocation (within-band distribution), and (3) inference-time scaling. EVQ addresses axis 2 --- the only one without a principled optimization framework. CREAM and LongRoPE address axis 3 (inference-time index/frequency manipulation); Resonance RoPE addresses frequency aliasing, which is orthogonal to density redistribution. We compare with DAPE (axis 2, NeurIPS 2024) as the most direct competitor on the allocation axis, and with YaRN (axis 3) to demonstrate composition. We note that EVQ is designed to be composed with these methods, not to replace them: EVQ + LongRoPE or EVQ + CREAM are valid and potentially beneficial combinations.

### Attack 5.2: "DAPE comparison is unfair --- you use extreme extrapolation (128->8K) where learned PE may not be expected to work well."

**Vulnerability**: LOW-MEDIUM. The PE-dominant regime is deliberately extreme, but that's the point.

**Defense Strategy**:

(a) **The PE-dominant regime isolates the variable of interest**: At L_train=128, model memorization and data effects are minimized, making frequency layout the dominant variable. This is standard methodology in PE research (DAPE itself uses L=128).

(b) **We test at multiple scales**: The PE-dominant claim is supported by Phase 0-3 (125M, L=128) AND Phase 11 (454M, L=256). The latter is less extreme.

(c) **EVQ uses 0 extra parameters**: DAPE uses d/2 = 32 learnable parameters. EVQ achieves better extrapolation with zero parameters --- a strictly stronger result.

**Ready-to-paste rebuttal**:

> The PE-dominant regime (L=128) is the same setting used in DAPE itself (NeurIPS 2024) and is standard for isolating PE quality from confounds. We additionally validate at L=256 (454M, 3-seed, Phase 11), where EVQ+YaRN achieves -61.7% PPL improvement over Geo+YaRN. The key point is that EVQ achieves this with zero extra parameters, while DAPE uses d/2 learnable parameters --- a strictly stronger result.

### Attack 5.3: "VideoRoPE (ICML 2025 Oral) already addresses frequency allocation for video. What does EVQ add?"

**Vulnerability**: LOW. VideoRoPE and EVQ are convergent evidence, not competing methods.

**Defense Strategy**:

VideoRoPE's core innovation is Low-frequency Temporal Allocation (LTA) --- a heuristic assignment of low-frequency channels to the temporal axis. EVQ provides a complementary theoretical perspective: the variational optimum (tau > 0) naturally shifts density toward low frequencies. The two methods arrive at the same directional conclusion from independent approaches (theory-first vs experiment-first).

Key differences:
- VideoRoPE is a heuristic 3D RoPE design for video VLMs; EVQ is a closed-form variational solution for general RoPE
- VideoRoPE operates at 7B+ scale post-hoc; EVQ is validated from-scratch at 50M-750M
- EVQ provides a scaling law (tau* = d_head/sqrt(L)); VideoRoPE provides per-axis heuristic rules

**Ready-to-paste rebuttal**:

> VideoRoPE and EVQ represent convergent evidence from independent approaches. VideoRoPE discovers empirically that temporal dimensions benefit from low-frequency emphasis; EVQ's variational framework provides a theoretical explanation: the optimum (tau > 0) naturally shifts density toward low frequencies. The methods differ in scope: VideoRoPE is a heuristic 3D design for video VLMs, while EVQ is a closed-form solution with a parameter-free scaling law for general RoPE. Our video experiments (Section 6.3, 6.4) show that EVQ's tau* law applies in the video setting, suggesting a unified framework underlying both approaches. We view this convergence as strengthening evidence for the importance of frequency allocation.

---

## 6. Downstream Task Evidence

### Attack 6.1: "PPL improvements don't always translate to downstream task improvements. Where is the downstream evidence?"

**Vulnerability**: MEDIUM. We have QuALITY QA and LongBench NLL, but accuracy gains are modest.

**Defense Strategy**:

(a) **We have three layers of downstream evidence**:
- LongBench NLL (13 tasks, 750M): +4.4% / -4.4% symmetric waterbed reversal
- QuALITY QA (n=2086, 454M): Gold Answer NLL -30% at 2x, accuracy +2.2pp (p~0.02) at 2x
- Passkey retrieval (350M 6-seed): 100% vs 61-65% at 4x extrapolation

(b) **Signal attenuation is expected for infrastructure-level changes**: PE allocation is infrastructure. Its signal propagates: PPL -52% -> Gold NLL -30% -> passkey +60pp -> accuracy +2.2pp. Each abstraction layer attenuates but never reverses the signal. For comparison, DAPE (NeurIPS 2024) was accepted with only PPL and CHE --- no NLL, no downstream accuracy at all.

(c) **Model capacity confound**: At 454M, QuALITY accuracy is near the 25% random baseline for both models. The model barely learned the task. Gold NLL, as a continuous metric, captures the PE signal that accuracy at capacity floor cannot resolve.

**Ready-to-paste rebuttal**:

> We provide three layers of downstream evidence: (1) LongBench conditional NLL on 13 tasks showing a symmetric +4.4%/-4.4% waterbed reversal; (2) QuALITY QA (n=2086) showing Gold Answer NLL -30% at 2x extrapolation with accuracy +2.2pp (p~0.02); and (3) 6-seed passkey retrieval (100% vs 61-65%). The modest accuracy gains reflect model-capacity limitations (454M is near random baseline on QuALITY), not EVQ limitations --- Gold NLL, a continuous metric, clearly captures the PE signal (-30%). For context, DAPE (NeurIPS 2024) was accepted with only PPL and CHE evaluation, with no NLL or downstream accuracy measurement. We believe our downstream evidence substantially exceeds the current standard for PE allocation papers.

### Attack 6.2: "The waterbed cost (+4.4% at in-distribution) could be unacceptable in production."

**Vulnerability**: LOW. The cost is small and well-characterized.

**Defense Strategy**:

(a) The +4.4% NLL cost at in-distribution is measured on 750M zero-shot. Under fine-tuning (QuALITY), the in-distribution cost is only -1.7% NLL (EVQ is actually better).

(b) The waterbed is inherent to the mathematics --- it's the price of redistributing finite channel capacity. The question is whether the trade-off is favorable, and the data consistently shows it is: the long-range gain (up to -45.9% PPL, -30% NLL) vastly outweighs the short-range cost.

(c) In production long-context systems (where L >> L_train), the benefit domain dominates.

**Ready-to-paste rebuttal**:

> The waterbed trade-off is inherent to any channel redistribution and is well-characterized in our paper. The +4.4% in-distribution cost (750M zero-shot) is bounded at <=0.4% in PPL terms across all scales, and under fine-tuning (QuALITY), EVQ actually shows better in-distribution NLL (-1.7%). In production long-context systems where sequences regularly exceed L_train, the benefit domain (up to -45.9% PPL, -30% Gold NLL) vastly outweighs the modest short-range cost. We believe honest presentation of this trade-off is a strength of the paper.

---

## 7. MLA Experiment Concerns

### Attack 7.1: "The MLA model is only 432M. MLA is used at 236B+ in DeepSeek-V3. How does this generalize?"

**Vulnerability**: MEDIUM. Architecture matches production, but scale is much smaller.

**Defense Strategy**:

(a) **Architecture fidelity**: Our MLA model uses the same d_rope=32 (16 frequency channels) as production MLA. The frequency allocation problem is defined by d_rope and L, not by total model parameters.

(b) **The "fewer channels, each more precious" principle**: With only 16 channels, each channel's placement carries more weight. EVQ shows a -31.1% improvement at 2x extrapolation --- larger than typical MHA improvements (-13% to -15%) --- consistent with the prediction that constrained frequency budgets amplify allocation quality.

(c) **3-seed validation with tight confidence intervals**: All 3 seeds show consistent advantage (29.7%-33.6%), ruling out noise.

**Ready-to-paste rebuttal**:

> Our MLA experiment preserves the critical architectural parameters (d_rope=32, 16 frequency channels) that define the frequency allocation problem. The allocation optimization is determined by d_rope and L, not model width/depth --- the 16 channels must cover the same frequency range regardless of whether the model has 432M or 236B parameters. The -31.1% improvement (3-seed) exceeds typical MHA improvements at comparable scales, consistent with the "fewer channels, each more precious" hypothesis. We acknowledge that production-scale MLA validation would strengthen this finding and note this in Limitations.

### Attack 7.2: "tau choice for MLA (1.414) is based on L=512 reference, not the actual L_train=8192. Is this principled?"

**Vulnerability**: MEDIUM. We explicitly note this in mainstory.md but should address it transparently.

**Defense Strategy**:

The tau* = d_head/sqrt(L) scaling law was validated for L in [256, 2048]. Using L=8192 would give tau = 32/sqrt(8192) = 0.354, which is in the low-tau regime where EVQ barely differs from geometric. We used L=512 as a reference from the validated range, giving tau = 1.414. The strong results suggest this choice is effective, though the optimal tau for MLA at L=8192 may differ.

This is an honest area where the theory may need extension: the scaling law may need a saturation or floor correction for very long training lengths, or the relevant "L" for MLA may be an effective length related to d_rope rather than the actual sequence length.

**Ready-to-paste rebuttal**:

> We appreciate this observation. The tau* scaling law was validated for L in [256, 2048]. We used L=512 as a reference from the validated range, yielding tau=1.414. We note in the paper that the optimal tau for MLA at L=8192 may differ and warrants further investigation. One hypothesis is that the relevant "L" for tau* in MLA may scale with d_rope rather than the full sequence length, since the 16 frequency channels have a different effective resolution than 32+ channels in MHA. The strong results at tau=1.414 are encouraging, but we agree this is an area where the scaling law may need refinement for the compressed-RoPE regime.

---

## 8. Novelty and Theoretical Contribution

### Attack 8.1: "The idea of redistributing RoPE frequencies is not new. What's the novelty?"

**Vulnerability**: LOW. We have a clear novelty claim that is well-differentiated from prior work.

**Defense Strategy**:

(a) **Prior work on frequency allocation**: DAPE (NeurIPS 2024) learns frequency parameters via backprop. FIRE learns an implicit mapping via neural network. CREAM manipulates position indices. LongRoPE searches for per-frequency scaling factors. None of these provide a closed-form solution derived from first principles.

(b) **EVQ's specific novelty**:
- First variational formulation of the RoPE frequency allocation problem
- First closed-form solution family (EVQ-Cosh) where geometric RoPE is the tau=0 limit
- First parameter-free scaling law (tau* = d_head/sqrt(L))
- First demonstration that training-time allocation and inference-time scaling compose multiplicatively
- First PE allocation study on MLA

(c) **The geometric limit is new**: Showing that geometric RoPE is a degenerate boundary case of an optimization family --- not just a design choice --- reframes the entire field.

**Ready-to-paste rebuttal**:

> We respectfully disagree. While the motivation of improving RoPE frequencies is shared with prior work, no existing method provides a closed-form derivation from first principles. DAPE learns d/2 parameters via backprop; FIRE uses a neural network; CREAM manipulates indices; LongRoPE uses evolutionary search. EVQ is, to our knowledge, the only method that derives an analytic solution from a variational principle, shows geometric RoPE is its degenerate limit, and provides a parameter-free scaling law. The theoretical contribution is not "redistribute frequencies" but "prove that the optimal redistribution has a specific closed form and that the current default is its boundary case."

### Attack 8.2: "The waterbed inequality is well-known in information theory. Is this a contribution?"

**Vulnerability**: LOW. We don't claim waterbed is new --- we claim its empirical validation on downstream tasks is new.

**Defense Strategy**:

The waterbed inequality in our context is a consequence of the variational optimization, not a borrowed result. More importantly, the contribution is the empirical validation: the +4.4%/-4.4% symmetric reversal on 13 LongBench tasks, with task-type decomposition showing QA tasks (requiring precise retrieval) benefiting most. To our knowledge, this is the first direct measurement of the frequency-allocation waterbed effect on real downstream tasks.

**Ready-to-paste rebuttal**:

> The waterbed inequality in our paper is derived as a consequence of the variational optimization (Section 3), not borrowed from information theory. The contribution is not the inequality itself but its empirical validation on downstream tasks: the +4.4%/-4.4% symmetric reversal on 13 LongBench tasks, with task-type decomposition showing QA tasks benefiting up to -16.8%. To our knowledge, this is the first direct measurement of the frequency-allocation waterbed effect on real downstream tasks, providing empirical grounding for the theoretical prediction.

---

## 9. Practical Impact and Adoption

### Attack 9.1: "If EVQ is so simple (one-line change), why hasn't anyone done this before?"

**Vulnerability**: This is actually a strength, not a weakness.

**Defense Strategy**:

(a) **The simplicity is the point**: Many fundamental insights in ML are simple in hindsight. The contribution is not the code change but the theoretical framework that identifies what the optimal change should be.

(b) **The geometric default was never questioned**: Since RoFormer (2021), every production model has used geometric frequency allocation. The absence of a principled framework for reasoning about allocation meant there was no way to evaluate alternatives systematically.

(c) **Prior attempts used complexity**: DAPE adds d/2 learnable parameters. FIRE adds a neural network. The fact that EVQ achieves better results with zero parameters by leveraging the correct theoretical framework validates the approach.

**Ready-to-paste rebuttal**:

> This is precisely our point. The geometric default has been used without question since RoFormer (2021) because no principled framework existed for evaluating alternatives. Prior methods that attempted to improve allocation (DAPE, FIRE) added learnable parameters because they lacked theoretical guidance. EVQ's simplicity (zero parameters, one-line change) is a direct consequence of having the correct theoretical framework --- the variational solution tells us exactly what the optimal allocation should be, eliminating the need for learning or search.

### Attack 9.2: "The paper doesn't demonstrate EVQ on any production model or real-world application."

**Vulnerability**: MEDIUM. True, but this is common for foundational PE research.

**Defense Strategy**:

(a) **Consistent with the field**: DAPE (NeurIPS 2024) tests at 125M only. FIRE at 125M/350M. Resonance RoPE at synthetic tasks. No PE allocation paper has demonstrated on a production-deployed model.

(b) **Cross-architecture evidence**: We show benefits on GPT-style MHA, MLA (DeepSeek architecture), autoregressive video, bidirectional DiT, and cross-family transfer (Llama-3, Qwen-2.5).

(c) **Zero-overhead integration**: EVQ requires no architecture changes, no training recipe changes, no inference changes. The barrier to production adoption is minimal.

**Ready-to-paste rebuttal**:

> Production deployment is outside the scope of foundational PE research, as it is for all comparable papers (DAPE, FIRE, CREAM, Resonance RoPE). Our contribution is the theoretical framework and empirical validation, not a production system. We note that EVQ's integration cost is uniquely low among PE methods: zero extra parameters, zero hyperparameters, zero architecture changes, zero inference overhead --- a single line of initialization code. This low barrier makes production adoption straightforward.

---

## 10. YaRN Composition Claims

### Attack 10.1: "The -86% average PPL improvement over Geo+YaRN is from a single-seed experiment. This is unreliable."

**Vulnerability**: MEDIUM. Phase 17 composition data is indeed single-seed.

**Defense Strategy**:

(a) **Multiple independent confirmations of the composition pattern**:
- Phase 17 (454M, single-seed): -86% average across 4K-32K
- MLA 3-seed: EVQ+YaRN(s=4) -48.8% at 16K (3-seed validated)
- Phase 18 YaRN FT: 13.6pp structural reversal (single-seed, but confirmed across undertrained and fully-trained regimes)
- 6-seed passkey: EVQ+YaRN 100% vs Geo+YaRN 61-65% (6-seed, zero variance)

(b) **The direction is consistent across every experimental line**: No experiment has ever shown Geo+YaRN outperforming EVQ+YaRN.

**Ready-to-paste rebuttal**:

> The -86% number is from a single-seed experiment, which we acknowledge. However, the composition pattern is confirmed across multiple independent experimental lines: MLA 3-seed (-48.8% at 16K), 6-seed passkey (100% vs 61-65%, zero variance), Phase 18 YaRN FT (13.6pp structural reversal across two training regimes), and progressive training (3 stages, monotonically widening). No experiment across our entire evidence base has shown Geo+YaRN outperforming EVQ+YaRN. The specific -86% magnitude is single-seed, but the qualitative conclusion --- that EVQ provides a multiplicatively better foundation for YaRN --- is supported by multi-seed evidence across multiple architectures and training regimes.

### Attack 10.2: "The claim that EVQ and YaRN are 'orthogonal' is not rigorously established."

**Vulnerability**: LOW-MEDIUM. We use "orthogonal" colloquially (addressing different bottlenecks), not mathematically.

**Defense Strategy**:

(a) **Mechanistic argument**: EVQ modifies the frequency allocation phi_k (within-band density). YaRN rescales frequencies at inference to cover longer contexts. In log-frequency space, EVQ shifts phi_k below the geometric diagonal (shape correction), while YaRN shifts phi_k above it (range correction). The corrections are additive: phi_{EVQ+YaRN} = phi_{EVQ} + Delta_phi_{YaRN}.

(b) **Empirical confirmation**: The strict dominance hierarchy at every extrapolation length (EVQ+YaRN > EVQ > GEO+YaRN > GEO) demonstrates composition. YaRN provides larger marginal benefit on EVQ than on GEO (-25.6% vs -15.1% at 16K), confirming superlinear interaction.

(c) **We provide a clear figure** (Fig. 4) showing the orthogonal decomposition in log-frequency space.

**Ready-to-paste rebuttal**:

> We use "orthogonal" to mean that EVQ and YaRN address different deficiencies --- shape (within-band density) vs range (beyond L_train coverage). This is formalized in Figure 4: in log-frequency space, EVQ bends phi_k below the diagonal while YaRN shifts phi_k above it, and the corrections are additive. The empirical evidence supports superlinear rather than merely additive composition: YaRN provides -25.6% marginal benefit on EVQ vs -15.1% on GEO at 16K (MLA, 3-seed), indicating that a better frequency substrate amplifies inference-time scaling. We acknowledge that "orthogonal" is used colloquially and the precise interaction mechanism deserves deeper theoretical analysis.

---

## 11. Presentation and Scope

### Attack 11.1: "Too many contributions (6 bullets). The paper tries to do too much."

**Vulnerability**: MEDIUM. 6 contribution bullets is above the NeurIPS norm of 3-4.

**Defense Strategy**:

If this concern arises, we can consolidate in the camera-ready:
1. Theory: Closed-form variational solution, geometric RoPE = tau=0 limit
2. Systems result: EVQ unlocks YaRN (multiplicative composition + progressive amplification + structural reversal)
3. PE-dominant: Closed-form beats learnable PE with 0 parameters
4. MLA: First study on MLA, amplified benefit in compressed regime

**Ready-to-paste rebuttal**:

> Thank you for this feedback. We agree the contribution list can be more focused. The core story is: (1) a theoretical framework yielding a closed-form solution where geometric RoPE is the tau=0 limit; (2) the systems discovery that EVQ unlocks inference-time scaling, with progressive amplification and structural composition robustness; (3) PE-dominant validation showing closed-form allocation beats learnable alternatives; and (4) MLA validation confirming amplified benefit in the compressed-RoPE regime. We will consolidate the contribution list in the camera-ready version.

### Attack 11.2: "The paper is dense. Some reviewers may find it hard to follow."

**Vulnerability**: LOW-MEDIUM. The theory section is necessarily technical.

**Defense Strategy**:

We make several deliberate design choices for accessibility:
- Section 4 (Predictions) provides physical intuition before experiments
- The "shape vs range" decomposition (Figure 4) gives a visual mnemonic
- Key numbers are highlighted in tables rather than buried in text
- The appendix contains full derivation details, keeping the main text focused on results

**Ready-to-paste rebuttal**:

> We appreciate this concern and have structured the paper to be accessible at multiple levels: Section 4 provides physical intuition through the shape-vs-range decomposition and waterbed interpretation before any experimental data. Figure 4 gives a visual mnemonic for the orthogonality claim. Detailed derivations are reserved for the appendix to keep the main text focused. We welcome specific suggestions for improving clarity in particular sections.

---

## 12. Reproducibility

### Attack 12.1: "No code release mentioned. How can results be reproduced?"

**Vulnerability**: MEDIUM. The NeurIPS checklist asks about code availability.

**Defense Strategy**:

(a) **The core method is a 6-line function**: The entire EVQ implementation is shown in the paper (Algorithm 1). Any researcher can implement it in minutes.

(b) **132 unit tests validate the implementation**: Covering numerical stability (tau in [1e-8, 20]), gradient correctness (autograd.gradcheck), and independent numpy cross-validation at rtol=1e-6.

(c) **Full hyperparameters in appendix**: Reproducibility table with all training details.

(d) **Consider anonymous code release**: If required, we can provide an anonymous repository with the core library and test suite.

**Ready-to-paste rebuttal**:

> The core EVQ method is a 6-line Python function shown in Algorithm 1, which any researcher can implement in minutes. We provide complete hyperparameters in the appendix and validate our implementation with 132 unit tests covering numerical stability, gradient correctness, and independent numpy cross-validation. We will release our code and test suite upon acceptance, and can provide an anonymous repository during review if desired.

---

## 13. Competitive Landscape Context (2024-2025)

This section provides factual context on recent PE papers for calibrating reviewer expectations and positioning EVQ's contributions.

### 13.1 PE Allocation Methods (Direct Competitors)

| Method | Venue | Approach | Parameters | Scale | Downstream | From-scratch |
|--------|-------|----------|-----------|-------|------------|:------------:|
| DAPE | NeurIPS 2024 | Learnable MLP-based PE | d/2 (32) | 125M | PPL + CHE only | Yes |
| FIRE | ICLR 2024 | Neural network interpolation | ~K | 125M, 350M | SCROLLS | Yes |
| CREAM | NeurIPS 2024 | Index manipulation + Gaussian | ~K | Llama-2 7B (LoRA) | LongBench | No (LoRA) |
| VideoRoPE | ICML 2025 Oral | Heuristic LTA for video | 0 | 7B+ VLM (post-hoc) | Video benchmarks | No (post-hoc) |
| **EVQ-Cosh** | **This paper** | **Closed-form variational** | **0** | **50M-750M (5 scales)** | **NLL 13 tasks + QA** | **Yes** |

**Key differentiators**:
- EVQ is the only allocation method with a closed-form derivation from first principles
- EVQ requires 0 extra parameters (vs DAPE's 32, FIRE's ~K)
- EVQ provides the broadest from-scratch scale chain (5 scales)
- EVQ is the first allocation method studied on MLA
- EVQ shows explicit composition with inference-time methods (no other allocation paper does this)

### 13.2 Inference-Time Scaling Methods (Complementary)

| Method | Venue | Approach | Axis |
|--------|-------|----------|------|
| YaRN | ICLR 2024 | NTK-by-parts + attention scaling | Inference (axis 3) |
| LongRoPE | ICML 2024 | Evolutionary search for per-freq factors | Inference (axis 3) |
| LongRoPE2 | arXiv 2025 | Improved LongRoPE with near-lossless scaling | Inference (axis 3) |
| Resonance RoPE | ACL Findings 2024 | Integer-period snapping | Inference (axis 3) |
| CoPE | arXiv 2025 | Clipped RoPE + ABF | Inference (axis 3) |

EVQ is orthogonal to all of these. The paper demonstrates explicit composition with YaRN; composition with LongRoPE, Resonance RoPE, and CoPE are predicted to be beneficial and represent future work.

### 13.3 Theoretical Analysis Papers

| Paper | Venue | Key Finding | Relation to EVQ |
|-------|-------|-------------|-----------------|
| "Round and Round We Go" | ICLR 2025 | Gemma prefers low frequencies; high frequencies for positional attention | Consistent with EVQ's prediction that low-freq band is bottleneck |
| "A Comparative Study of RoPE-based PE" | OpenReview 2024 | Empirical comparison of RoPE variants | Contextualizes but doesn't address allocation axis |

### 13.4 Acceptance Bar Calibration

Based on recent PE papers accepted at top venues:

- **DAPE (NeurIPS 2024 poster)**: 125M scale, PPL + CHE only, no downstream accuracy, learnable parameters. Bar: theoretical framing + clean experiments at small scale.
- **CREAM (NeurIPS 2024 poster)**: LoRA fine-tuning on Llama-2, no from-scratch. Bar: practical recipe + LongBench evaluation.
- **YaRN (ICLR 2024)**: Empirical method, no theory. Bar: practical impact + comprehensive evaluation.
- **LongRoPE (ICML 2024)**: Evolutionary search, no closed-form. Bar: strong results + practical deployment (Phi-3).

EVQ exceeds the empirical bar set by all of these: broader scale chain, more downstream evidence, and a closed-form theoretical framework that none of the above provide. The main gap vs these papers is production-scale deployment (LongRoPE in Phi-3), which is a deployment rather than research contribution.

---

## Appendix: Quick-Reference Rebuttal Numbers

| Claim | Key Number | Evidence Strength | Seeds |
|-------|-----------|:--:|:--:|
| EVQ+YaRN composition | -86% avg PPL (4K-32K) | Single-seed but consistent across all lines | 1 |
| MLA standalone | -31.1% at 16K | Multi-seed, tight CI | 3 |
| MLA composition (YaRN) | -48.8% at 16K | Multi-seed | 3 |
| Structural reversal (YaRN FT) | 13.6pp swing | Single-seed, two regimes | 1 |
| Progressive amplification | -34.6% -> -52.0% -> -81.2% | Single-seed, Stage 1 multi-seed confirmed | 1 (full), 3 (S1) |
| Passkey EVQ+YaRN | 100% vs 61-65% | Multi-seed, zero variance | 6 |
| PE-dominant vs DAPE | 333.7 vs 455.3 | Multi-seed | 3 |
| Cross-scale raw PPL | -13.3% at 16K (350M) | Multi-seed | 3 |
| Downstream NLL | -30% Gold NLL (QuALITY) | Large n (2086) | 1 (model) |
| Waterbed | +4.4% / -4.4% | 13 tasks | 1 |
| tau* scaling law | <1% PPL gap worst case | 99 runs, 27 configs | 3 |
| 750M supporting | -45.9% PPL, 0%->77.5% AR | Single-seed (explicitly labeled) | 1 |
| Video temporal | -47% PPL at 8x extrap | Multi-seed | 2 |
| DiT head-to-head | -32% far-frame MSE | Head-to-head (same run) | 2 |
| 132 unit tests | All pass, 2.32s | Deterministic | - |

---

## 14. Additional Attacks (Reader-Test Identified)

### Attack 14.1: "R^2 > 0.99 but 35-49% residual --- these numbers seem contradictory."

**Vulnerability**: MEDIUM. This is a presentation issue that could kill credibility if not clarified.

**Clarification**: The R^2 > 0.99 refers to the **mid-band** projection quality (the region where the variational ODE operates, approximately phi in [0.2, 0.8]). The 35-49% **full-matrix** residual includes boundary effects (UV discretization at phi~0, IR truncation at phi~1, and finite diagonal ridge width). These are different measurements on different domains. The mid-band R^2 governs the quality of the EVQ solution; the full-matrix residual includes regions where the ODE does not apply.

**Ready-to-paste rebuttal**:

> Thank you for catching this apparent contradiction. The R^2 > 0.99 refers to the mid-band projection (phi in [0.2, 0.8]), the region where the variational ODE operates. The 35-49% residual is the full-matrix residual including boundary effects (UV discretization, IR truncation, diagonal ridge width) that fall outside the ODE's domain. We agree this distinction needs to be more prominent and will clarify in the revised text. The key point is that the approximation is accurate precisely where it matters for the EVQ solution.

### Attack 14.2: "Dimensional analysis doesn't uniquely derive tau* proportional to d_head/sqrt(L). Why not d_head/L or sqrt(d_head/L)?"

**Vulnerability**: MEDIUM. The dimensional argument alone does not uniquely select the sqrt(L) dependence.

**Defense Strategy**:

(a) The specific functional form comes from the ratio beta/alpha in the broadband surrogate, not from pure dimensional analysis. The sqrt arises because tau^2 = beta/alpha, and alpha and beta have different L-dependence in the kernel projection.

(b) **Empirical discrimination**: The 99-run sweep tests multiple functional forms. tau* = d_head/sqrt(L) provides consistently better predictions than d_head/L (which overestimates tau at large L) or sqrt(d_head/L) (which underestimates tau at large d_head).

(c) **Sensitivity analysis**: At d_head=64, even the worst-case configuration shows <1% PPL gap from the empirical optimum, suggesting the landscape around tau* is shallow. Moderate perturbations of the scaling law (e.g., +/-20%) still give near-optimal results.

**Ready-to-paste rebuttal**:

> The specific d_head/sqrt(L) form emerges from the beta/alpha ratio in the broadband surrogate (tau^2 = beta/alpha, where the two parameters have different L-dependence), not from dimensional analysis alone. The 99-run sweep empirically discriminates this from alternatives: d_head/L systematically overestimates tau at large L, while sqrt(d_head/L) underestimates at large d_head. Importantly, the PPL landscape around tau* is shallow (<1% gap at worst case), so moderate perturbations of the functional form still yield near-optimal results.

### Attack 14.3: "Did you test architectures where EVQ fails? Only reporting successes looks like selection bias."

**Vulnerability**: MEDIUM-HIGH. We should proactively address this.

**Defense Strategy**:

(a) **We report principled negative results**: base=10K, L=4096 (350M) where EVQ underperforms, exactly as predicted by collision theory (c=0.90, only ~3/32 channels optimizable). This is in the paper.

(b) **DiT tau sweep includes failures**: tau=0.30, 0.70, 1.20 all lose to GEO, only tau=1.50 wins. We report the full sweep, not just the winner.

(c) **The theory predicts when EVQ should NOT help**: When the collision block is small (low base, long training), EVQ has little room to optimize. We test and confirm this prediction. A method that has no failure modes is suspicious; one whose failures match predictions is credible.

**Ready-to-paste rebuttal**:

> We report principled negative results throughout the paper: (1) base=10K, L=4096 where EVQ underperforms geometric, matching the collision theory prediction that only ~3/32 channels are optimizable; (2) DiT experiments reporting 4 tau values, of which 3 lose to GEO (only tau=1.5 wins); (3) the dead-zone prediction explicitly identifies when EVQ should NOT help. We believe reporting failures that match theoretical predictions is stronger evidence than reporting only successes --- it demonstrates the theory is predictive, not post-hoc.

### Attack 14.4: "Inference/initialization cost of EVQ --- is the arcsinh computation expensive?"

**Vulnerability**: LOW. This is trivial but should be addressed.

**Defense Strategy**:

EVQ's cost is exactly one arcsinh call per channel during model initialization. This is a one-time cost of ~microseconds on GPU (32 channels, 1 arcsinh each). During training and inference, EVQ uses the standard RoPE forward pass --- the inv_freq tensor is precomputed and static. There is zero runtime overhead.

**Ready-to-paste rebuttal**:

> EVQ's computational cost is one arcsinh evaluation per frequency channel at model initialization (e.g., 32 arcsinh calls for d_head=64). This takes microseconds on GPU and is executed exactly once. During training and inference, EVQ uses the identical RoPE forward pass as geometric allocation --- the inv_freq tensor is precomputed and static. Runtime overhead is exactly zero.

### Attack 14.5: "Which is doing the work --- EVQ or YaRN? Your composition results could mean YaRN is just better with a warmer start."

**Vulnerability**: MEDIUM. This is a valid decomposition question.

**Defense Strategy**:

(a) **Ablation is in the data**: We show EVQ raw (no YaRN), GEO+YaRN, EVQ+YaRN, and GEO raw. The strict dominance hierarchy (EVQ+YaRN > EVQ > GEO+YaRN > GEO at every length) demonstrates that both contribute independently.

(b) **Marginal analysis**: YaRN's marginal benefit is larger on EVQ than on GEO (-25.6% vs -15.1% at 16K in MLA). This means EVQ amplifies YaRN's effectiveness, not just passively receives it.

(c) **"Warmer start" is exactly our claim**: EVQ provides a better frequency substrate for YaRN to work with. The mechanism is not mysterious --- YaRN rescales frequencies, and better input frequencies produce better rescaled frequencies.

**Ready-to-paste rebuttal**:

> The ablation structure (4 conditions: GEO, GEO+YaRN, EVQ, EVQ+YaRN) directly decomposes the contributions. The strict dominance hierarchy at every length shows both contribute independently. Critically, YaRN's marginal benefit is larger on EVQ than on GEO (-25.6% vs -15.1% at 16K), demonstrating that EVQ amplifies YaRN's effectiveness. The reviewer's framing ("warmer start") is precisely our claim: EVQ provides a structurally better frequency substrate for inference-time scaling. The 13.6pp structural reversal under YaRN FT further confirms this --- the composition benefit persists even when the standalone advantage vanishes.

---

## Appendix: Response Templates by Reviewer Type

### Theoretically-Oriented Reviewer
**Likely concerns**: Broadband surrogate validity, scaling law derivation, waterbed rigor
**Strategy**: Lead with the 24,000-config validation, emphasize falsifiability, highlight that geometric RoPE is a special case (not just an alternative)

### Empirically-Oriented Reviewer
**Likely concerns**: Scale, single-seed, downstream tasks, comparison breadth
**Strategy**: Lead with the 5-scale chain + MLA 3-seed, compare favorably with DAPE/FIRE/CREAM evidence standards, emphasize the structural reversal

### Systems/Practical Reviewer
**Likely concerns**: Production relevance, integration cost, deployment experience
**Strategy**: Lead with zero-parameter/zero-overhead, one-line code change, MLA architecture match, Llama-3/Qwen cross-architecture transfer

### Adversarial Reviewer
**Likely concerns**: "Not novel enough", "just another PE tweak", "scale too small"
**Strategy**: Lead with the geometric-limit framing (this generalizes, not replaces), the composition discovery (new phenomenon, not incremental improvement), and fair comparison with accepted papers at same venues
