# EVQ-Cosh Rebuttal Playbook: NeurIPS 2026

> **Purpose**: Comprehensive pre-rebuttal preparation. For each anticipated reviewer attack, we provide: the attack framing, an honest self-assessment of vulnerability, the defense strategy with specific evidence pointers, and ready-to-paste rebuttal text.
>
> **Methodology**: Attacks are derived from (1) the paper's own limitations section, (2) common NeurIPS reviewer patterns for PE/long-context papers, (3) competitive landscape analysis of 2024-2025 PE papers (DAPE, CREAM, FIRE, LongRoPE, Resonance RoPE, VideoRoPE, CoPE, "Round and Round We Go"), and (4) known weaknesses identified in our internal audit.
>
> **Last updated**: 2026-06-01

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

**Vulnerability**: MEDIUM-HIGH. This is our most honest limitation. Primary multi-seed evidence is at 50M-454M. 750M is single-seed. The 1B/4K MLA row is supporting-only and includes a raw EVQ reversal at 8K/16K, so it cannot be used as broad scale validation.

**Defense Strategy**:

(a) **Scale-independence of the mechanism**: EVQ modifies only the RoPE inverse-frequency initialization --- a one-line change that is architecture-agnostic. The theoretical prediction (tau* = d_head / sqrt(L)) depends on d_head and L, not model width or depth. The mechanism operates at the frequency-channel level, which is invariant to model scale.

(b) **Mechanism-scale coverage, with caveats**: We provide a broad from-scratch PE allocation study across 50M, 125M, 350M (3-seed), 454M, and 750M (single-seed/supporting). The primary mechanism stress tests are consistent, but supporting rows should not be described as universal scale validation. For comparison:
- DAPE (NeurIPS 2024): 125M only
- FIRE (ICLR 2024): 125M, 350M
- CREAM (NeurIPS 2024): Llama-2 7B but LoRA fine-tuning only, not from-scratch
- LongRoPE (ICML 2024): Llama-2/3, Phi-3 --- but these are post-hoc scaling, not from-scratch training

(c) **MLA validation in a production-relevant compressed-RoPE regime**: Our 432M MLA experiment (3-seed) tests the same compressed-RoPE problem family used in DeepSeek-V2/V3. It is not production-identical: our stress test uses d_rope=32/base=500K, while production DeepSeek configurations use different rotary width/base choices. The defensible claim is that scarce rotary channels make allocation quality more visible.

(d) **Cross-architecture transfer**: Llama-3-8B and Qwen-2.5-8B LoRA experiments show EVQ benefits transfer to pretrained 8B models, though these are preliminary.

**Ready-to-paste rebuttal**:

> We acknowledge the scale gap and have stated this explicitly in Limitations. EVQ modifies only the frequency initialization, so the mechanism is defined at the frequency-channel level rather than by model width/depth. Our evidence is therefore best read as a mechanism study across several small-to-mid-scale settings, not as frontier-scale validation. The MLA experiment tests a production-relevant compressed-RoPE regime, but it is not production-identical. A 1B/4K supporting row also exposes a raw EVQ reversal in the old sparse-channel setup, which we treat as a limitation and a target for artifact-level audit rather than as scale evidence. Production-scale validation remains future work.

### Attack 1.2: "750M result is single-seed. It could be a lucky run."

**Vulnerability**: MEDIUM. The 750M result (PPL -45.9% at 16K, AR exact 0%->77.5%) is large, but formally it is n=1 and should stay supporting-only.

**Defense Strategy**:

The 750M experiment is explicitly labeled as "supporting evidence" in the paper, not a primary claim. The magnitude of the effect (-45.9% PPL, 0% to 77.5% AR exact) is large relative to the 350M 3-seed variance, but n=1 means it should support directionality only, not precise effect size or durability.

**Ready-to-paste rebuttal**:

> The 750M result is explicitly presented as single-seed supporting evidence. Its role is to show that the mechanism signal can appear beyond the smaller primary runs, not to establish a precise effect size or durability law. We therefore keep the primary claim anchored in multi-seed mechanism stress tests and treat 750M as supporting context.

---

## 2. Undertraining / Training Amount Objection

### Attack 2.1: "EVQ's advantage may simply be an artifact of insufficient training. With enough training, geometric RoPE would converge to similar performance."

**Vulnerability**: HIGH unless Phase 18 is scoped carefully, because the same row contains a raw EVQ reversal.

**Defense Strategy**:

This is the single most important attack to answer, and the evidence is mixed:

(a) **Phase 18 structural reversal, scoped carefully**: At 4K/1B tokens, EVQ *loses* standalone 8K extrapolation by +11.1%, while EVQ+YaRN+FT is better than GEO+YaRN+FT at the target length by -2.5%. This is useful evidence that the trained substrate can still matter under YaRN+FT, but it is single-seed, changes train length/data relative to the primary MLA run, and should also be presented as a raw-EVQ failure mode.

(b) **Two-component decomposition**: EVQ's advantage has two separable components:
- Raw extrapolation benefit: diminishes with training amount (as expected --- more training lets the model compensate for suboptimal frequencies)
- Structural composition benefit: tested under matched settings, because YaRN inherits the quality of the training-time frequency layout

(c) **Progressive training is supporting context**: Three-stage progressive training (512->1024->2048) shows the EVQ+YaRN advantage *growing* from -34.6% to -52.0% to -81.2%, but the full chain is single-seed and should not be used as primary durability proof.

(d) **MLA training progression**: EVQ's advantage at 16K is -29.0% at 50% training and -31.1% at 100% training --- monotonically increasing, not decreasing. The in-distribution cost simultaneously decreases from +1.4% to +0.9%.

**Ready-to-paste rebuttal**:

> This is an important concern, and we do not claim that more training universally preserves raw EVQ gains. In fact, the 1B/4K MLA supporting row shows a raw EVQ reversal at 8K/16K, which we treat as a real limitation of that sparse old-frequency setup. The narrower point is that training-time allocation still affects how inference-time scaling acts on the learned substrate: in the same supporting row, EVQ+YaRN+FT remains better than GEO+YaRN+FT at the target length, while the primary 8K/500M MLA run shows a stable 3-seed EVQ benefit at 16K. Progressive and continuation rows are supporting evidence only. Together these data argue against reducing the effect to mere undertraining, but they do not establish universal durability.

### Attack 2.2: "The models are trained on relatively few tokens (50M-500M tokens). Production models train on trillions of tokens."

**Vulnerability**: MEDIUM. True, but this applies equally to all PE allocation papers.

**Defense Strategy**:

(a) **Fair comparison with literature**: DAPE (NeurIPS 2024) trains on sequence length 128 with limited tokens. FIRE trains at 125M/350M scale. No PE allocation paper has trained at trillion-token scale --- this is a shared limitation of the subfield, not specific to EVQ.

(b) **The frequency initialization is consumed at step 0**: EVQ changes the initialization of inv_freq, which is used from the very first forward pass. The benefit is architectural, not dependent on training duration.

(c) **Phase 18 evidence, with limitation**: The 1B/4K MLA row shows that raw EVQ can converge toward or lose to Geo in a sparse-frequency window. The target-length EVQ+YaRN+FT comparison remains favorable, but the row should be used as scoped evidence for substrate effects under scaling, not as proof of no convergence.

**Ready-to-paste rebuttal**:

> This concern applies broadly to PE allocation research: DAPE trains in a PE-dominant small-length regime, FIRE at 125M/350M scale, and none of these works establish trillion-token from-scratch behavior. Our evidence should be read as mechanism validation, not production-scale convergence proof. The 1B/4K MLA row is especially informative because raw EVQ loses in early extrapolation, yet target-length EVQ+YaRN+FT remains mildly favorable. This narrows the claim: EVQ's value is not guaranteed raw dominance, but a frequency substrate that can give inference-time scaling higher leverage in the settings we test.

---

## 3. Single-Seed Evidence

### Attack 3.1: "Progressive training chain (Claims 4) is single-seed."

**Vulnerability**: MEDIUM. The progressive chain (454M, seed=42, 512->1024->2048) is indeed single-seed for the full 3-stage pipeline.

**Defense Strategy**:

(a) **Stage 1 is now multi-seed confirmed**: Seeds 42/43/44 all show consistent EVQ advantage at Stage 1 (PPL@4K: -16.5%, NIAH@1K: +26pp with zero variance).

(b) **Magnitude is useful but not definitive**: The Stage 3 advantage is -81.2% at 16K. This is large enough to motivate follow-up, but the full Stage 2-3 pipeline is still single-seed and should not be treated as statistically closed.

(c) **Corroborating evidence from independent experimental lines**: 750M (single-seed, -45.9%), 350M 3-seed (-13.3%), and passkey-mix EVQ+YaRN vs Geo+YaRN (3 seeds per method; 100% vs about 61%) show the same directional pattern in their scoped settings.

(d) **The progressive widening pattern**: The fact that the advantage monotonically increases (34.6% -> 52.0% -> 81.2%) across stages would be extremely unlikely to occur by chance.

**Ready-to-paste rebuttal**:

> We acknowledge the progressive chain is single-seed for the full pipeline and state this explicitly. Stage 1 has multi-seed support, and the later-stage effect size is large, but we use the full progressive chain only as supporting context. The primary claim remains anchored in the matched-scale 454M passkey table and the 3-seed MLA stress test; multi-seed validation of Stages 2-3 would be needed before making a durability claim.

---

## 4. Broadband Surrogate Validity

### Attack 4.1: "The entire theory rests on a single approximation (broadband surrogate). If this approximation fails, the derivation is invalid."

**Vulnerability**: MEDIUM. The surrogate derivation is correct conditional on the fitted broadband surrogate, but the reviewer can fairly attack the surrogate-to-trained-model gap.

**Defense Strategy**:

(a) **24,000-configuration numerical sweep**: The broadband projection K_approx = alpha*I + beta*A^{-1} achieves R^2 > 0.99 across the regime where RoPE-based models operate (base 8K-100K, L >= 4096). This is not a cherry-picked result --- it systematically maps the 6-dimensional boundary of validity.

(b) **35-49% full-matrix residual is understood**: The residual comes from three identifiable boundary effects (UV discretization, IR wavelength truncation, finite diagonal ridge width). The mid-band where the variational ODE operates is well-captured.

(c) **GPT-2 cross-validation**: Real attention distance distributions from 12x12 heads yield power-law fits consistent with the theoretical assumption (alpha > 0.8 for local heads, which are most sensitive to RoPE allocation).

(d) **The approximation is falsifiable, and the paper provides the falsification criteria**: We explicitly state when it fails (e.g., token co-occurrence distance priors give R^2 = 0.645-0.664). This transparency strengthens rather than weakens the claim.

**Ready-to-paste rebuttal**:

> We appreciate this important theoretical concern. The broadband surrogate is indeed the single approximation in the derivation, and we believe this transparency is a strength of our approach. We validate it with a 24,000-configuration sweep covering 6 dimensions (base, L, alpha, grid, mid-band, method), finding R^2 > 0.99 under D(delta) proportional to 1/delta for base in [8K, 100K] and L >= 4096 --- precisely the operating regime of modern RoPE models. We also explicitly show where it fails (token co-occurrence: R^2 ~ 0.65), providing falsification criteria. The residual (35-49%) is attributable to three identifiable boundary effects that do not affect the mid-band where the variational ODE operates (see Appendix A for detailed analysis).

### Attack 4.2: "The scaling law tau* = d_head/sqrt(L) is empirical, not theoretically derived."

**Vulnerability**: MEDIUM. The scaling rule is an operating default/basin selector supported by theory and sweeps, not a theorem of trained attention.

**Defense Strategy**:

The tau* rule should be described as arising from the separate small-τ softmax-transport argument plus empirical sweep validation. The broadband surrogate alone does not force the sqrt(L) exponent, and the flat basin means the rule mainly selects a stable non-geometric operating region.

**Ready-to-paste rebuttal**:

> We treat tau* = d_head/sqrt(L) as an operating default, not a global optimum theorem. The sqrt(L) dependence comes from a separate small-τ softmax-transport argument and is then validated by sweep evidence; the variational surrogate itself gives the closed-form allocation family rather than a complete trained-attention scaling law. The empirical basin is shallow, so the important claim is that a non-geometric allocation is robustly useful in the tested regimes, while direct L_eff^J measurement remains an open validation.

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

(c) **EVQ uses 0 extra parameters**: DAPE uses d/2 = 32 learnable parameters. EVQ's seed-42 diagnostic row is favorable in this protocol, but the comparison should remain seed- and protocol-scoped.

**Ready-to-paste rebuttal**:

> The PE-dominant regime (L=128) is a deliberately extreme diagnostic for isolating PE quality. In the reported seed-42 Geo/DAPE/EVQ contrast, EVQ is favorable while adding zero learned parameters. We additionally report L=256 supporting evidence. We do not present this as a broad learned-PE dominance result; stronger claims would require additional seeds and tuned learned-PE/range-scaling baselines.

### Attack 5.3: "VideoRoPE (ICML 2025 Oral) already addresses frequency allocation for video. What does EVQ add?"

**Vulnerability**: MEDIUM. VideoRoPE and EVQ are convergent in motivation, but video rows are supporting and should not carry the main text claim.

**Defense Strategy**:

VideoRoPE's core innovation is Low-frequency Temporal Allocation (LTA) --- a heuristic assignment of low-frequency channels to the temporal axis. EVQ provides a complementary theoretical perspective: the variational optimum (tau > 0) naturally shifts density toward low frequencies. The two methods arrive at the same directional conclusion from independent approaches (theory-first vs experiment-first).

Key differences:
- VideoRoPE is a heuristic 3D RoPE design for video VLMs; EVQ is a closed-form variational solution for general RoPE
- VideoRoPE operates at 7B+ scale post-hoc; EVQ is validated from-scratch at 50M-750M
- EVQ provides a closed-form allocation family and an operating tau rule for text RoPE; VideoRoPE provides per-axis heuristic rules

**Ready-to-paste rebuttal**:

> VideoRoPE and EVQ represent convergent motivation from independent approaches: both point toward low-frequency temporal/long-range allocation as important. The methods differ in scope: VideoRoPE is a heuristic 3D design for video VLMs, while EVQ is a closed-form allocation family for standard RoPE. Our video rows are supporting evidence that frequency allocation matters beyond text, but they should not be used as primary proof of the text tau rule.

---

## 6. Downstream Task Evidence

### Attack 6.1: "PPL improvements don't always translate to downstream task improvements. Where is the downstream evidence?"

**Vulnerability**: MEDIUM. We have QuALITY QA and LongBench NLL, but accuracy gains are modest.

**Defense Strategy**:

(a) **We have three layers of downstream evidence**:
- LongBench NLL (13 tasks, 750M): +4.4% / -4.4% symmetric waterbed reversal
- QuALITY QA (n=2086, 454M): Gold Answer NLL -30% at 2x, accuracy +2.2pp (p~0.02) at 2x
- Passkey retrieval (3 seeds per method): 100% vs about 61% at 4x extrapolation

(b) **Signal attenuation is expected for infrastructure-level changes**: PE allocation is infrastructure. In our diagnostics, the signal is strongest in PPL and teacher-forced NLL-style metrics, and weaker in end-task accuracy. That attenuation is expected and should be framed as scope, not as a downstream SOTA claim.

(c) **Model capacity confound**: At 454M, QuALITY accuracy is near the 25% random baseline for both models. The model barely learned the task. Gold NLL, as a continuous metric, captures the PE signal that accuracy at capacity floor cannot resolve.

**Ready-to-paste rebuttal**:

> We provide three layers of downstream or task-adjacent evidence: (1) LongBench conditional NLL on 13 tasks showing a symmetric +4.4%/-4.4% waterbed reversal; (2) QuALITY QA (n=2086) showing Gold Answer NLL -30% at 2x extrapolation with accuracy +2.2pp (p~0.02); and (3) passkey-mix retrieval with 3 seeds per method (100% vs about 61%). The modest accuracy gains reflect model-capacity limitations (454M is near random baseline on QuALITY), so Gold NLL and teacher-forced NLL-gap retrieval should be described as PE-diagnostic signals rather than full downstream task wins.

### Attack 6.2: "The waterbed cost (+4.4% at in-distribution) could be unacceptable in production."

**Vulnerability**: MEDIUM. The short-range cost is measured and usually small in our reports, but production acceptability is application-dependent.

**Defense Strategy**:

(a) The +4.4% NLL cost at in-distribution is measured on 750M zero-shot. Under fine-tuning (QuALITY), the in-distribution cost is only -1.7% NLL (EVQ is actually better).

(b) The waterbed is inherent to the mathematics --- it's the price of redistributing finite channel capacity. The question is whether the trade-off is favorable, and the data consistently shows it is: the long-range gain (up to -45.9% PPL, -30% NLL) vastly outweighs the short-range cost.

(c) In long-context systems where sequences regularly exceed L_train, the trade-off may be favorable, but this is an application-level decision rather than a theorem.

**Ready-to-paste rebuttal**:

> The waterbed trade-off is inherent to finite channel redistribution and we report it rather than hiding it. In our supporting downstream rows, the in-distribution cost is modest relative to the long-range NLL/PPL gains, but acceptability depends on the target application and context-length distribution. We therefore frame EVQ as a frequency-allocation trade-off, not a universally free production improvement.

---

## 7. MLA Experiment Concerns

### Attack 7.1: "The MLA model is only 432M. MLA is used at 236B+ in DeepSeek-V3. How does this generalize?"

**Vulnerability**: MEDIUM-HIGH. The experiment targets the compressed-RoPE problem family used by MLA, but the exact rotary width/base and model scale are not production-identical.

**Defense Strategy**:

(a) **Architecture relevance**: Our MLA model uses a compressed RoPE subspace with only 16 frequency channels. This isolates the same "scarce rotary budget" mechanism that motivates the MLA stress test, while the paper should explicitly note that production DeepSeek configurations use different rotary width/base choices.

(b) **The "fewer channels, each more precious" principle**: With only 16 channels, each channel's placement carries more weight. EVQ shows a -31.1% improvement at 2x extrapolation --- larger than typical MHA improvements (-13% to -15%) --- consistent with the prediction that constrained frequency budgets amplify allocation quality.

(c) **3-seed validation with tight confidence intervals**: All 3 seeds show consistent advantage (29.7%-33.6%), ruling out noise.

**Ready-to-paste rebuttal**:

> Our MLA experiment is a scarce-channel stress test for the compressed-RoPE problem family used by MLA, not a production-identical DeepSeek reproduction. With only 16 rotary frequency channels, allocation mistakes are amplified, making it a targeted mechanism test for the finite-spectral-budget claim. The -31.1% 16K improvement is 3-seed validated in this setting, but production-scale MLA validation and per-channel convention ablations remain future work.

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

(a) **Prior work on frequency allocation**: DAPE (NeurIPS 2024) learns frequency parameters via backprop. FIRE learns an implicit mapping via neural network. CREAM manipulates position indices. LongRoPE searches for per-frequency scaling factors. The defensible novelty claim is that EVQ provides a closed-form solution for a stated broadband surrogate, not that it is the only conceivable way to redistribute frequencies.

(b) **EVQ's specific novelty**:
- A variational formulation of the RoPE frequency allocation problem
- First closed-form solution family (EVQ-Cosh) where geometric RoPE is the tau=0 limit
- A zero-learned-parameter operating rule for selecting tau
- Evidence that training-time allocation and inference-time scaling compose in matched settings
- First PE allocation study on MLA

(c) **The geometric limit is new**: Showing that geometric RoPE is a degenerate boundary case of an optimization family --- not just a design choice --- reframes the entire field.

**Ready-to-paste rebuttal**:

> While the broad motivation of improving RoPE frequencies is shared with prior work, EVQ targets a different cell: a closed-form training-time allocation derived from a stated broadband surrogate, with geometric RoPE recovered as the tau-to-zero limit. DAPE learns d/2 parameters, FIRE uses a neural network, and LongRoPE searches inference-time scaling factors. Our theoretical contribution is not simply "redistribute frequencies," but a compact variational allocation family that can be tested and composed with inference-time scaling.

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

> This is precisely our point. The geometric default has often been treated as neutral infrastructure rather than as an allocation choice. EVQ's simplicity follows from narrowing the problem to a closed-form surrogate optimum: it gives a concrete, zero-learned-parameter allocation to test, while retaining the standard RoPE forward pass. We do not claim this exhausts all possible training-time schedules; it gives a principled and reproducible one.

### Attack 9.2: "The paper doesn't demonstrate EVQ on any production model or real-world application."

**Vulnerability**: MEDIUM. True, but this is common for foundational PE research.

**Defense Strategy**:

(a) **Consistent with the field**: DAPE (NeurIPS 2024) tests at 125M only. FIRE at 125M/350M. Resonance RoPE at synthetic tasks. No PE allocation paper has demonstrated on a production-deployed model.

(b) **Cross-architecture evidence with scope**: We report benefits on GPT-style MHA and the MLA scarce-channel stress test, plus supporting video and cross-family adaptation rows. The LoRA rows remain exploratory because matched Geo+LoRA controls are incomplete.

(c) **Zero-overhead integration**: EVQ requires no architecture change and no inference-time change beyond the initialized `inv_freq` table. Production adoption still requires validation under the target recipe.

**Ready-to-paste rebuttal**:

> Production deployment is outside the scope of this mechanism paper. Our contribution is the frequency-allocation framework and controlled evidence, not a deployed system. EVQ's integration cost is low -- it changes the initialized inverse-frequency table and keeps the standard RoPE forward pass -- but production adoption should be validated under the target model, data, and range-scaling recipe.

---

## 10. YaRN Composition Claims

### Attack 10.1: "The -86% average PPL improvement over Geo+YaRN is from a single-seed experiment. This is unreliable."

**Vulnerability**: MEDIUM. Phase 17 composition data is indeed single-seed.

**Defense Strategy**:

(a) **Multiple independent confirmations of the composition pattern**:
- Phase 17 (454M, single-seed): -86% average across 4K-32K
- MLA 3-seed: EVQ+YaRN(s=4) -48.8% at 16K (3-seed validated)
- Phase 18 YaRN FT: target-length EVQ+YaRN+FT remains better despite raw EVQ reversal (single-seed; beyond-target lengths are mixed)
- Passkey mix: EVQ+YaRN 100% vs Geo+YaRN about 61% (3 seeds per method; EVQ row has zero std)

(b) **The targeted composition direction is consistent in primary matched-scale settings**: The primary matched-scale rows favor EVQ+YaRN, but supporting Phase18 also contains beyond-target lengths where GEO+YaRN+FT is better. Do not claim universal dominance.

**Ready-to-paste rebuttal**:

> The -86% number is from a single-seed supporting experiment, which we acknowledge. The primary composition evidence should instead be the matched-scale 454M passkey table and the 3-seed MLA table: in those settings, YaRN has higher leverage on the EVQ-trained substrate than on Geo. Phase18 is useful as a supporting target-length YaRN+FT comparison, but it also contains beyond-target lengths where GEO+YaRN+FT is better, so it should not be used to claim universal dominance. The defensible conclusion is complementarity under matched tested settings, not that EVQ+YaRN beats every tuned or fine-tuned Geo+YaRN variant.

### Attack 10.2: "The claim that EVQ and YaRN are 'orthogonal' is not rigorously established."

**Vulnerability**: LOW-MEDIUM. We use "orthogonal" colloquially (addressing different bottlenecks), not mathematically.

**Defense Strategy**:

(a) **Mechanistic argument**: EVQ modifies the frequency allocation phi_k (within-band density). YaRN rescales frequencies at inference to cover longer contexts. In log-frequency space, EVQ shifts phi_k below the geometric diagonal (shape correction), while YaRN shifts phi_k above it (range correction). The corrections are additive: phi_{EVQ+YaRN} = phi_{EVQ} + Delta_phi_{YaRN}.

(b) **Empirical confirmation**: In the primary MLA table, YaRN provides larger marginal benefit on EVQ than on GEO at 16K (-25.6% vs -15.1%), supporting complementarity under the matched setting. This should not be generalized to every tuned scale, training length, or beyond-target evaluation point.

(c) **We provide a clear figure** (Fig. 4) showing the orthogonal decomposition in log-frequency space.

**Ready-to-paste rebuttal**:

> We use "orthogonal" to mean that EVQ and YaRN address different deficiencies --- shape (within-band density) vs range (beyond L_train coverage). In log-frequency space, EVQ changes the training-time allocation while YaRN rescales frequencies at inference. The empirical evidence supports complementarity in matched settings: YaRN provides -25.6% marginal benefit on EVQ vs -15.1% on GEO at 16K in the 3-seed MLA table. We acknowledge that "orthogonal" is colloquial rather than a theorem, and that tuned-scale baselines remain an important scope limitation.

---

## 11. Presentation and Scope

### Attack 11.1: "Too many contributions (6 bullets). The paper tries to do too much."

**Vulnerability**: MEDIUM. 6 contribution bullets is above the NeurIPS norm of 3-4.

**Defense Strategy**:

If this concern arises, we can consolidate in the revised manuscript:
1. Theory: Closed-form variational solution, geometric RoPE = tau=0 limit
2. Systems result: EVQ increases fixed-scale YaRN leverage (multiplicative composition + progressive amplification + structural reversal)
3. PE-dominant: Closed-form beats learnable PE with 0 parameters
4. MLA: First study on MLA, amplified benefit in compressed regime

**Ready-to-paste rebuttal**:

> Thank you for this feedback. We agree the contribution list can be more focused. The core story is: (1) a theoretical framework yielding a closed-form allocation family where geometric RoPE is the tau=0 limit; (2) the mechanism evidence that EVQ can give matched inference-time scaling higher leverage; (3) PE-dominant validation showing closed-form allocation is competitive with learned alternatives in the tested diagnostic; and (4) MLA validation showing amplified benefit in a scarce-channel stress test. We will consolidate the contribution list in the revised manuscript.

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

(b) **Focused implementation tests validate the released audit/core paths**: Current repository tests cover the RoPE core schedule, checkpoint-loaded `inv_freq` handling, artifact manifests, training-artifact audits, and audit-doc consistency.

(c) **Full hyperparameters in appendix**: Reproducibility table with all training details.

(d) **Consider anonymous code release**: If required, we can provide an anonymous repository with the core library and test suite.

**Ready-to-paste rebuttal**:

> The core EVQ method is a short inverse-CDF function shown in Algorithm 1, and the repository includes the schedule API, focused tests, and audit scripts for checkpoint `inv_freq` provenance. We provide hyperparameters in the appendix and will release the reviewer-facing code path needed to reproduce the reported claims, with missing external artifacts represented by sanitized manifests rather than private run roots.

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
- EVQ provides a closed-form allocation for a stated broadband surrogate
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

EVQ is complementary in principle to these inference-time methods because it
changes the training-time frequency substrate. The paper demonstrates explicit
composition with YaRN under matched tested settings; composition with LongRoPE,
Resonance RoPE, and CoPE remains future work.

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

EVQ's evidence profile is different from these papers: stronger on closed-form mechanism and training-time allocation, weaker on production-scale deployment and tuned inference-scaler baselines. That is the right comparison point for rebuttal; do not frame EVQ as simply exceeding every empirical bar.

---

## Appendix: Quick-Reference Rebuttal Numbers

| Claim | Key Number | Evidence Strength | Seeds |
|-------|-----------|:--:|:--:|
| EVQ+YaRN composition | -86% avg PPL (4K-32K) | Single-seed supporting; use primary matched-scale tables for rebuttal | 1 |
| MLA standalone | -31.1% at 16K | Multi-seed, tight CI | 3 |
| MLA composition (YaRN) | -48.8% at 16K | Multi-seed | 3 |
| Structural reversal (YaRN FT) | 13.6pp swing | Single-seed, two regimes | 1 |
| Progressive amplification | -34.6% -> -52.0% -> -81.2% | Single-seed, Stage 1 multi-seed confirmed | 1 (full), 3 (S1) |
| Passkey EVQ+YaRN | 100% vs about 61% | 3 seeds per method; teacher-forced NLL-gap; EVQ row has zero std | 3/method |
| PE-dominant vs DAPE | 333.7 vs 455.3 | Seed-42 diagnostic for Geo/DAPE/EVQ; learnable-tau row is multi-seed | 1--3 by row |
| Cross-scale raw PPL | -13.3% at 16K (350M) | Multi-seed | 3 |
| Downstream NLL | -30% Gold NLL (QuALITY) | Large n (2086) | 1 (model) |
| Waterbed | +4.4% / -4.4% | 13 tasks | 1 |
| tau* scaling law | <1% PPL gap worst case | 99 runs, 27 configs | 3 |
| 750M supporting | -45.9% PPL, 0%->77.5% AR | Single-seed (explicitly labeled) | 1 |
| Video temporal | -47% PPL at 8x extrap | Multi-seed | 2 |
| DiT head-to-head | -32% far-frame MSE | Head-to-head (same run) | 2 |
| Focused repo tests | RoPE core, checkpoint `inv_freq`, artifact manifests, audit docs | Deterministic helper coverage | - |

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

(a) The specific functional form should be attributed to the separate small-tau
softmax-transport argument, not to dimensional analysis alone and not to the
broadband surrogate alone.

(b) **Empirical support**: The sweep evidence supports the default operating
region and shows that tau=0 is often outside the useful basin. It should not be
over-sold as a theorem selecting a unique exponent for all trained attention
regimes.

(c) **Sensitivity analysis**: At d_head=64, even the worst-case configuration shows <1% PPL gap from the empirical optimum, suggesting the landscape around tau* is shallow. Moderate perturbations of the scaling law (e.g., +/-20%) still give near-optimal results.

**Ready-to-paste rebuttal**:

> The d_head/sqrt(L) form should be read as an operating rule supported by a small-tau softmax-transport argument and sweep validation, not as a dimensional-analysis theorem. The sweep evidence is useful because it places the default inside a broad non-geometric basin and separates it from tau=0 in the tested regimes. Direct L_eff^J measurements and additional functional-form ablations would be needed before making a stronger trained-attention scaling claim.

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

(a) **Ablation is in the data**: We show EVQ raw (no YaRN), GEO+YaRN, EVQ+YaRN, and GEO raw in the matched settings. The primary MLA and passkey rows support higher YaRN leverage on EVQ, while supporting Phase18 reminds us not to generalize this to every beyond-target length.

(b) **Marginal analysis**: YaRN's marginal benefit is larger on EVQ than on GEO (-25.6% vs -15.1% at 16K in MLA). This means EVQ amplifies YaRN's effectiveness, not just passively receives it.

(c) **"Warmer start" is exactly our claim**: EVQ provides a better frequency substrate for YaRN to work with. The mechanism is not mysterious --- YaRN rescales frequencies, and better input frequencies produce better rescaled frequencies.

**Ready-to-paste rebuttal**:

> The ablation structure (4 conditions: GEO, GEO+YaRN, EVQ, EVQ+YaRN) decomposes the matched-setting contributions. In the 3-seed MLA table, YaRN's marginal benefit is larger on EVQ than on GEO (-25.6% vs -15.1% at 16K), supporting the claim that EVQ gives inference-time scaling a better trained substrate in this setting. The reviewer's framing ("warmer start") is close to our mechanism claim: YaRN rescales the frequencies it inherits. We should not claim this implies universal dominance over tuned Geo+YaRN variants or every beyond-target length.

---

## Appendix: Response Templates by Reviewer Type

### Theoretically-Oriented Reviewer
**Likely concerns**: Broadband surrogate validity, scaling law derivation, waterbed rigor
**Strategy**: Lead with the 24,000-config validation, emphasize falsifiability, highlight that geometric RoPE is a special case (not just an alternative)

### Empirically-Oriented Reviewer
**Likely concerns**: Scale, single-seed, downstream tasks, comparison breadth
**Strategy**: Lead with the 454M matched-scale table and MLA 3-seed stress test; use the broader scale chain only as supporting context and acknowledge the structural reversal

### Systems/Practical Reviewer
**Likely concerns**: Production relevance, integration cost, deployment experience
**Strategy**: Lead with zero-parameter/zero-overhead, one-line code change, MLA architecture match, Llama-3/Qwen cross-architecture transfer

### Adversarial Reviewer
**Likely concerns**: "Not novel enough", "just another PE tweak", "scale too small"
**Strategy**: Lead with the geometric-limit framing (this generalizes, not replaces), the composition discovery (new phenomenon, not incremental improvement), and fair comparison with accepted papers at same venues
