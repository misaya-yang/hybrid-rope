# NeurIPS 2026 Review Committee Report (Full Simulation)

> 3位审稿人 + AC Meta-Review + Rebuttal Preparation
> 评分: R1=5/10 Borderline Reject, R2=5/10 Borderline Reject, R3=6/10 Borderline Accept
> AC: Major Revision needed, leaning Reject

---

## Reviewer 1 (Theory & Foundations)

### Summary of Core Contribution

The paper reframes the RoPE frequency table as a finite spectral-allocation problem rather than an intrinsic part of the rotary operator. It introduces a quadratic collision surrogate over log-frequency densities, derives a unique cosh-shaped minimizer and closed-form inverse-CDF quantization, and then selects the practical parameter (\tau=d_{\mathrm{eff}}/\sqrt{L}) using a separate small-(\tau) softmax-transport argument. The resulting method changes only the fixed inverse-frequency table and adds no learned parameters. I reviewed the complete 41-page submission, including Appendices A–D and the checklist.

### Strengths

- The conditional variational result is clean and largely self-contained. Section 3.3 and Appendix A.1 establish existence, strict convexity, uniqueness, positivity, the derivative boundary conditions, and the normalized solution
  $$\rho_\tau(\phi)=\frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau}.$$
  The proof correctly handles the non-negativity constraint by showing that the unconstrained Euler solution is strictly positive.
- The inverse-CDF construction in Section 3.4 and Appendix A.3 is unusually convenient for deployment: the CDF is analytically invertible, geometric RoPE is recovered as (\tau\to0), and Equation (4) fully specifies the implementation.
- The paper is commendably explicit about epistemic status. Table 1, Section 3.7, and Appendix A.11 distinguish the exact result conditional on (C_{\mathrm{app}}), the practical pure-tether branch, and the semi-analytic rather than globally optimal status of the (\tau) rule.
- Appendix A.6 provides a relevant functional rather than pointwise validation of the surrogate: the EVQ allocation reduces the exact-kernel collision score by 24–92% and raises effective rank across the 12 listed configurations in Table 5. This is stronger evidence than merely fitting the smooth surrogate to an oscillatory kernel.
- Appendix A.19 turns the qualitative "scarce channels matter more" intuition into explicit (K^{-1}) transport and (K^{-2}) high-resolution distortion terms. Although the trained-model validation is not yet controlled enough, this is a useful formal direction.

### Weaknesses / Major Concerns

- The theory does not derive the complete deployed method from a unified objective. The minimizer of the stated surrogate uses (\tau_{\mathrm{surr}}=\sqrt{\beta/\alpha}), but Appendix A.11 explicitly finds (\tau_{\mathrm{surr}}\sim\sqrt{d_{\mathrm{head}}}L^{-0.11}), not the deployed (d_{\mathrm{head}}L^{-1/2}). The latter is obtained from a different stiffness–utility objective. Thus, the variational surrogate supplies the cosh shape, while a separately motivated model supplies the scale. This is legitimate as engineering, but substantially weaker than a single variational derivation of EVQ-Cosh.
- The central surrogate remains weakly grounded in the actual RoPE kernel. Section 3.2 posits a constant diagonal plus (\min(\phi,\psi)) Green kernel; Appendix A.5 acknowledges that the continuum stationary-phase diagonal is exponentially (\phi)-dependent and leads to a different Bessel-type solution. Constant (\alpha) is selected mainly for positivity, inverse-CDF tractability, and discrete fitting. The paper does not report fit residuals for ((\alpha,\beta)), sensitivity to the fitting objective, or validation under realistic causal/empirical distance priors rather than primarily the uniform separation prior.
- Proposition 1 depends materially on modeling choices that are not yet empirically verified. The diffuse baseline (p_0=1/L), the Pearson-(\chi^2) stiffness, the (d_{\mathrm{head}}) normalization, and the utility coefficient all determine the scaling. Table 8 shows substantial finite-range sensitivity to the stiffness functional, and Appendix A.15 explicitly leaves direct measurement of (L_{\mathrm{eff}}^J) as a future falsification test. The paper therefore derives a plausible structural exponent under assumptions, not a robust theorem about trained attention.
- The pure-tether approximation is not demonstrably controlled in the regimes where the strongest PE-dominant result uses (\tau=4). Appendix A.16 shows only (L^1)/CDF suppression of the Fisher-forcing branch, with constants depending on an unmeasured Fisher coefficient, and notes inverse-CDF amplification by (\sinh\tau/\tau). The prescribed diagnostic (R_F) is not reported for the primary experiments, nor is there a trained pure-tether-versus-forced-branch ablation.
- The MLA operating convention is particularly under-derived. Appendix C.1 uses (d_{\mathrm{eff}}=d_{\mathrm{head}}=128), although only (d_{\mathrm{rope}}=32) dimensions participate in RoPE and the surrogate diagonal depends on (d_{\mathrm{rope}}). The authors explicitly identify (\tau=d_{\mathrm{rope}}/\sqrt{L}) as the natural missing sanity check. Without this ablation, the claimed architecture-specific rule is calibrated rather than theoretically supported.

### Detailed Section-by-Section Comments

- **Introduction:**
  The finite-spectral-budget framing is compelling and provides a useful conceptual separation among operator, range scaling, and training-time allocation. The sentence "the rotation is the operator; the frequency table is a design choice" captures the central contribution well. However, the language around a "derived" allocation should consistently preserve the distinction that only the family shape is derived from (C_{\mathrm{app}}), while the deployed parameter is selected from a separate model and empirical basin.
- **Related Work:**
  The three-axis organization in Section 2 is effective. The novelty claim is nevertheless established mainly by identifying an unoccupied taxonomy cell in Table 25. Occupying a cell is not itself evidence that the selected variational surrogate or cosh family is theoretically preferable. The paper would be stronger with a more analytical comparison to simple alternative allocation families and with a clearer discussion of whether tuned-base geometric RoPE can approximate the same active spectral region.
- **Method/Theory:**
  Sections 3.1–3.4 are the most convincing part of the theoretical presentation. Theorem 1 and Theorem 2 are straightforward but correct and practically useful.
  Section 3.6's "waterbed" result is mathematically an entropy/divergence lower bound for departing from uniform allocation. It does not itself prove that long-range improvement must trade against short-range degradation, nor does it connect the surrogate cost quantitatively to PPL. The interpretation should be narrowed accordingly.
  Section 3.7 should be presented as a model-based scaling hypothesis supported by sweeps. At (\tau=4), the small-(\tau) expansion is far outside a regime where the leading coefficients alone are quantitatively reliable, even if the empirical basin happens to be broad.
- **Experiments if applicable:**
  The experiments demonstrate that the schedule can matter, but they do not isolate whether the cosh variational form is responsible. Missing controls include a linear or power-law density, optimized geometric base, random/jittered schedules, and a numerically collision-optimized allocation trained end-to-end. Such controls are especially important because Appendix A.14's numerical "oracle" is not a reliable optimum and is worse than EVQ in one row.
- **Conclusion:**
  Section 5 is appropriately cautious about the surrogate and the scope of the diagnostics. I recommend replacing any implication that the full training-time rule is theoretically derived with the more precise claim that EVQ is a surrogate-derived allocation family with a semi-analytic, empirically validated operating rule.
- **Relevant Appendix Parts:**
  Appendix A.1 is rigorous and valuable. Appendices A.5–A.6 honestly expose and then functionally test the main approximation. Appendices A.10–A.12 contain important caveats that should arguably be moved into the main text, particularly the mismatch between the surrogate's own scale and the deployed scale, and the stiffness sensitivity in Table 8. Appendix A.15 is a promising route to stronger theory, but the absence of the proposed (L_{\mathrm{eff}}^J) measurements leaves the central trained-attention assumption untested. Appendix A.16 and Appendix C.1 identify concrete validity checks that should be executed rather than deferred.

### Scores

- Quality (technical soundness & rigor): 2.5/4
- Clarity: 2.5/4
- Significance & Originality: 3/4
- Overall Score: **5/10 – Borderline Reject**; the core variational calculation is sound and the framing is interesting, but the theoretical story for the deployed rule is not yet unified or sufficiently validated.

**Recommendation**: Borderline Reject
**Confidence**: High

---

## Reviewer 2 (Empirical ML, Experiments & Reproducibility)

### Summary of Core Contribution

The paper evaluates a fixed, zero-parameter change to RoPE frequency allocation in three main settings: a 454M MHA model combined factorially with YaRN, a highly PE-dominant 125M short-training-length diagnostic, and a 432M MLA model with only 16 rotary channels. The reported effects are large in several long-range metrics, especially EVQ+YaRN passkey retrieval and MLA extrapolation PPL. However, baseline tuning, replication of one primary result, internal result consistency, and reproduction details are not yet at the standard needed for a confident empirical acceptance.

### Strengths

- Primary I uses a useful (2\times2) design: Geo, Geo+YaRN, EVQ, and EVQ+YaRN, with the same stated YaRN scale and three seeds. Table 3 and Figure 2 show a large and apparently robust effect at 8K: 100% versus (61%) teacher-forced NLL-gap retrieval, with supporting PPL improvements.
- Primary III is also multi-seed and reports means and standard deviations at all lengths. Table 18 shows a substantial 16K reduction from (138.8\pm5.5) to (95.6\pm4.1), while the 8K in-range change is small.
- The authors separate primary, robustness, and supporting evidence in Table 2. This is good experimental hygiene, and the main abstract does not rely on the single-seed LoRA, progressive-training, or 750M results.
- The paper reports both in-range costs and extrapolation gains rather than only favorable long-context endpoints. Tables 20 and 24, as well as the MLA results, make the tradeoff visible.
- The appendices contain several useful mechanism probes: exact-kernel collision scores in Table 5, matched-scale YaRN leverage in Table 19, video base sweeps in Table 17, and attention-distance analyses in Figure 6.

### Weaknesses / Major Concerns

- The most important missing baselines are tuned geometric base and tuned range scaling. All primary text experiments use (b=500\text{K}), a regime the paper itself says creates large headroom for non-geometric allocation. The video sweep in Table 17 shows that when the base is small enough to keep channels alive, geometric allocation can win. A text-side base sweep is therefore essential. Similarly, using the same YaRN scale is not equivalent to giving Geo and EVQ equally optimized YaRN configurations; a scale sweep for both is required to support complementarity beyond one matched-scale point.
- Primary II is not sufficiently replicated. Geo, DAPE, and fixed EVQ in Table 4 are all seed 42, while only learnable (\tau) is three-seed. The paper calls this a primary anchor and compares EVQ to a learned operator, but a single retained seed cannot establish the reported 35% and 11.4% relative differences robustly. All methods should be run on the same seeds with uncertainty.
- There are serious internal figure/table inconsistencies. On PDF page 36, Figure 8 visually plots QA accuracy, is titled "EVQ Wins at All Context Lengths," and includes a 32K point, while its caption describes gold-answer NLL. Table 21 instead reports different accuracy values, no 32K row, and EVQ is slightly worse in accuracy at 16K. Figure 9 labels the 454M long-range gain as approximately (-81%), whereas Table 20 reports (-13.3%) for the three-seed 454M FineWeb-Edu row; the figure appears to mix a different progressive-training setting with the nominal cross-scale comparison. These are not cosmetic errors—they make it unclear which numbers support which claims.
- The manuscript alone is not sufficiently reproducible. Tables 10–11 omit exact model architecture configurations, tokenizer/vocabulary, training-token counts and steps for several primary runs, exact global token batch, evaluation-set sizes, passkey generation and placement protocol, NLL-gap decision rule, DAPE implementation details, and full YaRN parameters. Values such as "batch size 2–4" are not exact. The checklist says an archive exists, but the paper should still make the primary comparisons auditable.
- Longer-training evidence raises a substantive robustness concern. Appendix D reports that at 1B MLA training tokens the raw EVQ advantage reverses to (+11.1%), while EVQ+YaRN retains only a (-2.5%) single-seed gain. This may indicate that the 500M-token primary result is training-stage-dependent. A multi-seed saturation study is needed before claiming that the allocation advantage is intrinsic rather than transient.

### Detailed Section-by-Section Comments

- **Introduction:**
  The empirical questions Q1–Q3 are clear, and the intervention is appropriately described as minimal. However, calling the PE-dominant result a primary stress test while using a single seed for the principal Geo/DAPE/EVQ contrast is not aligned with the otherwise careful evidence hierarchy.
- **Related Work:**
  The positioning distinguishes training-time allocation from post-hoc range extension, but the empirical comparison set is narrower than the related-work discussion. No trained comparison is made to a simple searched per-channel allocation, clipping-style schedule, or alternative closed-form density. The paper therefore supports "allocation matters" more strongly than "EVQ-Cosh is the right allocation."
- **Method/Theory:**
  Equation (4) is simple enough to reproduce at the initializer level. The paper states that midpoint versus endpoint quantization changes PPL by less than 1% across tested (K\ge16), but the underlying table is not shown. More importantly, changing midpoint quantiles also changes the realized extremal frequencies; a range/endpoints-matched geometric control would help isolate density shape.
- **Experiments if applicable:**
  **Primary I:** The four-way comparison is strong, but the headline metric is teacher-forced NLL-gap retrieval rather than autoregressive exact match. The number of evaluation examples, insertion distribution, success threshold, and binomial uncertainty are not given. Multi-seed AR exact would materially strengthen the systems claim. The same-scale YaRN experiment should be accompanied by a Geo/EVQ scale sweep because YaRN partitions or rescales channels based on their frequencies, so "same scale" need not imply matched effective intervention.
  **Primary II:** The 128-to-8K setup is an intentionally extreme 64× diagnostic. That is acceptable for mechanism isolation, but it increases the need for replication, tuned-base geometric controls, and a complete DAPE reproduction. It is unclear whether "DAPE-style" uses the strongest configuration and training protocol from the referenced method.
  **Primary III:** Table 18 is one of the strongest results. Yet the claimed channel-scarcity mechanism is not isolated by comparing MLA at (K=16) with an MHA row at (K=64): architecture, training length, extrapolation ratio, and content pathway all change. A within-architecture sweep over (d_{\mathrm{rope}}) is needed. The missing (d_{\mathrm{rope}}/\sqrt L) operating-point ablation is also critical.
  **Robustness and Supporting Evidence:** Table 20 mixes datasets and training regimes across model sizes, so it should not be interpreted as a controlled scaling law. The video results are interesting, but "sharing identical weights and optimizer state" between allocations requires a much clearer description: such coupling controls randomness but may also cause interference between conditions. The LoRA result in Table 23 lacks a matched Geo+LoRA+LongAlign control and has a 30% in-distribution PPL penalty; it cannot isolate EVQ.
- **Conclusion:**
  The limitation language is appropriately cautious. It should explicitly foreground the absence of tuned-base text baselines, the single-seed DAPE comparison, and the longer-training MLA reversal, as these materially affect the empirical interpretation.
- **Relevant Appendix Parts:**
  Appendix B.2 is a helpful start but not a full reproduction specification. Appendix B.6's paired/shared-weight video protocol needs pseudocode and independent-run confirmation. Appendix C.1 should report the full (\tau) sweep, including (d_{\mathrm{rope}}/\sqrt L). Appendix D appropriately labels exploratory results, but the 1B-token reversal is important enough to discuss in the main limitations. Figures 8 and 9 on page 36 must be regenerated from the final tables and checked against their captions.

### Scores

- Quality (technical soundness & rigor): 2.5/4
- Clarity: 2/4
- Significance & Originality: 3/4
- Overall Score: **5/10 – Borderline Reject**; the effect sizes are promising, but baseline fairness, one-seed primary evidence, saturation uncertainty, and inconsistent reported results prevent a reliable empirical conclusion.

**Recommendation**: Borderline Reject
**Confidence**: High

---

## Reviewer 3 (Applications, Impact, Clarity & Broader Contribution)

### Summary of Core Contribution

EVQ-Cosh is an appealingly lightweight intervention: it replaces the fixed RoPE inverse-frequency schedule with a closed-form non-geometric schedule while preserving architecture, parameter count, and runtime structure. The strongest practical evidence is that it composes with YaRN and benefits an MLA architecture with a compressed rotary subspace. The current submission is best understood as a promising positional-mechanism study rather than a demonstrated improvement to general long-context applications.

### Strengths

- Deployment is unusually simple. Equation (4) requires only a one-time inverse-frequency calculation; there is no auxiliary loss, added parameter, learned module, or inference-time search.
- The paper clearly states what it is not claiming. Sections 1, 4.1, and 5 distinguish PE diagnostics from universal downstream performance, and Table 2 explicitly labels the evidence tiers.
- Compatibility with existing systems is a meaningful practical contribution. Primary I suggests EVQ can improve the substrate on which YaRN operates, while Primary III targets MLA, where rotary channels are particularly scarce.
- The supporting evidence tests multiple contexts rather than repeating one synthetic benchmark: MHA, MLA, progressive training, LoRA injection, and video DiT. The video base sweep is especially useful because it exposes a regime—base 100 with no dead channels—in which EVQ does not help.
- The ethical risk is low, and the authors correctly avoid equating positional-diagnostic gains with deployment reliability or safety guarantees in Section 5.

### Weaknesses / Major Concerns

- Application-level evidence is limited. The principal retrieval result is teacher-forced and trained with a 10% passkey mixture. QuALITY accuracy in Table 21 remains near random, and the 8B LoRA experiment reportedly does not improve RULER. There is no strong downstream result at a model scale capable of performing the task.
- Practical generality is uncertain. EVQ is most beneficial in large-base/dead-channel regimes; Table 17 shows Geo winning when all video channels are alive. The 1B-token MLA result in Appendix D also suggests raw EVQ performance can reverse after longer training. Users therefore lack clear guidance on when EVQ should or should not be enabled.
- The paper's conceptual message is simple, but the exposition is much more complicated than the method. Terms such as "pure tether," "Fisher forcing," "waterbed," "habitable zone," "epistemic stratification," and several overlapping dimensions ((d_{\mathrm{head}},d_{\mathrm{rot}},d_{\mathrm{eff}},d_{\mathrm{rope}})) make the paper difficult to navigate. A notation table and one practical decision diagram would help.
- The internal visual inconsistencies on page 36 substantially reduce reader trust. Figure 8's content, title, caption, and Table 21 do not agree; Figure 9 does not transparently correspond to Table 20. These issues are especially problematic in an application-facing paper where readers rely on plots for the empirical takeaway.
- The broader-impact discussion is too speculative. The paper suggests reduced wasted long-context compute, but it does not measure training or inference savings, memory reduction, latency, or task-level reliability. Since the initializer itself does not reduce nominal compute, the claim should be framed as a potential indirect benefit rather than a demonstrated impact.

### Detailed Section-by-Section Comments

- **Introduction:**
  The opening framing is memorable and accessible. The distinction between the RoPE operator and the allocated frequency table is likely to be useful to practitioners. The introduction could improve further by giving a one-sentence operational criterion, such as: EVQ is expected to help when a large fraction of geometric channels accumulates negligible phase over the training window.
- **Related Work:**
  The three-axis taxonomy is useful for readers outside the immediate RoPE literature. The section is nevertheless dense and citation-heavy. A compact visual showing operator changes, training-time allocation, and inference-time scaling would communicate the positioning more effectively than the long prose inventory.
- **Method/Theory:**
  The deployable method should appear earlier and more prominently. An algorithm box containing five lines—choose (d_{\mathrm{eff}}), compute (\tau), create quantiles, apply the inverse CDF, replace `inv_freq`—would make the paper substantially more accessible. The paper also needs practical guidance for nonstandard architectures, because the MLA and video corrections are currently empirical conventions.
- **Experiments if applicable:**
  The passkey and extrapolation-PPL results are appropriate mechanism diagnostics, but they do not yet establish real-world long-context understanding. A convincing application evaluation would use a model with non-trivial baseline accuracy and report generation-level outcomes, not only gold-answer NLL.
  The 750M AR exact result is promising but single-seed. The 8B experiment would be more relevant with a matched Geo+LoRA control and a task suite on which the adapted model is actually instruction-following at 16K–32K.
  The video experiments are an interesting cross-modal check, but Oscillating Moving MNIST is far from modern video generation. The audit of dead temporal channels in Table 16 is useful motivation, yet it is not a substitute for multi-seed evaluation on a realistic video benchmark.
- **Conclusion:**
  The conclusion appropriately calls the work a mechanism study. It should give concrete deployment boundaries: supported base range, tested (\tau) range, warning conditions from Appendix A.16, and the uncertainty for long or saturated pretraining.
- **Relevant Appendix Parts:**
  Appendices B–D contain valuable practical details that should be distilled rather than simply accumulated. Table 17 is particularly important because it reveals failure conditions. Table 23 should not call a 30% in-distribution PPL increase "modest" without stronger downstream benefits. The page-36 plots require correction before publication.

### Scores

- Quality (technical soundness & rigor): 3/4
- Clarity: 2.5/4
- Significance & Originality: 3/4
- Overall Score: **6/10 – Borderline Accept**; the intervention is simple, potentially useful, and honestly scoped, but the current real-world evidence and presentation quality are not yet fully convincing.

**Recommendation**: Borderline Accept
**Confidence**: Medium

---

## Area Chair Meta-Review

### Overview of Reviewer Consensus and Key Disagreements

All reviewers agree that the paper contains a compelling central idea: RoPE's geometric frequency table is a design choice, and a fixed non-geometric allocation can materially alter long-context behavior without adding parameters. They also agree that the closed-form inverse-CDF implementation is elegant, that the authors are unusually transparent about evidence tiers, and that the three-seed EVQ+YaRN and MLA results are promising.

The main disagreement concerns whether this is already sufficient for acceptance as a mechanism paper. Reviewer 3 gives substantial credit to deployability, honest scope, and the observed gains. Reviewers 1 and 2 place more weight on the disconnected shape/scale derivation, insufficiently tuned controls, the single-seed DAPE contrast, and internal result inconsistencies.

### AC's Overall Assessment of the paper's contribution to the field

This is a potentially important contribution. The finite-frequency-budget perspective is likely to influence how practitioners think about RoPE, and the method is simple enough to have practical uptake. However, the current submission does not yet establish that EVQ-Cosh, rather than non-geometric reallocation more generally or correction of an oversized base, is responsible for the observed advantages. The theoretical results rigorously solve the chosen surrogate, but the surrogate does not derive the deployed parameter scale, and several key assumptions remain unmeasured in trained models. Empirically, two anchors are strong, but the baseline set and reporting consistency are not adequate for a definitive NeurIPS claim.

### Strengths that the AC finds most compelling (with references)

- The exact convex variational solution and elementary inverse-CDF map in Sections 3.3–3.5 and Appendix A.1.
- The minimal implementation and zero-parameter nature of Equation (4).
- The large three-seed matched-scale EVQ+YaRN effect in Table 3 and Figure 2.
- The three-seed MLA improvement in Table 18, which motivates further work on allocation under compressed rotary subspaces.
- The explicit epistemic map in Table 1 and evidence-tiering in Table 2.
- The functional exact-kernel validation in Appendix A.6 and the failure-regime evidence in the video base sweep of Table 17.

### Critical Weaknesses that must be addressed (with references)

- The deployed (\tau=d_{\mathrm{eff}}/\sqrt L) is not the parameter obtained from the Section 3.2 surrogate; Appendix A.11 explicitly documents the mismatch. The paper needs either a more unified theoretical account or more modest framing of what is derived.
- No text-side tuned-base geometric baseline or full Geo-versus-EVQ YaRN-scale sweep is reported. These are necessary to distinguish allocation shape from correction of base/range mismatch.
- The PE-dominant Geo/DAPE/EVQ comparison in Table 4 is single-seed. The baseline fidelity and statistical robustness of this primary claim are therefore uncertain.
- The MLA mechanism is not isolated through a controlled rotary-channel-count ablation, and the (d_{\mathrm{eff}}=d_{\mathrm{head}}) convention lacks the sanity check explicitly requested in Appendix C.1.
- The 1B-token MLA reversal in Appendix D raises uncertainty about whether the benefit persists under training saturation.
- Figure 8, Figure 9, Table 20, and Table 21 are internally inconsistent. All result-bearing figures must be regenerated from an auditable source of truth.
- The manuscript's reproduction specification is incomplete without relying heavily on the claimed supplementary archive.

### AC Recommendation and justification

**Major Revision needed.** The idea is strong enough to merit serious reconsideration, but the current evidence does not cleanly separate EVQ-Cosh from simpler allocation/base alternatives, and the reporting inconsistencies are too significant to overlook. Under a binary NeurIPS decision, I would currently recommend **Reject**, with encouragement to resubmit after the controlled baselines, multi-seed replication, and result audit are completed.

### Simulated Program Committee Decision

**Borderline, leaning Reject.** The committee would likely view the work as promising and potentially high-impact, but not yet sufficiently mature or internally consistent for acceptance in its present form.

---

## Potential Rebuttal Questions & Preparation Guidance

### Q1: How should reviewers reconcile the variational parameter (\tau=\sqrt{\beta/\alpha}) with the deployed rule (\tau=d_{\mathrm{eff}}/\sqrt L)?

**Underlying concern:** Appendix A.11 shows that the original surrogate predicts a materially different scaling, so the full method is not the minimizer of the stated variational problem.

**Preparation:** Clearly separate the shape theorem from the operating-point model. Provide either a combined objective deriving both, or a concise claim revision stating that the cosh family is variationally derived while (\tau) is semi-analytic and empirically selected.

### Q2: Why is the constant-diagonal-plus-(\min)-kernel surrogate the appropriate model of RoPE collision?

**Underlying concern:** Appendix A.5 derives a different (\phi)-dependent diagonal from stationary phase, and no quantitative surrogate-fit errors are reported.

**Preparation:** Report the fitting objective, fitted (\alpha,\beta), residuals, and allocation rankings under multiple priors: uniform, causal triangular, passkey-specific, and empirical attention-weighted distance distributions.

### Q3: Does the (L^{-1/2}) rule survive direct measurement of trained-attention curvature?

**Underlying concern:** Proposition 1 assumes diffuse attention, while Appendix A.15 introduces (L_{\mathrm{eff}}^J) precisely because trained attention may not be diffuse.

**Preparation:** Measure (L_{\mathrm{eff}}^J) or at least its entropy proxy on representative checkpoints and show whether it scales approximately with (L_{\mathrm{train}}) in the primary settings.

### Q4: How sensitive is the operating rule to the selected stiffness functional?

**Underlying concern:** Table 8 shows that finite-range exponents vary substantially across reasonable divergences, and Pearson (\chi^2) does not exactly produce 0.5 under the reported finite-grid optimization.

**Preparation:** Provide trained-model sweep results for two or three alternative stiffness-motivated rules, or show that their predicted (\tau) values all fall within the same measured PPL basin.

### Q5: Is dropping the Fisher-forcing branch justified at (\tau=4)?

**Underlying concern:** Appendix A.16 warns that CDF errors can be amplified by (\sinh\tau/\tau), while the Fisher coefficient and (R_F) diagnostic are not measured.

**Preparation:** Report (R_F) estimates in the primary configurations and train at least one numerical forced-branch allocation against pure-tether EVQ at the PE-dominant operating point.

### Q6: Why does MLA use (d_{\mathrm{eff}}=d_{\mathrm{head}}) rather than (d_{\mathrm{rope}}), and what happens under the alternative?

**Underlying concern:** The surrogate normalization and number of rotary channels depend on (d_{\mathrm{rope}}), but the deployed (\tau) uses the full head dimension.

**Preparation:** Add the direct (d_{\mathrm{rope}}/\sqrt L) baseline, a local (\tau) sweep, and, ideally, intermediate dimension formulas such as (\sqrt{d_{\mathrm{head}}d_{\mathrm{rope}}/L}).

### Q7: Does the cosh allocation outperform other non-geometric allocation shapes?

**Underlying concern:** The experiments establish that changing allocation can help, but not that the variational cosh shape is specifically responsible.

**Preparation:** Compare against at least a linear density, power-law warp, random monotone perturbation, and a numerically collision-optimized schedule, all with matched endpoints and channel count.

### Q8: How does EVQ compare with a tuned geometric RoPE base in the primary text settings?

**Underlying concern:** The primary base (500\text{K}) may create dead channels that a smaller geometric base could reactivate; Table 17 demonstrates this effect in video.

**Preparation:** Run a text base sweep for Geo and EVQ, report best validation-selected values, and compare at matched spectral span or matched number of active channels.

### Q9: Would Geo close the gap if YaRN scale and other YaRN parameters were tuned separately?

**Underlying concern:** Equal YaRN scale is useful for an interaction test but does not establish superiority over equally optimized range extension.

**Preparation:** Provide a factorial scale sweep for both Geo and EVQ, including validation-selected scales and an interaction analysis with confidence intervals.

### Q10: Can the PE-dominant DAPE comparison be replicated across matched seeds?

**Underlying concern:** The fixed Geo/DAPE/EVQ rows in Table 4 use only seed 42, making the primary learned-operator comparison statistically fragile.

**Preparation:** Run all methods on at least three identical seeds, report mean, standard deviation, and paired seed differences, and document DAPE-specific hyperparameter tuning.

### Q11: What exactly constitutes a successful "teacher-forced NLL-gap retrieval," and how does it correlate with autoregressive exact match?

**Underlying concern:** The headline 100% result is not a generation-level success metric, and the evaluation-set size and threshold are unspecified.

**Preparation:** State the decision rule, number of examples, passkey positions and lengths, confidence intervals, and multi-seed AR exact results for the 454M primary model.

### Q12: Does the MLA gain persist after training saturation?

**Underlying concern:** Appendix D reports that raw EVQ becomes 11.1% worse at 1B tokens in one seed, potentially contradicting the interpretation of an intrinsic allocation benefit.

**Preparation:** Replicate the 1B-token result over multiple seeds, show learning curves with uncertainty, and distinguish raw EVQ, EVQ+YaRN, and any post-training adaptation.

### Q13: Can channel scarcity be isolated within one architecture?

**Underlying concern:** Comparing MHA and MLA changes more than the number of rotary channels, and the extrapolation ratios in the cited rows are not matched.

**Preparation:** Sweep (d_{\mathrm{rot}}) or (d_{\mathrm{rope}}) within an otherwise fixed architecture and test whether EVQ's gain scales monotonically with (1/K) or (1/K^2).

### Q14: How do the authors reconcile Figure 8 with Table 21 and Figure 9 with Table 20?

**Underlying concern:** The current plots appear to contain stale or mixed-setting results, undermining confidence in the numerical pipeline.

**Preparation:** Provide corrected figures generated directly from released result files, an errata table identifying every changed value, and a script that regenerates all manuscript tables and plots.

### Q15: What are the exact contents of the "99 trained models across 27 settings" sweep?

**Underlying concern:** Figure 4, Table 7, and the surrounding text are difficult to reconcile into a row-level accounting of configurations, seeds, and (\tau) candidates.

**Preparation:** Release a flat table with one row per run: architecture, dataset, seed, (L), (d_{\mathrm{head}}), base, (\tau), token count, validation PPL, evidence tier, and selection status.

### Q16: Can the authors provide a complete, manuscript-level reproduction specification for the primary anchors?

**Underlying concern:** The current appendix omits exact architectures, tokenizers, total steps/tokens, global token batches, evaluation sizes, scoring details, and some baseline configurations.

**Preparation:** Add full configuration tables and exact commands or config-file references for Primary I–III, including all YaRN and DAPE parameters.

### Q17: What evidence supports practical long-context benefit beyond positional diagnostics?

**Underlying concern:** QuALITY accuracy is near chance, the LoRA experiment lacks a matched control, and RULER reportedly does not improve.

**Preparation:** Add a matched Geo+LoRA control and at least one capable-model evaluation with non-trivial baseline task performance, reporting both generation-level accuracy and probabilistic metrics.
