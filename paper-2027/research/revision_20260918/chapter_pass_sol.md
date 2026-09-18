# Sol: independent PDF comparison

Model: gpt-5.6-sol. Fresh context. Identical prompt: `chapter_pass_review_prompt.txt`.
Inputs: the two frozen PDFs identified in `chapter_pass_identity.json`.
The reviewer volunteered a rating; the prompt requested neither a score nor a preferred outcome. This rating is not treated as a verified venue-rubric assessment.

Recommendation: **Accept (7/10), confidence 4/5.** The revision is a net improvement with no blocking regression or newly introduced technical error.

### Novelty and significance

- **Strong central contribution — pp. 2–3, §§2–3.** Fixed-support experiments isolate frequency placement; equal-displacement controls further separate placement shape from total movement. Supporting evidence includes OLMo’s 0.56/61.04/60.47 fixed-range comparison, the equal-displacement BM–Uni result, and TailSpline–C at 32K (+2.10 pp, CI [1.11, 3.08]). Contrary evidence is properly retained: TailSpline–C is neutral at 16K, and Cosh is indistinguishable from the matched exponential family in the short factorial. This supports allocation as a genuine design variable, but not a universally preferred shape. **Small repair:** state in the main discussion that the preferred allocation is length- and support-dependent.
- **Meaningful distinction from prior frequency modification — pp. 3–5, §4.** The manuscript separates spectrum, coordinate assignment, and complete-pair positional overlap. The crossed-table and spectrum-permutation interventions support learned frequency-coordinate compatibility. The contribution is significant because it turns “changing RoPE frequencies helps” into a more precise attribution claim. No correctness repair is needed.

### Theory and method

- **Sound, appropriately scoped geometry — pp. 3–5; Appendix A, pp. 14–17.** The complete sine–cosine subspace measure, whitening invariance, exact effective-rank identity, and slow-frequency convergence form a coherent analysis. The manuscript also presents contrary evidence: lower positional effective rank can accompany better task performance, so rank is explicitly treated as diagnostic rather than a selection rule. This honesty strengthens the contribution. **Small repair:** repeat the “diagnostic, not optimizer” limitation once in §4 rather than leaving the sharpest caveat to Appendix A.6.
- **Transparent TailSpline construction — pp. 5–6; Appendix B, pp. 17–20.** The one-sided log-gap objective has a unique positive closed-form minimizer; the equal-displacement control is exact; the rendered Eqs. (7)–(10) and supporting derivations are intact. The phase results explain approach to a chosen stretched-distance reference, but do not prove task optimality; Appendix B correctly says so. The one-sided boundary condition remains a design prior. **Small repair:** call it an explicit boundary prior in §5.1 and say the phase theorem motivates the profile without predicting downstream accuracy.
- **Important method gap — NCP, Appendix E.1, pp. 30–31.** The objective is reproducible, but the choices λ=9/4, shift bound 2/9, and 32 Fourier modes receive little selection rationale or sensitivity analysis. This narrows confidence in the native-window construction, whose gains are smaller and task-dependent. **Small repair:** add an objective-only rationale or a compact neighboring-parameter sensitivity table, clearly separated from task fitting.

### Experimental and practical value

- **Unusually broad positive evidence — pp. 7–9, §6; Appendices D–F.** TailSpline is tested across Llama, OLMo, Qwen, GLM, Kanana, and 70B NF4; large Llama/OLMo panels confirm the strongest effects. Results include retrieval, pooled language modeling, natural QA, fixed-support training, continuation, and adaptation. Matched inputs, paired uncertainty, complete task tables, source-cluster resampling, and equal-displacement controls make the core evidence persuasive.
- **Material boundary conditions are mostly confined to the appendix — pp. 30, 32–34.** At higher extension factors, Llama book-QA decreases and Qwen long-book PPL worsens; several learned Cosh studies trade native-window quality for extrapolation, including the 8B LoRA 8K result. Natural-QA evidence is also mixed: OLMo is the clearest positive case, while most other intervals include zero. These results do not invalidate the central s=4 claim, but they limit any universal practical interpretation. **Small repair:** add one sentence to §8 summarizing task-, length-, and native-quality tradeoffs; calibrate the abstract’s natural-QA wording to “point gains, with the clearest improvement on OLMo.”
- **Reproducibility gap, moderate — p. 10 and appendices.** Algorithms, grids, datasets, seeds, sample counts, scoring, and many precision details are strong. The PDF itself lacks a compact compute/runtime/software table for the large training and 70B evaluations, and some supplementary results lack clustered uncertainty, such as Kanana 128K and the 70B s=16 PPL panel. **Small repair:** add one appendix table for accelerator/runtime/software information and intervals where source-level replication permits them.
- **Optional extension:** Larger direct panels for Qwen and Kanana, and a clearer cost/assumption comparison against search- or learning-based long-context methods, would strengthen external positioning. The existing large Llama/OLMo confirmations are sufficient for the current core claim.

### Narrative and clarity

The revision improves the main-paper story. The new Figure 1 on p. 2 explains allocation as channel-wise distance stretch before introducing the algebra; the equal-displacement result now appears immediately after Figure 4 on p. 7; same-target context-use evidence leads §6.2; and several causal-sounding phrases were weakened appropriately. The renamed “complete-pair effective-rank identity” is also more precise.

The remaining narrative issue is breadth: TailSpline, Cosh, and NCP serve different operating regimes, but readers must reconstruct that map. **Small repair:** add a three-row summary mapping each construction to frozen extension, learned extrapolation, or native-window intervention, along with what is fixed and the primary endpoint.

### AC synthesis

The paper offers a clear conceptual contribution, exact and carefully limited theory, and extensive controlled evidence. Its strongest claim—interior frequency allocation matters beyond support and total displacement—is directly tested. Practical effects are broad at the principal extension regime, while failures and conditional results remain visible in the appendices. The remaining issues concern scope calibration, NCP hyperparameter rationale, and surfacing limitations, rather than validity.

### Regression audit

- **Figure 1 replacement, p. 2:** The new mechanism figure is more legible and better connected to TailSpline. The old empirical fixed-support and crossed-table panels no longer have main-text visual prominence, but their numbers remain in §§3.1–3.2 and Appendix C/Table 6. This is a minor presentation tradeoff, not lost evidence. A compact numeric callout would recover the old figure’s immediacy.
- **Equal-displacement evidence, pp. 7–8:** Moving it before direct-baseline discussion strengthens the contribution-to-control link.
- **Claim calibration, pp. 5, 8, 30:** “Establishes value,” “application-level value of improved retrieval,” and the result-leading NCP sentence were replaced by more defensible wording.
- **Theory wording, p. 14:** The invariance explanation and theorem name are clearer without changing substance.
- **Visual integrity:** All changed figures, tables, and rendered formulas inspected cleanly. No clipping, overlap, malformed equation, missing table, page-count change, or substantive numerical loss was found.
