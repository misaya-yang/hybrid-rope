# Independent manuscript review — Round 1

**Manuscript:** *Beyond the Base: Frequency Allocation in RoPE*  
**Review type:** Author-requested internal simulation using the supplied ICLR 2027 review criteria; not an official conference review.  
**Input boundary:** Only the frozen 59-page `input.pdf` was consulted. I read all nine main pages, examined the supplementary material, and visually inspected every main page and its figures. I did not consult repository sources, source archives, previous reviews, memory, or external literature. Page numbers below are PDF page numbers, which match the printed numbers. Small arithmetic checks used only formulas printed in the PDF.

## 中文执行摘要

论文最有说服力的贡献是：通过固定频率端点的配对实验，证明 RoPE 的内部频率分配确实具有独立价值；TailSpline 在 Llama 的 2,600 条干净 RULER 配对输入上，相比 MrRoPE-Pro 提升 11.72 个百分点，证据明确。主要数学结论在声明的假设下成立。

最值得修改的三点是：①“原生窗口代价低于 3%”只能表述为当前样本的点估计，置信区间并未排除更大退化；②正文应直接报告等总位移 T/C 对照尚无明确优势，而且运行批次不一致；③将 TailSpline 的实用性结论限定到实际测试的模型、比较方法和任务，不能用 BM 的自然问答结果替代 TailSpline 的证据。建议为 **弱接收，内部评分 6/10，置信度中等**；该判断基于控制实验提供的新知识，并不要求达到 SOTA。

## 1. Summary and assessment

The paper asks whether the interior placement of RoPE frequencies matters after their sampled endpoints are fixed, and how this freedom can support learning and frozen extension. It separates support, total native-relative displacement, allocation shape, amplitude, and frequency-to-coordinate assignment. It then offers two explicit constructions: Cosh, obtained from a continuous density objective, and TailSpline, obtained by smoothing displacement increments at the slow-band junction.

The strongest scientific result is the controlled identification of allocation as an independently useful variable, including its interaction with support policy. The strongest deployment result is the clean Llama-3-8B-Instruct RULER comparison against MrRoPE-Pro. The paper is less conclusive about why TailSpline's particular shape is preferable to simpler same-dose profiles, or how broadly its deployment advantage transfers.

| Criterion | Assessment |
|---|---|
| Specific question | Clear and experimentally addressable. “Same range” is defined using actual sampled endpoints. |
| Motivation and positioning | Sensible. The paper acknowledges that frequency learning and rescaling already exist; its distinction is the controls plus explicit constructions. |
| Correctness and evidence | Main algebra appears sound. Primary paired experiments support bounded claims; some headline wording outruns the uncertainty, and several secondary studies are exploratory or runtime-confounded. |
| Significance/new knowledge | A useful, moderate contribution: separating allocation from support and learned assignment is informative even without a best-in-class method. The practical advantage of the particular TailSpline prior remains comparator-dependent. |

## 2. Strengths

1. **The main identification experiment is well controlled.** Section 3.1, pp. 2–3, and Appendix B.1, pp. 24–25, hold initialization, data order, optimizer, token budget, and actual endpoints fixed within three training-seed pairs. Table 6 reports every seed rather than pooling evaluation anchors as independent training replications. The reversal under retargeted support is valuable negative evidence, not an inconvenience hidden from the main argument.

2. **The clean deployment result is substantial and reasonably transparent.** Section 6.2, p. 7, and Appendix H.7, pp. 52–53/Table 39, report 2,600 paired prompts, all 13 task means, output-health counts, a task-equal estimand, paired uncertainty, and leave-one-task-out checks. The +11.72-point contrast with interval [10.32, 13.11] is not explained by one isolated task. The negative multivalue task remains in the aggregate.

3. **The theory generally respects its scope.** The full sine–cosine treatment avoids a phase-dependent cosine-only collision criterion. Proposition 1 and the effective-rank identity, p. 4/Eqs. (4)–(5), are supported by the expansions and Gram calculations in Appendix A.1–A.4, pp. 14–18. The discrete kernel criterion is correctly limited to exact bilinear equivalence and frequencies in (0, π). The paper does not mistake it for a theorem about inability to adapt after training.

4. **The constructions are explicit and inexpensive.** Cosh's positive unique density minimizer follows from the stated strictly convex objective (pp. 20–22, Theorem 8). TailSpline's finite-grid solution follows from its positive-definite quadratic form (p. 49, H.1). I checked the printed TailSpline normalization, positivity, objective value, and equal-dose identity for n = 1, 2, 17, and 18 with exact arithmetic. These checks support the construction formulas, not their model quality.

5. **The appendix preserves failures and protocol differences.** Examples include the Cosh support-policy reversal, the matched exponential result, BM's Qwen losses, unsuccessful adaptation/readout outcomes, and the response-count counterexample. The distinctions among official YaRN, a historical fixed-index operator, and the MLA wavelength blend are especially useful (p. 30, D.2).

## 3. Prioritized decision-relevant weaknesses and repairs

### W1 — Native retention is presented more strongly than its uncertainty supports

**Type:** Claim-calibration/evidence gap. **Priority:** High.

**Locators:** p. 1, introduction lines 43–45; p. 8, “Native-window trade-off”; p. 9, conclusion; p. 54, Appendix H.8, lines 2869–2874.

The 2.33% relative reduction is an accurate point estimate: Native scores 91.8846% and TailSpline 89.7436% on 130 classic 8K prompts. However, the paired TailSpline-minus-Native interval is **[−6.1410, +1.9231] percentage points**, and no noninferiority margin was specified. The repeated “below 3%” formulation can therefore be read as a demonstrated retention guarantee when the experiment has not excluded materially larger losses. The 0.37% PPL increase is a different endpoint and cannot close the task-retention uncertainty. Also, the native measurement is a classic padded panel, whereas the clean long-context gain is measured under another input construction.

**Smallest repair:** Describe an **observed 2.33% relative reduction on the 130-prompt native panel**, put its paired interval next to the point estimate in the main text, and explicitly identify the different native/long-context panel contracts. Remove threshold-like language from the introduction and conclusion. Additional evaluation is needed only if the authors wish to establish a specified retention margin; wording can be corrected immediately without new model runs.

### W2 — The main text explains TailSpline's equal-dose hypothesis but omits its inconclusive outcome

**Type:** Evidence gap and narrative imbalance. **Priority:** High.

**Locators:** p. 6, Eqs. (12)–(13) and “Separating shape from total displacement”; p. 7, Table 1; p. 49, H.2; p. 53, H.8, lines 2850–2861.

The main text spends substantial space on a narrow hypothesis that the T/C exchange improves task utility. The measured primary contrast appears only in H.8: **T minus C Full-13 AUC = −0.4103 points, interval [−2.6282, +1.8205]**. Moreover, T uses batch 1 and C batch 2 with reordered batches; legacy backend/revision fields are incomplete. The 39-row replay is a useful diagnostic but cannot establish runtime equivalence for the complete comparison.

This does not refute the paper's general allocation claim: the separate BM/Uni same-dose comparison in G.8 supports that claim. It does mean that the clean T/MrPro gain cannot establish a benefit attributable to TailSpline's residual shape beyond total displacement, and the displayed interval does not establish T/C equivalence either. The one-sided smoothness objective remains a valid construction prior whose specific task advantage is unresolved.

**Smallest repair:** Add the T/C point estimate, interval, and runtime qualification to the paragraph on p. 6 or the results on p. 8. Label E1 a diagnostic in Table 1. Reconcile H.2's future-tense “would isolate”/“No T–C task result is included here” with the completed diagnostic in H.8. If a stronger residual-shape claim is retained, complete the comparison under a common runtime; a positive result should not be presumed or required. Do not replace Full-13 with the more favorable NIAH secondary endpoint.

### W3 — The practical TailSpline claim has a narrower comparator and transfer base than the surrounding evidence volume suggests

**Type:** Experimental completeness/significance gap. **Priority:** Medium–high.

**Locators:** pp. 7–8, Section 6.2/Fig. 4; p. 46, G.5/Table 33; p. 48, G.8/Table 35; p. 53, H.7, lines 2844–2846.

The large clean result evaluates one checkpoint, one long length, and MrRoPE-Pro. That is a legitimate and useful result. The older YaRN comparisons concern other profiles and panels; BM's five-task natural-QA gain is also a different method. Neither supplies a direct clean TailSpline/YaRN comparison or natural-QA transfer for TailSpline. The second TailSpline model uses the classic depth-selected, prefix-padded contract and a much weaker MrPro reference. The paper commendably discloses these distinctions, but they constrain how generally the deployment recommendation can be read.

**Smallest repair:** State the tested model, scale, and comparator in the abstract's deployment sentence, and retain the distinction between the clean Llama endpoint and classic cross-model curves. For a stronger practical claim, the highest-value bounded addition is a correctly implemented frozen YaRN arm on the existing clean prompts with the same decoder and explicit gain policy. This is a useful calibration against an established alternative, not a demand for SOTA, many new models, or paid training. Natural-QA evaluation of TailSpline is a further extension only if broader natural-output claims are pursued.

### W4 — Cosh establishes a useful family, but the evidence does not select its exact objective or operating rule

**Type:** Theory-to-method evidence limitation. **Priority:** Medium.

**Locators:** p. 5, Eq. (6) and strength discussion; p. 23, A.13/Eqs. (43)–(48); p. 27, B.3/Table 8; pp. 34–35, E.1/Table 17.

The paper correctly says that the geometry does not uniquely derive either objective (p. 4). The supporting factorial reinforces this limitation: reference Cosh versus the deformation-matched exponential is +0.00074 NLL with interval [−0.006, +0.008]. The reference rule improves 7/12 configuration means; its own aggregate interval includes zero. That factorial uses only 8.39M training tokens and four evaluation windows per length on repeated WikiText-2, so it is a short-run diagnostic rather than strong evidence for a universal strength law. The larger MLA result is useful, but its unanchored midpoint table changes support, and the best extrapolation still has high absolute PPL.

**Smallest repair:** Say plainly that Cosh is **one effective explicit allocation family**, and that the reference strength is an empirical operating choice supported in particular protocols, not an optimizer of model loss. Put the “matched exponential performs similarly” result into the main summary of Appendix B.3. Keep the larger MLA result as learning utility, as Table 1 already does. A new broad sweep or learned-frequency benchmark is unnecessary for this calibrated contribution.

### W5 — A few central empirical protocols are not reconstructible from the PDF alone

**Type:** Reproducibility/clarity gap. **Priority:** Medium.

**Locators:** pp. 20 and 54–55, A.7/I.3/Fig. 15; p. 34, E.1, lines 1816–1826; p. 53, H.8.

The slot-permutation result appears in the first main figure, but the PDF does not specify the actual permutation or its generation rule, the exact installed reference table for each model, and a complete OLMo evaluation sample/length contract. “Permuting interior slots” defines a class of interventions, not the particular one producing 3.10423 → 6.86493 NLL. The MLA evaluation description identifies a local cache filename and token count but does not identify its public split/revision and construction as precisely as B.1 does. H.8 explicitly acknowledges missing runtime identity fields.

**Smallest repair:** Add a compact specification of the permutation vector or deterministic rule, reference table, input sample counts, evaluation length, and scoring window; document how the MLA cache was constructed from a public split. State unresolved legacy identity fields explicitly rather than implying a fully replayable experiment. The source archive may contain further details, but I did not inspect it and therefore make no claim that these artifacts are absent there.

## 4. Genuine errors versus unresolved evidence

- **No fatal mathematical error identified.** The main density optimum, finite-grid TailSpline solution, effective-rank identity, and discrete equivalence argument are consistent with their declared assumptions. The slow-frequency limit concerns normalized positional subspaces; it is not a claim that learned attention loses exactly the same effective dimension.
- **No arithmetic error in the headline clean score or native point estimate identified.** W1 concerns uncertainty and presentation, not an incorrect subtraction.
- **Editorial inconsistency to resolve:** H.2 on p. 49 reads as though the T/C task comparison remains absent, while H.8 reports a completed diagnostic. A direct forward reference and the diagnostic label would remove the ambiguity.
- **Unresolved empirical hypotheses:** superiority of T's equal-dose residual shape, a native retention margin, and TailSpline transfer to natural QA. Their current uncertainty should not be described as either a proven success or a demonstrated failure.

## 5. Clarity, title, and optional improvements

The title is appropriate for the central identification question and avoids an unsupported optimality claim. The abstract is readable but would be more informative with the actual clean Llama contrast and an explicit statement that the two constructions use different design objectives. Main-page layout and figures are legible; Fig. 4 correctly separates the clean endpoint from classic curves. I found no main-text page-limit issue: the main material occupies nine pages, and references and appendices follow. The AI-use statement is present.

Optional improvements:

- Give the primary claim-to-result map a visible column for “supported,” “conditional,” and “unresolved.” The appendix glossary helps, but readers currently must traverse many protocols to learn the status of the T/C result.
- Identify uniform-separation geometry as a diagnostic in one short main-text clause. Appendix A already explains the effects of centering, feature scale, and changing the separation measure; repeating all that theory is unnecessary.
- Reduce repeated historical summaries and replace them with direct cross-references where space is needed for the uncertainty and T/C result. The many supporting studies should not obscure the handful of experiments that determine the paper's conclusion.

## 6. Questions for the authors

1. Will “below 3%” be restricted to the observed relative point estimate, or is a formal native-retention margin part of the intended claim?
2. Given H.8, is TailSpline presented as a useful analytic construction beating MrPro, or as evidence that its equal-dose residual exchange improves task utility? The current data support the former more clearly.
3. Can the clean Llama panel support a direct established-baseline comparison, or will the deployment claim remain explicitly MrPro-specific?
4. What exact reference tables and deterministic slot permutation produced Fig. 1(c)/Fig. 15(c,d), and how was the MLA evaluation cache constructed?

## 7. Recommendation and confidence

**Recommendation: Weak accept / borderline positive.**  
**Internal review score: 6/10** — this is an internal simulation scale, not a claim about the official ICLR form.  
**Confidence: Moderate (3/5, internal).**

The controlled evidence that interior allocation matters, the clear support-policy interaction, and the substantial clean TailSpline/MrPro result constitute useful new knowledge. The theory is mostly an explicit characterization and construction toolkit rather than a predictive account of model quality; the paper largely acknowledges that. I would not reject solely because Cosh resembles another useful deformation or because TailSpline has not achieved SOTA.

The highest-value revisions are claim calibration and exposing the current control outcome. A clean established-baseline comparison would strengthen the practical significance most efficiently. A stronger acceptance recommendation would require greater confidence in method-specific utility or deployment breadth; the present evidence does not justify a general retention guarantee or an established equal-dose TailSpline advantage.
