# Astra: independent PDF comparison

Model: gpt-6-astra. Fresh context. Identical prompt: `chapter_pass_review_prompt.txt`.
Inputs: the two frozen PDFs identified in `chapter_pass_identity.json`.

## Overall assessment

The revision preserves the scientific evidence and makes the argument easier to follow. I found **no newly introduced material correctness error or lost experimental result**. Its strongest contribution is the controlled identification of internal allocation effects, supported by useful frozen-model constructions—not a universal theory of optimal frequencies.

### 1. Novelty and significance

**Strength.** Sections 2–3 distinguish frequency range, total displacement, interior shape and coordinate assignment. The equal-displacement Llama comparison is particularly consequential: +2.10 points at 32K with interval [1.11, 3.08], while the 16K difference is negligible (§6.1, p.7; Appendix D.3, pp.24–25). This supports a specific allocation effect beyond merely moving frequencies farther.

**Boundary.** Frequency modification itself is established prior work, as §7 acknowledges. The contribution is the controlled decomposition plus public-parameter constructions. The revised introduction states this distinction adequately; no additional novelty disclaimer is needed.

### 2. Theory and method

**Strength.** The complete-pair analysis correctly separates positional redundancy from content-coordinate capacity (§4; Appendices A.1–A.6). The lower effective rank yet higher task quality of TailSpline is especially useful contrary evidence against interpreting rank as a quality objective. TailSpline’s discrete objective, closed form and terminal-phase comparison are internally coherent (§5.1; Appendix B).

The revision improves precision by replacing the loose whitening “cancellation” explanation with the orthogonal-factor argument (p.14), and by renaming the effective-rank theorem descriptively.

**Limitation, not an error.** The one-sided tail objective is a declared design preference. Its solution and phase bound do not prove task optimality; Appendix B.5 explicitly supplies the opposing native-distance trade-off. The paper already respects this distinction. A broader optimization theorem would be an optional extension.

I checked rendered formulas: the absolute value in Eq.39 and the radical in the NCP curvature bound are present. Apparent omissions in extracted text are **not manuscript errors**.

### 3. Experimental and practical value

**Strength.** The evidence is considerably broader than the small Full-13 baseline panels alone suggest: large paired RULER comparisons, 200-input-per-task retrieval baselines, 46-document language modeling, natural QA, complete-book evaluations and quantized 70B transfer. Native-window NCP also has a particularly informative same-target context-use comparison (§6.2; Appendix E.4).

**Presentation weakness, pre-existing — native-length cost.**
Location: §6.1, p.7; Appendix D.2, p.24.

- **Evidence:** “slight reduction” describes 90.03→85.16% at native 8K. The appendix reports −4.87 points [−6.57, −3.15], with FWE falling from 87.33 to 9.33 and 45 empty TailSpline responses.
- **Contrary evidence:** the aggregate numbers and failure are retained, and the other twelve tasks have a positive descriptive mean.
- **Effect:** the adjective understates a meaningful deployment trade-off in a paper emphasizing quality across the working window.
- **Smallest repair:** replace “slight reduction” with the numerical decline and a short reference to its concentration in FWE. No new experiment is required.

**Reproducibility gap, pre-existing — coordinate permutation.**
Location: Appendix C.6, p.23.

- **Evidence:** the PDF reports large performance changes from an interior-frequency permutation but does not specify the permutation, its generation seed, or the historical “log-law” table’s exact construction.
- **Contrary evidence:** it states that source identities remain in the numerical supplement; I cannot infer that the accompanying materials lack them.
- **Effect:** a PDF reader cannot reproduce this particular intervention, and different permutations can have materially different severity. The broader allocation result remains independently supported.
- **Smallest repair:** give the permutation specification and an exact constructor/table identifier.

Natural-task conclusions remain appropriately narrower than the retrieval findings. Table 2 and Appendix D.7 retain uncertain contrasts; Appendix D.8 retains the Llama high-factor book-QA loss and Qwen PPL reversal.

### 4. Narrative and clarity

The new Figure 1 gives a concrete meaning to allocation before the formal definition. Moving the equal-displacement result immediately after the main length comparison strengthens the causal story. Leading §6.2 with the same-target history experiment also better supports “context use.”

**Minor presentation trade-off introduced by the revision:** the old Figure 1’s crossed-weight heatmap disappears from the main paper. Its exact values remain in Table 6, and §3.2 retains the result, so this is **reduced visual accessibility, not evidence loss**. A compact numerical sentence in §3.2 would restore immediacy if space permits; restoring the entire figure is optional.

## AC synthesis

The manuscript presents a persuasive controlled study with practical frozen-model value and appropriately bounded mathematical analysis. The revised organization strengthens that case. The two concrete repairs above improve deployment interpretation and reproducibility; neither overturns the central contribution.

## Regression audit

- **Preserved:** fixed-support effects, support-policy reversal, equal-displacement controls, crossed weights, coordinate reassignment, theory, all reported numerical tables, uncertain natural-task results and negative high-factor/native-window results.
- **Improved:** allocation/distance intuition, whitening explanation, positioning of the decisive shape control, and NCP context-use narrative.
- **Lost:** no substantive result; only the main-text visualization of the crossed-weight evidence.
- **Rendering:** inspected main-paper figures, tables and relevant appendix formulas; no new clipping, broken formula or material layout defect identified.
- **Conclusion:** **regression pass**, with the pre-existing reporting/reproducibility repairs identified above.
