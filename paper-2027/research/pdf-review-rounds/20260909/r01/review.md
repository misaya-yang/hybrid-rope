# Round 1: independent PDF-only review

Input: `paper.pdf`, SHA256 `e9944c699252dd81d574f4e1d36ba6f2c3821e5b0c617c3b77f2d6c815f2c92a`. Reviewer: fresh Sol, high reasoning; no source or prior-review context.

Core contribution read by reviewer: finite exponent allocation at fixed log-frequency range, full sin/cos geometry and a Cosh family, with empirical compatibility effects involving range and learned weights.

Verdict: borderline, slightly favorable; medium-high confidence. The reviewer checked the main propositions/theorems and did not find a formula error overturning the argument.

## Comments and primary-agent disposition

1. **Cosh family versus a unique deployment recipe** (pp4,20–21,24,28,31,35). Accepted the reproducibility issue: added a protocol-level tau/grid/selection table and defined EVQ-Cosh as a one-parameter family. Did not adopt a new requirement to prove a universal zero-search rule, which the paper does not claim. Video's sweep-selected setting is identified.
2. **8B headline trade-off and attribution** (pp1,5,31). Added 8K6.82→10.07 alongside 32K991.5→127.9 in the abstract; model results are described as complete-table interventions. Kept the separate 516-step RULER continuation distinct from the 300-step PPL pair.
3. **M4 cross-shape inference too strong** (pp3,27). Replaced the categorical useful-intervention conclusion with observed mean directions, configuration dependence, and uncertainty; kept unadjusted tests in the table. No equivalence inference from p=.836.
4. **Geometry-to-surrogate bridge** (pp4,15–16,19). Kept the explicitly chosen cumulative-tail criterion and replaced functional-validation language in the cosine-slice appendix with descriptive diagnostic language. Behavioral effects remain separate measured findings.
5. **Training replications versus evaluation rows** (pp5,23–35). Added replication counts directly to the main training table and expanded the appendix protocol map for the new frozen-model studies. Absolute NLL for the 151M three-seed primary comparison was requested; the recovered primary receipt contains contrasts and source hashes, so no absolute values were reconstructed from unrelated evaluations.
6. **Exact runtime formulas / opaque numerical examples** (pp25–26,16). Added R(L)=(K−1)/K logL, static factor-four crossing formulas, discrete causal measure, residual projection, SVD cutoff, FP32 endpoint installation and gain. Replaced opaque search-derived numerical examples with fully specified small frequency vectors. Closed-form Gram calculations agree with independent Gauss–Legendre quadrature to <4e−14; this is mathematical verification, not a model experiment.
7. **Labels/figures** (pp1–2,7,15). Named MrRoPE-Pro in the natural-QA abstract comparison; made CI confidence explicit; labeled all9 OOD comparisons; enlarged appendix figures and separated the 48 label from the legend. Kept EVQ-Cosh as the established method name and defined its mathematical family, rather than inventing an unsupported acronym expansion.

## Additional issues found by the primary agent

- The 750M grid was incorrectly described as midpoint in the inherited appendix. The historical script and run report use r=0, inclusive u=k/(K−1), then map the Cosh quantiles to the original Geo endpoints. Corrected the formula and protocol table; numerical results retained.
- Future result tables had floated into the Cosh derivation, and appendix floats interrupted later protocols. Added proportionate float barriers and preferred local placement.

Output is the next compiled PDF, reviewed independently in round 2.
