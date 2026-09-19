# Independent PDF-only comparative research review

## Scope

I reviewed both PDFs in full, including all appendices: Paper A, *MrRoPE: Mixed-Radix Rotary Position Embedding* (16 pages), and Paper B, *Beyond the Base: Frequency Allocation in RoPE* (35 pages). I used the supplied text extractions for coverage and rendered the main figures, tables, and formula-heavy pages to check layout and extraction fidelity. This review uses only the two PDFs. It applies the same criteria to each paper and does not require theory to prove task optimality that the paper does not claim.

## Overall comparative judgment

Paper B is the stronger research contribution overall. Its central claim is narrower and better identified: interior frequency allocation matters even when support is fixed, and part of the effect remains when total log-frequency displacement is also fixed. It supports that claim with several distinct interventions, formal analysis with explicit limits, paired uncertainty estimates, and unusually detailed protocols. Its practical method, TailSpline, also beats Paper A's MrRoPE-Pro in strong matched comparisons on Llama and OLMo and transfers across additional models, although results are less decisive on some cross-family and natural-task panels.

Paper A contributes a useful organizing lens and a simple training-free method. MrRoPE-Pro is practically attractive and the PDF reports consistent improvements over YaRN across several long-context endpoints. However, the paper presents the radix analogy as a unifying theory more strongly than its derivation warrants, contains concrete mathematical and reporting inconsistencies, and provides much weaker causal isolation and uncertainty accounting. Its empirical case for MrRoPE-Pro is meaningful; its mechanistic and theoretical conclusions are not established to the same standard.

| Criterion | Paper A | Paper B |
|---|---|---|
| Conceptual contribution | Mixed-radix interpretation and progressive middle-band scaling | Frequency allocation as an independent design variable, plus explicit frozen, native-window, and learned constructions |
| Theory relative to claims | Intuitive but often stated as formal equivalence or proof without sufficient justification | Formal positional geometry and construction results; usually careful to distinguish structural statements from task guarantees |
| Causal isolation | Common-recipe MrRoPE-Pro versus YaRN/NTK, but no equal-displacement, spectrum-reassignment, or crossed weights/tables controls | Fixed-support training, equal-displacement controls, spectrum-preserving reassignment, crossed weights/tables, and matched frozen deployment |
| Evidence quality | Broad tasks and lengths, but sparse protocol detail, no uncertainty estimates, and several table/identity inconsistencies | Large paired panels for key claims, confidence intervals, replication units, negative results, and detailed appendices; some transfer panels remain small |
| Practical value | Simple training-free extension through 128K on three model families | Strong frozen extension on five families plus a 70B transfer, native-window LM, training, and adaptation; mixed results at higher factors are disclosed |
| Narrative and reproducibility | Concise and easy to follow, but imprecise and internally inconsistent | Dense and method-rich, but traceable and substantially more reproducible |

## Paper A: main strengths

1. **A compact and usable intervention.** The cumulative-scaling parameterization makes NTK-aware scaling, YaRN, and the proposed uniform/progressive transitions easy to compare. MrRoPE-Pro has a closed-form middle-band schedule and reuses YaRN's public band and amplitude machinery (PDF pp. 4-6, Sec. 3). Whatever one concludes about the radix interpretation, this is a simple training-free frequency-table construction.

2. **Broad practical evidence for the direct method.** The paper tests three model families, native lengths of 4K/8K/32K, and extension factors up to 16. The reported direct comparisons favor MrRoPE-Pro over YaRN on most perplexity entries (p. 7, Table 1; p. 15, Table 4), RULER through 128K (p. 8, Table 2), NIAH (pp. 7-8, Fig. 4), InfiniteBench (p. 8, Table 3), and LongBench v2 (p. 16, Table 5). The consistency across several endpoint types is a real strength, especially because no weight updates are used.

3. **A clear research question.** The paper asks whether YaRN's linear/regressive treatment of the transition band is the best choice and answers with a concrete progressive alternative. That question is more useful than the broader rhetoric about a complete RoPE theory.

4. **Concise presentation.** The main method and headline empirical comparison can be understood quickly. Figure 3 and Eqs. 13-16 make the three transition profiles visually and algebraically comparable (pp. 5-6).

## Paper A: substantive weaknesses

### 1. The radix framing is an analogy, not the demonstrated equivalence claimed

Section 2.2 begins from a genuine visual resemblance between a RoPE phase and a radix digit, but the operations are materially different. A radix digit requires a floor and a modulo by the radix; a RoPE channel is a continuous phase modulo (2\pi). Equation 7 forms a weighted sum of wrapped phases and calls it a recovered, biased position estimate, but it is neither an inverse of RoPE nor a proof that RoPE performs radix conversion. Figure 2 only shows that this chosen statistic has a locally linear trend for some bases (PDF p. 3, Sec. 2.2).

The later claim that any frequency rescaling expressible through cumulative products of \(\lambda_j\) is a mixed-radix conversion is also largely a reparameterization: any positive frequency table can be written through adjacent ratios. This lens may be useful, but it does not by itself establish representational range, generalization, or why one \(\lambda\) profile should improve a learned model. The paper repeatedly elevates this heuristic into statements such as "prove this hypothesis," "unified theory," and "theoretical basis" (pp. 3-4, 8-10). Those are unsupported at the stated level.

There is also a concrete dimensional error in the definition of incomplete-cycle channels: the condition is written as \(L/\theta_j < 2\pi\) on p. 4, Sec. 3.1, whereas a channel's accumulated phase is \(L\theta_j\). This is not a request for stronger or task-optimal theory; it is a correctness problem in the theory the paper does claim.

### 2. The mechanism analysis is not sufficient for its conclusions

The RoPE-bound plot uses the first root of a sum of cosines as a proxy for an attainable context bound (p. 9, Fig. 5). Even accepting the cited prior framework, the paper does not establish that a larger root is an upper bound on usable model context in the reported tasks, nor that MrRoPE-Pro "pushes the upper bound ... to its maximum" (p. 10). No optimization or maximality result is given.

Figure 6 samples 50 token pairs and plots middle-band partial attention scores, but the layer/head selection, sampling unit, uncertainty, and link from those raw partial scores to final softmax competition are underspecified (p. 9). The claimed mechanism - stabilized attention distributions that better match pretraining - is therefore a suggestive diagnostic, not a demonstrated causal explanation.

### 3. The direct empirical evidence is useful but under-reported

The PDF generally reports point estimates only. RULER has no stated inputs per task or uncertainty intervals (p. 8, Table 2). InfiniteBench randomly selects 100 examples per subset but gives no selection seed, interval, or paired analysis (p. 8, Table 3). NIAH's heatmap gives no aggregate uncertainty. These omissions matter because some reported differences are only a few points.

The paper's comparison to GPT-4, Kimi, and Yi in Table 3 is descriptive rather than controlled: model scale, training, and system differ. It is fair to say MrRoPE-Pro beats YaRN on the same Llama checkpoint and happens to exceed those systems on selected subsets. The conclusion that it "surpasses strong baselines, including closed-source models" (p. 10) is broader than Table 3, where GPT-4's average is 68.3 versus 49.8 for MrRoPE-Pro.

### 4. Several internal inconsistencies reduce confidence and reproducibility

- In the Qwen block of Table 1, MrRoPE-Pro at 32K is printed as 3.579 while YaRN/MrRoPE-Uni are 2.589/2.581 and neighboring MrRoPE-Pro values are 2.881/2.328. The accompanying prose says MrRoPE-Pro is lowest at every length. This is almost certainly a typo, but as printed the table contradicts the claim (p. 7).
- The hyperparameter convention is inconsistent. Appendix A.2 gives YaRN's usual \(\alpha=1,\beta=32\), while Appendix B.1 concludes that \(\alpha=32,\beta=1\) is the strong default (pp. 13-15). The figures label the one-turn/32-turn boundaries in a way that suggests names have been swapped rather than a real reversal. This needs correction because it changes implementation.
- Main text names LLaMA3-8B for LongBench v2, but Table 5 labels the corresponding block "LLaMA2-7B-Instruct" and uses a 4K-to-64K setting (pp. 8, 15-16). The exact evaluated checkpoint is therefore unclear.
- The proof that YaRN is always a regressive conversion uses \(c=\beta-s\alpha\) and concludes an inequality without discussing cases where this term changes sign (p. 13, App. A.2.1). The stated universal conclusion is broader than the shown conditions.

These are real reporting/correctness gaps, not requests for optional future experiments.

## Paper B: main strengths

### 1. The central novelty is well isolated

Paper B cleanly separates support (endpoints and log-frequency span), allocation (interior normalized coordinates), total log-frequency displacement, amplitude, and assignment to learned coordinates (pp. 1-3, Secs. 1-3). This is conceptually important relative to the prior work as the PDF describes it: YaRN and MrRoPE already change frequencies, but they do not establish what interior placement contributes after range and movement are controlled.

The strongest causal evidence is deliberately redundant:

- Fixed-support paired training changes only the 30 interior coordinates across three matched seeds (p. 3, Sec. 3.2; p. 21, Table 5).
- The TailSpline-C control matches endpoints, outer bands, gain, and total log displacement, yet differs by +2.10 Full-13 points at 32K with a paired interval [1.11, 3.08] (p. 7; p. 24, App. D.3).
- A spectrum-preserving intervention permutes only the assignment of the same frequencies to learned coordinates and sharply changes frozen-model performance (p. 23, App. C.6).
- Crossed weights and runtime tables show that each trained weight set prefers a table derived from its own training allocation (pp. 21-22, Table 6).

Together these controls justify the paper's main scientific claim substantially better than a single new-method-versus-baseline comparison would.

### 2. The theory is technically adequate and appropriately bounded

The complete sine-cosine pair analysis is a meaningful correction to cosine-only collision heuristics. The paper defines a phase-invariant pair overlap through canonical correlations, derives the exact effective-rank identity, proves the slow-frequency shared-subspace limit, and gives explicit counterexamples where cosine-only or single-length orderings mislead (pp. 3-5; pp. 14-16, Apps. A.1-A.3). The integer-position kernel criterion and slow-block construction carefully distinguish positional dependence from content-coordinate capacity (pp. 5, 16-17).

Crucially, the paper does not claim that effective rank determines task quality. It explicitly shows TailSpline can improve RULER while lowering effective rank and calls the softmax relation an explanatory fixed-state identity rather than a task-optimality criterion (p. 5; p. 17, App. A.6). Likewise, its phase intervals are explicitly said not to be effective model context lengths (p. 20, App. B.5). This calibration makes the theory adequate for the claims actually made.

TailSpline's derivation is also internally complete: the transition objective, unique closed-form minimizer, endpoint behavior, finite phase ratio, and exact equal-displacement control are derived in pp. 5-6 and Apps. B.1-B.2 (pp. 17-18). The objective is a declared smooth tail-connection preference, not presented as a proof of task optimality.

### 3. Evidence quality and disclosure are unusually strong

The main Llama and OLMo RULER comparisons use 650-2600 paired prompts and report paired intervals (p. 7, Table 1; p. 24, Table 8). Direct three-arm comparisons add YaRN, while PPL and NIAH use matched documents or prompts (pp. 26-28, Tables 12-15). The paper reports sample sizes, cluster/bootstrap units, generation caps, empty/capped outputs, input ranges, model-specific tokenization, and exact evaluation revisions. It also separates large confirmation panels from small transfer panels.

The paper reports unfavorable and mixed results rather than hiding them. TailSpline is 4.87 Full-13 points below original RoPE at native 8K, with the loss concentrated in one task that produces many empty responses (p. 24). At higher factors it improves Llama Full-13 and PPL but hurts Llama book QA; on Qwen 256K it improves single-needle retrieval while worsening five-book PPL (p. 30, Table 18). NCP improves native-window same-target NLL and RULER but not Native-QA (pp. 31-32). These counterexamples materially increase confidence in the paper's evidence discipline.

### 4. Practical value extends beyond one checkpoint or regime

For frozen extension, TailSpline uses only public RoPE parameters and no checkpoint calibration. It improves strong matched panels on Llama and OLMo, leads the point estimates across Qwen, GLM, and Kanana Full-13 panels, and transfers the same 8B table to Llama-70B NF4 (pp. 6-8, Tables 1-2; pp. 27-30). The largest, most decisive gains are on Llama and OLMo; Qwen/Kanana intervals versus MrRoPE-Pro include zero, and most natural-QA intervals are wide.

The paper also establishes that allocation matters during learning and adaptation, not only frozen extension. Three-seed 432M MLA training, a matched 750M continuation, a 1.485B OLMo pair, matched Llama LoRA, and explicit long-gap OLMo adaptation all show extended-length gains, while native-length costs are reported (pp. 8-9; pp. 33-35, Apps. F.1-F.2). These protocols are heterogeneous, but each supports a different scoped claim rather than being pooled into one effect.

### 5. Reproducibility is much stronger than Paper A

The reproducibility statement specifies what is in the source archive and enumerates the major token budgets (p. 10). The appendices give frequency formulas, exact tables or constructors, seeds, datasets and revisions, optimization settings, sampling rules, scoring units, and uncertainty procedures. A reader could reconstruct most comparisons from the PDF plus archive. Paper A says code and configurations are supplied, but its PDF leaves several core identities unresolved.

## Paper B: substantive weaknesses and residual concerns

1. **The paper is too broad for its narrative bandwidth.** TailSpline, Cosh, and NCP are three distinct constructions serving frozen extension, training, and native-window use. They are unified by allocation, but the 35-page paper still reads partly like several projects joined under one thesis. The central identification result is strong enough that the presentation could more sharply distinguish the primary claim from supporting regimes.

2. **TailSpline's design objective is principled but not uniquely motivated.** Penalizing extra-gap curvature and the low-frequency junction yields a clean closed form, but it remains one declared boundary prior among many. The equal-displacement control demonstrates that its residual shape matters at Llama 32K, not that the minimum-bending objective is the causal reason or generally best allocation. The paper mostly avoids claiming this, but phrases such as "these properties explain the distance preference" should be read as design interpretation, not model mechanism (pp. 5-6).

3. **Cross-family and natural-task conclusions need qualification.** The decisive large panels are Llama and OLMo. Qwen and Kanana T-P intervals include zero; GLM is positive on Full-13 but based on 10 inputs per task (p. 7, Table 1). In natural QA, only OLMo's marginal T-P interval excludes zero; Qwen's marginal interval does, but its simultaneous interval includes zero, and Llama/GLM/70B intervals are inconclusive (p. 8, Table 2; p. 30, Table 17). The abstract's phrase "gains on natural question answering" is defensible as an existence statement but should not be read as a uniform cross-model result.

4. **Several training studies are strong demonstrations but not interchangeable replications.** The 151.9M fixed-support experiment is the cleanest training identification. The 50.9M factorial is very short (8.39M tokens per arm), and the larger MLA, OLMo, continuation, and LoRA studies use different architectures, data, grids, objectives, and strengths (pp. 21-22, 33-35). This breadth supports robustness of the broader design variable, but it does not estimate one common effect size.

5. **External validity remains checkpoint- and task-dependent.** The paper explicitly shows reversals when support policy changes (p. 3; p. 21, Table 5), native-length degradation for the frozen extension (p. 24), and task-specific reversals at high extension factors (p. 30). The correct conclusion is that allocation is consequential and can be designed profitably under specified regimes, not that one allocation dominates across all lengths and tasks. The main text is mostly consistent with this, though its concluding language about quality "throughout the working context range" is more optimistic than the full appendix record.

## Paired interpretation of the papers

Paper A's clearest durable contribution is the practical progressive transition that Paper B uses as a matched baseline. Paper B shows why that comparison should be framed as allocation rather than radix conversion, and it adds the controls Paper A lacks. Under Paper A's stated common YaRN recipe, MrRoPE-Pro-versus-YaRN is evidence that the complete transition profile matters, but it does not separate total movement from higher-order shape or test spectrum assignment to learned coordinates. Paper B's TailSpline-versus-MrPro comparison holds support, outer bands, amplitude, inputs, weights, and decoder fixed; TailSpline-versus-C also holds total displacement fixed. This makes Paper B's causal claim qualitatively stronger.

The papers also differ in how they use theory. Paper A uses a heuristic positional statistic and partial-attention plots to argue that its method raises a theoretical context bound and stabilizes attention. Paper B proves narrower structural facts, then demonstrates empirically that those facts are not themselves task-selection rules. That separation between mathematical structure, fixed-content response, and model/task evidence is more convincing.

Paper A remains more compact and easier to deploy from the main text. Its 128K evidence on Llama and Qwen is practically relevant, and Paper B itself confirms MrRoPE-Pro as a strong baseline rather than a straw man. The appropriate comparative conclusion is therefore not that Paper A lacks value; it is that Paper A's method is more convincing than its theory, while Paper B's scientific identification and reporting are stronger than either component of Paper A.

## Residual uncertainty from this PDF-only review

- I did not inspect code, raw outputs, supplementary archives beyond the PDFs, or external literature. Numerical integrity and novelty relative to work not described in the papers therefore remain unverified.
- Paper A's Qwen PPL and model-name inconsistencies may be typesetting errors rather than run-level errors, but the PDF alone cannot resolve them.
- Paper B's detailed protocols substantially reduce ambiguity, yet the large number of studies and endpoints makes multiplicity and historical selection difficult to assess from the PDF alone. Its strongest conclusions are the pre-specified matched controls and large paired panels, not every nominal improvement.
- Neither paper establishes a universally optimal frequency allocation, and neither needs to. Paper B establishes that allocation is an independent, consequential design variable and supplies useful constructions; Paper A establishes that a simple progressive rule is a competitive training-free extension on the reported checkpoints.
