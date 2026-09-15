# Independent manuscript review

**Manuscript:** *Beyond the Base: Frequency Allocation in RoPE*  
**Review type:** Author-requested internal simulation using the supplied ICLR 2027-style rubric; not an official conference review.  
**Material consulted:** Only the supplied 60-page PDF. I read the nine main pages and the supplementary material, including the relevant proofs and experimental protocols, and visually inspected all nine main pages and their figures. I did not consult repository files, code, raw experimental artifacts, prior reviews, memory, or external literature. Page references below are PDF page numbers, which agree with the printed page numbers.

## 中文执行摘要

**倾向弱接收，内部评分 6/10。** 论文最有说服力的贡献是把频率端点、内部配置、总位移和已学习的坐标对应关系区分开，并用配对训练和冻结模型实验验证内部配置确实有用。Llama 的 clean RULER 结果支持 TailSpline 相对 MrRoPE-Pro 的明确收益；主要数学结论在各自假设下成立。

目前不足主要是证据覆盖范围，而不是已发现的致命数学错误：TailSpline 的具体边界设计尚未优于同位移控制得到可靠验证；原生窗口代价的区间较宽；432M 实验的评估缓存缺少可核实的训练排除记录。优先补齐同运行条件的 C 对照、用明确独立语料复评已有 432M 检查点，并在摘要/结论中更明确地限定原生窗口及部署收益的适用范围。无需以全面刷新 SOTA 或重做所有训练作为修复前提。

## Summary and assessment

The paper asks whether interior RoPE frequency placement matters after fixing the sampled frequency endpoints, and how that freedom can support both learning and frozen context extension. It provides a useful decomposition into support, allocation, displacement, assignment, and amplitude; controlled experiments; full sine–cosine subspace analysis; and two explicit allocation constructions, Cosh and TailSpline.

The clearest scientific result is an existence result with meaningful controls: changing interior allocation can change and sometimes improve model behavior at fixed support. The 151.9M paired experiment isolates that intervention well. The stronger practical result is TailSpline's task-equal gain of 11.72 percentage points over MrRoPE-Pro on 2,600 paired clean 32K Llama prompts. Neither the theory nor the experiments establish a universally preferred allocation or a task-optimal smoothing objective, and the manuscript usually acknowledges this distinction.

**Recommendation: weak accept. Internal review score: 6/10** on an explicitly internal scale, where 5 is borderline and 6 is weak accept; this is not asserted to be the official ICLR scoring form. **Confidence: 4/5** in the within-manuscript assessment. Confidence in exhaustive novelty is lower because this review deliberately does not inspect the cited literature; runtime and artifact claims also remain unverified beyond their PDF descriptions.

The main results provide useful new knowledge without needing a SOTA claim. The evidence and explicit limitations merit acceptance narrowly above the borderline. Stronger claims about the best frozen rule, broadly preserved native performance, or predictive mechanism would require additional evidence.

## Strengths

1. **The experimental question is specific and genuinely controlled.** Section 2, p. 2, Eq. (1) and the control table distinguish quantities often conflated in frequency scaling. Section 3.1 and Appendix B.1, pp. 25–26, match actual endpoints, initialization, token order, optimizer, budget, and evaluation targets within three training-seed pairs. The disjoint train/validation shards and common final-128-token targets are important strengths. Improvements at retained support and the opposite ordering after support retargeting are both reported in Table 6, rather than presenting the favorable policy alone.

2. **The clean frozen result is substantial and reasonably well documented.** Section 6.2, pp. 7–8, and Appendix H.7, pp. 53–54/Table 39, use 200 source-order inputs per task, unpadded batch 1, matching gain and decoder, all 13 tasks, and paired within-task intervals. The gain is not caused by omitting the negative multivalue task or empty/capped outputs. Leave-one-task-out gains and task-family decompositions support breadth within this benchmark. The clean experiment directly addresses the unusual padding and selection in the classic curves.

3. **The mathematical objects and their limits are largely stated correctly.** The pairwise canonical-correlation measure handles both sine and cosine and is invariant to within-pair basis changes. The effective-rank identity follows directly from the block Gram expansion (pp. 4, 15–16, Eqs. (4)–(5), Theorem 4). The slow-subspace argument handles the near-singular representation by rescaling the sine direction; Appendix A.1 separately discusses raw feature scale and the centered softmax metric. The integer-position result has appropriate frequency, invertibility, and all-content assumptions (p. 4; pp. 20–21). The proofs of the Cosh density optimum and finite-grid TailSpline solution are coherent for their declared objectives (pp. 21–23 and 50).

4. **The paper preserves informative negative evidence.** Examples include the support-policy reversal, the lack of Cosh superiority over a displacement-matched exponential (p. 3; Table 8, p. 28), BM's small Qwen losses (pp. 36–37), unsuccessful natural-output conversion for an adapted Llama model (p. 43), and the inconclusive T/C comparison (pp. 6, 54). These delimit the contribution more credibly than a collection of wins would.

5. **The constructions are simple enough to be useful.** Eqs. (7), (8), and (11), plus the installation recipe on p. 6, define finite tables without a new rotary operator. TailSpline's low implementation cost and lack of weight updates or per-model calibration are practically attractive. The title is accurate and appropriately broader than either named construction.

## Prioritized decision-relevant weaknesses and repairs

### W1. The evidence identifies useful allocation more strongly than it identifies TailSpline's specific boundary choice

**Type:** Evidence gap; medium priority for the current bounded claim, high priority for a stronger method-selection claim.  
**Locators:** Section 5.2, p. 6, Eqs. (10)–(13); Section 6.2, pp. 7–8; Appendix G.8, p. 49/Table 35; Appendix H.2, p. 50; H.7–H.8, p. 54.

TailSpline versus MrPro changes both shape and total log displacement. This is a valid allocation comparison at fixed support, but it does not show that the one-sided tail-junction objective is the reason to select TailSpline over a simpler profile with its displacement. The explicit equal-dose C control is therefore highly relevant. Its recorded Full-13 AUC is slightly higher than TailSpline's, with T–C −0.41 points and an interval spanning both signs. More importantly, T and C use different batch sizes/orderings and lack complete runtime identity. The 39-row replay checks some sensitivity but does not establish comparability of the complete arms. The manuscript accurately calls this unresolved; I do not interpret the result as proving either a T loss or equivalence.

BM–Uni supplies some separate evidence for shape at equal dose, but its six-task 48-input follow-up is much smaller and uses another construction/model. The larger clean TailSpline panel compares only MrPro. Consequently, the paper offers a successful simple rule but limited guidance on why a practitioner should select that particular rule.

**Bounded repair:** Complete one runtime-matched T/C comparison using a declared primary endpoint and identical input/order/backend settings. Reuse valid existing T outputs if the runtime can be matched exactly. If a narrower predeclared subset is used for cost reasons, report its precision honestly. A same-contract YaRN comparison would separately establish whether the clean gain extends beyond the nearest MrPro comparator; it is valuable, but an exhaustive baseline sweep is unnecessary. If these additions are unavailable, retain the present unresolved statement and explicitly describe TailSpline as one empirically effective boundary prior, not a validated optimal choice.

### W2. The larger Cosh learning result has an unresolved evaluation-data provenance gap

**Type:** Evidence/reproducibility gap; medium-to-high priority.  
**Locators:** Section 6.1, p. 7/Fig. 3; Appendix E.1, pp. 35–36/Table 17, especially p. 35, lines 1880–1886.

The 432M comparison is the strongest multi-seed learning utility result at a realistic training length, but the archived evaluation cache lacks its original build log, pinned revision, and document-exclusion manifest. The described cache-miss loader samples the public training split and has dataset fallbacks. The PDF therefore establishes a paired-cache difference, but does not establish that this prominent evaluation is disjoint from model training or reconstructible as the specified FineWeb-Edu evaluation distribution.

This is not evidence that contamination actually occurred, nor does it invalidate the three-seed difference on the common cache. It does limit the interpretation as clean held-out extrapolation. The separately documented disjoint 151.9M experiment supports the core allocation claim, but cannot resolve the provenance of the 432M result.

**Bounded repair:** Evaluate the existing six checkpoints on one pinned, explicitly disjoint text collection with common offsets. No additional training is needed. Report paired per-seed NLL/PPL, the data-exclusion rule, and native as well as extended lengths. Until then, surface the paired-cache qualification next to Fig. 3 or its paragraph, rather than leaving the limitation only in Appendix E.1.

### W3. The native-window deployment claim has much less precision and a different input contract than the clean long-context result

**Type:** Evidence scope and emphasis; medium priority.  
**Locators:** Abstract and introduction, p. 1; Section 6.2, p. 8; conclusion, p. 9; Appendix H.8, p. 55, lines 2916–2928.

The observed 2.33% relative native task reduction is correctly computed from a 2.14-point difference, and the text says “observed.” However, it is repeatedly used to characterize a small native trade-off, while its paired interval permits a loss of 6.14 percentage points. It uses only 130 classic padded 8K prompts, whereas the clean 32K benefit uses 2,600 source-order prompts. There is no native noninferiority margin, and shorter ordinary prompts are not covered by this particular static-table comparison. The small whole-prefix PPL increase measures another endpoint and does not bound generated-task loss.

The appendix adequately discloses these facts, so this is not a hidden comparison error. Nevertheless, the prominence of the exact relative percentage can communicate more certainty and deployment coverage than the result has.

**Bounded repair:** In the abstract/conclusion, describe a small point estimate with uncertain native task cost, or include the absolute effect and interval at the principal summary location. A clean source-order native panel under the same static table would materially strengthen the use case. Choose a practical tolerance before making a noninferiority claim; do not infer one from the current nonsignificant interval. A small representative shorter-input stratum is useful only if the intended claim includes those inputs.

### W4. The theory explains design degrees of freedom but only weakly guides the successful operating choices

**Type:** Contribution limitation, not an identified theorem error.  
**Locators:** Sections 4–5, pp. 4–6; Appendix A.3–A.4, pp. 18–19; A.13, p. 24; Table 8, p. 28; H.6, p. 53.

The subspace result is a useful structural account. It does not predict model loss: whitening removes scale, content coefficients matter, separation measures change rankings, and full rank can coexist with recurrence. The paper itself supplies these counterexamples. Cosh's tail-energy objective and TailSpline's boundary objective are consequently chosen priors, rather than consequences that optimize an established task-relevant geometry. The Cosh strength rule is based on an explicitly local small-τ argument, with substantial finite-τ approximation error; the matched exponential is similarly effective in the short factorial.

This reduces the novelty of the theoretical contribution to a careful synthesis and explicit construction, rather than a generally predictive design theory. That contribution is still useful, particularly with the controlled evidence. The manuscript should preserve this calibration throughout.

**Bounded repair:** Add a compact statement of what each objective predicts and what it does not, and explain the intended practical choice of Cosh strength and support policy. Use existing results to illustrate one favorable and one unfavorable regime. A new mechanism study is optional; a demonstrated causal mediation theory is not required for the present paper.

### W5. The clearest same-spectrum empirical demonstration cannot be exactly reconstructed from the described record

**Type:** Reproducibility gap; lower priority than W1–W3.  
**Locators:** Section 3.2, p. 3, lines 155–161; Appendix I.3–I.4, pp. 55–56/Fig. 15.

The archived slot experiment lacks the realized permutation, seed override/NumPy version, and exact OLMo tail-scoring window. This is explicitly acknowledged. The algebraic compensation identity is exact, but it is not a measured compensated recovery. The well-specified 151.9M table crossing is an important complementary experiment, yet changes the spectrum and does not directly replace a reproducible same-multiset intervention.

**Bounded repair:** For one already studied checkpoint, record one fixed permutation vector, its frequency table, input IDs, scoring window, runtime, and reference/permuted scores. Add a compensated numerical check if convenient, while distinguishing it from full-model task evidence. Alternatively, keep the historical values clearly secondary and avoid making exact replay of that experiment part of the reproducibility promise. There is no need for a permutation sweep.

## Errors, statistical interpretation, and presentation

### Genuine errors

I did **not identify a decision-changing algebraic or numerical contradiction** in the central formulas or reported comparisons. The construction proofs solve their stated objectives; they do not prove task optimality, which the text generally recognizes. The separate AI-use statement on p. 10 does not make the main text exceed the supplied nine-page limit: substantive main text ends on p. 9, followed by the exempt statement, references, and appendices.

One minor narrative issue is the heading **“The same spectrum can have different learned uses”** on p. 3: its first supporting experiment crosses two different derived spectra. The actual fixed-multiset experiment appears later in that paragraph. Renaming the heading to “Learned compatibility and slot assignment,” or separating those controls, would align the heading with the evidence.

### Statistics and fairness

The paired-seed design and disclosure of seed-specific results are appropriate. The three training seeds are limited but real replications; the 32 anchors should not be treated as 96 independent training runs, and the paper does not do so. The clean frozen bootstrap is explicitly conditional on fixed tasks and selected source records. Its narrow interval should not be read as uncertainty across models or new QA corpora. The factorial's configuration bootstrap and unadjusted sign-flip tests are disclosed, and its roughly 0.01 NLL improvements on very short repeated-corpus training should remain supporting evidence, not broad model-scale validation.

Matching endpoints, gain, prompt rows, and decoder supports fairness of the principal allocation contrasts. The MLA endpoint change, classic padding, adaptation supervision, differing historical trainers, and T/C batch mismatch are all disclosed. None should be silently pooled into one effect estimate. BM's natural-QA result supports allocation design but is not evidence of TailSpline natural-QA transfer; Section 6.2 and H.7 correctly distinguish this. Testing TailSpline on the existing natural-QA protocol would strengthen practical significance, but is an optional extension under the present benchmark-specific claims.

### Clarity and visual presentation

The main pages are readable, the formulas fit, and the four main figures convey the intended comparisons without obvious clipping. Fig. 4 appropriately separates clean and classic contracts. The title is informative, and the abstract largely matches the demonstrated scope. The main weakness is the volume of heterogeneous supporting studies: 60 PDF pages, changing grid conventions, several historical operators, and multiple adaptation protocols place substantial bookkeeping demands on the reader. Tables 1–3 help. A short results hierarchy distinguishing the decisive experiments from exploratory/historical support would help more than additional appendix length. Main-text references to Fig. 10 where Fig. 1 already carries the central retained-support result could also be simplified.

The PDF gives enough detail to understand and independently implement the main constructions and fixed-support training protocol. It points to packaged commands and artifacts, but I did not inspect those resources and cannot certify that the package executes or reproduces the numbers.

## Questions for the authors

1. Can the C arm be evaluated under the exact T runtime, and does the primary paired T/C contrast change? What practical advantage justifies choosing TailSpline if the matched result remains indistinguishable?
2. Can the existing 432M checkpoints be evaluated on a pinned, explicitly disjoint corpus, or can the original cache's source/exclusion identity be recovered?
3. What native task degradation is acceptable for the intended static deployment? Does a clean native panel support that tolerance, rather than merely a small point estimate?
4. Was the clean 32K panel excluded from all choices of method, operating settings, and primary endpoint before any outputs were inspected? Its source-order construction is clear, but a concise selection timeline would sharpen the meaning of “confirmation.”
5. Can one exact same-spectrum permutation intervention be made reproducible with its realized vector and scoring specification?

## Most valuable revision sequence

1. Resolve the matched-runtime T/C comparison, keeping the primary endpoint fixed.
2. Re-evaluate the existing 432M checkpoints on a documented disjoint corpus.
3. Calibrate the native-cost summary and, if feasible, add a clean native task comparison.
4. Repair the small reproducibility and narrative gaps without expanding the paper into a general SOTA or mechanism study.

These changes would strengthen the specific submitted contribution. They are not a request to retrain every model, prove a universal allocation optimum, or evaluate unrelated tasks.
