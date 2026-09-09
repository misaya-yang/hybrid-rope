# Round 2: independent PDF-only review

Input: `paper.pdf`, SHA256 `0021523e8dad79bfc9273d2da252ee69205e5cf7503243eb467453947ec9a89c`. Reviewer: fresh Sol, high reasoning; only this PDF and derived images/text.

Core contribution read by reviewer: exponent allocation separated from range, with causal fixed-endpoint training and interactions with learned weights. Verdict: borderline, leaning accept; medium-high confidence. No decisive algebra error found in the cross-Gram, rank budget, slow limit, Cosh solution, or BM construction.

## Comments and primary-agent disposition

1. **Cosh specificity** (pp3,26–29). The reviewer asks to distinguish an identified allocation effect from a uniquely superior function. The current abstract explicitly calls Cosh a construction from a specified convex criterion; Section 3.2 already reports the exponential comparison and its interval. Kept these positive, bounded claims. Did not invent a Cosh-uniqueness target or add repeated denials of it.
2. **Comparison types** (pp7,24,26,31,33–35). Complete-table wording already introduces the model cases; main frozen controls match endpoints/gain, while Index/YaRN gains are tabulated. Kept those distinctions and the protocol map. No new public checklist or claim that all model cases independently identify pure shape.
3. **Training replication unit** (pp1,5,31). Added “one matched pair” at the 750M and Llama first mentions and in the Llama abstract headline. Retained training-seed counts in the main table, with clearer headings and spacing.
4. **Geometry-to-construction language** (pp4,14–19). Abstract now states that geometry characterizes how allocation changes the positional basis. The chosen cumulative-tail criterion remains explicit. No inferred LM-optimality statement added.
5. **Gold-block intervention aggregate** (pp5,33–34). Verified `_answer_nll` and `_phase1_aggregate` in the historical evaluator: pooled answer-token NLL, 30 tokens from ten three-token cases. Added the aggregate definition, all-ten positive EVQ directions, and documented case-level hit@16 ranges. The historical report supports these values; no case-level NLL confidence interval was invented from absent rows.
6. **Artifact scope** (p9,p38). Reproducibility statement now names the actual manuscript archive contents and distinguishes paper/numerical reconstruction from checkpoint-level reproduction. The source archive will include numerical summaries and a hash manifest; model weights and training streams are separate.
7. **Small clarity/layout issues** (pp5,20,25). Defined pooled 750M AR as 2K/4K/8K, 40 trials per length and 120 total. Removed a stale unintroduced three-region sentence in the crossing proof. Defined FMR in the crossing caption and the relation between k/K and normalized k/(K−1).

## Additional primary-agent correction

The inherited hyperparameter table said 750M batch 2–4. The Phase15 run report specifies microbatch 7 and accumulation 2, effective batch 14; corrected it. Historical report: `docs/exp/2026-03/2026-03-06_phase15_750m_2k_to_4k_continue_results.md`. No new model run.

Output is the next compiled PDF, independently reviewed in round 3.
