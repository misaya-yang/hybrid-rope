# Round 4: independent PDF-only review

Input SHA256: `d0c2e485e4963ca4250243e86f1199dce2816affdbc8048b673edd2f8104c006`. Fresh Sol, high reasoning; only the 38-page PDF and derived pages/text. Verdict: accept leaning, medium-high confidence. The reviewer checked all main mathematical families without finding an error overturning them.

## Disposition

1. **Qwen configuration attribution** (p1 intro, p7). Valid inconsistency: the result paragraph already said table plus amplitude, but the introduction shortened it to table alone. Made the introduction consistent. The reviewer's initial abstract locator was corrected by the reviewer; the abstract's OLMo BM comparison has matched amplitude.
2. **151.9M anchors and absolute NLL** (pp2–3,26–27). Recovered the original anchor sampler and evaluator from `main_0726`. Added seed20260723, stratified end-position sampling with2048-token guards, suffix windows `[e−L,e)`, and the common final128 target tokens. The stored primary receipt has per-seed differences and hashes; absolute values from different crossing tables were not substituted. Recovering the original stream is an archive task, not a missing experiment.
3. **Official RULER versus whole-response exact** (pp6,30,33). Added one sentence giving the16K normalized exact0/1.54% and noting official task partial credit. Kept the actual metric names and existing full table.
4. **Tau selection**. The paper already defines a family and lists each setting procedure. No new universal-rule disclaimer or duplicated selection table was added. Rewrote the appendix's lengthy non-claims into its actual reference rule, local calculation and neighboring-strength results.
5. **Repeated evidence tiers in Discussion**. Not adopted. The main table and protocol paragraphs already give seed/pair counts. The author's explicit priority is a paper organized by its scientific question, not an extended review-defense narrative.
6. **External versions**. Added a compact pointer to the bundled recorded model/tokenizer/evaluator identities and the verified RULER commit for the index/placement studies. Existing records are used; unrecorded revisions are not inferred from later runs.
7. **Math and layout**. Explicitly took the slow-frequency limit through positive frequencies, defined sinc, and identified the23/24 slow-pair grids together in the appendix. The alleged Seeds/Train-length “38K” concatenation was not reproduced in the supplied page or layout-text output; no arbitrary table restructuring was performed. Moved the temporal table ahead of the source-intervention paragraph and kept the enlarged figure annotation within its panel.

## Author-directed prose pass

After this review, the author reaffirmed the exponent-distribution research question and prohibited defensive writing. The primary agent read the active manuscript and removed repeated inventories of unclaimed universal optima, missing theory targets, evidence ceilings, and capability caveats. Experiment identities, equations, actual scores, selection procedures and statistical units remain. Range-retarget ordering and weight-table preference are presented as findings. A separate agent is examining the seven supplied reference papers' actual treatment of tradeoffs; it is not part of the ten PDF-only review rounds.

The next immutable manuscript is reviewed in round5.
