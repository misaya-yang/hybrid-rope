# Round 1 integration

Review: [independent PDF-only report](review.md), internal 6/10, weak accept.
The review score describes the frozen input, not the optimized output.

| Finding | Decision and implemented change | Verification / remaining boundary |
|---|---|---|
| W1 native wording | Accepted. Introduction/conclusion now name an **observed** 2.33% relative trade-off. Main results put the paired −6.14/+1.92pp interval beside the point estimate and identify the classic-native versus clean-long panel contracts. | Values read from the completed Native report and portable input. Positive gain/trade-off framing retained; no retention bound asserted. |
| W2 T/C result hidden | Accepted. The construction paragraph now reports −0.41pp [−2.63,+1.82] and batch1/batch2; Table 1 labels E1 a diagnostic. Appendix H.2 points to the completed H.8 result instead of claiming none exists. | Primary Full-13 endpoint preserved; no secondary-endpoint substitution. No new run required for this correction. |
| W3 deployment scope / baseline | Accepted scope correction: abstract now names Llama and MrRoPE-Pro; separate clean and classic panels remain. New YaRN GPU suggestion remains a bounded optional follow-up, already prepared outside the running queues. | Author's no-numbers-in-abstract rule retained. Pending Natural-QA/YaRN outcomes are not included. |
| W4 Cosh family vs exact objective | Accepted. Main text describes Cosh as an effective explicit family and states the strength convention. It now includes the existing M4 outcomes: 7/12 reference, 10/12 preassigned stronger Cosh, 9/12 matched exponential, and +0.00074 [−0.006,+0.008] Cosh–exponential NLL. | Independently recomputed from the 12 configuration rows. Short-training budget and three paired seeds named; no universal optimizer claimed. |
| W5 slot protocol | Partially repaired. Appendix identifies the historical reference grids, scale, gain, counts and table hashes; distinguishes the available permutation-generator default from the unretained realized vector/seed receipt. | The exact old vector and OLMo tail-window receipt were not recovered; they are not invented. Figure 1 now uses the fully specified two-seed 151.9M weights-by-derived-table crossing. The historical slot result remains clearly identified in text/appendix. |
| W5 MLA cache | Repaired the provenance description with inspectable loader behavior and τ=1.414 in main. Appendix states the cache-first path, shuffle rule, public source candidate and absent original cache-building/revision/exclusion receipts. | Code behavior does not prove the provenance of a cache hit. Reported paired-cache result preserved; disjoint-shard evidence comes from the separate fixed-support experiment. |
| Optional geometry bridge | Reused existing parity-lattice theorem to explain the resolution/coverage trade-off, rather than adding another proxy or curve. | Existing proof retained; no task-ranking claim added. |

## High-value repository reuse

- **Promoted:** the existing 151.9M controlled crossing, with matrix
  3.426432/5.775900 and 4.454919/3.479045 NLL, two training seeds, 32 paired
  anchors each, final 128 targets at 1K. Source:
  `paper-2027/research/attention-aware-retrofit/evidence/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json`.
  Its exact runtime-table construction is already in the PDF appendix.
- **Promoted:** complete fixed-support two-family M4 summary. All selected counts
  were recalculated, not taken from the old malformed `best_cosh_multiplier_counts`
  summary field. No historical raw artifact was rewritten.
- **Reused conceptually:** the parity-lattice recurrence counterexample clarifies
  why finite-window effective rank and out-of-window coverage are different design
  concerns.
- **Retained in appendix:** 15M-token learned-frequency comparison and historical
  native-grid study; they do not replace the longer strict primary experiment.
- **Pending:** Natural-QA T is complete but P was partial when checked; no partial
  scores were inspected or inserted. No new model/GPU work launched.

## Sources inspected by integrator, not reviewer

The current claim map and asset registry; M4 portable configuration rows and
original complete report; the scale-consistent log-profile owner; the
`dilation_allocation_fork.py` constructor; `run_gqa_evq_experiment.py` and its
`run_evq_sweep.py` data loader; current Natural-QA completion counts. The reviewer
was not given any of these sources or this disposition.
