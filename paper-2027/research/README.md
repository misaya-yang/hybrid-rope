# ICLR 2027 claim and evidence router

This is the claim-level routing layer for the active `paper-2027/` manuscript.
It tells an agent which file owns each claim and which material is only a plan,
audit, or external review.

## Scope

This file is the **claim-level** router for `paper-2027/`: which owner holds
each number and what its maximum role is. Repository-level navigation, the
theory index, the closed-route ledger, and the research agenda live one level
up in [`../../INDEX.md`](../../INDEX.md). Do not duplicate them here.

## Start here

1. [`../../AGENTS.md`](../../AGENTS.md) — stable scientific, submission, safety,
   and workspace rules.
2. [`../../INDEX.md`](../../INDEX.md) — theory/evidence/code index and agenda.
3. [`../HANDOFF.md`](../HANDOFF.md) — live manuscript/worktree state and the
   only current action queue.
4. [`ICLR2027_NARRATIVE_OPTIMIZATION_PLAN_20260826.md`](ICLR2027_NARRATIVE_OPTIMIZATION_PLAN_20260826.md)
   — current Codex-facing 9-page narrative edit order; no numerical ownership.
   The prior simulated-review memo is
   [`ICLR2027_MANUSCRIPT_OPTIMIZATION_AND_SIMULATED_REVIEW_20260826.md`](ICLR2027_MANUSCRIPT_OPTIMIZATION_AND_SIMULATED_REVIEW_20260826.md).
   The related-work / novelty Codex patch is
   [`ICLR2027_CITATION_NOVELTY_AUDIT_20260826.md`](ICLR2027_CITATION_NOVELTY_AUDIT_20260826.md).
5. [`ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md`](ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md)
   — current conceptual grammar: physical table, causal variables, method
   stages, and the role of the zero-training replacement.
6. [`ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md`](ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md)
   — current post-submission theory state, empirical constraints, missing
   identification bridge, and method-entry conditions. The executable protocol
   is [`attention-aware-retrofit/preflights/MATCHED_CONTENT_PHASE_2X2_BRIDGE_PREFLIGHT_20260827.md`](attention-aware-retrofit/preflights/MATCHED_CONTENT_PHASE_2X2_BRIDGE_PREFLIGHT_20260827.md).
   Its separate protected-progressive companion is
   [`attention-aware-retrofit/preflights/PROTECTED_PROGRESSIVE_SHIFT_PREFLIGHT_20260827.md`](attention-aware-retrofit/preflights/PROTECTED_PROGRESSIVE_SHIFT_PREFLIGHT_20260827.md);
   both are design-only until separately authorized.
7. [`ICLR2027_RESEARCH_SYNTHESIS_20260819.md`](ICLR2027_RESEARCH_SYNTHESIS_20260819.md)
   — implemented claim architecture.
8. [`EXACT_RANGE_151M_3SEED_RESULT_20260820.md`](EXACT_RANGE_151M_3SEED_RESULT_20260820.md)
   — raw-hash-receipted three-training-seed fixed-support result.
9. [`FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`](FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md)
   — canonical theory, finite-K counterexamples, and 50M crossing.
10. [`attention-aware-retrofit/README.md`](attention-aware-retrofit/README.md)
   — current mature-checkpoint retrofit results, negative routes, and receipts.

Do not start from the newest date, an external review, or a preflight.

## Current scientific architecture

The realised RoPE object is the ordered frequency tensor `Omega`. For a mature
checkpoint, fixing its Native base as a coordinate convention gives an
equivalent exponent curve `e`; this is the honest implementation-level
description of scalar-base, learned-table, and direct-tensor methods. It does
not collapse their causal identities: what is permitted to change, what is
held fixed, and whether the intervention occurs before training, after
training, or at serving time remain decisive.

For causal analysis, the manuscript separates sampled support from normalized
interior allocation:

\[
x_k=-\log\omega_k=a+Rz_k.
\]

Its central claim is that `z` is a separately identifiable training-time
variable even when `(a,R)` is fixed; it changes full sin/cos subspace geometry
and trained behaviour, while model weights co-adapt to the table used during
training. This decomposition is an experimental accounting system, not a
two-module implementation. EVQ-Cosh is one closed-form intervention on this
axis, not a universal optimum.

The new frozen-checkpoint case study remains a different estimand. Its complete
zero-training intervention contains a deterministic long table, a fixed
attention amplitude, and a session-static Native/long route. Same-support and
frequency-by-gain owners probe those components under their own matched
protocols rather than forming one pooled factorial: fixed-support `z` remains
consequential on OLMo and Qwen, while a coarse fixed-index ramp control matches
the detailed derived profile at the tested points. The defensible novelty is fixed-support
identification and its mature-checkpoint corollary, not merely producing a
non-geometric table or a new YaRN family.

Frozen-table in-window loss must not be promoted into an allocation
impossibility. It contains both the intrinsic effect of the chosen allocation
and table/weight co-adaptation mismatch. Co-adapted training owners, including
the internal phase-chord Pareto result, show that one specified table can be
near-parity in-window while improving all tested extrapolation lengths. The
session route is a verified frozen-checkpoint deployment fallback, not the
theoretical solution to the joint objective.

## Directory map

| Path | Role |
| --- | --- |
| research-root dated owners | conceptual foundation, central synthesis, exact-range owner, canonical theory, and early theory architecture |
| [`attention-aware-retrofit/`](attention-aware-retrofit/) | mature retrofit results, receipts, analyses, theory, and preflights, each in a separate subdirectory |
| [`audits/`](audits/) | internal manuscript/theory/evidence audits and falsified internal measures |
| [`external-reviews/`](external-reviews/) | untrusted independent-model recomputations and proposals |
| [`three_completions/`](three_completions/) | supplementary derivations, verification scripts, and rendered internal note |

Additional durable theory/supporting files at research root:

- [`ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md`](ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md)
  — current continuation owner; it supersedes dated next-step reasoning without
  superseding numerical result owners;
- [`ICLR2027_THEORY_ARCHITECTURE.md`](ICLR2027_THEORY_ARCHITECTURE.md) — early
  theory design exploration; current synthesis and canonical report win on
  conflict;
- [`three_completions/README.md`](three_completions/README.md) — index for the
  long-form derivation/verification bundle.

## Canonical evidence routing

| Question | Canonical source | Maximum role |
| --- | --- | --- |
| Current manuscript narrative edit order | [`ICLR2027_NARRATIVE_OPTIMIZATION_PLAN_20260826.md`](ICLR2027_NARRATIVE_OPTIMIZATION_PLAN_20260826.md) | Codex-facing 9-page wording/layout only; never a numerical owner |
| Related-work citation sufficiency and remaining novelty-attack papers | [`ICLR2027_CITATION_NOVELTY_AUDIT_20260826.md`](ICLR2027_CITATION_NOVELTY_AUDIT_20260826.md) | Codex-facing related-work / bib patch only; not an evidence owner; do not restore the NeurIPS PE zoo |
| Prior simulated-review decision | [`ICLR2027_MANUSCRIPT_OPTIMIZATION_AND_SIMULATED_REVIEW_20260826.md`](ICLR2027_MANUSCRIPT_OPTIMIZATION_AND_SIMULATED_REVIEW_20260826.md) | historical reviewer-risk record; most P0/P1 items are already in live TeX |
| Conceptual grammar and causal-variable separation | [`ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md`](ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md) | notation and routing only; never a numerical owner |
| Post-submission theory state and missing bridge | [`ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md`](ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md) | theory continuation and protocol design only; no new result or compute authorization |
| Matched-content phase 2x2 bridge preflight | [`attention-aware-retrofit/preflights/MATCHED_CONTENT_PHASE_2X2_BRIDGE_PREFLIGHT_20260827.md`](attention-aware-retrofit/preflights/MATCHED_CONTENT_PHASE_2X2_BRIDGE_PREFLIGHT_20260827.md) | frozen protocol design only; separates table-by-phase interaction from gain, adaptation, and routing |
| Protected progressive shift preflight | [`attention-aware-retrofit/preflights/PROTECTED_PROGRESSIVE_SHIFT_PREFLIGHT_20260827.md`](attention-aware-retrofit/preflights/PROTECTED_PROGRESSIVE_SHIFT_PREFLIGHT_20260827.md) | separate new protection test for the simple progressive curve; Stage 2 may invoke the matched-content bridge |
| Full sin/cos geometry, collision, stable-rank identity | [`FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`](FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) | main theory; static basis, not LM-quality predictor |
| Pure fixed-support interior-allocation identification | [`EXACT_RANGE_151M_3SEED_RESULT_20260820.md`](EXACT_RANGE_151M_3SEED_RESULT_20260820.md) plus M4 historical owner | main causal experiment |
| Exact frozen Q/K transplant obstruction | `../../rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` | exact impossibility for fixed static maps, not all approximate adapters |
| Weights/table co-adaptation | canonical full-RoPE report plus `../../scripts/analysis/attention_fisher_50m_probe.py` | diagnostic; keep separate from exact-range estimand |
| Scarce-budget systems flagship | `../../data/curated/table18_mla_3seed_aggregate.json` | three-seed 432M MLA result |
| Training-stage and scale persistence | `../../docs/exp/2026-03-06_phase15_750m_2k_to_4k_continue_results.md` and `../../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md` | 750M continuation and 1.485B trend |
| Cross-modal video-DiT breadth | [`VIDEO_DIT_HEAD_TO_HEAD_SEED42_RESULT_20260826.md`](VIDEO_DIT_HEAD_TO_HEAD_SEED42_RESULT_20260826.md) and `../../data/curated/video_dit_seed42_head_to_head_20260826.json` | one seed-42 matched head-to-head comparison; supporting modality breadth only |
| Matched mature phase exposure | `../../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md` | protocol-specific capability evidence |
| Mature fixed-support `z` controls and 151.9M crossing | [`attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`](attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) | internal causal case study; not a new operator-family claim |
| Zero-training practical session policy | [`attention-aware-retrofit/results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md`](attention-aware-retrofit/results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md) | internal OLMo deployment/capability evidence |
| Fresh natural-text policy persistence and mechanism controls | [`attention-aware-retrofit/results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md`](attention-aware-retrofit/results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md) | one-checkpoint, one-new-shard teacher-forced NLL; bundled policy, fixed-support `z`, profile detail, and routing remain separate estimands |
| Mature fixed-support co-adaptation oracle | [`attention-aware-retrofit/results/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md`](attention-aware-retrofit/results/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md) | one-checkpoint mechanism study: registered phase gate fails; matched natural-LM controls identify long-tail redistribution without full-NLL or 2Wiki dominance |
| Mature fixed-support dose response | [`attention-aware-retrofit/results/ALLOCATION_DOSE_RESPONSE_RESULT_20260826.md`](attention-aware-retrofit/results/ALLOCATION_DOSE_RESPONSE_RESULT_20260826.md) | one-checkpoint mechanism study: analytic joint gate fails; small frozen moves establish a graded full/tail response, not a selector or new method |
| Native 4K RULER diagnostic | [`attention-aware-retrofit/results/NATIVE_4K_RULER_DIAGNOSTIC_RESULT_20260826.md`](attention-aware-retrofit/results/NATIVE_4K_RULER_DIAGNOSTIC_RESULT_20260826.md) | descriptive task cells only; differing cross-length rows prevent a model-ceiling versus position-code conclusion |
| Failed mature direct-`z` calibration and analytic single-table routes | [`attention-aware-retrofit/results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md`](attention-aware-retrofit/results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md) and [`attention-aware-retrofit/results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md`](attention-aware-retrofit/results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md) | closed negative method gates; not incomplete downstream queues |
| Internal joint in-window/extrapolation allocation feasibility | [`attention-aware-retrofit/results/EXPERIMENT_REPORT_20260821.md`](attention-aware-retrofit/results/EXPERIMENT_REPORT_20260821.md) | two-seed phase-chord Pareto evidence; establishes feasibility internally, but method selection and seed scope block manuscript promotion |
| Target-free allocation screens on the M4 harness | [`attention-aware-retrofit/results/PHASE_ISOTROPY_50M_M4_RESULT_20260824.md`](attention-aware-retrofit/results/PHASE_ISOTROPY_50M_M4_RESULT_20260824.md) and [`attention-aware-retrofit/results/PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md`](attention-aware-retrofit/results/PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md) | `SCREEN_UNRESOLVED`; the small-model regime differs from the 151.9M control, but cross-protocol sign difference is not a noise-floor estimate or candidate rejection. See `../../INDEX.md` §6.2 |
| Historical post-GPU interpretation | [`attention-aware-retrofit/analysis/POST_GPU_REFLECTION_AND_PROBLEM2_ROADMAP_20260824.md`](attention-aware-retrofit/analysis/POST_GPU_REFLECTION_AND_PROBLEM2_ROADMAP_20260824.md) | historical decision memo; superseded for action priority by the 2026-08-25 mature co-adaptation owner and `../../INDEX.md` §6 |
| LeRoPE related-work facts | `../../rebuttal/rebuttal_0723/theory_results/LEROPE_CONCURRENT_WORK_NOTE_20260728.md` plus primary paper | positioning only |
| Failed LeRoPE profile oracle | [`audits/LEROPE_PROFILE_ORACLE_AUDIT_20260820.md`](audits/LEROPE_PROFILE_ORACLE_AUDIT_20260820.md) | internal negative |
| Failed attention-measure ordering gate | [`audits/KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md`](audits/KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md) | internal negative |

## Reviewer objections

Routing says who owns a number. This table says whether the number survives the
strongest objection a reviewer can raise against it. It is an internal
adversarial ledger: it is not a limitations inventory, and none of it belongs
verbatim in reviewer-facing text.

`defended` = an owner answers the objection on its own protocol. `partial` =
answered, but the answer is not where a reviewer will look. `open` = the
manuscript currently has no answer.

| # | What a reviewer takes away | Strongest objection | Current defense | Status |
| --- | --- | --- | --- | --- |
| R1 | Interior allocation `z` is a real training variable | "This is a base or support change under another name." | Both endpoints held bitwise; only the 30 interior frequencies move; 3/3 seeds, `-0.281/-0.176/-0.146` at `2x/4x/8x`. `EXACT_RANGE_151M_3SEED_RESULT_20260820.md` | defended |
| R2 | The proposed table is a good place on that axis | "Is \evq{} close to the best behavioural allocation, or merely one effective intervention?" | No behavioural ceiling is known. The static $r_2$ search in `../../scripts/analysis/third_axis_ceiling.py` is an internal geometry diagnostic and cannot answer this objection. The paper needs only the bounded construction claim it already makes. | open |
| R3 | Allocation improves long-context behaviour | "Match the support to the target length and your method loses: `+0.060/+0.227/+0.460`, 0/3 seeds, monotone in length." | Same owner §4 reports it; App.~\ref{sec:identification-details} carries it; the claim is identification, not additive gain over range transport | partial |
| R4 | The theory explains the effect | "You state that full-RoPE geometry does not predict LM quality, and your own 50M crossing (`7.14/76.20/23.05/7.16`) shows table-by-weights dominates. Then the theorem is decorative." | The identity is a budget account and the obstruction theorem is exact; neither is offered as a ranker. `AGENTS.md` claim ceilings; `../../INDEX.md` §3.4 | partial |
| R5 | The result holds at scale | "Causal identification is 151.9M and 50M. Everything larger is a different estimand." | Roles are labelled rather than pooled; 1.485B same-initialisation is the stated pretraining ceiling | defended |
| R6 | Closed form beats learning the table | "Why not learn it? LeRoPE does." | Closed form removes the table-search stage; LeRoPE is compatible evidence, and no matched comparator is claimed | partial |
| R7 | Retrofit is competitive with deployed practice | "Your `YaRN-style` operator is not cited YaRN." | Locked nomenclature separates them; the RULER contrast is stated against the repository operator | defended |

R2 is an open method-development question, not a defect in fixed-support
identification. R3 is a measured support-allocation interaction. Static geometry
does not turn either one into the other; `../../INDEX.md` §6 owns the research
response.

## Protocol boundaries

General claim ceilings, evidence-identity traps, and the "a plan/script/launch
log is not a result" rule live in [`../../AGENTS.md`](../../AGENTS.md) §2–§3 and
are not restated here. Below are only the boundaries specific to this
manuscript's owners.

- A scalar-base rule and an exponent-allocation rule are distinguished by
  their restricted intervention family and stage, even though a realised
  tensor admits multiple textual base/exponent parameterisations.
- The 151.9M crossing joins the 50M crossing as a co-adaptation diagnostic.
- The zero-training session policy owns the complete table/gain/router
  intervention; same-support and gain 2x2 owners identify its components.
- The fresh FineWeb owner closes the previously missing natural OLMo
  geometric/ramp/derived test. Its target-aware comparison changes support and
  table together and therefore owns no pure-allocation claim.
- Fixed-support allocation effects are length- and checkpoint-conditional.
  Neither geometric nor non-geometric membership predicts in-window or
  extrapolation quality; owners compare specified `z` values only.
- NLL/PPL, teacher-forced NLL gap, strict generation, token F1, exact match,
  RULER, and causal source-use are distinct endpoints.
- Row bootstraps condition on a fixed checkpoint and task set; they are not
  model-, task-population-, or training-seed uncertainty.

## Stop conditions

There is no stop list here. Stop conditions have two homes and one owner each:

- **volatile queue stops** (what not to rerun for the current submission) —
  [`../HANDOFF.md`](../HANDOFF.md) §6;
- **permanently closed routes** (what has been falsified and must not be
  reproposed) — [`../../INDEX.md`](../../INDEX.md) §3.4.

The Qwen aliased-`0.6175` correction is now an evidence-identity trap in
[`../../AGENTS.md`](../../AGENTS.md) §3.

The promoted frozen case study remains bounded to its stated protocols. The
natural OLMo same-support comparison is now complete; natural Qwen and
checkpoint-population generalisation remain unestablished, but neither is an
active queue. Joint in-window/extrapolation allocation remains the method
frontier and is not contradicted by the frozen-candidate failures. Consult the
handoff before proposing new compute.

## Placement rules

| New material | Destination |
| --- | --- |
| live mutable state | only `../HANDOFF.md` |
| central paper-facing owner | research root only when it joins the main claim architecture |
| mature retrofit result | `attention-aware-retrofit/results/` |
| mechanism/falsification | `attention-aware-retrofit/analysis/` |
| preregistration or revoked protocol | `attention-aware-retrofit/preflights/` |
| compact receipt | the relevant `evidence/` directory |
| manuscript/theory audit | `audits/` |
| external-model review | `external-reviews/<source-date>/` |

Raw checkpoints, evaluation rows, caches, server paths, and secrets stay
outside `paper-2027/`. External reviews never supersede canonical owners.
