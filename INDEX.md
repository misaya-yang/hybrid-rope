# INDEX — claim and owner router

- **Updated:** 2026-09-03
- **Evidence cut-off:** owners available through 2026-09-03.
- **Role:** route an exact question to its current owner, correction, or scoped
  negative. This is not a report, timeline, or live handoff.

Do not read every linked file. Search this index for the question, open the
smallest matching row, then read its owner and raw/receipt artifact only to the
depth the task requires. Rules are in `AGENTS.md`; paper orientation is in
`README.md`; live state is in `paper-2027/HANDOFF.md`.

## 0. Current paper status

- **Paper identity:** a finite RoPE table decomposes into sampled support
  $(a,R)$ and interior allocation $z$; fixed-support interventions identify
  $z$, target-aware retargeting shows interaction, and exact geometry exposes
  the finite spectral budget.
- **Analytic construction:** EVQ-Cosh is closed-form and zero-learned-parameter,
  unique only within its stated convex surrogate.
- **Strongest practical consequence:** fully frozen model-relative structured
  allocations produce large no-update extrapolation/downstream gains. Matched
  adaptation and from-training/co-adapted evidence supply distinct lifecycle
  consequences and breadth.
- **Submission state:** the manuscript design is frozen around completed
  evidence. No new submission experiment is planned; current work is the
  September title/abstract/metadata and full-paper verification sequence.
- **Not claimed:** arbitrary allocations always help, universal/unique optimum,
  SOTA, or static geometry as a trained-model quality predictor.

## 1. Paper-level owners

| Question | Current answer | Owner |
| --- | --- | --- |
| Does allocation matter at fixed support during training? | Yes in the 151.9M three-seed protocol; retain support/seed scope | [`EXACT_RANGE_151M_3SEED_RESULT_20260820`](paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) + [JSON](paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.json) |
| Does the effect persist across exact-range configurations? | Yes in M4; matched non-Cosh shape remains competitive | [`M4_EXACT_RANGE_FACTORIAL_RESULT_20260726`](rebuttal/rebuttal_0723/theory_results/M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md) |
| What does full sin/cos geometry prove? | Redundancy/effective dimension and counterexamples; not LM ranking | [`FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819`](paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) |
| How are support and allocation separated? | `x_k = a + R z_k`; notation/intervention grammar, not a number owner | [`ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823`](paper-2027/research/foundations/ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md) |
| Can RoPE/attention structure alone determine optimal `z`, a frequency system, or mature-checkpoint movement? | **Boundary only:** distribution-free behavioural optimality is non-identifiable. This does not close repository-constrained selection from completed evidence. | [`ROPE_OPTIMALITY_IDENTIFIABILITY_AND_CONDITIONAL_EQUATIONS_20260903`](paper-2027/research/foundations/ROPE_OPTIMALITY_IDENTIFIABILITY_AND_CONDITIONAL_EQUATIONS_20260903.md) |
| What is the exact frozen transplant boundary? | Position-independent invertible Q/K compensation requires matching multisets up to sign/permutation | [`OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726`](rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md) |
| What is the bounded EVQ-Cosh theorem? | Unique only for its stated convex surrogate | [`03_theory.tex`](paper-2027/sections/03_theory.tex) + [`a1_proofs.tex`](paper-2027/appendix/a1_proofs.tex) |
| What supports the matched-adaptation route? | Protocol-specific task-family length transfer at 1.485B and causal source use at 8B; not pure frozen-`z` or pretraining-scale evidence | [`OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729`](rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md), [`EVQ_8B_ADAPTATION_EVIDENCE_20260724`](rebuttal/rebuttal_0723/theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md) |
| What breadth supports the paper? | Protocol-specific 432M MLA, 750M continuation, existing 1.485B scale line, and Video-DiT | [`table18 MLA`](data/curated/table18_mla_3seed_aggregate.json), [`750M report`](docs/exp/2026-03/2026-03-06_phase15_750m_2k_to_4k_continue_results.md), [`OLMO2 1B`](rebuttal/rebuttal_0723/theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md), [`Video-DiT`](paper-2027/research/evidence/VIDEO_DIT_HEAD_TO_HEAD_SEED42_RESULT_20260826.md) |

## 2. Mature-checkpoint owners

Open the full local catalogue only when needed:
[`results/README.md`](paper-2027/research/attention-aware-retrofit/results/README.md).

| Question | Current status | Owner |
| --- | --- | --- |
| What is the strongest practical no-update result? | Fully frozen derived allocation changes OLMo 16K RULER from `0.0056` to `0.6047`; coarse label-free allocation reaches `0.6104`; pure-`z` and broader deployment claims remain separate | [`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823`](paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) |
| Does it persist on fresh natural text? | Length-conditional NLL effect; not universal ranking | [`FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md) |
| What is the strongest tracked static-table result? | One OLMo table passes tested 1x gates and improves longer endpoints; ordered permutation can collapse | [`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831`](paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md) |
| Under one static table and one path, which completed form is retained, and what same-table LoRA follows? | The sequential OLMo stop tree retains the full 64-slot legacy-u p2 mask installed as log-s4 with fixed `c=.074`; this is a capability-first engineering incumbent among the named historical candidates, not a global optimum. Same-substrate Q/K LoRA is specified but unexecuted. | [`SINGLE_STATIC_LOG_P2_SELECTION_AND_LORA_20260903`](paper-2027/research/attention-aware-retrofit/theory/SINGLE_STATIC_LOG_P2_SELECTION_AND_LORA_20260903.md) |
| What is supported across K32/K128? | Normalized pair index is the best-tested coordinate, not a law or K-causal result | [`K32 confirmation`](paper-2027/research/attention-aware-retrofit/results/K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md), [`K128 confirmation`](paper-2027/research/attention-aware-retrofit/results/K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md), [`full RULER-13`](paper-2027/research/attention-aware-retrofit/results/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md) |
| Does long signal convert to natural QA? | **Unresolved:** 9/2 raw owners missing; constructed 38-row assay invalid | [`ZERO_TRAINING_TWO_DAY...`](paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md) |
| Do headwise clocks solve the joint objective? | **Exploratory/report-only:** variable-length capped panel, adaptive row reuse, no tracked executed bundle | [`HEADWISE_FACTORIZED...`](paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md) |
| Does calibration-frozen attention-displacement Selective-31 beat layer-matched random/reverse masks? | **Negative at exact candidate/protocol scope:** target-long generation is floor-equal while short KL/Top-1 and answer-token NLL reverse the expected Selective advantage; matched global controls resolve the reused panel | [`HEAD_SELECTIVE_ZERO_TRAINING...`](paper-2027/research/attention-aware-retrofit/results/HEAD_SELECTIVE_ZERO_TRAINING_SIX_ARM_RESULT_20260903.md) |
| Can Native checkpoint structure uniquely determine an ordered movement profile? | **No without an added preference.** The declared squared Native-geometry surrogate uniquely constructs `m = Iso(1-u)` and its executed OLMo arm improves natural retention/likelihood, but fresh core-4 is materially worse at 4K/8K; it exposes an endpoint-dependent tradeoff, not a latent law or current-p2 replacement. | [`theory`](paper-2027/research/attention-aware-retrofit/theory/NATIVE_ONLY_MOVEMENT_PROFILE_IDENTIFIABILITY_20260903.md), [`result`](paper-2027/research/attention-aware-retrofit/results/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md) |
| Can one non-Native static table guarantee exact Native short behaviour and change long geometry? | **No under universal exact-preservation and standard stationary-RoPE assumptions.** The completed per-request Native/s4 policy is the existing behavioural escape. A prefix-preserving long-frame key handoff removes the old cross-boundary phase mismatch in CPU algebra, but has no model-quality evidence. | [`STATIC_NATIVE_NO_HARM_AND_PREFIX_HANDOFF_20260903`](paper-2027/research/attention-aware-retrofit/theory/STATIC_NATIVE_NO_HARM_AND_PREFIX_HANDOFF_20260903.md), [`session-policy result`](paper-2027/research/attention-aware-retrofit/results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md) |
| Is the Selective-31 calibration score a universal functional sensitivity, and does joint Q/K--frequency relabeling invalidate the ordered-coupling results? | **No.** The score is exact endpoint attention-map displacement on a frozen calibration pack, not `chi_func`; exact joint relabeling is a gauge identity, while existing frequency-only permutations intentionally hold Q/K fixed. | [`LOCAL_FUNCTIONAL_COMPATIBILITY_AND_GAUGE_AUDIT_20260903`](paper-2027/research/attention-aware-retrofit/theory/LOCAL_FUNCTIONAL_COMPATIBILITY_AND_GAUGE_AUDIT_20260903.md) |
| What can the 9/2 first-principles memo support? | Only explicitly retained identities under assumptions; T4/T5/T7 and behavioural generalizations are retracted/disputed | [`FIRST_PRINCIPLES...`](paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md) |

Endpoint boundaries remain strict: NLL/PPL, answer-token NLL, teacher-forced
gap, strict generation, token F1, exact match, RULER/NIAH, QA, causal source
use, adaptation, and transfer are different evidence tiers.

## 3. Correction ledger

| Search hit | Current use | Replacement or reason |
| --- | --- | --- |
| 28 direct-hybrid zero-score receipts (2026-07-26) | **Invalid method evidence** | Native/EVQ buffer alias; use [`OLMO2_POSTHOC...`](rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md) |
| old exact-range three-seed aggregate | **Superseded; never splice** | use raw-backed [`EXACT_RANGE...`](paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) |
| Qwen 128K `0.6175` | **Invalid aliased value** | corrected result is `0.5400` in [`SAME_SUPPORT...`](paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) |
| generated `FAILED_50M_GATE` | **Superseded: `SCREEN_UNRESOLVED`** | positive control failed; use the two owners in §4 |
| old Gemma 16K zero with 8K reference | **Superseded/confounded** | use [`REFERENCE_CORRECTED_K128_RESULT_20260901`](paper-2027/research/attention-aware-retrofit/results/REFERENCE_CORRECTED_K128_RESULT_20260901.md) |
| 38-row “16K Hotpot” Fact D | **Invalid for claims/gates** | selected constructed stress, non-official filler, raw missing |
| 9/2 gain-sweep session facts | **Unverified** | forensic lead only; no optimum, mechanism, or class conclusion |
| Hotpot-200 headwise comparison | **Report-only exploratory** | variable length capped at 16K, adaptive reuse, executed/raw bundle untracked |
| first-principles T4/T5/T7 and exact-conditioning claims | **Retracted/disputed** | invalid bound division, unrestricted torus claim, and novelty ratio |
| first-principles T1 arc-length proof | **Corrected; conclusion retained under its injective finite-arc assumptions** | old proof did not establish uniform per-slot scaling; use the tangent-ray proof in [`ROPE_OPTIMALITY...` §4.5](paper-2027/research/foundations/ROPE_OPTIMALITY_IDENTIFIABILITY_AND_CONDITIONAL_EQUATIONS_20260903.md#45-correction-to-the-historical-pi-arc-proof) |
| 9/3 optimality owner used as a terminal method verdict | **Corrected: supporting boundary only** | distribution-free impossibility does not answer the author-required single-static-table selection problem; that question remains active |
| two-day synthesis “only allowed” route | **Superseded candidate negative** | holdout failure; not a queue or method-class result |

An author-chosen threshold such as `0.875` is an operational tolerance, not a
theorem. Passing or missing it does not create a scientific discontinuity.

## 4. Scoped negatives and unresolved questions

| Object | Exact status | Owner |
| --- | --- | --- |
| cosine-only collision / lower collision-logdet as behavioural rankers | universal sufficiency refuted; regularizer use remains open | [`FULL_ROPE...`](paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) |
| attention-Fisher `kappa_att` | tested ranker negative, not all attention-aware metrics | [`KAPPA_ATTENTION_MEASURE_AUDIT_20260820`](paper-2027/research/audits/KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md) |
| LeRoPE `w^(1/3)` oracle | published-shape operationalization negative, not the curvature class | [`LEROPE_PROFILE_ORACLE_AUDIT_20260820`](paper-2027/research/audits/LEROPE_PROFILE_ORACLE_AUDIT_20260820.md) |
| `D*`, coverage residual, phase-risk selectors | tested selector negatives in the registered panel | [`RETROFIT_AXIS_FALSIFICATION_20260822`](paper-2027/research/attention-aware-retrofit/analysis/RETROFIT_AXIS_FALSIFICATION_20260822.md) |
| direct-`z` two-document calibration | candidate/protocol negative | [`DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md) |
| two analytic Native-support tables | two candidates failed; no inherent trade-off theorem | [`ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md) |
| continuous-boundary-slope operator | only this implementation is closed | [`ZERO_TRAINING_MECHANISM_AND_CEILING_20260826`](paper-2027/research/attention-aware-retrofit/analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md) |
| exact Native no-harm from one non-Native static table | class-level obstruction under universal content/short-position equality; approximate retention and nonstandard/dynamic operators remain open | [`STATIC_NATIVE_NO_HARM_AND_PREFIX_HANDOFF_20260903`](paper-2027/research/attention-aware-retrofit/theory/STATIC_NATIVE_NO_HARM_AND_PREFIX_HANDOFF_20260903.md) |
| phase-isotropy / pair-volume / min-eigenvalue | `SCREEN_UNRESOLVED`, not negative | [`PHASE_ISOTROPY...`](paper-2027/research/attention-aware-retrofit/results/PHASE_ISOTROPY_50M_M4_RESULT_20260824.md), [`PHASE_ALLOCATION...`](paper-2027/research/attention-aware-retrofit/results/PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md) |
| Native-retention + natural long-QA/EOS assay | unresolved validity | §3 corrections; validity precedes method selection |
| Native-compatible/long-capable intervention | one arm passes the reused natural 1x double gate and retains long likelihood/RULER, but fresh core-4 is negative at 4K/8K; no universal jointly passing law | [`NATIVE_ISOTONIC_PROFILE_RESULT_20260903`](paper-2027/research/attention-aware-retrofit/results/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md) |

No content-blind static scalar score in this repository has prospectively
ranked LM behaviour across the required regimes. A numerical static search is
only best-found under its stated support/measure/optimizer/restarts and is not a
global or behavioural ceiling.

## 5. What to do

### Submission — active

1. By 2026-09-17, freeze the title, abstract, author roster, and author metadata
   after owner-level and live-policy checks.
2. Submit the matching official abstract and metadata by 2026-09-18, 11:59 PM
   AoE, and record the platform receipt in the handoff.
3. Preserve the reviewer path: decomposition → identification/retargeting →
   exact geometry → EVQ-Cosh construction → frozen/adaptation/from-training
   consequences. Repair the title, abstract, first page, and Figure 1 only where
   this path is unclear or scientifically wrong.
4. Complete the owner-by-owner number/protocol audit, rebuild the curated
   supplement, run final build/anonymity/policy/visual checks, and submit the
   verified paper by 2026-09-25.

Exact live progress and authorization belong only in
[`HANDOFF.md`](paper-2027/HANDOFF.md).

### Post-submission method work — historical single-table selection resolved

The completed-history answer for the **deterministic static pure-`z` table**
question is routed by
[`SINGLE_STATIC_LOG_P2_SELECTION_AND_LORA_20260903`](paper-2027/research/attention-aware-retrofit/theory/SINGLE_STATIC_LOG_P2_SELECTION_AND_LORA_20260903.md).
Under the paper's capability-first sequential replacement rules, the retained
OLMo form is the full 64-slot legacy-u p2 movement installed as log-s4 with one
fixed `c=.074` gain. It uses one table and one path at every length. It is not a
formal optimum: Native-isotonic has better natural-likelihood/retention points
but loses the fresh structured 4K/8K replacement contrast.

This resolves the **completed-history deployment selection** only. It does not
satisfy or close the separate README direction that asks for a new candidate
derived prospectively before LM outcomes: p2 and `.074` retain their disclosed
outcome-selection history.

A request-level Native/long router, two tables, cache branching, frequency
segmentation, or a length-time switch is not this answer. The Qwen result is
long-capability construction transfer only; K32/K128 normalized-index results
belong to a distinct C2-derived family.

The declared LoRA follow-up freezes this same table/gain at every training and
evaluation length and adapts Q/K only; it is an unexecuted same-substrate
specification, not a transplanted Stage-A result. No new LoRA result, global
training-free SOTA comparison, GPU method-development experiment, or model
execution is currently authorized.

Protocol/assay validity, controls, executed identity, raw-owner output, budget,
and stop conditions must pass preflight before any authorized compute. These are
execution gates, not a replacement research direction. No GPU method-development
experiment is currently active or authorized.

## 6. On-demand routes

| Need | Open |
| --- | --- |
| Why the question changed | [`TIMELINE.md`](paper-2027/research/history/TIMELINE.md) |
| Paper-level theory/evidence | [`research/README.md`](paper-2027/research/README.md) |
| Mature-result catalogue | [`results/README.md`](paper-2027/research/attention-aware-retrofit/results/README.md) |
| Mature theory status | [`theory/README.md`](paper-2027/research/attention-aware-retrofit/theory/README.md) |
| Historical reports | [`docs/exp/`](docs/exp/) |
| July review/evidence | [`rebuttal/rebuttal_0723/README.md`](rebuttal/rebuttal_0723/README.md) |
| Reusable RoPE code | [`scripts/lib/rope/`](scripts/lib/rope/) |
| Static rank diagnostic | [`scripts/analysis/third_axis_ceiling.py`](scripts/analysis/third_axis_ceiling.py) — algebraic only, not an LM owner |
| Current evaluation utilities | [`scripts/eval/`](scripts/eval/) — require live protocol and authorization |

Do not copy credentials, private paths, raw checkpoints, caches, or ignored
evidence into Git to bridge machines.
