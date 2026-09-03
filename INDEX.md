# INDEX — claim and owner router

- **Updated:** 2026-09-03
- **Evidence cut-off:** owners available through 2026-09-02; later material has
  not been reconciled here.
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
| What is supported across K32/K128? | Normalized pair index is the best-tested coordinate, not a law or K-causal result | [`K32 confirmation`](paper-2027/research/attention-aware-retrofit/results/K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md), [`K128 confirmation`](paper-2027/research/attention-aware-retrofit/results/K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md), [`full RULER-13`](paper-2027/research/attention-aware-retrofit/results/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md) |
| Does long signal convert to natural QA? | **Unresolved:** 9/2 raw owners missing; constructed 38-row assay invalid | [`ZERO_TRAINING_TWO_DAY...`](paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md) |
| Do headwise clocks solve the joint objective? | **Exploratory/report-only:** variable-length capped panel, adaptive row reuse, no tracked executed bundle | [`HEADWISE_FACTORIZED...`](paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md) |
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
| phase-isotropy / pair-volume / min-eigenvalue | `SCREEN_UNRESOLVED`, not negative | [`PHASE_ISOTROPY...`](paper-2027/research/attention-aware-retrofit/results/PHASE_ISOTROPY_50M_M4_RESULT_20260824.md), [`PHASE_ALLOCATION...`](paper-2027/research/attention-aware-retrofit/results/PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md) |
| Native-retention + natural long-QA/EOS assay | unresolved validity | §3 corrections; validity precedes method selection |
| Native-compatible/long-capable intervention | no jointly passing valid arm; no impossibility theorem | tracked 9/1 owners + report-only 9/2 headwise work |

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

### Post-submission research — separate and not authorized

The author-ordered first direction is a deterministic static pure-`z` table on
a frozen checkpoint. Derive the candidate before LM evaluation without learning
or loss-based frequency search. Endpoint movement is allowed; the first gate is
a declared small Native-window cost plus improvement at `2x`/`4x`, followed by
untouched downstream evaluation.

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
