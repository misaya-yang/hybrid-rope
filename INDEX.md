# INDEX — claim and owner router

- **Updated:** 2026-09-04
- **Evidence cut-off:** owners available through 2026-09-04.
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
  evidence; later research enters it only after owner-backed validation and an
  explicit author decision. Manuscript work does not prohibit active research.
- **Active method theme:** one static table/gain under about `0.12` separate
  Native NLL and downstream damage: maximize zero-training reach toward 8x,
  then test small physical-2x/4x LoRA on untouched 8x/16x/32x capability.
- **Not claimed:** arbitrary allocations always help, universal/unique optimum,
  SOTA, or static geometry as a trained-model quality predictor.

## Start the next research session here

Live server experiment outcomes and independent FFN review: [execution report](paper-2027/research/attention-aware-retrofit/results/SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904.md). Read it before the historical preflight; it includes the [source-only control guard](scripts/experiments/source_only_generation_guard.py) [Native-window guard](scripts/experiments/native_window_guard.py), [NIAH retention canary](scripts/experiments/niah_retention_canary.py), and [simple capability canary](scripts/experiments/simple_capability_canary.py).

| Intent | Read only this first | Deliverable |
| --- | --- | --- |
| Understand or improve the two core problems | [First-principles contract](paper-2027/research/attention-aware-retrofit/theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md) | Fixed witness, compact/near/far, actual Native constraints, reviewed dossier decisions |
| Identify measured compute and available model assets | [Compute/model snapshot, 2026-09-04](docs/overview/CURRENT_RESEARCH_COMPUTE_AND_MODEL_ASSETS_20260904.md) | Actual parameter counts, Native windows, file identities and completed execution boundaries; recheck volatile state |
| Run tomorrow on 5090/4080 Super 32GB | [Independent staged protocol](paper-2027/research/attention-aware-retrofit/preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md) | Prepare -> qualify assay -> compare -> stop/confirm, with raw receipts |
| Combine with YaRN and narrow the next paper increment | [YaRN correspondence and budget comparison](paper-2027/research/attention-aware-retrofit/theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md#10-yarn-correspondence-supervision-density-and-a-smaller-next-claim) | Static versus dynamic, dense LM versus answer supervision, three benchmark families; no automatic new sweep |
| Review FFN learning/forgetting and choose the next task from results | [Mechanism review §9](paper-2027/research/attention-aware-retrofit/theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md#9-ffn-review-learning-under-a-native-constraint), [E2 stage driver](scripts/train/run_native_constrained_transfer.sh), [receipt reviewer](scripts/analysis/review_native_constrained_transfer.py), [contract tests](tests/test_native_constrained_transfer.py) | Fresh Native endpoints and paired C/N/F; execution amendment governs unresolved resume/controls; no proxy promotion |
| Implement or audit the new assay | [Independent experiment](scripts/experiments/single_table_generation.py), [fixed controls](scripts/analysis/export_single_table_controls.py), [constrained trainer](scripts/train/train_single_table_native_constrained.py), [contracts](scripts/lib/rope/generation_contract.py), [tests](tests/test_single_table_generation.py) | Verify actual input/source placement, raw full output/EOS and controls |
| Improve the manuscript | [Outcome-dependent revision plan](paper-2027/REVISION_BRIEF.md#9-outcome-dependent-manuscript-edits-2026-09-04) | A specific source change backed by an existing or newly admitted owner |
| Investigate a previous failure | Search the correction/negative tables below | Explain exactly what failed and why a new test distinguishes a remaining alternative |

GPT-6-led sessions follow the same evidence rules as any other model. Historical
multi-model analyses are provenance, not authority or an automatic work queue.
Do not return to the old runner merely because its scripts already exist.

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
| What is the strongest practical no-update result? | Fully frozen derived allocation changes OLMo 16K RULER from `0.0056` to `0.6047`; coarse label-free allocation reaches `0.6104`; pure-`z` and broader deployment claims remain separate | [`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823`](paper-2027/research/attention-aware-retrofit/results/causal-mechanism/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) |
| Does it persist on fresh natural text? | Length-conditional NLL effect; not universal ranking | [`FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/causal-mechanism/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md) |
| What is the strongest tracked static-table result? | One OLMo table passes tested 1x gates and improves longer endpoints; ordered permutation can collapse | [`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831`](paper-2027/research/attention-aware-retrofit/results/coupling-transfer/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md) |
| Under one static table and one path, which completed form is retained, and what same-table LoRA follows? | The sequential OLMo stop tree retains the full 64-slot legacy-u p2 mask installed as log-s4 with fixed `c=.074`; this is a capability-first engineering incumbent among the named historical candidates, not a global optimum. The later exact Q/K-LoRA screen improves PG-19 but not measured generated capability. | [`selection/specification`](paper-2027/research/attention-aware-retrofit/theory/SINGLE_STATIC_LOG_P2_SELECTION_AND_LORA_20260903.md), [`LoRA result`](paper-2027/research/attention-aware-retrofit/results/adaptation-coadaptation/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md) |
| What is supported across K32/K128? | Normalized pair index is the best-tested coordinate, not a law or K-causal result | [`K32 confirmation`](paper-2027/research/attention-aware-retrofit/results/coupling-transfer/K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md), [`K128 confirmation`](paper-2027/research/attention-aware-retrofit/results/coupling-transfer/K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md), [`full RULER-13`](paper-2027/research/attention-aware-retrofit/results/coupling-transfer/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md) |
| Does long signal convert to natural QA? | **Unresolved:** 9/2 raw owners missing; constructed 38-row assay invalid | [`ZERO_TRAINING_TWO_DAY...`](paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md) |
| Do headwise clocks solve the joint objective? | **Exploratory/report-only:** variable-length capped panel, adaptive row reuse, no tracked executed bundle | [`HEADWISE_FACTORIZED...`](paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md) |
| Does calibration-frozen attention-displacement Selective-31 beat layer-matched random/reverse masks? | **Negative at exact candidate/protocol scope:** target-long generation is floor-equal while short KL/Top-1 and answer-token NLL reverse the expected Selective advantage; matched global controls resolve the reused panel | [`HEAD_SELECTIVE_ZERO_TRAINING...`](paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/HEAD_SELECTIVE_ZERO_TRAINING_SIX_ARM_RESULT_20260903.md) |
| Can Native checkpoint structure uniquely determine an ordered movement profile? | **No without an added preference.** The declared squared Native-geometry surrogate uniquely constructs `m = Iso(1-u)` and its executed OLMo arm improves natural retention/likelihood, but fresh core-4 is materially worse at 4K/8K; it exposes an endpoint-dependent tradeoff, not a latent law or current-p2 replacement. | [`theory`](paper-2027/research/attention-aware-retrofit/theory/NATIVE_ONLY_MOVEMENT_PROFILE_IDENTIFIABILITY_20260903.md), [`result`](paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md) |
| Can one non-Native static table guarantee exact Native short behaviour and change long geometry? | **No under universal exact-preservation and standard stationary-RoPE assumptions.** The completed per-request Native/s4 policy is the existing behavioural escape. A prefix-preserving long-frame key handoff removes the old cross-boundary phase mismatch in CPU algebra, but has no model-quality evidence. | [`STATIC_NATIVE_NO_HARM_AND_PREFIX_HANDOFF_20260903`](paper-2027/research/attention-aware-retrofit/theory/STATIC_NATIVE_NO_HARM_AND_PREFIX_HANDOFF_20260903.md), [`session-policy result`](paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md) |
| Is the Selective-31 calibration score a universal functional sensitivity, and does joint Q/K--frequency relabeling invalidate the ordered-coupling results? | **No.** The score is exact endpoint attention-map displacement on a frozen calibration pack, not `chi_func`; exact joint relabeling is a gauge identity, while existing frequency-only permutations intentionally hold Q/K fixed. | [`LOCAL_FUNCTIONAL_COMPATIBILITY_AND_GAUGE_AUDIT_20260903`](paper-2027/research/attention-aware-retrofit/theory/LOCAL_FUNCTIONAL_COMPATIBILITY_AND_GAUGE_AUDIT_20260903.md) |
| Do finite scale-orbit boundary and Fourier-rank quantities predict mature-model behaviour? | **Negative selector result.** Exact boundary count changes `6 -> 64` under a behaviourally invisible ULP perturbation; p2 and the failed exact chain share zero Gram lower bound and saturated operator error but have opposite 4x utility. A CPU follow-up finds old `D*` plus phase safety diagnose these extreme failures, but the earlier one-turn-floor counterexample still falsifies `D*` as a general selector. | [`result`](paper-2027/research/attention-aware-retrofit/results/operator-analysis/SCALE_ORBIT_BOUNDARY_VALIDATION_RESULT_20260904.md), [`transport preflight`](paper-2027/research/attention-aware-retrofit/preflights/operator-analysis/SCALE_ORBIT_TRANSPORT_RESIDUAL_PREFLIGHT_20260904.md), [`prior axis falsification`](paper-2027/research/attention-aware-retrofit/analysis/RETROFIT_AXIS_FALSIFICATION_20260822.md) |
| What survives a proof, novelty, and tightness audit of the supplied finite scale-covariance derivation? | **The mathematics survives; the current empirical-tightness route does not.** Theorem 5 extends to continuous finite-dimensional real orthogonal RPE and has a dimension-free separation-order corollary. Exact obstruction/boundary leakage are prior art and novelty is not certified. A resolving synthetic control passes, but 45 bounded-condition trajectories select identity and saturate near error `2`; Ky-Fan is zero/tiny and non-ranking. Multilevel is stopped. | [`proof/novelty owner`](paper-2027/research/attention-aware-retrofit/theory/FINITE_SCALE_COVARIANCE_PROOF_NOVELTY_AND_TIGHTNESS_AUDIT_20260904.md), [`tightness result`](paper-2027/research/attention-aware-retrofit/results/operator-analysis/SCALE_CONJUGACY_TIGHTNESS_RESULT_20260904.md), [`preflight`](paper-2027/research/attention-aware-retrofit/preflights/operator-analysis/SCALE_CONJUGACY_TIGHTNESS_PREFLIGHT_20260904.md) |
| Has same-substrate log-p2 Q/K LoRA already been executed? | **Yes, at unit gain and at the retained `c=.074`.** Both improve paired PG-19 NLL; neither establishes generated-task capability improvement. | [`LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904`](paper-2027/research/attention-aware-retrofit/results/adaptation-coadaptation/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md) |
| Does exact log-p2 plus `c=.074` benefit from matched Q/K-only adaptation? | **Likelihood only in the measured panel.** PG-19 improves at 1x/4x, five-task macros are slightly negative/unresolved, and fresh core-4 changes `-.0100/+.0025/-.0225` at 4K/8K/16K. | [`LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904`](paper-2027/research/attention-aware-retrofit/results/adaptation-coadaptation/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md) |
| What is the corrected next experiment for one-table zero-training and low-step adaptation? | **Prospective, independent redesign.** Qualify compact/near/far lawful worlds and exact EOS; confirm fixed N/Z/G/Y; use all-linear r16 with original-Native functional constraints on qualified natural data. Earlier QK/source-contrast prototype is superseded. | [`CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904`](paper-2027/research/attention-aware-retrofit/preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md) |
| Where is the earlier single-table synthesis preserved? | **Historical research brief.** Mixed C2/p2 fitted ceiling and phase-cost lower-bound use are corrected; its prototype execution order is superseded by the independent protocol above. | [`SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904`](paper-2027/research/attention-aware-retrofit/analysis/SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904.md) |
| What can the 9/2 first-principles memo support? | Only explicitly retained identities under assumptions; T4/T5/T7 and behavioural generalizations are retracted/disputed | [`FIRST_PRINCIPLES...`](paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md) |

Endpoint boundaries remain strict: NLL/PPL, answer-token NLL, teacher-forced
gap, strict generation, token F1, exact match, RULER/NIAH, QA, causal source
use, adaptation, and transfer are different evidence tiers.

## 3. Correction ledger

| Search hit | Current use | Replacement or reason |
| --- | --- | --- |
| 28 direct-hybrid zero-score receipts (2026-07-26) | **Invalid method evidence** | Native/EVQ buffer alias; use [`OLMO2_POSTHOC...`](rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md) |
| p2 Native boundary near 3.91/4.01 from s2/s4 quadratic | **Superseded as same-path prediction** | s2 is compressed C2/G(x), s4 is full-p2; no matched tensor path. [Correction and replacement](paper-2027/research/attention-aware-retrofit/theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md) |
| phase-cost box/isotonic ceiling in the 9/4 brief | **Unresolved lower-bound step** | no certified feasible-set/domination argument or finite-region Native curvature; same replacement owner |
| old factor-frontier / paired-view LoRA runner and first new QK/source-margin prototype | **Superseded execution plans** | use fixed-witness diagnostic and Native-constrained engine; no prototype run was executed |
| old exact-range three-seed aggregate | **Superseded; never splice** | use raw-backed [`EXACT_RANGE...`](paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) |
| Qwen 128K `0.6175` | **Invalid aliased value** | corrected result is `0.5400` in [`SAME_SUPPORT...`](paper-2027/research/attention-aware-retrofit/results/causal-mechanism/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) |
| generated `FAILED_50M_GATE` | **Superseded: `SCREEN_UNRESOLVED`** | positive control failed; use the two owners in §4 |
| old Gemma 16K zero with 8K reference | **Superseded/confounded** | use [`REFERENCE_CORRECTED_K128_RESULT_20260901`](paper-2027/research/attention-aware-retrofit/results/coupling-transfer/REFERENCE_CORRECTED_K128_RESULT_20260901.md) |
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
| direct-`z` two-document calibration | candidate/protocol negative | [`DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md) |
| two analytic Native-support tables | two candidates failed; no inherent trade-off theorem | [`ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md) |
| continuous-boundary-slope operator | only this implementation is closed | [`ZERO_TRAINING_MECHANISM_AND_CEILING_20260826`](paper-2027/research/attention-aware-retrofit/analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md) |
| exact Native no-harm from one non-Native static table | class-level obstruction under universal content/short-position equality; approximate retention and nonstandard/dynamic operators remain open | [`STATIC_NATIVE_NO_HARM_AND_PREFIX_HANDOFF_20260903`](paper-2027/research/attention-aware-retrofit/theory/STATIC_NATIVE_NO_HARM_AND_PREFIX_HANDOFF_20260903.md) |
| phase-isotropy / pair-volume / min-eigenvalue | `SCREEN_UNRESOLVED`, not negative | [`PHASE_ISOTROPY...`](paper-2027/research/attention-aware-retrofit/results/PHASE_ISOTROPY_50M_M4_RESULT_20260824.md), [`PHASE_ALLOCATION...`](paper-2027/research/attention-aware-retrofit/results/PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md) |
| Native-retention + natural long-QA/EOS assay | unresolved validity | §3 corrections; validity precedes method selection |
| Native-compatible/long-capable intervention | one arm passes the reused natural 1x double gate and retains long likelihood/RULER, but fresh core-4 is negative at 4K/8K; no universal jointly passing law | [`NATIVE_ISOTONIC_PROFILE_RESULT_20260903`](paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md) |

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

### Active method work — two-track programme

The target is one static table/gain with about `0.12` maximum damage separately
on Native PPL and downstream tasks. Use strict .88 retention; .875 is a labelled
historical sensitivity. The retained full-p2 s4/c=.074 point remains a marginal
incumbent, not an optimum or a proven 4x ceiling.

**Z:** identify whether a frozen candidate's failure is already Native/local,
source-distance dependent, or output/EOS dependent while confirming the existing
witness. No new factor/gain/curve search. Table factor is not useful physical reach.

**F:** physical <=16K task exposure; untouched 8x/16x/32x capability.
The [Native-constrained engine](scripts/train/train_single_table_native_constrained.py)
uses fixed all-linear r16 N/Z/Y arms, lawful-world complete-trajectory loss and
original-Native teacher constraints. Qualified natural/replay assets are required;
the earlier QK/source-contrast ladder is not the execution plan.
All length renderings of a semantic group stay together. No sealed outcome may
select an adapter, loss, table, gain or new early-stop rule.

The [first-principles owner](paper-2027/research/attention-aware-retrofit/theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md)
explains the derivations and alternative openings; the
[execution contract](paper-2027/research/attention-aware-retrofit/preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md)
defines tomorrow's commands and stop rules. The older
[SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904](paper-2027/research/attention-aware-retrofit/analysis/SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904.md)
is context only. Per the recorded handoff, the prior "no GPU run is active" state is superseded by author-authorized execution; see HANDOFF.md; verify live
machine state before spending compute. No run is launched by documentation.

Every result must change a named decision. A valid candidate failure closes
that candidate/protocol, never the class. Unresolved controls close neither.

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
