# Research protocol reference

- **Status/date:** 2026-09-06; research conventions and historical routing annotations extracted from root documents.
- **Purpose:** explain research objectives, metric conventions, evidence labels,
  protocol interpretation and method identities when a task needs them.
- **Sources:** the [first-principles owner](attention-aware-retrofit/theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md),
  the [amended execution protocol](attention-aware-retrofit/preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md),
  and [retention implementation](../../scripts/lib/rope/generation_contract.py).
- **Evidence boundary:** this is a documentation consolidation, not a result
  owner, new measurement, validation receipt, experiment queue or run authorization.
  It replaces only the explanatory material formerly embedded in AGENTS.md;
  existing scientific owners are not superseded. Their exact protocols govern
  claims; [HANDOFF.md](../HANDOFF.md) governs live state.
- **Use:** read the relevant section on demand through [INDEX.md](../../index.md).
  Numeric gates and current-research descriptions below are the dated convention,
  not permanent agent instructions. Recheck the linked amended protocol before use.
  Backticked repository paths are relative to the repository root.

## Research context and metric conventions

**Historical programme snapshot.** The constraints and numeric gates in this
section belong to the pre-reconstruction programme. For current planning and new
confirmation thresholds, use [REVISION_BRIEF v5](../REVISION_BRIEF.md).
Old gates remain attached to old results; no new threshold is adopted here.


The preceding programme studied two questions, with its owners routed in `INDEX.md`:
zero training with one global static table and frozen weights, and light
adaptation with a fixed deployment table. Training exposure, physical reach,
and multiples of the checkpoint's verified Native window are different quantities.
Do not infer useful reach from table factor or label within-Native evaluation
as extrapolation.

- The preceding protocol gated Native PPL and generated-task retention separately
  at `>= 0.88`, with `0.875` reported as marginal. `scripts/lib/rope/generation_contract.py`
  defines PPL retention as `exp(native_nll - candidate_nll)` and task retention
  as the candidate/Native score ratio. The approximate `0.12` damage budget is
  not a raw NLL-difference threshold. Preserve the protocol's endpoint and
  uncertainty requirements; these are operational gates, not discontinuities.
- Qualify complete-output/EOS scoring, Native-compact solvability, matched
  near/far source twins, and deleted-source controls before interpreting
  generation or selecting a candidate. Keep synthetic diagnosis, official
  benchmarks, and natural-task generalization separate. Qualified natural data
  owns transfer claims; proxy losses, attention/operator diagnostics, sampled
  KL, and gradients do not establish capability or Native retention.
- Student replay uses its actual deployment table against the original-Native
  teacher. Keep every length rendering of a semantic group in the same split.
  Exposed confirmation or sealed test outcomes cannot tune a table, adapter,
  loss, gain, or stopping rule, or become fresh independent confirmation.
- Follow the current amended protocol, not historical step/resume commands.
  The prepared comparison is routed through protocol §10 in `INDEX.md` and
  `scripts/experiments/matched_transfer_round.py`; its CPU template is not run
  authorization. Do not reopen stopped candidates or new curve/gain/scale
  searches from an old memo. Architecture choices such as FFN-only updates are
  hypotheses, not project-wide requirements.
- Apply the declared candidate/family stop rules only within a resolving
  protocol. A valid zero primary long-generation score or failed Native gate
  stops the affected candidate/matrix. Unresolved controls select or close
  neither a candidate nor a route. A candidate failure does not close a method
  class; class closure needs a matching impossibility result or prospective
  resolving evidence.

## Evidence vocabulary and interpretation

Classify load-bearing scientific statements by their evidential status:

| Label | Use |
| --- | --- |
| Observation | Measured under a validated protocol with an owner |
| Derived result | Checked proof under stated assumptions |
| Interpretation | Explanation not uniquely identified by observations |
| Working hypothesis | Prospective, falsifiable, not established |
| Negative result | Valid evidence against a named claim/candidate and scope |
| Invalid | Known protocol, implementation, data, or measurement defect |
| Unresolved | Insufficient validity, resolving power, or evidence |
| Superseded | Explicit replacement owner; retain historical record |

Before using an outcome, verify the question/estimand; checkpoint, table/operator,
reference length, gain/routing and intervention parity; split, tokenized prompt,
position/evidence placement, reference/request lengths, generation reserve,
decoder and scorer; resolving controls; and recoverable executed code, raw rows,
hashes, failures and exclusions. Preserve the budget, metric and uncertainty
unit: seeds, rows, tasks and trajectories are not interchangeable. “Same” needs
verified identical fields; “matched” needs a declared contract.

- A plan, script, launch log, checkpoint inventory or summary does not prove a
  completed valid measurement. Model reviews and confidence are analysis to
  check, never scientific authority. Use the most direct valid owner of the
  estimand; filenames, recency and routing summaries confer no authority.
- Invalid or unresolved assays cannot tune parameters, pass/fail claims,
  supersede observations or enter synthesis as measurements. Repair provenance
  or validity before spending compute to repeat completed work.
- New or amended owners state status/date, exact question, protocol/assumptions,
  artifact/receipt identity, supported and unsupported claims, and corrections
  near the top. Route every new owner in `INDEX.md` in the same change.
- For conflicting owners, compare estimand and protocol first. Preserve distinct
  regimes; for the same regime, trace raw artifacts and mark the unsupported
  claim superseded or unresolved. A correction must leave a visible notice on
  the stale searchable source and route its replacement in `INDEX.md`.
- Cross-checkpoint, cross-`K`, cross-task and cross-scale claims need matched
  factorization. A numerical search gives only a best-found value under its
  support, measure, optimizer and restarts. Post-outcome theory is postdiction
  until a frozen independent behavioural prediction succeeds.
- Check proof quantifiers, dimensions, limits and minimal counterexamples.
  Do not divide upper bounds, infer lower bounds from upper bounds, or promote
  a construction sketch to a theorem.

Preserve method identities: `Geo` is the geometric training baseline; `Native`
is the unmodified checkpoint/table; `FMRoPE` is the exact-range paper-faithful
arm; anchored EVQ-Cosh is endpoint-normalized to FMRoPE; repository `YaRN-style`
is not an exact cited-YaRN reproduction; the run-specific MLA name is
`MLA wavelength-blend operator`.

## Historical prepared-round entrypoints

These are reference commands for the prepared protocol, not a standing request
for execution. Run from the repository root after resolving the applicable
protocol in INDEX.md.

| Need | Command / environment |
| --- | --- |
| Inspect the prepared schema without model access | `python3 scripts/experiments/matched_transfer_round.py template` |
| Prepared-round regression checks | `conda run --no-capture-output -n aidemo python -m pytest tests/test_matched_transfer_round.py -q` on the work machine; the import chain needs PyTorch |

The old `tests/test_repository_navigation.py` is absent from the slim tree;
its historical pass does not validate current routing. CPU preparation and
unit checks do not establish model results or GPU readiness.

## Prior routing annotations

The following text preserves the previous root INDEX's owner summaries,
corrections and scoped negatives. It is routing history, not a fresh result
validation or the current execution plan. In particular, it does not settle
new baseline-fidelity or scoring questions raised by the supplied cross-audit.
Read the exact linked owner and receipts before using a claim. Section numbers
inside these preserved annotations refer to their former index sections.

### Prior 1. Paper-level owners

| Question | Current answer | Owner |
| --- | --- | --- |
| Does allocation matter at fixed support during training? | Yes in the 151.9M three-seed protocol; retain support/seed scope | [`EXACT_RANGE_151M_3SEED_RESULT_20260820`](../../paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) + [JSON](../../paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.json) |
| Does the effect persist across exact-range configurations? | Yes in M4; matched non-Cosh shape remains competitive | `main_0726:rebuttal/rebuttal_0723/theory_results/M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md` (M4_EXACT_RANGE_FACTORIAL_RESULT_20260726) |
| What does full sin/cos geometry prove? | Redundancy/effective dimension and counterexamples; not LM ranking | [`FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819`](../../paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) |
| How are support and allocation separated? | `x_k = a + R z_k`; notation/intervention grammar, not a number owner | [`ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823`](../../paper-2027/research/foundations/ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md) |
| Can RoPE/attention structure alone determine optimal `z`, a frequency system, or mature-checkpoint movement? | **Boundary only:** distribution-free behavioural optimality is non-identifiable. This does not close repository-constrained selection from completed evidence. | [`ROPE_OPTIMALITY_IDENTIFIABILITY_AND_CONDITIONAL_EQUATIONS_20260903`](../../paper-2027/research/foundations/ROPE_OPTIMALITY_IDENTIFIABILITY_AND_CONDITIONAL_EQUATIONS_20260903.md) |
| What is the exact frozen transplant boundary? | Position-independent invertible Q/K compensation requires matching multisets up to sign/permutation | `main_0726:rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` (OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726) |
| What is the bounded EVQ-Cosh theorem? | Unique only for its stated convex surrogate | [`03_theory.tex`](../../paper-2027/sections/03_theory.tex) + [`a1_proofs.tex`](../../paper-2027/appendix/a1_proofs.tex) |
| What supports the matched-adaptation route? | Protocol-specific task-family length transfer at 1.485B and causal source use at 8B; not pure frozen-`z` or pretraining-scale evidence | `main_0726:rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md` (OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729), `main_0726:rebuttal/rebuttal_0723/theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md` (EVQ_8B_ADAPTATION_EVIDENCE_20260724) |
| What breadth supports the paper? | Protocol-specific 432M MLA, 750M continuation, existing 1.485B scale line, and Video-DiT | `main_0726:data/curated/table18_mla_3seed_aggregate.json` (table18 MLA), [`750M report`](../../docs/exp/2026-03/2026-03-06_phase15_750m_2k_to_4k_continue_results.md), `main_0726:rebuttal/rebuttal_0723/theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md` (OLMO2 1B), [`Video-DiT`](../../paper-2027/research/evidence/VIDEO_DIT_HEAD_TO_HEAD_SEED42_RESULT_20260826.md) |

### Prior 2. Mature-checkpoint owners

Open the full local catalogue only when needed:
[`results/README.md`](../../paper-2027/research/attention-aware-retrofit/results/README.md).

| Question | Current status | Owner |
| --- | --- | --- |
| What is the strongest practical no-update result? | Fully frozen derived allocation changes OLMo 16K RULER from `0.0056` to `0.6047`; coarse label-free allocation reaches `0.6104`; pure-`z` and broader deployment claims remain separate | [`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823`](../../paper-2027/research/attention-aware-retrofit/results/causal-mechanism/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) |
| Does it persist on fresh natural text? | Length-conditional NLL effect; not universal ranking | [`FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824`](../../paper-2027/research/attention-aware-retrofit/results/causal-mechanism/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md) |
| What is the strongest tracked static-table result? | One OLMo table passes tested 1x gates and improves longer endpoints; ordered permutation can collapse | [`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831`](../../paper-2027/research/attention-aware-retrofit/results/coupling-transfer/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md) |
| Under one static table and one path, which completed form is retained, and what same-table LoRA follows? | The sequential OLMo stop tree retains the full 64-slot legacy-u p2 mask installed as log-s4 with fixed `c=.074`; this is a capability-first engineering incumbent among the named historical candidates, not a global optimum. The later exact Q/K-LoRA screen improves PG-19 but not measured generated capability. | [`selection/specification`](../../paper-2027/research/attention-aware-retrofit/theory/SINGLE_STATIC_LOG_P2_SELECTION_AND_LORA_20260903.md), [`LoRA result`](../../paper-2027/research/attention-aware-retrofit/results/adaptation-coadaptation/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md) |
| What is supported across K32/K128? | Normalized pair index is the best-tested coordinate, not a law or K-causal result | [`K32 confirmation`](../../paper-2027/research/attention-aware-retrofit/results/coupling-transfer/K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md), [`K128 confirmation`](../../paper-2027/research/attention-aware-retrofit/results/coupling-transfer/K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md), [`full RULER-13`](../../paper-2027/research/attention-aware-retrofit/results/coupling-transfer/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md) |
| Does long signal convert to natural QA? | **Unresolved:** 9/2 raw owners missing; constructed 38-row assay invalid | [`ZERO_TRAINING_TWO_DAY...`](../../paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md) |
| Do headwise clocks solve the joint objective? | **Exploratory/report-only:** variable-length capped panel, adaptive row reuse, no tracked executed bundle | [`HEADWISE_FACTORIZED...`](../../paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md) |
| Does calibration-frozen attention-displacement Selective-31 beat layer-matched random/reverse masks? | **Negative at exact candidate/protocol scope:** target-long generation is floor-equal while short KL/Top-1 and answer-token NLL reverse the expected Selective advantage; matched global controls resolve the reused panel | [`HEAD_SELECTIVE_ZERO_TRAINING...`](../../paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/HEAD_SELECTIVE_ZERO_TRAINING_SIX_ARM_RESULT_20260903.md) |
| Can Native checkpoint structure uniquely determine an ordered movement profile? | **No without an added preference.** The declared squared Native-geometry surrogate uniquely constructs `m = Iso(1-u)` and its executed OLMo arm improves natural retention/likelihood, but fresh core-4 is materially worse at 4K/8K; it exposes an endpoint-dependent tradeoff, not a latent law or current-p2 replacement. | [`theory`](../../paper-2027/research/attention-aware-retrofit/theory/NATIVE_ONLY_MOVEMENT_PROFILE_IDENTIFIABILITY_20260903.md), [`result`](../../paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md) |
| Can one non-Native static table guarantee exact Native short behaviour and change long geometry? | **No under universal exact-preservation and standard stationary-RoPE assumptions.** The completed per-request Native/s4 policy is the existing behavioural escape. A prefix-preserving long-frame key handoff removes the old cross-boundary phase mismatch in CPU algebra, but has no model-quality evidence. | [`STATIC_NATIVE_NO_HARM_AND_PREFIX_HANDOFF_20260903`](../../paper-2027/research/attention-aware-retrofit/theory/STATIC_NATIVE_NO_HARM_AND_PREFIX_HANDOFF_20260903.md), [`session-policy result`](../../paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md) |
| Is the Selective-31 calibration score a universal functional sensitivity, and does joint Q/K--frequency relabeling invalidate the ordered-coupling results? | **No.** The score is exact endpoint attention-map displacement on a frozen calibration pack, not `chi_func`; exact joint relabeling is a gauge identity, while existing frequency-only permutations intentionally hold Q/K fixed. | [`LOCAL_FUNCTIONAL_COMPATIBILITY_AND_GAUGE_AUDIT_20260903`](../../paper-2027/research/attention-aware-retrofit/theory/LOCAL_FUNCTIONAL_COMPATIBILITY_AND_GAUGE_AUDIT_20260903.md) |
| Do finite scale-orbit boundary and Fourier-rank quantities predict mature-model behaviour? | **Negative selector result.** Exact boundary count changes `6 -> 64` under a behaviourally invisible ULP perturbation; p2 and the failed exact chain share zero Gram lower bound and saturated operator error but have opposite 4x utility. A CPU follow-up finds old `D*` plus phase safety diagnose these extreme failures, but the earlier one-turn-floor counterexample still falsifies `D*` as a general selector. | [`result`](../../paper-2027/research/attention-aware-retrofit/results/operator-analysis/SCALE_ORBIT_BOUNDARY_VALIDATION_RESULT_20260904.md), [`transport preflight`](../../paper-2027/research/attention-aware-retrofit/preflights/operator-analysis/SCALE_ORBIT_TRANSPORT_RESIDUAL_PREFLIGHT_20260904.md), [`prior axis falsification`](../../paper-2027/research/attention-aware-retrofit/analysis/RETROFIT_AXIS_FALSIFICATION_20260822.md) |
| What survives a proof, novelty, and tightness audit of the supplied finite scale-covariance derivation? | **The mathematics survives; the current empirical-tightness route does not.** Theorem 5 extends to continuous finite-dimensional real orthogonal RPE and has a dimension-free separation-order corollary. Exact obstruction/boundary leakage are prior art and novelty is not certified. A resolving synthetic control passes, but 45 bounded-condition trajectories select identity and saturate near error `2`; Ky-Fan is zero/tiny and non-ranking. Multilevel is stopped. | [`proof/novelty owner`](../../paper-2027/research/attention-aware-retrofit/theory/FINITE_SCALE_COVARIANCE_PROOF_NOVELTY_AND_TIGHTNESS_AUDIT_20260904.md), [`tightness result`](../../paper-2027/research/attention-aware-retrofit/results/operator-analysis/SCALE_CONJUGACY_TIGHTNESS_RESULT_20260904.md), [`preflight`](../../paper-2027/research/attention-aware-retrofit/preflights/operator-analysis/SCALE_CONJUGACY_TIGHTNESS_PREFLIGHT_20260904.md) |
| Has same-substrate log-p2 Q/K LoRA already been executed? | **Yes, at unit gain and at the retained `c=.074`.** Both improve paired PG-19 NLL; neither establishes generated-task capability improvement. | [`LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904`](../../paper-2027/research/attention-aware-retrofit/results/adaptation-coadaptation/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md) |
| Does exact log-p2 plus `c=.074` benefit from matched Q/K-only adaptation? | **Likelihood only in the measured panel.** PG-19 improves at 1x/4x, five-task macros are slightly negative/unresolved, and fresh core-4 changes `-.0100/+.0025/-.0225` at 4K/8K/16K. | [`LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904`](../../paper-2027/research/attention-aware-retrofit/results/adaptation-coadaptation/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md) |
| What is the corrected next experiment for one-table zero-training and low-step adaptation? | **Prospective, independent redesign.** Qualify compact/near/far lawful worlds and exact EOS; confirm fixed N/Z/G/Y; use all-linear r16 with original-Native functional constraints on qualified natural data. Earlier QK/source-contrast prototype is superseded. | [`CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904`](../../paper-2027/research/attention-aware-retrofit/preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md) |
| Where is the earlier single-table synthesis preserved? | **Historical research brief.** Mixed C2/p2 fitted ceiling and phase-cost lower-bound use are corrected; its prototype execution order is superseded by the independent protocol above. | [`SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904`](../../paper-2027/research/attention-aware-retrofit/analysis/SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904.md) |
| What can the 9/2 first-principles memo support? | Only explicitly retained identities under assumptions; T4/T5/T7 and behavioural generalizations are retracted/disputed | [`FIRST_PRINCIPLES...`](../../paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md) |

Endpoint boundaries remain strict: NLL/PPL, answer-token NLL, teacher-forced
gap, strict generation, token F1, exact match, RULER/NIAH, QA, causal source
use, adaptation, and transfer are different evidence tiers.

### Prior 3. Correction ledger

| Search hit | Current use | Replacement or reason |
| --- | --- | --- |
| 28 direct-hybrid zero-score receipts (2026-07-26) | **Invalid method evidence** | Native/EVQ buffer alias; use `main_0726:rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` (OLMO2_POSTHOC...) |
| p2 Native boundary near 3.91/4.01 from s2/s4 quadratic | **Superseded as same-path prediction** | s2 is compressed C2/G(x), s4 is full-p2; no matched tensor path. [Correction and replacement](../../paper-2027/research/attention-aware-retrofit/theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md) |
| phase-cost box/isotonic ceiling in the 9/4 brief | **Unresolved lower-bound step** | no certified feasible-set/domination argument or finite-region Native curvature; same replacement owner |
| old factor-frontier / paired-view LoRA runner and first new QK/source-margin prototype | **Superseded execution plans** | use fixed-witness diagnostic and Native-constrained engine; no prototype run was executed |
| old exact-range three-seed aggregate | **Superseded; never splice** | use raw-backed [`EXACT_RANGE...`](../../paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) |
| Qwen 128K `0.6175` | **Invalid aliased value** | corrected result is `0.5400` in [`SAME_SUPPORT...`](../../paper-2027/research/attention-aware-retrofit/results/causal-mechanism/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) |
| generated `FAILED_50M_GATE` | **Superseded: `SCREEN_UNRESOLVED`** | positive control failed; use the two owners in §4 |
| old Gemma 16K zero with 8K reference | **Superseded/confounded** | use [`REFERENCE_CORRECTED_K128_RESULT_20260901`](../../paper-2027/research/attention-aware-retrofit/results/coupling-transfer/REFERENCE_CORRECTED_K128_RESULT_20260901.md) |
| 38-row “16K Hotpot” Fact D | **Invalid for claims/gates** | selected constructed stress, non-official filler, raw missing |
| 9/2 gain-sweep session facts | **Unverified** | forensic lead only; no optimum, mechanism, or class conclusion |
| Hotpot-200 headwise comparison | **Report-only exploratory** | variable length capped at 16K, adaptive reuse, executed/raw bundle untracked |
| first-principles T4/T5/T7 and exact-conditioning claims | **Retracted/disputed** | invalid bound division, unrestricted torus claim, and novelty ratio |
| first-principles T1 arc-length proof | **Corrected; conclusion retained under its injective finite-arc assumptions** | old proof did not establish uniform per-slot scaling; use the tangent-ray proof in [`ROPE_OPTIMALITY...` §4.5](../../paper-2027/research/foundations/ROPE_OPTIMALITY_IDENTIFIABILITY_AND_CONDITIONAL_EQUATIONS_20260903.md#45-correction-to-the-historical-pi-arc-proof) |
| 9/3 optimality owner used as a terminal method verdict | **Corrected: supporting boundary only** | distribution-free impossibility does not answer the author-required single-static-table selection problem; that question remains active |
| two-day synthesis “only allowed” route | **Superseded candidate negative** | holdout failure; not a queue or method-class result |

An author-chosen threshold such as `0.875` is an operational tolerance, not a
theorem. Passing or missing it does not create a scientific discontinuity.

### Prior 4. Scoped negatives and unresolved questions

| Object | Exact status | Owner |
| --- | --- | --- |
| cosine-only collision / lower collision-logdet as behavioural rankers | universal sufficiency refuted; regularizer use remains open | [`FULL_ROPE...`](../../paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) |
| attention-Fisher `kappa_att` | tested ranker negative, not all attention-aware metrics | [`KAPPA_ATTENTION_MEASURE_AUDIT_20260820`](../../paper-2027/research/audits/KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md) |
| LeRoPE `w^(1/3)` oracle | published-shape operationalization negative, not the curvature class | [`LEROPE_PROFILE_ORACLE_AUDIT_20260820`](../../paper-2027/research/audits/LEROPE_PROFILE_ORACLE_AUDIT_20260820.md) |
| `D*`, coverage residual, phase-risk selectors | tested selector negatives in the registered panel | [`RETROFIT_AXIS_FALSIFICATION_20260822`](../../paper-2027/research/attention-aware-retrofit/analysis/RETROFIT_AXIS_FALSIFICATION_20260822.md) |
| direct-`z` two-document calibration | candidate/protocol negative | [`DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824`](../../paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md) |
| two analytic Native-support tables | two candidates failed; no inherent trade-off theorem | [`ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824`](../../paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md) |
| continuous-boundary-slope operator | only this implementation is closed | [`ZERO_TRAINING_MECHANISM_AND_CEILING_20260826`](../../paper-2027/research/attention-aware-retrofit/analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md) |
| exact Native no-harm from one non-Native static table | class-level obstruction under universal content/short-position equality; approximate retention and nonstandard/dynamic operators remain open | [`STATIC_NATIVE_NO_HARM_AND_PREFIX_HANDOFF_20260903`](../../paper-2027/research/attention-aware-retrofit/theory/STATIC_NATIVE_NO_HARM_AND_PREFIX_HANDOFF_20260903.md) |
| phase-isotropy / pair-volume / min-eigenvalue | `SCREEN_UNRESOLVED`, not negative | [`PHASE_ISOTROPY...`](../../paper-2027/research/attention-aware-retrofit/results/PHASE_ISOTROPY_50M_M4_RESULT_20260824.md), [`PHASE_ALLOCATION...`](../../paper-2027/research/attention-aware-retrofit/results/PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md) |
| Native-retention + natural long-QA/EOS assay | unresolved validity | §3 corrections; validity precedes method selection |
| Native-compatible/long-capable intervention | one arm passes the reused natural 1x double gate and retains long likelihood/RULER, but fresh core-4 is negative at 4K/8K; no universal jointly passing law | [`NATIVE_ISOTONIC_PROFILE_RESULT_20260903`](../../paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md) |

No content-blind static scalar score in this repository has prospectively
ranked LM behaviour across the required regimes. A numerical static search is
only best-found under its stated support/measure/optimizer/restarts and is not a
global or behavioural ceiling.
