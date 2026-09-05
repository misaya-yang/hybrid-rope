# Retrofit preregistrations and revoked protocols

A preflight records what was frozen before execution. It is never evidence that
an experiment ran.

## Directory layout

| Directory | Protocol family | Files |
| --- | --- | ---: |
| [`causal-mechanism/`](causal-mechanism/) | Causal controls and matched-content mechanism tests | 4 |
| [`zero-training-deployment/`](zero-training-deployment/) | Static-table, routing, and zero-weight protocols | 7 |
| [`coupling-transfer/`](coupling-transfer/) | Finite-K, reference, cross-model, and transport protocols | 10 |
| [`adaptation-coadaptation/`](adaptation-coadaptation/) | LoRA, residual retrofit, and co-adaptation protocols | 7 |
| [`operator-analysis/`](operator-analysis/) | Scale-covariance, orbit, and conjugacy validation | 4 |

The current [constrained frontier and adaptation protocol](CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md)
stays at its stable top-level path.

[`NATIVE_ISOTONIC_PROFILE_PREFLIGHT_20260903.md`](zero-training-deployment/NATIVE_ISOTONIC_PROFILE_PREFLIGHT_20260903.md)
freezes the exact-u/pinned-Iso composite, the legacy/exact-u by p2/Iso
attribution table, inherited-gain controls, numerical entrance, formal 1x
replay, cheap long diagnostic, conditional breadth, monitoring, evidence, and
shutdown contract. It authorizes no parameter rescue or post-outcome sweep.

## September 1–2 protocol status (historical)

Section-number references inside individual preflights record the `INDEX.md`
layout that existed when they were frozen. Current evidence, closed routes, and
agenda live in repository `INDEX.md` §§2, 3, and 5. Nothing in this directory is
a current queue.

[`COUPLING_NEGATIVE_ATTRIBUTION_PREFLIGHT_20260901.md`](coupling-transfer/COUPLING_NEGATIVE_ATTRIBUTION_PREFLIGHT_20260901.md)
registers Native-only P0 reference-length calibration and independent
confirmation. It adds no RoPE candidate or curve parameter. The conditional
[`reference-correct K128 protocol`](coupling-transfer/REFERENCE_CORRECTED_K128_PREFLIGHT_20260901.md)
has completed s2 and s4. The
[`same-generation Qwen s2 protocol`](coupling-transfer/QWEN_S2_SAME_FAMILY_IDENTIFICATION_PREFLIGHT_20260901.md)
now admits only its fixed baseline-completion panel. Mechanism, selector and
broad-baseline stages are still conditional. Current priority remains owned
by `INDEX.md`.

[`K32_PAIRED_CROSSING_CONFIRMATION_PREFLIGHT_20260901.md`](coupling-transfer/K32_PAIRED_CROSSING_CONFIRMATION_PREFLIGHT_20260901.md)
registered a separate fresh-seed, 80-row/task confirmation of the unchanged
Native/physical/index profiles. It completed with an unresolved coordinate
ordering; the result is owned by
`../results/coupling-transfer/K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md`.

[`NATIVE_QK_FINITE_KL_PREFLIGHT_20260901.md`](coupling-transfer/NATIVE_QK_FINITE_KL_PREFLIGHT_20260901.md)
froze a conditional P3 same-QK diagnostic, including all 256 NLL-aligned query
positions and aggregation. Its `CONFIRMED_CROSSING` entrance failed, so no
model execution or mechanism result exists.

[`K32_FRESH_YARN_MATCHED_BASELINE_PREFLIGHT_20260901.md`](coupling-transfer/K32_FRESH_YARN_MATCHED_BASELINE_PREFLIGHT_20260901.md)
registered one official-equation YaRN-s2 completion arm on the same fresh K32
rows. It completed with index favored at 64K; this remains a matched baseline
completion, not a new method holdout. The result shares the K32 confirmation
owner.

[`K128_COORDINATE_RANKING_CONFIRMATION_PREFLIGHT_20260901.md`](coupling-transfer/K128_COORDINATE_RANKING_CONFIRMATION_PREFLIGHT_20260901.md)
froze a new-seed, single-16K physical/index K128 confirmation. It completed
with index favored; the result has its own canonical owner and does not add a
length, baseline, or profile parameter.

[`K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_PREFLIGHT_20260901.md`](coupling-transfer/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_PREFLIGHT_20260901.md)
froze the new-seed full-RULER confirmation for the selected engineering
representative. It completed as `CLEAR_ADVANCE`; the result has its own owner.

[`K32_PACKED_NATURAL_NLL_CONFIRMATION_PREFLIGHT_20260901.md`](coupling-transfer/K32_PACKED_NATURAL_NLL_CONFIRMATION_PREFLIGHT_20260901.md)
froze the packed-natural likelihood gate. The prior session completed the
model evaluation and summarized it in
[`../results/zero-training-deployment/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`](../results/zero-training-deployment/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md),
but its remote raw result files were not recovered; the statistics therefore
remain internal.

## Retired recent protocols

| Preflight | Lifecycle state |
| --- | --- |
| [`ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830.md`](zero-training-deployment/ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830.md) | retired after execution; preserves the old W0/F1 success-first registration only, with no current action |
| [`MATCHED_CONTENT_PHASE_2X2_BRIDGE_PREFLIGHT_20260827.md`](causal-mechanism/MATCHED_CONTENT_PHASE_2X2_BRIDGE_PREFLIGHT_20260827.md) | retired conditional diagnostic; not part of the current deterministic zero-training design |

The leave-one-band-out design in
[`../analysis/PROTECTED_RAMP_RIGOROUS_COMPOSITE_20260828.md`](../analysis/PROTECTED_RAMP_RIGOROUS_COMPOSITE_20260828.md)
§9 is invalid as written: several abrupt band restorations make the frequency
table non-monotone, B0 changes support, and B4 is not an exact sham. It has no
preflight or execution authorization and cannot be revived without a new
monotonicity-checked construction.

## Prepared current protocol

| Preflight | Current state |
| --- | --- |
| [`CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md`](CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md) | factor-frontier code and a LoRA prototype are prepared; the scientific LoRA gate still needs route-explicit scoring and a QK-first/parameter-matched-QKVO screen; 8x/16x/32x remain behind the evaluation firewall; the author shut the work machine down and no new run started |

## Executed protocols

| Preflight | Final state |
| --- | --- |
| [`SCALE_CONJUGACY_TIGHTNESS_PREFLIGHT_20260904.md`](operator-analysis/SCALE_CONJUGACY_TIGHTNESS_PREFLIGHT_20260904.md) | executed; synthetic nontrivial positive control passes, but all 45 primary trajectories select identity and sampled error saturates near `2`; multilevel stopped; result owned by `../results/operator-analysis/SCALE_CONJUGACY_TIGHTNESS_RESULT_20260904.md` |
| [`FINITE_SCALE_COVARIANCE_COMPLETE_VALIDATION_PREFLIGHT_20260904.md`](operator-analysis/FINITE_SCALE_COVARIANCE_COMPLETE_VALIDATION_PREFLIGHT_20260904.md) | supplied Pro text is Sections 5--12 only; 21 theorem checks plus best-`D_j` optimizer prepared; local 102/102 and work-machine 24/24 suites plus both five-table no-card preflights pass; proofs/general extension are audited, while operator tightness, novelty certification, paper judgment, and Sections 1--4 remain unclaimed |
| [`SCALE_ORBIT_BOUNDARY_VALIDATION_PREFLIGHT_20260903.md`](operator-analysis/SCALE_ORBIT_BOUNDARY_VALIDATION_PREFLIGHT_20260903.md) | executed; exact-count/Gram/operator quantities fail as selectors, result owned by `../results/operator-analysis/SCALE_ORBIT_BOUNDARY_VALIDATION_RESULT_20260904.md` |
| [`SCALE_ORBIT_TRANSPORT_RESIDUAL_PREFLIGHT_20260904.md`](operator-analysis/SCALE_ORBIT_TRANSPORT_RESIDUAL_PREFLIGHT_20260904.md) | executed CPU-only; diagnoses today's extreme failures but does not revive `D*` as a general selector; same result owner |
| [`LOG_P2_UNIT_GAIN_QK_LORA_REPLAY_PREFLIGHT_20260903.md`](adaptation-coadaptation/LOG_P2_UNIT_GAIN_QK_LORA_REPLAY_PREFLIGHT_20260903.md) | executed; unit-gain Q/K adapter improves PG-19 but generated tasks are heterogeneous, result owned by `../results/adaptation-coadaptation/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md` |
| [`LOG_P2_C074_QK_LORA_PREFLIGHT_20260904.md`](adaptation-coadaptation/LOG_P2_C074_QK_LORA_PREFLIGHT_20260904.md) | executed; exact retained-gain adapter improves PG-19 but not natural-generation or fresh core-4 macros; same result owner |
| [`ALLOCATION_DOSE_RESPONSE_PREFLIGHT_20260826.md`](adaptation-coadaptation/ALLOCATION_DOSE_RESPONSE_PREFLIGHT_20260826.md) | executed; registered primary construction failed the joint in-window/long-tail gate; result owned by `../results/adaptation-coadaptation/ALLOCATION_DOSE_RESPONSE_RESULT_20260826.md` |
| [`COADAPTIVE_ALLOCATION_ORACLE_PREFLIGHT_20260825.md`](adaptation-coadaptation/COADAPTIVE_ALLOCATION_ORACLE_PREFLIGHT_20260825.md) | executed; registered shell gate failed; attribution and matched recovery are owned by `../results/adaptation-coadaptation/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md` |
| [`PHASE_ALLOCATION_M4_EXTENDED_PREFLIGHT_20260824.md`](zero-training-deployment/PHASE_ALLOCATION_M4_EXTENDED_PREFLIGHT_20260824.md) | executed; its own decision rule ("label the regime as unresolved rather than attributing the result to the new score" when the anchored-Cosh control is neutral/negative at 50M base-256) is applied by the canonical report. See `../../../../INDEX.md` §3.2 |
| [`ZERO_PARAMETER_PROTECTED_BAND_PREFLIGHT_20260824.md`](zero-training-deployment/ZERO_PARAMETER_PROTECTED_BAND_PREFLIGHT_20260824.md) | executed; failed 1x PG-19 no-harm gate and stopped before capability evaluation |
| [`ZERO_PARAMETER_SINGLE_TABLE_PREFLIGHT_20260824.md`](zero-training-deployment/ZERO_PARAMETER_SINGLE_TABLE_PREFLIGHT_20260824.md) | executed; both static analytic tables stopped by `../results/zero-training-deployment/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md` |
| [`DIRECT_Z_ZERO_WEIGHT_PREFLIGHT_20260824.md`](zero-training-deployment/DIRECT_Z_ZERO_WEIGHT_PREFLIGHT_20260824.md) | executed; stopped by `../results/zero-training-deployment/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md` before downstream evaluation |
| [`SAME_SUPPORT_CONTROL_PREFLIGHT_20260823.md`](causal-mechanism/SAME_SUPPORT_CONTROL_PREFLIGHT_20260823.md) | executed; superseded by `../results/causal-mechanism/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md` |
| [`SMALL_MODEL_RETROFIT_CAUSAL_PREFLIGHT_20260823.md`](causal-mechanism/SMALL_MODEL_RETROFIT_CAUSAL_PREFLIGHT_20260823.md) | executed; superseded by the 151.9M section of the same owner |
| [`FAR_PASS_CHORD_RESIDUAL_PREFLIGHT_20260821.md`](adaptation-coadaptation/FAR_PASS_CHORD_RESIDUAL_PREFLIGHT_20260821.md) | executed; route stopped by `../results/adaptation-coadaptation/FAR_PASS_CHORD_EXPERIMENT_REPORT_20260821.md` |

## Superseded, revoked, or deprioritized designs

| Preflight | Final state |
| --- | --- |
| [`PROTECTED_PROGRESSIVE_SHIFT_PREFLIGHT_20260827.md`](zero-training-deployment/PROTECTED_PROGRESSIVE_SHIFT_PREFLIGHT_20260827.md) | historical, not executed, and superseded as an active route; protected-ramp formula scanning was retired, and the later abrupt band-restoration design is invalid rather than conditionally runnable |
| [`FUNCTION_MORPH_PREFLIGHT_20260821.md`](causal-mechanism/FUNCTION_MORPH_PREFLIGHT_20260821.md) | old 18-target audit remains deprioritized and unexecuted; its former F1/F2 reuse path is retired and creates no execution queue |
| [`PHASE_CHORD_LORA_RETROFIT_PREFLIGHT_20260822.md`](adaptation-coadaptation/PHASE_CHORD_LORA_RETROFIT_PREFLIGHT_20260822.md) | revoked before GPU because the target was not identifiable from model input |
| [`PHASE_ADAROPE_LORA_PREFLIGHT_20260822.md`](adaptation-coadaptation/PHASE_ADAROPE_LORA_PREFLIGHT_20260822.md) | prepared no-GPU protocol; not current queue and no result |

Do not launch anything from this directory without a new explicit user
authorization and a live owner. Every experiment must state its falsifiable
hypothesis, the decision it can change, why the answer is not already known,
and its stop rule. Entering a result into the manuscript is a later, separate
evidence decision.
