# Retrofit preregistrations and revoked protocols

A preflight records what was frozen before execution. It is never evidence that
an experiment ran.

| Preflight | Final state |
| --- | --- |
| [`PROTECTED_PROGRESSIVE_SHIFT_PREFLIGHT_20260827.md`](PROTECTED_PROGRESSIVE_SHIFT_PREFLIGHT_20260827.md) | frozen protocol design; separate new protected-simple-curve test; not executed; requires separate explicit compute authorization; the protected-ramp scanning direction was superseded as the active investigation direction on 2026-08-28 by the band-attribution design in `../analysis/PROTECTED_RAMP_RIGOROUS_COMPOSITE_20260828.md` §9 |
| [`MATCHED_CONTENT_PHASE_2X2_BRIDGE_PREFLIGHT_20260827.md`](MATCHED_CONTENT_PHASE_2X2_BRIDGE_PREFLIGHT_20260827.md) | frozen protocol design; not executed; requires separate explicit compute authorization; remains the next decision-valuable protocol per `../../../../INDEX.md` §6.2 |
| [`ALLOCATION_DOSE_RESPONSE_PREFLIGHT_20260826.md`](ALLOCATION_DOSE_RESPONSE_PREFLIGHT_20260826.md) | executed; registered primary construction failed the joint in-window/long-tail gate; result owned by `../results/ALLOCATION_DOSE_RESPONSE_RESULT_20260826.md` |
| [`COADAPTIVE_ALLOCATION_ORACLE_PREFLIGHT_20260825.md`](COADAPTIVE_ALLOCATION_ORACLE_PREFLIGHT_20260825.md) | executed; registered shell gate failed; attribution and matched recovery are owned by `../results/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md` |
| [`PHASE_ALLOCATION_M4_EXTENDED_PREFLIGHT_20260824.md`](PHASE_ALLOCATION_M4_EXTENDED_PREFLIGHT_20260824.md) | executed; its own decision rule ("label the regime as unresolved rather than attributing the result to the new score" when the anchored-Cosh control is neutral/negative at 50M base-256) is applied by the canonical report. See `../../../../INDEX.md` §6.2 |
| [`ZERO_PARAMETER_PROTECTED_BAND_PREFLIGHT_20260824.md`](ZERO_PARAMETER_PROTECTED_BAND_PREFLIGHT_20260824.md) | executed; failed 1x PG-19 no-harm gate and stopped before capability evaluation |
| [`ZERO_PARAMETER_SINGLE_TABLE_PREFLIGHT_20260824.md`](ZERO_PARAMETER_SINGLE_TABLE_PREFLIGHT_20260824.md) | executed; both static analytic tables stopped by `../results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md` |
| [`DIRECT_Z_ZERO_WEIGHT_PREFLIGHT_20260824.md`](DIRECT_Z_ZERO_WEIGHT_PREFLIGHT_20260824.md) | executed; stopped by `../results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md` before downstream evaluation |
| [`SAME_SUPPORT_CONTROL_PREFLIGHT_20260823.md`](SAME_SUPPORT_CONTROL_PREFLIGHT_20260823.md) | executed; superseded by `../results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md` |
| [`SMALL_MODEL_RETROFIT_CAUSAL_PREFLIGHT_20260823.md`](SMALL_MODEL_RETROFIT_CAUSAL_PREFLIGHT_20260823.md) | executed; superseded by the 151.9M section of the same owner |
| [`FAR_PASS_CHORD_RESIDUAL_PREFLIGHT_20260821.md`](FAR_PASS_CHORD_RESIDUAL_PREFLIGHT_20260821.md) | executed; route stopped by `../results/FAR_PASS_CHORD_EXPERIMENT_REPORT_20260821.md` |
| [`FUNCTION_MORPH_PREFLIGHT_20260821.md`](FUNCTION_MORPH_PREFLIGHT_20260821.md) | prepared but deprioritized; no result |
| [`PHASE_CHORD_LORA_RETROFIT_PREFLIGHT_20260822.md`](PHASE_CHORD_LORA_RETROFIT_PREFLIGHT_20260822.md) | revoked before GPU because the target was not identifiable from model input |
| [`PHASE_ADAROPE_LORA_PREFLIGHT_20260822.md`](PHASE_ADAROPE_LORA_PREFLIGHT_20260822.md) | prepared no-GPU protocol; not current queue and no result |

Do not launch anything from this directory without a new explicit user
authorization and a live owner stating why it can change the paper.
