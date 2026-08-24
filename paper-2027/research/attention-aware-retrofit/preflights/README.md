# Retrofit preregistrations and revoked protocols

A preflight records what was frozen before execution. It is never evidence that
an experiment ran.

| Preflight | Final state |
| --- | --- |
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
