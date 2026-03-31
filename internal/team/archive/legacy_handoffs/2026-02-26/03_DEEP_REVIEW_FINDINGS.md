# 03 Deep Review Findings (Code + Protocol)

This review focuses on regressions, protocol compliance, and operability.

## Findings (ordered by severity)

1. [P1] Adapter detection false-positive risk in Plan B eval
- File: `scripts/plan_b_eval_longbench.py`
- Issue: directory could be treated as adapter-valid with config only but missing weight file.
- Fix: adapter resolution now requires both config and weight (`adapter_model.safetensors|bin`).
- Status: RESOLVED.

2. [P1] `custom_inv_freq.pt` discovery path gap for checkpoint/final_lora layout
- File: `scripts/plan_b_eval_longbench.py`
- Issue: anchored inv tensor under run root could be missed when adapter path is nested.
- Fix: added parent and grand-parent artifact lookup.
- Status: RESOLVED.

3. [P2] Ambiguous adapter resolved path in training summary when adapter missing
- File: `scripts/train_cross_model_lora_fast_tuned.py`
- Issue: `adapter_resolved_path` could point to run dir even when no adapter exists.
- Fix: now empty string when layout is `none`.
- Status: RESOLVED.

4. [P2] Registry fallback path for non-ready rows
- File: `scripts/build_model_registry.py`
- Issue: non-ready row defaulted to non-existent `final_lora` path.
- Fix: fallback now points to run root for clearer diagnostics.
- Status: RESOLVED.

## Residual risks

- Full runtime behavior of eval scripts still depends on server environment packages and GPU runtime.
- Statistical validity still depends on future run completion (this round lands interface/contracts only).

## Review verdict

- Code-level blockers: NONE.
- Protocol/schema blockers: NONE.
- Ready for server-side execution under existing gates.
