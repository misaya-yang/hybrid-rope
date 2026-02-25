# 02 Validation Snapshot (2026-02-26)

## Executed checks

1. Syntax checks
- `python3 -m py_compile` on all changed scripts.
- Result: PASS.

2. Plan B dry-run wiring
- Ran `scripts/plan_b_eval_longbench.py --dry_run` with synthetic model/run dirs.
- Verified:
  - custom task selection wiring (`--tasks`)
  - stress knobs wiring
  - `manifest_json` propagation to NIAH/Passkey
  - `final_lora` adapter discovery
- Result: PASS.

3. Registry behavior
- Ran `scripts/build_model_registry.py` on synthetic run root.
- Verified `ready` row includes new adapter layout fields.
- Result: PASS.

4. Data mixer smoke test
- Ran `scripts/prepare_long_instruction_mix.py` on toy JSONL.
- Verified outputs:
  - `train/valid/test` text files
  - `mix_manifest.json` with required source/audit fields
- Result: PASS.

## Environment limitations during local validation

- Local environment lacks runtime deps for full eval execution (`numpy`, `matplotlib`).
- Therefore full LongBench/NIAH/Passkey runtime execution was not performed locally.
- Mitigation: syntax + dry-run + schema checks completed; full execution deferred to server environment.
