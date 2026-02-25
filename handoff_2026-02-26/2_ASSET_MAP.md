# Asset Map (2026-02-26)

## Code files changed in this round

- `scripts/train_cross_model_lora_fast_tuned.py`
- `scripts/build_model_registry.py`
- `scripts/plan_b_eval_longbench.py`
- `scripts/eval_longbench.py`
- `scripts/eval_niah_recall.py`
- `scripts/eval_passkey_teacher_forcing.py`
- `scripts/prepare_long_instruction_mix.py` (new)

## Output contracts

Training run dir:

```text
<run_dir>/
  adapter_config.json                  (compat)
  adapter_model.safetensors|bin        (compat)
  final_lora/
    adapter_config.json                (canonical)
    adapter_model.safetensors|bin
  artifacts/
    summary.json
    custom_inv_freq.pt
```

Plan B eval run dir:

```text
<output_root>/<method>_seed<seed>/
  longbench_*.json
  longbench_manifest.json
  niah/niah_recall_results.json
  passkey/passkey_tf_summary.json
  repro_manifest/
```

## Data mixer outputs

`prepare_long_instruction_mix.py` outputs:
- `train.txt`
- `valid.txt`
- `test.txt`
- `mix_manifest.json`
