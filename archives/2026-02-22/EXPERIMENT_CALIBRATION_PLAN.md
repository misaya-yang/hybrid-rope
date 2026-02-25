# Llama-3-8B Fair Suite: Multi-Calibration Plan

Goal: avoid "good PPL but broken retrieval" false positives.

## Gate 0: Environment Freeze

1. Record:
- `transformers` version
- `torch` version
- GPU model and driver
2. Confirm local-only loading path for model/tokenizer/data.
3. Keep one fixed seed for all method comparisons.

## Gate 1: Injection Correctness

Run `--calibration_only` first for each method.

Checks:
1. `model.config.rope_scaling` is `None` before and after injection.
2. Rotary modules found and patched count > 0.
3. New `inv_freq` shape/dtype exactly match original.
4. `anchored_hybrid` rigid core (`j0=12`) is bitwise equal to baseline in every patched layer.
5. For `baseline`, reinjection logit drift <= tolerance (default `1e-4`).

Artifacts:
- `artifacts/calibration_report.json`

## Gate 2: Data/Template Integrity

Checks:
1. `tokenizer.apply_chat_template(..., tokenize=True)` returns valid non-empty ids.
2. Sample decode sanity: user/assistant headers exist in template path.
3. Labels mask only assistant tail region (`-100` elsewhere).

## Gate 3: Short Smoke Training

Run 40-80 steps first:
- `--max_steps 80`
- Keep all other hyperparameters unchanged.

Pass criteria:
1. Loss finite and decreasing trend exists.
2. No NaN/Inf.
3. No sudden collapse in training logs.

## Gate 4: Full Fair Run

Run full settings:
- steps 400
- batch 2
- grad accum 2
- lr 2e-4
- bf16
- LoRA rank 64 / alpha 128

Methods to compare:
1. `baseline`
2. `pi`
3. `yarn`
4. `anchored_hybrid`

## Gate 5: Post-Train Cross-Validation

Do not trust PPL only.

Must report together:
1. PPL at 16K / 32K
2. NIAH (single + multi needle)
3. LongBench subset (qasper, hotpotqa, gov_report)
4. Teacher-forcing passkey sanity (true vs false key)

Only accept method-level claims when all 4 are directionally consistent.
