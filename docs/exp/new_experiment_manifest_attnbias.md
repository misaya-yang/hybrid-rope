# New Experiment Manifest (Attention-Integrated, Isolated)

- Generated: `2026-02-26`
- Base model lock: `meta-llama/Meta-Llama-3-8B-Instruct` (8K-native only)
- Forbidden: `Llama-3.1`, any `128K` model
- RoPE lock: `inv_freq.copy_()` injection only
- Theta lock: `500000`

## SHA256 (new files)

- See `artifacts/reviewer_2026-02-26/attn_hashes_2026-02-26.txt` for exact SHA256 after final sync.

## Commands (do not run during sacred active job)

### 0) Queue launcher (recommended)

```bash
bash scripts/isolated/attn/next_attn_lora_queue.sh
```

Notes:
- auto-waits sacred Qwen processes to exit.
- auto-falls back to `base` conda env if `aidemo` is unavailable.

### 1) Read-only audit

```bash
python scripts/isolated/attn/attn_audit_readonly.py \
  --remote_host connect.bjb1.seetacloud.com \
  --remote_port 52592 \
  --remote_user root \
  --remote_password '***' \
  --report_path docs/exp/reports/attn_audit_report_2026-02-26.md
```

### 2) Train (seed 42)

```bash
~/miniconda3/bin/conda run -n aidemo python scripts/isolated/attn/new_lora_longalpaca_attnbias_train.py \
  --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --run_name attnbias_seed42 \
  --seed 42 \
  --attn_bias_mode bias \
  --gamma_mode constant \
  --gamma 1e-4 \
  --max_steps 800 \
  --lora_rank 32
```

### 3) Train (seed 1337)

```bash
~/miniconda3/bin/conda run -n aidemo python scripts/isolated/attn/new_lora_longalpaca_attnbias_train.py \
  --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --run_name attnbias_seed1337 \
  --seed 1337 \
  --attn_bias_mode bias \
  --gamma_mode constant \
  --gamma 1e-4 \
  --max_steps 800 \
  --lora_rank 32
```

### 4) Evaluate (seed 42)

```bash
~/miniconda3/bin/conda run -n aidemo python scripts/isolated/attn/new_eval_longbench_attnbias.py \
  --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --adapter_path artifacts/new_attnbias_v1/train/attnbias_seed42/adapter \
  --custom_inv_freq_path artifacts/new_attnbias_v1/train/attnbias_seed42/artifacts/custom_inv_freq.pt \
  --output_root artifacts/new_attnbias_v1/eval/seed42 \
  --seed 42 \
  --attn_bias_mode bias \
  --gamma_mode constant \
  --gamma 1e-4
```

### 5) Evaluate (seed 1337)

```bash
~/miniconda3/bin/conda run -n aidemo python scripts/isolated/attn/new_eval_longbench_attnbias.py \
  --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --adapter_path artifacts/new_attnbias_v1/train/attnbias_seed1337/adapter \
  --custom_inv_freq_path artifacts/new_attnbias_v1/train/attnbias_seed1337/artifacts/custom_inv_freq.pt \
  --output_root artifacts/new_attnbias_v1/eval/seed1337 \
  --seed 1337 \
  --attn_bias_mode bias \
  --gamma_mode constant \
  --gamma 1e-4
```

## Expected outputs

- `artifacts/new_attnbias_v1/train/<run_name>/run_config.json`
- `artifacts/new_attnbias_v1/train/<run_name>/adapter/`
- `artifacts/new_attnbias_v1/eval/<seed>/results.json`
- `artifacts/new_attnbias_v1/eval/<seed>/longbench_lb21.json`
- `artifacts/new_attnbias_v1/eval/<seed>/length_buckets.json`
- `artifacts/new_attnbias_v1/eval/<seed>/niah/`
- `artifacts/new_attnbias_v1/eval/<seed>/passkey/`
