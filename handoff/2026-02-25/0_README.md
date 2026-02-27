# Handoff 2026-02-25 (Qwen Evidence Chain)

This is the **current entrypoint** for the NeurIPS reviewer-remediation experiment rebuild.

Primary goal:
- Build a **protocol-clean, auditable** evidence chain on **Qwen2.5-7B-Instruct** (not a SOTA race).

If you only run three commands, run these (server paths shown; adapt locally if needed):

1) Build model/LoRA registry (so every AI can locate artifacts reliably)

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
/root/miniconda3/bin/python scripts/build_model_registry.py \
  --repo_root . \
  --model_cache_dir /root/autodl-tmp/dfrope/ms_models \
  --run_roots artifacts/cross_model_fast_tuned_b1_gc
```

2) Baseline gold (Qwen, full lb21, per-sample traces, strict parity)

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
/root/miniconda3/bin/python scripts/eval_longbench.py \
  --base_model_path /root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct \
  --task_set lb21 \
  --max_samples_per_task 0 \
  --max_input_tokens 16384 \
  --batch_size 2 \
  --prompt_source official \
  --chat_template auto \
  --truncate_mode middle \
  --max_new_tokens_policy official \
  --score_scale pct \
  --strict_parity_check \
  --save_per_sample_traces 1 \
  --trace_output_max_chars 1024 \
  --repro_manifest_dir artifacts/repro_manifest/baseline_gold_qwen_seed42 \
  --manifest_json artifacts/manifests/longbench_manifest_qwen_ctx16384_seed42.json \
  --seed 42 \
  --output_json artifacts/reviewer_2026-02-25/qwen_baseline_gold_seed42/longbench_lb21.json
```

3) Anchored tuned (LoRA or inference-only) and statistics

See:
- `handoff/2026-02-25/3_RUNBOOK.md` (exact command matrix)

Stop conditions (hard):
- Any comparison without identical manifest/template/decode/injection metadata is `INVALID`.
- Any run missing `inv_freq` hash or repro manifest is `PENDING` (no claims).
