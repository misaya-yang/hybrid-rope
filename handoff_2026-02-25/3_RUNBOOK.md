# Runbook (Qwen Evidence Chain)

Last updated: 2026-02-25

This runbook is written so another AI/engineer can switch machines and continue experiments without ambiguity.

## 0) Principles (non-negotiable)

- Fix pipeline first, then scale experiments.
- Never compare across mismatched manifest/template/decode/injection settings.
- Always save per-sample traces and reproducibility manifest.
- If a condition loses, keep it (no cherry-pick).

## 1) Environment

Server python:
- `/root/miniconda3/bin/python`

Offline flags (recommended):
```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

## 2) Gate A: LongBench pipeline parity (must be green)

Run a parity audit (small subset) before full lb21:
```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
/root/miniconda3/bin/python scripts/import_2024/longbench_pipeline_audit.py \
  --out_dir artifacts/reviewer_2026-02-25/longbench_parity_audit
```

Expected: errors are within the parity threshold in `handoff_2026-02-23/longbench_pipeline_parity.md`.

## 3) Gate B: Baseline Gold (Qwen, full lb21)

Throughput tip (96GB GPU):
- Start with `--batch_size 2`. If stable, try `--batch_size 4`. If OOM, fall back to `1`.

Seed 42:
```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
mkdir -p artifacts/reviewer_2026-02-25/qwen_baseline_gold_seed42
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

Seed 1337:
- same command but replace `seed`, manifest, and output paths.

## 4) Gate C: Anchored tuned (LoRA) vs baseline

### 4.1 Training (400 steps)

Train baseline and anchored (seed 42):
```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
BASE=/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct
OUT=artifacts/cross_model_fast_tuned_b1_gc
DATA=/root/autodl-tmp/wikitext_data

/root/miniconda3/bin/python scripts/train_cross_model_lora_fast_tuned.py \
  --method baseline --base_model_path "$BASE" --output_dir "$OUT/qwen2_5_7b_instruct_baseline_42" \
  --run_name qwen2_5_7b_instruct_baseline_42 --data_dir "$DATA" --seed 42 \
  --max_seq_len 16384 --max_steps 400 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 --warmup_steps 50 --save_steps 200 --logging_steps 10 \
  --lora_rank 16 --lora_alpha 32 --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --bf16 --optim paged_adamw_8bit --load_in_4bit --gradient_checkpointing --local_files_only --trust_remote_code

/root/miniconda3/bin/python scripts/train_cross_model_lora_fast_tuned.py \
  --method anchored_sigmoid --base_model_path "$BASE" --output_dir "$OUT/qwen2_5_7b_instruct_anchored_sigmoid_42" \
  --run_name qwen2_5_7b_instruct_anchored_sigmoid_42 --data_dir "$DATA" --seed 42 \
  --max_seq_len 16384 --max_steps 400 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 --warmup_steps 50 --save_steps 200 --logging_steps 10 \
  --lora_rank 16 --lora_alpha 32 --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --bf16 --optim paged_adamw_8bit --load_in_4bit --gradient_checkpointing --local_files_only --trust_remote_code \
  --anchor_factor 4 --slope_raw 20 --center_ratio 0.70
```

Repeat for seed 1337 by changing output dirs and `--seed`.

### 4.2 Evaluation (full lb21, paired manifest, per-sample traces)

Example (anchored seed 42):
```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
/root/miniconda3/bin/python scripts/eval_longbench.py \
  --base_model_path /root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct \
  --adapter_path artifacts/cross_model_fast_tuned_b1_gc/qwen2_5_7b_instruct_anchored_sigmoid_42/final_lora \
  --custom_inv_freq_path artifacts/cross_model_fast_tuned_b1_gc/qwen2_5_7b_instruct_anchored_sigmoid_42/artifacts/custom_inv_freq.pt \
  --task_set lb21 --max_samples_per_task 0 --max_input_tokens 16384 \
  --batch_size 2 \
  --prompt_source official --chat_template auto --truncate_mode middle --max_new_tokens_policy official \
  --score_scale pct --strict_parity_check \
  --save_per_sample_traces 1 --trace_output_max_chars 1024 \
  --manifest_json artifacts/manifests/longbench_manifest_qwen_ctx16384_seed42.json \
  --repro_manifest_dir artifacts/repro_manifest/qwen_fast400_anchored_seed42 \
  --seed 42 \
  --output_json artifacts/reviewer_2026-02-25/qwen_fast400_seed42/anchored_lb21.json
```

## 5) Modern anchor (NTK-aware dynamic)

If LongRoPE is not ready, run `ntk_dynamic` first:
```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
/root/miniconda3/bin/python scripts/eval_longbench.py \
  --base_model_path /root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct \
  --variant ntk_dynamic \
  --task_set lb21 --max_samples_per_task 0 --max_input_tokens 16384 \
  --batch_size 2 \
  --prompt_source official --chat_template auto --truncate_mode middle --max_new_tokens_policy official \
  --score_scale pct --strict_parity_check \
  --save_per_sample_traces 1 --trace_output_max_chars 1024 \
  --manifest_json artifacts/manifests/longbench_manifest_qwen_ctx16384_seed42.json \
  --repro_manifest_dir artifacts/repro_manifest/qwen_ntk_seed42 \
  --seed 42 \
  --output_json artifacts/reviewer_2026-02-25/qwen_ntk_seed42/longbench_lb21.json
```

## 6) Statistics (per-sample + FDR)

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
TASKS=\"narrativeqa,qasper,multifieldqa_en,multifieldqa_zh,hotpotqa,2wikimqa,musique,dureader,gov_report,qmsum,multi_news,vcsum,trec,triviaqa,samsum,lsht,passage_count,passage_retrieval_en,passage_retrieval_zh,lcc,repobench-p\"
/root/miniconda3/bin/python scripts/import_2024/significance_test.py \
  --data_dir artifacts/reviewer_2026-02-25/qwen_fast400_seed42 \
  --seed_grouped artifacts/reviewer_2026-02-25/qwen_fast400_seed1337 \
  --task_list \"$TASKS\" \
  --hierarchical_bootstrap \
  --fdr_method both \
  --n_bootstrap 5000 \
  --output_prefix significance_full21_fdr_qwen
```

## 7) Theory bridge (paper-ready figures)

- `scripts/run_attn_hist.py` (save hist, fit power-law)
- `scripts/import_2024/functional_residual_real_prior.py`
- `scripts/import_2024/theorem3_adversarial_bimodal.py`

All outputs should be copied into `paper_exports/<date>_qwen_evidence/`.
