# Runbook (Next-Day Continuation)

Last updated: 2026-02-26 01:20 CST

## 0) Preflight

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
git pull --ff-only
/root/miniconda3/bin/python -m py_compile \
  scripts/train_cross_model_lora_fast_tuned.py \
  scripts/build_model_registry.py \
  scripts/plan_b_eval_longbench.py \
  scripts/prepare_long_instruction_mix.py \
  scripts/eval_longbench.py \
  scripts/eval_niah_recall.py \
  scripts/eval_passkey_teacher_forcing.py
```

## 1) Do not touch running experiment process

Read-only checks only:

```bash
pgrep -af 'train_cross_model_lora_fast_tuned.py|eval_longbench.py|plan_b_'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

If critical run is active, do not kill/restart it.

## 2) Build mixed long-instruction training text

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
/root/miniconda3/bin/python scripts/prepare_long_instruction_mix.py \
  --source "name=src_en_qa;path=/root/autodl-tmp/dfrope/data/src_en_qa.jsonl;lang=en;task_type=qa_retrieval;filter_rule=dedupe_len_guard" \
  --source "name=src_zh_sum;path=/root/autodl-tmp/dfrope/data/src_zh_sum.jsonl;lang=zh;task_type=summary;filter_rule=dedupe_len_guard" \
  --source "name=src_mix_chat;path=/root/autodl-tmp/dfrope/data/src_mix_chat.jsonl;lang=auto;task_type=dialogue_code_structured;filter_rule=dedupe_len_guard" \
  --tokenizer_path /root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --output_dir artifacts/plan_b_data/long_instruction_mix_2026-02-26 \
  --target_train_samples 0 \
  --language_ratio en:0.7,zh:0.3 \
  --task_ratio qa_retrieval:0.4,summary:0.3,dialogue_code_structured:0.3 \
  --bucket_ratio 2k:1,4k:2,8k:3,16k:4 \
  --seed 42
```

## 3) Llama strict training (Plan B)

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
/root/miniconda3/bin/python scripts/plan_b_train_anchored_v2.py \
  --base_model_path /root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --long_instruction_jsonl /root/autodl-tmp/dfrope/data/long_instruction_master.jsonl \
  --prepared_data_dir artifacts/plan_b_data/long_instruction_mix_2026-02-26 \
  --methods baseline,anchored_sigmoid \
  --seeds 42,1337 \
  --max_steps 400 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --max_seq_len 16384
```

## 4) Llama eval (lb6 + stress)

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
/root/miniconda3/bin/python scripts/plan_b_eval_longbench.py \
  --base_model_path /root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --runs_root artifacts/plan_b_runs \
  --output_root artifacts/plan_b_eval \
  --methods baseline,anchored_sigmoid \
  --seeds 42,1337 \
  --task_set lb6 \
  --max_samples_per_task 0 \
  --niah_needles_per_prompt 4 \
  --niah_trials_per_cell 1 \
  --passkey_trials_per_cell 20
```

## 5) Stats gate

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
/root/miniconda3/bin/python scripts/import_2024/significance_test.py \
  --data_dir artifacts/plan_b_eval/baseline_seed42 \
  --seed_grouped artifacts/plan_b_eval/anchored_sigmoid_seed42 \
  --fdr_method both \
  --n_bootstrap 5000 \
  --hierarchical_bootstrap \
  --output_prefix significance_planb_lb6
```

Only claim "significant" when FDR-BH passes.
