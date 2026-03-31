# 0_README (Operator Entry)

Last updated: 2026-02-26 01:20 CST

Primary objective for next session:
- Continue experiments from a protocol-locked codebase that already includes:
  - adapter layout compatibility (`root_adapter|final_lora`)
  - expanded Plan B eval controls
  - unified result schema fields
  - long-instruction mix data builder

If only three commands are run first, use these:

1) Build/refresh model registry

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
/root/miniconda3/bin/python scripts/build_model_registry.py \
  --repo_root . \
  --model_cache_dir /root/autodl-tmp/dfrope/ms_models \
  --run_roots artifacts/cross_model_fast_tuned_b1_gc,artifacts/plan_b_runs
```

2) Dry-run Plan B matrix and verify command surface

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
/root/miniconda3/bin/python scripts/plan_b_eval_longbench.py \
  --base_model_path /root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --runs_root artifacts/plan_b_runs \
  --output_root artifacts/plan_b_eval \
  --methods baseline,anchored_sigmoid \
  --seeds 42,1337 \
  --task_set lb6 \
  --tasks qasper,hotpotqa,multi_news,gov_report,narrativeqa,2wikimqa \
  --max_samples_per_task 0 \
  --niah_needles_per_prompt 4 \
  --niah_trials_per_cell 1 \
  --passkey_trials_per_cell 20 \
  --dry_run
```

3) Launch full Plan B eval (remove `--dry_run`) after lock checks pass

Stop conditions:
- Any parity mismatch (`prompt_source/chat_template/truncate/max_new_tokens_policy`) across compared rows.
- Any result file missing `protocol_lock`, `manifest_json`, `per_sample_scores_raw`, or `inv_sha256`.
