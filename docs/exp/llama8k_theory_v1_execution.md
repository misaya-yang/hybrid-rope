# LLaMA-3-8B (8K) 理论验证执行说明 v1

## Scope
- 只验证 `8K` 区间下 `EVQ-Cosh RoPE + LoRA` 相对 `Geometric` 的增益。
- 仅使用 `Meta-Llama-3-8B-Instruct`。
- 不启用 attention bias/gate/额外损失。

## Canonical Entrypoints
- 训练入口：`/Users/yang/projects/hybrid-rope/scripts/isolated/longinst/new_lora_longinst_train_v1.py`
- 调度入口：`/Users/yang/projects/hybrid-rope/scripts/isolated/longinst/run_llama8k_theory_v1.py`
- 统计入口：`/Users/yang/projects/hybrid-rope/scripts/isolated/longinst/paired_stats_llama8k_theory_v1.py`
- 评测入口：`/Users/yang/projects/hybrid-rope/scripts/eval_longbench.py`

## Locked Protocol
- Data source: `must use prebuilt mixed dataset via --mixed_dataset_dir` (paper-grade no on-the-fly fallback)
- RoPE injection: `inv_freq.copy_()`
- Sequence length: `8192`
- LoRA modules: `q_proj,k_proj,v_proj,o_proj`
- Train core defaults: `lr=2e-5, warmup=50, attn=sdpa`
- Gate rule:
  - `qasper_lora >= qasper_base`
  - `musique_lora >= musique_base - 1.0`
- Claim grade: `significant / directional / inconclusive`
- Seed-replication claim guard: if full-eval run-pairs `< 2`, stats claim is forced to `insufficient_seed_replication`.
- Stats protocol parity guard: paired stats will refuse to run if geometric/EVQ have mismatched `code_hash` or `data_hash`.
- Prebuilt mixed dataset guard: trainer enforces source token ratios (`wiki>=0.05`, `synthetic>=0.10`) and manifest-target drift tolerance.
- Source-prior mapping lock: `power_law_base->wiki`, `bimodal_reasoning->synthetic`, `uniform_scaffold->long`.

## Current Job Plan
- Stage A (seed=42, full LB21)
  - A1 geometric (`tau=0.0`) r32 s800
  - A2 evq_cosh (`tau=1.5`) r32 s800
- Stage B (seed=1337, full LB21)
  - B1 geometric (`tau=0.0`) r32 s800
  - B2 evq_cosh (`tau=1.5`) r32 s800

## Run Command
```bash
cd /Users/yang/projects/hybrid-rope
.venv/bin/python scripts/isolated/longinst/run_llama8k_theory_v1.py \
  --execute \
  --mixed_dataset_dir /root/autodl-tmp/dfrope/hybrid-rope/artifacts/datasets/mixed_prior_v1_YYYYMMDD_HHMMSS \
  --mixed_dataset_split train \
  --base_model_path /root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --longalpaca_path /root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.min64.jsonl \
  --longqa_path /root/autodl-tmp/dfrope/datasets/LongQA.jsonl \
  --wikitext_train_path /root/autodl-tmp/wikitext_data/train.txt \
  --longbench_local_data_dir /root/autodl-tmp/dfrope/ms_datasets/LongBench/data \
  --morning_reference_json artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/downstream_eval_fast400/longbench/anchored_sigmoid.json \
  --qwen_seed42_json artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/downstream_eval_fast400/longbench/anchored_sigmoid.json \
  --qwen_seed1337_json artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/downstream_eval_fast400/longbench/anchored_sigmoid.json
```

> 说明：调度器默认启用 `--require_mixed_dataset_dir`，未提供 `--mixed_dataset_dir` 会直接拒绝执行。

## Artifacts
- Train/Eval: `artifacts/llama8k_theory_v1/train/<run_name>/...`
- Registry: `docs/exp/llama8k_theory_v1_registry.csv`
- Report: `docs/exp/llama8k_theory_v1_report.md`
- Run manifest: `artifacts/llama8k_theory_v1/stats/run_manifest.json`

## Fail-Fast Rules
- `assistant_tokens_lt64_ratio > 0.10` -> stop training
- continuation-dominant corpus detected and no override -> stop training
- missing `--mixed_dataset_dir` with `--execute` -> stop immediately
- Stats protocol mismatch (`code_hash/data_hash`) -> stop stats generation
