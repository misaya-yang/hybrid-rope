# LLaMA-3-8B (8K) 理论验证执行说明 v1

## Scope
- 只验证 `8K` 区间下 `Anchored-RoPE + LoRA` 相对 `Geometric` 的增益。
- 仅使用 `Meta-Llama-3-8B-Instruct`。
- 不启用 attention bias/gate/额外损失。

## Canonical Entrypoints
- 训练入口：`/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/scripts/isolated/longinst/new_lora_longinst_train_v1.py`
- 调度入口：`/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/scripts/isolated/longinst/run_llama8k_theory_v1.py`
- 统计入口：`/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/scripts/isolated/longinst/paired_stats_llama8k_theory_v1.py`
- 评测入口：`/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/scripts/eval_longbench.py`

## Locked Protocol
- Data mix: `70% long-instruction + 10% general instruction + 20% synthetic long-QA`
- RoPE injection: `inv_freq.copy_()`
- Sequence length: `8192`
- LoRA modules: `q_proj,k_proj,v_proj,o_proj`
- Train core defaults: `lr=2e-5, warmup=50, attn=sdpa`
- Gate rule:
  - `qasper_lora >= qasper_base`
  - `musique_lora >= musique_base - 1.0`
- Claim grade: `significant / directional / inconclusive`

## Two-Stage 8 Jobs
- Stage A (seed=42, gate only)
  - A1 geometric r32 s800
  - A2 anchored r32 s800
  - A3 anchored r48 s800
  - A4 anchored r32 s600
- Stage B
  - B1 geometric(best) seed1337 gate
  - B2 anchored(best) seed1337 gate
  - B3 geometric(best) seed42 full LB21
  - B4 anchored(best) seed42 full LB21

## Run Command
```bash
~/miniconda3/bin/conda run -n aidemo python scripts/isolated/longinst/run_llama8k_theory_v1.py \
  --execute \
  --base_model_path /root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --longalpaca_path /root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.min64.jsonl \
  --longqa_path /root/autodl-tmp/dfrope/datasets/LongQA.jsonl \
  --wikitext_train_path /root/autodl-tmp/wikitext_data/train.txt \
  --longbench_local_data_dir /root/autodl-tmp/dfrope/ms_datasets/LongBench/data \
  --morning_reference_json artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/downstream_eval_fast400/longbench/anchored_sigmoid.json \
  --qwen_seed42_json artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/downstream_eval_fast400/longbench/anchored_sigmoid.json \
  --qwen_seed1337_json artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/downstream_eval_fast400/longbench/anchored_sigmoid.json
```

> 说明：`LongAlpaca-12k.min64.jsonl` 为已完成监督长度过滤（assistant_tokens>=64）的版本，优先用于本计划。

## Artifacts
- Train/Eval: `artifacts/llama8k_theory_v1/train/<run_name>/...`
- Registry: `docs/exp/llama8k_theory_v1_registry.csv`
- Report: `docs/exp/llama8k_theory_v1_report.md`
- Run manifest: `artifacts/llama8k_theory_v1/stats/run_manifest.json`

## Fail-Fast Rules
- `assistant_tokens_lt64_ratio > 0.10` -> stop training
- continuation-dominant corpus detected and no override -> stop training
- Stage B gate failed -> cancel full LB21 jobs
