# A800 Llama-3-8B Hybrid-LoRA Result Summary (2026-02-13)

## 1. Training Setup (completed)
- Base model: `/opt/dfrope/models_alt/LLM-Research/Meta-Llama-3-8B-Instruct`
- Method: `hybrid_a0.2_t100k` + LoRA(q/k/v/o)
- Sequence length: `8192`
- Data source: `allenai/c4:en`
- Max steps: `600`
- Wall time: `3.883 hours`

## 2. Evaluation: Hybrid-LoRA vs Standard Unfinetuned

| Length | Standard (PPL) | Hybrid-LoRA (PPL) | Relative Improvement |
|---|---:|---:|---:|
| 2048 | 15.889 | 15.352 | 3.38% |
| 8192 | 13.633 | 13.506 | 0.93% |
| 16384 | 190.566 | 15.400 | 91.92% |

## 3. Key Takeaways
- 16K result is the key signal: Hybrid-LoRA significantly outperforms unfinetuned base model.
- At 16K, PPL drops from `190.566` to `15.400` (about `91.92%` relative reduction).
- 2K and 8K remain stable, showing no short-context collapse.

## 4. Notes
- `base_yarn_x2` was attempted but failed in current runtime with error: `'rope_type'`.
- This comparison is **Hybrid-LoRA vs Standard Unfinetuned** (no Geo-LoRA baseline in this run).

## 5. Raw Files
- `artifacts/a800_2026-02-13/llama3_hybrid_lora/summary.json`
- `artifacts/a800_2026-02-13/llama3_hybrid_lora/run.log`
- `artifacts/a800_2026-02-13/llama3_hybrid_lora/adapter_config.json`
- `artifacts/a800_2026-02-13/llama3_hybrid_lora_eval/results.json`
- `artifacts/a800_2026-02-13/llama3_hybrid_lora_eval/run.log`
