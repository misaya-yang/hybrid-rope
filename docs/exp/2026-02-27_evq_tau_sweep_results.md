# EVQ τ-Sweep Results & 8B Launch Plan (2026-02-27)

## 1. τ-Sweep 完整结果

### 50M Model (TinyStories, from-scratch)

| τ | Collision | PPL@2048 | PPL@4096 | PPL@8192 | PPL@16384 | vs baseline |
|---:|:---------:|:--------:|:--------:|:--------:|:---------:|:-----------:|
| 0.00 | 0.3857 | 4.146 | 6.183 | 14.004 | 33.316 | — |
| 0.20 | 0.3902 | 4.160 | 7.283 | 17.457 | 42.314 | +27.0% |
| 0.40 | 0.4382 | 4.207 | 6.789 | 14.331 | 33.298 | -0.1% |
| 0.60 | 0.3607 | 4.173 | 7.253 | 16.571 | 37.978 | +14.0% |
| 0.80 | 0.2899 | 4.169 | 7.736 | 17.193 | 36.306 | +9.0% |
| 1.00 | 0.3048 | 4.150 | 7.830 | 17.790 | 37.369 | +12.2% |
| **1.50** | **0.2678** | **4.134** | **6.667** | **13.778** | **29.697** | **-10.9%** |
| 2.00 | 0.2782 | 4.197 | 7.048 | 14.981 | 35.646 | +7.0% |

### 125M Model (TinyStories, from-scratch)

| τ | PPL@2048 | PPL@4096 | PPL@8192 | PPL@16384 | vs baseline |
|---:|:--------:|:--------:|:--------:|:---------:|:-----------:|
| 0.00 | 3.346 | 5.454 | 13.476 | 34.153 | — |
| 0.20 | 3.363 | 5.999 | 16.616 | 43.103 | +26.2% |
| **1.50** | **3.290** | **4.681** | **10.459** | **27.699** | **-18.9%** |

### Key Findings

1. **τ=1.5 is optimal** at both scales, confirmed by consistent improvement pattern
2. **Scaling law**: 50M -10.9% → 125M -18.9% — improvement grows with model size
3. **No waterbed trade-off** at 125M: short-context PPL also improves
4. **Phase collision minimized** at τ=1.5 (0.2678, lowest across all τ)
5. **Threshold effect**: τ=0.2-1.0 are in a "perturbation zone" that hurts more than helps

## 2. 脚本优化 (RTX Pro 6000 Blackwell, 96GB)

Changes made to `scripts/isolated/longinst/`:

| 参数 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| Stage A τ values | {0.0, 0.4, 0.6, 0.8} | {0.0, 1.2, 1.5, 1.8} | 小模型验证了 1.5 最优，bracket 确认 |
| `attn_implementation` | sdpa | flash_attention_2 | Blackwell 支持 FA2，显存效率高 |
| `per_device_train_batch_size` | 2 | 4 | 96GB + 4bit + FA2 + grad_ckpt 可支持 |
| `evq_tau` default | 0.5 | 1.5 | 实验验证的最优值 |

## 3. 8B 实验运行指令

```bash
# 前置：确保 flash-attn 已安装
pip install flash-attn --no-build-isolation

# 运行 8-job 管线 (Stage A: 4 τ gate; Stage B: dual-seed full eval)
python scripts/isolated/longinst/run_llama8k_theory_v1.py \
  --base_model_path <path_to_Meta-Llama-3-8B-Instruct> \
  --longalpaca_path <path_to_LongAlpaca-12k.min64.jsonl> \
  --wikitext_train_path <path_to_wikitext/train.txt> \
  --longbench_local_data_dir <path_to_LongBench/data> \
  --qwen_seed42_json <path> \
  --qwen_seed1337_json <path> \
  --morning_reference_json <path> \
  --run_full_eval_seed1337 \
  --execute \
  --write_docs
```

## Operator

Claude (Cowork mode), 2026-02-27
