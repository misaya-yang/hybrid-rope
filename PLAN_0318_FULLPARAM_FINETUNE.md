# 2026-03-18 LLaMA-3-8B EVQ Continued Pretraining Plan

## Goal

Continued pretraining of LLaMA-3-8B-Instruct with EVQ-Cosh frequency allocation.
证明EVQ在7B+规模预训练模型上的长程PPL和passkey优势。

## Model

- **LLaMA-3-8B-Instruct** (Meta-Llama-3-8B-Instruct)
- head_dim=128, rope_theta=500000, native context=8192
- 64 inv_freq values

## EVQ-Cosh Config

- tau = 128 / sqrt(8192) = **1.414**
- base = 500000 (same as original)
- EVQ replaces inv_freq before training

## Training

- **Method**: LoRA rank=64, target=Q/K projections only
- **Data**: PG-19 (60%) + C4 long (20%) + C4 short + WikiText (20%)
  - Packed into train_packed.pt (4000 chunks x 8192 tokens = 32M tokens)
- **Steps**: 2000, batch=1, grad_accum=8, effective_batch=8
- **LR**: 2e-5, cosine schedule, 5% warmup
- **Precision**: bf16, gradient checkpointing, SDPA
- **VRAM**: ~28GB (fits RTX 5090 32GB), ~30GB with longer context

## H2H Design

Two configs, same data/steps/seed, only inv_freq differs:
1. **GEO** (control): original LLaMA-3 frequencies
2. **EVQ** (treatment): EVQ-Cosh tau=1.414

## Evaluation

| Metric | Lengths | Method |
|--------|---------|--------|
| PPL (sliding window) | 8K, 16K, 32K | PG-19 test split |
| Passkey retrieval | 8K, 16K, 32K | 5-digit number, 20 trials |
| YaRN scaling | Applied at 16K, 32K | NTK-aware interpolation |

## Files

| File | Purpose |
|------|---------|
| `scripts/text_eval/llama3_continued_pretrain.py` | Main experiment script |
| `scripts/text_eval/eval_454m_multilength.py` | 454M multi-length eval (RTX PRO 6000) |
| `scripts/text_eval/prepare_training_data.py` | Data tokenization and packing |

## Server Status

### AWS T4 (52.65.136.42)
- LLaMA-3-8B-Instruct: READY (15GB, ModelScope mirror)
- Training data packed: READY (train_packed.pt 151MB + test sets)
- Scripts uploaded: READY
- Running: tau fine sweep (DiT, PID 12162, since Mar 17)

### RTX PRO 6000 (Blackwell, port 23173)
- 454M phase17c checkpoints: available
- Plan: 1hr eval of 454M at multi-length PPL + passkey

## Execution Order

1. **RTX PRO 6000 (1hr)**: Run eval_454m_multilength.py on phase17c checkpoints
2. **Transfer**: SCP packed data from AWS to RTX PRO 6000
3. **RTX PRO 6000 (2hr)**: Run llama3_continued_pretrain.py full h2h
4. Or **AWS T4**: Smoke test with `--pilot --steps 5`

## Smoke Test (T4)

```bash
source /home/ubuntu/venv/bin/activate
python /home/ubuntu/llama3_continued_pretrain.py \
    --model_dir /home/ubuntu/models/LLM-Research/Meta-Llama-3-8B-Instruct \
    --data_dir /home/ubuntu/data/packed \
    --output_dir /home/ubuntu/results/llama3_evq \
    --steps 5 --pilot
```

## Expected Results

If EVQ-Cosh works at 8B scale:
- 8K PPL: EVQ ≈ GEO (zero-cost at training length)
- 16K PPL: EVQ < GEO (extrapolation advantage)
- 32K PPL: EVQ << GEO (larger advantage at larger extrapolation)
- Passkey: EVQ >= GEO at all lengths
