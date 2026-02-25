# Fine-tuning Guide for Sparse Attention

## Overview

This directory contains a complete fine-tuning framework for prior-guided sparse attention. Since zero-shot sparse attention failed (PPL +950%), **fine-tuning is essential** to adapt the model to sparse attention patterns.

## Hardware Requirements

| Setup | GPU | Memory | Suitable For |
|-------|-----|--------|--------------|
| **M4 Max** | Apple Silicon | 36GB | GPT-2 Small (117M), GPT-2 Medium (345M) with LoRA |
| **A100 40GB** | NVIDIA | 40GB | GPT-2 Large (774M) with LoRA, GPT-2 Medium full fine-tune |
| **A100 80GB** | NVIDIA | 80GB | GPT-2 XL (1.5B) with LoRA, GPT-2 Large full fine-tune |

## Quick Start

### Option 1: M4 Max (Local)

```bash
# Single experiment: GPT-2 Small, LoRA, 3 epochs
bash run_finetune_m4.sh

# Or manually:
python finetune_sparse.py \
    --model gpt2 \
    --method lora \
    --lam 0.01 \
    --gamma 1.0 \
    --epochs 3 \
    --batch_size 2 \
    --max_length 256
```

### Option 2: A100/H100 (Cloud GPU)

```bash
# SSH into your GPU instance, then:
bash run_finetune_a100.sh

# Full Pareto search (finds optimal γ, λ)
bash run_pareto_search.sh
```

## Detailed Usage

### 1. Single Experiment

Train one configuration:

```bash
python finetune_sparse.py \
    --model gpt2 \
    --method lora \
    --variant prior_sparse \
    --lam 0.01 \          # Prior strength
    --gamma 2.0 \         # Sparsemax temperature
    --epochs 3 \
    --batch_size 4 \
    --max_length 512 \
    --lr 5e-5 \
    --use_amp             # Enable for CUDA GPUs
```

**Key Parameters:**
- `--lam` (λ): Prior strength. Start with 0.01, increase if model doesn't use prior.
- `--gamma` (γ): Sparsemax temperature. Lower = sparser. Try 0.5, 1.0, 2.0, 5.0.
- `--method`: `full` (all parameters) or `lora` (low-rank adapters, memory efficient).

### 2. Pareto Search (Recommended)

Grid search to find the best sparsity-accuracy trade-off:

```bash
python search_pareto.py \
    --model gpt2 \
    --method lora \
    --epochs 3 \
    --lambdas 0.0 0.005 0.01 0.02 0.05 \
    --gammas 0.5 1.0 2.0 5.0 10.0
```

This will:
1. Train a dense baseline
2. Train sparse models for each (λ, γ) combination
3. Generate Pareto frontier plot
4. Report sweet spot configurations

### 3. Dense Baseline (For Comparison)

Train a dense model with same settings:

```bash
python finetune_sparse.py \
    --model gpt2 \
    --method lora \
    --no_sparse \         # Uses standard softmax
    --epochs 3
```

## Expected Results

### Without Fine-tuning (Zero-shot)
| Model | Sparsity | PPL | Status |
|-------|----------|-----|--------|
| GPT-2 | 97% | 277 (+950%) | ❌ Fails |

### With Fine-tuning (Target)
| Model | Sparsity | PPL | Status |
|-------|----------|-----|--------|
| GPT-2 | 70% | ~28 (+5%) | ✅ Target |
| GPT-2 | 85% | ~32 (+20%) | ✅ Acceptable |

**If fine-tuning works**, you should see PPL recover to within 5-20% of dense baseline while achieving 70-85% sparsity.

## Output Structure

```
outputs/finetune/
├── gpt2_lora_prior_sparse_gamma1.0_20260225_120000/
│   ├── config.json           # Experiment configuration
│   ├── summary.json          # Final metrics
│   ├── history.json          # Per-epoch training history
│   ├── best_model/           # Best checkpoint (lowest eval PPL)
│   │   └── adapter_config.json
│   │   └── adapter_model.bin
│   └── final_model/          # Final checkpoint
│
└── pareto_search_20260225_130000/
    ├── pareto_results.csv    # All (λ, γ) combinations
    └── pareto_frontier.png   # Visualization
```

## Monitoring Training

Training progress is printed to stdout:

```
Epoch 1/3
Epoch 1 | Loss: 3.1234 | PPL: 22.72 | LR: 4.23e-05 | Sparsity: 72.3% | NNZ: 35.2
  Train - Loss: 2.9876, PPL: 19.85
  Eval  - Loss: 2.8453, PPL: 17.21, Sparsity: 74.2%, NNZ: 32.8
  *** New best PPL: 17.21 ***
```

Key metrics:
- **PPL**: Perplexity (lower is better)
- **Sparsity**: % of zeros in allowed attention region
- **NNZ**: Average non-zeros per token (lower = more sparse)

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 1

# Reduce sequence length
--max_length 256

# Enable gradient checkpointing (already on by default)

# Use LoRA instead of full fine-tuning
--method lora
```

### Training Too Slow
```bash
# Enable mixed precision (CUDA only)
--use_amp

# Reduce max_train_samples for quick tests
--max_train_samples 10000
```

### PPL Not Improving
1. **Increase epochs**: Try 5-10 epochs instead of 3
2. **Adjust learning rate**: Try 1e-4 or 2e-5
3. **Reduce gamma**: Lower γ = more sparsity but harder to train
4. **Check prior**: If λ is too high, prior may dominate semantics

### PPL Good but No Sparsity
- Decrease gamma: `--gamma 0.5`
- Increase lambda: `--lam 0.05`
- Check stats are being computed (see logs for "Sparsity: XX%")

## Scaling to Larger Models

### GPT-2 Medium (345M)
```bash
python finetune_sparse.py \
    --model gpt2-medium \
    --method lora \
    --batch_size 2 \
    --max_length 512
```

### GPT-2 Large (774M) - Needs A100
```bash
python finetune_sparse.py \
    --model gpt2-large \
    --method lora \
    --lora_rank 8 \       # Lower rank for memory
    --batch_size 4 \
    --max_length 512 \
    --use_amp
```

### GPT-2 XL (1.5B) - Needs A100 80GB
```bash
python finetune_sparse.py \
    --model gpt2-xl \
    --method lora \
    --lora_rank 8 \
    --batch_size 2 \
    --max_length 512 \
    --use_amp
```

## Paper Checklist

After running experiments, you should have:

- [ ] Dense baseline PPL (for comparison)
- [ ] Sparse model PPL after fine-tuning
- [ ] Pareto curve plot (PPL vs Sparsity)
- [ ] At least one "sweet spot" config with ≥70% sparsity and ≤+5% PPL
- [ ] Training curves showing convergence
- [ ] Reproducible command lines (in logs)

## Next Steps

1. **Run dense baseline**:
   ```bash
   python finetune_sparse.py --no_sparse --epochs 3
   ```

2. **Run Pareto search** (on A100):
   ```bash
   bash run_pareto_search.sh
   ```

3. **Analyze results**:
   - Check `pareto_results.csv` for best config
   - Open `pareto_frontier.png` to see trade-off curve

4. **Final validation**:
   - Train best config for more epochs (5-10)
   - Report final numbers in paper

## Citation

If you use this code, please cite:

```bibtex
@misc{prior_sparse_attention_2026,
  title={Prior-Guided Variational Sparse Attention},
  year={2026},
  note={Experimental framework for fine-tuning sparse attention}
}
```
