# Prior-guided Variational Sparse Attention: Reproducible Experiment

## Quick Start

```bash
# 1. Quick test (5-10 min)
python test_quick.py

# 2. Full experiment (1-2 hours)
bash run_experiment.sh

# 3. Custom run
python main_experiment.py \
    --output_dir outputs/my_run \
    --max_tokens 100000 \
    --seq_len 1024
```

## What's Included

| File | Purpose |
|------|---------|
| `attention_patch.py` | Monkey-patch GPT2Attention with three variants |
| `main_experiment.py` | Full experimental protocol |
| `test_quick.py` | Quick validation (5-10 min) |
| `run_experiment.sh` | Bash wrapper for full experiment |
| `PROTOCOL.md` | Detailed experimental protocol |

## Requirements

```bash
pip install torch transformers datasets entmax matplotlib tqdm
```

Tested on:
- Python 3.9-3.11
- PyTorch 2.0+ (MPS support for Apple Silicon)
- transformers 4.30+
- entmax 1.0+

## Three Attention Variants

### A. Baseline (Standard Softmax)
```python
attn = softmax(QK^T / √d)
```

### B. Prior-Biased Softmax
```python
attn = softmax(QK^T / √d + λ * log D(Δ))
```

### C. Prior-Guided Sparse Attention
```python
Z = (QK^T / √d + λ * log D(Δ)) / γ
attn = sparsemax(Z)
```

## Key Implementation Details

1. **Real Forward**: Patches `GPT2Attention._attn()` to actually use sparsemax during forward pass
2. **True PPL**: Computes perplexity from model outputs with real attention weights
3. **Proper Sparsity**: Only counts zeros in allowed (lower-triangular) region
4. **Validated**: Checks row sums to 1.0 and non-negativity

## Expected Output

```
outputs/variational_sparse_attn/20240225_1430/
├── results.json              # Raw data
├── conclusion.txt            # Paper-ready conclusion
├── env.txt                   # Environment info
├── figures/
│   ├── gamma_tradeoff.png   # Main result: γ vs PPL/Sparsity
│   └── pareto_curve.png     # Pareto front: Sparsity vs PPL
└── README.md                # This file
```

## Interpreting Results

### Success Criteria
- Find γ where **Sparsity ≥ 70%** AND **PPL increase ≤ 5%**
- All sanity checks pass
- Smooth trade-off curve

### Example Good Result
```
γ=0.5: PPL=29.2 (+2.5%), Sparsity=75.3%
→ PASS: Sweet spot identified
```

### Example Marginal Result
```
γ=0.3: PPL=30.8 (+8.1%), Sparsity=82.1%
→ MARGINAL: High sparsity but PPL degradation > 5%
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM / Memory error | Reduce `--seq_len` to 512 or `--max_tokens` to 50000 |
| MPS error | Will auto-fallback to CPU; or set `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Slow on CPU | Reduce `--max_tokens` to 20000 for quick test |
| No exact zeros | Check entmax installation: `pip install entmax` |
| Determinism fail | Check no dropout/evaluation mode is on |

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{yourpaper2024,
  title={Prior-Guided Variational Sparse Attention},
  author={...},
  year={2024}
}
```

## License

MIT License - Free for research use.
