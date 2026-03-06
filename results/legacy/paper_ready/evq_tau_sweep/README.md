# EVQ τ-Sweep Paper-Ready Data

> Generated: 2026-02-27
> Hardware: RTX 5090 32GB (AutoCloud server)
> Training: TinyStories from-scratch, 500k steps, base=500000

## Files

| File | Description |
|------|-------------|
| `evq_sweep_paper_table.csv` | Flat CSV for all experiments (LaTeX table generation) |
| `50m_seed42.json` | 50M full 8-point τ-sweep (to be synced from server) |
| `125m_seed42.json` | 125M partial sweep seed=42 (to be synced from server) |
| `125m_seed137.json` | 125M cross-seed validation (to be synced from server) |

## Key Results

- **Optimal τ**: 1.5 (consistent across 50M and 125M)
- **50M PPL@16K gain**: -10.9% (τ=1.5 vs τ=0)
- **125M PPL@16K gain**: -18.9% (seed=42), -5.8% (seed=137)
- **Waterbed**: Not observed (short-context PPL preserved)
- **Phase collision**: Minimum at τ=1.5 (0.268)

## Server Data Paths

```
/root/evq_sweep/results/50m/results_final.json
/root/evq_sweep/results/125m/results_final.json
/root/evq_sweep/results/125m_seed137/results_final.json
```

## LaTeX Usage

```python
import pandas as pd
df = pd.read_csv("evq_sweep_paper_table.csv")

# Table 1: 50M full sweep
t1 = df[df["model"] == "50M"]

# Table 2: Cross-scale comparison (τ=0 vs τ=1.5 only)
t2 = df[df["tau"].isin([0.0, 1.5])]
```
