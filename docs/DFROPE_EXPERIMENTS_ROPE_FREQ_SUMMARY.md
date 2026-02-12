# RoPE Frequency Experiments Summary (Unified Search + Hybrid/AnchPoly)

Last updated: 2026-02-12

This file is a consolidated snapshot of the experiments we ran after pivoting from DF-RoPE to **RoPE frequency distribution design** (TI-preserving: phase remains linear in distance; we only redesign the frequency set `omega_k` / `inv_freq`).

## Key Takeaways (So Far)

- Under the **same training/eval loop** as `unified_search.py` (TinyStories, 50M train tokens, seq_len=2048, BF16, 50.9M model), simply raising `theta` (e.g. `geo_500k`) already yields a large extrapolation win vs small `theta`.
- The “**hybrid**” frequency distribution (a convex combination of a geometric baseline and an anchored-polynomial reshaping) can **match or beat** a very-large-theta baseline **without having to push theta to extremes**:
  - On A800 split B, `hybrid_a0.2_t100k` achieved `PPL@16384=16.316` vs `geo_500k_ALIGN=17.947` (about **9.1% lower PPL**).
  - On the same run, `hybrid_a0.2_t500k` achieved `PPL@16384=17.980` (roughly tie with `geo_500k_ALIGN`).
- We also observed a consistent but smaller “**micro-lead**” from anchored polynomial over pure geometric at the same high theta (A100 split A):
  - `anchpoly_p3.9_omf0.3_t500k` achieved `PPL@16384=16.459` vs `geo_500k_ALIGN=16.991` (about **3.1% lower PPL**).
- **Seed robustness is not yet proven** in our repo snapshot: the “3 configs × 3 seeds” job was started and is running/partial. We should not claim seed-stable improvements until all three seeds finish for all three configs.

## Repro/Code Artifacts

### Unified Search Script

- Remote path (both machines): `/opt/dfrope/unified_search.py`
- Identical file checksum on both machines when run:
  - `sha256=48c83359c045d188955233a0b96e8fb4c85433173508db143996f1aa7db21556`

### 3-Config 3-Seed Script (In Progress)

- Remote path (A100): `/opt/dfrope/unified_search_3cfg_3seed.py`
- Purpose: run only 3 configs × 3 seeds, report `PPL@2048` and `PPL@16384` with mean±std.
- Seeds: `[42, 123, 7]`

## Experiment: Unified Search (50.9M, TinyStories 50M)

### What We Ran (Process Timeline)

- Step 1: Built a single-file `unified_search.py` that hard-codes:
  - model architecture, tokenizer, streaming dataset ingestion, training hyperparams, evaluation slicing
  - a small list of candidate frequency parameterizations (geometric / sigmoid / anchored polynomial / hybrid mixtures)
- Step 2: Deployed the **exact same file bytes** to two machines and verified checksum equality.
- Step 3: Launched split A on A100 and split B on A800 to cover a wider candidate set in parallel.
- Step 4: Validated alignment by comparing `geo_10k_ALIGN` and `geo_500k_ALIGN` across machines.
- Step 5: Recorded winners and losers and used the leaderboard to decide the next search direction (focus on anchored poly + hybrid; sigmoid de-prioritized).

### Fixed Training/Eval Settings (as implemented by `unified_search.py`)

- Model: 6 layers, `hidden_size=512`, `heads=8`, `head_dim=64`, `intermediate=2048`, vocab 50304, dropout 0.0, max_position_embeddings 2048
- Tokenizer: `EleutherAI/gpt-neox-20b`
- Data:
  - Train: `roneneldan/TinyStories`, split `train`, streaming, first 50,000,000 tokens after tokenization
  - Val: `roneneldan/TinyStories`, split `validation`, streaming, up to 5,000,000 tokens
- Train:
  - `seq_len=2048`
  - `batch_size=32`
  - `lr=6e-4`, cosine schedule, warmup 2%
  - AdamW betas `(0.9,0.95)`, weight_decay `0.1`
  - grad clip `1.0`
  - BF16 autocast
- Eval:
  - lengths: `[2048, 4096, 8192, 16384]`
  - chunks: 10
  - sequential slicing: chunk `i` uses `val[i*L:(i+1)*L]`
- Output JSON:
  - A100: `/opt/dfrope/results/unified_search/results_A.json`
  - A800: `/opt/dfrope/results/unified_search/results_B.json`

### Machine A (A100, split A) Results

Remote results source: `/opt/dfrope/results/unified_search/results_A.json`

Best configs by `PPL@16384` (lower is better):

| Config | PPL@2048 | PPL@16384 |
|---|---:|---:|
| `anchpoly_p3.9_omf0.3_t500k` | 6.578 | **16.459** |
| `geo_500k_ALIGN` | 6.804 | 16.991 |
| `sig_s8_m0.6_omf0.3_t10k` | 6.717 | 25.610 |
| `sig_s7_m0.5_omf0.3_t10k` | 6.790 | 26.588 |

Notes:
- `anchpoly_p3.9_omf0.3_t500k` beat `geo_500k` on this run by ~0.53 PPL at 16K.
- Many sigmoid variants at base `theta=10k` are not competitive at 16K (some are >30 PPL).

### Machine B (A800, split B) Results

Remote results source: `/opt/dfrope/results/unified_search/results_B.json`

Best configs by `PPL@16384` (lower is better):

| Config | PPL@2048 | PPL@16384 |
|---|---:|---:|
| `hybrid_a0.2_t100k` | 6.658 | **16.316** |
| `geo_500k_ALIGN` | 6.852 | 17.947 |
| `hybrid_a0.2_t500k` | 6.817 | 17.980 |
| `mix_sig8t10k_geo500k_a0.5` | 6.756 | 19.355 |

Notes:
- Hybrid at base `theta=100k` is the strongest seen in split B.
- Sigmoid at high theta did not beat the best hybrid/high-theta baselines in this sweep.

### Cross-Machine “Alignment” Sanity Check

`geo_500k_ALIGN` @ 16384:
- A100: 16.991
- A800: 17.947

Relative difference ~5.6% (acceptable for a streaming-tokenization pipeline, but not perfect determinism).

## Experiment: 3 Configs × 3 Seeds (50.9M, TinyStories 50M) [In Progress]

Goal: estimate variance across seeds for the three finalists:

1. `geo_500k`
2. `hybrid_a0.2_t100k`
3. `anchpoly_p3.9_omf0.3_t500k`

Remote paths (A100):
- Log: `/opt/dfrope/results/unified_search_3cfg_3seed/log.txt`
- Results: `/opt/dfrope/results/unified_search_3cfg_3seed/results.json`

Local snapshot captured during run:
- `/Users/misaya.yanghejazfs.com.au/dfrope/remote_results/3cfg_3seed_results.json` (may be partial)

**Interpretation rules (for paper-grade claims):**
- For each config, compute mean±std over seeds for `PPL@2048` and `PPL@16384`.
- Only claim “wins” if:
  - mean is better, and
  - variance is not exploding (std small relative to mean-gap), and
  - ideally at least 2/3 seeds show improvement (not just one lucky seed).

## Data Files Synced Locally (Snapshot)

Stored under:
- `/Users/misaya.yanghejazfs.com.au/dfrope/remote_results/`

Includes:
- `results_A.json` (A100 unified search)
- `log_A.txt` (A100 unified search log)
- `3cfg_3seed_results.json` (partial snapshot, if run not finished)
- `3cfg_3seed_log.txt` (run log snapshot)

## Immediate Next Steps

1. Finish the 3-seed run and update this summary with mean±std tables.
2. After seed-robustness is confirmed, do a **targeted local search** around the winners:
   - anchored poly: sweep `p` and `omf` at high theta (e.g. `theta=500k`), because it already shows a small but meaningful lead.
   - hybrid: sweep `alpha` and `theta_base` (e.g. `100k/200k/500k`) to test the “hybrid replaces high theta” claim under seeds.
3. (If mentor requests) scale-up study on larger model (e.g. 350M) using the same frequency functions, but keep the exact train/eval protocol fixed to avoid “definition drift”.
