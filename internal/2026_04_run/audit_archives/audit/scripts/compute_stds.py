#!/usr/bin/env python3
"""
Compute per-seed std for paper Tables 4 (EVQ x YaRN, paper file table2_evq_yarn_main.tex)
and Table 5 (PE-dominant, paper file table4_pe_dominant.tex).

Source data:
  - Table 4: docs/exp/2026-03-03_passkey_mix_results.md (Section 1.1, 10% mix per-seed)
            + Section 2.2 means for the YaRN rows (per-seed PPL not preserved locally)
  - Table 5: docs/exp/2026-02-24_128tok_baseline_report.md (Phase 0-3, single-seed for Geo/DAPE/EVQ;
            3-seed only for Learnable tau)

Usage: python3 audit/scripts/compute_stds.py
"""
from __future__ import annotations

import math
from statistics import mean, stdev


def report(label, vals):
    m = mean(vals)
    if len(vals) > 1:
        s = stdev(vals)  # ddof=1 (numpy default)
    else:
        s = float("nan")
    print(f"  {label}: n={len(vals)}, mean={m:.2f}, std(ddof=1)={s:.3f}")
    return m, s


# ============================================================================
# TABLE 4 (paper file: tables/table2_evq_yarn_main.tex)
# Data: 10% mix, 3 seeds (42, 123, 7), 350M, scale=8 fair comparison
# Source: docs/exp/2026-03-03_passkey_mix_results.md Section 1.1 (raw),
#         Section 2.2 (YaRN rows, only means available)
# ============================================================================

# Per-seed RAW (no YaRN) PPL@2K, 4K, 8K, 16K — Section 1.1
geo_raw_per_seed = {
    42:  {"PPL@2K": 67.4, "PPL@4K":  94.9, "PPL@8K": 156.5, "PPL@16K": 251.9,
          "PK@2K": 1.00, "PK@4K": 0.42, "PK@8K": 0.46},
    123: {"PPL@2K": 66.3, "PPL@4K": 100.2, "PPL@8K": 170.9, "PPL@16K": 278.0,
          "PK@2K": 1.00, "PK@4K": 0.74, "PK@8K": 0.36},
    7:   {"PPL@2K": 68.0, "PPL@4K": 101.4, "PPL@8K": 158.2, "PPL@16K": 256.1,
          "PK@2K": 1.00, "PK@4K": 0.60, "PK@8K": 0.40},
}
evq_raw_per_seed = {
    42:  {"PPL@2K": 68.0, "PPL@4K":  95.3, "PPL@8K": 152.5, "PPL@16K": 240.8,
          "PK@2K": 1.00, "PK@4K": 0.82, "PK@8K": 0.60},
    123: {"PPL@2K": 67.3, "PPL@4K":  89.5, "PPL@8K": 144.3, "PPL@16K": 230.8,
          "PK@2K": 1.00, "PK@4K": 0.58, "PK@8K": 0.44},
    7:   {"PPL@2K": 68.3, "PPL@4K":  98.1, "PPL@8K": 154.0, "PPL@16K": 239.9,
          "PK@2K": 1.00, "PK@4K": 0.66, "PK@8K": 0.56},
}

# Per-seed retrieval rates @8K under YaRN (only PK@8K; PPL not preserved locally)
# Source: Section 2.2 last sub-table
yarn_retrieval_per_seed_at_8k = {
    "Geo+YaRN":  {123: 0.58, 42: 0.62, 7: 0.64},
    "EVQ+YaRN":  {123: 1.00, 42: 1.00, 7: 1.00},
}

# 3-seed MEAN values for the YaRN composition rows (PPL per-seed UNAVAILABLE locally)
# Source: docs/exp/2026-03-03_passkey_mix_results.md Section 2.2
table4_means = {
    "Geo":      {"PK@8K": 0.41, "PK@12K": 0.57, "PK@16K": 0.51,
                 "PPL@8K": 161.9, "PPL@16K": 253.2},
    "Geo+YaRN": {"PK@8K": 0.61, "PK@12K": 0.59, "PK@16K": 0.51,
                 "PPL@8K":  82.9, "PPL@16K": 157.7},
    "EVQ":      {"PK@8K": 0.53, "PK@12K": 0.63, "PK@16K": 0.50,
                 "PPL@8K": 150.3, "PPL@16K": 229.5},
    "EVQ+YaRN": {"PK@8K": 1.00, "PK@12K": 0.79, "PK@16K": 0.68,
                 "PPL@8K":  70.9, "PPL@16K": 107.5},
}


print("=" * 80)
print("TABLE 4 (paper Table 4 / file table2_evq_yarn_main.tex): EVQ x YaRN")
print("Source: docs/exp/2026-03-03_passkey_mix_results.md (350M, 10% mix, 3 seeds)")
print("=" * 80)

print("\n--- Per-seed std for RAW PPL (Geo, EVQ raw) ---")
for cfg, ppl_dict in [("Geo raw", geo_raw_per_seed), ("EVQ raw", evq_raw_per_seed)]:
    print(f"\n{cfg}:")
    for L in ["PPL@2K", "PPL@4K", "PPL@8K", "PPL@16K"]:
        vals = [seed[L] for seed in ppl_dict.values()]
        report(f"{L:8s}", vals)

print("\n--- Per-seed std for retrieval @8K (PK@8K) ---")
for cfg in ["Geo+YaRN", "EVQ+YaRN"]:
    vals = list(yarn_retrieval_per_seed_at_8k[cfg].values())
    print(f"\n{cfg}:")
    report("PK@8K (retrieval)", vals)

# Cross-check: the 3-seed mean PPL@8K for Geo raw should be ~161.9 (matches)
print("\n--- Sanity: mean cross-check vs paper values ---")
for cfg, ppl_dict in [("Geo raw", geo_raw_per_seed), ("EVQ raw", evq_raw_per_seed)]:
    vals = [seed["PPL@8K"] for seed in ppl_dict.values()]
    m = mean(vals)
    print(f"  {cfg:10s} PPL@8K mean = {m:.2f}  (paper says: 161.9 / 150.3)")
    vals16 = [seed["PPL@16K"] for seed in ppl_dict.values()]
    print(f"  {cfg:10s} PPL@16K mean = {mean(vals16):.2f}  (paper says: 253.2 / 229.5)")

# Note: PPL@16K mean for Geo raw = (251.9+278.0+256.1)/3 = 262.0 (matches Table 3, NOT Table 4)
# Paper uses scale=8 protocol numbers (253.2) for Table 4

# ============================================================================
# TABLE 5 (paper file: tables/table4_pe_dominant.tex)
# Source: docs/exp/2026-02-24_128tok_baseline_report.md
# 125M, L_train=128, FineWeb-Edu
# ============================================================================

print("\n" + "=" * 80)
print("TABLE 5 (paper Table 5 / file table4_pe_dominant.tex): PE-dominant 128->8K")
print("Source: docs/exp/2026-02-24_128tok_baseline_report.md (Phase 0-3, 125M)")
print("=" * 80)

# Geo: Phase 1 A1, single seed only (seed 42)
# Learnable tau: Phase 3 A4/C1/C2, 3 seeds (42, 137, 256)
# DAPE (B2 lr_mult=100): Phase 2, single seed only (seed 42)
# EVQ tau=5.0: Phase 6 sweep, single seed (seed 42)

# Per-seed Learnable tau (3 seeds 42/137/256)
learnable_per_seed = {
    42:  {"PPL@128": 182.3, "PPL@8K": 441.4, "tau_final": 1.139},
    137: {"PPL@128": 181.6, "PPL@8K": 448.1, "tau_final": 1.144},
    256: {"PPL@128": 179.7, "PPL@8K": 424.4, "tau_final": 1.138},
}

print("\n--- Learnable tau (3 seeds; only row with multi-seed) ---")
for L in ["PPL@128", "PPL@8K", "tau_final"]:
    vals = [s[L] for s in learnable_per_seed.values()]
    report(f"{L:12s}", vals)

# Single-seed rows
print("\n--- Single-seed rows (NO STD AVAILABLE) ---")
print("  Geo (seed 42):              PPL@128=184.9, PPL@8K=513.7  [SINGLE SEED]")
print("  DAPE B2 lr=100 (seed 42):   PPL@128=183.6, PPL@8K=455.3  [SINGLE SEED]")
print("  EVQ tau=5.0 (seed 42):      PPL@128=182.0, PPL@8K=333.7  [SINGLE SEED]")

print("\n*** Paper Table 5 caption says '3-seed mean' but only Learnable tau row is 3-seed ***")
print("*** Geo, DAPE, and EVQ rows are all single-seed (seed 42) ***")
