#!/usr/bin/env python3
"""
Test 3: 大规模 sweep 找到 R²_mid > 0.99 的精确边界。

扫描维度:
  1. base ∈ [500, 10M]  (细粒度)
  2. L ∈ [64, 8192]
  3. α ∈ [0.3, 1.5]  (power-law exponent)
  4. n_grid ∈ [16, 256]
  5. 真实 attention D(Δ) (per-head, per-layer, global)
  6. 拟合方法变体 (两步 vs 联合)
  7. mid-band 边界 (5% vs 10% vs 20%)

纯 CPU numpy 计算，单 config ~1ms, 总计 ~万级 configs < 1min.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "m4_max_36gb"


# ============================================================
# Core R² computation (optimized for batch sweep)
# ============================================================

def compute_r2_batch(
    D: np.ndarray,
    base: int,
    n_grid: int = 64,
    mid_frac: float = 0.1,
    method: str = "two_step",
) -> dict:
    """Compute broadband R². method: 'two_step' or 'joint'."""
    L = len(D)
    phi = np.linspace(0, 1, n_grid)
    dphi = phi[1] - phi[0]
    omega = base ** (-phi)
    M = np.minimum(phi[:, None], phi[None, :])

    deltas = np.arange(1, L + 1, dtype=np.float64)
    cos_table = np.cos(np.outer(omega, deltas))
    weighted = cos_table * D[np.newaxis, :]
    K = weighted @ cos_table.T

    if method == "two_step":
        # Standard: off-diag → (c0, β), then diag → α
        mask = ~np.eye(n_grid, dtype=bool)
        A_fit = np.column_stack([np.ones(mask.sum()), M[mask]])
        coeffs, _, _, _ = np.linalg.lstsq(A_fit, K[mask], rcond=None)
        c0, beta = coeffs
        K_diag = np.diag(K)
        resid_diag = K_diag - c0 - beta * phi
        alpha_fit = resid_diag.mean() * dphi

    elif method == "joint":
        # Joint: fit all elements simultaneously
        # K_ij ≈ c0 + β·M_ij + α·δ_ij/Δφ
        I_scaled = np.eye(n_grid) / dphi
        n2 = n_grid * n_grid
        K_flat = K.reshape(n2)
        X = np.column_stack([
            np.ones(n2),
            M.reshape(n2),
            I_scaled.reshape(n2),
        ])
        coeffs, _, _, _ = np.linalg.lstsq(X, K_flat, rcond=None)
        c0, beta, alpha_fit = coeffs

    # Reconstruct
    I_mat = np.eye(n_grid) * (alpha_fit / dphi)
    K_approx = c0 + beta * M + I_mat

    # R² full
    ss_res = np.sum((K - K_approx) ** 2)
    ss_tot = np.sum((K - K.mean()) ** 2)
    r2_full = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # R² mid-band
    lo = int(n_grid * mid_frac)
    hi = int(n_grid * (1 - mid_frac))
    if hi <= lo:
        r2_mid = r2_full
    else:
        K_mid = K[lo:hi, lo:hi]
        Ka_mid = K_approx[lo:hi, lo:hi]
        ss_res_m = np.sum((K_mid - Ka_mid) ** 2)
        ss_tot_m = np.sum((K_mid - K_mid.mean()) ** 2)
        r2_mid = 1 - ss_res_m / ss_tot_m if ss_tot_m > 0 else 0

    tau_fit = math.sqrt(beta / alpha_fit) if alpha_fit > 0 and beta > 0 else float("nan")

    return {
        "r2_full": r2_full,
        "r2_mid": r2_mid,
        "alpha_fit": float(alpha_fit),
        "beta": float(beta),
        "tau_fit": tau_fit,
    }


def make_power_law(L, alpha):
    d = np.arange(1, L + 1, dtype=np.float64)
    D = d ** (-alpha)
    return D / D.sum()


# ============================================================
# Sweep 1: (base, L, α) 三维细粒度
# ============================================================

def sweep_base_L_alpha():
    print("=" * 80)
    print("SWEEP 1: (base, L, α) — 三维细粒度扫描")
    print("=" * 80)

    bases = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 500000, 1000000, 10000000]
    L_values = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    alphas = np.arange(0.3, 1.55, 0.05)

    hits = []  # configs where R²_mid > 0.99
    near = []  # configs where R²_mid > 0.95

    total = len(bases) * len(L_values) * len(alphas)
    count = 0

    for base in bases:
        for L in L_values:
            best_r2 = -999
            best_alpha = None
            for alpha in alphas:
                D = make_power_law(L, alpha)
                r = compute_r2_batch(D, base)
                count += 1

                if r["r2_mid"] > best_r2:
                    best_r2 = r["r2_mid"]
                    best_alpha = alpha

                if r["r2_mid"] > 0.99:
                    hits.append({
                        "base": base, "L": L, "alpha": round(float(alpha), 3),
                        "r2_mid": round(r["r2_mid"], 6), "r2_full": round(r["r2_full"], 6),
                        "tau_fit": round(r["tau_fit"], 2) if not math.isnan(r["tau_fit"]) else None,
                    })
                elif r["r2_mid"] > 0.95:
                    near.append({
                        "base": base, "L": L, "alpha": round(float(alpha), 3),
                        "r2_mid": round(r["r2_mid"], 6),
                    })

            print(f"  base={base:>10d}  L={L:>5d}  best_α={best_alpha:.2f}  "
                  f"R²_mid={best_r2:.6f}{'  *** >0.99!' if best_r2 > 0.99 else ('  ** >0.95' if best_r2 > 0.95 else '')}")

    print(f"\n  Total configs: {count}")
    print(f"  R²_mid > 0.99: {len(hits)} configs")
    print(f"  R²_mid > 0.95: {len(near) + len(hits)} configs")

    if hits:
        print(f"\n  === R²_mid > 0.99 HITS ===")
        print(f"  {'base':>10s}  {'L':>5s}  {'α':>6s}  {'R²_mid':>10s}  {'R²_full':>10s}  {'τ_fit':>8s}")
        print(f"  " + "-" * 55)
        for h in sorted(hits, key=lambda x: -x["r2_mid"]):
            tau_str = f"{h['tau_fit']:8.2f}" if h['tau_fit'] else "     NaN"
            print(f"  {h['base']:10d}  {h['L']:5d}  {h['alpha']:6.3f}  "
                  f"{h['r2_mid']:10.6f}  {h['r2_full']:10.6f}  {tau_str}")

    return {"hits": hits, "near": near}


# ============================================================
# Sweep 2: 拟合方法和 mid-band 边界
# ============================================================

def sweep_method_midband():
    print("\n" + "=" * 80)
    print("SWEEP 2: 拟合方法 × mid-band 边界")
    print("=" * 80)

    # Use configs near the boundary from Sweep 1
    test_configs = [
        (10000, 2048, 1.0),
        (10000, 4096, 1.0),
        (10000, 1024, 1.0),
        (500000, 2048, 0.95),
        (500000, 4096, 0.95),
        (100000, 2048, 1.0),
        (5000, 2048, 1.0),
        (2000, 2048, 1.0),
    ]

    methods = ["two_step", "joint"]
    mid_fracs = [0.05, 0.10, 0.15, 0.20, 0.25]

    hits = []

    print(f"\n  {'base':>10s}  {'L':>5s}  {'α':>5s}  {'method':>10s}  {'mid%':>5s}  "
          f"{'R²_full':>10s}  {'R²_mid':>10s}")
    print(f"  " + "-" * 65)

    for base, L, alpha in test_configs:
        D = make_power_law(L, alpha)
        for method in methods:
            for mf in mid_fracs:
                r = compute_r2_batch(D, base, method=method, mid_frac=mf)
                marker = " ***" if r["r2_mid"] > 0.99 else ""
                print(f"  {base:10d}  {L:5d}  {alpha:5.2f}  {method:>10s}  {mf:5.2f}  "
                      f"{r['r2_full']:10.6f}  {r['r2_mid']:10.6f}{marker}")
                if r["r2_mid"] > 0.99:
                    hits.append({
                        "base": base, "L": L, "alpha": alpha,
                        "method": method, "mid_frac": mf,
                        "r2_mid": round(r["r2_mid"], 6),
                        "r2_full": round(r["r2_full"], 6),
                    })

    return {"hits": hits}


# ============================================================
# Sweep 3: n_grid 敏感性
# ============================================================

def sweep_ngrid():
    print("\n" + "=" * 80)
    print("SWEEP 3: n_grid 敏感性")
    print("=" * 80)

    test_configs = [
        (10000, 2048, 1.0),
        (10000, 4096, 1.0),
        (5000, 2048, 1.0),
        (2000, 4096, 1.0),
    ]
    n_grids = [8, 12, 16, 20, 24, 32, 48, 64, 96, 128]

    hits = []

    for base, L, alpha in test_configs:
        D = make_power_law(L, alpha)
        print(f"\n  base={base}, L={L}, α={alpha}")
        for ng in n_grids:
            r = compute_r2_batch(D, base, n_grid=ng)
            marker = " ***" if r["r2_mid"] > 0.99 else (" **" if r["r2_mid"] > 0.95 else "")
            print(f"    n_grid={ng:4d}  R²_full={r['r2_full']:.6f}  R²_mid={r['r2_mid']:.6f}{marker}")
            if r["r2_mid"] > 0.99:
                hits.append({
                    "base": base, "L": L, "alpha": alpha,
                    "n_grid": ng, "r2_mid": round(r["r2_mid"], 6),
                })

    return {"hits": hits}


# ============================================================
# Sweep 4: 真实 attention D(Δ) per-head
# ============================================================

def sweep_attention_per_head():
    print("\n" + "=" * 80)
    print("SWEEP 4: Per-head attention D(Δ) × (base, L)")
    print("=" * 80)

    # Load cached attention D
    ph_path = RESULTS_DIR / "D_attention_per_head.npy"
    if not ph_path.exists():
        print("  No cached per-head data. Run test3_attention_prior.py --part A first.")
        return {"hits": []}

    D_per_head = np.load(ph_path)  # (12, 12, max_delta)
    n_layers, n_heads, max_delta = D_per_head.shape

    bases = [500, 1000, 2000, 5000, 10000, 50000, 100000, 500000]
    L_values = [128, 256, 512, 1024]

    hits = []
    best_per_base = {}

    for base in bases:
        for L in L_values:
            L_eff = min(L, max_delta)
            best_r2 = -999
            best_head = None

            for l in range(n_layers):
                for h in range(n_heads):
                    D_h = D_per_head[l, h, :L_eff].copy()
                    s = D_h.sum()
                    if s < 1e-10:
                        continue
                    D_h /= s
                    r = compute_r2_batch(D_h, base)

                    if r["r2_mid"] > best_r2:
                        best_r2 = r["r2_mid"]
                        best_head = (l, h)

                    if r["r2_mid"] > 0.99:
                        hits.append({
                            "base": base, "L": L, "layer": l, "head": h,
                            "r2_mid": round(r["r2_mid"], 6),
                        })

            if best_head:
                marker = " ***" if best_r2 > 0.99 else (" **" if best_r2 > 0.95 else "")
                print(f"  base={base:>10d}  L={L:>5d}  best=L{best_head[0]:02d}H{best_head[1]:02d}  "
                      f"R²_mid={best_r2:.6f}{marker}")

    if hits:
        print(f"\n  Per-head R²_mid > 0.99: {len(hits)} configs")
        # Show unique heads
        unique_heads = set((h["layer"], h["head"]) for h in hits)
        print(f"  Unique heads: {len(unique_heads)}")
        for l, h in sorted(unique_heads):
            count = sum(1 for x in hits if x["layer"] == l and x["head"] == h)
            print(f"    L{l:02d}H{h:02d}: {count} configs with R²>0.99")

    return {"hits": hits}


# ============================================================
# Sweep 5: 混合先验 D(Δ) = w·Δ^{-α} + (1-w)·uniform
# ============================================================

def sweep_mixed_prior():
    print("\n" + "=" * 80)
    print("SWEEP 5: 混合先验 D(Δ) = w·Δ^{-α} + (1-w)·uniform")
    print("=" * 80)

    bases = [10000, 100000, 500000]
    L = 2048
    alphas = [0.8, 0.9, 1.0, 1.1]
    weights = np.arange(0.5, 1.01, 0.05)

    hits = []
    deltas = np.arange(1, L + 1, dtype=np.float64)
    D_uniform = np.ones(L) / L

    for base in bases:
        for alpha in alphas:
            D_pl = deltas ** (-alpha)
            D_pl /= D_pl.sum()

            best_r2 = -999
            best_w = None
            for w in weights:
                D_mix = w * D_pl + (1 - w) * D_uniform
                D_mix /= D_mix.sum()
                r = compute_r2_batch(D_mix, base)
                if r["r2_mid"] > best_r2:
                    best_r2 = r["r2_mid"]
                    best_w = w
                if r["r2_mid"] > 0.99:
                    hits.append({
                        "base": base, "alpha": alpha, "w": round(float(w), 2),
                        "r2_mid": round(r["r2_mid"], 6),
                    })

            marker = " ***" if best_r2 > 0.99 else (" **" if best_r2 > 0.95 else "")
            print(f"  base={base:>10d}  α={alpha:.1f}  best_w={best_w:.2f}  "
                  f"R²_mid={best_r2:.6f}{marker}")

    return {"hits": hits}


# ============================================================
# Sweep 6: 真实 attention D + 多 base 精细扫
# ============================================================

def sweep_attention_base_fine():
    print("\n" + "=" * 80)
    print("SWEEP 6: 真实 attention D(Δ) × 精细 base sweep")
    print("=" * 80)

    gl_path = RESULTS_DIR / "D_attention_global.npy"
    if not gl_path.exists():
        print("  No cached attention data.")
        return {"hits": []}

    D_global = np.load(gl_path)

    # Very fine base sweep
    bases = list(range(100, 1001, 100)) + list(range(1000, 10001, 500)) + \
            list(range(10000, 100001, 10000)) + [500000, 1000000, 10000000]
    L_values = [128, 256, 512, 1024]

    hits = []

    for L in L_values:
        L_eff = min(L, len(D_global))
        D_trunc = D_global[:L_eff].copy()
        D_trunc /= D_trunc.sum()

        print(f"\n  L = {L}")
        for base in bases:
            r = compute_r2_batch(D_trunc, base)
            if r["r2_mid"] > 0.99:
                hits.append({
                    "base": base, "L": L, "r2_mid": round(r["r2_mid"], 6),
                    "source": "attention_global",
                })
                print(f"    base={base:>10d}  R²_mid={r['r2_mid']:.6f} ***")
            elif r["r2_mid"] > 0.97:
                print(f"    base={base:>10d}  R²_mid={r['r2_mid']:.6f} (close)")

    return {"hits": hits}


# ============================================================
# Main
# ============================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Run all sweeps
    all_results["sweep1_base_L_alpha"] = sweep_base_L_alpha()
    all_results["sweep2_method_midband"] = sweep_method_midband()
    all_results["sweep3_ngrid"] = sweep_ngrid()
    all_results["sweep4_attention_per_head"] = sweep_attention_per_head()
    all_results["sweep5_mixed_prior"] = sweep_mixed_prior()
    all_results["sweep6_attention_base_fine"] = sweep_attention_base_fine()

    # Grand summary
    print("\n" + "=" * 80)
    print("GRAND SUMMARY: All R² > 0.99 hits")
    print("=" * 80)

    total_hits = 0
    for name, res in all_results.items():
        hits = res.get("hits", [])
        if hits:
            print(f"\n  {name}: {len(hits)} hits")
            for h in sorted(hits, key=lambda x: -x.get("r2_mid", 0))[:10]:
                print(f"    {h}")
        total_hits += len(hits)

    if total_hits == 0:
        print("\n  *** NO CONFIG REACHED R²_mid > 0.99 ***")
        print("  Closest configs (>0.95) — check sweep1 'near' list")
    else:
        print(f"\n  Total R²_mid > 0.99 configs: {total_hits}")

    # Find the boundary
    print("\n  R² > 0.99 BOUNDARY CONDITIONS:")
    s1 = all_results["sweep1_base_L_alpha"]
    if s1["hits"]:
        bases_hit = sorted(set(h["base"] for h in s1["hits"]))
        print(f"    base range: {min(bases_hit)} - {max(bases_hit)}")
        Ls_hit = sorted(set(h["L"] for h in s1["hits"]))
        print(f"    L range: {min(Ls_hit)} - {max(Ls_hit)}")
        alphas_hit = sorted(set(h["alpha"] for h in s1["hits"]))
        print(f"    α range: {min(alphas_hit):.3f} - {max(alphas_hit):.3f}")

    save_path = str(RESULTS_DIR / "test3_boundary_sweep_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved → {save_path}")


if __name__ == "__main__":
    main()
