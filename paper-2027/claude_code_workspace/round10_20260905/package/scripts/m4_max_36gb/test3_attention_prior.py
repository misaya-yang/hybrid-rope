#!/usr/bin/env python3
"""
Test 3 终极验证: 从真实 attention patterns 提取 D(Δ)，计算 broadband R²。

三层验证:
  Part A: GPT-2 125M attention → 逐 head D(Δ) → per-head R²
  Part B: 逐 head D(Δ) 拟合 power-law α → 验证 α ≈ 1.0 假设
  Part C: 大规模 sweep (α, base, L, n_grid, 拟合方法) → 找 R² > 0.99 的精确边界

本机 M4 Max MPS, 纯 inference + numpy 计算, ~10-15min total.

Usage:
    python test3_attention_prior.py                    # 全部
    python test3_attention_prior.py --part A           # 只跑 attention extraction
    python test3_attention_prior.py --part B           # 只跑 power-law fitting
    python test3_attention_prior.py --part C           # 只跑大规模 sweep
    python test3_attention_prior.py --n-seqs 200       # 更多序列 (更准)
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ============================================================
# Part A: Extract attention distance distributions from GPT-2
# ============================================================

def extract_attention_D(
    n_seqs: int = 100,
    seq_len: int = 1024,
    device: str = "mps",
    dataset_name: str = "wikitext",
) -> dict:
    """
    Run GPT-2 on real text, extract per-head attention distance distribution.

    Returns: {
        "D_per_head": np.array (n_layers, n_heads, max_delta),  # normalized per head
        "D_global": np.array (max_delta,),  # averaged over all heads
        "D_per_layer": np.array (n_layers, max_delta),
        "meta": {...}
    }
    """
    from transformers import GPT2LMHeadModel, AutoTokenizer
    from datasets import load_dataset

    print("=" * 70)
    print("PART A: Extracting attention D(Δ) from GPT-2 125M")
    print("=" * 70)

    # Load model
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    n_layers = model.config.n_layer  # 12
    n_heads = model.config.n_head  # 12

    # Load text data
    print(f"Loading {dataset_name} for text...")
    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)

    # Tokenize sequences
    print(f"Preparing {n_seqs} sequences of length {seq_len}...")
    all_ids = []
    buffer = []
    for ex in ds:
        text = ex.get("text", "")
        if not text or len(text.strip()) < 50:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(ids)
        while len(buffer) >= seq_len:
            all_ids.append(torch.tensor(buffer[:seq_len], dtype=torch.long))
            buffer = buffer[seq_len:]
            if len(all_ids) >= n_seqs:
                break
        if len(all_ids) >= n_seqs:
            break

    n_seqs_actual = len(all_ids)
    print(f"Got {n_seqs_actual} sequences")

    # Accumulate attention distance histograms
    max_delta = seq_len - 1
    # D_accum[layer, head, delta] accumulates attention mass at each distance
    D_accum = np.zeros((n_layers, n_heads, max_delta), dtype=np.float64)

    print(f"Running inference ({n_seqs_actual} seqs, MPS)...")
    t0 = time.time()

    with torch.no_grad():
        for i, ids in enumerate(all_ids):
            input_ids = ids.unsqueeze(0).to(device)  # (1, seq_len)

            outputs = model(input_ids, output_attentions=True)
            # outputs.attentions: tuple of (1, n_heads, seq_len, seq_len) per layer

            for layer_idx, attn in enumerate(outputs.attentions):
                # attn: (1, n_heads, seq_len, seq_len)
                attn_np = attn[0].cpu().float().numpy()  # (n_heads, seq_len, seq_len)

                # For each query position q, attention to key position k at distance d = q - k
                # Sum attention weights by distance
                for d in range(1, max_delta + 1):
                    # Diagonal band at distance d: attn[h, q, q-d] for q >= d
                    # This is the sub-diagonal at offset d
                    band = np.diagonal(attn_np, offset=-d, axis1=1, axis2=2)  # (n_heads, seq_len-d)
                    D_accum[layer_idx, :, d - 1] += band.sum(axis=1)  # sum over query positions

            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_seqs_actual - i - 1) / rate
                print(f"  [{i+1}/{n_seqs_actual}] {rate:.1f} seq/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s ({n_seqs_actual/elapsed:.1f} seq/s)")

    # Normalize per head
    D_per_head = np.zeros_like(D_accum)
    for l in range(n_layers):
        for h in range(n_heads):
            total = D_accum[l, h].sum()
            if total > 0:
                D_per_head[l, h] = D_accum[l, h] / total

    # Global average
    D_global = D_accum.sum(axis=(0, 1))
    D_global /= D_global.sum()

    # Per-layer average
    D_per_layer = D_accum.sum(axis=1)
    for l in range(n_layers):
        total = D_per_layer[l].sum()
        if total > 0:
            D_per_layer[l] /= total

    return {
        "D_per_head": D_per_head,
        "D_global": D_global,
        "D_per_layer": D_per_layer,
        "meta": {
            "n_seqs": n_seqs_actual,
            "seq_len": seq_len,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "max_delta": max_delta,
            "time_s": round(elapsed, 1),
        },
    }


# ============================================================
# Part B: Fit power-law to per-head D(Δ)
# ============================================================

def fit_power_law(D: np.ndarray, fit_range=(10, 500)) -> dict:
    """Fit D(Δ) ~ Δ^{-α} in log-log space over given range."""
    max_delta = len(D)
    lo, hi = fit_range
    hi = min(hi, max_delta)
    deltas = np.arange(1, max_delta + 1, dtype=np.float64)

    mask = (deltas >= lo) & (deltas <= hi) & (D > 0)
    if mask.sum() < 5:
        return {"alpha": float("nan"), "r2_fit": 0, "intercept": 0}

    log_d = np.log(deltas[mask])
    log_D = np.log(D[mask])

    # Linear regression in log-log
    coeffs = np.polyfit(log_d, log_D, 1)
    alpha = -coeffs[0]  # D ~ Δ^{-alpha}
    intercept = coeffs[1]

    # R² of log-log fit
    predicted = np.polyval(coeffs, log_d)
    ss_res = np.sum((log_D - predicted) ** 2)
    ss_tot = np.sum((log_D - log_D.mean()) ** 2)
    r2_fit = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {"alpha": float(alpha), "r2_fit": float(r2_fit), "intercept": float(intercept)}


def analyze_attention_D(attn_data: dict) -> dict:
    """Part B: Fit power-law to each head's D(Δ)."""
    print("\n" + "=" * 70)
    print("PART B: Power-law fitting to per-head D(Δ)")
    print("=" * 70)

    D_per_head = attn_data["D_per_head"]
    n_layers, n_heads, max_delta = D_per_head.shape

    print(f"\n  {'Layer':>5s}  {'Head':>4s}  {'α':>8s}  {'R²_fit':>8s}  {'type':>12s}")
    print(f"  " + "-" * 45)

    head_fits = []
    for l in range(n_layers):
        for h in range(n_heads):
            fit = fit_power_law(D_per_head[l, h], fit_range=(10, 500))
            alpha = fit["alpha"]
            r2 = fit["r2_fit"]

            # Classify head type
            if alpha > 1.5:
                htype = "very local"
            elif alpha > 0.8:
                htype = "local"
            elif alpha > 0.3:
                htype = "mixed"
            else:
                htype = "global"

            head_fits.append({
                "layer": l, "head": h,
                "alpha": round(alpha, 3),
                "r2_fit": round(r2, 4),
                "type": htype,
            })
            print(f"  L{l:02d}    H{h:02d}  {alpha:8.3f}  {r2:8.4f}  {htype:>12s}")

    # Statistics
    alphas = [f["alpha"] for f in head_fits if not math.isnan(f["alpha"])]
    print(f"\n  α statistics:")
    print(f"    mean = {np.mean(alphas):.3f}")
    print(f"    median = {np.median(alphas):.3f}")
    print(f"    std = {np.std(alphas):.3f}")
    print(f"    min = {np.min(alphas):.3f}, max = {np.max(alphas):.3f}")

    # Type breakdown
    types = [f["type"] for f in head_fits]
    for t in ["very local", "local", "mixed", "global"]:
        cnt = types.count(t)
        print(f"    {t}: {cnt}/{len(types)} ({100*cnt/len(types):.0f}%)")

    # Global fit
    D_global = attn_data["D_global"]
    global_fit = fit_power_law(D_global, fit_range=(10, 500))
    print(f"\n  Global D(Δ) fit: α = {global_fit['alpha']:.3f}, R²_fit = {global_fit['r2_fit']:.4f}")

    return {"head_fits": head_fits, "global_fit": global_fit, "alpha_stats": {
        "mean": round(float(np.mean(alphas)), 3),
        "median": round(float(np.median(alphas)), 3),
        "std": round(float(np.std(alphas)), 3),
    }}


# ============================================================
# Broadband R² computation (shared)
# ============================================================

def compute_r2(D: np.ndarray, base: int, n_grid: int = 64) -> dict:
    """Compute broadband R² for given D(Δ) and base."""
    L = len(D)
    phi = np.linspace(0, 1, n_grid)
    dphi = phi[1] - phi[0]
    omega = base ** (-phi)
    M = np.minimum(phi[:, None], phi[None, :])

    deltas = np.arange(1, L + 1, dtype=np.float64)

    cos_table = np.cos(np.outer(omega, deltas))
    weighted = cos_table * D[np.newaxis, :]
    K = weighted @ cos_table.T

    # Two-step fit
    mask = ~np.eye(n_grid, dtype=bool)
    A_fit = np.column_stack([np.ones(mask.sum()), M[mask]])
    coeffs, _, _, _ = np.linalg.lstsq(A_fit, K[mask], rcond=None)
    c0, beta = coeffs

    K_diag = np.diag(K)
    resid_diag = K_diag - c0 - beta * phi
    alpha_fit = resid_diag.mean() * dphi

    I_mat = np.eye(n_grid) * (alpha_fit / dphi)
    K_approx = c0 + beta * M + I_mat

    ss_res = np.sum((K - K_approx) ** 2)
    ss_tot = np.sum((K - K.mean()) ** 2)
    r2_full = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    lo, hi = int(n_grid * 0.1), int(n_grid * 0.9)
    K_mid = K[lo:hi, lo:hi]
    K_approx_mid = K_approx[lo:hi, lo:hi]
    ss_res_mid = np.sum((K_mid - K_approx_mid) ** 2)
    ss_tot_mid = np.sum((K_mid - K_mid.mean()) ** 2)
    r2_mid = 1 - ss_res_mid / ss_tot_mid if ss_tot_mid > 0 else 0

    tau_fit = math.sqrt(beta / alpha_fit) if alpha_fit > 0 and beta > 0 else float("nan")

    return {
        "r2_full": round(r2_full, 6),
        "r2_mid": round(r2_mid, 6),
        "alpha_fit": float(alpha_fit),
        "beta": float(beta),
        "tau_fit": round(tau_fit, 4) if not math.isnan(tau_fit) else None,
    }


# ============================================================
# Part C: Comprehensive R² computation with attention D(Δ)
# ============================================================

def compute_attention_r2(attn_data: dict, bases=None, L_values=None) -> dict:
    """Part C: R² with real attention D(Δ) across configurations."""
    print("\n" + "=" * 70)
    print("PART C: Broadband R² with attention-based D(Δ)")
    print("=" * 70)

    if bases is None:
        bases = [10_000, 50_000, 100_000, 500_000, 1_000_000, 10_000_000]
    if L_values is None:
        L_values = [256, 512, 1024]

    D_per_head = attn_data["D_per_head"]
    D_global = attn_data["D_global"]
    D_per_layer = attn_data["D_per_layer"]
    n_layers, n_heads, max_delta = D_per_head.shape

    results = {}

    # --- C1: Global D(Δ) ---
    print("\n--- C1: Global attention D(Δ) ---")
    print(f"  {'base':>10s}  {'L':>6s}  {'R²_full':>10s}  {'R²_mid':>10s}  {'τ_fit':>10s}")
    print(f"  " + "-" * 50)

    global_results = []
    for base in bases:
        for L in L_values:
            D_trunc = D_global[:L].copy()
            D_trunc /= D_trunc.sum()
            r = compute_r2(D_trunc, base)
            global_results.append({"base": base, "L": L, **r})
            tau_str = f"{r['tau_fit']:10.2f}" if r['tau_fit'] else "       NaN"
            marker = " ***" if r["r2_mid"] > 0.99 else (" **" if r["r2_mid"] > 0.95 else "")
            print(f"  {base:10d}  {L:6d}  {r['r2_full']:10.6f}  {r['r2_mid']:10.6f}  "
                  f"{tau_str}{marker}")
    results["global"] = global_results

    # --- C2: Per-head R² (at base=500K, L=512) ---
    print(f"\n--- C2: Per-head R² (base=500K, L=512) ---")
    print(f"  {'Layer':>5s}  {'Head':>4s}  {'R²_mid':>10s}  {'α_head':>8s}")
    print(f"  " + "-" * 35)

    base_focus, L_focus = 500_000, 512
    head_r2s = []
    for l in range(n_layers):
        for h in range(n_heads):
            D_h = D_per_head[l, h, :L_focus].copy()
            s = D_h.sum()
            if s < 1e-10:
                continue
            D_h /= s
            r = compute_r2(D_h, base_focus)
            alpha_h = fit_power_law(D_per_head[l, h], (10, 500))["alpha"]
            head_r2s.append({
                "layer": l, "head": h,
                "r2_mid": r["r2_mid"],
                "alpha_head": round(alpha_h, 3),
            })
            marker = " ***" if r["r2_mid"] > 0.99 else (" **" if r["r2_mid"] > 0.95 else "")
            print(f"  L{l:02d}    H{h:02d}  {r['r2_mid']:10.6f}  {alpha_h:8.3f}{marker}")

    results["per_head"] = head_r2s

    # R² vs α correlation
    if head_r2s:
        r2s = np.array([h["r2_mid"] for h in head_r2s])
        alphas = np.array([h["alpha_head"] for h in head_r2s])
        valid = ~np.isnan(alphas)
        if valid.sum() > 5:
            corr = np.corrcoef(alphas[valid], r2s[valid])[0, 1]
            print(f"\n  Correlation(α_head, R²_mid) = {corr:.3f}")

            # Best heads
            sorted_idx = np.argsort(r2s)[::-1]
            print(f"\n  Top 5 heads by R²_mid:")
            for i in sorted_idx[:5]:
                h = head_r2s[i]
                print(f"    L{h['layer']:02d} H{h['head']:02d}: R²_mid={h['r2_mid']:.4f}, α={h['alpha_head']:.3f}")
            print(f"  Bottom 5 heads by R²_mid:")
            for i in sorted_idx[-5:]:
                h = head_r2s[i]
                print(f"    L{h['layer']:02d} H{h['head']:02d}: R²_mid={h['r2_mid']:.4f}, α={h['alpha_head']:.3f}")

    # --- C3: Per-layer D(Δ) ---
    print(f"\n--- C3: Per-layer R² (base=500K, L=512) ---")
    layer_results = []
    for l in range(n_layers):
        D_l = D_per_layer[l, :L_focus].copy()
        D_l /= D_l.sum()
        r = compute_r2(D_l, base_focus)
        alpha_l = fit_power_law(D_per_layer[l], (10, 500))["alpha"]
        layer_results.append({"layer": l, "r2_mid": r["r2_mid"], "alpha": round(alpha_l, 3)})
        marker = " ***" if r["r2_mid"] > 0.99 else (" **" if r["r2_mid"] > 0.95 else "")
        print(f"  Layer {l:2d}: R²_mid={r['r2_mid']:.6f}, α={alpha_l:.3f}{marker}")
    results["per_layer"] = layer_results

    # --- C4: Comparison table ---
    print(f"\n--- C4: D(Δ) comparison (base=500K, L=512) ---")
    print(f"  {'Prior':>30s}  {'R²_mid':>10s}")
    print(f"  " + "-" * 42)

    comparison = {}
    # Attention global
    D_g = D_global[:L_focus].copy()
    D_g /= D_g.sum()
    r = compute_r2(D_g, base_focus)
    comparison["attention_global"] = r["r2_mid"]
    print(f"  {'Attention (global avg)':>30s}  {r['r2_mid']:10.6f}")

    # Power-law priors
    deltas = np.arange(1, L_focus + 1, dtype=np.float64)
    for alpha in [0.5, 0.8, 0.9, 0.95, 1.0, 1.1, 1.5]:
        D_pl = deltas ** (-alpha)
        D_pl /= D_pl.sum()
        r = compute_r2(D_pl, base_focus)
        name = f"Δ^{{-{alpha}}}"
        comparison[name] = r["r2_mid"]
        marker = " ***" if r["r2_mid"] > 0.99 else (" **" if r["r2_mid"] > 0.95 else "")
        print(f"  {name:>30s}  {r['r2_mid']:10.6f}{marker}")

    # Uniform
    D_u = np.ones(L_focus) / L_focus
    r = compute_r2(D_u, base_focus)
    comparison["uniform"] = r["r2_mid"]
    print(f"  {'Uniform':>30s}  {r['r2_mid']:10.6f}")

    results["comparison"] = comparison

    # --- C5: n_grid sensitivity ---
    print(f"\n--- C5: n_grid sensitivity (base=500K, L=512, α=1.0) ---")
    D_1 = deltas ** (-1.0)
    D_1 /= D_1.sum()
    for ng in [16, 32, 48, 64, 96, 128, 256]:
        r = compute_r2(D_1, base_focus, n_grid=ng)
        marker = " ***" if r["r2_mid"] > 0.99 else (" **" if r["r2_mid"] > 0.95 else "")
        print(f"  n_grid={ng:4d}  R²_mid={r['r2_mid']:.6f}{marker}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test 3: Attention-based D(Δ) and broadband R²")
    parser.add_argument("--part", type=str, default="ABC", help="Which parts to run (A/B/C)")
    parser.add_argument("--n-seqs", type=int, default=100, help="Number of sequences for attention extraction")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results" / "m4_max_36gb"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    attn_data = None

    # Part A: Extract attention D(Δ)
    if "A" in args.part.upper():
        attn_data = extract_attention_D(
            n_seqs=args.n_seqs,
            seq_len=args.seq_len,
            device=args.device,
        )
        all_results["meta"] = attn_data["meta"]

        # Save D tensors
        np.save(results_dir / "D_attention_global.npy", attn_data["D_global"])
        np.save(results_dir / "D_attention_per_head.npy", attn_data["D_per_head"])
        np.save(results_dir / "D_attention_per_layer.npy", attn_data["D_per_layer"])
        print(f"\n  Saved D tensors to {results_dir}/D_attention_*.npy")

    # Load cached if Part A not run
    if attn_data is None:
        global_path = results_dir / "D_attention_global.npy"
        if global_path.exists():
            print("Loading cached attention D(Δ)...")
            attn_data = {
                "D_global": np.load(results_dir / "D_attention_global.npy"),
                "D_per_head": np.load(results_dir / "D_attention_per_head.npy"),
                "D_per_layer": np.load(results_dir / "D_attention_per_layer.npy"),
                "meta": {"n_layers": 12, "n_heads": 12},
            }
        else:
            print("ERROR: No cached attention data. Run with --part A first.")
            sys.exit(1)

    # Part B: Power-law fitting
    if "B" in args.part.upper():
        fit_results = analyze_attention_D(attn_data)
        all_results["power_law_fits"] = fit_results

    # Part C: Broadband R²
    if "C" in args.part.upper():
        r2_results = compute_attention_r2(attn_data)
        all_results["broadband_r2"] = r2_results

    # Save
    save_path = args.save or str(results_dir / "test3_attention_prior_results.json")

    # Convert numpy types for JSON serialization
    def json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=json_default)
    print(f"\nResults saved → {save_path}")


if __name__ == "__main__":
    main()
