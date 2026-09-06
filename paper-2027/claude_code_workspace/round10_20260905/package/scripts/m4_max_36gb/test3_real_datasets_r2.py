#!/usr/bin/env python3
"""
Test 3 真实数据集验证: Broadband R² with real D(Δ)

从 HuggingFace 下载 5 个不同领域的数据集，各取 ~50M tokens，
用 measure_distance_distribution 测量真实 D(Δ)，
然后用标准 Algorithm 1 方法计算 Broadband R²。

对比合成 Δ^{-1.5} 先验的结果，验证 R² 差异来自先验。

Usage:
    python test3_real_datasets_r2.py              # 全部 5 个数据集
    python test3_real_datasets_r2.py --quick       # 每个只取 5M tokens (快速验证)
    python test3_real_datasets_r2.py --dataset wikitext  # 单个数据集
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.rope.learnable_evq import measure_distance_distribution


# ============================================================
# Dataset configs
# ============================================================

DATASETS = {
    "fineweb-edu": {
        "hf_name": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "text_key": "text",
        "streaming": True,
        "description": "Curated educational web text",
    },
    "openwebtext": {
        "hf_name": "Skylion007/openwebtext",
        "split": "train",
        "text_key": "text",
        "streaming": True,
        "description": "Open reproduction of WebText (Reddit-filtered)",
    },
    "wikitext": {
        "hf_name": "wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "text_key": "text",
        "streaming": False,
        "description": "Wikipedia articles (cleaned)",
    },
    "c4": {
        "hf_name": "allenai/c4",
        "config": "en",
        "split": "train",
        "text_key": "text",
        "streaming": True,
        "description": "Colossal Clean Crawled Corpus",
    },
    "tinystories": {
        "hf_name": "roneneldan/TinyStories",
        "split": "train",
        "text_key": "text",
        "streaming": True,
        "description": "Synthetic children's stories (simple narrative)",
    },
}


# ============================================================
# Tokenization & D(Δ) measurement
# ============================================================

def tokenize_and_measure(
    dataset_name: str,
    target_tokens: int = 50_000_000,
    max_delta: int = 4096,
    sample_size: int = 500_000,
    cache_dir: str | None = None,
) -> dict:
    """Download, tokenize, measure D(Δ), return results dict."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    config = DATASETS[dataset_name]
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} — {config['description']}")
    print(f"Target: {target_tokens/1e6:.0f}M tokens, max_delta={max_delta}")
    print(f"{'='*60}")

    # Load tokenizer (fast tokenizer for speed)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    # Load dataset
    load_kwargs = {"path": config["hf_name"], "split": config["split"]}
    if "config" in config:
        load_kwargs["name"] = config["config"]
    if config.get("streaming"):
        load_kwargs["streaming"] = True
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    t0 = time.time()
    print("Loading dataset...")
    ds = load_dataset(**load_kwargs)

    # Tokenize with batch encoding for speed
    print("Tokenizing (batch mode)...")
    all_tokens = []
    total = 0
    n_docs = 0
    batch_texts = []
    BATCH_SIZE = 1000

    for example in ds:
        text = example.get(config["text_key"], "")
        if not text or len(text.strip()) < 50:
            continue
        batch_texts.append(text)
        n_docs += 1

        if len(batch_texts) >= BATCH_SIZE:
            encoded = tokenizer(batch_texts, add_special_tokens=False,
                                truncation=False, return_attention_mask=False)
            for ids in encoded["input_ids"]:
                all_tokens.extend(ids)
                total += len(ids)
            batch_texts = []
            if n_docs % 10000 == 0:
                print(f"  {n_docs} docs, {total/1e6:.1f}M tokens...")
            if total >= target_tokens:
                break

    # Flush remaining batch
    if batch_texts and total < target_tokens:
        encoded = tokenizer(batch_texts, add_special_tokens=False,
                            truncation=False, return_attention_mask=False)
        for ids in encoded["input_ids"]:
            all_tokens.extend(ids)
            total += len(ids)

    elapsed_tok = time.time() - t0
    print(f"Tokenized {n_docs} docs → {total/1e6:.1f}M tokens in {elapsed_tok:.1f}s")

    if total < max_delta * 2:
        return {"dataset": dataset_name, "status": "failed",
                "error": f"Only got {total} tokens, need at least {max_delta*2}"}

    # Convert to tensor
    token_tensor = torch.tensor(all_tokens[:min(total, target_tokens)], dtype=torch.long)

    # Measure D(Δ)
    print(f"Measuring D(Δ) with sample_size={sample_size}...")
    t1 = time.time()
    D_hist = measure_distance_distribution(
        token_tensor, max_delta=max_delta, sample_size=sample_size
    )
    elapsed_d = time.time() - t1
    print(f"D(Δ) measured in {elapsed_d:.1f}s")

    # Compute tail exponent via log-log regression on Δ > 500
    deltas = np.arange(1, max_delta + 1, dtype=np.float64)
    D_np = D_hist.numpy()
    tail_mask = deltas >= 500
    D_tail = D_np[tail_mask]
    d_tail = deltas[tail_mask]
    # Avoid log(0)
    valid = D_tail > 0
    if valid.sum() > 10:
        log_d = np.log(d_tail[valid])
        log_D = np.log(D_tail[valid])
        slope, intercept = np.polyfit(log_d, log_D, 1)
    else:
        slope, intercept = 0, 0

    # Summary stats
    D_mean = D_np.mean()
    D_max = D_np.max()
    D_short = D_np[:10].mean()  # short-range average
    D_long = D_np[500:].mean()  # long-range average

    result = {
        "dataset": dataset_name,
        "status": "success",
        "n_tokens": int(token_tensor.shape[0]),
        "n_docs": n_docs,
        "max_delta": max_delta,
        "D_mean": float(D_mean),
        "D_max": float(D_max),
        "D_short_range_mean": float(D_short),
        "D_long_range_mean": float(D_long),
        "tail_exponent": float(slope),
        "time_tokenize_s": round(elapsed_tok, 1),
        "time_measure_s": round(elapsed_d, 1),
    }

    print(f"  D_mean={D_mean:.6f}, D_max={D_max:.6f}")
    print(f"  D_short(1-10)={D_short:.6f}, D_long(>500)={D_long:.6f}")
    print(f"  Tail exponent: D ~ Δ^{slope:.3f}")

    return result, D_hist


# ============================================================
# Broadband R² calculation (matches theory_numerical_verification.py Test 3)
# ============================================================

def compute_broadband_r2(
    D_hist: torch.Tensor,
    bases: list[int] | None = None,
    L_values: list[int] | None = None,
    n_grid: int = 64,
) -> list[dict]:
    """
    Compute broadband R² using real D(Δ).

    Same method as Test 3 in theory_numerical_verification.py,
    but uses real D(Δ) instead of synthetic Δ^{-1.5}.
    """
    if bases is None:
        bases = [10_000, 100_000, 500_000, 10_000_000]
    if L_values is None:
        L_values = [256, 512, 1024, 2048]

    D_np = D_hist.numpy().astype(np.float64)
    max_delta = len(D_np)

    phi = np.linspace(0, 1, n_grid)
    dphi = phi[1] - phi[0]
    M = np.minimum(phi[:, None], phi[None, :])

    results = []

    for base in bases:
        omega = base ** (-phi)

        for L in L_values:
            L_eff = min(L, max_delta)
            deltas = np.arange(1, L_eff + 1, dtype=np.float64)
            D_trunc = D_np[:L_eff].copy()
            D_trunc /= D_trunc.sum() + 1e-30  # renormalize to L

            # Build kernel K_ij = Σ_Δ D(Δ) cos(ω_i Δ) cos(ω_j Δ)
            cos_table = np.cos(np.outer(omega, deltas))  # (n_grid, L_eff)
            weighted = cos_table * D_trunc[np.newaxis, :]
            K = weighted @ cos_table.T  # (n_grid, n_grid)

            # Fit: K ≈ c₀ + α·I/Δφ + β·M
            # Step 1: off-diagonal → β, c₀
            mask = ~np.eye(n_grid, dtype=bool)
            K_off = K[mask]
            M_off = M[mask]
            A_fit = np.column_stack([np.ones_like(K_off), M_off])
            coeffs, _, _, _ = np.linalg.lstsq(A_fit, K_off, rcond=None)
            c0, beta = coeffs

            # Step 2: diagonal → α
            K_diag = np.diag(K)
            resid_diag = K_diag - c0 - beta * phi
            alpha = max(resid_diag.mean() * dphi, 1e-10)

            # Reconstruct
            I_mat = np.eye(n_grid) * (alpha / dphi)
            K_approx = c0 + beta * M + I_mat

            # R² full
            ss_res = np.sum((K - K_approx) ** 2)
            ss_tot = np.sum((K - K.mean()) ** 2)
            r2_full = 1 - ss_res / ss_tot

            # R² mid-band (exclude edge 10%)
            lo, hi = int(n_grid * 0.1), int(n_grid * 0.9)
            K_mid = K[lo:hi, lo:hi]
            K_approx_mid = K_approx[lo:hi, lo:hi]
            ss_res_mid = np.sum((K_mid - K_approx_mid) ** 2)
            ss_tot_mid = np.sum((K_mid - K_mid.mean()) ** 2)
            r2_mid = 1 - ss_res_mid / ss_tot_mid if ss_tot_mid > 0 else 0

            # τ from fit
            tau_fit = math.sqrt(beta / alpha) if alpha > 1e-10 and beta > 0 else 0

            results.append({
                "base": base,
                "L": L,
                "r2_full": round(r2_full, 4),
                "r2_mid": round(r2_mid, 4),
                "alpha": float(alpha),
                "beta": float(beta),
                "c0": float(c0),
                "tau_fit": round(tau_fit, 4),
            })

    return results


def compute_synthetic_r2(
    alpha_exponent: float = -1.5,
    bases: list[int] | None = None,
    L_values: list[int] | None = None,
    n_grid: int = 64,
) -> list[dict]:
    """Compute R² with synthetic power-law D(Δ) ∝ Δ^alpha for comparison."""
    if bases is None:
        bases = [10_000, 100_000, 500_000, 10_000_000]
    if L_values is None:
        L_values = [256, 512, 1024, 2048]

    phi = np.linspace(0, 1, n_grid)
    dphi = phi[1] - phi[0]
    M = np.minimum(phi[:, None], phi[None, :])

    results = []

    for base in bases:
        omega = base ** (-phi)
        for L in L_values:
            deltas = np.arange(1, L + 1, dtype=np.float64)
            D = deltas ** alpha_exponent
            D /= D.sum()

            cos_table = np.cos(np.outer(omega, deltas))
            weighted = cos_table * D[np.newaxis, :]
            K = weighted @ cos_table.T

            mask = ~np.eye(n_grid, dtype=bool)
            K_off = K[mask]
            M_off = M[mask]
            A_fit = np.column_stack([np.ones_like(K_off), M_off])
            coeffs, _, _, _ = np.linalg.lstsq(A_fit, K_off, rcond=None)
            c0, beta = coeffs

            K_diag = np.diag(K)
            resid_diag = K_diag - c0 - beta * phi
            alpha_val = max(resid_diag.mean() * dphi, 1e-10)

            I_mat = np.eye(n_grid) * (alpha_val / dphi)
            K_approx = c0 + beta * M + I_mat

            ss_res = np.sum((K - K_approx) ** 2)
            ss_tot = np.sum((K - K.mean()) ** 2)
            r2_full = 1 - ss_res / ss_tot

            lo, hi = int(n_grid * 0.1), int(n_grid * 0.9)
            K_mid = K[lo:hi, lo:hi]
            K_approx_mid = K_approx[lo:hi, lo:hi]
            ss_res_mid = np.sum((K_mid - K_approx_mid) ** 2)
            ss_tot_mid = np.sum((K_mid - K_mid.mean()) ** 2)
            r2_mid = 1 - ss_res_mid / ss_tot_mid if ss_tot_mid > 0 else 0

            results.append({
                "base": base, "L": L,
                "r2_full": round(r2_full, 4),
                "r2_mid": round(r2_mid, 4),
            })

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test 3: Real dataset broadband R²")
    parser.add_argument("--quick", action="store_true",
                        help="Use 5M tokens per dataset (fast)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run single dataset (e.g. 'wikitext')")
    parser.add_argument("--max-delta", type=int, default=4096)
    parser.add_argument("--sample-size", type=int, default=500_000)
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to JSON")
    args = parser.parse_args()

    target_tokens = 5_000_000 if args.quick else 50_000_000

    # Select datasets
    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {list(DATASETS.keys())}")
            sys.exit(1)
        dataset_names = [args.dataset]
    else:
        dataset_names = list(DATASETS.keys())

    results_dir = PROJECT_ROOT / "results" / "m4_max_36gb"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Also compute synthetic baseline for comparison
    print("\n" + "=" * 60)
    print("BASELINE: Synthetic D(Δ) ∝ Δ^{-1.5}")
    print("=" * 60)
    synth_r2 = compute_synthetic_r2(alpha_exponent=-1.5)
    for r in synth_r2:
        print(f"  base={r['base']:>10d}  L={r['L']:>5d}  "
              f"R²_full={r['r2_full']:.4f}  R²_mid={r['r2_mid']:.4f}")

    # Run real datasets
    all_results = {"synthetic_baseline": synth_r2, "datasets": {}}

    for name in dataset_names:
        try:
            result, D_hist = tokenize_and_measure(
                name,
                target_tokens=target_tokens,
                max_delta=args.max_delta,
                sample_size=args.sample_size,
            )

            if result["status"] != "success":
                print(f"  FAILED: {result.get('error', 'unknown')}")
                all_results["datasets"][name] = result
                continue

            # Save D(Δ) tensor
            D_path = results_dir / f"D_{name.replace('-', '_')}.pt"
            torch.save(D_hist, D_path)
            print(f"  Saved D(Δ) → {D_path}")

            # Compute R²
            print(f"\n  Computing broadband R²...")
            r2_results = compute_broadband_r2(D_hist)

            # Print comparison table
            print(f"\n  {'base':>10s}  {'L':>5s}  {'R²_full':>8s}  {'R²_mid':>8s}  "
                  f"{'α':>10s}  {'β':>10s}  {'τ_fit':>8s}")
            print(f"  " + "-" * 75)
            for r in r2_results:
                print(f"  {r['base']:>10d}  {r['L']:>5d}  {r['r2_full']:>8.4f}  "
                      f"{r['r2_mid']:>8.4f}  {r['alpha']:>10.4e}  {r['beta']:>10.4e}  "
                      f"{r['tau_fit']:>8.4f}")

            result["r2_results"] = r2_results
            all_results["datasets"][name] = result

        except Exception as e:
            print(f"\n  ERROR on {name}: {e}")
            import traceback
            traceback.print_exc()
            all_results["datasets"][name] = {
                "dataset": name, "status": "error", "error": str(e)
            }

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Real vs Synthetic R²_mid (base=500K, L=2048)")
    print("=" * 70)

    # Synthetic baseline
    synth_500k_2048 = [r for r in synth_r2
                       if r["base"] == 500_000 and r["L"] == 2048]
    if synth_500k_2048:
        print(f"  Synthetic Δ^{{-1.5}}:  R²_mid = {synth_500k_2048[0]['r2_mid']:.4f}")

    for name, data in all_results["datasets"].items():
        if data.get("status") != "success":
            print(f"  {name:20s}:  FAILED — {data.get('error', '')[:50]}")
            continue
        r2_match = [r for r in data.get("r2_results", [])
                    if r["base"] == 500_000 and r["L"] == 2048]
        if r2_match:
            tail = data.get("tail_exponent", 0)
            print(f"  {name:20s}:  R²_mid = {r2_match[0]['r2_mid']:.4f}  "
                  f"(tail ~ Δ^{tail:.2f})")

    # Save
    save_path = args.save or str(results_dir / "test3_real_r2_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved → {save_path}")


if __name__ == "__main__":
    main()
