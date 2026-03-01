#!/usr/bin/env python3
"""Phase 0: Algorithm 1 blind prediction of τ* from data statistics.

Measures D̂(Δ) from FineWeb-Edu tokens, computes τ* = √(β/α).
This is a pure CPU computation — no GPU, no training, no gradients.

Usage:
    python experiments/run_algorithm1_predict.py --max_delta 128 --base 500000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Use HuggingFace mirror for Chinese mainland servers
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
# Fix: hf-mirror.com returns pagination Link headers pointing to huggingface.co
try:
    import huggingface_hub.utils._pagination as _hf_pag
    _orig_get_next_page = _hf_pag._get_next_page
    def _patched_get_next_page(response):
        url = _orig_get_next_page(response)
        if url and "huggingface.co" in url:
            mirror = os.environ.get("HF_ENDPOINT", "").rstrip("/")
            if mirror and mirror != "https://huggingface.co":
                url = url.replace("https://huggingface.co", mirror)
        return url
    _hf_pag._get_next_page = _patched_get_next_page
except Exception:
    pass

# Add project root to path
_proj_root = str(Path(__file__).resolve().parents[1])
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import torch
from rope.learnable_evq import measure_distance_distribution, estimate_tau_from_distance_prior


def main():
    parser = argparse.ArgumentParser(description="Algorithm 1: τ* prediction from data statistics")
    parser.add_argument("--max_delta", type=int, default=128,
                        help="Maximum distance to measure (should match training seq_len)")
    parser.add_argument("--base", type=float, default=500000.0,
                        help="RoPE base frequency")
    parser.add_argument("--sample_size", type=int, default=100000,
                        help="Number of token positions to sample")
    parser.add_argument("--n_tokens", type=int, default=2000000,
                        help="Number of tokens to load from dataset")
    parser.add_argument("--dataset", type=str, default="fineweb-edu")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Algorithm 1: τ* Prediction (Phase 0)")
    print(f"  max_delta={args.max_delta}  base={args.base}")
    print(f"  sample_size={args.sample_size}  n_tokens={args.n_tokens}")
    print(f"{'='*60}\n")

    # Load tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    print(f"  Tokenizer loaded: vocab_size={tok.vocab_size}")

    # Load data — stream tokenize
    print(f"\n  Loading {args.n_tokens/1e6:.1f}M tokens from {args.dataset}...")
    t0 = time.time()

    from datasets import load_dataset
    if args.dataset == "fineweb-edu":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        text_key = "text"
    else:
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        text_key = "text"

    ids = []
    for x in ds:
        txt = x.get(text_key, "")
        if txt:
            ids.extend(tok.encode(txt, add_special_tokens=False))
        if len(ids) >= args.n_tokens:
            break

    token_ids = torch.tensor(ids[:args.n_tokens], dtype=torch.long)
    elapsed = time.time() - t0
    print(f"  Got {len(token_ids)/1e6:.2f}M tokens in {elapsed:.1f}s")

    # Measure D̂(Δ)
    print(f"\n  Measuring D̂(Δ) for Δ ∈ [1, {args.max_delta}]...")
    t1 = time.time()
    D_hist = measure_distance_distribution(
        token_ids, max_delta=args.max_delta, sample_size=args.sample_size
    )
    elapsed = time.time() - t1
    print(f"  D̂(Δ) computed in {elapsed:.1f}s")

    # Print top-10 distances by weight
    top_vals, top_idx = D_hist.topk(min(10, len(D_hist)))
    print(f"\n  Top-10 distances by D̂(Δ):")
    for v, i in zip(top_vals, top_idx):
        print(f"    Δ={i.item()+1:4d}: D̂={v.item():.6f}")

    # Estimate τ*
    print(f"\n  Running Algorithm 1 (base={args.base}, n_grid=64)...")
    t2 = time.time()
    tau_star, alpha, beta, residual = estimate_tau_from_distance_prior(
        D_hist, base=args.base, n_grid=64, max_delta=args.max_delta
    )
    elapsed = time.time() - t2
    print(f"  Computed in {elapsed:.3f}s")

    # Results
    print(f"\n{'='*60}")
    print(f"  ALGORITHM 1 RESULTS")
    print(f"{'='*60}")
    print(f"  τ* = {tau_star:.6f}")
    print(f"  α  = {alpha:.6e}  (diagonal ridge)")
    print(f"  β  = {beta:.6e}  (off-diagonal coupling)")
    print(f"  β/α = {beta/alpha:.6f}" if alpha > 0 else "  β/α = N/A (α ≤ 0)")
    print(f"  Residual ‖K - K_approx‖_F / ‖K‖_F = {residual:.4f}  ({residual*100:.1f}%)")
    print(f"\n  Interpretation:")
    if tau_star > 1.5:
        print(f"    τ* > 1.5 → data favours heavy high-frequency allocation")
    elif tau_star > 0.5:
        print(f"    τ* ∈ [0.5, 1.5] → moderate non-geometric allocation")
    elif tau_star > 0.1:
        print(f"    τ* ∈ [0.1, 0.5] → near-geometric, slight high-freq bias")
    else:
        print(f"    τ* ≈ 0 → geometric allocation is near-optimal")

    # Also compute for base=10000 for comparison
    tau_10k, alpha_10k, beta_10k, res_10k = estimate_tau_from_distance_prior(
        D_hist, base=10000.0, n_grid=64, max_delta=args.max_delta
    )
    print(f"\n  Comparison with base=10000: τ*={tau_10k:.6f} (α={alpha_10k:.2e}, β={beta_10k:.2e}, res={res_10k:.4f})")

    result = {
        "algorithm": "Algorithm 1 (data-driven τ estimation)",
        "dataset": args.dataset,
        "max_delta": args.max_delta,
        "base": args.base,
        "n_tokens": len(token_ids),
        "sample_size": args.sample_size,
        "tau_star": round(tau_star, 6),
        "alpha": alpha,
        "beta": beta,
        "beta_over_alpha": beta / alpha if alpha > 0 else None,
        "relative_residual": round(residual, 6),
        "D_hist_top10": {str(i.item()+1): round(v.item(), 8)
                        for v, i in zip(top_vals, top_idx)},
        "comparison_base10000": {
            "tau_star": round(tau_10k, 6),
            "alpha": alpha_10k,
            "beta": beta_10k,
            "residual": round(res_10k, 6),
        },
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved to {out_path}")
    else:
        print(f"\n  JSON result:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
