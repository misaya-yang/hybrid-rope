#!/usr/bin/env python3
"""
Phase 11 YaRN Post-Eval: Load trained checkpoints, apply YaRN scaling, re-evaluate PPL.
Tests EVQ+YaRN synergy vs Geo+YaRN at various extrapolation ratios.

Multi-scale: for each eval length, test with multiple YaRN scales to find optimal.
"""

import json, math, os, sys, gc
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import numpy as np

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import GPT, DEVICE, DTYPE, USE_AUTOCAST
from phase11_L256_extrap import (
    geometric_inv_freq, evq_cosh_inv_freq, hybrid_evq_inv_freq,
    CFG_350M, EVAL_LENGTHS,
)

WORK = Path("/root/autodl-tmp/evq_phase11_L256")
TRAIN_LEN = 256
BASE = 500_000.0
DIM = 64

# YaRN scales to test — from exact match to over-scaled
YARN_SCALES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def build_yarn_inv_freq(base_inv, head_dim, scale):
    """YaRN progressive frequency scaling with smoothstep ramp.

    Follows the standard YaRN formula:
    - Low-freq dims (position-encoding): scale normally (divide by scale)
    - High-freq dims (local): leave untouched
    - Middle dims: smoothstep interpolation
    - Attention temperature correction
    """
    if scale <= 1.0:
        return base_inv.clone()
    K = head_dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    # YaRN ramp parameters (standard: beta_fast=32, beta_slow=1 → 20%-90% range)
    start = int(0.20 * K)
    end = int(0.90 * K)
    if end <= start:
        end = min(K - 1, start + 1)
    ramp = torch.clamp((idx - start) / float(max(1, end - start)), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep
    # Temperature correction
    temperature = 1.0 + 0.07 * math.log2(scale)
    yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
    return (base_inv.double() / yarn_scale).float()


def build_ntk_aware_inv_freq(base_inv, head_dim, base, scale):
    """NTK-Aware: scaled_base = base * scale^(d/(d-2)), recompute geometric."""
    if scale <= 1.0:
        return base_inv.clone()
    d = head_dim
    scaled_base = base * (scale ** (d / (d - 2)))
    K = d // 2
    idx = torch.arange(K, dtype=torch.float64)
    inv = 1.0 / (scaled_base ** (2.0 * idx / d))
    return inv.float()


def eval_ppl_single(model, val_data, inv_freq, L, n_chunks=8):
    """Evaluate PPL at single length after swapping inv_freq."""
    for block in model.blocks:
        block.attn.rope.inv_freq.copy_(inv_freq)
    model.blocks[0].attn.rope._build(L + 100)

    model.eval()
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    rng = np.random.RandomState(9999)

    max_start = len(val_data) - L
    if max_start <= 0:
        return None
    offsets = sorted(rng.choice(max_start, size=min(n_chunks, max_start // L), replace=False))
    losses = []
    for offset in offsets:
        chunk = val_data[offset:offset + L].unsqueeze(0).to(DEVICE)
        try:
            with torch.no_grad(), ctx:
                logits = model(chunk[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       chunk[:, 1:].reshape(-1))
            losses.append(loss.item())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
            raise
        finally:
            del chunk
    if losses:
        return round(math.exp(sum(losses) / len(losses)), 3)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,137,256")
    parser.add_argument("--methods", default="geo,evq2.0,evq4.0")
    parser.add_argument("--r", type=int, default=16)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    methods = args.methods.split(",")

    val_path = Path("/root/autodl-tmp/evq_phase9/data/val_fineweb-edu_5000000.pt")
    print(f"Loading val data from {val_path}...")
    val_data = torch.load(val_path, weights_only=True)

    cfg = CFG_350M.copy()
    cfg["seq_len"] = TRAIN_LEN
    cfg["max_position_embeddings"] = TRAIN_LEN

    # Build base inv_freqs
    base_inv_freqs = {}
    for m in methods:
        if m == "geo":
            base_inv_freqs[m] = geometric_inv_freq()
        elif m.startswith("evq"):
            tau = float(m.replace("evq", ""))
            base_inv_freqs[m] = evq_cosh_inv_freq(tau=tau)
        elif m.startswith("hybrid"):
            tau = float(m.replace("hybrid", ""))
            base_inv_freqs[m] = hybrid_evq_inv_freq(tau=tau, r=args.r)

    all_results = {}

    for method in methods:
        for seed in seeds:
            run_id = f"350m_{method}_seed{seed}"
            model_path = WORK / run_id / "model.pt"

            if not model_path.exists():
                print(f"\n  SKIP {run_id} (no checkpoint)")
                continue

            result_path = WORK / run_id / "yarn_multi_scale.json"
            if result_path.exists():
                print(f"\n  SKIP {run_id} (yarn eval exists)")
                with open(result_path) as f:
                    all_results[run_id] = json.load(f)
                continue

            print(f"\n{'='*70}")
            print(f"  YARN MULTI-SCALE EVAL: {run_id}")
            print(f"{'='*70}")

            base_inv = base_inv_freqs[method]
            model = GPT(cfg, base_inv).to(DEVICE)
            state = torch.load(model_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state)

            result = {"run_id": run_id, "method": method, "seed": seed}

            # 1. Raw PPL (no scaling) at all lengths
            print(f"\n  [raw] No scaling:")
            raw_ppl = {}
            for L in EVAL_LENGTHS:
                ppl = eval_ppl_single(model, val_data, base_inv, L)
                if ppl:
                    raw_ppl[str(L)] = ppl
                    print(f"    L={L:>5d}: PPL={ppl:.2f}")
            result["raw"] = raw_ppl

            # 2. YaRN at multiple scales for each eval length
            print(f"\n  [yarn] Multi-scale:")
            yarn_results = {}
            for L in EVAL_LENGTHS:
                if L <= TRAIN_LEN:
                    continue
                best_ppl = None
                best_scale = None
                scale_ppls = {}
                for scale in YARN_SCALES:
                    if scale < 1.0:
                        continue
                    yarn_inv = build_yarn_inv_freq(base_inv, DIM, scale)
                    ppl = eval_ppl_single(model, val_data, yarn_inv, L)
                    if ppl:
                        scale_ppls[f"s{scale:.0f}"] = ppl
                        if best_ppl is None or ppl < best_ppl:
                            best_ppl = ppl
                            best_scale = scale
                yarn_results[str(L)] = {
                    "scales": scale_ppls,
                    "best_ppl": best_ppl,
                    "best_scale": best_scale,
                }
                print(f"    L={L:>5d}: {scale_ppls}  best=s{best_scale}({best_ppl:.1f})")
            result["yarn_multi"] = yarn_results

            # 3. YaRN with auto scale (L/L_train) — the standard approach
            print(f"\n  [yarn_auto] scale = L/L_train:")
            yarn_auto = {}
            for L in EVAL_LENGTHS:
                if L <= TRAIN_LEN:
                    ppl = eval_ppl_single(model, val_data, base_inv, L)
                else:
                    scale = L / TRAIN_LEN
                    yarn_inv = build_yarn_inv_freq(base_inv, DIM, scale)
                    ppl = eval_ppl_single(model, val_data, yarn_inv, L)
                if ppl:
                    yarn_auto[str(L)] = ppl
                    print(f"    L={L:>5d} (scale={L/TRAIN_LEN:>5.1f}×): PPL={ppl:.2f}")
            result["yarn_auto"] = yarn_auto

            # 4. NTK-Aware for comparison
            print(f"\n  [ntk] NTK-Aware, scale = L/L_train:")
            ntk_results = {}
            for L in EVAL_LENGTHS:
                if L <= TRAIN_LEN:
                    ppl = eval_ppl_single(model, val_data, base_inv, L)
                else:
                    scale = L / TRAIN_LEN
                    ntk_inv = build_ntk_aware_inv_freq(base_inv, DIM, BASE, scale)
                    ppl = eval_ppl_single(model, val_data, ntk_inv, L)
                if ppl:
                    ntk_results[str(L)] = ppl
                    print(f"    L={L:>5d} (scale={L/TRAIN_LEN:>5.1f}×): PPL={ppl:.2f}")
            result["ntk_auto"] = ntk_results

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            all_results[run_id] = result

            del model, state
            gc.collect()
            torch.cuda.empty_cache()

    # Save aggregate
    agg_path = WORK / "yarn_multi_scale_results.json"
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Print comprehensive summary ────────────────────────────────────
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"  COMPREHENSIVE RESULTS: L_train={TRAIN_LEN}, 350M, 3-seed mean")
    print(f"{sep}")

    # Compute 3-seed means
    def mean_ppl(method, key, L):
        vals = []
        for s in seeds:
            r = all_results.get(f"350m_{method}_seed{s}", {})
            v = r.get(key, {}).get(str(L))
            if v is not None:
                vals.append(v)
        return sum(vals) / len(vals) if vals else None

    header = f"  {'Method':>12s} {'Scale':>8s}"
    for L in EVAL_LENGTHS:
        header += f" {'L='+str(L):>8s}"
    print(header)
    print("  " + "-" * 90)

    for method in methods:
        # Raw
        raw_line = f"  {method:>12s} {'raw':>8s}"
        for L in EVAL_LENGTHS:
            m = mean_ppl(method, "raw", L)
            raw_line += f" {m:>8.1f}" if m else f" {'--':>8s}"
        print(raw_line)

        # YaRN auto
        yarn_line = f"  {method:>12s} {'yarn':>8s}"
        for L in EVAL_LENGTHS:
            m = mean_ppl(method, "yarn_auto", L)
            yarn_line += f" {m:>8.1f}" if m else f" {'--':>8s}"
        print(yarn_line)

        # NTK
        ntk_line = f"  {method:>12s} {'ntk':>8s}"
        for L in EVAL_LENGTHS:
            m = mean_ppl(method, "ntk_auto", L)
            ntk_line += f" {m:>8.1f}" if m else f" {'--':>8s}"
        print(ntk_line)
        print()

    # Delta table: EVQ vs Geo
    print(f"\n  DELTA vs GEO (raw → raw, yarn → yarn):")
    for method in methods:
        if method == "geo":
            continue
        for key in ["raw", "yarn_auto", "ntk_auto"]:
            label = f"{method}+{key.split('_')[0]}"
            line = f"  {label:>18s}:"
            for L in EVAL_LENGTHS:
                geo_m = mean_ppl("geo", key, L)
                evq_m = mean_ppl(method, key, L)
                if geo_m and evq_m:
                    delta = (evq_m / geo_m - 1) * 100
                    line += f" {delta:>+7.1f}%"
                else:
                    line += f" {'--':>8s}"
            print(line)
        print()

    # YaRN improvement over raw
    print(f"\n  YARN IMPROVEMENT (yarn_auto PPL / raw PPL - 1):")
    for method in methods:
        line = f"  {method:>12s}:"
        for L in EVAL_LENGTHS:
            raw_m = mean_ppl(method, "raw", L)
            yarn_m = mean_ppl(method, "yarn_auto", L)
            if raw_m and yarn_m and L > TRAIN_LEN:
                delta = (yarn_m / raw_m - 1) * 100
                line += f" {delta:>+7.1f}%"
            else:
                line += f" {'--':>8s}"
        print(line)

    print(f"\n  Results saved to {agg_path}")


if __name__ == "__main__":
    main()
