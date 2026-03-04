#!/usr/bin/env python3
"""Token Identity Test — distance-swept positional information propagation.

Tests whether a model can propagate token identity information across distance Δ
using a primed-vs-unprimed design that cancels out token frequency bias.

For each trial at distance Δ:
  - Primed:   [filler] MARKER [filler_Δ] MARKER   (marker at Δ AND at end)
  - Unprimed: [filler] [filler_Δ+1]      MARKER   (marker only at end)

  delta_NLL = NLL_unprimed(MARKER_end) - NLL_primed(MARKER_end)

  - Positive: seeing MARKER at distance Δ reduces end-NLL → info propagates
  - Zero: model can't reach back to distance Δ → collision/extrapolation failure

This directly measures how much positional encoding enables information flow
over distance, without requiring any "task understanding" from the model.

Four curves: Geo, EVQ, Geo+YaRN, EVQ+YaRN

Usage:
    python eval_token_identity.py \
        --geo_dir /path/to/350m_tau0.00_seed42 \
        --evq_dir /path/to/350m_tau1.50_seed42 \
        --tier 350m --train_len 2048 --base 500000 \
        --num_trials 200 --batch_size 16
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import GPT, TIER_CONFIGS, get_device_and_dtype, load_val
from eval_pe_baselines import build_yarn_inv_freq, _swap_inv_freq, _restore_inv_freq

DEVICE, DTYPE = get_device_and_dtype()
USE_AUTOCAST = DEVICE == "cuda" and DTYPE != torch.float32

# Pool of rare marker tokens to randomize across trials
# (avoids any single-token bias in the model)
MARKER_POOL = [
    31822, 31906, 32011, 32099, 32145, 32200, 32301, 32400,
    32500, 32600, 32700, 32800, 32900, 33000, 33100, 33200,
    33300, 33400, 33500, 33600, 33700, 33800, 33900, 34000,
]


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------

def build_primed_pair(
    filler_tokens: torch.Tensor,
    marker_id: int,
    total_length: int,
    distance: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a primed/unprimed sequence pair.

    Primed:   [filler_before] MARKER [filler_between] MARKER
    Unprimed: [filler_before] FILLER [filler_between] MARKER

    Both sequences are identical except position (total_length - 1 - distance)
    has MARKER (primed) vs original filler (unprimed).

    Returns:
        (primed_ids, unprimed_ids) — both shape (total_length,)
    """
    # Start with filler
    if len(filler_tokens) < total_length:
        # Repeat filler if needed
        repeats = (total_length // len(filler_tokens)) + 1
        filler = filler_tokens.repeat(repeats)[:total_length]
    else:
        filler = filler_tokens[:total_length].clone()

    unprimed = filler.clone()
    primed = filler.clone()

    # Place MARKER at end (both sequences)
    primed[-1] = marker_id
    unprimed[-1] = marker_id

    # Place MARKER at distance Δ from end (primed only)
    prime_pos = max(0, total_length - 1 - distance)
    primed[prime_pos] = marker_id
    # unprimed keeps original filler at prime_pos

    return primed, unprimed


# ---------------------------------------------------------------------------
# Batched NLL at final position (reuses split-lm_head optimization)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _batched_final_nll(
    model: GPT,
    ctx,
    batch_ids: torch.Tensor,   # (B, T)
    target_token: torch.Tensor,  # (B,) or scalar
) -> List[float]:
    """Compute NLL of target token at the final prediction position.

    Runs transformer backbone on full batch, then applies lm_head only
    at the last position (1 position per sequence).
    """
    B, T = batch_ids.shape
    batch_ids = batch_ids.to(DEVICE)
    if target_token.dim() == 0:
        target_token = target_token.expand(B)
    target_token = target_token.to(DEVICE)

    with ctx:
        x = model.emb(batch_ids[:, :-1])
        for b in model.blocks:
            x = b(x)
        hidden = model.ln(x)  # (B, T-1, hidden_dim)

    # lm_head at last position: predict token at position T-1
    h_last = hidden[:, -1, :]  # (B, hidden_dim)
    logits = model.head(h_last).float()  # (B, V)

    log_probs = F.log_softmax(logits, dim=-1)
    nlls = -log_probs.gather(1, target_token.unsqueeze(1)).squeeze(1)

    del hidden, logits, log_probs
    return nlls.cpu().tolist()


# ---------------------------------------------------------------------------
# Single model evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_identity_single_model(
    model: GPT,
    inv_freq_orig: torch.Tensor,
    head_dim: int,
    train_len: int,
    base: float,
    filler_tokens: torch.Tensor,
    eval_length: int,
    distances: List[int],
    num_trials: int,
    batch_size: int = 16,
    apply_yarn: bool = False,
    seed: int = 42,
) -> Dict:
    """Evaluate token identity discrimination across distances."""
    model.eval()
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()

    scale = eval_length / train_len
    if apply_yarn and scale > 1.0:
        yarn_inv = build_yarn_inv_freq(inv_freq_orig, head_dim, scale)
        _swap_inv_freq(model, yarn_inv, eval_length)
    else:
        model.extend_rope(eval_length + 100)

    rng = random.Random(seed)
    results = {}

    for dist in distances:
        dist_ratio = dist / train_len

        # Pre-build all trial pairs
        primed_seqs = []
        unprimed_seqs = []
        marker_ids = []

        for t in range(num_trials):
            # Random filler offset
            offset = rng.randint(0, max(1, len(filler_tokens) - eval_length - 1))
            filler_slice = filler_tokens[offset: offset + eval_length]
            if len(filler_slice) < eval_length:
                filler_slice = filler_tokens[:eval_length]

            # Random marker from pool
            marker = rng.choice(MARKER_POOL)

            primed, unprimed = build_primed_pair(
                filler_slice, marker, eval_length, dist
            )
            primed_seqs.append(primed)
            unprimed_seqs.append(unprimed)
            marker_ids.append(marker)

        # Batch process: interleave [primed_0, unprimed_0, primed_1, unprimed_1, ...]
        nll_primed_all = []
        nll_unprimed_all = []

        for batch_start in range(0, num_trials, batch_size):
            batch_end = min(batch_start + batch_size, num_trials)

            all_ids = []
            all_targets = []
            for i in range(batch_start, batch_end):
                all_ids.append(primed_seqs[i])
                all_ids.append(unprimed_seqs[i])
                all_targets.append(marker_ids[i])
                all_targets.append(marker_ids[i])

            batch_tensor = torch.stack(all_ids, dim=0)
            target_tensor = torch.tensor(all_targets, dtype=torch.long)
            nlls = _batched_final_nll(model, ctx, batch_tensor, target_tensor)

            for j in range(batch_end - batch_start):
                nll_primed_all.append(nlls[2 * j])
                nll_unprimed_all.append(nlls[2 * j + 1])

        nll_p = np.array(nll_primed_all)
        nll_u = np.array(nll_unprimed_all)
        delta = nll_u - nll_p  # positive = priming helps = info propagates

        results[f"D={dist}_R={dist_ratio:.2f}x"] = {
            "distance": dist,
            "distance_ratio": round(dist_ratio, 2),
            "mean_nll_primed": round(float(nll_p.mean()), 4),
            "mean_nll_unprimed": round(float(nll_u.mean()), 4),
            "delta_nll": round(float(delta.mean()), 4),
            "delta_std": round(float(delta.std()), 4),
            "delta_positive_frac": round(float((delta > 0).mean()), 4),
            "num_trials": num_trials,
        }

        print(f"      Δ={dist:>6} ({dist_ratio:.2f}×)  "
              f"primed={nll_p.mean():.3f}  unprimed={nll_u.mean():.3f}  "
              f"δ={delta.mean():+.4f} ±{delta.std():.4f}  "
              f"pos={100*(delta>0).mean():.0f}%")

    if apply_yarn:
        _restore_inv_freq(model, inv_freq_orig, train_len)

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(results: Dict, train_len: int) -> Dict:
    points = []
    for key, val in sorted(results.items(), key=lambda x: x[1]["distance"]):
        points.append((val["distance_ratio"], val["delta_nll"]))

    extrap = [(r, d) for r, d in points if r >= 1.0]
    auc = 0.0
    if len(extrap) >= 2:
        for i in range(len(extrap) - 1):
            r0, d0 = extrap[i]
            r1, d1 = extrap[i + 1]
            auc += (r1 - r0) * (d0 + d1) / 2
        r_range = extrap[-1][0] - extrap[0][0]
        if r_range > 0:
            auc /= r_range

    # Collapse point: where delta_nll first drops to ≤ 0
    collapse = None
    for ratio, delta in points:
        if delta <= 0 and ratio >= 0.5:
            collapse = ratio
            break

    return {
        "auc_delta_extrap": round(auc, 4),
        "collapse_point": collapse,
        "points": points,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Token Identity Test — primed vs unprimed NLL"
    )
    parser.add_argument("--geo_dir", type=str, required=True)
    parser.add_argument("--evq_dir", type=str, required=True)
    parser.add_argument("--tier", choices=["50m", "125m", "350m", "500m"],
                        required=True)
    parser.add_argument("--train_len", type=int, default=2048)
    parser.add_argument("--base", type=float, default=500000.0)
    parser.add_argument("--eval_multiplier", type=int, default=4,
                        help="L_eval = multiplier × L_train (default: 4)")
    parser.add_argument("--distance_ratios", type=str,
                        default="0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0",
                        help="Comma-separated Δ/L_train ratios")
    parser.add_argument("--num_trials", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="fineweb-edu",
                        choices=["fineweb-edu", "tinystories"])
    args = parser.parse_args()

    eval_length = args.train_len * args.eval_multiplier
    distance_ratios = [float(x) for x in args.distance_ratios.split(",")]
    distances = [int(r * args.train_len) for r in distance_ratios]
    cfg = TIER_CONFIGS[args.tier].copy()
    head_dim = cfg["head_dim"]

    print(f"\n{'#'*60}")
    print(f"  Token Identity Test (Primed vs Unprimed)")
    print(f"  tier={args.tier}  train_len={args.train_len}  eval_len={eval_length}")
    print(f"  distances (tokens): {distances}")
    print(f"  distance ratios: {distance_ratios}")
    print(f"  marker pool size: {len(MARKER_POOL)}")
    print(f"  num_trials={args.num_trials}  batch_size={args.batch_size}")
    print(f"  seed={args.seed}")
    print(f"{'#'*60}\n")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    print(f"  Sample markers: {[tok.decode([m]) for m in MARKER_POOL[:5]]}")

    cache_dir = str(Path(args.geo_dir).parent)
    filler_tokens = load_val(tok, dataset=args.dataset, cache_dir=cache_dir)

    configs = [
        ("Geo",          args.geo_dir, False),
        ("EVQ",          args.evq_dir, False),
        ("Geo+YaRN",     args.geo_dir, True),
        ("EVQ+YaRN",     args.evq_dir, True),
    ]

    all_results = {
        "metadata": {
            "tier": args.tier,
            "train_len": args.train_len,
            "eval_length": eval_length,
            "base": args.base,
            "distance_ratios": distance_ratios,
            "num_trials": args.num_trials,
            "batch_size": args.batch_size,
            "marker_pool": MARKER_POOL,
            "seed": args.seed,
            "device": DEVICE,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "curves": {},
    }

    t_total = time.time()

    for label, ckpt_dir, use_yarn in configs:
        ckpt_path = Path(ckpt_dir)
        model_path = ckpt_path / "model.pt"
        inv_freq_path = ckpt_path / "inv_freq.npy"

        if not model_path.exists():
            print(f"  ERROR: {model_path} not found, skipping {label}")
            continue

        print(f"\n  {'='*50}")
        print(f"  {label}  (ckpt: {ckpt_dir})")
        print(f"  {'='*50}")

        inv_freq = np.load(str(inv_freq_path))
        inv_freq_t = torch.from_numpy(inv_freq)

        eval_cfg = cfg.copy()
        eval_cfg["seq_len"] = args.train_len
        eval_cfg["max_position_embeddings"] = args.train_len

        model = GPT(eval_cfg, inv_freq_t).to(DEVICE)
        state = torch.load(str(model_path), map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        t0 = time.time()
        curve = eval_identity_single_model(
            model=model,
            inv_freq_orig=inv_freq_t,
            head_dim=head_dim,
            train_len=args.train_len,
            base=args.base,
            filler_tokens=filler_tokens,
            eval_length=eval_length,
            distances=distances,
            num_trials=args.num_trials,
            batch_size=args.batch_size,
            apply_yarn=use_yarn,
            seed=args.seed,
        )
        elapsed = time.time() - t0

        summary = compute_summary(curve, args.train_len)
        all_results["curves"][label] = {
            "results": curve,
            "summary": summary,
            "eval_time_sec": round(elapsed, 1),
        }

        print(f"\n    AUC(δ): {summary['auc_delta_extrap']:.4f}  "
              f"collapse: {summary['collapse_point']}")
        print(f"    Time: {elapsed:.0f}s")

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    total_time = time.time() - t_total
    all_results["metadata"]["total_time_min"] = round(total_time / 60, 1)
    all_results["metadata"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.geo_dir).parent / "token_identity_results.json"

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    _print_comparison(all_results, args.train_len)

    print(f"\n{'='*60}")
    print(f"  TOKEN IDENTITY TEST COMPLETE  |  {total_time / 60:.1f} min total")
    print(f"  Results: {out_path}")
    print(f"{'='*60}")


def _print_comparison(all_results: Dict, train_len: int) -> None:
    curves = all_results.get("curves", {})
    if not curves:
        return

    all_dists = set()
    for c in curves.values():
        for key, val in c.get("results", {}).items():
            all_dists.add(val["distance"])
    all_dists = sorted(all_dists)

    print(f"\n  {'─'*85}")
    print(f"  Token Identity: δNLL (primed advantage) by Distance")
    print(f"  positive = info propagates over Δ; zero/negative = position encoding fails")
    print(f"  {'─'*85}")

    header = f"  {'Method':>15}"
    for d in all_dists:
        ratio = d / train_len
        header += f"  {ratio:.2f}×"
    header += "    AUC  collapse"
    print(header)

    for label, data in curves.items():
        row = f"  {label:>15}"
        results = data.get("results", {})
        for d in all_dists:
            for key, val in results.items():
                if val["distance"] == d:
                    delta = val["delta_nll"]
                    row += f"  {delta:+.3f}"
                    break
            else:
                row += f"  {'—':>6}"

        summary = data.get("summary", {})
        auc = summary.get("auc_delta_extrap", 0)
        cp = summary.get("collapse_point")
        cp_str = f"{cp:.1f}×" if cp is not None else "none"
        row += f"  {auc:+.4f}  {cp_str}"
        print(row)

    print(f"  {'─'*85}")


if __name__ == "__main__":
    main()
