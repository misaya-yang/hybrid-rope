#!/usr/bin/env python3
"""DSR v2: Distance-Swept Retrieval — mechanism-aligned evaluation.

Measures retrieval accuracy as a function of query-needle *distance* (not
sequence length), isolating the RoPE extrapolation effect from length confounds.

Setup:
  - Fixed L_eval = 8 × L_train (default 16384 for L_train=2048)
  - Query always at the end
  - Needle placed at position p = L_eval - Δ  (distance = Δ)
  - Δ/L_train ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0}

Four curves:
  1. Geo           (τ=0, raw frequencies)
  2. EVQ           (τ=1.5, raw frequencies)
  3. Geo + YaRN    (τ=0, per-length YaRN scaling)
  4. EVQ + YaRN    (τ=1.5, per-length YaRN scaling)

Output:
  - Accuracy(Δ) curves (retrieval rate vs distance)
  - AUC over normalized distance [1×, 8×]
  - Break-point Δ_50 (first distance where accuracy < 50%)

Usage:
    python eval_dsr.py \
        --geo_dir  /path/to/350m_tau0.00_seed42 \
        --evq_dir  /path/to/350m_tau1.50_seed42 \
        --tier 350m --train_len 2048 --base 500000 \
        --num_trials 50

    # Quick test (10 trials):
    python eval_dsr.py \
        --geo_dir ... --evq_dir ... --tier 350m --num_trials 10
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import GPT, TIER_CONFIGS, get_device_and_dtype, load_val
from eval_pe_baselines import build_yarn_inv_freq, _swap_inv_freq, _restore_inv_freq
from eval_passkey_scratch import (
    build_passkey_eval_sequence,
    _compute_nll_at_positions,
    _autoregressive_probe,
    PASSKEY_PREFIX,
    PASSKEY_SUFFIX,
    DASH,
)

DEVICE, DTYPE = get_device_and_dtype()
USE_AUTOCAST = DEVICE == "cuda" and DTYPE != torch.float32


# ---------------------------------------------------------------------------
# DSR core: build sequence with needle at specific distance from query
# ---------------------------------------------------------------------------

def build_dsr_sequence(
    filler_tokens: torch.Tensor,
    passkey: str,
    tokenizer,
    total_length: int,
    distance: int,
) -> Tuple[torch.Tensor, int, int]:
    """Build a DSR eval sequence with needle at a specific distance from query.

    Layout::
        [filler_before] <<PASS:{passkey}>> [filler_after] <<PASS:

    The probe (query) is always at the end. The needle is placed such that
    the token distance from needle to probe equals `distance`.

    Args:
        filler_tokens: 1-D tensor of filler token IDs.
        passkey: Dash-separated digit string, e.g. "7-4-2-9-1".
        tokenizer: HuggingFace tokenizer.
        total_length: Total sequence length in tokens.
        distance: Desired token distance from needle start to probe start.

    Returns:
        (input_ids, passkey_start, probe_start)
    """
    # Convert distance to depth_percent:
    # probe is at the end, needle at (total_length - distance) from start
    # depth_percent = needle_position / filler_budget
    full_marker_text = f"{PASSKEY_PREFIX}{passkey}{PASSKEY_SUFFIX}"
    probe_text = PASSKEY_PREFIX
    full_marker_ids = tokenizer.encode(full_marker_text, add_special_tokens=False)
    probe_ids = tokenizer.encode(probe_text, add_special_tokens=False)
    filler_budget = total_length - len(full_marker_ids) - len(probe_ids)

    # needle_start should be at: total_length - len(probe_ids) - distance
    # = filler_before_len (since seq = before + marker + after + probe)
    # filler_before_len = total_length - len(probe_ids) - distance - len(full_marker_ids) ... wait no
    # Actually: probe_start = before_len + len(full_marker_ids) + after_len
    # distance = probe_start - passkey_start = len(full_marker_ids) + after_len
    # So: after_len = distance - len(full_marker_ids)
    after_len = max(0, distance - len(full_marker_ids))
    before_len = filler_budget - after_len

    if before_len < 0:
        # distance too large for this total_length, clip
        before_len = 0
        after_len = filler_budget

    # Use depth_percent to delegate to existing function
    depth = before_len / max(filler_budget, 1)
    depth = max(0.0, min(1.0, depth))

    return build_passkey_eval_sequence(
        filler_tokens, passkey, tokenizer, total_length, depth
    )


def make_random_passkey(rng: random.Random) -> str:
    """Generate a random 5-digit dash-separated passkey."""
    digits = [str(rng.randint(0, 9)) for _ in range(5)]
    return DASH.join(digits)


def make_wrong_passkey(correct: str, rng: random.Random) -> str:
    """Generate a wrong passkey differing in every digit."""
    digits = correct.split(DASH)
    wrong = []
    for d in digits:
        candidates = [str(x) for x in range(10) if str(x) != d]
        wrong.append(rng.choice(candidates))
    return DASH.join(wrong)


# ---------------------------------------------------------------------------
# Single model evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_dsr_single_model(
    model: GPT,
    inv_freq_orig: torch.Tensor,
    head_dim: int,
    train_len: int,
    base: float,
    filler_tokens: torch.Tensor,
    tokenizer,
    eval_length: int,
    distances: List[int],
    num_trials: int,
    apply_yarn: bool = False,
    seed: int = 42,
) -> Dict:
    """Evaluate one model across all distances.

    Returns:
        Dict with per-distance retrieval rates and summary stats.
    """
    model.eval()
    from contextlib import nullcontext
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()

    # Apply YaRN if requested
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
        trial_results = []

        for t in range(num_trials):
            passkey = make_random_passkey(rng)
            wrong_passkey = make_wrong_passkey(passkey, rng)

            # Randomize filler offset for each trial
            filler_offset = rng.randint(0, max(1, len(filler_tokens) - eval_length))
            trial_filler = filler_tokens[filler_offset:]

            input_ids, pk_start, probe_start = build_dsr_sequence(
                trial_filler, passkey, tokenizer, eval_length, dist
            )

            # NLL gap scoring
            answer_ids = tokenizer.encode(
                f"{passkey}{PASSKEY_SUFFIX}", add_special_tokens=False
            )
            n_answer = len(answer_ids)

            nll_correct = _compute_nll_at_positions(
                model, ctx, input_ids, probe_start, n_answer
            )

            # Build wrong sequence for comparison
            wrong_ids = input_ids.clone()
            wrong_marker = tokenizer.encode(
                f"{PASSKEY_PREFIX}{wrong_passkey}{PASSKEY_SUFFIX}",
                add_special_tokens=False,
            )
            # Replace the original passkey with wrong one at pk_start
            for i, tok in enumerate(wrong_marker):
                if pk_start + i < len(wrong_ids):
                    wrong_ids[pk_start + i] = tok

            nll_wrong = _compute_nll_at_positions(
                model, ctx, wrong_ids, probe_start, n_answer
            )

            nll_gap = nll_wrong - nll_correct
            retrieved = nll_gap > 0

            trial_results.append({
                "trial": t,
                "passkey": passkey,
                "nll_correct": round(nll_correct, 4),
                "nll_wrong": round(nll_wrong, 4),
                "nll_gap": round(nll_gap, 4),
                "retrieved": bool(retrieved),
            })

        # Aggregate
        retrieval_rate = sum(1 for r in trial_results if r["retrieved"]) / num_trials
        mean_gap = np.mean([r["nll_gap"] for r in trial_results])

        results[f"D={dist}_R={dist_ratio:.1f}x"] = {
            "distance": dist,
            "distance_ratio": round(dist_ratio, 2),
            "retrieval_rate": round(retrieval_rate, 4),
            "mean_nll_gap": round(float(mean_gap), 4),
            "num_trials": num_trials,
            "trials": trial_results,
        }

        print(f"      Δ={dist:>6} ({dist_ratio:.1f}×)  "
              f"acc={retrieval_rate*100:5.1f}%  gap={mean_gap:+.3f}")

    # Restore original frequencies
    if apply_yarn:
        _restore_inv_freq(model, inv_freq_orig, train_len)

    return results


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

def compute_summary(results: Dict, train_len: int) -> Dict:
    """Compute AUC and break-point from distance-swept results."""
    points = []
    for key, val in sorted(results.items(), key=lambda x: x[1]["distance"]):
        ratio = val["distance_ratio"]
        rate = val["retrieval_rate"]
        points.append((ratio, rate))

    # AUC over extrapolation region (ratio >= 1.0)
    extrap_points = [(r, a) for r, a in points if r >= 1.0]
    auc = 0.0
    if len(extrap_points) >= 2:
        for i in range(len(extrap_points) - 1):
            r0, a0 = extrap_points[i]
            r1, a1 = extrap_points[i + 1]
            auc += (r1 - r0) * (a0 + a1) / 2  # trapezoidal
        # Normalize by range
        r_range = extrap_points[-1][0] - extrap_points[0][0]
        if r_range > 0:
            auc /= r_range

    # Break-point: first ratio where accuracy < 50%
    breakpoint_50 = None
    breakpoint_70 = None
    for ratio, rate in points:
        if breakpoint_70 is None and rate < 0.70:
            breakpoint_70 = ratio
        if breakpoint_50 is None and rate < 0.50:
            breakpoint_50 = ratio

    return {
        "auc_extrap": round(auc, 4),
        "breakpoint_50": breakpoint_50,
        "breakpoint_70": breakpoint_70,
        "points": points,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DSR v2: Distance-Swept Retrieval evaluation"
    )
    parser.add_argument("--geo_dir", type=str, required=True,
                        help="Checkpoint dir for Geometric (τ=0) model")
    parser.add_argument("--evq_dir", type=str, required=True,
                        help="Checkpoint dir for EVQ (τ=1.5) model")
    parser.add_argument("--tier", choices=["50m", "125m", "350m", "500m"],
                        required=True)
    parser.add_argument("--train_len", type=int, default=2048)
    parser.add_argument("--base", type=float, default=500000.0)
    parser.add_argument("--eval_multiplier", type=int, default=8,
                        help="L_eval = eval_multiplier × L_train (default: 8)")
    parser.add_argument("--distance_ratios", type=str,
                        default="0.5,1.0,1.5,2.0,2.5,3.0,4.0,6.0,8.0",
                        help="Comma-separated Δ/L_train ratios")
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: auto)")
    parser.add_argument("--dataset", type=str, default="fineweb-edu",
                        choices=["fineweb-edu", "tinystories"])
    args = parser.parse_args()

    eval_length = args.train_len * args.eval_multiplier
    distance_ratios = [float(x) for x in args.distance_ratios.split(",")]
    distances = [int(r * args.train_len) for r in distance_ratios]
    cfg = TIER_CONFIGS[args.tier].copy()
    head_dim = cfg["head_dim"]

    print(f"\n{'#'*60}")
    print(f"  DSR v2: Distance-Swept Retrieval")
    print(f"  tier={args.tier}  train_len={args.train_len}  eval_len={eval_length}")
    print(f"  distances (tokens): {distances}")
    print(f"  distance ratios: {distance_ratios}")
    print(f"  num_trials={args.num_trials}  seed={args.seed}")
    print(f"{'#'*60}\n")

    # Load tokenizer & filler data
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Use parent of geo_dir as cache
    cache_dir = str(Path(args.geo_dir).parent)
    filler_tokens = load_val(tok, dataset=args.dataset, cache_dir=cache_dir)

    # Define the 4 configurations
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
        print(f"  {label}")
        print(f"  {'='*50}")

        # Load model
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
        curve = eval_dsr_single_model(
            model=model,
            inv_freq_orig=inv_freq_t,
            head_dim=head_dim,
            train_len=args.train_len,
            base=args.base,
            filler_tokens=filler_tokens,
            tokenizer=tok,
            eval_length=eval_length,
            distances=distances,
            num_trials=args.num_trials,
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

        print(f"\n    AUC(extrap): {summary['auc_extrap']:.3f}  "
              f"Δ_50: {summary['breakpoint_50']}  "
              f"Δ_70: {summary['breakpoint_70']}")

        # Free GPU memory before loading next model
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Final save
    total_time = time.time() - t_total
    all_results["metadata"]["total_time_min"] = round(total_time / 60, 1)
    all_results["metadata"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.geo_dir).parent / "dsr_results.json"

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # Print comparison table
    _print_comparison(all_results, args.train_len)

    print(f"\n{'='*60}")
    print(f"  DSR COMPLETE  |  {total_time / 60:.1f} min total")
    print(f"  Results: {out_path}")
    print(f"{'='*60}")


def _print_comparison(all_results: Dict, train_len: int) -> None:
    """Print a formatted comparison table."""
    curves = all_results.get("curves", {})
    if not curves:
        return

    # Collect all distances
    all_dists = set()
    for c in curves.values():
        for key, val in c.get("results", {}).items():
            all_dists.add(val["distance"])
    all_dists = sorted(all_dists)

    print(f"\n  {'─'*70}")
    print(f"  DSR Comparison: Retrieval Rate (%) by Distance")
    print(f"  {'─'*70}")

    header = f"  {'Method':>15}"
    for d in all_dists:
        ratio = d / train_len
        header += f"  {ratio:.1f}×"
    header += "   AUC  Δ_50"
    print(header)

    for label, data in curves.items():
        row = f"  {label:>15}"
        results = data.get("results", {})
        for d in all_dists:
            found = False
            for key, val in results.items():
                if val["distance"] == d:
                    row += f"  {val['retrieval_rate']*100:4.0f}"
                    found = True
                    break
            if not found:
                row += f"  {'—':>4}"

        summary = data.get("summary", {})
        auc = summary.get("auc_extrap", 0)
        bp50 = summary.get("breakpoint_50")
        bp_str = f"{bp50:.1f}×" if bp50 is not None else ">8×"
        row += f"  {auc:.3f}  {bp_str}"
        print(row)

    print(f"  {'─'*70}")


if __name__ == "__main__":
    main()
