#!/usr/bin/env python3
"""Passkey retrieval evaluation for EVQ from-scratch models.

Measures positional retrieval accuracy at various context lengths and depths.
Loads checkpoints saved by run_evq_sweep.py.

Usage:
    python scripts/m4_evq_sweep/eval_passkey.py --work_dir ~/evq_sweep --tiers 50m,125m,500m
    python scripts/m4_evq_sweep/eval_passkey.py --work_dir ~/evq_sweep --tiers 500m --taus 0.0,1.5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import model & config from sweep script
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_evq_sweep import (
    GPT,
    TIER_CONFIGS,
    evq_cosh_inv_freq,
    get_device_and_dtype,
)

DEVICE, DTYPE = get_device_and_dtype()
USE_AUTOCAST = DEVICE == "cuda" and DTYPE != torch.float32

# ---------------------------------------------------------------------------
# Passkey sequence builder
# ---------------------------------------------------------------------------

# Number words that TinyStories models should know
NUMBER_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# Regex to detect number-word contamination in filler text
_NUM_PATTERN = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine)\b", re.IGNORECASE
)

PASSKEY_TEMPLATE = "The special number is {num}."
QUERY_TEMPLATE = " The special number is"


def _sanitize_filler(tokenizer, filler_ids: torch.Tensor, window: int) -> torch.Tensor:
    """Remove filler regions that contain number words to prevent contamination.

    Decodes the filler in chunks, replaces contaminated chunks with other
    filler content. Returns cleaned filler_ids of the same length.
    """
    chunk_size = 256
    total = len(filler_ids)
    ids_list = filler_ids.tolist()
    clean_ids: List[int] = []
    fallback_start = total // 2  # Use second half as fallback source

    for i in range(0, total, chunk_size):
        chunk = ids_list[i : i + chunk_size]
        text = tokenizer.decode(chunk, skip_special_tokens=True)
        if _NUM_PATTERN.search(text):
            # Replace with fallback filler (from a different region)
            fb = ids_list[fallback_start : fallback_start + len(chunk)]
            fallback_start += len(chunk)
            if fallback_start >= total:
                fallback_start = 0
            clean_ids.extend(fb)
        else:
            clean_ids.extend(chunk)

    return torch.tensor(clean_ids[: total], dtype=torch.long)


def build_passkey_sequence(
    tokenizer,
    filler_ids: torch.Tensor,
    context_length: int,
    depth_ratio: float,
    passkey_idx: int,
    seed: int,
) -> Tuple[torch.Tensor, List[int]]:
    """Build a passkey retrieval sequence.

    Args:
        tokenizer: Tokenizer instance
        filler_ids: 1D tensor of sanitized filler token IDs
        context_length: Total sequence length in tokens
        depth_ratio: Where to place the passkey (0.0=start, 1.0=end)
        passkey_idx: Index into NUMBER_WORDS for the passkey
        seed: Random seed for filler selection

    Returns:
        (input_ids, target_token_ids) where target_token_ids are the tokens
        that should follow the query template.
    """
    rng = random.Random(seed)

    num_word = NUMBER_WORDS[passkey_idx]
    passkey_text = PASSKEY_TEMPLATE.format(num=num_word)
    query_text = QUERY_TEMPLATE

    passkey_ids = tokenizer.encode(passkey_text, add_special_tokens=False)
    query_ids = tokenizer.encode(query_text, add_special_tokens=False)
    target_ids = tokenizer.encode(" " + num_word, add_special_tokens=False)

    # Available space for filler
    filler_needed = context_length - len(passkey_ids) - len(query_ids)
    if filler_needed < 10:
        raise ValueError(
            f"context_length={context_length} too short for passkey+query "
            f"({len(passkey_ids)}+{len(query_ids)} tokens)"
        )

    # Split filler into before/after passkey
    insert_pos = max(1, int(filler_needed * depth_ratio))
    before_len = insert_pos
    after_len = filler_needed - before_len

    # Sample filler from validation data
    total_filler = len(filler_ids)
    start = rng.randint(0, max(0, total_filler - filler_needed - 100))
    filler_chunk = filler_ids[start : start + filler_needed].tolist()

    # Pad if needed
    while len(filler_chunk) < filler_needed:
        filler_chunk.extend(filler_ids[: filler_needed - len(filler_chunk)].tolist())

    before = filler_chunk[:before_len]
    after = filler_chunk[before_len : before_len + after_len]

    # Assemble: [filler_before] [passkey] [filler_after] [query]
    seq = before + passkey_ids + after + query_ids
    seq = seq[:context_length]  # Trim to exact length

    return torch.tensor(seq, dtype=torch.long), target_ids


# ---------------------------------------------------------------------------
# Passkey evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_passkey_single(
    model: GPT,
    tokenizer,
    filler_ids: torch.Tensor,
    context_length: int,
    depth_ratio: float,
    num_trials: int = 50,
    base_seed: int = 12345,
) -> Dict[str, float]:
    """Run passkey retrieval at a specific context length and depth.

    Returns accuracy, mean_rank, mean_prob, and standard error.
    """
    model.eval()
    model.extend_rope(context_length + 100)

    correct = 0
    ranks = []
    probs = []

    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()

    for trial in range(num_trials):
        passkey_idx = trial % len(NUMBER_WORDS)
        seed = base_seed + trial * 1000 + context_length

        try:
            seq, target_ids = build_passkey_sequence(
                tokenizer, filler_ids, context_length, depth_ratio, passkey_idx, seed
            )
        except ValueError:
            continue

        input_ids = seq.unsqueeze(0).to(DEVICE)

        with ctx:
            logits = model(input_ids)  # (1, L, vocab)

        # Get logits at the last position (prediction for next token)
        last_logits = logits[0, -1, :]  # (vocab,)
        probs_dist = F.softmax(last_logits.float(), dim=-1)

        # Check if the first target token is predicted correctly
        target_token = target_ids[0]
        pred_token = last_logits.argmax().item()

        if pred_token == target_token:
            correct += 1

        # Rank of target token
        sorted_indices = last_logits.argsort(descending=True)
        rank = (sorted_indices == target_token).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
        probs.append(probs_dist[target_token].item())

    n = len(ranks)
    if n == 0:
        return {"accuracy": 0.0, "se": 0.0, "mean_rank": -1, "mean_prob": 0.0, "trials": 0}

    acc = correct / n
    # Wilson score interval standard error (for binomial proportion)
    se = math.sqrt(acc * (1 - acc) / n) if n > 1 else 0.0

    return {
        "accuracy": round(acc, 4),
        "se": round(se, 4),
        "ci95_low": round(max(0.0, acc - 1.96 * se), 4),
        "ci95_high": round(min(1.0, acc + 1.96 * se), 4),
        "mean_rank": round(sum(ranks) / n, 1),
        "mean_prob": round(sum(probs) / n, 6),
        "trials": n,
    }


def eval_passkey_full(
    model: GPT,
    tokenizer,
    filler_ids: torch.Tensor,
    context_lengths: List[int],
    depth_ratios: List[float],
    num_trials: int = 50,
) -> Dict[str, dict]:
    """Run full passkey evaluation across lengths and depths."""
    results = {}
    for L in context_lengths:
        for d in depth_ratios:
            key = f"L={L}_d={d:.1f}"
            print(f"    {key} ...", end=" ", flush=True)
            try:
                r = eval_passkey_single(
                    model, tokenizer, filler_ids, L, d, num_trials
                )
                print(f"acc={r['accuracy']:.0%} +/-{r['se']:.0%}  rank={r['mean_rank']:.0f}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("OOM, skipping")
                    r = {"accuracy": -1, "se": 0, "mean_rank": -1, "mean_prob": -1, "trials": 0, "error": "OOM"}
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    elif DEVICE == "mps":
                        torch.mps.empty_cache()
                else:
                    raise
            results[key] = r
    return results


# ---------------------------------------------------------------------------
# Load model from checkpoint
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    tier: str, tau: float, seed: int, work_dir: Path, base: float = 500000.0
) -> GPT:
    """Load a trained model from sweep checkpoint."""
    cfg = TIER_CONFIGS[tier].copy()
    run_id = f"{tier}_tau{tau:.2f}_seed{seed}"
    ckpt_path = work_dir / run_id / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    inv_freq = evq_cosh_inv_freq(cfg["head_dim"], tau, base)
    model = GPT(cfg, inv_freq)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    # Rebuild cos/sin caches from loaded inv_freq (state_dict overwrites inv_freq
    # buffer but cos_c/sin_c are non-persistent and come from constructor).
    model.blocks[0].attn.rope._build(cfg["max_position_embeddings"])
    model = model.to(DEVICE)
    model.eval()
    print(f"  Loaded {run_id} from {ckpt_path}")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Passkey retrieval eval for EVQ sweep")
    parser.add_argument("--work_dir", type=str, required=True,
                        help="Sweep work directory containing checkpoints")
    parser.add_argument("--tiers", type=str, default="500m",
                        help="Comma-separated tiers to evaluate")
    parser.add_argument("--taus", type=str, default="0.0,1.5",
                        help="Comma-separated tau values")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--base", type=float, default=500000.0)
    parser.add_argument("--context_lengths", type=str, default="1024,2048,4096,8192,16384",
                        help="Comma-separated context lengths to test")
    parser.add_argument("--depth_ratios", type=str, default="0.1,0.5,0.9",
                        help="Comma-separated depth ratios (0=start, 1=end)")
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--val_tokens", type=int, default=5_000_000,
                        help="Max validation tokens for filler text")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    tiers = [t.strip() for t in args.tiers.split(",")]
    taus = [float(t) for t in args.taus.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    context_lengths = [int(L) for L in args.context_lengths.split(",")]
    depth_ratios = [float(d) for d in args.depth_ratios.split(",")]

    # Load tokenizer & validation data
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    print("Loading validation data for filler text...")
    from run_evq_sweep import load_val
    raw_filler = load_val(tokenizer, args.val_tokens)

    print("Sanitizing filler (removing number-word contamination)...")
    filler_ids = _sanitize_filler(tokenizer, raw_filler, 256)
    print(f"  Filler ready: {len(filler_ids)} tokens")

    all_results = {
        "metadata": {
            "tiers": tiers,
            "taus": taus,
            "seeds": seeds,
            "context_lengths": context_lengths,
            "depth_ratios": depth_ratios,
            "num_trials": args.num_trials,
            "device": DEVICE,
            "filler_sanitized": True,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "experiments": {},
    }

    for tier in tiers:
        for tau in taus:
            for seed in seeds:
                run_id = f"{tier}_tau{tau:.2f}_seed{seed}"
                print(f"\n{'='*60}")
                print(f"  PASSKEY EVAL: {run_id}")
                print(f"{'='*60}")

                try:
                    model = load_model_from_checkpoint(
                        tier, tau, seed, work_dir, args.base
                    )
                except FileNotFoundError as e:
                    print(f"  SKIP: {e}")
                    continue

                results = eval_passkey_full(
                    model, tokenizer, filler_ids,
                    context_lengths, depth_ratios,
                    args.num_trials,
                )

                # Compute summary: average accuracy +/- SE per context length
                summary = {}
                for L in context_lengths:
                    accs = []
                    for d in depth_ratios:
                        key = f"L={L}_d={d:.1f}"
                        if key in results and results[key].get("accuracy", -1) >= 0:
                            accs.append(results[key]["accuracy"])
                    if accs:
                        avg = sum(accs) / len(accs)
                        std = (sum((a - avg) ** 2 for a in accs) / max(len(accs) - 1, 1)) ** 0.5
                        summary[f"L={L}_avg_acc"] = round(avg, 4)
                        summary[f"L={L}_std_acc"] = round(std, 4)

                all_results["experiments"][run_id] = {
                    "details": results,
                    "summary": summary,
                }

                # Free model memory
                del model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                elif DEVICE == "mps":
                    torch.mps.empty_cache()

                # Checkpoint
                out_path = work_dir / "passkey_results.json"
                with open(out_path, "w") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                print(f"  [checkpoint] saved to {out_path}")

    # Final save
    all_results["metadata"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path = work_dir / "passkey_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary table
    print(f"\n{'='*60}")
    print("  PASSKEY RETRIEVAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<30} ", end="")
    for L in context_lengths:
        print(f" L={L:>5}", end="")
    print()
    print("  " + "-" * (30 + 7 * len(context_lengths)))

    for run_id, exp in sorted(all_results["experiments"].items()):
        print(f"  {run_id:<30} ", end="")
        for L in context_lengths:
            acc = exp["summary"].get(f"L={L}_avg_acc", -1)
            if acc >= 0:
                print(f" {acc:>5.1%}", end="")
            else:
                print(f"   N/A", end="")
        print()

    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
