#!/usr/bin/env python3
"""Eval sanity check: diagnose long-length PPL explosion.

Tests:
  A) Position-split loss: compare loss at positions 0..2047 vs 2048+ within a long sequence.
     If 0..2047 ≈ PPL@2K but 2048+ is much worse → genuine extrapolation failure, not eval bug.
  B) Debug prints: shapes, NaN check, logit stats, position verification.
  C) Compare standard eval vs position-aware eval to ensure consistency.

Usage:
    python experiments/eval_sanity.py --checkpoint_dir /path/to/125m_tau1.00_seed42 \
        --val_data /path/to/val_*.pt --tier 125m --base 500000 --tau 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext

# Add project root
_proj_root = str(Path(__file__).resolve().parents[1])
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

# Import from sweep script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "m4_evq_sweep"))
from run_evq_sweep import (
    GPT, RotaryEmbedding, LearnableRotaryEmbedding,
    evq_cosh_inv_freq, TIER_CONFIGS, eval_model,
    get_device_and_dtype,
)

DEVICE, DTYPE = get_device_and_dtype()
USE_AUTOCAST = DEVICE == "cuda" and DTYPE != torch.float32


def load_model_from_checkpoint(ckpt_dir: Path, cfg: dict, tau: float, base: float,
                                learnable: bool = False, tau_init: float = 1.0):
    """Load a trained model from checkpoint."""
    ckpt_path = ckpt_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No model.pt in {ckpt_dir}")

    inv_freq = evq_cosh_inv_freq(cfg["head_dim"], tau, base)

    if learnable:
        learnable_rope = LearnableRotaryEmbedding(
            head_dim=cfg["head_dim"],
            max_seq=cfg["max_position_embeddings"],
            base=base,
            tau_init=tau_init,
        ).to(DEVICE)
        model = GPT(cfg, inv_freq, learnable_rope=learnable_rope).to(DEVICE)
    else:
        model = GPT(cfg, inv_freq).to(DEVICE)

    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def position_split_eval(model, val_data: torch.Tensor, L: int, train_len: int = 2048,
                         n_chunks: int = 5, seed: int = 9999):
    """Evaluate loss split by position range: [0, train_len) vs [train_len, L)."""
    model.eval()
    model.extend_rope(L + 100)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()

    rng = np.random.RandomState(seed)
    max_start = len(val_data) - L
    if max_start <= 0:
        print(f"  val_data too short for L={L}")
        return None

    n_chunks = min(n_chunks, max_start // L)
    offsets = sorted(rng.choice(max_start, size=n_chunks, replace=False))

    results = {
        "L": L,
        "train_len": train_len,
        "in_range_losses": [],    # positions 0..train_len-1
        "extrap_losses": [],      # positions train_len..L-1
        "full_losses": [],        # all positions
    }

    for offset in offsets:
        chunk = val_data[offset: offset + L].unsqueeze(0).to(DEVICE)
        input_ids = chunk[:, :-1]   # (1, L-1)
        targets = chunk[:, 1:]      # (1, L-1)

        with torch.no_grad(), ctx:
            logits = model(input_ids)  # (1, L-1, vocab)

        # Debug prints for first chunk only
        if offset == offsets[0]:
            print(f"\n  === DEBUG for L={L} ===")
            print(f"  input_ids shape: {input_ids.shape}")
            print(f"  targets shape:   {targets.shape}")
            print(f"  logits shape:    {logits.shape}")
            print(f"  logits dtype:    {logits.dtype}")
            print(f"  logits has NaN:  {torch.isnan(logits).any().item()}")
            print(f"  logits has Inf:  {torch.isinf(logits).any().item()}")
            print(f"  logits min/max/mean: {logits.min().item():.4f} / {logits.max().item():.4f} / {logits.mean().item():.4f}")

            # Check RoPE positions implicitly
            rope = model.blocks[0].attn.rope
            if isinstance(rope, RotaryEmbedding):
                print(f"  RoPE cache size: {rope._max}")
                print(f"  cos_c shape:     {rope.cos_c.shape}")
                print(f"  cos_c[:3, :4]:   {rope.cos_c[:3, :4]}")
                print(f"  cos_c[-3:, :4]:  {rope.cos_c[-3:, :4]}")
            elif isinstance(rope, LearnableRotaryEmbedding):
                print(f"  LearnableRoPE max: {rope._max}")
                tau_val = rope.evq.get_tau_value()
                print(f"  Learned tau:       {tau_val:.6f}")

        # Per-token cross-entropy (no reduction)
        per_token_loss = F.cross_entropy(
            logits.squeeze(0), targets.squeeze(0), reduction="none"
        )  # (L-1,)

        # Check for NaN/Inf in loss
        if torch.isnan(per_token_loss).any() or torch.isinf(per_token_loss).any():
            n_nan = torch.isnan(per_token_loss).sum().item()
            n_inf = torch.isinf(per_token_loss).sum().item()
            print(f"  WARNING: {n_nan} NaN, {n_inf} Inf in per-token loss!")

        # Split by position
        # Positions 0..train_len-2 are "in range" (target positions 1..train_len-1)
        # But note: the INPUT position that matters is the query position.
        # Position i in the input predicts token i+1. The model uses RoPE position i.
        # "In-range" = model used RoPE positions 0..train_len-1
        # Input has positions 0..L-2. We want to split at train_len-1.
        split_at = min(train_len - 1, L - 1)  # position index in the loss tensor

        in_range = per_token_loss[:split_at]
        extrap = per_token_loss[split_at:]

        full_loss = per_token_loss.mean().item()
        in_range_loss = in_range.mean().item() if len(in_range) > 0 else float("nan")
        extrap_loss = extrap.mean().item() if len(extrap) > 0 else float("nan")

        results["full_losses"].append(full_loss)
        results["in_range_losses"].append(in_range_loss)
        results["extrap_losses"].append(extrap_loss)

        del logits, per_token_loss, chunk
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Aggregate
    def safe_ppl(losses):
        valid = [l for l in losses if not math.isnan(l) and not math.isinf(l)]
        if not valid:
            return float("nan")
        return math.exp(sum(valid) / len(valid))

    results["full_ppl"] = safe_ppl(results["full_losses"])
    results["in_range_ppl"] = safe_ppl(results["in_range_losses"])
    results["extrap_ppl"] = safe_ppl(results["extrap_losses"])

    print(f"\n  L={L}: Position-split analysis ({n_chunks} chunks)")
    print(f"    Full PPL:                {results['full_ppl']:.3f}")
    print(f"    In-range [0..{train_len-1}] PPL:  {results['in_range_ppl']:.3f}")
    if not math.isnan(results["extrap_ppl"]):
        print(f"    Extrapolated [{train_len}..{L-1}] PPL: {results['extrap_ppl']:.3f}")
    else:
        print(f"    Extrapolated [{train_len}..{L-1}] PPL: N/A (all in range)")

    return results


def per_position_loss_curve(model, val_data: torch.Tensor, L: int,
                             n_chunks: int = 5, seed: int = 9999):
    """Compute per-position loss averaged over multiple chunks, for plotting."""
    model.eval()
    model.extend_rope(L + 100)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()

    rng = np.random.RandomState(seed)
    max_start = len(val_data) - L
    n_chunks = min(n_chunks, max_start // L)
    offsets = sorted(rng.choice(max_start, size=n_chunks, replace=False))

    all_losses = []
    for offset in offsets:
        chunk = val_data[offset: offset + L].unsqueeze(0).to(DEVICE)
        with torch.no_grad(), ctx:
            logits = model(chunk[:, :-1])
        per_token = F.cross_entropy(
            logits.squeeze(0), chunk[:, 1:].squeeze(0), reduction="none"
        )
        all_losses.append(per_token.cpu())
        del logits, chunk
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Average across chunks
    stacked = torch.stack(all_losses)  # (n_chunks, L-1)
    mean_loss = stacked.mean(dim=0).numpy()  # (L-1,)

    return mean_loss


def main():
    parser = argparse.ArgumentParser(description="Eval sanity check for long-length PPL")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing model.pt")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to val_*.pt file. If not set, auto-detect from work_dir.")
    parser.add_argument("--work_dir", type=str, default=None,
                        help="Work dir to auto-detect val data from")
    parser.add_argument("--tier", type=str, default="125m")
    parser.add_argument("--base", type=float, default=500000.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--learnable", action="store_true")
    parser.add_argument("--tau_init", type=float, default=1.0)
    parser.add_argument("--eval_lengths", type=str, default="2048,4096,8192",
                        help="Comma-separated eval lengths")
    parser.add_argument("--n_chunks", type=int, default=5)
    parser.add_argument("--save_curve", action="store_true",
                        help="Save per-position loss curve to JSON")
    args = parser.parse_args()

    cfg = TIER_CONFIGS[args.tier].copy()
    ckpt_dir = Path(args.checkpoint_dir)
    eval_lengths = [int(x) for x in args.eval_lengths.split(",")]

    # Load val data
    if args.val_data:
        val_data = torch.load(args.val_data, map_location="cpu", weights_only=True)
    else:
        # Auto-detect from work_dir or checkpoint_dir parent
        work_dir = Path(args.work_dir) if args.work_dir else ckpt_dir.parent
        candidates = list(work_dir.glob("val_*.pt"))
        if not candidates:
            print(f"ERROR: No val_*.pt found in {work_dir}")
            sys.exit(1)
        val_path = candidates[0]
        print(f"  Auto-detected val data: {val_path}")
        val_data = torch.load(val_path, map_location="cpu", weights_only=True)

    print(f"  Val data: {len(val_data)} tokens")

    # Load model
    print(f"\n  Loading model from {ckpt_dir}")
    model = load_model_from_checkpoint(
        ckpt_dir, cfg, args.tau, args.base,
        learnable=args.learnable, tau_init=args.tau_init,
    )

    # ---- Test A: Standard eval (same as run_evq_sweep.py) ----
    print(f"\n{'='*60}")
    print(f"  TEST A: Standard eval_model (same as training script)")
    print(f"{'='*60}")
    ppl_standard = eval_model(model, val_data, eval_lengths, eval_chunks=args.n_chunks)
    print(f"  Standard PPL: {ppl_standard}")

    # ---- Test B: Position-split eval ----
    print(f"\n{'='*60}")
    print(f"  TEST B: Position-split loss analysis")
    print(f"{'='*60}")
    split_results = {}
    for L in eval_lengths:
        try:
            res = position_split_eval(model, val_data, L, train_len=cfg["seq_len"],
                                       n_chunks=args.n_chunks)
            if res:
                split_results[L] = res
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  L={L}: OOM, skipping")
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'L':>6}  {'Std PPL':>10}  {'InRange PPL':>12}  {'Extrap PPL':>12}  {'Ratio':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*8}")
    for L in eval_lengths:
        std_ppl = ppl_standard.get(str(L), float("nan"))
        if L in split_results:
            sr = split_results[L]
            in_ppl = sr["in_range_ppl"]
            ex_ppl = sr["extrap_ppl"]
            if not math.isnan(ex_ppl) and not math.isnan(in_ppl) and in_ppl > 0:
                ratio = ex_ppl / in_ppl
            else:
                ratio = float("nan")
            print(f"  {L:>6}  {std_ppl:>10.3f}  {in_ppl:>12.3f}  {ex_ppl:>12.3f}  {ratio:>8.2f}x")
        else:
            print(f"  {L:>6}  {std_ppl:>10.3f}  {'N/A':>12}  {'N/A':>12}  {'N/A':>8}")

    print(f"\n  INTERPRETATION:")
    if split_results:
        # Check if in-range PPL at longer L matches PPL@2K
        base_ppl = ppl_standard.get(str(cfg["seq_len"]), None)
        if base_ppl:
            print(f"  - PPL@{cfg['seq_len']} (full, baseline): {base_ppl:.3f}")
            for L in eval_lengths:
                if L > cfg["seq_len"] and L in split_results:
                    in_ppl = split_results[L]["in_range_ppl"]
                    ex_ppl = split_results[L]["extrap_ppl"]
                    in_ratio = in_ppl / base_ppl
                    print(f"  - L={L}: in-range PPL={in_ppl:.1f} ({in_ratio:.2f}x baseline), "
                          f"extrap PPL={ex_ppl:.1f}")
                    if in_ratio > 1.5:
                        print(f"    ^^^ IN-RANGE PPL much worse than baseline → POSSIBLE EVAL BUG")
                    elif not math.isnan(ex_ppl) and ex_ppl / in_ppl > 2.0:
                        print(f"    ^^^ Extrapolation degradation: {ex_ppl/in_ppl:.1f}x → "
                              f"genuine extrapolation failure (model never trained beyond {cfg['seq_len']})")

    # Save per-position curve if requested
    if args.save_curve and eval_lengths:
        curve_L = max(L for L in eval_lengths if L <= 8192)  # avoid OOM
        print(f"\n  Computing per-position loss curve for L={curve_L}...")
        curve = per_position_loss_curve(model, val_data, curve_L, n_chunks=args.n_chunks)
        curve_path = ckpt_dir / "position_loss_curve.json"
        with open(curve_path, "w") as f:
            json.dump({
                "L": curve_L,
                "train_len": cfg["seq_len"],
                "per_position_loss": curve.tolist(),
            }, f)
        print(f"  Saved to {curve_path}")

    # Save full results
    out_path = ckpt_dir / "eval_sanity_results.json"
    out = {
        "standard_ppl": ppl_standard,
        "position_split": {str(L): {
            "full_ppl": split_results[L]["full_ppl"],
            "in_range_ppl": split_results[L]["in_range_ppl"],
            "extrap_ppl": split_results[L]["extrap_ppl"],
        } for L in split_results},
        "tier": args.tier,
        "tau": args.tau,
        "base": args.base,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Full results saved to {out_path}")


if __name__ == "__main__":
    main()
