#!/usr/bin/env python3
"""Super-extrapolation eval: per-length YaRN on Geo/EVQ checkpoints.

Evaluates a trained checkpoint at extreme extrapolation ratios (up to 64x)
using per-length Dynamic YaRN. For each eval length, applies YaRN with
scale = eval_len / train_len, so each length gets its optimal configuration.

Also runs raw baseline (no YaRN) for comparison.

Usage:
    python eval_super_extrap.py \
        --checkpoint_dir /path/to/350m_tau1.50_seed42 \
        --tier 350m --train_len 512 --base 500000 \
        --eval_lengths 512,2048,4096,8192,16384,32768 \
        --passkey_lengths 512,2048,4096,8192,16384

    # Batch mode: eval all checkpoints in a work_dir
    python eval_super_extrap.py \
        --work_dir /path/to/evq_super_extrap_512 \
        --tier 350m --train_len 512 --base 500000 \
        --eval_lengths 512,2048,4096,8192,16384,32768 \
        --passkey_lengths 512,2048,4096,8192,16384
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Imports from sweep scripts
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import (
    GPT,
    TIER_CONFIGS,
    eval_model,
    load_val,
    get_device_and_dtype,
)
from eval_pe_baselines import (
    build_yarn_inv_freq,
    _swap_inv_freq,
    _restore_inv_freq,
)
from eval_passkey_scratch import eval_passkey_nll_gap

DEVICE, DTYPE = get_device_and_dtype()
USE_AUTOCAST = DEVICE == "cuda" and DTYPE != torch.float32


# ---------------------------------------------------------------------------
# Core eval logic
# ---------------------------------------------------------------------------

def eval_checkpoint(
    checkpoint_dir: Path,
    cfg: dict,
    train_len: int,
    base: float,
    val_data: torch.Tensor,
    tokenizer,
    eval_lengths: List[int],
    passkey_lengths: List[int],
    raw_eval_lengths: Optional[List[int]] = None,
    raw_passkey_lengths: Optional[List[int]] = None,
    eval_chunks: int = 10,
) -> Dict:
    """Evaluate a single checkpoint with raw baseline + per-length YaRN.

    Args:
        checkpoint_dir: Path to checkpoint (must contain model.pt + inv_freq.npy)
        cfg: Model config dict (from TIER_CONFIGS)
        train_len: Training sequence length
        base: RoPE base frequency
        val_data: Validation token IDs (1-D tensor)
        tokenizer: Tokenizer for passkey eval
        eval_lengths: Lengths for per-length YaRN eval (PPL)
        passkey_lengths: Lengths for per-length YaRN eval (passkey)
        raw_eval_lengths: Lengths for raw baseline PPL (default: up to 16x train_len)
        raw_passkey_lengths: Lengths for raw baseline passkey (default: same as raw_eval)
        eval_chunks: Number of PPL eval chunks per length

    Returns:
        Dict with raw and yarn eval results
    """
    run_name = checkpoint_dir.name
    print(f"\n{'='*60}")
    print(f"  SUPER-EXTRAP EVAL: {run_name}")
    print(f"  train_len={train_len}  base={base}")
    print(f"{'='*60}")

    # Defaults for raw eval
    if raw_eval_lengths is None:
        raw_eval_lengths = [train_len * m for m in [1, 2, 4, 8, 16]
                            if train_len * m <= 16384]
    if raw_passkey_lengths is None:
        raw_passkey_lengths = raw_eval_lengths

    # Load checkpoint
    inv_freq_path = checkpoint_dir / "inv_freq.npy"
    model_path = checkpoint_dir / "model.pt"
    if not model_path.exists():
        print(f"  ERROR: {model_path} not found, skipping")
        return {}

    inv_freq = np.load(str(inv_freq_path))
    inv_freq_t = torch.from_numpy(inv_freq)

    # Override cfg for this model
    eval_cfg = cfg.copy()
    eval_cfg["seq_len"] = train_len
    eval_cfg["max_position_embeddings"] = train_len

    model = GPT(eval_cfg, inv_freq_t).to(DEVICE)
    state = torch.load(str(model_path), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    head_dim = cfg["head_dim"]
    results = {
        "run_name": run_name,
        "train_len": train_len,
        "base": base,
        "head_dim": head_dim,
        "raw": {},
        "yarn": {},
    }

    # ---- A) Raw baseline (no YaRN) ----
    print(f"\n  --- Raw baseline (no YaRN) ---")
    print(f"  PPL eval lengths: {raw_eval_lengths}")
    raw_ppl = eval_model(model, val_data, raw_eval_lengths, eval_chunks)
    results["raw"]["ppl"] = raw_ppl

    print(f"\n  Passkey eval lengths: {raw_passkey_lengths}")
    try:
        raw_pk = eval_passkey_nll_gap(
            model, tokenizer, val_data,
            lengths=raw_passkey_lengths,
            depths=[0.25, 0.50, 0.75],
            num_trials=10,
        )
        results["raw"]["passkey"] = raw_pk
    except Exception as e:
        print(f"  Raw passkey eval failed: {e}")
        results["raw"]["passkey"] = {"error": str(e)}

    # ---- B) Per-length YaRN eval ----
    print(f"\n  --- Per-length YaRN eval ---")
    yarn_ppl = {}
    yarn_passkey = {}

    for L in sorted(set(eval_lengths)):
        scale_L = max(L / train_len, 1.0)
        print(f"\n  [YaRN] L={L}  scale={scale_L:.1f}x")

        if scale_L <= 1.0:
            # At or below train length — use raw results
            if str(L) in raw_ppl:
                yarn_ppl[str(L)] = raw_ppl[str(L)]
            print(f"    Using raw result (scale <= 1)")
            continue

        # Build YaRN frequencies for this scale
        yarn_inv = build_yarn_inv_freq(inv_freq_t, head_dim, scale_L)
        _swap_inv_freq(model, yarn_inv, L)

        # PPL
        try:
            ppl_result = eval_model(model, val_data, [L], eval_chunks)
            yarn_ppl.update(ppl_result)
        except Exception as e:
            print(f"    PPL eval failed at L={L}: {e}")

        # Passkey
        if L in passkey_lengths:
            try:
                pk_result = eval_passkey_nll_gap(
                    model, tokenizer, val_data,
                    lengths=[L],
                    depths=[0.25, 0.50, 0.75],
                    num_trials=10,
                )
                yarn_passkey[str(L)] = pk_result
            except Exception as e:
                print(f"    Passkey eval failed at L={L}: {e}")

        # Restore original frequencies
        _restore_inv_freq(model, inv_freq_t, train_len)

    results["yarn"]["ppl"] = yarn_ppl
    results["yarn"]["passkey"] = yarn_passkey

    # Cleanup
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ---- Print summary table ----
    _print_summary(results, train_len, eval_lengths, passkey_lengths)

    return results


def _print_summary(
    results: Dict,
    train_len: int,
    eval_lengths: List[int],
    passkey_lengths: List[int],
) -> None:
    """Print a formatted comparison table."""
    run_name = results.get("run_name", "?")
    raw_ppl = results.get("raw", {}).get("ppl", {})
    yarn_ppl = results.get("yarn", {}).get("ppl", {})
    raw_pk = results.get("raw", {}).get("passkey", {})
    yarn_pk = results.get("yarn", {}).get("passkey", {})

    print(f"\n  {'─'*70}")
    print(f"  Method: {run_name}  (train_len={train_len})")
    print(f"  {'─'*70}")
    print(f"  {'Eval_L':>8}  {'Scale':>6}  {'PPL(raw)':>10}  {'PPL(+YaRN)':>12}  "
          f"{'PK(raw)':>8}  {'PK(+YaRN)':>10}")

    all_lengths = sorted(set(eval_lengths) | set(passkey_lengths))
    for L in all_lengths:
        scale = L / train_len
        scale_str = f"{scale:.0f}x"

        # Raw PPL
        rp = raw_ppl.get(str(L))
        rp_str = f"{rp:.1f}" if rp is not None else "-"

        # YaRN PPL
        yp = yarn_ppl.get(str(L))
        yp_str = f"{yp:.1f}" if yp is not None else "-"

        # Raw passkey retrieval rate
        rk_str = "-"
        if raw_pk and isinstance(raw_pk, dict) and "summary" in raw_pk:
            for key, val in raw_pk["summary"].items():
                if key.startswith(f"L={L}_"):
                    rate = val.get("retrieval_rate", None)
                    if rate is not None:
                        rk_str = f"{rate*100:.0f}%"
                    break

        # YaRN passkey retrieval rate
        yk_str = "-"
        ypk = yarn_pk.get(str(L))
        if ypk and isinstance(ypk, dict):
            g = ypk.get("global", {})
            rate = g.get("retrieval_rate", None)
            if rate is not None:
                yk_str = f"{rate*100:.0f}%"

        print(f"  {L:>8}  {scale_str:>6}  {rp_str:>10}  {yp_str:>12}  "
              f"{rk_str:>8}  {yk_str:>10}")

    print(f"  {'─'*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Super-extrapolation eval with per-length YaRN"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint_dir", type=str,
                       help="Single checkpoint directory to evaluate")
    group.add_argument("--work_dir", type=str,
                       help="Work directory containing multiple checkpoint dirs")

    parser.add_argument("--tier", choices=["50m", "125m", "350m", "500m"],
                        required=True)
    parser.add_argument("--train_len", type=int, required=True,
                        help="Training sequence length (e.g. 512, 256)")
    parser.add_argument("--base", type=float, default=500000.0,
                        help="RoPE base frequency")
    parser.add_argument("--eval_lengths", type=str, required=True,
                        help="Comma-separated YaRN eval lengths "
                             "(e.g. 512,2048,4096,8192,16384,32768)")
    parser.add_argument("--passkey_lengths", type=str, default=None,
                        help="Comma-separated passkey eval lengths "
                             "(default: same as eval_lengths minus largest)")
    parser.add_argument("--eval_chunks", type=int, default=10,
                        help="Number of PPL eval chunks per length")
    parser.add_argument("--dataset", type=str, default="fineweb-edu",
                        choices=["fineweb-edu", "tinystories"])
    args = parser.parse_args()

    eval_lengths = [int(x) for x in args.eval_lengths.split(",")]
    if args.passkey_lengths:
        passkey_lengths = [int(x) for x in args.passkey_lengths.split(",")]
    else:
        # Default: all eval lengths except the largest (may OOM for passkey)
        passkey_lengths = eval_lengths[:-1] if len(eval_lengths) > 1 else eval_lengths

    cfg = TIER_CONFIGS[args.tier].copy()
    cfg["seq_len"] = args.train_len
    cfg["max_position_embeddings"] = args.train_len

    print(f"\n{'#'*60}")
    print(f"  SUPER-EXTRAPOLATION EVAL")
    print(f"  tier={args.tier}  train_len={args.train_len}  base={args.base}")
    print(f"  eval_lengths={eval_lengths}")
    print(f"  passkey_lengths={passkey_lengths}")
    print(f"{'#'*60}\n")

    # Load tokenizer & validation data
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    cache_dir = None
    if args.work_dir:
        cache_dir = args.work_dir
    elif args.checkpoint_dir:
        cache_dir = str(Path(args.checkpoint_dir).parent)
    val_data = load_val(tok, dataset=args.dataset, cache_dir=cache_dir)

    # Determine checkpoint dirs to evaluate
    if args.checkpoint_dir:
        ckpt_dirs = [Path(args.checkpoint_dir)]
    else:
        work = Path(args.work_dir)
        # Find all dirs matching pattern: {tier}_tau*_seed* or {tier}_r*_seed*
        ckpt_dirs = sorted([
            d for d in work.iterdir()
            if d.is_dir() and (d / "model.pt").exists()
            and d.name.startswith(f"{args.tier}_")
        ])
        if not ckpt_dirs:
            print(f"  ERROR: No checkpoint dirs found in {work}")
            return

    print(f"  Found {len(ckpt_dirs)} checkpoint(s) to evaluate")

    # Evaluate each checkpoint
    all_results = {
        "metadata": {
            "tier": args.tier,
            "train_len": args.train_len,
            "base": args.base,
            "eval_lengths": eval_lengths,
            "passkey_lengths": passkey_lengths,
            "device": DEVICE,
            "dtype": str(DTYPE),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "checkpoints": {},
    }

    t_total = time.time()
    for ckpt_dir in ckpt_dirs:
        print(f"\n  >>> Evaluating: {ckpt_dir.name}")
        t0 = time.time()

        result = eval_checkpoint(
            checkpoint_dir=ckpt_dir,
            cfg=cfg,
            train_len=args.train_len,
            base=args.base,
            val_data=val_data,
            tokenizer=tok,
            eval_lengths=eval_lengths,
            passkey_lengths=passkey_lengths,
            eval_chunks=args.eval_chunks,
        )

        result["eval_time_sec"] = round(time.time() - t0, 1)
        all_results["checkpoints"][ckpt_dir.name] = result

        # Save incremental results
        out_dir = ckpt_dir.parent
        results_path = out_dir / "super_extrap_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"  [saved] {results_path}")

    total_time = time.time() - t_total
    all_results["metadata"]["total_time_min"] = round(total_time / 60, 1)
    all_results["metadata"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Final save
    if args.work_dir:
        final_path = Path(args.work_dir) / "super_extrap_results.json"
    else:
        final_path = Path(args.checkpoint_dir).parent / "super_extrap_results.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"  SUPER-EXTRAP EVAL COMPLETE  |  {total_time / 60:.1f} min total")
    print(f"  Results: {final_path}")
    print(f"{'='*60}")

    # Final cross-checkpoint comparison
    _print_cross_comparison(all_results, args.train_len)


def _print_cross_comparison(all_results: Dict, train_len: int) -> None:
    """Print a cross-checkpoint comparison table."""
    checkpoints = all_results.get("checkpoints", {})
    if len(checkpoints) < 2:
        return

    print(f"\n  {'='*70}")
    print(f"  CROSS-CHECKPOINT COMPARISON (YaRN PPL)")
    print(f"  {'='*70}")

    # Collect all eval lengths
    all_L = set()
    for ck in checkpoints.values():
        all_L.update(int(k) for k in ck.get("yarn", {}).get("ppl", {}).keys())
    all_L = sorted(all_L)

    # Header
    header = f"  {'Run':>35}"
    for L in all_L:
        scale = L / train_len
        header += f"  {L}({scale:.0f}x)"
    print(header)

    # Rows
    for name, ck in sorted(checkpoints.items()):
        row = f"  {name:>35}"
        yppl = ck.get("yarn", {}).get("ppl", {})
        for L in all_L:
            v = yppl.get(str(L))
            if v is not None:
                row += f"  {v:>10.1f}"
            else:
                row += f"  {'—':>10}"
        print(row)

    # YaRN Passkey comparison
    print(f"\n  CROSS-CHECKPOINT COMPARISON (YaRN Passkey Retrieval Rate)")
    header = f"  {'Run':>35}"
    for L in all_L:
        scale = L / train_len
        header += f"  {L}({scale:.0f}x)"
    print(header)

    for name, ck in sorted(checkpoints.items()):
        row = f"  {name:>35}"
        ypk = ck.get("yarn", {}).get("passkey", {})
        for L in all_L:
            pk_data = ypk.get(str(L))
            if pk_data and isinstance(pk_data, dict):
                g = pk_data.get("global", {})
                rate = g.get("retrieval_rate")
                if rate is not None:
                    row += f"  {rate*100:>9.0f}%"
                else:
                    row += f"  {'—':>10}"
            else:
                row += f"  {'—':>10}"
        print(row)

    print(f"  {'='*70}")


if __name__ == "__main__":
    main()
