#!/usr/bin/env python3
"""PE baseline comparison: evaluate inference-time position encoding methods
on a trained Geometric (tau=0) checkpoint.

Methods evaluated:
  - Geo (baseline): original geometric frequencies, no modification
  - PI: geo_inv / scale
  - YaRN: smoothstep ramp interpolation + temperature (scale relative to train_len)
  - NTK-aware: scaled_base = base * scale^(d/(d-2)), recompute geometric
  - Dynamic NTK: per-eval-length scale, recompute geometric each time

Usage:
    python scripts/m4_evq_sweep/eval_pe_baselines.py \
        --checkpoint_dir /root/autodl-tmp/evq_passkey_mix_10pct/350m_tau0.00_seed42 \
        --tier 350m --base 500000

Notes on interpretation (written to results file):
  - EVQ is a TRAINING-TIME method: trained from scratch with EVQ frequencies
  - PI/YaRN/NTK are INFERENCE-TIME methods: zero extra training, applied to Geo checkpoint
  - Even if Geo + SOTA inference PE < EVQ trained => frequency allocation should be
    optimized at training time, not patched at inference time
"""

from __future__ import annotations

import argparse
import hashlib
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
    evq_cosh_inv_freq,
    load_val,
    get_device_and_dtype,
    phase_collision_score,
)
from eval_passkey_scratch import eval_passkey_nll_gap

DEVICE, DTYPE = get_device_and_dtype()
USE_AUTOCAST = DEVICE == "cuda" and DTYPE != torch.float32


# ---------------------------------------------------------------------------
# Frequency builders for each PE method
# ---------------------------------------------------------------------------

def build_geo_inv_freq(head_dim: int, base: float) -> torch.Tensor:
    """Geometric RoPE frequencies matching EVQ tau=0 (midpoint quantization)."""
    return evq_cosh_inv_freq(head_dim, tau=0.0, base=base)


def build_pi_inv_freq(geo_inv: torch.Tensor, scale: float) -> torch.Tensor:
    """Position Interpolation: divide frequencies by scale factor."""
    return geo_inv / scale


def build_yarn_inv_freq(
    geo_inv: torch.Tensor, head_dim: int, scale: float
) -> torch.Tensor:
    """YaRN: smoothstep ramp interpolation + attention temperature.

    Uses train_len-relative scale (NOT the hardcoded 8192 in schedules.py).
    """
    K = head_dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    start = int(0.20 * K)
    end = int(0.90 * K)
    if end <= start:
        end = min(K - 1, start + 1)
    ramp = torch.clamp((idx - start) / float(max(1, end - start)), 0.0, 1.0)
    # Smoothstep
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    temperature = 1.0 + 0.07 * math.log2(scale) if scale > 1.0 else 1.0
    yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
    return (geo_inv.double() / yarn_scale).float()


def build_ntk_aware_inv_freq(
    head_dim: int, base: float, scale: float
) -> torch.Tensor:
    """NTK-aware: scaled_base = base * scale^(d/(d-2)), recompute geometric."""
    d = head_dim
    scaled_base = base * (scale ** (d / (d - 2)))
    K = d // 2
    idx = torch.arange(K, dtype=torch.float64)
    inv = 1.0 / (scaled_base ** (2.0 * idx / d))
    return inv.float()


def build_dynamic_ntk_inv_freq(
    head_dim: int, base: float, train_len: int, eval_len: int
) -> torch.Tensor:
    """Dynamic NTK: per-eval-length scale, recompute geometric."""
    scale = max(eval_len / train_len, 1.0)
    if scale <= 1.0:
        return build_geo_inv_freq(head_dim, base)
    return build_ntk_aware_inv_freq(head_dim, base, scale)


# ---------------------------------------------------------------------------
# Swap-and-eval helpers
# ---------------------------------------------------------------------------

def _swap_inv_freq(model: GPT, new_inv: torch.Tensor, max_len: int) -> None:
    """Replace model's RoPE inv_freq and rebuild cos/sin cache."""
    model.blocks[0].attn.rope.inv_freq.copy_(new_inv)
    model.blocks[0].attn.rope._build(max_len + 100)


def _restore_inv_freq(model: GPT, orig_inv: torch.Tensor, train_len: int) -> None:
    """Restore original inv_freq and rebuild cache."""
    model.blocks[0].attn.rope.inv_freq.copy_(orig_inv)
    model.blocks[0].attn.rope._build(train_len)


def eval_static_method(
    model: GPT,
    method_name: str,
    inv_freq: torch.Tensor,
    orig_inv: torch.Tensor,
    val_data: torch.Tensor,
    tokenizer,
    eval_lengths: List[int],
    passkey_lengths: List[int],
    eval_chunks: int,
    train_len: int,
    skip_swap: bool = False,
) -> dict:
    """Swap inv_freq -> eval PPL + passkey -> restore. Returns result dict."""
    inv_hash = hashlib.sha256(inv_freq.numpy().tobytes()).hexdigest()[:16]
    pc = phase_collision_score(inv_freq)

    print(f"\n{'='*60}")
    print(f"  Evaluating: {method_name}")
    print(f"  inv_freq: max={inv_freq.max().item():.6f}  "
          f"min={inv_freq.min().item():.8f}  hash={inv_hash}")
    print(f"  Phase collision total={pc['total']:.6f}")
    print(f"{'='*60}")

    max_L = max(eval_lengths)
    if not skip_swap:
        _swap_inv_freq(model, inv_freq, max_L)
    else:
        model.extend_rope(max_L + 100)

    # PPL eval
    t0 = time.time()
    ppl = eval_model(model, val_data, eval_lengths, eval_chunks)
    ppl_time = time.time() - t0

    # Passkey eval (only at passkey_lengths, which may be shorter than eval_lengths)
    t1 = time.time()
    passkey = eval_passkey_nll_gap(
        model, tokenizer, val_data,
        lengths=passkey_lengths,
        depths=[0.10, 0.25, 0.50, 0.75, 0.90],
        num_trials=10,
    )
    passkey_time = time.time() - t1

    # Restore
    if not skip_swap:
        _restore_inv_freq(model, orig_inv, train_len)

    return {
        "method": method_name,
        "type": "inference_time",
        "ppl": ppl,
        "passkey": passkey,
        "phase_collision": pc,
        "inv_freq_hash": inv_hash,
        "ppl_time_sec": round(ppl_time, 1),
        "passkey_time_sec": round(passkey_time, 1),
    }


def eval_dynamic_ntk(
    model: GPT,
    orig_inv: torch.Tensor,
    val_data: torch.Tensor,
    tokenizer,
    eval_lengths: List[int],
    passkey_lengths: List[int],
    eval_chunks: int,
    head_dim: int,
    base: float,
    train_len: int,
) -> dict:
    """Dynamic NTK: per-eval-length inv_freq swap."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: Dynamic NTK (per-length)")
    print(f"{'='*60}")

    ppl_all: Dict[str, float] = {}
    passkey_all_summary: Dict[str, dict] = {}
    passkey_all_details: Dict[str, dict] = {}
    global_gaps: List[float] = []
    global_retrieved: List[bool] = []
    global_ar_match: List[bool] = []

    t0 = time.time()

    # PPL eval at all lengths
    for L in eval_lengths:
        inv_freq = build_dynamic_ntk_inv_freq(head_dim, base, train_len, L)
        scale_L = max(L / train_len, 1.0)
        inv_hash = hashlib.sha256(inv_freq.numpy().tobytes()).hexdigest()[:16]
        print(f"\n  [PPL] L={L}: scale={scale_L:.2f}  hash={inv_hash}")

        _swap_inv_freq(model, inv_freq, L)
        ppl_L = eval_model(model, val_data, [L], eval_chunks)
        ppl_all.update(ppl_L)
        _restore_inv_freq(model, orig_inv, train_len)

    # Passkey eval only at passkey_lengths
    for L in passkey_lengths:
        inv_freq = build_dynamic_ntk_inv_freq(head_dim, base, train_len, L)
        scale_L = max(L / train_len, 1.0)
        print(f"\n  [Passkey] L={L}: scale={scale_L:.2f}")

        _swap_inv_freq(model, inv_freq, L)
        passkey_L = eval_passkey_nll_gap(
            model, tokenizer, val_data,
            lengths=[L],
            depths=[0.10, 0.25, 0.50, 0.75, 0.90],
            num_trials=10,
        )
        passkey_all_summary.update(passkey_L.get("summary", {}))
        for key, detail in passkey_L.get("details", {}).items():
            passkey_all_details[key] = detail
            global_gaps.append(detail["nll_gap"])
            global_retrieved.append(detail["retrieved"])
            global_ar_match.append(detail["ar_exact_match"])
        _restore_inv_freq(model, orig_inv, train_len)

    total_time = time.time() - t0

    n_global = len(global_gaps)
    passkey_combined = {
        "summary": passkey_all_summary,
        "details": passkey_all_details,
        "global": {
            "total_trials": n_global,
            "mean_nll_gap": round(
                sum(global_gaps) / n_global, 4
            ) if n_global else float("nan"),
            "retrieval_rate": round(
                sum(1 for r in global_retrieved if r) / n_global, 4
            ) if n_global else float("nan"),
            "ar_exact_match_rate": round(
                sum(1 for m in global_ar_match if m) / n_global, 4
            ) if n_global else float("nan"),
        },
    }

    # Phase collision for the max-length variant
    max_inv = build_dynamic_ntk_inv_freq(head_dim, base, train_len, max(eval_lengths))
    pc = phase_collision_score(max_inv)
    max_hash = hashlib.sha256(max_inv.numpy().tobytes()).hexdigest()[:16]

    return {
        "method": "Dynamic NTK",
        "type": "inference_time",
        "ppl": ppl_all,
        "passkey": passkey_combined,
        "phase_collision": pc,
        "inv_freq_hash": max_hash,
        "eval_time_sec": round(total_time, 1),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _per_length_retrieval(passkey_results: dict, lengths: List[int]) -> Dict[str, float]:
    """Average retrieval rate across depths for each context length."""
    summary = passkey_results.get("summary", {})
    per_length = {}
    for L in lengths:
        rates = []
        for key, val in summary.items():
            if key.startswith(f"L={L}_"):
                rate = val.get("retrieval_rate", float("nan"))
                if not math.isnan(rate):
                    rates.append(rate)
        if rates:
            per_length[str(L)] = round(sum(rates) / len(rates), 4)
        else:
            per_length[str(L)] = float("nan")
    return per_length


def print_comparison_table(
    results: Dict[str, dict],
    eval_lengths: List[int],
    passkey_lengths: List[int],
    evq_entry: Optional[dict] = None,
) -> None:
    """Print a formatted comparison table."""
    # Header
    len_strs = [f"{L // 1024}K" for L in eval_lengths]
    pk_len_strs = [f"{L // 1024}K" for L in passkey_lengths]
    ppl_headers = [f"PPL@{s}" for s in len_strs]
    pk_headers = [f"Pass@{s}" for s in pk_len_strs]

    col_w = 10
    method_w = 20
    header = f"  {'Method':<{method_w}}"
    for h in ppl_headers + pk_headers:
        header += f"  {h:>{col_w}}"
    header += f"  {'PhaseCol':>{col_w}}"

    sep = "  " + "─" * (method_w + (col_w + 2) * (len(ppl_headers + pk_headers) + 1))

    print(f"\n{sep}")
    print(header)
    print(sep)

    # Method order
    method_order = ["Geo (baseline)", "PI", "YaRN", "NTK-aware", "Dynamic NTK"]

    for method_name in method_order:
        entry = results.get(method_name)
        if entry is None:
            continue
        ppl = entry.get("ppl", {})
        pk_rates = _per_length_retrieval(entry.get("passkey", {}), passkey_lengths)

        row = f"  {method_name:<{method_w}}"
        for L in eval_lengths:
            val = ppl.get(str(L), float("nan"))
            row += f"  {val:>{col_w}.2f}" if not math.isnan(val) else f"  {'N/A':>{col_w}}"
        for L in passkey_lengths:
            val = pk_rates.get(str(L), float("nan"))
            if math.isnan(val):
                row += f"  {'N/A':>{col_w}}"
            else:
                row += f"  {val:>{col_w}.0%}"
        pc = entry.get("phase_collision", {}).get("total", float("nan"))
        row += f"  {pc:>{col_w}.4f}" if not math.isnan(pc) else f"  {'N/A':>{col_w}}"
        print(row)

    # EVQ row (if available)
    if evq_entry:
        print(sep)
        ppl = evq_entry.get("ppl", {})
        pk_rates = _per_length_retrieval(evq_entry.get("passkey", {}), passkey_lengths)
        tau = evq_entry.get("tau", "?")
        name = f"EVQ t={tau} (*)"
        row = f"  {name:<{method_w}}"
        for L in eval_lengths:
            val = ppl.get(str(L), float("nan"))
            row += f"  {val:>{col_w}.2f}" if not math.isnan(val) else f"  {'N/A':>{col_w}}"
        for L in passkey_lengths:
            val = pk_rates.get(str(L), float("nan"))
            if math.isnan(val):
                row += f"  {'N/A':>{col_w}}"
            else:
                row += f"  {val:>{col_w}.0%}"
        print(row)
        print(f"  (*) = trained from scratch with EVQ frequencies, "
              f"not inference-time swap")

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PE baseline comparison on Geo (tau=0) checkpoint"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to Geo checkpoint dir (e.g., .../350m_tau0.00_seed42)",
    )
    parser.add_argument("--tier", type=str, default="350m",
                        choices=list(TIER_CONFIGS.keys()))
    parser.add_argument("--base", type=float, default=500000.0)
    parser.add_argument("--train_len", type=int, default=2048)
    parser.add_argument(
        "--eval_lengths", type=str, default="",
        help="Comma-separated PPL eval lengths (default: from tier config)",
    )
    parser.add_argument(
        "--passkey_lengths", type=str, default="2048,4096,8192",
        help="Comma-separated passkey eval lengths (default: 2048,4096,8192)",
    )
    parser.add_argument(
        "--results_json", type=str, default="",
        help="Path to results_checkpoint.json for EVQ comparison row",
    )
    parser.add_argument("--evq_tau", type=float, default=1.5,
                        help="EVQ tau value to include in comparison table")
    parser.add_argument("--evq_seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TIER_CONFIGS[args.tier].copy()
    eval_lengths = (
        [int(x) for x in args.eval_lengths.split(",")]
        if args.eval_lengths
        else list(cfg["eval_lengths"])
    )
    passkey_lengths = [int(x) for x in args.passkey_lengths.split(",")]
    head_dim = cfg["head_dim"]
    train_len = args.train_len
    scale = max(max(eval_lengths) / train_len, 1.0)

    print(f"\n{'#'*60}")
    print(f"  PE BASELINE COMPARISON")
    print(f"  tier={args.tier}  base={args.base}  train_len={train_len}")
    print(f"  eval_lengths={eval_lengths}  scale={scale:.2f}")
    print(f"  passkey_lengths={passkey_lengths}")
    print(f"  device={DEVICE}  dtype={DTYPE}")
    print(f"{'#'*60}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Load val data (uses disk cache)
    ckpt_dir = Path(args.checkpoint_dir)
    work_dir = ckpt_dir.parent
    val_data = load_val(tok, cache_dir=str(work_dir))

    # Load model from Geo checkpoint
    print(f"\n  Loading Geo checkpoint from: {ckpt_dir}")
    geo_inv = build_geo_inv_freq(head_dim, args.base)
    model = GPT(cfg, geo_inv)
    ckpt_path = ckpt_dir / "model.pt"
    if not ckpt_path.exists():
        print(f"  ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    print(f"  Model loaded successfully")

    # Save original inv_freq for restore
    orig_inv = model.blocks[0].attn.rope.inv_freq.clone()

    # ──────────────────────────────────────────────────────────────
    # Evaluate each method
    # ──────────────────────────────────────────────────────────────
    all_results: Dict[str, dict] = {}
    t_total = time.time()

    # 1. Geo (baseline) — no swap needed
    geo_result = eval_static_method(
        model, "Geo (baseline)", geo_inv, orig_inv,
        val_data, tok, eval_lengths, passkey_lengths,
        cfg["eval_chunks"], train_len, skip_swap=True,
    )
    geo_result["type"] = "trained"
    all_results["Geo (baseline)"] = geo_result

    # 2. PI
    pi_inv = build_pi_inv_freq(geo_inv, scale)
    pi_result = eval_static_method(
        model, "PI", pi_inv, orig_inv,
        val_data, tok, eval_lengths, passkey_lengths,
        cfg["eval_chunks"], train_len,
    )
    all_results["PI"] = pi_result

    # 3. YaRN (with train_len-relative scale)
    yarn_inv = build_yarn_inv_freq(geo_inv, head_dim, scale)
    yarn_result = eval_static_method(
        model, "YaRN", yarn_inv, orig_inv,
        val_data, tok, eval_lengths, passkey_lengths,
        cfg["eval_chunks"], train_len,
    )
    all_results["YaRN"] = yarn_result

    # 4. NTK-aware
    ntk_inv = build_ntk_aware_inv_freq(head_dim, args.base, scale)
    ntk_result = eval_static_method(
        model, "NTK-aware", ntk_inv, orig_inv,
        val_data, tok, eval_lengths, passkey_lengths,
        cfg["eval_chunks"], train_len,
    )
    all_results["NTK-aware"] = ntk_result

    # 5. Dynamic NTK (per-length)
    dntk_result = eval_dynamic_ntk(
        model, orig_inv, val_data, tok,
        eval_lengths, passkey_lengths, cfg["eval_chunks"],
        head_dim, args.base, train_len,
    )
    all_results["Dynamic NTK"] = dntk_result

    total_time = time.time() - t_total
    print(f"\n  All methods evaluated in {total_time / 60:.1f} min")

    # ──────────────────────────────────────────────────────────────
    # Optionally load EVQ results for comparison
    # ──────────────────────────────────────────────────────────────
    evq_entry = None
    if args.results_json:
        rj = Path(args.results_json)
        if rj.exists():
            with open(rj) as f:
                sweep_results = json.load(f)
            evq_run_id = f"{args.tier}_tau{args.evq_tau:.2f}_seed{args.evq_seed}"
            evq_data = sweep_results.get("experiments", {}).get(evq_run_id, {})
            if evq_data:
                evq_entry = {
                    "tau": args.evq_tau,
                    "ppl": evq_data.get("ppl", {}),
                    "passkey": evq_data.get("passkey", {}),
                }
                print(f"  Loaded EVQ comparison from: {evq_run_id}")
            else:
                print(f"  WARNING: {evq_run_id} not found in {rj}")
        else:
            print(f"  WARNING: results_json not found: {rj}")

    # ──────────────────────────────────────────────────────────────
    # Print comparison table
    # ──────────────────────────────────────────────────────────────
    print_comparison_table(all_results, eval_lengths, passkey_lengths, evq_entry)

    # ──────────────────────────────────────────────────────────────
    # Save results
    # ──────────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "tier": args.tier,
            "base": args.base,
            "train_len": train_len,
            "eval_lengths": eval_lengths,
            "scale": scale,
            "device": DEVICE,
            "dtype": str(DTYPE),
            "checkpoint_dir": str(ckpt_dir),
            "total_time_min": round(total_time / 60, 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "methods": {},
        "interpretation_note": (
            "EVQ is a TRAINING-TIME method: trained from scratch with EVQ "
            "frequencies, same compute budget as Geo. "
            "PI/YaRN/NTK are INFERENCE-TIME methods: zero extra training, "
            "applied to the Geo checkpoint at eval time. "
            "If EVQ outperforms Geo + best inference-time PE => frequency "
            "allocation should be optimized at training time."
        ),
    }

    for name, entry in all_results.items():
        # Strip large details from passkey to keep JSON manageable
        passkey_slim = {
            "summary": entry.get("passkey", {}).get("summary", {}),
            "global": entry.get("passkey", {}).get("global", {}),
        }
        per_len_ret = _per_length_retrieval(
            entry.get("passkey", {}), eval_lengths
        )
        output["methods"][name] = {
            "type": entry.get("type", "inference_time"),
            "ppl": entry.get("ppl", {}),
            "passkey_summary": passkey_slim,
            "per_length_retrieval": per_len_ret,
            "phase_collision": entry.get("phase_collision", {}),
            "inv_freq_hash": entry.get("inv_freq_hash", ""),
        }

    if evq_entry:
        per_len_ret = _per_length_retrieval(
            evq_entry.get("passkey", {}), eval_lengths
        )
        output["methods"][f"EVQ tau={args.evq_tau}"] = {
            "type": "trained",
            "ppl": evq_entry.get("ppl", {}),
            "passkey_summary": {
                "summary": evq_entry.get("passkey", {}).get("summary", {}),
                "global": evq_entry.get("passkey", {}).get("global", {}),
            },
            "per_length_retrieval": per_len_ret,
        }

    out_path = work_dir / "pe_baselines_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to: {out_path}")

    # Also save full passkey details separately
    details_path = work_dir / "pe_baselines_passkey_details.json"
    full_passkey = {}
    for name, entry in all_results.items():
        full_passkey[name] = entry.get("passkey", {})
    with open(details_path, "w") as f:
        json.dump(full_passkey, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Full passkey details saved to: {details_path}")


if __name__ == "__main__":
    main()
