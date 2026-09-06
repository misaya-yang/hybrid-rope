#!/usr/bin/env python3
"""
Phase 14C: supporting multi-scale EVQ+YaRN validation (50M, 125M).

Runs a supporting 5% passkey-mix check at smaller model scales.
3 seeds (42, 123, 7) × 2 methods (Geo, EVQ tau=1.5 r=0) × 2 tiers (50M, 125M) = 12 runs.

After training, runs YaRN@scale=8 passkey eval + PPL eval to verify
that the matched-scale EVQ+YaRN pattern persists across smaller scales.

Hyperparameters follow the supporting multiscale protocol:
  - 100M tokens FineWeb-Edu
  - seq_len=2048
  - base=500K
  - 5% passkey mix
  - head_dim=64
  - cosine LR schedule with 2% warmup
  - Eval: passkey at 2K/4K/8K/12K/16K, depths=[0.1,0.2,0.5,0.8,0.9], 10 trials
  - YaRN: scale=8, channel-index ramp (start=0.20*K, end=0.90*K), smoothstep + temperature

Usage (on 5090):
    EVQ_PHASE14C_WORK_DIR=results/core_text/phase14c \
      python scripts/core_text_phases/phase14c_multiscale_evq_yarn.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core_text_phases.run_evq_sweep import (
    GPT,
    DEVICE,
    DTYPE,
    USE_AUTOCAST,
    eval_model,
    evq_cosh_inv_freq,
    load_data,
    load_val,
    set_seed,
    get_batch_from_data,
    maybe_wrap_with_passkey_mix,
)
from scripts.supporting_eval.eval_passkey_scratch import eval_passkey_nll_gap

# ===================================================================
# Constants — MUST match 350M experiment exactly
# ===================================================================

BASE = 500_000.0
HEAD_DIM = 64
TRAIN_TOKENS = 100_000_000
SEQ_LEN = 2048
PASSKEY_MIX_RATIO = 0.05  # 5%
SEEDS = [42, 123, 7]
TAUS = [0.0, 1.5]  # 0.0 = Geo, 1.5 = EVQ (full, r=0)

# Eval config — same as 350M
EVAL_LENGTHS = [2048, 4096, 8192, 12288, 16384]
EVAL_CHUNKS = 8
PASSKEY_LENGTHS = [2048, 4096, 8192, 12288, 16384]
PASSKEY_DEPTHS = [0.1, 0.2, 0.5, 0.8, 0.9]
PASSKEY_TRIALS = 10
YARN_SCALE = 8.0

# Tier configs — architecture only, training params unified
TIERS = {
    "50m": dict(
        vocab_size=50304,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        head_dim=HEAD_DIM,
        intermediate_size=2048,
        max_position_embeddings=SEQ_LEN,
        lr=6e-4,
        batch_size=32,       # effective
        micro_batch_size=8,  # fits on 5090 32GB
        grad_accum=4,
    ),
    "125m": dict(
        vocab_size=50304,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        head_dim=HEAD_DIM,
        intermediate_size=3072,
        max_position_embeddings=SEQ_LEN,
        lr=3e-4,
        batch_size=16,       # effective
        micro_batch_size=4,  # fits on 5090 32GB
        grad_accum=4,
    ),
}

WORK = Path(os.environ.get("EVQ_PHASE14C_WORK_DIR", "results/core_text/phase14c"))
DATA_CACHE_DIR = Path(os.environ.get("EVQ_DATA_CACHE", str(WORK / "data")))
REUSE_CACHE_DIRS = [
    Path(p)
    for p in os.environ.get("EVQ_PHASE14C_REUSE_DIRS", "").split(os.pathsep)
    if p
]
# ===================================================================
# YaRN — exact copy from eval_pe_baselines.py (used in 350M experiment)
# ===================================================================

def build_yarn_inv_freq(
    geo_inv: torch.Tensor, head_dim: int, scale: float
) -> torch.Tensor:
    """YaRN: smoothstep ramp interpolation + attention temperature.

    Uses channel-index-based ramp (start=0.20*K, end=0.90*K).
    EXACT copy from eval_pe_baselines.py used in 350M experiment.
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


# ===================================================================
# Helpers
# ===================================================================

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def method_label(tau):
    return "EVQ" if tau > 0 else "Geo"


def run_tag(tier, tau, seed):
    return f"{tier}_tau{tau:.2f}_seed{seed}"


# ===================================================================
# Training — matches 350M exactly (cosine LR, 2% warmup, grad clip 1.0)
# ===================================================================

def train_model(model, data, cfg, seed=42):
    """Train model with cosine LR schedule + gradient accumulation. Matches 350M training."""
    model.train()
    lr = cfg["lr"]
    min_lr = lr * 0.1
    micro_bs = cfg.get("micro_batch_size", cfg["batch_size"])
    grad_accum = cfg.get("grad_accum", 1)
    effective_bs = micro_bs * grad_accum

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    total_chunks = len(data)
    steps = total_chunks // effective_bs
    warmup = int(steps * 0.02)

    print(f"  Training: micro_bs={micro_bs}, grad_accum={grad_accum}, "
          f"effective_bs={effective_bs}, steps={steps}, warmup={warmup}, lr={lr}")

    set_seed(seed)
    perm = torch.randperm(total_chunks)
    t0 = time.time()

    for s in range(steps):
        # Cosine LR with linear warmup
        if s < warmup:
            cur_lr = lr * s / max(warmup, 1)
        else:
            frac = (s - warmup) / max(steps - warmup, 1)
            cur_lr = min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for a in range(grad_accum):
            chunk_idx = s * effective_bs + a * micro_bs
            batch = get_batch_from_data(data, perm[chunk_idx : chunk_idx + micro_bs]).to(DEVICE)
            ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
            with ctx:
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )
                loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if s % 100 == 0 or s == steps - 1:
            elapsed = time.time() - t0
            eta = elapsed / (s + 1) * (steps - s - 1) if s > 0 else 0
            print(f"    step {s}/{steps}  loss={accum_loss:.4f}  lr={cur_lr:.2e}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed/60:.1f} min")
    return model, elapsed


# ===================================================================
# Evaluation — passkey + PPL, baseline + YaRN
# ===================================================================

def eval_run(model, tokenizer, filler_tokens, val_data, inv_freq_orig, tag):
    """Full evaluation: baseline passkey + PPL, then YaRN passkey + PPL."""
    results = {}

    # --- Baseline ---
    print(f"\n  [baseline] {method_label(0 if 'tau0' in tag else 1.5)} raw")
    model.eval()
    with torch.no_grad():
        # PPL
        ppl = eval_model(model, val_data, EVAL_LENGTHS, EVAL_CHUNKS)
        print(f"    PPL: {ppl}")

        # Passkey
        pk = eval_passkey_nll_gap(
            model, tokenizer, filler_tokens,
            lengths=PASSKEY_LENGTHS,
            depths=PASSKEY_DEPTHS,
            num_trials=PASSKEY_TRIALS,
            seed=42,
        )

    pk_global = pk.get("global", {})
    pk_summary = pk.get("summary", {})

    # Per-length passkey summary
    pk_per_length = {}
    for L in PASSKEY_LENGTHS:
        trials_at_L = [v for k, v in pk_summary.items() if k.startswith(f"L={L}_")]
        if trials_at_L:
            mean_ret = np.mean([t.get("retrieval_rate", 0) for t in trials_at_L])
            pk_per_length[str(L)] = round(mean_ret * 100)
    print(f"    PK per-length: {pk_per_length}")

    results["baseline"] = {
        "ppl": ppl,
        "passkey_global": pk_global,
        "passkey_summary": pk_summary,
        "passkey_per_length": pk_per_length,
    }

    # --- YaRN ---
    print(f"\n  [+YaRN] {method_label(0 if 'tau0' in tag else 1.5)} + YaRN (scale={YARN_SCALE})")
    orig_inv = inv_freq_orig.clone()
    yarn_inv = build_yarn_inv_freq(orig_inv, HEAD_DIM, YARN_SCALE)
    model.blocks[0].attn.rope.inv_freq.copy_(yarn_inv)
    model.blocks[0].attn.rope._build(max(EVAL_LENGTHS) + 100)

    with torch.no_grad():
        # PPL with YaRN
        ppl_yarn = eval_model(model, val_data, EVAL_LENGTHS, EVAL_CHUNKS)
        print(f"    PPL: {ppl_yarn}")

        # Passkey with YaRN
        pk_yarn = eval_passkey_nll_gap(
            model, tokenizer, filler_tokens,
            lengths=PASSKEY_LENGTHS,
            depths=PASSKEY_DEPTHS,
            num_trials=PASSKEY_TRIALS,
            seed=42,
        )

    pk_yarn_global = pk_yarn.get("global", {})
    pk_yarn_summary = pk_yarn.get("summary", {})

    pk_yarn_per_length = {}
    for L in PASSKEY_LENGTHS:
        trials_at_L = [v for k, v in pk_yarn_summary.items() if k.startswith(f"L={L}_")]
        if trials_at_L:
            mean_ret = np.mean([t.get("retrieval_rate", 0) for t in trials_at_L])
            pk_yarn_per_length[str(L)] = round(mean_ret * 100)
    print(f"    PK per-length: {pk_yarn_per_length}")

    results["yarn"] = {
        "ppl": ppl_yarn,
        "passkey_global": pk_yarn_global,
        "passkey_summary": pk_yarn_summary,
        "passkey_per_length": pk_yarn_per_length,
    }

    # Restore original inv_freq
    model.blocks[0].attn.rope.inv_freq.copy_(orig_inv)
    model.blocks[0].attn.rope._build(SEQ_LEN + 100)

    return results


# ===================================================================
# Single run: train + eval
# ===================================================================

def run_single(tier, tau, seed, train_data, val_data, filler, tokenizer):
    """Train one model and evaluate it."""
    tag = run_tag(tier, tau, seed)
    method = method_label(tau)
    run_dir = WORK / tier / tag

    result_file = run_dir / "result.json"
    if result_file.exists():
        print(f"\n[SKIP] {tag}: already done")
        with open(result_file) as f:
            return json.load(f)

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = TIERS[tier].copy()

    print(f"\n{'='*60}")
    print(f"  {tag}  ({method}, {tier}, seed={seed})")
    print(f"{'='*60}")

    # Build inv_freq: full EVQ (r=0) for tau>0, geometric for tau=0
    inv_freq = evq_cosh_inv_freq(HEAD_DIM, tau=tau, base=BASE)

    # Build and train model
    set_seed(seed)
    model = GPT(cfg, inv_freq).to(DEVICE)
    model, train_time = train_model(model, train_data, cfg, seed=seed)

    # Save model
    model_path = run_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    np.save(run_dir / "inv_freq.npy", inv_freq.cpu().numpy())

    # Evaluate
    inv_freq_orig = model.blocks[0].attn.rope.inv_freq.clone().cpu()
    eval_results = eval_run(model, tokenizer, filler, val_data, inv_freq_orig, tag)

    # Save passkey NLL details
    save_json(run_dir / "passkey_nll.json", eval_results)

    result = {
        "tag": tag,
        "tier": tier,
        "method": method,
        "tau": tau,
        "seed": seed,
        "train_time_sec": round(train_time, 1),
        **eval_results,
    }
    save_json(result_file, result)
    print(f"  Saved: {result_file}")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    return result


# ===================================================================
# Report generation
# ===================================================================

def generate_report(all_results):
    """Generate final summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  PHASE 14C: Multi-Scale EVQ+YaRN Synergy Validation")
    lines.append("  50M and 125M, 3 seeds (42/123/7), 5% passkey mix, YaRN scale=8")
    lines.append("=" * 70)

    for tier in ["50m", "125m"]:
        lines.append(f"\n{'='*60}")
        lines.append(f"  TIER: {tier}")
        lines.append(f"{'='*60}")

        # Collect per-method per-seed results
        for mode in ["baseline", "yarn"]:
            mode_label = "Baseline" if mode == "baseline" else "+YaRN (scale=8)"
            lines.append(f"\n  --- {mode_label} ---")

            # Header
            hdr = f"  {'Config':<20} {'PK@2K':>6} {'PK@4K':>6} {'PK@8K':>6} {'PK@12K':>7} {'PK@16K':>7}  {'PPL@2K':>7} {'PPL@8K':>7} {'PPL@16K':>8}"
            lines.append(hdr)
            lines.append("  " + "-" * (len(hdr) - 2))

            for tau in TAUS:
                method = method_label(tau)
                seed_pks = {str(L): [] for L in PASSKEY_LENGTHS}
                seed_ppls = {str(L): [] for L in EVAL_LENGTHS}

                for seed in SEEDS:
                    tag = run_tag(tier, tau, seed)
                    r = all_results.get(tag)
                    if r is None:
                        continue
                    mode_data = r.get(mode, {})
                    pk_pl = mode_data.get("passkey_per_length", {})
                    ppl = mode_data.get("ppl", {})

                    for L in PASSKEY_LENGTHS:
                        val = pk_pl.get(str(L))
                        if val is not None:
                            seed_pks[str(L)].append(val)
                    for L in EVAL_LENGTHS:
                        val = ppl.get(str(L))
                        if val is not None:
                            seed_ppls[str(L)].append(val)

                    # Per-seed line
                    pk_vals = " ".join(f"{pk_pl.get(str(L), '?'):>5}%" for L in PASSKEY_LENGTHS)
                    ppl_2k = ppl.get("2048", ppl.get(2048, "?"))
                    ppl_8k = ppl.get("8192", ppl.get(8192, "?"))
                    ppl_16k = ppl.get("16384", ppl.get(16384, "?"))
                    ppl_str = f"{ppl_2k:>7.1f} {ppl_8k:>7.1f} {ppl_16k:>8.1f}" if isinstance(ppl_2k, (int, float)) else f"{'?':>7} {'?':>7} {'?':>8}"
                    lines.append(f"  {method+' s'+str(seed):<20} {pk_vals}  {ppl_str}")

                # Mean +/- std line
                if all(len(seed_pks[str(L)]) >= 2 for L in PASSKEY_LENGTHS):
                    pk_summary = ""
                    for L in PASSKEY_LENGTHS:
                        vals = seed_pks[str(L)]
                        m, s = np.mean(vals), np.std(vals)
                        pk_summary += f" {m:>3.0f}±{s:<2.0f}%"
                    ppl_summary = ""
                    for L_key in ["2048", "8192", "16384"]:
                        vals = seed_ppls.get(L_key, [])
                        if vals:
                            m = np.mean(vals)
                            ppl_summary += f" {m:>7.1f}"
                        else:
                            ppl_summary += f" {'?':>7}"
                    lines.append(f"  {method+' MEAN±STD':<20}{pk_summary} {ppl_summary}")
                lines.append("")

        # Delta table
        lines.append(f"\n  --- Delta (EVQ+YaRN - Geo+YaRN), {tier} ---")
        for L in PASSKEY_LENGTHS:
            evq_vals = []
            geo_vals = []
            for seed in SEEDS:
                evq_tag = run_tag(tier, 1.5, seed)
                geo_tag = run_tag(tier, 0.0, seed)
                evq_r = all_results.get(evq_tag, {}).get("yarn", {}).get("passkey_per_length", {})
                geo_r = all_results.get(geo_tag, {}).get("yarn", {}).get("passkey_per_length", {})
                if str(L) in evq_r:
                    evq_vals.append(evq_r[str(L)])
                if str(L) in geo_r:
                    geo_vals.append(geo_r[str(L)])
            if evq_vals and geo_vals:
                evq_m, geo_m = np.mean(evq_vals), np.mean(geo_vals)
                delta = evq_m - geo_m
                lines.append(f"    L={L:>5}: EVQ+YaRN={evq_m:.0f}%  Geo+YaRN={geo_m:.0f}%  Delta={delta:+.0f}pp")

    # Cross-scale summary
    lines.append(f"\n{'='*60}")
    lines.append("  CROSS-SCALE SUMMARY: EVQ+YaRN @8K (the key metric)")
    lines.append(f"{'='*60}")
    for tier in ["50m", "125m"]:
        for method_tau, method_name in [(0.0, "Geo"), (1.5, "EVQ")]:
            baseline_vals = []
            yarn_vals = []
            for seed in SEEDS:
                tag = run_tag(tier, method_tau, seed)
                r = all_results.get(tag)
                if r is None:
                    continue
                b_pk = r.get("baseline", {}).get("passkey_per_length", {}).get("8192")
                y_pk = r.get("yarn", {}).get("passkey_per_length", {}).get("8192")
                if b_pk is not None:
                    baseline_vals.append(b_pk)
                if y_pk is not None:
                    yarn_vals.append(y_pk)
            if baseline_vals and yarn_vals:
                b_m, b_s = np.mean(baseline_vals), np.std(baseline_vals)
                y_m, y_s = np.mean(yarn_vals), np.std(yarn_vals)
                lines.append(f"  {tier} {method_name:<4}: baseline={b_m:.0f}±{b_s:.0f}%  +YaRN={y_m:.0f}±{y_s:.0f}%")

    report = "\n".join(lines)
    return report


# ===================================================================
# Main
# ===================================================================

def main():
    t_start = time.time()

    print("#" * 70)
    print("  Phase 14C: Multi-Scale EVQ+YaRN Synergy Validation")
    print("  Tiers: 50M, 125M  |  Seeds: 42, 123, 7")
    print("  Methods: Geo (tau=0), EVQ (tau=1.5, r=0)")
    print("  Training: 100M tokens, seq_len=2048, 5% passkey mix")
    print("  Eval: passkey @2K/4K/8K/12K/16K + YaRN scale=8")
    print("#" * 70)

    WORK.mkdir(parents=True, exist_ok=True)
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Load data.  Optional reuse paths are explicit via EVQ_PHASE14C_REUSE_DIRS;
    # the default path stays inside the repository-local results tree.
    print("\nLoading training data...")
    train_cache_name = "train_fineweb-edu_100000000_2048.pt"
    existing_caches = [DATA_CACHE_DIR / train_cache_name]
    existing_caches.extend(d / train_cache_name for d in REUSE_CACHE_DIRS)
    train_cache = None
    for p in existing_caches:
        if p.exists():
            train_cache = p
            break

    if train_cache and train_cache != DATA_CACHE_DIR / train_cache_name:
        # Symlink to avoid double storage
        dest = DATA_CACHE_DIR / train_cache_name
        if not dest.exists():
            os.symlink(str(train_cache), str(dest))
        print(f"  Reusing existing cache: {train_cache}")

    train_data_raw = load_data(
        tok,
        TRAIN_TOKENS,
        SEQ_LEN,
        "fineweb-edu",
        cache_dir=str(DATA_CACHE_DIR),
        strict_dataset=True,
    )

    print("Loading validation data...")
    val_cache_name = "val_fineweb-edu_5000000.pt"
    val_cache_dest = DATA_CACHE_DIR / "val_fineweb-edu_5000000.pt"
    for existing_val in [d / val_cache_name for d in REUSE_CACHE_DIRS]:
        if existing_val.exists() and not val_cache_dest.exists():
            os.symlink(str(existing_val), str(val_cache_dest))
            print(f"  Reusing existing validation cache: {existing_val}")
            break
    val_data = load_val(
        tok,
        5_000_000,
        "fineweb-edu",
        cache_dir=str(DATA_CACHE_DIR),
        strict_dataset=True,
    )
    filler = val_data[:50000]

    # Wrap with passkey mix (5%)
    train_data = maybe_wrap_with_passkey_mix(
        train_data=train_data_raw,
        filler_tokens=filler,
        tokenizer=tok,
        seq_len=SEQ_LEN,
        passkey_ratio=PASSKEY_MIX_RATIO,
    )

    # Run all experiments
    all_results = {}

    for tier in ["50m", "125m"]:
        for tau in TAUS:
            for seed in SEEDS:
                tag = run_tag(tier, tau, seed)
                result = run_single(tier, tau, seed, train_data, val_data, filler, tok)
                all_results[tag] = result

    # Generate report
    report = generate_report(all_results)
    print("\n" + report)

    # Save everything
    save_json(WORK / "all_results.json", all_results)
    report_path = WORK / "REPORT.txt"
    with open(report_path, "w") as f:
        f.write(report)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/3600:.1f} hours")
    print(f"  Results: {WORK}/all_results.json")
    print(f"  Report:  {WORK}/REPORT.txt")


if __name__ == "__main__":
    main()
