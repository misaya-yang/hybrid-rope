#!/usr/bin/env python3
"""
EXP-4: 350M Progressive Chain — Delayed τ Protocol Validation (M4 Max 36GB)

Paper Role:  Validates delayed-τ protocol at 454M-equivalent architecture (24L/16H/d=1024)
             De-risks flagship 454M results + provides multi-seed evidence for Fig 4
Input:       FineWeb-Edu streaming (auto-download)
Output:      results/m4_max_36gb/exp4_progressive_350m/

## Motivation

Phase 17C (454M, seed 42-44) showed EVQ winning NIAH but losing PPL@8192.
Root cause: τ retarget at every stage forces "dual adaptation" — model must
simultaneously adapt to new frequencies AND learn longer context.

Fix: DELAYED τ — keep τ*(512)=2.828 through all stages, only apply YaRN at eval.

## Design

3 methods × 3 stages × 2 seeds = 18 training runs + evals

Methods:
  GEO       — Geometric baseline (τ=0), no retarget needed
  EVQ-D     — EVQ, delayed τ: τ*(512)=2.828 kept through ALL stages
  EVQ-R     — EVQ, retarget τ: τ recalculated per stage (broken protocol, diagnostic)

Stages:
  Stage 1: L=512,  tokens=STAGE1_TOKENS (base training)
  Stage 2: L=1024, tokens=STAGE2_TOKENS (continuation from Stage 1 ckpt)
  Stage 3: L=2048, tokens=STAGE3_TOKENS (continuation from Stage 2 ckpt)

Seeds: 42, 123 (paper-grade, 2-seed mean ± std)
  EVQ-R runs only seed 42 (diagnostic, 1-seed sufficient)

## Key Verification

After Stage 3:
  ✓ EVQ-D PPL@4096 < GEO PPL@4096  (delayed protocol wins extrapolation)
  ✓ EVQ-R PPL@4096 > EVQ-D PPL@4096 (retarget penalty confirmed)
  ✓ EVQ-D NIAH@4096 ≥ GEO NIAH@4096

## Usage

  conda activate aidemo
  cd ~/neurIPS-2026/hybrid-rope

  # Step 1: Pilot (measure MPS throughput, 5M tokens, ~30min)
  python scripts/m4_max_36gb/exp4_progressive_chain_350m.py --pilot

  # Step 2: Full run (use pilot timing to adjust tokens)
  python scripts/m4_max_36gb/exp4_progressive_chain_350m.py

  # Step 3: Summary
  python scripts/m4_max_36gb/exp4_progressive_chain_350m.py --summary

  # Optional: Adjust token budget based on pilot (default: 50M/25M/25M per stage)
  python scripts/m4_max_36gb/exp4_progressive_chain_350m.py --stage1_tokens 30000000 --stage2_tokens 20000000 --stage3_tokens 20000000
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Imports from core infrastructure
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
CORE_DIR = SCRIPT_DIR.parent / "core_text_phases"
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(CORE_DIR.parent / "supporting_eval"))

from run_evq_sweep import (  # noqa: E402
    GPT,
    DEVICE,
    DTYPE,
    USE_AUTOCAST,
    eval_model,
    evq_cosh_inv_freq,
    get_batch_from_data,
    load_data,
    load_val,
    maybe_wrap_with_passkey_mix,
    set_seed,
)

try:
    from eval_passkey_scratch import eval_passkey_nll_gap  # noqa: E402
    HAS_PASSKEY = True
except ImportError:
    HAS_PASSKEY = False
    print("[warn] eval_passkey_scratch not available, passkey eval disabled")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE = 500_000.0
DIM = 64  # head_dim, universal across all tiers

CFG_350M = dict(
    vocab_size=50304,
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    head_dim=64,
    intermediate_size=4096,
    max_position_embeddings=2048,  # updated per stage
)

# Default token budgets (adjustable via CLI)
DEFAULT_STAGE1_TOKENS = 50_000_000   # 50M
DEFAULT_STAGE2_TOKENS = 25_000_000   # 25M
DEFAULT_STAGE3_TOKENS = 25_000_000   # 25M

PILOT_TOKENS = 5_000_000  # 5M for pilot throughput measurement

# Batch sizes per stage (conservative for 30GB MPS)
BATCH_SIZES = {
    512:  4,   # ~8-10GB, comfortable
    1024: 2,   # ~14-16GB, comfortable
    2048: 1,   # ~18-22GB, tight but feasible at 30GB
}

# Gradient accumulation per stage (Bug 2 fix)
GRAD_ACCUM = {
    512:  2,   # B=4 × ga=2 → effective 8 seqs → 4096 tok/step
    1024: 2,   # B=2 × ga=2 → effective 4 seqs → 4096 tok/step
    2048: 4,   # B=1 × ga=4 → effective 4 seqs → 8192 tok/step
}

# Passkey mix ratio (Bug 3 fix — matches phase17b's 5%)
PASSKEY_RATIO = 0.05

# Eval lengths per stage
EVAL_LENGTHS = {
    512:  [512, 1024, 2048, 4096],
    1024: [1024, 2048, 4096, 8192],
    2048: [2048, 4096, 8192, 16384],
}

# Passkey eval lengths (only at trained length + 2x)
PK_LENGTHS = {
    512:  [512, 1024],
    1024: [1024, 2048, 4096],
    2048: [2048, 4096, 8192],
}
PK_TRIALS = 10
PK_DEPTHS = [0.1, 0.5, 0.9]

SEEDS = [42, 123]
EVAL_CHUNKS = 8

# Methods
METHODS = {
    "geo":   {"tau_mode": "zero",      "description": "Geometric baseline"},
    "evq_d": {"tau_mode": "delayed",   "description": "EVQ delayed-tau (keep tau*(512))"},
    "evq_r": {"tau_mode": "retarget",  "description": "EVQ retarget-tau (per-stage)"},
}

STAGES = [
    {"seq_len": 512,  "name": "stage1"},
    {"seq_len": 1024, "name": "stage2"},
    {"seq_len": 2048, "name": "stage3"},
]


# ---------------------------------------------------------------------------
# tau computation
# ---------------------------------------------------------------------------

def get_tau(method: str, stage_seq_len: int) -> float:
    """Get tau value for a method at a given stage."""
    mode = METHODS[method]["tau_mode"]
    if mode == "zero":
        return 0.0
    elif mode == "delayed":
        # Always use tau*(512) = d_head / sqrt(512) = 2.828
        return DIM / math.sqrt(512)
    elif mode == "retarget":
        # Retarget to tau*(current_L) at each stage
        return DIM / math.sqrt(stage_seq_len)
    else:
        raise ValueError(f"Unknown tau_mode: {mode}")


def get_inv_freq(method: str, stage_seq_len: int) -> torch.Tensor:
    """Get inv_freq for a method at a given stage."""
    tau = get_tau(method, stage_seq_len)
    return evq_cosh_inv_freq(DIM, tau, BASE)


# ---------------------------------------------------------------------------
# Training with checkpoint save + memory management
# ---------------------------------------------------------------------------

def train_stage(
    model: GPT,
    data,
    cfg: dict,
    seed: int,
    save_path: Path,
    stage_name: str,
    grad_accum: int = 1,
) -> Tuple[GPT, float]:
    """Train one stage with gradient accumulation, save checkpoint."""
    model.train()
    lr = cfg["lr"]
    min_lr = lr * 0.1
    batch_size = cfg["batch_size"]  # micro batch size
    seq_len = cfg["seq_len"]
    effective_bs = batch_size * grad_accum
    steps = len(data) // effective_bs
    warmup = max(int(steps * 0.02), 10)

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    set_seed(seed)
    perm = torch.randperm(len(data))
    t0 = time.time()

    print(f"    [{stage_name}] micro_bs={batch_size}, grad_accum={grad_accum}, "
          f"effective_bs={effective_bs}, steps={steps}, "
          f"tok/step={effective_bs * seq_len}")

    for s in range(steps):
        # Cosine LR with warmup
        if s < warmup:
            cur_lr = lr * s / max(warmup, 1)
        else:
            cur_lr = min_lr + (lr - min_lr) * 0.5 * (
                1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1))
            )
        for g in opt.param_groups:
            g["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for a in range(grad_accum):
            idx_start = s * effective_bs + a * batch_size
            idx_end = idx_start + batch_size
            indices = perm[idx_start:idx_end]
            batch = get_batch_from_data(data, indices).to(DEVICE)

            # Forward pass
            ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
            with ctx:
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)
                )
                loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # MPS memory management
        if DEVICE == "mps" and s % 100 == 0:
            torch.mps.empty_cache()

        if s % 50 == 0:
            elapsed = time.time() - t0
            tokens_done = (s + 1) * effective_bs * seq_len
            tokens_total = steps * effective_bs * seq_len
            tps = tokens_done / elapsed if elapsed > 0 else 0
            eta = (tokens_total - tokens_done) / tps if tps > 0 else 0
            print(
                f"    [{stage_name}] step {s}/{steps}  loss={accum_loss:.4f}  "
                f"lr={cur_lr:.2e}  {tps/1e3:.1f}K tok/s  ETA={eta/60:.0f}min"
            )

    train_time = time.time() - t0
    tokens_total = steps * effective_bs * seq_len
    tps = tokens_total / train_time if train_time > 0 else 0
    print(f"  [{stage_name}] Training done: {train_time/60:.1f}min, "
          f"{tokens_total/1e6:.1f}M tokens, {tps/1e3:.1f}K tok/s")

    # Save checkpoint
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"  [{stage_name}] Saved: {save_path}")

    return model, train_time


# ---------------------------------------------------------------------------
# Full pipeline: one method x one seed
# ---------------------------------------------------------------------------

def run_method_seed(
    method: str,
    seed: int,
    stage_tokens: Dict[str, int],
    work_dir: Path,
    val_data: torch.Tensor,
    tokenizer,
    dry_run: bool = False,
    preloaded_stage_data: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict:
    """Run full 3-stage progressive chain for one method and one seed."""
    tag = f"{method}_seed{seed}"
    run_dir = work_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  METHOD={method} ({METHODS[method]['description']})  SEED={seed}")
    print(f"{'='*70}")

    results = {"method": method, "seed": seed, "stages": {}}
    prev_ckpt_path = None

    # Filler tokens for passkey mix (from validation data)
    filler_tokens = val_data[:50000] if val_data is not None else None

    for stage_idx, stage in enumerate(STAGES):
        seq_len = stage["seq_len"]
        stage_name = stage["name"]
        tokens = stage_tokens.get(stage_name, 25_000_000)

        tau = get_tau(method, seq_len)
        inv_freq = get_inv_freq(method, seq_len)

        print(f"\n  --- {stage_name}: L={seq_len}, tau={tau:.4f}, tokens={tokens/1e6:.0f}M ---")
        print(f"  inv_freq: max={inv_freq.max().item():.6f}, min={inv_freq.min().item():.8f}")

        # Config for this stage
        cfg = dict(CFG_350M)
        cfg["max_position_embeddings"] = seq_len
        cfg["seq_len"] = seq_len
        cfg["batch_size"] = BATCH_SIZES[seq_len]
        cfg["lr"] = 2e-4 if stage_idx == 0 else 5e-5  # lower LR for continuation

        ga = GRAD_ACCUM[seq_len]

        if dry_run:
            batch_size = cfg["batch_size"]
            effective_bs = batch_size * ga
            n_steps = tokens // (effective_bs * seq_len)
            print(f"  [DRY RUN] batch={batch_size}, grad_accum={ga}, "
                  f"effective_bs={effective_bs}, steps={n_steps}, "
                  f"tokens/step={effective_bs * seq_len}")
            # Bug 5: Print inv_freq hash at Stage 2 to verify methods diverge
            if stage_name == "stage2":
                inv_hash = hashlib.md5(inv_freq.numpy().tobytes()).hexdigest()[:12]
                print(f"  [DRY RUN] Stage 2 inv_freq hash: {inv_hash} "
                      f"(tau={tau:.4f})")
            results["stages"][stage_name] = {
                "seq_len": seq_len, "tau": tau, "tokens": tokens,
                "batch_size": batch_size, "grad_accum": ga,
                "steps": n_steps, "status": "dry_run",
            }
            continue

        # Check if this stage already completed
        ckpt_path = run_dir / f"{stage_name}_L{seq_len}" / "model.pt"
        result_path = run_dir / f"{stage_name}_L{seq_len}" / "result.json"
        if result_path.exists():
            print(f"  [{stage_name}] Already completed, loading results...")
            with open(result_path) as f:
                results["stages"][stage_name] = json.load(f)
            prev_ckpt_path = ckpt_path
            continue

        # Bug 1 fix: Use preloaded, non-overlapping data slices
        print(f"  [{stage_name}] Using preloaded training data ({tokens/1e6:.0f}M tokens)...")
        train_data_raw = preloaded_stage_data[stage_name]

        # Bug 3 fix: Wrap with passkey mix training (5%)
        train_data = maybe_wrap_with_passkey_mix(
            train_data_raw, filler_tokens, tokenizer,
            seq_len=seq_len, passkey_ratio=PASSKEY_RATIO,
        )

        # Build model
        model = GPT(cfg, inv_freq).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  [{stage_name}] Model params: {n_params/1e6:.1f}M")

        # Load previous checkpoint for continuation
        if prev_ckpt_path is not None and prev_ckpt_path.exists():
            print(f"  [{stage_name}] Loading checkpoint: {prev_ckpt_path}")
            state = torch.load(prev_ckpt_path, map_location=DEVICE, weights_only=True)
            # Strip rope buffers (we set inv_freq explicitly)
            state = {k: v for k, v in state.items() if ".rope." not in k}
            missing, unexpected = model.load_state_dict(state, strict=False)
            rope_missing = [k for k in missing if ".rope." not in k]
            if rope_missing:
                print(f"  WARNING: non-rope missing keys: {rope_missing}")
            # Force our inv_freq
            model.blocks[0].attn.rope.inv_freq.copy_(
                inv_freq.to(model.blocks[0].attn.rope.inv_freq.device)
            )
            model.blocks[0].attn.rope._build(seq_len)
            del state
        elif stage_idx > 0:
            print(f"  WARNING: No checkpoint found at {prev_ckpt_path}, training from scratch!")

        # Train (Bug 2 fix: gradient accumulation)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        model, train_time = train_stage(
            model, train_data, cfg, seed,
            save_path=ckpt_path,
            stage_name=f"{tag}/{stage_name}",
            grad_accum=ga,
        )

        # Free training data
        del train_data, train_data_raw
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

        # Eval: PPL
        print(f"\n  [{stage_name}] Evaluating PPL...")
        eval_lens = EVAL_LENGTHS[seq_len]
        ppl = eval_model(model, val_data, eval_lens, EVAL_CHUNKS)

        # Bug 4 fix: Memory cleanup after eval (eval extends rope buffers)
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

        # Eval: Passkey
        passkey_results = {}
        if HAS_PASSKEY and tokenizer is not None:
            pk_lens = PK_LENGTHS[seq_len]
            print(f"  [{stage_name}] Evaluating Passkey NLL-gap (lengths={pk_lens})...")
            try:
                passkey_results = eval_passkey_nll_gap(
                    model, tokenizer, val_data,
                    lengths=pk_lens,
                    depths=PK_DEPTHS,
                    num_trials=PK_TRIALS,
                )
                g = passkey_results.get("global", {})
                print(f"  [{stage_name}] Passkey retrieval: {g.get('retrieval_rate', 'N/A')}")
            except Exception as e:
                print(f"  [{stage_name}] Passkey eval failed: {e}")

            # Bug 4 fix: Memory cleanup after passkey eval
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()

        # Save inv_freq for verification
        inv_freq_stats = {
            "tau": tau,
            "tau_mode": METHODS[method]["tau_mode"],
            "max": round(inv_freq.max().item(), 8),
            "min": round(inv_freq.min().item(), 8),
            "ratio": round((inv_freq.max() / inv_freq.min()).item(), 2),
        }
        np.save(ckpt_path.parent / "inv_freq.npy", inv_freq.numpy())

        stage_result = {
            "seq_len": seq_len,
            "tau": tau,
            "tau_mode": METHODS[method]["tau_mode"],
            "tokens": tokens,
            "batch_size": cfg["batch_size"],
            "grad_accum": ga,
            "passkey_ratio": PASSKEY_RATIO,
            "train_time_sec": round(train_time, 1),
            "ppl": ppl,
            "passkey": passkey_results,
            "inv_freq_stats": inv_freq_stats,
            "n_params": n_params,
        }

        # Save per-stage result
        with open(result_path, "w") as f:
            json.dump(stage_result, f, indent=2)

        results["stages"][stage_name] = stage_result
        prev_ckpt_path = ckpt_path

        # Free model memory before next stage
        del model
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Data preloading: load once, split across stages (Bug 1 fix)
# ---------------------------------------------------------------------------

def preload_stage_data(
    tokenizer,
    stage_tokens: Dict[str, int],
    cache_dir: str,
) -> Dict[str, torch.Tensor]:
    """Load all training tokens once and split into non-overlapping per-stage chunks."""
    # Compute exact flat tokens needed per stage
    flat_needed = 0
    stage_specs = []
    for stage in STAGES:
        sn = stage["name"]
        sl = stage["seq_len"]
        tok = stage_tokens.get(sn, 25_000_000)
        n_chunks = tok // sl
        n_flat = n_chunks * sl
        stage_specs.append((sn, sl, n_chunks, n_flat))
        flat_needed += n_flat

    print(f"  [data] Total flat tokens needed: {flat_needed/1e6:.1f}M")
    for sn, sl, nc, nf in stage_specs:
        print(f"    {sn}: L={sl}, chunks={nc}, tokens={nf/1e6:.1f}M")

    # Load all tokens at seq_len=512 (smallest granularity, divides all stage seq_lens)
    print(f"  [data] Loading {flat_needed/1e6:.1f}M tokens at seq_len=512...")
    all_data_chunked = load_data(
        tokenizer, flat_needed, 512,
        dataset="fineweb-edu",
        cache_dir=cache_dir,
    )
    all_flat = all_data_chunked.reshape(-1)
    del all_data_chunked

    if all_flat.numel() < flat_needed:
        raise ValueError(
            f"Loaded {all_flat.numel()} tokens but need {flat_needed}. "
            f"Try reducing token budgets."
        )

    # Split into non-overlapping per-stage slices
    stage_data = {}
    offset = 0
    for sn, sl, nc, nf in stage_specs:
        stage_data[sn] = all_flat[offset:offset + nf].reshape(-1, sl).clone()
        print(f"  [data] {sn}: offset={offset/1e6:.1f}M, "
              f"shape={tuple(stage_data[sn].shape)}")
        offset += nf

    del all_flat
    gc.collect()
    return stage_data


# ---------------------------------------------------------------------------
# Pilot: measure throughput
# ---------------------------------------------------------------------------

def run_pilot(work_dir: Path, tokenizer) -> Dict:
    """Quick throughput measurement: 350M, L=512, 5M tokens, seed=42."""
    print("\n" + "=" * 70)
    print("  PILOT RUN: Measuring MPS throughput for 350M")
    print("=" * 70)

    seq_len = 512
    batch_size = BATCH_SIZES[seq_len]
    tokens = PILOT_TOKENS

    cfg = dict(CFG_350M)
    cfg["max_position_embeddings"] = seq_len
    cfg["seq_len"] = seq_len
    cfg["batch_size"] = batch_size
    cfg["lr"] = 2e-4

    inv_freq = evq_cosh_inv_freq(DIM, DIM / math.sqrt(seq_len), BASE)

    print(f"  Config: L={seq_len}, batch={batch_size}, tokens={tokens/1e6:.0f}M")
    print(f"  Loading data...")
    train_data = load_data(
        tokenizer, tokens, seq_len,
        dataset="fineweb-edu",
        cache_dir=str(work_dir / "data_cache"),
    )

    print(f"  Building model...")
    model = GPT(cfg, inv_freq).to(DEVICE)

    print(f"  Training...")
    _, train_time = train_stage(
        model, train_data, cfg, seed=42,
        save_path=work_dir / "pilot" / "model.pt",
        stage_name="pilot",
    )

    tps = tokens / train_time
    print(f"\n  PILOT RESULTS:")
    print(f"  Throughput: {tps/1e3:.1f}K tokens/sec")
    print(f"  Time for 5M tokens: {train_time/60:.1f}min")

    # Project full experiment time
    stage_tokens = {
        "stage1": DEFAULT_STAGE1_TOKENS,
        "stage2": DEFAULT_STAGE2_TOKENS,
        "stage3": DEFAULT_STAGE3_TOKENS,
    }

    # Estimate: Stage 2/3 are slower due to longer sequences
    # L=1024 ~ 0.6x throughput of L=512 (longer seqs, smaller batch)
    # L=2048 ~ 0.35x throughput of L=512 (longest seqs, batch=1)
    time_stage1 = stage_tokens["stage1"] / tps
    time_stage2 = stage_tokens["stage2"] / (tps * 0.6)
    time_stage3 = stage_tokens["stage3"] / (tps * 0.35)

    # Total: 2 seeds x (GEO + EVQ-D) + 1 seed x EVQ-R = 5 full chains
    time_per_chain = time_stage1 + time_stage2 + time_stage3
    n_chains = 2 * 2 + 1  # 2 methods x 2 seeds + 1 diagnostic
    total_est = time_per_chain * n_chains

    # Add 20% for eval overhead
    total_est *= 1.2

    print(f"\n  TIME PROJECTIONS (default tokens: {DEFAULT_STAGE1_TOKENS/1e6:.0f}M/{DEFAULT_STAGE2_TOKENS/1e6:.0f}M/{DEFAULT_STAGE3_TOKENS/1e6:.0f}M):")
    print(f"  Per-chain estimate: {time_per_chain/3600:.1f}h "
          f"(S1={time_stage1/3600:.1f}h + S2={time_stage2/3600:.1f}h + S3={time_stage3/3600:.1f}h)")
    print(f"  Total ({n_chains} chains + eval): {total_est/3600:.1f}h")

    if total_est > 50 * 3600:
        # Suggest reduced tokens
        reduction = (50 * 3600) / total_est
        s1 = int(DEFAULT_STAGE1_TOKENS * reduction)
        s2 = int(DEFAULT_STAGE2_TOKENS * reduction)
        s3 = int(DEFAULT_STAGE3_TOKENS * reduction)
        print(f"\n  WARNING: Exceeds 50h budget! Suggested reduced tokens:")
        print(f"  --stage1_tokens {s1} --stage2_tokens {s2} --stage3_tokens {s3}")
        print(f"  Projected time: ~48h")
    elif total_est < 40 * 3600:
        # Can increase tokens
        increase = (45 * 3600) / total_est
        s1 = int(DEFAULT_STAGE1_TOKENS * increase)
        s2 = int(DEFAULT_STAGE2_TOKENS * increase)
        s3 = int(DEFAULT_STAGE3_TOKENS * increase)
        print(f"\n  OK: Under 40h budget! Can increase tokens:")
        print(f"  --stage1_tokens {s1} --stage2_tokens {s2} --stage3_tokens {s3}")
        print(f"  Projected time: ~45h")
    else:
        print(f"\n  OK: Within 40-50h budget. Default tokens are good.")

    # Cleanup
    del model, train_data
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    pilot_result = {
        "throughput_tok_per_sec": round(tps, 1),
        "pilot_time_sec": round(train_time, 1),
        "projected_per_chain_sec": round(time_per_chain, 1),
        "projected_total_sec": round(total_est, 1),
        "projected_total_hours": round(total_est / 3600, 1),
    }

    pilot_path = work_dir / "pilot" / "pilot_result.json"
    pilot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pilot_path, "w") as f:
        json.dump(pilot_result, f, indent=2)

    return pilot_result


# ---------------------------------------------------------------------------
# Summary: aggregate results across methods/seeds
# ---------------------------------------------------------------------------

def generate_summary(work_dir: Path):
    """Aggregate all results and print comparison tables."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY: 350M Progressive Chain")
    print("=" * 70)

    all_results = {}
    for method in METHODS:
        for seed in SEEDS:
            if method == "evq_r" and seed != 42:
                continue  # EVQ-R only runs seed 42
            tag = f"{method}_seed{seed}"
            run_dir = work_dir / tag

            for stage in STAGES:
                stage_name = stage["name"]
                result_path = run_dir / f"{stage_name}_L{stage['seq_len']}" / "result.json"
                if result_path.exists():
                    with open(result_path) as f:
                        r = json.load(f)
                    key = (method, seed, stage_name)
                    all_results[key] = r

    if not all_results:
        print("  No results found!")
        return

    # PPL comparison table per stage
    for stage in STAGES:
        sn = stage["name"]
        sl = stage["seq_len"]
        print(f"\n  --- {sn} (L_train={sl}) ---")
        print(f"  {'Method':<12} {'Seed':<6} {'tau':<8}", end="")
        for el in EVAL_LENGTHS[sl]:
            print(f"  PPL@{el:<6}", end="")
        print()

        for method in METHODS:
            for seed in SEEDS:
                if method == "evq_r" and seed != 42:
                    continue
                key = (method, seed, sn)
                if key not in all_results:
                    continue
                r = all_results[key]
                ppl = r.get("ppl", {})
                tau = r.get("tau", "?")
                print(f"  {method:<12} {seed:<6} {tau:<8.4f}", end="")
                for el in EVAL_LENGTHS[sl]:
                    v = ppl.get(str(el), None)
                    if isinstance(v, (int, float)):
                        print(f"  {v:<10.2f}", end="")
                    else:
                        print(f"  {'---':<10}", end="")
                print()

    # Mean +/- std for 2-seed methods
    print(f"\n  --- Mean +/- Std (seeds {SEEDS}) ---")
    for stage in STAGES:
        sn = stage["name"]
        sl = stage["seq_len"]
        print(f"\n  {sn} (L_train={sl}):")
        for method in ["geo", "evq_d"]:
            ppls_by_len = {}
            for seed in SEEDS:
                key = (method, seed, sn)
                if key in all_results:
                    for el_str, v in all_results[key].get("ppl", {}).items():
                        ppls_by_len.setdefault(el_str, []).append(v)

            if ppls_by_len:
                print(f"    {method:<12}", end="")
                for el in EVAL_LENGTHS[sl]:
                    vals = ppls_by_len.get(str(el), [])
                    if len(vals) >= 2:
                        m = np.mean(vals)
                        s = np.std(vals)
                        print(f"  {m:.2f}+/-{s:.2f}", end="")
                    elif len(vals) == 1:
                        print(f"  {vals[0]:.2f}", end="")
                    else:
                        print(f"  {'---':>10}", end="")
                print()

    # Protocol comparison (EVQ-D vs EVQ-R at seed 42)
    print(f"\n  --- Protocol Comparison (seed=42): Delayed vs Retarget ---")
    for stage in STAGES:
        sn = stage["name"]
        sl = stage["seq_len"]
        evq_d = all_results.get(("evq_d", 42, sn), {}).get("ppl", {})
        evq_r = all_results.get(("evq_r", 42, sn), {}).get("ppl", {})
        if evq_d and evq_r:
            print(f"  {sn}:", end="")
            for el in EVAL_LENGTHS[sl]:
                d_val = evq_d.get(str(el))
                r_val = evq_r.get(str(el))
                if d_val and r_val:
                    delta = ((r_val / d_val) - 1) * 100
                    winner = "D" if d_val < r_val else "R"
                    print(f"  @{el}: D={d_val:.1f} R={r_val:.1f} ({delta:+.1f}% -> {winner})", end="")
            print()

    # Save summary
    summary = {
        "experiment": "exp4_progressive_350m",
        "methods": list(METHODS.keys()),
        "seeds": SEEDS,
        "stages": [s["seq_len"] for s in STAGES],
        "all_results": {
            f"{m}_s{s}_{sn}": v
            for (m, s, sn), v in all_results.items()
        },
    }
    summary_path = work_dir / "exp4_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EXP-4: 350M Progressive Chain")
    parser.add_argument("--pilot", action="store_true",
                        help="Run pilot to measure MPS throughput (~30min)")
    parser.add_argument("--summary", action="store_true",
                        help="Generate summary from existing results")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print plan without training")
    parser.add_argument("--stage1_tokens", type=int, default=DEFAULT_STAGE1_TOKENS,
                        help=f"Stage 1 tokens (default: {DEFAULT_STAGE1_TOKENS/1e6:.0f}M)")
    parser.add_argument("--stage2_tokens", type=int, default=DEFAULT_STAGE2_TOKENS,
                        help=f"Stage 2 tokens (default: {DEFAULT_STAGE2_TOKENS/1e6:.0f}M)")
    parser.add_argument("--stage3_tokens", type=int, default=DEFAULT_STAGE3_TOKENS,
                        help=f"Stage 3 tokens (default: {DEFAULT_STAGE3_TOKENS/1e6:.0f}M)")
    parser.add_argument("--methods", type=str, default="geo,evq_d,evq_r",
                        help="Comma-separated methods to run (default: geo,evq_d,evq_r)")
    parser.add_argument("--seeds", type=str, default="42,123",
                        help="Comma-separated seeds (default: 42,123)")
    parser.add_argument("--work_dir", type=str, default=None,
                        help="Override work directory")
    args = parser.parse_args()

    # Work directory
    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        repo_root = SCRIPT_DIR.parent.parent
        work_dir = repo_root / "results" / "m4_max_36gb" / "exp4_progressive_350m"
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"[exp4] Work dir: {work_dir}")
    print(f"[exp4] Device: {DEVICE}, dtype: {DTYPE}")

    # Summary mode
    if args.summary:
        generate_summary(work_dir)
        return

    # Load tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Pilot mode
    if args.pilot:
        run_pilot(work_dir, tok)
        return

    # Parse methods and seeds
    methods = [m.strip() for m in args.methods.split(",") if m.strip() in METHODS]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    stage_tokens = {
        "stage1": args.stage1_tokens,
        "stage2": args.stage2_tokens,
        "stage3": args.stage3_tokens,
    }

    total_tokens = sum(stage_tokens.values())
    print(f"\n[exp4] Configuration:")
    print(f"  Methods: {methods}")
    print(f"  Seeds: {seeds}")
    print(f"  Tokens per chain: {total_tokens/1e6:.0f}M "
          f"(S1={stage_tokens['stage1']/1e6:.0f}M + "
          f"S2={stage_tokens['stage2']/1e6:.0f}M + "
          f"S3={stage_tokens['stage3']/1e6:.0f}M)")
    print(f"  Grad accum: L512={GRAD_ACCUM[512]}, L1024={GRAD_ACCUM[1024]}, "
          f"L2048={GRAD_ACCUM[2048]}")
    print(f"  Passkey mix ratio: {PASSKEY_RATIO:.0%}")

    # Load validation data (shared across all runs)
    print("\n  Loading validation data...")
    val_data = load_val(tok, 5_000_000, "fineweb-edu",
                        cache_dir=str(work_dir / "data_cache"))

    # Bug 1 fix: Preload all training data once, split by stage
    preloaded_data = None
    if not args.dry_run:
        print("\n  Preloading training data (non-overlapping per stage)...")
        preloaded_data = preload_stage_data(
            tok, stage_tokens,
            cache_dir=str(work_dir / "data_cache"),
        )

    # Run all method x seed combinations
    all_results = {}
    run_order = []
    for method in methods:
        for seed in seeds:
            # EVQ-R only runs seed 42 (diagnostic)
            if method == "evq_r" and seed != 42:
                continue
            run_order.append((method, seed))

    n_runs = len(run_order)
    print(f"\n  Total chains to run: {n_runs}")
    if args.dry_run:
        print("  [DRY RUN MODE]")

    for i, (method, seed) in enumerate(run_order):
        print(f"\n{'#'*70}")
        print(f"  CHAIN {i+1}/{n_runs}: {method} seed={seed}")
        print(f"{'#'*70}")

        result = run_method_seed(
            method=method,
            seed=seed,
            stage_tokens=stage_tokens,
            work_dir=work_dir,
            val_data=val_data,
            tokenizer=tok,
            dry_run=args.dry_run,
            preloaded_stage_data=preloaded_data,
        )
        all_results[f"{method}_seed{seed}"] = result

    # Dry run: verify inv_freq divergence between EVQ-D and EVQ-R
    if args.dry_run:
        print(f"\n{'='*70}")
        print("  [DRY RUN] inv_freq divergence verification at Stage 2:")
        for m in ["evq_d", "evq_r"]:
            if m in [r[0] for r in run_order]:
                inv = get_inv_freq(m, 1024)
                tau = get_tau(m, 1024)
                h = hashlib.md5(inv.numpy().tobytes()).hexdigest()[:12]
                print(f"    {m}: tau={tau:.4f}, hash={h}, "
                      f"max={inv.max().item():.6f}, min={inv.min().item():.8f}")
        evq_d_inv = get_inv_freq("evq_d", 1024)
        evq_r_inv = get_inv_freq("evq_r", 1024)
        if torch.allclose(evq_d_inv, evq_r_inv):
            print("  ERROR: EVQ-D and EVQ-R have IDENTICAL inv_freq at Stage 2!")
        else:
            diff = (evq_d_inv - evq_r_inv).abs().max().item()
            print(f"  OK: EVQ-D and EVQ-R inv_freq differ (max_diff={diff:.6f})")
        print(f"{'='*70}")

    # Generate final summary
    if not args.dry_run:
        generate_summary(work_dir)

    print("\n[exp4] Done!")


if __name__ == "__main__":
    main()
