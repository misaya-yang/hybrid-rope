#!/usr/bin/env python3
"""
τ Theory Verification Experiment — M4 Max Overnight Run
========================================================

Verifies the NOVEL prediction that existing data does NOT cover:

  >>> At L=4096, d_head=64: the OLD formula gives τ*=1.0,
  >>> the NEW formula gives τ*=1.4 (floor kicks in).
  >>> This experiment DISTINGUISHES the two.

Design:
  Phase A: L=4096 sweep  (7 taus, ~3h) — THE FLOOR TEST
  Phase B: L=2048 sweep  (4 taus, ~1.5h) — confirmation
  Total: ~4.5h — fits well within one overnight session.

Usage:
  python scripts/analysis/exp_tau_theory_verify.py             # full run
  python scripts/analysis/exp_tau_theory_verify.py --phase A   # L=512 only
  python scripts/analysis/exp_tau_theory_verify.py --phase B   # L=2048 only
  python scripts/analysis/exp_tau_theory_verify.py --dry-run   # check setup

Requires: run_evq_sweep.py in scripts/core_text_phases/
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# ── project root ──
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "core_text_phases"))

import torch
import numpy as np

from run_evq_sweep import (
    GPT,
    RotaryEmbedding,
    evq_cosh_inv_freq,
)

# ══════════════════════════════════════════════════════════════════════
# Experiment configuration
# ══════════════════════════════════════════════════════════════════════

SEED = 42
BASE = 500_000.0
DATA_CACHE = str(ROOT / "results/theory/tau_sweep_verify/data_cache")

# ════════════════════════════════════════════════════════════════
# Benchmark-based design (M4 Max MPS, actual measured):
#   125M d64 L=512  bs=16: 1.6s/step → 0.5h/10M  ✓
#   125M d64 L=1024 bs=8:  1.7s/step → 0.6h/10M  ✓
#   50M  d64 L=2048 bs=8:  4.0s/step → 0.7h/10M  ✓
#   125M d32 L=1024 bs=8:  30s/step  → TOO SLOW   ✗
#   125M d64 L=2048 bs=4:  28s/step  → TOO SLOW   ✗
#
# Strategy: cross-L scaling law with d_head=64 only.
# The NEW data point is 50M@L=2048 (no existing τ sweep).
# ════════════════════════════════════════════════════════════════

_125M = dict(vocab_size=50304, hidden_size=768, num_layers=12,
             num_heads=12, head_dim=64, intermediate_size=3072)
_50M = dict(vocab_size=50304, hidden_size=512, num_layers=6,
            num_heads=8, head_dim=64, intermediate_size=2048)

# ── Phase A: 125M L=512 — formula τ*=2.83, full sweep ──
# Dense sweep to map the complete τ-PPL curve
# 8 runs × 0.5h = 4h
PHASE_A = dict(
    name="125M_L512_d64",
    seq_len=512,
    train_tokens=10_000_000,
    batch_size=16,
    lr=6e-4,
    taus=[0.0, 0.5, 1.0, 1.5, 2.0, 2.83, 3.5, 5.0],
    eval_lengths=[512, 1024, 2048, 4096, 8192],
    data_source_cache=DATA_CACHE,
    seed=42,
    model_cfg=_125M,
)

# ── Phase B: 50M L=2048 — formula τ*=1.41, NEW territory ──
# This is the MOST VALUABLE phase: first τ sweep at L=2048
# Tests the transition/floor region
# 6 runs × 0.7h = 4.2h
PHASE_B = dict(
    name="50M_L2048_d64",
    seq_len=2048,
    train_tokens=10_000_000,
    batch_size=8,
    lr=6e-4,
    taus=[0.0, 0.5, 1.0, 1.41, 2.0, 3.0],
    eval_lengths=[2048, 4096, 8192, 16384],
    data_source_cache=DATA_CACHE,
    seed=42,
    model_cfg=_50M,
)

# ── Phase C: 125M L=1024 — formula τ*=2.0, bridge point ──
# Links L=512 and L=2048, confirms d/√L tracking
# 5 runs × 0.6h = 3h
PHASE_C = dict(
    name="125M_L1024_d64",
    seq_len=1024,
    train_tokens=10_000_000,
    batch_size=8,
    lr=3e-4,
    taus=[0.0, 1.0, 2.0, 2.5, 3.0],
    eval_lengths=[1024, 2048, 4096, 8192],
    data_source_cache=DATA_CACHE,
    seed=42,
    model_cfg=_125M,
)

# ── Phase D: Seed=137 for Phase B (the new territory) ──
# Error bars at L=2048 where we have no prior data
# 3 runs × 0.7h = 2.1h
PHASE_D = dict(
    name="50M_L2048_d64_s137",
    seq_len=2048,
    train_tokens=10_000_000,
    batch_size=8,
    lr=6e-4,
    taus=[0.0, 1.0, 1.41],
    eval_lengths=[2048, 4096, 8192, 16384],
    data_source_cache=DATA_CACHE,
    seed=137,
    model_cfg=_50M,
)

RESULTS_DIR = ROOT / "results" / "theory" / "tau_theory_verify"


# ══════════════════════════════════════════════════════════════════════
# Training loop (adapted from run_evq_sweep.py)
# ══════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(tau: float, seq_len: int, model_cfg: dict, seed: int) -> GPT:
    """Build model with given τ and architecture config."""
    d_head = model_cfg["head_dim"]
    K = d_head // 2

    # Build inv_freq FIRST (GPT constructor needs it)
    if abs(tau) > 1e-8:
        inv_freq = evq_cosh_inv_freq(d_head, tau, BASE)
    else:
        inv_freq = torch.pow(
            torch.tensor(BASE, dtype=torch.float64),
            -torch.arange(K).float() / K,
        ).float()

    cfg = dict(
        **model_cfg,
        max_position_embeddings=max(seq_len * 8, 16384),
    )
    set_seed(seed)
    model = GPT(cfg, inv_freq)
    return model


@torch.no_grad()
def eval_ppl(model, val_data: torch.Tensor, seq_len: int, device: torch.device) -> float:
    """Evaluate perplexity at a specific sequence length."""
    model.eval()
    n_chunks = min(20, val_data.size(0) // seq_len)
    if n_chunks < 1:
        return float("nan")

    total_loss, total_tokens = 0.0, 0
    for i in range(n_chunks):
        chunk = val_data[i * seq_len : (i + 1) * seq_len].unsqueeze(0).to(device)
        x, y = chunk[:, :-1], chunk[:, 1:]
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    return math.exp(total_loss / total_tokens)


def train_one_run(
    tau: float,
    phase_cfg: dict,
    device: torch.device,
    run_dir: Path,
) -> dict:
    """Train one model and return results dict."""
    seq_len = phase_cfg["seq_len"]
    seed = phase_cfg.get("seed", SEED)
    d_head = phase_cfg["model_cfg"]["head_dim"]
    tag = f"tau{tau:.2f}_L{seq_len}_d{d_head}_s{seed}"
    result_file = run_dir / f"{tag}.json"

    # Skip if already completed
    if result_file.exists():
        print(f"  [{tag}] Already done, loading cached result")
        with open(result_file) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"  Training: τ={tau:.2f}, L={seq_len}, d_head={d_head}, seed={seed}")
    tau_formula = d_head / math.sqrt(seq_len)
    print(f"  Formula τ* = d/√L = {tau_formula:.3f}")
    print(f"{'='*60}")

    t0 = time.time()

    # Load data — directly from pre-cached .pt files (no network)
    cache_dir = Path(phase_cfg["data_source_cache"])
    # Find the largest cached file for this seq_len
    pattern = f"train_fineweb-edu_*_{seq_len}.pt"
    candidates = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No cached data for seq_len={seq_len} in {cache_dir}")
    train_file = candidates[0]
    print(f"  Loading: {train_file.name}")
    train_data = torch.load(str(train_file), weights_only=True)
    # Flatten to 1D for training loop
    if train_data.dim() == 2:
        train_data = train_data.reshape(-1)
    val_file = cache_dir / "val_fineweb-edu_5000000.pt"
    val_data = torch.load(str(val_file), weights_only=True)
    if val_data.dim() == 2:
        val_data = val_data.reshape(-1)

    # Build model
    model = build_model(tau, seq_len, phase_cfg["model_cfg"], seed).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: {n_params:.1f}M params, d_head={d_head}, device={device}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=phase_cfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Training
    bs = phase_cfg["batch_size"]
    total_tokens = phase_cfg["train_tokens"]
    tokens_per_step = bs * seq_len
    total_steps = total_tokens // tokens_per_step
    warmup_steps = max(1, total_steps // 50)

    model.train()
    step_losses = []
    data_ptr = 0

    for step in range(total_steps):
        # LR schedule: linear warmup + cosine decay
        if step < warmup_steps:
            lr = phase_cfg["lr"] * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr = phase_cfg["lr"] * 0.1 + 0.9 * phase_cfg["lr"] * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Get batch
        batch_tokens = bs * (seq_len + 1)
        if data_ptr + batch_tokens > train_data.numel():
            data_ptr = 0
        chunk = train_data[data_ptr : data_ptr + batch_tokens].reshape(bs, seq_len + 1)
        data_ptr += batch_tokens

        x = chunk[:, :-1].to(device)
        y = chunk[:, 1:].to(device)

        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step_losses.append(loss.item())

        if (step + 1) % max(1, total_steps // 10) == 0:
            avg = np.mean(step_losses[-100:])
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * (total_steps - step - 1)
            print(
                f"  step {step+1}/{total_steps}  loss={avg:.4f}  "
                f"lr={lr:.2e}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m"
            )

    train_time = time.time() - t0

    # Evaluation
    print(f"  Evaluating at lengths {phase_cfg['eval_lengths']}...")
    ppls = {}
    for eval_len in phase_cfg["eval_lengths"]:
        ppl = eval_ppl(model, val_data, eval_len, device)
        ppls[str(eval_len)] = round(ppl, 3)
        print(f"    L={eval_len}: PPL = {ppl:.3f}")

    # Build result
    result = dict(
        tau=tau,
        tau_formula=round(tau_formula, 4),
        seq_len=seq_len,
        d_head=d_head,
        base=BASE,
        seed=seed,
        train_tokens=total_tokens,
        train_time_min=round(train_time / 60, 1),
        final_train_loss=round(float(np.mean(step_losses[-50:])), 4),
        ppls=ppls,
        n_params_M=round(n_params, 1),
    )

    # Save
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    # Free memory
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return result


# ══════════════════════════════════════════════════════════════════════
# Summary & theory comparison
# ══════════════════════════════════════════════════════════════════════

def summarize_phase(results: list[dict], phase_name: str):
    """Print summary table and theory comparison."""
    if not results:
        return

    seq_len = results[0]["seq_len"]
    d_head = results[0]["d_head"]
    tau_formula = d_head / math.sqrt(seq_len)
    eval_lens = sorted(results[0]["ppls"].keys(), key=int)

    print(f"\n{'='*72}")
    print(f"RESULTS: {phase_name} (L={seq_len}, formula τ* = {tau_formula:.3f})")
    print(f"{'='*72}")

    # Header
    header = f"  {'tau':>6s} | {'loss':>7s}"
    for el in eval_lens:
        header += f" | {'PPL@'+el:>10s}"
    header += f" | {'note':>15s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Find best PPL at training length and at 4x extrapolation
    best_train = (None, 1e9)
    best_extrap = (None, 1e9)
    extrap_key = str(seq_len * 4)

    for r in sorted(results, key=lambda x: x["tau"]):
        tau = r["tau"]
        note = ""
        if abs(tau - tau_formula) < 0.05:
            note = "← formula"
        elif abs(tau) < 1e-8:
            note = "← geometric"

        ppl_train = r["ppls"].get(str(seq_len), float("nan"))
        ppl_extrap = r["ppls"].get(extrap_key, float("nan"))

        if ppl_train < best_train[1]:
            best_train = (tau, ppl_train)
        if ppl_extrap < best_extrap[1]:
            best_extrap = (tau, ppl_extrap)

        row = f"  {tau:6.2f} | {r['final_train_loss']:7.4f}"
        for el in eval_lens:
            ppl = r["ppls"].get(el, float("nan"))
            row += f" | {ppl:10.3f}"
        row += f" | {note:>15s}"
        print(row)

    print(f"\n  Best PPL@{seq_len} (in-range):        τ = {best_train[0]:.2f} (PPL {best_train[1]:.3f})")
    print(f"  Best PPL@{extrap_key} (extrapolation): τ = {best_extrap[0]:.2f} (PPL {best_extrap[1]:.3f})")
    print(f"  Formula prediction:                  τ = {tau_formula:.2f}")

    # Theory verification
    print(f"\n  --- Theory verification ---")
    geo = [r for r in results if abs(r["tau"]) < 1e-8]
    formula_run = [r for r in results if abs(r["tau"] - tau_formula) < 0.1]

    if geo and formula_run:
        geo_ppl = geo[0]["ppls"].get(extrap_key, float("nan"))
        frm_ppl = formula_run[0]["ppls"].get(extrap_key, float("nan"))
        if geo_ppl > 0 and frm_ppl > 0:
            improvement = (geo_ppl - frm_ppl) / geo_ppl * 100
            print(f"  EVQ (τ={tau_formula:.2f}) vs Geometric @{extrap_key}: {improvement:+.1f}%")

    # Habitable zone check
    hab_taus = [r for r in results if 1.0 <= r["tau"] <= 2.5]
    if hab_taus:
        hab_ppls = [r["ppls"].get(extrap_key, 1e9) for r in hab_taus]
        print(f"  Habitable zone τ∈[1.0, 2.5]: all extrapolation PPLs within "
              f"{min(hab_ppls):.1f} - {max(hab_ppls):.1f}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="τ Theory Verification")
    parser.add_argument("--phase", default="ABCD",
                        help="Any combo of A/B/C/D (default=ABCD). "
                             "A=L4096 floor, B=L2048 confirm, "
                             "C=d32 floor, D=seed137 replication")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan only, don't train")
    parser.add_argument("--tokens", type=int, default=None,
                        help="Override train_tokens per run")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    phases = []
    if "A" in args.phase:
        phases.append(("Phase_A: L4096 d64 FLOOR TEST", PHASE_A))
    if "B" in args.phase:
        phases.append(("Phase_B: L2048 d64 confirm", PHASE_B))
    if "C" in args.phase:
        phases.append(("Phase_C: L2048 d32 FLOOR TEST #2", PHASE_C))
    if "D" in args.phase:
        phases.append(("Phase_D: L4096 d64 seed137", PHASE_D))

    # Summary
    total_runs = sum(len(p["taus"]) for _, p in phases)
    print(f"\n{'='*72}")
    print(f"τ THEORY VERIFICATION EXPERIMENT")
    print(f"{'='*72}")
    print(f"  Model: 125M, Base: {BASE:.0f}")
    print(f"  Total runs: {total_runs}")

    for name, cfg in phases:
        tokens = args.tokens or cfg["train_tokens"]
        d_h = cfg["model_cfg"]["head_dim"]
        n_h = cfg["model_cfg"]["num_heads"]
        tau_f = d_h / math.sqrt(cfg["seq_len"])
        print(f"\n  {name}:")
        print(f"    L={cfg['seq_len']}, d_head={d_h}, {n_h}H, seed={cfg.get('seed', SEED)}")
        print(f"    d/√L = {tau_f:.3f}, max(d/√L, 1.4) = {max(tau_f, 1.4):.3f}")
        print(f"    τ sweep: {cfg['taus']}")
        print(f"    {tokens/1e6:.0f}M tokens, bs={cfg['batch_size']}, eval={cfg['eval_lengths']}")

    if args.dry_run:
        print("\n  [DRY RUN] — exiting without training.")
        return

    # Run
    t_start = time.time()
    run_dir = RESULTS_DIR

    for name, cfg in phases:
        if args.tokens:
            cfg = {**cfg, "train_tokens": args.tokens}

        print(f"\n\n{'#'*72}")
        print(f"# {name}")
        print(f"{'#'*72}")

        phase_results = []
        for tau in cfg["taus"]:
            result = train_one_run(tau, cfg, device, run_dir)
            phase_results.append(result)

        summarize_phase(phase_results, name)

    total_time = (time.time() - t_start) / 3600
    print(f"\n\nTotal experiment time: {total_time:.1f} hours")

    # Final consolidated summary
    print(f"\n{'='*72}")
    print("FINAL: Theory predictions vs experiment")
    print(f"{'='*72}")
    all_results = []
    for f in sorted(run_dir.glob("*.json")):
        with open(f) as fh:
            all_results.append(json.load(fh))

    if all_results:
        print(f"\n  {'d':>4s} {'L':>6s} {'seed':>5s} | {'d/√L':>6s} {'max()':>6s} | {'τ*_best':>8s} | {'verdict':>10s}")
        print("  " + "-" * 60)
        groups = set((r["d_head"], r["seq_len"], r["seed"]) for r in all_results)
        for d_h, sl, sd in sorted(groups):
            phase_r = [r for r in all_results if r["d_head"] == d_h and r["seq_len"] == sl and r["seed"] == sd]
            tau_f = d_h / math.sqrt(sl)
            tau_new = max(tau_f, 1.4)
            extrap = str(sl * 4)
            best_tau, best_ppl = None, 1e9
            for r in phase_r:
                p = r["ppls"].get(extrap, 1e9)
                if p < best_ppl:
                    best_ppl = p
                    best_tau = r["tau"]
            if best_tau is not None:
                gap_old = abs(best_tau - tau_f) / max(tau_f, 0.1) * 100
                gap_new = abs(best_tau - tau_new) / tau_new * 100
                v = "NEW ✓" if gap_new < gap_old else "OLD ✓"
                print(f"  {d_h:4d} {sl:6d} {sd:5d} | {tau_f:6.2f} {tau_new:6.2f} | {best_tau:8.2f} | {v:>10s}")


if __name__ == "__main__":
    main()
