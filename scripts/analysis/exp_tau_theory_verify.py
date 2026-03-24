#!/usr/bin/env python3
"""
τ Theory Verification Experiment — M4 Max Overnight Run
========================================================

Verifies the core predictions of the Softmax Transport theory:
  1. τ* = d_head/√L is optimal (or near-optimal) at each L
  2. Habitable zone: τ ∈ [1.0, 2.5] always works
  3. τ=0 (geometric) is suboptimal at extrapolation
  4. Floor effect: at large L, τ_floor ≈ 1.4 overrides d/√L

Design:
  Phase A: L=512 sweep  (8 taus × 1 seed, ~2.5h on M4 Max)
  Phase B: L=2048 sweep (6 taus × 1 seed, ~2h on M4 Max)
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
    load_data,
    load_val,
)

# ══════════════════════════════════════════════════════════════════════
# Experiment configuration
# ══════════════════════════════════════════════════════════════════════

SEED = 42
D_HEAD = 64
BASE = 500_000.0

# 125M architecture
MODEL_CFG = dict(
    vocab_size=50304,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    head_dim=D_HEAD,
    intermediate_size=3072,
)

# Phase A: L=512 — formula predicts τ* = 64/√512 = 2.83
PHASE_A = dict(
    name="L512",
    seq_len=512,
    train_tokens=50_000_000,
    batch_size=16,           # M4 Max can handle 16 at L=512
    lr=6e-4,
    taus=[0.0, 0.5, 1.0, 1.5, 2.0, 2.83, 3.5, 5.0],
    eval_lengths=[512, 1024, 2048, 4096, 8192],
    data_source_cache=str(ROOT / "results/core_text/phase18_base_sweep/data_cache"),
)

# Phase B: L=2048 — formula predicts τ* = 64/√2048 = 1.41
PHASE_B = dict(
    name="L2048",
    seq_len=2048,
    train_tokens=50_000_000,
    batch_size=4,            # M4 Max 36GB limit at L=2048
    lr=3e-4,
    taus=[0.0, 0.5, 1.0, 1.41, 2.0, 3.0],
    eval_lengths=[2048, 4096, 8192, 16384],
    data_source_cache=str(ROOT / "results/theory/tau_sweep_verify/data_cache"),
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


def build_model(tau: float, seq_len: int) -> GPT:
    """Build 125M model with given τ."""
    cfg = dict(
        **MODEL_CFG,
        max_position_embeddings=max(seq_len * 8, 16384),
        rope_base=BASE,
    )
    set_seed(SEED)
    model = GPT(**cfg)

    # Inject EVQ frequencies
    if abs(tau) > 1e-8:
        inv_freq = evq_cosh_inv_freq(D_HEAD, tau, BASE)
    else:
        K = D_HEAD // 2
        inv_freq = torch.pow(
            torch.tensor(BASE, dtype=torch.float64),
            -torch.arange(K).float() / K,
        ).float()

    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.inv_freq = torch.nn.Parameter(inv_freq, requires_grad=False)
            module._cos = None
            module._sin = None
            module._max = 0

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
    tag = f"tau{tau:.2f}_L{seq_len}"
    result_file = run_dir / f"{tag}.json"

    # Skip if already completed
    if result_file.exists():
        print(f"  [{tag}] Already done, loading cached result")
        with open(result_file) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"  Training: τ={tau:.2f}, L={seq_len}")
    tau_formula = D_HEAD / math.sqrt(seq_len)
    print(f"  Formula τ* = d/√L = {tau_formula:.3f}")
    print(f"{'='*60}")

    t0 = time.time()

    # Load data
    cache_dir = phase_cfg["data_source_cache"]
    train_data = load_data(
        cache_dir,
        max_tokens=phase_cfg["train_tokens"],
        seq_len=seq_len,
    )
    val_data = load_val(cache_dir, max_tokens=5_000_000)

    # Build model
    model = build_model(tau, seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: {n_params:.1f}M params, device={device}")

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
        d_head=D_HEAD,
        base=BASE,
        seed=SEED,
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
    tau_formula = D_HEAD / math.sqrt(seq_len)
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
    parser.add_argument("--phase", choices=["A", "B", "AB"], default="AB",
                        help="A=L512, B=L2048, AB=both (default)")
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
        phases.append(("Phase_A_L512", PHASE_A))
    if "B" in args.phase:
        phases.append(("Phase_B_L2048", PHASE_B))

    # Summary
    total_runs = sum(len(p["taus"]) for _, p in phases)
    print(f"\n{'='*72}")
    print(f"τ THEORY VERIFICATION EXPERIMENT")
    print(f"{'='*72}")
    print(f"  Model: 125M (d_head={D_HEAD}, 12L, 12H)")
    print(f"  Base: {BASE:.0f}")
    print(f"  Seed: {SEED}")
    print(f"  Total runs: {total_runs}")

    for name, cfg in phases:
        tokens = args.tokens or cfg["train_tokens"]
        tau_f = D_HEAD / math.sqrt(cfg["seq_len"])
        print(f"\n  {name}:")
        print(f"    L = {cfg['seq_len']}, formula τ* = {tau_f:.3f}")
        print(f"    τ sweep: {cfg['taus']}")
        print(f"    {tokens/1e6:.0f}M tokens, batch_size={cfg['batch_size']}")
        print(f"    Eval: {cfg['eval_lengths']}")

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
        print(f"\n  {'L':>6s} | {'τ*_formula':>10s} | {'τ*_best':>8s} | {'gap':>6s} | {'verdict':>10s}")
        print("  " + "-" * 50)
        for sl in sorted(set(r["seq_len"] for r in all_results)):
            phase_r = [r for r in all_results if r["seq_len"] == sl]
            tau_f = D_HEAD / math.sqrt(sl)
            extrap = str(sl * 4)
            best_tau, best_ppl = None, 1e9
            for r in phase_r:
                p = r["ppls"].get(extrap, 1e9)
                if p < best_ppl:
                    best_ppl = p
                    best_tau = r["tau"]
            if best_tau is not None:
                gap = abs(best_tau - tau_f) / tau_f * 100
                v = "✓" if gap < 30 else "✗"
                print(f"  {sl:6d} | {tau_f:10.3f} | {best_tau:8.2f} | {gap:5.1f}% | {v:>10s}")


if __name__ == "__main__":
    main()
