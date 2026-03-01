#!/usr/bin/env python3
"""Sanity checks for learnable τ before full experiment.

Check 1: Different init τ0 → converge to same region
Check 2: τ(t) + grad_τ(t) trajectory (smooth, no sign-flip chaos)
Check 3: Inference-time τ-sweep on trained model → PPL minimum near learned τ

Usage:
    python experiments/sanity_check_learnable.py --work_dir /root/autodl-tmp/sanity_check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SWEEP_SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "m4_evq_sweep" / "run_evq_sweep.py")

# Sanity check uses 15M tokens (1/3 of 50m tier) — enough to see τ convergence
TRAIN_TOKENS = "15000000"
DATASET = "fineweb-edu"


def run_cmd(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print(f"  FAILED with rc={ret.returncode}")
    return ret.returncode


def check1_multi_init(work_dir: Path):
    """Check 1: Three different τ0 → converge to same region."""
    print("\n" + "#"*60)
    print("  CHECK 1: Multi-init convergence")
    print("#"*60)

    inits = [0.01, 1.0, 2.0]
    for tau_init in inits:
        run_cmd([
            sys.executable, SWEEP_SCRIPT,
            "--tier", "50m", "--learnable",
            "--tau_init", str(tau_init), "--tau_lr_mult", "100",
            "--dataset", DATASET, "--train_tokens", TRAIN_TOKENS,
            "--work_dir", str(work_dir),
            "--seeds", "42", "--base", "500000.0",
            "--resume",
        ], desc=f"Learnable τ0={tau_init}")

    # Collect results
    print("\n  --- Check 1 Results ---")
    final_taus = []
    for tau_init in inits:
        run_id = f"50m_learnable_init{tau_init:.2f}_seed42"
        traj_path = work_dir / run_id / "tau_trajectory.json"
        if traj_path.exists():
            with open(traj_path) as f:
                traj = json.load(f)
            final_tau = traj[-1]["tau"]
            final_taus.append(final_tau)
            print(f"  τ0={tau_init:.2f} → τ_final={final_tau:.4f} (steps={traj[-1]['step']})")
        else:
            print(f"  τ0={tau_init:.2f} → MISSING trajectory")

    if len(final_taus) >= 2:
        spread = max(final_taus) - min(final_taus)
        mean_tau = sum(final_taus) / len(final_taus)
        print(f"\n  Spread: {spread:.4f}  Mean: {mean_tau:.4f}")
        if spread < 0.3:
            print(f"  PASS: all inits converge to τ≈{mean_tau:.2f} (spread < 0.3)")
        else:
            print(f"  WARN: spread={spread:.4f} > 0.3, may not converge")
    return final_taus


def check2_gradient_trajectory(work_dir: Path, tau_init: float = 1.0):
    """Check 2: τ(t) and grad_τ(t) smoothness."""
    print("\n" + "#"*60)
    print("  CHECK 2: Gradient trajectory analysis")
    print("#"*60)

    run_id = f"50m_learnable_init{tau_init:.2f}_seed42"
    traj_path = work_dir / run_id / "tau_trajectory.json"

    if not traj_path.exists():
        print(f"  SKIP: {traj_path} not found")
        return

    with open(traj_path) as f:
        traj = json.load(f)

    steps = [e["step"] for e in traj]
    taus = [e["tau"] for e in traj]
    grads = [e.get("grad_tau", None) for e in traj]
    grads_valid = [g for g in grads if g is not None]

    print(f"  Total entries: {len(traj)}")
    print(f"  τ range: [{min(taus):.4f}, {max(taus):.4f}]")

    if grads_valid:
        # Count sign flips
        sign_flips = 0
        for i in range(1, len(grads_valid)):
            if grads_valid[i] * grads_valid[i-1] < 0:
                sign_flips += 1
        flip_rate = sign_flips / max(len(grads_valid) - 1, 1)

        print(f"  Grad entries: {len(grads_valid)}")
        print(f"  Grad range: [{min(grads_valid):.6f}, {max(grads_valid):.6f}]")
        print(f"  Sign flips: {sign_flips}/{len(grads_valid)-1} ({flip_rate:.1%})")

        if flip_rate < 0.7:
            print(f"  PASS: gradient direction stable (flip rate < 70%)")
        else:
            print(f"  WARN: high sign flip rate ({flip_rate:.1%}), may be noisy")

        # Check τ monotonicity in last 50%
        n_half = len(taus) // 2
        late_taus = taus[n_half:]
        late_std = (sum((t - sum(late_taus)/len(late_taus))**2 for t in late_taus) / len(late_taus)) ** 0.5
        print(f"  Late-stage τ std: {late_std:.4f}")
        if late_std < 0.05:
            print(f"  PASS: τ converged (late std < 0.05)")
        else:
            print(f"  WARN: τ still moving (late std={late_std:.4f})")
    else:
        print(f"  NO gradient data in trajectory (re-run with updated code)")

    # Save plot data for manual inspection
    plot_data = {"steps": steps, "taus": taus, "grads": grads}
    plot_path = work_dir / f"check2_gradient_data_{tau_init:.2f}.json"
    with open(plot_path, "w") as f:
        json.dump(plot_data, f)
    print(f"  Saved: {plot_path}")


def check3_inference_sweep(work_dir: Path, tau_init: float = 1.0):
    """Check 3: Inference-time τ-sweep on trained checkpoint."""
    print("\n" + "#"*60)
    print("  CHECK 3: Inference-time τ-sweep")
    print("#"*60)

    run_id = f"50m_learnable_init{tau_init:.2f}_seed42"
    ckpt_path = work_dir / run_id / "model.pt"

    if not ckpt_path.exists():
        print(f"  SKIP: {ckpt_path} not found")
        return

    # Add paths
    proj_root = str(Path(__file__).resolve().parents[1])
    script_dir = str(Path(proj_root) / "scripts" / "m4_evq_sweep")
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    import torch
    import numpy as np

    from run_evq_sweep import (
        GPT, TIER_CONFIGS, evq_cosh_inv_freq, eval_model,
        get_device_and_dtype, load_val,
    )

    DEVICE, DTYPE = get_device_and_dtype()
    cfg = TIER_CONFIGS["50m"].copy()

    # Load val data
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    val_data = load_val(tok, dataset=DATASET, cache_dir=str(work_dir))

    # Read learned tau
    traj_path = work_dir / run_id / "tau_trajectory.json"
    with open(traj_path) as f:
        traj = json.load(f)
    learned_tau = traj[-1]["tau"]

    # Sweep τ values at inference time
    sweep_taus = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    results = {}

    for tau in sweep_taus:
        inv_freq = evq_cosh_inv_freq(cfg["head_dim"], tau, 500000.0)
        model = GPT(cfg, inv_freq).to(DEVICE)

        # Load trained weights (skip rope keys since we're replacing inv_freq)
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        # Filter out rope-related keys (inv_freq, cos_c, sin_c, EVQ keys)
        compatible = {k: v for k, v in state.items()
                      if not any(x in k for x in ["rope.", "inv_freq", "cos_c", "sin_c",
                                                    "raw_tau", "evq.", "pos"])}
        model.load_state_dict(compatible, strict=False)

        ppl = eval_model(model, val_data, [2048, 4096], eval_chunks=5)
        results[f"{tau:.2f}"] = ppl
        ppl_2k = ppl.get("2048", float("nan"))
        ppl_4k = ppl.get("4096", float("nan"))
        print(f"  τ={tau:.2f}  PPL@2K={ppl_2k:.2f}  PPL@4K={ppl_4k:.2f}")

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Find best τ
    ppl_4k_map = {float(k): v.get("4096", 999) for k, v in results.items()}
    best_tau = min(ppl_4k_map, key=ppl_4k_map.get)
    best_ppl = ppl_4k_map[best_tau]

    print(f"\n  Learned τ: {learned_tau:.4f}")
    print(f"  Best sweep τ: {best_tau:.2f} (PPL@4K={best_ppl:.2f})")
    print(f"  |learned - best_sweep| = {abs(learned_tau - best_tau):.4f}")

    if abs(learned_tau - best_tau) < 0.5:
        print(f"  PASS: learned τ near sweep optimum (gap < 0.5)")
    else:
        print(f"  WARN: gap={abs(learned_tau - best_tau):.4f} > 0.5")

    # Save
    sweep_path = work_dir / "check3_inference_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump({"learned_tau": learned_tau, "sweep": results,
                    "best_tau": best_tau, "best_ppl_4k": best_ppl}, f, indent=2)
    print(f"  Saved: {sweep_path}")


def main():
    parser = argparse.ArgumentParser(description="Sanity checks for learnable τ")
    parser.add_argument("--work_dir", type=str, default="/root/autodl-tmp/sanity_check")
    parser.add_argument("--check", type=str, default="all",
                        help="Which check to run: 1, 2, 3, or all")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if args.check in ("all", "1"):
        check1_multi_init(work_dir)

    if args.check in ("all", "2"):
        check2_gradient_trajectory(work_dir, tau_init=1.0)
        check2_gradient_trajectory(work_dir, tau_init=0.01)
        check2_gradient_trajectory(work_dir, tau_init=2.0)

    if args.check in ("all", "3"):
        check3_inference_sweep(work_dir, tau_init=1.0)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  ALL SANITY CHECKS DONE  |  {elapsed/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
