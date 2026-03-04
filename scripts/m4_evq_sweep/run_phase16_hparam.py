#!/usr/bin/env python3
"""Phase 16: Analytically Determined Hyperparameters validation.

Experiment A: r-sweep (r ∈ {0,4,8,14,16,24,32}, τ=1.5 fixed, 2 seeds)
  - Validates that r=0 (pure EVQ) is universally optimal or noise-equivalent.

Experiment B: τ* scaling law at 350M (L=512, L=1024, τ sweep, 1 seed)
  - Validates τ*(L) = d_head/√L at 350M scale (Phase 8D was 125M).

Usage:
    python run_phase16_hparam.py --experiment A   # r-sweep only
    python run_phase16_hparam.py --experiment B   # τ-scaling only
    python run_phase16_hparam.py --experiment AB  # both sequential

    # Dry run to verify commands:
    python run_phase16_hparam.py --experiment AB --dry_run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_SWEEP = SCRIPT_DIR / "run_evq_sweep.py"

# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

# Experiment A: r-sweep
EXP_A = {
    "tier": "350m",
    "r_values": "0,4,8,14,16,24,32",
    "fixed_tau": 1.5,
    "seeds": "42,123",
    "base": 500000.0,
    "passkey_mix_ratio": 0.10,
    "train_tokens": 50_000_000,
    "seq_len": 2048,  # default for 350m
    "work_dir_suffix": "phase16_rsweep",
}

# Experiment B: τ-scaling at different L_train
EXP_B_CONFIGS = [
    {
        "label": "B1 (L=512)",
        "tier": "350m",
        "seq_len": 512,
        "taus": "0.0,1.5,2.83,4.0",
        "seeds": "42",
        "base": 500000.0,
        "passkey_mix_ratio": 0.10,
        "train_tokens": 50_000_000,
        "work_dir_suffix": "phase16_tau_L512",
    },
    {
        "label": "B2 (L=1024)",
        "tier": "350m",
        "seq_len": 1024,
        "taus": "0.0,1.5,2.0,3.0",
        "seeds": "42",
        "base": 500000.0,
        "passkey_mix_ratio": 0.10,
        "train_tokens": 50_000_000,
        "work_dir_suffix": "phase16_tau_L1024",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_work_dir(suffix: str, base_dir: str | None) -> Path:
    """Resolve work directory from base or default."""
    if base_dir:
        return Path(base_dir) / suffix
    # Default: /root/autodl-tmp/<suffix> on server, ~/evq_m4_sweep/<suffix> locally
    autodl = Path("/root/autodl-tmp")
    if autodl.exists():
        return autodl / suffix
    return Path.home() / "evq_m4_sweep" / suffix


def build_cmd_exp_a(cfg: dict, work_dir: Path, resume: bool) -> list[str]:
    """Build command for Experiment A (r-sweep)."""
    cmd = [
        sys.executable, str(RUN_SWEEP),
        "--tier", cfg["tier"],
        "--r_values", cfg["r_values"],
        "--fixed_tau", str(cfg["fixed_tau"]),
        "--seeds", cfg["seeds"],
        "--base", str(cfg["base"]),
        "--passkey_mix_ratio", str(cfg["passkey_mix_ratio"]),
        "--train_tokens", str(cfg["train_tokens"]),
        "--work_dir", str(work_dir),
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def build_cmd_exp_b(cfg: dict, work_dir: Path, resume: bool) -> list[str]:
    """Build command for Experiment B (τ-scaling at custom L)."""
    cmd = [
        sys.executable, str(RUN_SWEEP),
        "--tier", cfg["tier"],
        "--taus", cfg["taus"],
        "--seeds", cfg["seeds"],
        "--base", str(cfg["base"]),
        "--seq_len", str(cfg["seq_len"]),
        "--passkey_mix_ratio", str(cfg["passkey_mix_ratio"]),
        "--train_tokens", str(cfg["train_tokens"]),
        "--work_dir", str(work_dir),
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def run_cmd(cmd: list[str], label: str, dry_run: bool = False) -> int:
    """Run a subprocess command with logging."""
    cmd_str = " \\\n    ".join(cmd)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Command:\n    {cmd_str}\n")

    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return 0

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    elapsed = time.time() - t0
    print(f"\n  [{label}] Finished in {elapsed / 60:.1f} min, "
          f"return code: {result.returncode}")
    return result.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 16: Analytically Determined Hyperparameters validation"
    )
    parser.add_argument(
        "--experiment", type=str, required=True,
        choices=["A", "B", "AB", "a", "b", "ab"],
        help="Which experiment(s) to run: A (r-sweep), B (τ-scaling), AB (both)"
    )
    parser.add_argument(
        "--base_dir", type=str, default=None,
        help="Base directory for work dirs (default: /root/autodl-tmp or ~/evq_m4_sweep)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume: skip runs that already have results"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing"
    )
    args = parser.parse_args()
    exp = args.experiment.upper()

    print(f"\n{'#'*60}")
    print(f"  PHASE 16: Analytically Determined Hyperparameters")
    print(f"  Experiments: {exp}")
    print(f"  Resume: {args.resume}  Dry run: {args.dry_run}")
    print(f"{'#'*60}")

    t_total = time.time()
    codes = []

    # --- Experiment A: r-sweep ---
    if "A" in exp:
        work_dir_a = resolve_work_dir(EXP_A["work_dir_suffix"], args.base_dir)
        work_dir_a.mkdir(parents=True, exist_ok=True)
        cmd = build_cmd_exp_a(EXP_A, work_dir_a, args.resume)
        rc = run_cmd(cmd, "Experiment A: r-sweep (7 values × 2 seeds = 14 runs)", args.dry_run)
        codes.append(("A", rc))

    # --- Experiment B: τ-scaling ---
    if "B" in exp:
        for b_cfg in EXP_B_CONFIGS:
            work_dir_b = resolve_work_dir(b_cfg["work_dir_suffix"], args.base_dir)
            work_dir_b.mkdir(parents=True, exist_ok=True)
            cmd = build_cmd_exp_b(b_cfg, work_dir_b, args.resume)
            rc = run_cmd(cmd, f"Experiment B: τ-scaling {b_cfg['label']}", args.dry_run)
            codes.append((b_cfg["label"], rc))

    # --- Summary ---
    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  PHASE 16 COMPLETE  |  {total_time / 60:.1f} min total")
    print(f"{'='*60}")
    for label, rc in codes:
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  {label}: {status}")

    # Print result locations
    if "A" in exp:
        work_dir_a = resolve_work_dir(EXP_A["work_dir_suffix"], args.base_dir)
        print(f"\n  Experiment A results: {work_dir_a / 'results_final.json'}")
    if "B" in exp:
        for b_cfg in EXP_B_CONFIGS:
            work_dir_b = resolve_work_dir(b_cfg["work_dir_suffix"], args.base_dir)
            print(f"  Experiment {b_cfg['label']} results: {work_dir_b / 'results_final.json'}")

    print(f"\n  Next: python analyze_phase16.py --base_dir {args.base_dir or '<work_dir>'}")

    # Exit with error if any experiment failed
    if any(rc != 0 for _, rc in codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
