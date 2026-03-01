#!/usr/bin/env python3
"""7-run experiment matrix: fixed-τ sweep + learnable-τ EVQ on 125M GPT-2.

Runs:
  1. Geometric (τ=0.0)
  2-5. Fixed τ ∈ {0.5, 1.0, 1.5, 2.0}
  6. Learnable τ, init=1.0
  7. Learnable τ, init=0.01

Usage:
    python experiments/run_125m_learnable.py --work_dir ~/evq_125m_learnable
    python experiments/run_125m_learnable.py --work_dir ~/evq_125m_learnable --resume
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SWEEP_SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "m4_evq_sweep" / "run_evq_sweep.py")

# Common flags for all runs
COMMON = [
    "--tier", "125m",
    "--seeds", "42",
    "--base", "500000.0",
    "--dataset", "fineweb-edu",
    "--eval_16k",
]

# The 7-run matrix
RUNS = [
    # (description, extra_flags)
    ("Geometric (τ=0.0)",       ["--taus", "0.0"]),
    ("Fixed τ=0.5",             ["--taus", "0.5"]),
    ("Fixed τ=1.0",             ["--taus", "1.0"]),
    ("Fixed τ=1.5",             ["--taus", "1.5"]),
    ("Fixed τ=2.0",             ["--taus", "2.0"]),
    ("Learnable init=1.0",      ["--learnable", "--tau_init", "1.0", "--tau_lr_mult", "10"]),
    ("Learnable init=0.01",     ["--learnable", "--tau_init", "0.01", "--tau_lr_mult", "10"]),
]


def main():
    parser = argparse.ArgumentParser(description="125M learnable-τ experiment matrix")
    parser.add_argument("--work_dir", type=str, default=str(Path.home() / "evq_125m_learnable"))
    parser.add_argument("--dataset", type=str, default="fineweb-edu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'#'*60}")
    print(f"  125M Learnable-τ Experiment Matrix")
    print(f"  {len(RUNS)} runs  |  work_dir={work_dir}")
    print(f"{'#'*60}\n")

    t_total = time.time()
    results = []

    for i, (desc, extra_flags) in enumerate(RUNS, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(RUNS)}] {desc}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, SWEEP_SCRIPT,
            *COMMON,
            "--work_dir", str(work_dir),
            "--dataset", args.dataset,
            *extra_flags,
        ]
        if args.resume:
            cmd.append("--resume")
        if args.dry_run:
            cmd.append("--dry_run")

        print(f"  CMD: {' '.join(cmd)}\n")
        t0 = time.time()
        ret = subprocess.run(cmd, check=False)
        elapsed = time.time() - t0

        status = "OK" if ret.returncode == 0 else f"FAILED (rc={ret.returncode})"
        results.append((desc, status, elapsed))
        print(f"\n  [{i}/{len(RUNS)}] {desc}: {status} in {elapsed/60:.1f} min")

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  ALL RUNS COMPLETE  |  {total_time/60:.1f} min total")
    print(f"{'='*60}")
    for desc, status, elapsed in results:
        print(f"  {desc:30s}  {status:10s}  {elapsed/60:.1f} min")

    print(f"\n  Results: {work_dir / 'results_checkpoint.json'}")
    print(f"  Plot:   python experiments/plot_tau_trajectory.py --work_dir {work_dir}")


if __name__ == "__main__":
    main()
