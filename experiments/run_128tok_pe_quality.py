#!/usr/bin/env python3
"""128-token PE Quality Test: EVQ vs DAPE (EXPERIMENT_AUDIT_V4 方案 A).

Trains 125M models on 128-token sequences, evaluates extrapolation to 8192.
Directly comparable to DAPE (NeurIPS 2024) experimental protocol.

Phase 1 (core, ~2h):
  A1: Geometric RoPE (τ=0)          — baseline
  A2: Fixed EVQ τ=1.0               — fixed mid
  A3: Fixed EVQ τ=1.5               — fixed high
  A4: Learnable EVQ (init=1.0)      — core experiment
  A5: Learnable EVQ (init=0.01)     — robustness

Phase 2 (DAPE comparison, ~1h):
  B1: DAPE (lr_mult=10)
  B2: DAPE (lr_mult=100)

Usage:
    python experiments/run_128tok_pe_quality.py --work_dir ~/evq_128tok
    python experiments/run_128tok_pe_quality.py --work_dir ~/evq_128tok --phase 2
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
    "--seq_len", "128",
    "--train_tokens", "15000000",
    "--eval_lengths", "128,256,512,1024,2048,4096,8192",
]

# Phase 1: Core comparison (5 runs)
PHASE1 = [
    ("A1: Geometric (τ=0)",       ["--taus", "0.0"]),
    ("A2: Fixed EVQ τ=1.0",       ["--taus", "1.0"]),
    ("A3: Fixed EVQ τ=1.5",       ["--taus", "1.5"]),
    ("A4: Learnable EVQ init=1.0", ["--learnable", "--tau_init", "1.0", "--tau_lr_mult", "100"]),
    ("A5: Learnable EVQ init=0.01",["--learnable", "--tau_init", "0.01", "--tau_lr_mult", "100"]),
]

# Phase 2: DAPE comparison (2 runs)
PHASE2 = [
    ("B1: DAPE lr_mult=10",  ["--dape", "--tau_lr_mult", "10"]),
    ("B2: DAPE lr_mult=100", ["--dape", "--tau_lr_mult", "100"]),
]

# Phase 3: Multi-seed confirmation (2 runs)
PHASE3 = [
    ("C1: Learnable seed=137", ["--learnable", "--tau_init", "1.0", "--tau_lr_mult", "100", "--seeds", "137"]),
    ("C2: Learnable seed=256", ["--learnable", "--tau_init", "1.0", "--tau_lr_mult", "100", "--seeds", "256"]),
]


def run_phase(name: str, runs: list, work_dir: Path, resume: bool, dry_run: bool):
    print(f"\n{'#'*60}")
    print(f"  {name}")
    print(f"  {len(runs)} runs  |  work_dir={work_dir}")
    print(f"{'#'*60}\n")

    results = []
    for i, (desc, extra_flags) in enumerate(runs, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(runs)}] {desc}")
        print(f"{'='*60}")

        cmd = [sys.executable, SWEEP_SCRIPT, *COMMON,
               "--work_dir", str(work_dir), *extra_flags]
        if resume:
            cmd.append("--resume")
        if dry_run:
            cmd.append("--dry_run")

        print(f"  CMD: {' '.join(cmd)}\n")
        t0 = time.time()
        ret = subprocess.run(cmd, check=False)
        elapsed = time.time() - t0

        status = "OK" if ret.returncode == 0 else f"FAIL(rc={ret.returncode})"
        results.append((desc, status, elapsed))
        print(f"\n  [{i}/{len(runs)}] {desc}: {status} in {elapsed/60:.1f} min")

    return results


def main():
    parser = argparse.ArgumentParser(description="128-token PE quality test (DAPE comparison)")
    parser.add_argument("--work_dir", type=str, default=str(Path.home() / "evq_128tok"))
    parser.add_argument("--phase", type=str, default="1",
                        help="Phase to run: 1, 2, 3, 12, 123, or all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    t_total = time.time()

    phases = args.phase.lower()
    if phases == "all":
        phases = "123"

    if "1" in phases:
        r = run_phase("PHASE 1: Core EVQ Comparison", PHASE1, work_dir, args.resume, args.dry_run)
        all_results.extend(r)

    if "2" in phases:
        r = run_phase("PHASE 2: DAPE Comparison", PHASE2, work_dir, args.resume, args.dry_run)
        all_results.extend(r)

    if "3" in phases:
        r = run_phase("PHASE 3: Multi-Seed Confirmation", PHASE3, work_dir, args.resume, args.dry_run)
        all_results.extend(r)

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  ALL PHASES COMPLETE  |  {total_time/60:.1f} min total")
    print(f"{'='*60}")
    for desc, status, elapsed in all_results:
        print(f"  {desc:35s}  {status:10s}  {elapsed/60:.1f} min")

    print(f"\n  Results: {work_dir / 'results_final.json'}")


if __name__ == "__main__":
    main()
