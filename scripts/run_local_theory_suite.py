#!/usr/bin/env python3
"""
Run local theory suite:
1) Phase-transition mixed-prior scan
2) Proxy-trap energy visualizations
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run local theory suite on current machine.")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--L", type=int, default=131072)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--base", type=float, default=10000.0)
    ap.add_argument("--p_points", type=int, default=41)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str, default="results/theory_2026-02-22")
    ap.add_argument("--data_dir", type=str, default="data/theory_2026-02-22")
    return ap.parse_args()


def run(cmd: list[str]) -> None:
    print("[suite] RUN:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    py = args.python

    run(
        [
            py,
            str(root / "scripts" / "generate_phase_transition_data.py"),
            "--head_dim",
            str(args.head_dim),
            "--base",
            str(args.base),
            "--L",
            str(args.L),
            "--gamma",
            str(args.gamma),
            "--p_points",
            str(args.p_points),
            "--device",
            args.device,
            "--out_dir",
            args.out_dir,
            "--data_dir",
            args.data_dir,
        ]
    )

    run(
        [
            py,
            str(root / "scripts" / "plot_proxy_trap_energy.py"),
            "--L",
            str(args.L),
            "--base",
            str(args.base),
            "--gamma",
            str(args.gamma),
            "--device",
            args.device,
            "--out_dir",
            args.out_dir,
            "--data_dir",
            args.data_dir,
        ]
    )

    print("[suite] completed", flush=True)


if __name__ == "__main__":
    main()

