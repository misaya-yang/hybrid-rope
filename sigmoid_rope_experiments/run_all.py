#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from experiments import (
    exp1_formula_validation,
    exp2_phase_collision,
    exp3_attention_pattern,
    exp4_passkey_retrieval,
    exp5_scaling_law,
)
from src.utils import env_info, get_device, set_seed


def ensure_dependencies() -> None:
    # torch is required; others can be installed if missing.
    required = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
        ("pandas", "pandas"),
        ("seaborn", "seaborn"),
    ]
    missing: List[Tuple[str, str]] = []
    for module_name, pip_name in required:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append((module_name, pip_name))
    if not missing:
        return

    print("[deps] Missing packages detected:", ", ".join(m for m, _ in missing))
    for _, pip_name in missing:
        print(f"[deps] Installing {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run all Sigmoid-RoPE experiments end-to-end.")
    ap.add_argument(
        "--search-mode",
        type=str,
        default="auto",
        choices=["auto", "coarse", "fine"],
        help="Grid search mode. auto/coarse are recommended first; fine is slower.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="Force CPU mode.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    (root_dir / "data").mkdir(parents=True, exist_ok=True)
    (root_dir / "results").mkdir(parents=True, exist_ok=True)

    ensure_dependencies()
    set_seed(args.seed)

    device = get_device(prefer_cuda=not args.cpu)
    print("[env]", env_info())
    print("[run] device:", device)
    if args.search_mode == "auto":
        print("[run] grid search mode=auto => using coarse for quick trend verification.")
        print("[run] if needed, rerun with --search-mode fine for exhaustive search.")

    t0 = time.time()
    outputs: Dict[str, Dict] = {}

    outputs["exp1_formula_validation"] = exp1_formula_validation.run(
        root_dir=root_dir,
        device=device,
        search_mode=args.search_mode,
    )
    outputs["exp2_phase_collision"] = exp2_phase_collision.run(
        root_dir=root_dir,
        device=device,
    )
    outputs["exp3_attention_pattern"] = exp3_attention_pattern.run(
        root_dir=root_dir,
        device=device,
    )
    outputs["exp4_passkey_retrieval"] = exp4_passkey_retrieval.run(
        root_dir=root_dir,
        device=device,
    )
    outputs["exp5_scaling_law"] = exp5_scaling_law.run(
        root_dir=root_dir,
        device=device,
    )

    elapsed = time.time() - t0
    print("\n=== ALL EXPERIMENTS COMPLETED ===")
    print(f"Elapsed: {elapsed:.2f}s")
    print("Outputs:")
    for k, v in outputs.items():
        print(f"- {k}")
        for kk, vv in v.items():
            print(f"  {kk}: {vv}")


if __name__ == "__main__":
    main()

