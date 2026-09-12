#!/usr/bin/env python3
"""Dry-run-first launcher for the stock OLMo evaluator on the E3 pair."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development-screen", type=Path, required=True)
    parser.add_argument("--prior-registry", type=Path, required=True)
    parser.add_argument("--reuse-generations", type=Path, help="prior partial/complete layer_screen output; writes a new recovery directory")
    parser.add_argument("--execute", action="store_true", help="explicitly authorize GPU evaluation")
    args = parser.parse_args()
    plan = {
        "status": "DRY_RUN" if not args.execute else "STARTING",
        "prepared": str(args.prepared.resolve()), "out": str(args.out.resolve()),
        "arms": ["C42", "C42V24"], "rows_per_arm": 980,
        "execution_path": "scripts.experiments.olmo_fast_screen.layer_screen",
        "recovery_source": None if args.reuse_generations is None else str(args.reuse_generations.resolve()),
    }
    from experiments.rope_fast_5090_20260912.e3_validate import validate
    plan["cpu_validation"] = validate(args.prepared.resolve(), args.development_screen.resolve(), args.prior_registry.resolve())
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    if capability != (12, 0) or "5090" not in properties.name.upper():
        raise RuntimeError(f"E3 requires RTX 5090 sm120; got {properties.name!r} {capability}")
    command = [sys.executable, "-m", "scripts.experiments.olmo_fast_screen.layer_screen",
        "--prepared", str(args.prepared.resolve()), "--spec", str((args.prepared / "e3_methods.json").resolve()),
        "--out", str(args.out.resolve())]
    if args.reuse_generations is not None:
        command.extend(["--reuse-generations", str(args.reuse_generations.resolve())])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
