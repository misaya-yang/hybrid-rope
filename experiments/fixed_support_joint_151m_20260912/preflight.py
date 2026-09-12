#!/usr/bin/env python3
"""CPU identity preflight for the first paired S1 execution block."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

from protocol import SPEC, table_for
from run import atomic_json, build_model, weight_state_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sys.path.insert(0, str(args.original_root.resolve()))
    from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m import (  # noqa: PLC0415
        run_experiment as old,
    )

    manifest = json.loads(args.data_manifest.resolve().read_text())
    train_path = Path(manifest["train"]["path"])
    if not train_path.is_file() or train_path.stat().st_size != 999_948_416:
        raise ValueError("frozen uint16 training array identity/size is unavailable")
    hashes = {}
    counts = {}
    for arm in ("geo", "cosh", "full_z"):
        model, full_z = build_model(old, arm, args.support, args.seed)
        hashes[arm] = weight_state_sha256(model)
        counts[arm] = sum(parameter.numel() for parameter in model.parameters())
        if arm == "full_z":
            assert full_z is not None
            if not torch.equal(full_z.realized_inv_freq(), table_for("geo", args.support)):
                raise RuntimeError("remote full-z initialization parity failed")
        del model, full_z
        gc.collect()
    if len(set(hashes.values())) != 1:
        raise RuntimeError(f"paired model-weight initialization differs: {hashes}")
    report = {
        "status": "PASS",
        "support": int(args.support),
        "seed": int(args.seed),
        "protocol_sha256": SPEC.fingerprint(),
        "weight_initialization_sha256": hashes,
        "parameter_counts": counts,
        "data_manifest": str(args.data_manifest.resolve()),
        "train_path": str(train_path),
        "train_bytes": train_path.stat().st_size,
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
