#!/usr/bin/env python3
"""Fail closed before reusing a completed LoRA comparison checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_evq_lora import load_frequency_artifact


def validate_checkpoint_artifact(
    checkpoint_dir: Path,
    expected_method: str,
) -> Dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    metadata_path = checkpoint_dir / "experiment_meta.json"
    if not adapter_path.is_file():
        raise FileNotFoundError("adapter_model.safetensors is missing")
    if not metadata_path.is_file():
        raise FileNotFoundError("experiment_meta.json is missing")

    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    recorded_method = metadata.get("rope_method", metadata.get("method"))
    if recorded_method != expected_method:
        raise RuntimeError(
            f"Checkpoint method mismatch: expected {expected_method}, found {recorded_method!r}"
        )

    provenance = None
    if expected_method in {"evq_cosh", "native_geo"}:
        _, _, provenance = load_frequency_artifact(
            checkpoint_dir / "custom_inv_freq.pt",
            expected_method=expected_method,
        )
    return {
        "checkpoint": checkpoint_dir.name,
        "method": recorded_method,
        "frequency_provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-method",
        choices=["native_geo", "evq_cosh", "yarn"],
        required=True,
    )
    args = parser.parse_args()
    result = validate_checkpoint_artifact(args.checkpoint, args.expected_method)
    print(f"validated {result['checkpoint']} ({result['method']})")


if __name__ == "__main__":
    main()
