#!/usr/bin/env python3
"""Create a no-GPU READY receipt for matched general-data Q/K LoRA."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .prepare_data import atomic_json, sha256_file
from .train_4k_general_qk import (
    READY_STATUS,
    bound_code_sha256,
    registered_protocol,
)
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import FixedView, load_fixed_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--frequency", choices=("native", "evq"), required=True
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=2
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
        ),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--natural-eval-rows", type=int, default=16)
    parser.add_argument("--natural-tail-tokens", type=int, default=1_024)
    parser.add_argument("--seed", type=int, default=20_260_727)
    parser.add_argument(
        "--minimum-free-bytes", type=int, default=8_000_000_000
    )
    return parser.parse_args()


def _verify_view(
    view: FixedView,
    *,
    expected_source: str,
) -> dict[str, Any]:
    if (
        str(view.manifest["source"]["id"]) != expected_source
        or tuple(view.input_ids.shape)[1] != 4_096
        or int(view.lengths.max()) > 4_096
        or len(view.training_rows) == 0
    ):
        raise RuntimeError(f"general-QK view drift: {view.path}")
    for name, entry in view.manifest["files"].items():
        path = view.path / name
        if (
            not path.is_file()
            or path.stat().st_size != int(entry["bytes"])
            or sha256_file(path) != entry["sha256"]
        ):
            raise RuntimeError(f"general-QK data hash drift: {path}")
    return {
        "path": str(view.path),
        "manifest_sha256": sha256_file(view.path / "manifest.json"),
        "source_id": str(view.manifest["source"]["id"]),
        "source_revision": str(view.manifest["source"]["revision"]),
        "shape": [int(value) for value in view.input_ids.shape],
        "training_rows": int(len(view.training_rows)),
        "maximum_active_length": int(view.lengths.max()),
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("general-QK preflight must hide CUDA")
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("general-QK preflight unexpectedly sees CUDA")
    receipt_path = args.receipt.resolve()
    output = args.output.resolve()
    if receipt_path.exists() or output.exists():
        raise FileExistsError(receipt_path if receipt_path.exists() else output)

    checkpoint = args.checkpoint.resolve()
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, args.checkpoint_ready_receipt.resolve()
    )
    prepared = args.prepared_data.resolve()
    longalign = load_fixed_view(prepared / "longalign_paired_L4096")
    tulu = load_fixed_view(prepared / "tulu3_replay_L4096")
    longalign_receipt = _verify_view(
        longalign, expected_source="zai-org/LongAlign-10k"
    )
    tulu_receipt = _verify_view(
        tulu,
        expected_source="allenai/tulu-3-sft-olmo-2-mixture-0225",
    )
    background_manifest = args.background_dir.resolve() / "manifest.json"
    if not background_manifest.is_file():
        raise FileNotFoundError(background_manifest)
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < int(args.minimum_free_bytes):
        raise RuntimeError("insufficient free space for general-QK run")

    receipt = {
        "status": READY_STATUS,
        "protocol": registered_protocol(args),
        "bound_code_sha256": bound_code_sha256(),
        "preflight_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {
            "checkpoint_sha256": checkpoint_digest,
            "longalign": longalign_receipt,
            "tulu": tulu_receipt,
            "background_manifest_sha256": sha256_file(
                background_manifest
            ),
        },
        "data_boundary": {
            "training_sources": [
                "zai-org/LongAlign-10k",
                "allenai/tulu-3-sft-olmo-2-mixture-0225",
            ],
            "explicit_2wiki_training_rows": 0,
            "explicit_ruler_training_rows": 0,
            "downstream_overlap_audit": (
                "required before downstream result promotion"
            ),
        },
        "output": str(output),
        "storage": {
            "free_bytes": int(free_bytes),
            "minimum_free_bytes": int(args.minimum_free_bytes),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": READY_STATUS,
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "output": str(output),
                "free_bytes": free_bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
