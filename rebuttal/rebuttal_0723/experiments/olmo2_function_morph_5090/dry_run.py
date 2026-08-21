#!/usr/bin/env python3
"""Bind assets and freeze targets without loading a model or using CUDA."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

from .protocol import (
    HEAD_DIM,
    LENGTHS,
    METHOD_ID,
    MODEL_REVISION,
    ROWS_PER_LENGTH,
    ContractError,
    load_json,
    sha256_file,
    validate_model_config,
)
from .targets import build_target_manifest


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_FILES = (
    "protocol.py",
    "targets.py",
    "authorization.py",
    "dry_run.py",
    "run_audit.py",
)


def _token_view(path: Path, *, length: int, receipt_path: Path | None) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise ContractError(f"missing token view: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise ContractError(f"token view is not a tensor: {path}")
    if value.ndim != 2 or value.shape[1] != length:
        raise ContractError(
            f"token view {path} must have shape [rows,{length}], got {tuple(value.shape)}"
        )
    if value.shape[0] < ROWS_PER_LENGTH or value.dtype != torch.long:
        raise ContractError(
            f"token view {path} needs >= {ROWS_PER_LENGTH} int64 rows"
        )
    digest = sha256_file(path)
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": digest,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "selected_rows": ROWS_PER_LENGTH,
    }
    del value
    if receipt_path is not None:
        receipt = load_json(receipt_path)
        if receipt.get("tensor_sha256") != digest:
            raise ContractError(f"token receipt hash mismatch: {receipt_path}")
        if receipt.get("length") != length:
            raise ContractError(f"token receipt length mismatch: {receipt_path}")
        if receipt.get("training_or_parameter_updates") is not False:
            raise ContractError(f"token receipt does not assert no training: {receipt_path}")
        result["receipt"] = str(receipt_path.resolve())
        result["receipt_sha256"] = sha256_file(receipt_path)
    return result


def run_dry_run(
    *,
    model_dir: Path,
    collection: Path,
    views: dict[int, Path],
    receipts: dict[int, Path | None],
    output_dir: Path,
) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    collection = collection.resolve()
    config_path = model_dir / "config.json"
    weight_path = model_dir / "model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise ContractError("model directory must contain config.json and model.safetensors")
    config = load_json(config_path)
    validate_model_config(config)

    import numpy as np
    import torch

    cuda_initialized_before = bool(torch.cuda.is_initialized())
    with np.load(collection, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        inv_freq = payload["inv_freq"]
        mass = payload["mass"]
    expected_metadata = {
        "backend": "hf_olmo2",
        "base": 500_000.0,
        "head_dim": HEAD_DIM,
        "layers": 16,
        "heads": 16,
        "length": 4_096,
        "model_revision": MODEL_REVISION,
        "training_or_parameter_updates": False,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ContractError(
                f"R0 metadata drift for {key}: expected {expected!r}, "
                f"observed {metadata.get(key)!r}"
            )
    if inv_freq.shape != (64,) or mass.shape != (16, 16, 4_096):
        raise ContractError(
            f"R0 array shape drift: inv_freq={inv_freq.shape}, mass={mass.shape}"
        )
    weight_sha256 = sha256_file(weight_path)
    weight_rows = metadata.get("weight_files")
    if not isinstance(weight_rows, list) or len(weight_rows) != 1:
        raise ContractError("R0 metadata must bind exactly one weight file")
    if weight_rows[0].get("sha256") != weight_sha256:
        raise ContractError("model weight hash differs from the R0 collection owner")

    token_views = {
        str(length): _token_view(
            views[length], length=length, receipt_path=receipts.get(length)
        )
        for length in LENGTHS
    }
    if token_views["4096"]["sha256"] != metadata.get("tokens_sha256"):
        raise ContractError("4096 token view differs from the R0 collection owner")

    targets = build_target_manifest(collection)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "target_manifest.json"
    target_path.write_text(
        json.dumps(targets, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_hashes = {
        name: sha256_file(PACKAGE_ROOT / name) for name in SOURCE_FILES
    }
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "method_id": METHOD_ID,
        "status": "READY_FOR_EXPLICIT_GPU_AUDIT",
        "assets": {
            "model_dir": str(model_dir),
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "weights": {
                "path": str(weight_path),
                "sha256": weight_sha256,
                "bytes": weight_path.stat().st_size,
            },
            "r0_collection": {
                "path": str(collection),
                "sha256": sha256_file(collection),
            },
            "token_views": token_views,
            "target_manifest": {
                "path": str(target_path.resolve()),
                "sha256": sha256_file(target_path),
                "content_sha256": targets["content_sha256"],
            },
        },
        "source_sha256": source_hashes,
        "execution_proof": {
            "training_attempted": False,
            "optimizer_created": False,
            "gradients_enabled": False,
            "model_loaded": False,
            "checkpoint_deserialized": False,
            "download_attempted": False,
            "network_access_attempted": False,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_initialized_before": cuda_initialized_before,
            "cuda_initialized_after": bool(torch.cuda.is_initialized()),
        },
        "platform": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    if receipt["execution_proof"]["cuda_initialized_after"]:
        raise ContractError("dry-run unexpectedly initialized CUDA")
    receipt_path = output_dir / "dry_run_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--r0-collection", type=Path, required=True)
    for length in LENGTHS:
        parser.add_argument(f"--view-{length}", type=Path, required=True)
        parser.add_argument(f"--receipt-{length}", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    views = {length: getattr(args, f"view_{length}") for length in LENGTHS}
    receipts = {length: getattr(args, f"receipt_{length}") for length in LENGTHS}
    receipt = run_dry_run(
        model_dir=args.model_dir,
        collection=args.r0_collection,
        views=views,
        receipts=receipts,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str((args.output_dir / "dry_run_receipt.json").resolve()),
                "cuda_available": receipt["execution_proof"]["cuda_available"],
                "model_loaded": False,
                "training_attempted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
