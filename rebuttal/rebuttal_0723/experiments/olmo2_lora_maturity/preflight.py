#!/usr/bin/env python3
"""Create the immutable READY receipt for the OLMo-2 maturity screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from .prepare_causal_data import verify_collection as verify_causal
from .prepare_data import (
    MODEL_ID,
    MODEL_REVISIONS,
    atomic_json,
    sha256_file,
    verify_collection as verify_training_data,
)
from .prepare_probe_background import verify_background


EXPECTED_WEIGHTS = {
    "step10000_21B": {
        "model-00001-of-00002.safetensors": {
            "bytes": 4_983_360_992,
            "sha256": (
                "3f872540aa31ec87a8b976418ca2b4d671534be1cdaad61e"
                "ab2c97c413935203"
            ),
        },
        "model-00002-of-00002.safetensors": {
            "bytes": 956_326_560,
            "sha256": (
                "82ef05439f5069f0de94dae3ca2cc31ccb9a38461585f939"
                "c1c94099cfc77a01"
            ),
        },
    },
    "step20000_42B": {
        "model-00001-of-00002.safetensors": {
            "bytes": 4_983_360_992,
            "sha256": (
                "8ee07a39fe521e7bf9ca557f58e69431e0948ed7848fdf56"
                "401625d702fc1167"
            ),
        },
        "model-00002-of-00002.safetensors": {
            "bytes": 956_326_560,
            "sha256": (
                "59c1b15495a32d05b4165fa68129628feaa094314e975063"
                "e0011bfe0a0ae094"
            ),
        },
    },
    "step30000_63B": {
        "model-00001-of-00002.safetensors": {
            "bytes": 4_983_360_992,
            "sha256": (
                "7d1186ad3506b5760cfdd3fb099ace4c1339e9f6b98f6e3"
                "3af12400d8cff080f"
            ),
        },
        "model-00002-of-00002.safetensors": {
            "bytes": 956_326_560,
            "sha256": (
                "f72521ef281a54c337238d8f836661c9f5f50c93b8d397c"
                "66435939e37abeb75"
            ),
        },
    },
}


def parse_checksum_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        values[relative] = digest
    return values


def composite_hash(files: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_checkpoint(
    models_root: Path,
    name: str,
    checksums: dict[str, str],
    checksum_mtime_ns: int,
) -> dict[str, Any]:
    checkpoint = models_root / name
    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )
    expected_config = {
        "model_type": "olmo2",
        "hidden_size": 2_048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "max_position_embeddings": 4_096,
        "rope_theta": 500_000.0,
        "vocab_size": 100_352,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"{name} config drift for {key}: {config.get(key)!r}"
            )
    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )
    indexed_shards = set(index["weight_map"].values())
    expected_shards = set(EXPECTED_WEIGHTS[name])
    if indexed_shards != expected_shards:
        raise RuntimeError(f"{name} shard-index drift")
    files: dict[str, dict[str, Any]] = {}
    for filename, expected in EXPECTED_WEIGHTS[name].items():
        path = checkpoint / filename
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"{path} size mismatch")
        relative = str(path.relative_to(models_root.parent))
        actual = checksums.get(relative)
        if actual != expected["sha256"]:
            raise RuntimeError(f"{path} checksum mismatch")
        if path.stat().st_mtime_ns > checksum_mtime_ns:
            raise RuntimeError(f"{path} is newer than checksum receipt")
        files[filename] = dict(expected)
    return {
        "status": "verified",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISIONS[name],
        "config": expected_config,
        "files": files,
        "composite_sha256": composite_hash(files),
        "tokenizer_sha256": sha256_file(
            checkpoint / "tokenizer.json"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--checksum-file", type=Path, required=True)
    parser.add_argument("--checksum-exit-file", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--causal-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    checksum_file = args.checksum_file.resolve()
    checksum_exit = args.checksum_exit_file.resolve()
    if checksum_exit.read_text(encoding="utf-8").strip() != "0":
        raise RuntimeError("model checksum command did not exit successfully")
    checksums = parse_checksum_file(checksum_file)
    models_root = args.models_root.resolve()
    checkpoints = {
        name: verify_checkpoint(
            models_root,
            name,
            checksums,
            checksum_file.stat().st_mtime_ns,
        )
        for name in EXPECTED_WEIGHTS
    }
    training_data = verify_training_data(
        args.prepared_data.resolve() / "collection_manifest.json"
    )
    background = verify_background(args.background_dir.resolve())
    causal = verify_causal(args.causal_data.resolve())
    receipt = {
        "format_version": 1,
        "status": "OLMO2_MATURITY_ASSETS_VERIFIED",
        "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "models_root_name": models_root.name,
        "checksum_file_sha256": sha256_file(checksum_file),
        "checkpoints": checkpoints,
        "training_data": training_data,
        "background": background,
        "causal": causal,
    }
    atomic_json(output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output),
                "output_sha256": sha256_file(output),
                "checkpoints": {
                    name: value["composite_sha256"]
                    for name, value in checkpoints.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
