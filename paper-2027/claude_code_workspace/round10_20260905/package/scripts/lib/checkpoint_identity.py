"""Dependency-light checkpoint identity helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safetensors_weight_set_sha256(checkpoint: Path) -> str:
    """Return the file hash for one weight or a manifest hash for many shards."""

    weights = sorted(checkpoint.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"no safetensors weights under {checkpoint}")
    if len(weights) == 1:
        return sha256_file(weights[0])
    identity = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in weights
    ]
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
