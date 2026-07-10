#!/usr/bin/env python3
"""Create one byte-level LLaMA model manifest for all six legacy arms."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

try:
    from .legacy_lora_protocol import sha256_file
except ImportError:
    from legacy_lora_protocol import sha256_file


_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def validate_model_manifest(
    model_dir: Path,
    manifest: Mapping[str, Any],
    *,
    verify_hashes: bool = False,
) -> None:
    model_dir = Path(model_dir)
    if manifest.get("model") != model_dir.name:
        raise ValueError("model manifest identifier mismatch")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("model manifest has no file records")
    names = []
    for record in records:
        name = str(record.get("name", ""))
        if not name or Path(name).name != name:
            raise ValueError("model manifest file names must be path-safe basenames")
        names.append(name)
        path = model_dir / name
        if not path.is_file() or path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"model file missing or size-mismatched: {name}")
        if path.stat().st_mtime_ns != int(record.get("mtime_ns", -1)):
            raise ValueError(f"model file changed after full-hash manifest creation: {name}")
        if not _HEX_64.fullmatch(str(record.get("sha256", ""))):
            raise ValueError(f"model file SHA-256 missing: {name}")
        if verify_hashes and sha256_file(path) != record["sha256"]:
            raise ValueError(f"model file SHA-256 mismatch: {name}")
    if len(names) != len(set(names)):
        raise ValueError("model manifest contains duplicate file records")
    if "config.json" not in names or len([name for name in names if name.startswith("model") and name.endswith(".safetensors")]) < 4:
        raise ValueError("model manifest does not prove a complete sharded model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    required = [
        args.model_dir / "config.json",
        args.model_dir / "tokenizer.json",
        args.model_dir / "tokenizer_config.json",
    ]
    shards = sorted(args.model_dir.glob("model*.safetensors"))
    if len(shards) < 4:
        raise ValueError("expected a complete sharded LLaMA-3-8B safetensors model")
    required.extend(shards)
    for name in (
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        path = args.model_dir / name
        if path.is_file():
            required.append(path)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"model manifest inputs missing: {', '.join(missing)}")
    manifest = {
        "format_version": 1,
        "model": args.model_dir.name,
        "files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": sha256_file(path),
            }
            for path in required
        ],
    }
    validate_model_manifest(args.model_dir, manifest, verify_hashes=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "manifest": str(args.output),
        "manifest_sha256": sha256_file(args.output),
        "files": len(required),
    }, indent=2))


if __name__ == "__main__":
    main()
