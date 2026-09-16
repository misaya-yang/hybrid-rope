#!/usr/bin/env python3
"""Download and verify the pinned public GLM-4-9B-0414 snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "zai-org/GLM-4-9B-0414"
REVISION = "645b8482494e31b6b752272bf7f7f273ef0f3caf"
FILES = {
    "LICENSE": (1064, "d08ac7b31e69ddf9845d56f2705325e69873f87b0b34b4bd53a3185428a784ae"),
    "chat_template.jinja": (1015, "db700f25fa300e53634c6fc78dee86b7fbd6d27e624edb855b18a4078c83a822"),
    "config.json": (689, "98794d8150da03ca7baec63208584810d01caf43427e376071c00e770f4b4fa7"),
    "generation_config.json": (160, "56ee0be8c595f22a6a80a822096533e10455fcd1e859e0fc36a6e760088147fa"),
    "model-00001-of-00004.safetensors": (4984283160, "f428f8621a3b09c8dc85640d35b147fd52515c4de3eeed71a0bd24fe28a475e8"),
    "model-00002-of-00004.safetensors": (4895274600, "5fe3a4f6c126a8f831df97dcd3436660b5e744f7307b84e8512d4847c369b359"),
    "model-00003-of-00004.safetensors": (4895274616, "9716302a052d79c2750e3c98d34f4c4048871092831fc158cd89286513df83d9"),
    "model-00004-of-00004.safetensors": (4025786080, "d81b5e2c294f5f2aecd6c82ae522f86a4ce25c00d24bab19bb41386ab6874de4"),
    "model.safetensors.index.json": (43614, "76d38dfcac7d59e0250f1fcf03e25061127e6a2d1be7c2b7096c6561b7ffa45a"),
    "special_tokens_map.json": (601, "35a6dd9ff6e2474ba72b180db853851d2a84fce3d12095089285f447c7474541"),
    "tokenizer.json": (19966496, "76ebeac0d8bd7879ead7b43c16b44981f277e47225de2bd7de9ae1a6cc664a8c"),
    "tokenizer_config.json": (4170, "e316474d867ad1f91165a12d90c73b27a7e0f9a1c9293831cca66c792bfe9a36"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify(destination: Path) -> list[dict]:
    receipts = []
    for name, (expected_bytes, expected_sha) in FILES.items():
        path = destination / name
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        digest = sha256(path)
        if size != expected_bytes or digest != expected_sha:
            raise ValueError(f"pinned file identity differs: {name} {size} {digest}")
        receipts.append({"name": name, "bytes": size, "sha256": digest})
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("/root/models/GLM-4-9B-0414"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.destination != Path("/root/models/GLM-4-9B-0414"):
        raise ValueError("destination is frozen to the requested system-disk path")
    args.destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "0")
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=args.destination,
        allow_patterns=list(FILES),
        max_workers=args.workers,
    )
    files = verify(args.destination)
    receipt = {
        "status": "DOWNLOAD_COMPLETE_VERIFIED",
        "source_model": MODEL_ID,
        "hf_revision": REVISION,
        "destination": str(args.destination),
        "files": files,
        "model_execution": False,
        "gpu_tasks_modified": False,
    }
    atomic_json(args.destination / "DOWNLOAD_RECEIPT.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
