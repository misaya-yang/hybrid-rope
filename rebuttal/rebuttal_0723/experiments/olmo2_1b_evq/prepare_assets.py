#!/usr/bin/env python3
"""Download and verify pinned OLMo-2 model/config assets without a GPU."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    GEO1000_REVISION,
    GEO1000_WEIGHT_FILES,
    MODEL_ID,
    OFFICIAL_CONFIG_SHA256,
    OFFICIAL_CONFIG_URL,
    STEP0_REVISION,
    STEP0_WEIGHT_FILES,
    relative_file_manifest,
    sha256_file,
    validate_weight_files,
)


MODEL_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "README.md",
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def download_url(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "evq-olmo-preflight/1"})
    temporary = output.with_suffix(output.suffix + ".incomplete")
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as handle:
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    temporary.replace(output)


def download_snapshot(
    output: Path,
    revision: str,
    *,
    max_workers: int,
    attempts: int,
    retry_base_seconds: float,
) -> None:
    if attempts < 1:
        raise ValueError("download attempts must be positive")
    if retry_base_seconds < 0:
        raise ValueError("retry base seconds must be non-negative")
    output.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            snapshot_download(
                repo_id=MODEL_ID,
                revision=revision,
                local_dir=str(output),
                allow_patterns=MODEL_PATTERNS,
                max_workers=max_workers,
            )
            return
        except Exception as error:
            if attempt == attempts:
                raise
            delay = min(retry_base_seconds * (2 ** (attempt - 1)), 120.0)
            print(
                f"snapshot transport failed for {revision} "
                f"(attempt {attempt}/{attempts}: {type(error).__name__}); "
                f"resuming in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)


def validate_snapshot(
    output: Path,
    revision: str,
    expected_weights: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    required = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for name in required:
        if not (output / name).is_file():
            raise FileNotFoundError(output / name)
    weight_rows = validate_weight_files(output, expected_weights)
    tokenizer_files = [
        output / name
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "merges.txt",
            "vocab.json",
        )
        if (output / name).is_file()
    ]
    return {
        "model_id": MODEL_ID,
        "revision": revision,
        "weights": weight_rows,
        "config_sha256": sha256_file(output / "config.json"),
        "tokenizer_files": relative_file_manifest(output, tokenizer_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument(
        "--include-geo1000",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--download-attempts", type=int, default=8)
    parser.add_argument("--retry-base-seconds", type=float, default=5.0)
    args = parser.parse_args()

    root = args.asset_root.resolve()
    step0 = root / "step0"
    geo1000 = root / "geo1000"
    official_config = root / "upstream" / "OLMo2-1B-stage1.yaml"

    if not args.validate_only:
        download_snapshot(
            step0,
            STEP0_REVISION,
            max_workers=args.max_workers,
            attempts=args.download_attempts,
            retry_base_seconds=args.retry_base_seconds,
        )
        if args.include_geo1000:
            download_snapshot(
                geo1000,
                GEO1000_REVISION,
                max_workers=args.max_workers,
                attempts=args.download_attempts,
                retry_base_seconds=args.retry_base_seconds,
            )
        if not official_config.exists():
            download_url(OFFICIAL_CONFIG_URL, official_config)

    if sha256_file(official_config) != OFFICIAL_CONFIG_SHA256:
        raise RuntimeError("official OLMo training config hash mismatch")

    snapshots = {
        "step0": validate_snapshot(step0, STEP0_REVISION, STEP0_WEIGHT_FILES)
    }
    if args.include_geo1000:
        snapshots["geo1000"] = validate_snapshot(
            geo1000, GEO1000_REVISION, GEO1000_WEIGHT_FILES
        )

    verified = {
        "status": "ASSETS_VERIFIED",
        "asset_root": ".",
        "official_config": {
            "path": str(official_config.relative_to(root)),
            "url": OFFICIAL_CONFIG_URL,
            "sha256": OFFICIAL_CONFIG_SHA256,
        },
        "snapshots": snapshots,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    }
    manifest_path = root / "asset_manifest.json"
    if args.validate_only:
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
        if frozen.get("status") != "ASSETS_VERIFIED":
            raise RuntimeError("asset manifest is not ASSETS_VERIFIED")
        comparable_frozen = dict(frozen)
        comparable_verified = dict(verified)
        # Endpoint is transport provenance, not model content, and validation
        # must not rewrite it based on the current shell environment.
        comparable_frozen.pop("hf_endpoint", None)
        comparable_verified.pop("hf_endpoint", None)
        if comparable_frozen != comparable_verified:
            raise RuntimeError("frozen asset manifest content drift")
        print(json.dumps(frozen, indent=2, sort_keys=True))
        return
    write_json(manifest_path, verified)
    print(json.dumps(verified, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
