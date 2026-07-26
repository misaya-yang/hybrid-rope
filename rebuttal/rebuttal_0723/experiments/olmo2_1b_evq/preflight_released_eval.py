#!/usr/bin/env python3
"""CPU-only gate for released native-RoPE checkpoint evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    ACTUAL_PARAMETER_COUNT,
    GEO1000_REVISION,
    GEO1000_WEIGHT_FILES,
    GEO2000_REVISION,
    GEO2000_WEIGHT_FILES,
    GEO5000_REVISION,
    GEO5000_WEIGHT_FILES,
    MODEL_CONTRACT,
    sha256_file,
    sha256_json,
    validate_weight_files,
)


SNAPSHOTS = {
    "geo1000": {
        "revision": GEO1000_REVISION,
        "weights": GEO1000_WEIGHT_FILES,
    },
    "geo2000": {
        "revision": GEO2000_REVISION,
        "weights": GEO2000_WEIGHT_FILES,
    },
    "geo5000": {
        "revision": GEO5000_REVISION,
        "weights": GEO5000_WEIGHT_FILES,
    },
}
TOKENIZER_FILES = (
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
REQUIRED_CODE = (
    "contract.py",
    "evaluate.py",
    "evaluate_ruler.py",
    "prepare_eval_32k.py",
    "prepare_ruler_data.py",
    "preflight.py",
    "preflight_released_eval.py",
    "run_5090_released_eval.sh",
    "train.py",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def validate_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    actual = {
        "architectures": list(config["architectures"]),
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "num_attention_heads": int(config["num_attention_heads"]),
        "num_key_value_heads": int(config["num_key_value_heads"]),
        "head_dim": int(
            config.get(
                "head_dim",
                config["hidden_size"] // config["num_attention_heads"],
            )
        ),
        "max_position_embeddings": int(config["max_position_embeddings"]),
        "rope_theta": float(config["rope_theta"]),
        "vocab_size": int(config["vocab_size"]),
        "tie_word_embeddings": bool(config["tie_word_embeddings"]),
    }
    if actual != MODEL_CONTRACT:
        raise RuntimeError(f"released checkpoint config drift: {actual}")
    return {
        "sha256": sha256_file(path),
        "model_contract": actual,
        "actual_parameter_count": ACTUAL_PARAMETER_COUNT,
    }


def validate_snapshot(root: Path, name: str) -> dict[str, Any]:
    expected = SNAPSHOTS[name]
    snapshot = root / name
    if list(snapshot.glob("*.aria2")):
        raise RuntimeError(f"{name} still has incomplete aria2 artifacts")
    weights = validate_weight_files(snapshot, expected["weights"])
    config = validate_config(snapshot / "config.json")
    index = snapshot / "model.safetensors.index.json"
    index_payload = json.loads(index.read_text(encoding="utf-8"))
    metadata = index_payload.get("metadata", {})
    if int(metadata.get("total_parameters", -1)) != ACTUAL_PARAMETER_COUNT:
        raise RuntimeError(f"{name} parameter-count metadata drift")
    expected_total = ACTUAL_PARAMETER_COUNT * 4
    if int(metadata.get("total_size", -1)) != expected_total:
        raise RuntimeError(f"{name} total-size metadata drift")
    tokenizer = {}
    for filename in TOKENIZER_FILES:
        path = snapshot / filename
        if not path.is_file():
            raise RuntimeError(f"{name} missing tokenizer file: {filename}")
        tokenizer[filename] = sha256_file(path)
    return {
        "revision": expected["revision"],
        "snapshot": str(snapshot.resolve()),
        "config": config,
        "index_sha256": sha256_file(index),
        "weights": weights,
        "tokenizer": tokenizer,
    }


def code_receipt(code_root: Path) -> dict[str, Any]:
    files = {}
    for name in REQUIRED_CODE:
        path = code_root / name
        files[name] = sha256_file(path)
    return {
        "root": str(code_root.resolve()),
        "files": files,
        "combined_sha256": sha256_json(files),
    }


def validate_eval_anchor_bundle(manifest_path: Path) -> dict[str, Any]:
    """Verify the immutable evaluation inputs needed on the GPU machine."""
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "EVAL_DATA_VERIFIED":
        raise RuntimeError("evaluation dataset is not EVAL_DATA_VERIFIED")
    if manifest.get("tokenizer") != "allenai_dolma2":
        raise RuntimeError("evaluation tokenizer drift")
    if not manifest.get("held_out"):
        raise RuntimeError("evaluation data is not marked held-out")
    expected = {
        "long_documents": (128, 16_384),
        "official_validation": (256, 4_096),
    }
    anchors: dict[str, Any] = {}
    for name, (rows, length) in expected.items():
        anchor = manifest["anchors"][name]
        if anchor["rows"] != rows or anchor["length"] != length:
            raise RuntimeError(f"evaluation anchor contract drift: {name}")
        path = root / anchor["path"]
        if sha256_file(path) != anchor["sha256"]:
            raise RuntimeError(f"evaluation anchor hash drift: {name}")
        metadata_path = root / anchor["metadata_path"]
        if sha256_file(metadata_path) != anchor["metadata_sha256"]:
            raise RuntimeError(f"evaluation metadata hash drift: {name}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if len(metadata) != rows:
            raise RuntimeError(f"evaluation metadata row drift: {name}")
        if name == "long_documents":
            documents = [row["source"] for row in metadata]
            if len(set(documents)) != rows:
                raise RuntimeError(
                    "long evaluation is not document-disjoint by row"
                )
        anchors[name] = {
            "path": anchor["path"],
            "sha256": anchor["sha256"],
            "metadata_path": anchor["metadata_path"],
            "metadata_sha256": anchor["metadata_sha256"],
            "rows": rows,
            "length": length,
        }
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "anchor_only_transport": True,
        "source_provenance_retained_in_manifest": True,
        "anchors": anchors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument(
        "--snapshots",
        nargs="+",
        choices=tuple(SNAPSHOTS),
        default=list(SNAPSHOTS),
    )
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parent
    data_manifest = args.data_manifest.resolve()
    if not data_manifest.is_file():
        raise FileNotFoundError(data_manifest)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    free_bytes = os.statvfs(output_root)
    available_bytes = free_bytes.f_bavail * free_bytes.f_frsize
    if available_bytes < 2 * 1024**3:
        raise RuntimeError("evaluation output filesystem has less than 2 GiB free")

    snapshots = {
        name: validate_snapshot(args.asset_root.resolve(), name)
        for name in args.snapshots
    }
    evaluation = validate_eval_anchor_bundle(args.eval_manifest.resolve())
    receipt = {
        "status": "RELEASED_BASELINE_EVAL_CPU_READY",
        "scope": {
            "schedules": [
                f"released_native_rope_step{name.removeprefix('geo')}"
                for name in args.snapshots
            ],
            "lengths": [2048, 4096, 8192, 16384],
            "automatic_32k": False,
            "precision": "bf16_autocast_fp32_weights",
            "attention": "flash_only_sdpa_no_fallback",
            "compile_mode": "max-autotune-no-cudagraphs",
            "ruler": {
                "suite_default": "quick",
                "lengths": [4096, 8192, 16384],
                "limit_per_cell_default": 100,
            },
        },
        "snapshots": snapshots,
        "evaluation_dataset": evaluation,
        "data_manifest": {
            "path": str(data_manifest),
            "sha256": sha256_file(data_manifest),
        },
        "code": code_receipt(code_root),
        "environment": {
            "python": platform.python_version(),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "safetensors": package_version("safetensors"),
            "numpy": package_version("numpy"),
        },
        "output": {
            "root": str(output_root),
            "available_bytes": available_bytes,
        },
        "gpu_checks_pending": [
            "RTX_5090_identity_and_compute_capability",
            "Flash_SDPA_eligibility_at_4K_and_16K",
            "compile_success_and_latency",
            "finite_NLL_and_peak_VRAM",
        ],
    }
    write_json(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
