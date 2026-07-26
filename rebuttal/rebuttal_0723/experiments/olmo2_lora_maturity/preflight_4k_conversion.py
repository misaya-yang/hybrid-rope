#!/usr/bin/env python3
"""Fail-closed READY receipt for the step-30K 4K-only conversion run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .prepare_4k_binding_data import LENGTH, SLOTS
from .prepare_causal_data import verify_collection as verify_causal
from .prepare_data import atomic_json, sha256_file, verify_collection
from .prepare_probe_background import verify_background


EXPECTED_CHECKPOINT = {
    "model-00001-of-00002.safetensors": {
        "bytes": 4_983_360_992,
        "sha256": (
            "7d1186ad3506b5760cfdd3fb099ace4c1339e9f6b98f6e33af12400d8cff080f"
        ),
    },
    "model-00002-of-00002.safetensors": {
        "bytes": 956_326_560,
        "sha256": (
            "f72521ef281a54c337238d8f836661c9f5f50c93b8d397c66435939e37abeb75"
        ),
    },
}
EXPECTED_REVISION = "6251e24cf3f303f9d64c78456a155a5dbe2a35e8"
REQUIRED_BINDING_SETS = {
    "train_anchor": (2, "stage_b1_training"),
    "train_broad": (2, "stage_b2_training"),
    "calibration_anchor": (3, "stage_b1_calibration"),
    "validation_ood": (3, "stage_b2_validation"),
    "final_test": (3, "final_test_do_not_monitor"),
}
CODE_FILES = (
    "experiments/native_rope_evq_150m/__init__.py",
    "experiments/native_rope_evq_150m/model.py",
    "scripts/__init__.py",
    "scripts/lib/__init__.py",
    "scripts/lib/rope/__init__.py",
    "scripts/lib/rope/schedules.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_1b_evq/__init__.py",
    "rebuttal/rebuttal_0723/experiments/small_model_lora_conversion.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_1b_evq/contract.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_1b_evq/train.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_1b_evq/prepare_retrieval_data.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_1b_evq/prepare_ruler_data.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_1b_evq/evaluate_ruler.py",
    "rebuttal/rebuttal_0723/experiments/evaluate_ruler_lora.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_conversion.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_generalization.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_ood_factorial.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/__init__.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/causal_data.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/protocol.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/prepare_data.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/prepare_probe_background.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/prepare_causal_data.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/prepare_4k_binding_data.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/train_screen.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/train_4k_stage_a.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/train_4k_stage_b.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/evaluate_4k_conversion.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/preflight_4k_conversion.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/gate_4k_conversion.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/summarize_4k_conversion.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/run_pro6000_4k_conversion.sh",
)


def _verify_checkpoint(checkpoint: Path) -> dict[str, Any]:
    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )
    expected_config = {
        "model_type": "olmo2",
        "hidden_size": 2_048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "max_position_embeddings": LENGTH,
        "rope_theta": 500_000.0,
        "vocab_size": 100_352,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"checkpoint config drift for {key}: {config.get(key)!r}"
            )
    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )
    if set(index["weight_map"].values()) != set(EXPECTED_CHECKPOINT):
        raise RuntimeError("checkpoint shard index drift")

    files: dict[str, dict[str, Any]] = {}
    for name, expected in EXPECTED_CHECKPOINT.items():
        path = checkpoint / name
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"checkpoint size mismatch: {path}")
        actual = sha256_file(path)
        if actual != expected["sha256"]:
            raise RuntimeError(f"checkpoint checksum mismatch: {path}")
        files[name] = {
            "bytes": int(expected["bytes"]),
            "sha256": actual,
            "mtime_ns": path.stat().st_mtime_ns,
        }
    return {
        "status": "verified",
        "checkpoint_path": str(checkpoint.resolve()),
        "revision": EXPECTED_REVISION,
        "config": expected_config,
        "files": files,
        "composite_sha256": ":".join(
            files[name]["sha256"] for name in sorted(files)
        ),
        "tokenizer_sha256": sha256_file(checkpoint / "tokenizer.json"),
    }


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _verify_binding_set(
    root: Path,
    *,
    expected_variants: int,
    expected_purpose: str,
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("purpose") != expected_purpose:
        raise RuntimeError(f"binding purpose drift: {root}")
    shape = tuple(int(value) for value in manifest["shape"])
    if len(shape) != 3 or shape[1:] != (expected_variants, LENGTH):
        raise RuntimeError(f"binding shape drift: {root}: {shape}")
    if int(manifest["slots_per_sequence"]) != SLOTS:
        raise RuntimeError(f"binding slot count drift: {root}")
    placeholder_id = int(manifest["answer_placeholder_token_id"])
    for name, expected_hash in manifest["files"].items():
        path = root / name
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"binding file checksum mismatch: {path}")

    inputs = np.load(root / "input_ids.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(root / "labels.npy", mmap_mode="r", allow_pickle=False)
    if tuple(inputs.shape) != shape or inputs.dtype != np.uint32:
        raise RuntimeError(f"binding input contract drift: {root}")
    if tuple(labels.shape) != shape or labels.dtype != np.int32:
        raise RuntimeError(f"binding label contract drift: {root}")
    if int((labels != -100).sum()) != shape[0] * shape[1] * SLOTS:
        raise RuntimeError(f"binding supervised-token count drift: {root}")

    metadata = _iter_jsonl(root / "rows.jsonl")
    if len(metadata) != shape[0]:
        raise RuntimeError(f"binding metadata row-count drift: {root}")
    for row_index, row in enumerate(metadata):
        variant_names = [str(value) for value in row["variant_names"]]
        if variant_names != list(manifest["variant_names"]):
            raise RuntimeError(f"binding variant order drift: {root}:{row_index}")
        answer_positions = [int(value) for value in row["answer_positions"]]
        if len(answer_positions) != SLOTS or len(set(answer_positions)) != SLOTS:
            raise RuntimeError(f"binding answer positions drift: {root}:{row_index}")
        if min(answer_positions) <= 0 or max(answer_positions) >= LENGTH:
            raise RuntimeError(f"binding answer position out of range: {root}:{row_index}")
        source_positions = [
            int(value) for value in row["source_value_positions"]
        ]
        original_targets = [
            int(value) for value in row["value_token_ids"]
        ]
        swapped_targets = [
            int(value) for value in row["swapped_value_token_ids"]
        ]
        if not (
            len(source_positions)
            == len(original_targets)
            == len(swapped_targets)
            == SLOTS
        ):
            raise RuntimeError(f"binding source metadata drift: {root}:{row_index}")
        for variant, variant_name in enumerate(variant_names):
            active = np.flatnonzero(labels[row_index, variant] != -100)
            if list(active) != answer_positions:
                raise RuntimeError(
                    f"binding label positions drift: {root}:{row_index}:{variant}"
                )
            if not np.all(
                inputs[row_index, variant, active] == placeholder_id
            ):
                raise RuntimeError(
                    f"binding answer placeholder drift: "
                    f"{root}:{row_index}:{variant}"
                )
            if np.any(labels[row_index, variant, active] == placeholder_id):
                raise RuntimeError(
                    f"binding target aliases placeholder: "
                    f"{root}:{row_index}:{variant}"
                )
            expected_targets = (
                swapped_targets
                if variant_name == "swapped"
                else original_targets
            )
            if not np.array_equal(
                labels[row_index, variant, active],
                np.asarray(expected_targets, dtype=np.int32),
            ):
                raise RuntimeError(
                    f"binding target association drift: "
                    f"{root}:{row_index}:{variant}"
                )
            if variant_name in {"sourced", "swapped"} and not np.array_equal(
                inputs[row_index, variant, source_positions],
                np.asarray(expected_targets, dtype=np.uint32),
            ):
                raise RuntimeError(
                    f"binding source association drift: "
                    f"{root}:{row_index}:{variant}"
                )
        if not np.array_equal(
            np.sort(inputs[row_index, 0]),
            np.sort(inputs[row_index, -1]),
        ):
            raise RuntimeError(
                f"source/swap token multiset drift: {root}:{row_index}"
            )
        distances = [
            int(value) for value in row["actual_value_to_answer_distances"]
        ]
        if len(distances) != SLOTS or any(
            value <= 0 or value >= LENGTH for value in distances
        ):
            raise RuntimeError(f"binding distance drift: {root}:{row_index}")
    return {
        "status": "verified",
        "purpose": expected_purpose,
        "shape": list(shape),
        "manifest_sha256": sha256_file(manifest_path),
        "input_sha256": sha256_file(root / "input_ids.npy"),
        "labels_sha256": sha256_file(root / "labels.npy"),
    }


def _verify_binding_collection(root: Path) -> dict[str, Any]:
    collection_path = root / "collection_manifest.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if int(collection["hard_maximum_training_length"]) != LENGTH:
        raise RuntimeError("binding collection exceeds the 4K contract")
    if int(collection["slots_per_sequence"]) != SLOTS:
        raise RuntimeError("binding collection slot-count drift")
    placeholder_id = int(collection["answer_placeholder_token_id"])
    train_values = {int(value) for value in collection["train_value_token_ids"]}
    eval_values = {int(value) for value in collection["eval_value_token_ids"]}
    if not train_values or not eval_values or train_values & eval_values:
        raise RuntimeError("binding value pools are empty or overlapping")
    if placeholder_id in train_values or placeholder_id in eval_values:
        raise RuntimeError("binding placeholder aliases a value token")
    entries = {entry["name"]: entry for entry in collection["sets"]}
    if set(entries) != set(REQUIRED_BINDING_SETS):
        raise RuntimeError("binding collection set inventory drift")
    verified = {}
    for name, (variants, purpose) in REQUIRED_BINDING_SETS.items():
        entry = entries[name]
        manifest_path = root / entry["relative_path"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["answer_placeholder_token_id"]) != placeholder_id:
            raise RuntimeError(f"binding placeholder drift: {name}")
        if sha256_file(manifest_path) != entry["manifest_sha256"]:
            raise RuntimeError(f"binding manifest checksum mismatch: {name}")
        verified[name] = _verify_binding_set(
            root / entry["relative_path"],
            expected_variants=variants,
            expected_purpose=purpose,
        )
    return {
        "status": "verified",
        "collection_sha256": sha256_file(collection_path),
        "hard_maximum_training_length": LENGTH,
        "sets": verified,
    }


def _verify_code(code_root: Path) -> dict[str, Any]:
    files = {}
    for relative in CODE_FILES:
        path = code_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = sha256_file(path)
    return {"status": "verified", "files": files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--binding-data", type=Path, required=True)
    parser.add_argument("--causal-data", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--minimum-free-bytes", type=int, default=12_000_000_000
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt_path = args.receipt.resolve()
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("preflight must run with CUDA_VISIBLE_DEVICES empty")

    import torch
    import transformers

    if torch.cuda.is_available():
        raise RuntimeError("preflight unexpectedly sees a CUDA device")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_root).free
    if free_bytes < int(args.minimum_free_bytes):
        raise RuntimeError(
            f"insufficient output storage: {free_bytes} bytes free"
        )

    receipt = {
        "format_version": 1,
        "status": "OLMO2_STEP30_4K_CONVERSION_READY",
        "scientific_contract": {
            "base_checkpoint": "OLMo-2 1B step-30K / 63B tokens",
            "inference_operator": "unchanged softmax attention",
            "frequency": "endpoint EVQ-Cosh tau=2",
            "maximum_optimizer_sequence_length": LENGTH,
            "stage_a": "4K full-token next-token CE",
            "stage_b": (
                "paired sourced/swapped full-vocabulary CE with identical "
                "token multiset and positions"
            ),
            "forbidden": [
                "position interpolation",
                "attention temperature or operator change",
                "teacher hidden-state distillation",
                "native-RoPE training control",
            ],
        },
        "checkpoint": _verify_checkpoint(args.checkpoint.resolve()),
        "training_data": verify_collection(
            args.prepared_data.resolve() / "collection_manifest.json"
        ),
        "background": verify_background(args.background_dir.resolve()),
        "binding_data": _verify_binding_collection(args.binding_data.resolve()),
        "causal_data": verify_causal(args.causal_data.resolve()),
        "code": _verify_code(args.code_root.resolve()),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "transformers": transformers.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "storage": {
            "output_root": str(output_root),
            "free_bytes": free_bytes,
            "minimum_free_bytes": int(args.minimum_free_bytes),
        },
        "gates": {
            "stage_a": (
                "finite loss; 16K held-out natural NLL improves by >=0.2 "
                "over EVQ-injected base; 4K regression <=0.15"
            ),
            "stage_b1": (
                "calibration full-vocabulary exact >=0.50, source-deletion "
                "NLL gap >0, swap-follow positive fraction >=0.80"
            ),
            "stage_b2": (
                "held-out full-vocabulary exact leaves baseline floor, "
                "source-deletion gap >0, swap-follow positive fraction >0.5"
            ),
        },
    }
    atomic_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "checkpoint": receipt["checkpoint"]["composite_sha256"],
                "free_bytes": free_bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
