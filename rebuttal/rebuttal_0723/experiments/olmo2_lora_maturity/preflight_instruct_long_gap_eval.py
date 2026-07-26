#!/usr/bin/env python3
"""Create a no-GPU READY receipt for the 8K n=100 long-gap screen."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from .freeze_conversion_evidence import extract_key_values
from .prepare_data import atomic_json, sha256_file
from .prepare_instruct_ruler_long_gap import STATUS


MODEL_SHA256 = (
    "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
)
TOKENIZER_SHA256 = (
    "73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca"
)
READY_SHA256 = (
    "3988096b444212439ce288787a26c5a35104c73345aed61688ccbb5f544ab22d"
)
EXPECTED_ADAPTERS = {
    "native_seed20260725": {
        "sha256": (
            "6570ab94aec68431dd4e261eb3ef342e"
            "f72253357aa65a0d37b0a018df2f3f8d"
        ),
        "frequency": "native",
        "seed": 20_260_725,
    },
    "evq_seed20260725": {
        "sha256": (
            "95ceeb70117c73233915760a9756b9b2"
            "a98416ec188b125054da8ced75cad16a"
        ),
        "frequency": "evq",
        "seed": 20_260_725,
    },
    "evq_seed20260726": {
        "sha256": (
            "fdf6dfc249cb216c3effe22a2ee96fe4"
            "39a5aff9a11a99007f022c5fe7c3b085"
        ),
        "frequency": "evq",
        "seed": 20_260_726,
    },
}


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_adapter(
    name: str,
    path: Path,
) -> dict[str, Any]:
    expected = EXPECTED_ADAPTERS[name]
    record = file_record(path)
    if record["sha256"] != expected["sha256"]:
        raise RuntimeError(f"adapter SHA drift: {name}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = dict(payload.get("metadata", {}))
    expected_metadata = {
        "base_checkpoint_sha256": MODEL_SHA256,
        "frequency": expected["frequency"],
        "adaptation": "qkvo_answer",
        "rank": 64,
        "alpha": 128.0,
        "training_sequence_length": 4_096,
        "seed": expected["seed"],
    }
    for field, value in expected_metadata.items():
        if metadata.get(field) != value:
            raise RuntimeError(
                f"adapter metadata drift: {name}:{field}"
            )
    record["metadata"] = metadata
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument(
        "--native-seed20260725",
        type=Path,
        required=True,
    )
    parser.add_argument("--evq-seed20260725", type=Path, required=True)
    parser.add_argument("--evq-seed20260726", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("preflight must hide CUDA")
    if torch.cuda.is_available():
        raise RuntimeError("preflight unexpectedly sees CUDA")

    checkpoint = args.checkpoint.resolve()
    ready_receipt = args.ready_receipt.resolve()
    data_root = args.data_root.resolve()
    code_root = args.code_root.resolve()
    output_root = args.output_root.resolve()
    receipt_path = args.receipt.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if receipt_path.exists():
        raise FileExistsError(receipt_path)

    model = file_record(checkpoint / "model.safetensors")
    tokenizer_file = file_record(checkpoint / "tokenizer.json")
    if model["sha256"] != MODEL_SHA256:
        raise RuntimeError("model SHA drift")
    if tokenizer_file["sha256"] != TOKENIZER_SHA256:
        raise RuntimeError("tokenizer SHA drift")
    if sha256_file(ready_receipt) != READY_SHA256:
        raise RuntimeError("parent READY receipt drift")

    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "status": STATUS,
        "task": "niah_single_1",
        "lengths": [8_192],
        "samples_per_length": 100,
        "seed": 20_260_727,
    }
    for field, value in expected_manifest.items():
        if manifest.get(field) != value:
            raise RuntimeError(f"data manifest drift: {field}")
    if manifest["training_gap_support"] != {
        "rows": 1024,
        "minimum_tokens": 62,
        "maximum_tokens": 3933,
        "rows_sha256": (
            "1d7837e40c7bbc5a52d56cd2e111a515"
            "521ed6c6b5e561d40cf66e527887bea2"
        ),
    }:
        raise RuntimeError("training gap receipt drift")
    if any(
        int(item["count"]) != 0
        for item in manifest[
            "selected_overlap_with_forbidden"
        ].values()
    ):
        raise RuntimeError("long-gap data overlaps forbidden identities")
    data_entry = manifest["files"]["8192"]
    data_path = data_root / data_entry["relative_path"]
    if sha256_file(data_path) != data_entry["sha256"]:
        raise RuntimeError("long-gap data SHA drift")
    if int(data_entry["rows"]) != 100:
        raise RuntimeError("long-gap data row-count drift")
    if int(
        data_entry["minimum_generation_boundary_gap_tokens"]
    ) <= 3933:
        raise RuntimeError("long-gap minimum is not out of support")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    rows = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 100:
        raise RuntimeError("long-gap row count changed")
    recomputed_gaps = []
    for row in rows:
        chat_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["input"]}],
            add_generation_prompt=True,
        )
        prefix_ids = tokenizer(
            row.get("answer_prefix", ""),
            add_special_tokens=False,
        ).input_ids
        gap = (
            len(chat_ids)
            + len(prefix_ids)
            - int(row["token_position_answer"])
        )
        recomputed_gaps.append(gap)
    if min(recomputed_gaps) <= 3933:
        raise RuntimeError("independent long-gap recomputation failed")
    if (
        min(recomputed_gaps)
        != int(data_entry["minimum_generation_boundary_gap_tokens"])
        or max(recomputed_gaps)
        != int(data_entry["maximum_generation_boundary_gap_tokens"])
    ):
        raise RuntimeError("long-gap extrema receipt drift")

    forbidden_rechecks = {}
    selected_values = extract_key_values(data_path)
    for item in manifest["forbidden_sources"]:
        path = Path(item["path"]).resolve()
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError("forbidden source SHA drift")
        values = extract_key_values(path)
        forbidden_rechecks[str(path)] = {
            field: len(
                set(selected_values[field]) & set(values[field])
            )
            for field in (
                "source_keys",
                "source_values",
                "queries",
                "answers",
            )
        }
    if any(
        count
        for record in forbidden_rechecks.values()
        for count in record.values()
    ):
        raise RuntimeError("independent overlap recheck failed")

    adapter_paths = {
        "native_seed20260725": args.native_seed20260725.resolve(),
        "evq_seed20260725": args.evq_seed20260725.resolve(),
        "evq_seed20260726": args.evq_seed20260726.resolve(),
    }
    adapters = {
        name: verify_adapter(name, path)
        for name, path in adapter_paths.items()
    }
    code_relatives = (
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
        "evaluate_instruct_ruler_screen.py",
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
        "prepare_instruct_ruler_long_gap.py",
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
        "preflight_instruct_long_gap_eval.py",
    )
    code = {
        relative: file_record(code_root / relative)
        for relative in code_relatives
    }

    output_root.mkdir(parents=True)
    free_bytes = shutil.disk_usage(output_root).free
    if free_bytes < 1_000_000_000:
        raise RuntimeError("less than 1 GB output space remains")
    module = (
        "rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity."
        "evaluate_instruct_ruler_screen"
    )
    commands = {}
    for name, adapter in adapter_paths.items():
        frequency = str(EXPECTED_ADAPTERS[name]["frequency"])
        output = output_root / name
        commands[name] = [
            sys.executable,
            "-m",
            module,
            "--checkpoint",
            str(checkpoint),
            "--ready-receipt",
            str(ready_receipt),
            "--data-root",
            str(data_root),
            "--output",
            str(output),
            "--frequency",
            frequency,
            "--adapter",
            str(adapter),
            "--task",
            "niah_single_1",
            "--lengths",
            "8192",
            "--limit-per-length",
            "100",
        ]

    receipt = {
        "format_version": 1,
        "status": "OLMO2_INSTRUCT_LONG_GAP_N100_READY",
        "review_concerns": ["R27bE.2", "R27bE.5", "AC.2"],
        "scientific_contract": {
            "existing_evidence": (
                "mixed-gap 8K n=100 is Native 0/100 and EVQ "
                "69/100, 67/100"
            ),
            "smallest_missing_evidence": (
                "same-task n=100 with every source gap beyond all "
                "routing-training gaps"
            ),
            "training": "none",
            "arms": list(adapter_paths),
            "primary_metric": "strict first-number exact",
            "stop_condition": (
                "stop after all three frozen-adapter arms; do not "
                "launch training or another mechanism"
            ),
        },
        "checkpoint": {
            "model": model,
            "tokenizer": tokenizer_file,
        },
        "parent_ready_receipt": file_record(ready_receipt),
        "data": {
            "manifest": file_record(manifest_path),
            "file": file_record(data_path),
            "minimum_recomputed_gap_tokens": min(recomputed_gaps),
            "maximum_recomputed_gap_tokens": max(recomputed_gaps),
            "independent_forbidden_overlap": forbidden_rechecks,
        },
        "adapters": adapters,
        "code": code,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
        },
        "storage": {
            "output_root": str(output_root),
            "free_bytes": int(free_bytes),
        },
        "commands": commands,
    }
    atomic_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(receipt_path),
                "sha256": sha256_file(receipt_path),
                "minimum_gap": min(recomputed_gaps),
                "maximum_gap": max(recomputed_gaps),
                "free_bytes": int(free_bytes),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
