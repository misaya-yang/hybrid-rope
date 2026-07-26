#!/usr/bin/env python3
"""Create a no-GPU READY receipt for the mature-Instruct RULER matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .prepare_data import atomic_json, sha256_file


EXPECTED_MODEL_SHA256 = (
    "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
)
EXPECTED_TOKENIZER_SHA256 = (
    "73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca"
)
EXPECTED_RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
EXPECTED_PUNKT_TAB_SHA256 = (
    "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
)
EXPECTED_ADAPTERS = {
    "native_stage_a": (
        "f4bc6c471cd395eb0df25f6513d150d3d66a7133c2ac3a57406765708843574e",
        "native",
    ),
    "evq_stage_a": (
        "47e72c5e58be443a3f088787b58df415538f7cdb12e9f76d799dbb2b85155ee0",
        "evq",
    ),
    "native_final": (
        "6570ab94aec68431dd4e261eb3ef342ef72253357aa65a0d37b0a018df2f3f8d",
        "native",
    ),
    "evq_final": (
        "95ceeb70117c73233915760a9756b9b2a98416ec188b125054da8ced75cad16a",
        "evq",
    ),
}
CODE_FILES = (
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
    "prepare_instruct_ruler_screen.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
    "evaluate_instruct_ruler_screen.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
    "preflight_instruct_ruler_matrix.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
    "run_pro6000_instruct_ruler_matrix.sh",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
    "run_pro6000_instruct_stagea_matrix.sh",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
    "train_4k_stage_a.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
    "train_4k_counterfactual_routing.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_conversion.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_ood_factorial.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_1b_evq/evaluate_ruler.py",
    "scripts/lib/rope/schedules.py",
)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": sha256_file(path),
    }


def verify_adapter(path: Path, expected_sha: str, frequency: str) -> dict[str, Any]:
    record = file_record(path)
    if record["sha256"] != expected_sha:
        raise RuntimeError(f"adapter checksum drift: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = dict(payload.get("metadata", {}))
    expected = {
        "base_checkpoint_sha256": EXPECTED_MODEL_SHA256,
        "frequency": frequency,
        "adaptation": "qkvo_answer",
        "rank": 64,
        "alpha": 128.0,
        "training_sequence_length": 4_096,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"adapter metadata drift for {path.name}:{key}"
            )
    record["metadata"] = metadata
    return record


def verify_dataset(
    root: Path,
    *,
    task: str,
    seed: int,
    checkpoint: Path,
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "OLMO2_INSTRUCT_RULER_SCREEN_PREPARED":
        raise RuntimeError(f"invalid dataset status: {root}")
    expected = {
        "task": task,
        "seed": seed,
        "samples_per_length": 128,
        "lengths": [4_096, 8_192, 16_384],
        "ruler_commit": EXPECTED_RULER_COMMIT,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"dataset drift for {task}:{key}")
    if Path(manifest["checkpoint"]).resolve() != checkpoint.resolve():
        raise RuntimeError(f"dataset checkpoint drift for {task}")
    files = {}
    for length in expected["lengths"]:
        entry = manifest["files"][str(length)]
        path = root / entry["relative_path"]
        record = file_record(path)
        if record["sha256"] != entry["sha256"]:
            raise RuntimeError(f"dataset file hash drift for {task}:{length}")
        if int(entry["rows"]) != 128:
            raise RuntimeError(f"dataset row-count drift for {task}:{length}")
        files[str(length)] = record
    return {
        "root": str(root.resolve()),
        "manifest": file_record(manifest_path),
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--data-single-1", type=Path, required=True)
    parser.add_argument("--data-single-2", type=Path, required=True)
    parser.add_argument("--native-stage-a", type=Path, required=True)
    parser.add_argument("--evq-stage-a", type=Path, required=True)
    parser.add_argument("--native-final", type=Path, required=True)
    parser.add_argument("--evq-final", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20_260_726)
    parser.add_argument(
        "--matrix-kind",
        choices=("final", "stage-a"),
        default="final",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("preflight must hide CUDA")
    if torch.cuda.is_available():
        raise RuntimeError("preflight unexpectedly sees CUDA")

    receipt_path = args.receipt.resolve()
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    checkpoint = args.checkpoint.resolve()
    model = file_record(checkpoint / "model.safetensors")
    tokenizer = file_record(checkpoint / "tokenizer.json")
    if model["sha256"] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("checkpoint checksum drift")
    if tokenizer["sha256"] != EXPECTED_TOKENIZER_SHA256:
        raise RuntimeError("tokenizer checksum drift")

    ruler_root = args.ruler_root.resolve()
    ruler_commit = subprocess.run(
        ["git", "-C", str(ruler_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if ruler_commit != EXPECTED_RULER_COMMIT:
        raise RuntimeError("RULER commit drift")
    essay_provenance_path = (
        ruler_root
        / "scripts/data/synthetic/json/"
        "PaulGrahamEssays.provenance.json"
    )
    essay_provenance = json.loads(
        essay_provenance_path.read_text(encoding="utf-8")
    )
    if (
        essay_provenance.get("status")
        != "PARTIAL_OFFICIAL_RULER_ESSAY_CORPUS"
        or int(essay_provenance["downloaded_html_essay_count"]) != 169
    ):
        raise RuntimeError("essay corpus provenance drift")
    asset_root = ruler_root.parent.parent
    punkt_path = (
        asset_root / "nltk_data/tokenizers/punkt_tab.zip"
    )
    if sha256_file(punkt_path) != EXPECTED_PUNKT_TAB_SHA256:
        raise RuntimeError("NLTK punkt_tab checksum drift")

    overlap_path = args.data_single_1.resolve().parent / "overlap_audit.json"
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    if overlap.get("status") != (
        "OLMO2_INSTRUCT_RULER_FRESH_OVERLAP_AUDITED"
    ):
        raise RuntimeError("fresh-data overlap audit status drift")
    if any(int(value) != 0 for value in overlap["overlap"].values()):
        raise RuntimeError("fresh data overlaps training or prior evaluation")

    adapter_paths = {
        "native_stage_a": args.native_stage_a.resolve(),
        "evq_stage_a": args.evq_stage_a.resolve(),
        "native_final": args.native_final.resolve(),
        "evq_final": args.evq_final.resolve(),
    }
    adapters = {
        name: verify_adapter(path, *EXPECTED_ADAPTERS[name])
        for name, path in adapter_paths.items()
    }
    code_root = args.code_root.resolve()
    code = {
        relative: file_record(code_root / relative)
        for relative in CODE_FILES
    }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_root).free
    if free_bytes < 8_000_000_000:
        raise RuntimeError("less than 8 GB output space remains")

    if args.matrix_kind == "final":
        scientific_contract = {
            "existing_evidence": (
                "single-seed n=20 matched native/EVQ niah_single_1"
            ),
            "smallest_missing_evidence": (
                "fresh-seed n=128 confirmation plus essay-haystack shift"
            ),
            "arms": ["native_final", "evq_final"],
            "interpretation": (
                "confirm schedule-dependent length transfer after identical "
                "counterfactual routing repair"
            ),
        }
    else:
        scientific_contract = {
            "existing_evidence": (
                "n=20 Stage-A screen: native 4K=100%, EVQ 4K=5%, "
                "both 8K/16K=0%"
            ),
            "smallest_missing_evidence": (
                "fresh-seed n=128 Stage-A control on noise and essay "
                "haystacks"
            ),
            "arms": ["native_stage_a", "evq_stage_a"],
            "interpretation": (
                "identify the capability delta attributable to "
                "counterfactual routing after full-token PPL adaptation"
            ),
        }
    scientific_contract.update(
        {
            "tasks": ["niah_single_1", "niah_single_2"],
            "lengths": [4_096, 8_192, 16_384],
            "primary_metric": "strict first-number autoregressive exact",
            "secondary_metric": "official RULER substring match",
            "training": "none",
            "stop_condition": (
                "do not interpret long-context single_2 if both arms are "
                "below 25% strict exact at 4K"
            ),
        }
    )
    receipt = {
        "format_version": 1,
        "status": "OLMO2_INSTRUCT_PRO6000_RULER_MATRIX_READY",
        "created_at_unix": int(__import__("time").time()),
        "review_concerns": ["R27bE.2", "R27bE.5", "AC.2", "AC.4"],
        "matrix_kind": args.matrix_kind,
        "scientific_contract": scientific_contract,
        "checkpoint": {
            "checkpoint_path": str(checkpoint),
            "files": {"model.safetensors": model},
            "composite_sha256": model["sha256"],
            "tokenizer_sha256": tokenizer["sha256"],
        },
        "tokenizer": tokenizer,
        "ruler": {
            "root": str(ruler_root),
            "commit": ruler_commit,
            "essay_corpus_provenance": {
                "file": file_record(essay_provenance_path),
                "content": essay_provenance,
            },
            "punkt_tab": file_record(punkt_path),
        },
        "adapters": adapters,
        "datasets": {
            "niah_single_1": verify_dataset(
                args.data_single_1.resolve(),
                task="niah_single_1",
                seed=int(args.seed),
                checkpoint=checkpoint,
            ),
            "niah_single_2": verify_dataset(
                args.data_single_2.resolve(),
                task="niah_single_2",
                seed=int(args.seed),
                checkpoint=checkpoint,
            ),
            "overlap_audit": {
                "file": file_record(overlap_path),
                "content": overlap,
            },
        },
        "code": code,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "html2text": importlib.metadata.version("html2text"),
            "beautifulsoup4": importlib.metadata.version("beautifulsoup4"),
            "wonderwords": importlib.metadata.version("wonderwords"),
            "tenacity": importlib.metadata.version("tenacity"),
            "nltk": importlib.metadata.version("nltk"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "storage": {
            "output_root": str(output_root),
            "free_bytes": int(free_bytes),
        },
    }
    atomic_json(receipt_path, receipt)
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(receipt_path),
                "sha256": digest,
                "free_bytes": free_bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
