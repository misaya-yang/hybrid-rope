#!/usr/bin/env python3
"""Create the no-GPU READY receipt for the clean Tulu recovery arm."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    assert_frequency_contract,
)

from .train_4k_stage_a import composite_checkpoint_sha256


STATUS = "OLMO2_4K_TULU_RECOVERY_READY"
LENGTH = 4_096
EXPECTED_TRAIN_ROWS = 3_968
EXPECTED_VALIDATION_ROWS = 128
TULU_DATASET = "allenai/tulu-3-sft-olmo-2-mixture-0225"
TULU_REVISION = "d91a0785ade02942520280fb484866fce41e448f"
CRITICAL_CODE_PATHS = (
    "rebuttal/rebuttal_0723/experiments/"
    "olmo2_lora_maturity/train_4k_tulu_recovery.py",
    "rebuttal/rebuttal_0723/experiments/"
    "olmo2_lora_maturity/evaluate_instruct_ruler_transfer.py",
    "rebuttal/rebuttal_0723/experiments/"
    "olmo2_lora_maturity/train_screen.py",
    "rebuttal/rebuttal_0723/experiments/"
    "olmo2_lora_maturity/prepare_data.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_conversion.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_ood_factorial.py",
    "rebuttal/rebuttal_0723/experiments/"
    "small_model_lora_conversion.py",
    "rebuttal/rebuttal_0723/experiments/"
    "olmo2_1b_evq/contract.py",
    "rebuttal/rebuttal_0723/experiments/"
    "olmo2_1b_evq/train.py",
    "rebuttal/rebuttal_0723/experiments/"
    "olmo2_1b_evq/prepare_retrieval_data.py",
    "rebuttal/rebuttal_0723/experiments/"
    "olmo2_1b_evq/evaluate_ruler.py",
    "rebuttal/rebuttal_0723/experiments/"
    "olmo2_lora_generalization.py",
    "rebuttal/rebuttal_0723/experiments/"
    "olmo2_lora_maturity/causal_data.py",
    "rebuttal/rebuttal_0723/experiments/"
    "olmo2_lora_maturity/evaluate_instruct_ruler_screen.py",
    "rebuttal/rebuttal_0723/experiments/"
    "olmo2_lora_maturity/prepare_instruct_ruler_transfer.py",
    "rebuttal/rebuttal_0723/experiments/"
    "olmo2_lora_maturity/train_4k_stage_a.py",
    "scripts/lib/rope/schedules.py",
)


def file_entry(path: Path) -> dict[str, Any]:
    return {
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def checkpoint_entry(checkpoint: Path) -> dict[str, Any]:
    weights = sorted(checkpoint.glob("model-*.safetensors"))
    if not weights:
        single = checkpoint / "model.safetensors"
        if not single.is_file():
            raise RuntimeError("checkpoint has no safetensor weights")
        weights = [single]
    required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    required.extend(path.name for path in weights)
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        required.append(index.name)
    files = {}
    for name in required:
        path = checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = file_entry(path)
    return {
        "path": str(checkpoint),
        "files": files,
        "composite_sha256": composite_checkpoint_sha256(checkpoint),
    }


def tulu_entry(
    view: Path,
    *,
    checkpoint_tokenizer_sha256: str,
) -> dict[str, Any]:
    required = (
        "manifest.json",
        "input_ids.npy",
        "assistant_mask.npy",
        "lengths.npy",
        "rows.jsonl",
        "split.npy",
    )
    files = {}
    for name in required:
        path = view / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = file_entry(path)
    manifest = json.loads(
        (view / "manifest.json").read_text(encoding="utf-8")
    )
    source = manifest.get("source", {})
    if (
        source.get("id") != TULU_DATASET
        or source.get("revision") != TULU_REVISION
    ):
        raise RuntimeError("view is not the pinned official Tulu source")
    if (
        manifest.get("rendering", {}).get("labels")
        != "assistant_content_and_eos_only"
    ):
        raise RuntimeError("Tulu view objective drift")
    manifest_tokenizer = (
        manifest.get("tokenizer", {})
        .get("files", {})
        .get("tokenizer.json", {})
        .get("sha256")
    )
    if manifest_tokenizer != checkpoint_tokenizer_sha256:
        raise RuntimeError("Tulu/Instruct tokenizer hash mismatch")

    input_ids = np.load(
        view / "input_ids.npy", mmap_mode="r", allow_pickle=False
    )
    assistant_mask = np.load(
        view / "assistant_mask.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    lengths = np.load(
        view / "lengths.npy", mmap_mode="r", allow_pickle=False
    )
    split = np.load(
        view / "split.npy", mmap_mode="r", allow_pickle=False
    )
    if (
        input_ids.shape != assistant_mask.shape
        or input_ids.shape[1] != LENGTH
        or lengths.shape != (input_ids.shape[0],)
        or split.shape != (input_ids.shape[0],)
    ):
        raise RuntimeError("Tulu fixed-view shape drift")
    if (
        input_ids.dtype != np.int32
        or assistant_mask.dtype != np.uint8
        or lengths.dtype != np.int32
        or split.dtype != np.uint8
    ):
        raise RuntimeError("Tulu fixed-view dtype drift")
    if int(input_ids.min()) < 0 or int(input_ids.max()) >= 100_352:
        raise RuntimeError("Tulu token id exceeds model vocabulary")
    if not np.all((assistant_mask == 0) | (assistant_mask == 1)):
        raise RuntimeError("Tulu assistant mask is not binary")
    train_rows = np.flatnonzero(split == 0)
    validation_rows = np.flatnonzero(split == 1)
    if (
        len(train_rows) != EXPECTED_TRAIN_ROWS
        or len(validation_rows) != EXPECTED_VALIDATION_ROWS
    ):
        raise RuntimeError("Tulu split-size drift")
    if np.any(lengths <= 1) or np.any(lengths > LENGTH):
        raise RuntimeError("Tulu row length escaped the 4K maximum")
    positions = np.arange(LENGTH, dtype=np.int32)[None, :]
    for offset in range(0, len(lengths), 256):
        local_lengths = lengths[offset : offset + 256, None]
        local_mask = assistant_mask[offset : offset + 256]
        if np.any(local_mask[positions >= local_lengths]):
            raise RuntimeError("assistant mask labels padded positions")
    supervised = int(
        assistant_mask[train_rows, 1:].sum(dtype=np.int64)
    )
    if supervised <= 0:
        raise RuntimeError("Tulu training view has no assistant labels")

    row_metadata = [
        json.loads(line)
        for line in (view / "rows.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    if len(row_metadata) != len(input_ids):
        raise RuntimeError("Tulu row-metadata count drift")
    if [int(row["row"]) for row in row_metadata] != list(
        range(len(row_metadata))
    ):
        raise RuntimeError("Tulu row metadata is out of order")
    train_sources = {
        str(row_metadata[index]["source_id"])
        for index in train_rows
    }
    validation_sources = {
        str(row_metadata[index]["source_id"])
        for index in validation_rows
    }
    if train_sources & validation_sources:
        raise RuntimeError("Tulu train/validation source overlap")
    source_markers = ("ruler", "niah", "needle-in-a-haystack")
    explicit_benchmark_sources = sorted(
        source_id
        for source_id in train_sources
        if any(marker in source_id.lower() for marker in source_markers)
    )
    if explicit_benchmark_sources:
        raise RuntimeError(
            "Tulu view contains explicitly named RULER/NIAH sources"
        )

    def content_hash(index: int) -> str:
        length = int(lengths[index])
        raw = np.ascontiguousarray(
            input_ids[index, :length]
        ).view(np.uint8)
        return hashlib.sha256(raw).hexdigest()

    train_content = {content_hash(int(index)) for index in train_rows}
    validation_content = {
        content_hash(int(index)) for index in validation_rows
    }
    if train_content & validation_content:
        raise RuntimeError("Tulu train/validation token overlap")

    def set_hash(values: set[str]) -> str:
        digest = hashlib.sha256()
        for value in sorted(values):
            digest.update(value.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    return {
        "path": str(view),
        "files": files,
        "shape": [int(value) for value in input_ids.shape],
        "training_rows": int(len(train_rows)),
        "validation_rows": int(len(validation_rows)),
        "training_assistant_tokens": supervised,
        "train_source_ids_sha256": set_hash(train_sources),
        "validation_source_ids_sha256": set_hash(
            validation_sources
        ),
        "train_content_hashes_sha256": set_hash(train_content),
        "validation_content_hashes_sha256": set_hash(
            validation_content
        ),
        "train_validation_source_overlap": 0,
        "train_validation_content_overlap": 0,
        "explicit_ruler_or_niah_source_ids": 0,
        "manifest": manifest,
    }


def parent_entry(
    adapter: Path,
    *,
    checkpoint_sha256: str,
    frequency_sha256: str,
) -> dict[str, Any]:
    payload = torch.load(
        adapter, map_location="cpu", weights_only=True
    )
    state = payload.get("state", {})
    metadata = dict(payload.get("metadata", {}))
    if (
        metadata.get("frequency") != "evq"
        or metadata.get("frequency_sha256_float32")
        != frequency_sha256
        or metadata.get("adaptation") != "qkvo_answer"
        or int(metadata.get("training_sequence_length", -1))
        != LENGTH
        or int(metadata.get("rank", -1)) != 64
        or float(metadata.get("alpha", -1.0)) != 128.0
    ):
        raise RuntimeError("parent Stage-A adapter contract drift")
    if metadata.get("base_checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("parent adapter base-checkpoint mismatch")
    expected = {}
    for layer in range(16):
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            prefix = (
                f"model.model.layers.{layer}.self_attn."
                f"{projection}"
            )
            expected[f"{prefix}.a"] = (64, 2_048)
            expected[f"{prefix}.b"] = (2_048, 64)
    if set(state) != set(expected):
        raise RuntimeError("parent adapter parameter-key drift")
    for name, shape in expected.items():
        value = state[name]
        if (
            tuple(value.shape) != shape
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise RuntimeError(
                f"parent adapter tensor contract drift: {name}"
            )
    return {
        "path": str(adapter),
        **file_entry(adapter),
        "metadata": metadata,
        "state_tensors": len(state),
        "state_contract": "16_layers_x_qkvo_x_ab_r64_fp32_finite",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--tulu-view", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    checkpoint = args.checkpoint.resolve()
    parent = args.parent_adapter.resolve()
    tulu_view = args.tulu_view.resolve()
    background = args.background_dir.resolve()
    code_root = args.code_root.resolve()
    trainer = args.trainer.resolve()
    evaluator = args.evaluator.resolve()

    checkpoint_data = checkpoint_entry(checkpoint)
    tulu_data = tulu_entry(
        tulu_view,
        checkpoint_tokenizer_sha256=checkpoint_data["files"][
            "tokenizer.json"
        ]["sha256"],
    )
    parent_data = parent_entry(
        parent,
        checkpoint_sha256=checkpoint_data["composite_sha256"],
        frequency_sha256=assert_frequency_contract()[
            "evq_sha256_float32"
        ],
    )
    background_files = {}
    for name in (
        "manifest.json",
        "documents_L16384.npy",
        "documents_L16384.metadata.json",
    ):
        path = background / name
        if not path.is_file():
            raise FileNotFoundError(path)
        background_files[name] = file_entry(path)
    for path in (code_root, trainer, evaluator):
        if not path.exists():
            raise FileNotFoundError(path)
    code_entries = {}
    for relative in CRITICAL_CODE_PATHS:
        path = code_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        code_entries[relative] = file_entry(path)
    if trainer != code_root / CRITICAL_CODE_PATHS[0]:
        raise RuntimeError("trainer path is outside the frozen code map")
    if evaluator != code_root / CRITICAL_CODE_PATHS[1]:
        raise RuntimeError("evaluator path is outside the frozen code map")

    from liger_kernel.transformers import (
        LigerFusedLinearCrossEntropyLoss,
    )

    LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100,
        reduction="mean",
        return_z_loss=False,
        accum_dtype=torch.float32,
    )
    liger_version = importlib.metadata.version("liger-kernel")

    receipt = {
        "status": STATUS,
        "reviewer_concerns": ["R27bE.2", "R27bE.5", "AC.2"],
        "existing_evidence": (
            "LongAlign-only Stage A improves long-context NLL but the old "
            "adapter is below Native plus official YaRN on full RULER."
        ),
        "smallest_missing_evidence": (
            "Whether restoring instruction following with official 4K "
            "Tulu data transfers to RULER tasks never used for adaptation."
        ),
        "smallest_executable_plan": (
            "Load the frozen EVQ Stage-A adapter; run one deterministic "
            "496-step pass over the frozen 3,968-row official-source Tulu "
            "view; evaluate the frozen full RULER matrix at 4K, 8K, and "
            "16K."
        ),
        "stop_condition": (
            "Exactly one pass over the frozen Tulu view; no selection or "
            "tuning on RULER."
        ),
        "inputs": {
            "checkpoint": checkpoint_data,
            "parent_adapter": parent_data,
            "tulu_view": tulu_data,
            "background": {
                "path": str(background),
                "files": background_files,
            },
        },
        "code": {
            "root": str(code_root),
            "critical_files": code_entries,
            "preflight": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "trainer": {
                "path": str(trainer),
                "sha256": sha256_file(trainer),
            },
            "evaluator": {
                "path": str(evaluator),
                "sha256": sha256_file(evaluator),
            },
        },
        "protocol": {
            "maximum_training_length": LENGTH,
            "epochs": 1,
            "training_rows": EXPECTED_TRAIN_ROWS,
            "validation_rows": EXPECTED_VALIDATION_ROWS,
            "global_batch_size": 8,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "steps": 496,
            "rank": 64,
            "alpha": 128.0,
            "learning_rate": 2e-5,
            "warmup_ratio": 0.05,
            "compile_mode": "max-autotune-no-cudagraphs",
            "natural_eval_rows": 16,
            "seed": 20_260_728,
            "explicit_custom_binding_or_ruler_training_rows": 0,
            "allowed_gpu_name_substrings": [
                "RTX 5090",
                "RTX PRO 6000",
            ],
            "required_compute_capability": [12, 0],
            "required_compile_cache": True,
            "required_expandable_segments": True,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "liger_kernel": liger_version,
            "cuda_api_used": False,
        },
    }
    atomic_json(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
