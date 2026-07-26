"""Shared contracts for the Llama-3-8B physical-8K RULER mixture."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch


RULER_COMMIT = "38da79d79519ef87aa46ae804f838e1eab7f86d7"
TRAIN_LENGTH = 8_192
DEFAULT_EVAL_LENGTHS = (4_096, 8_192, 16_384, 32_768)
TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
GENERATION_TOKENS = {
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multivalue": 128,
    "niah_multiquery": 128,
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa_1": 32,
    "qa_2": 32,
}
OFFICIAL_METRIC = {
    task: (
        "string_match_part"
        if task in {"qa_1", "qa_2"}
        else "string_match_all"
    )
    for task in TASKS
}
SOURCE_STATUS = "LLAMA8B_RULER_SOURCES_READY_V1"
TRAIN_SOURCE_STATUS = "LLAMA8B_RULER_TRAIN_SOURCES_READY_V1"
EVAL_SOURCE_STATUS = "LLAMA8B_RULER_EVAL_SOURCES_READY_V1"
VIEW_STATUS = "LLAMA8B_PHYSICAL_8K_RULER_MIX_READY_V1"
READY_STATUS = "LLAMA8B_RULER_MIX_NO_GPU_READY_V1"
RESULT_STATUS = "LLAMA8B_PHYSICAL_8K_RULER_MIX_COMPLETE_V1"
EVAL_STATUS = "LLAMA8B_RULER_MIX_EVAL_COMPLETE_V1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def row_sha256(row: dict[str, Any]) -> str:
    return canonical_json_sha256(row)


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def append_jsonl(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_target(task: str, outputs: Iterable[Any]) -> str:
    values = [str(value) for value in outputs]
    if not values:
        raise RuntimeError(f"{task} row has no outputs")
    if task == "cwe":
        return " " + " ".join(
            f"{index + 1}. {value}"
            for index, value in enumerate(values)
        )
    if task in {
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multivalue",
        "niah_multiquery",
        "vt",
        "fwe",
    }:
        return " " + ", ".join(values)
    if task in {"qa_1", "qa_2"}:
        return " " + values[0]
    raise ValueError(f"unknown task: {task}")


def official_score(
    prediction: str,
    references: Iterable[Any],
    metric: str,
) -> float:
    values = [str(value).lower() for value in references]
    if not values:
        raise RuntimeError("RULER row has no references")
    lowered = "".join(
        char if char.isprintable() else "\n" for char in prediction
    ).strip().lower()
    if metric == "string_match_all":
        return sum(float(value in lowered) for value in values) / len(values)
    if metric == "string_match_part":
        return max(float(value in lowered) for value in values)
    raise ValueError(f"unknown official metric: {metric}")


def configure_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_sdp_enabled": bool(
            torch.backends.cuda.flash_sdp_enabled()
        ),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(
            torch.backends.cuda.mem_efficient_sdp_enabled()
        ),
        "cudnn_sdp_enabled": (
            bool(torch.backends.cuda.cudnn_sdp_enabled())
            if hasattr(torch.backends.cuda, "cudnn_sdp_enabled")
            else None
        ),
    }
