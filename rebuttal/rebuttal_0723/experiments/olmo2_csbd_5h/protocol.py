#!/usr/bin/env python3
"""Frozen constants and artifact helpers for the five-hour CSBD experiment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EXPERIMENT_ID = "olmo2_step30k_csbd_5h_s20260726"
SEED = 20_260_726

MODEL_ID = "allenai/OLMo-2-0425-1B-early-training"
MODEL_REVISION = "6251e24cf3f303f9d64c78456a155a5dbe2a35e8"
MODEL_WEIGHT_CONTRACT = {
    "model-00001-of-00002.safetensors": {
        "bytes": 4_983_360_992,
        "sha256": (
            "7d1186ad3506b5760cfdd3fb099ace4c1339e9f6b98f6e33"
            "af12400d8cff080f"
        ),
    },
    "model-00002-of-00002.safetensors": {
        "bytes": 956_326_560,
        "sha256": (
            "f72521ef281a54c337238d8f836661c9f5f50c93b8d397c"
            "66435939e37abeb75"
        ),
    },
}

SHORT_LENGTH = 4_096
LONG_LENGTH = 16_384
EXTRAPOLATION_LENGTHS = (24_576, 32_768)
FACTS = 32
QUERIES = 8

TRAIN_VALUE_COUNT = 2_048
EVAL_VALUE_COUNT = 512

TRAIN_ROWS_PER_BAND = 512
VALIDATION_ROWS = 128
FINAL_TEST_ROWS = 128
EXTRAPOLATION_ROWS = 64

TRAIN_DISTANCE_BANDS = {
    "train_low": (4_200, 5_800),
    "train_mid": (8_200, 9_800),
    "train_high": (12_200, 13_800),
}
HELDOUT_DISTANCE_BANDS = (
    (6_200, 7_800),
    (10_200, 11_800),
    (14_200, 15_300),
)
EXTRAPOLATION_DISTANCE_BANDS = {
    24_576: ((17_000, 19_000), (20_000, 22_000)),
    32_768: ((23_000, 25_000), (27_000, 30_000)),
}

TEACHER_RANK = 64
TEACHER_ALPHA = 128.0
TEACHER_MAX_STEPS = 500
TEACHER_GATE_EXACT = 0.95

STUDENT_RANK = 64
STUDENT_ALPHA = 128.0
STUDENT_MAX_STEPS = 1_000
STUDENT_GATE_STEP = 500

LOSS_WEIGHTS = {
    "ce": 1.0,
    "top1": 0.5,
    "binding": 1.0,
    "delta_hidden": 0.25,
    "retention": 0.1,
}
TOP1_MARGIN = 1.0
BINDING_MARGIN = 2.0
DELTA_NORM_WEIGHT = 0.25
DELTA_EPS = 1.0e-6
RETENTION_EVERY_STEPS = 10

PRIMARY_SUCCESS_GATES = {
    "validation_16k_first_query_exact_min": 0.60,
    "csbd_minus_ce_16k_exact_pp_min": 30.0,
    "validation_16k_preference_flip_min": 0.90,
    "extrapolation_24k_first_query_exact_min": 0.30,
    "median_top1_margin_strictly_positive": True,
    "natural_4k_ppl_relative_regression_max": 0.02,
    "short_binding_exact_drop_pp_max": 2.0,
}


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    array = np.ascontiguousarray(value)
    view = memoryview(array).cast("B")
    for start in range(0, len(view), 16 * 1024 * 1024):
        digest.update(view[start : start + 16 * 1024 * 1024])
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def file_manifest(paths: Iterable[Path], root: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for path in sorted(Path(value) for value in paths):
        relative = str(path.relative_to(root))
        rows[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return rows


def verify_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = Path(path).resolve()
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_config = {
        "model_type": "olmo2",
        "hidden_size": 2_048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "max_position_embeddings": 4_096,
        "rope_theta": 500_000.0,
        "vocab_size": 100_352,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"step-30K config drift for {key}: "
                f"{config.get(key)!r} != {expected!r}"
            )
    weights: dict[str, Any] = {}
    for name, expected in MODEL_WEIGHT_CONTRACT.items():
        source = checkpoint / name
        if source.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"{source} byte-size mismatch")
        actual = sha256_file(source)
        if actual != expected["sha256"]:
            raise RuntimeError(f"{source} SHA-256 mismatch")
        weights[name] = dict(expected)
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "checkpoint": str(checkpoint),
        "config": expected_config,
        "config_sha256": sha256_file(config_path),
        "tokenizer_sha256": sha256_file(checkpoint / "tokenizer.json"),
        "weights": weights,
    }

