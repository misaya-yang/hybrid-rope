#!/usr/bin/env python3
"""Canonical 151.9M contract and R1'/R2' matrix definitions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .schedule import DEFAULT_BASE, DEFAULT_K, LAMBDA_VALUES, sha256_file


SEEDS: tuple[int, ...] = (42, 137, 256)
CANONICAL_OWNER = Path(__file__).resolve().parents[4] / (
    "paper-2027/research/EXACT_RANGE_151M_3SEED_RESULT_20260820.json"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_canonical_protocol(path: str | Path = CANONICAL_OWNER) -> dict[str, Any]:
    """Validate and return immutable fields required by the new matrix."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("canonical exact-range JSON has no protocol object")
    expected = {
        "model_parameters": 151_898_880,
        "training_length": 256,
        "training_tokens_per_arm": 499_974_144,
        "optimizer_steps": 7_629,
        "evaluation_anchor_count": 32,
        "evaluation_tail_tokens": 128,
        "global_batch_sequences": 256,
        "aggregation_unit": "training_seed",
    }
    for key, value in expected.items():
        actual = protocol.get(key)
        if actual != value:
            raise ValueError(f"canonical protocol mismatch for {key}: {actual!r} != {value!r}")
    seeds = tuple(int(seed) for seed in payload.get("training_seeds", ()))
    if seeds != SEEDS:
        raise ValueError(f"canonical seeds {seeds!r} do not equal {SEEDS!r}")
    evaluation_lengths = tuple(int(v) for v in protocol["evaluation_lengths"])
    if evaluation_lengths != (256, 512, 1024, 2048):
        raise ValueError("canonical evaluation lengths changed")
    return {
        "source_path": str(source),
        "source_relpath": str(source.relative_to(repo_root())),
        "source_sha256": sha256_file(source),
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
        "paper_role": payload.get("paper_role"),
        "training_seeds": list(seeds),
        **expected,
        "evaluation_lengths": list(evaluation_lengths),
        "seed_micro_batch_accumulation": {"42": [64, 4], "137": [128, 2], "256": [128, 2]},
        "optimizer": {
            "name": "fused AdamW",
            "beta1": 0.9,
            "beta2": 0.95,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "peak_lr": 6e-4,
            "minimum_lr": 6e-5,
            "warmup_steps": 762,
            "schedule": "cosine",
        },
        "r0_data_receipts": dict(payload.get("new_run_data_receipts", {})),
        "claim_boundary": dict(payload.get("claim_boundary", {})),
    }


def protocol_fingerprint(protocol: dict[str, Any]) -> str:
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def r2_arm_names() -> tuple[str, ...]:
    return (
        "geo",
        "cosh_tau4",
        *(f"demand_lambda_{value:g}".replace(".", "p") for value in LAMBDA_VALUES),
    )


def r1_arm_names() -> tuple[str, ...]:
    names: list[str] = []
    for mode in ("anchored_tail", "mid_only"):
        names.extend(f"{mode}_demand_lambda_{value:g}".replace(".", "p") for value in LAMBDA_VALUES)
    return tuple(names)


def r2_training_matrix() -> list[dict[str, Any]]:
    return [
        {"suite": "R2", "arm": arm, "seed": seed, "status": "NOT_STARTED", "training_authorized": False}
        for seed in SEEDS
        for arm in r2_arm_names()
    ]


def r1_training_matrix() -> list[dict[str, Any]]:
    """Register all anchored-tail/mid-only controls for all three seeds."""

    return [
        {
            "suite": "R1",
            "arm": arm,
            "seed": seed,
            "status": "NOT_STARTED",
            "training_authorized": False,
        }
        for seed in SEEDS
        for arm in r1_arm_names()
    ]
