"""Fail-closed receipts and authorization helpers for phase AdaRoPE."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


METHOD_ID = "olmo2_phase_adarope_headwise_v1"
GPU_AUTH_ENV = "PHASE_ADAROPE_GPU_AUTHORIZED"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return sha256_file(path)


def require_dual_authorization(*, cli_authorize: bool, environment: Mapping[str, str] | None = None) -> None:
    env = os.environ if environment is None else environment
    if not cli_authorize or env.get(GPU_AUTH_ENV) != "1":
        raise PermissionError("GPU action requires --authorize and PHASE_ADAROPE_GPU_AUTHORIZED=1")


def no_cuda_preflight() -> dict[str, Any]:
    return {
        "status": "CPU_PREFLIGHT_ONLY",
        "cuda_initialized": False,
        "training_attempted": False,
        "evaluation_attempted": False,
        "gpu_authorization_required_for_run": True,
    }


def stage_receipt(*, stage: str, arm: str, parent_sha256: str | None, contract: Mapping[str, Any], status: str) -> dict[str, Any]:
    if stage not in {"stage0", "stage1", "stage2"}:
        raise ValueError("unknown stage")
    receipt = {
        "method_id": METHOD_ID,
        "stage": stage,
        "arm": arm,
        "parent_sha256": parent_sha256,
        "contract": dict(contract),
        "status": status,
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    return receipt


def assert_sidecar_hash(path: Path, expected_sha256: str) -> None:
    if sha256_file(path.resolve()) != expected_sha256:
        raise ValueError("frequency sidecar hash drift")


def assert_peft_roundtrip(metadata: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key, value in expected.items():
        if key == "target_modules":
            if set(metadata.get(key, ())) != set(value):
                raise ValueError(f"PEFT roundtrip metadata drift: {key}")
            continue
        if metadata.get(key) != value:
            raise ValueError(f"PEFT roundtrip metadata drift: {key}")
