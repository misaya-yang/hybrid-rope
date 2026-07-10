#!/usr/bin/env python3
"""Fail-closed validator for final legacy LongAlign LoRA adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    from .legacy_lora_protocol import (
        LEGACY_OBJECTIVE,
        canonical_json_sha256,
        sha256_file,
        validate_legacy_protocol,
    )
except ImportError:
    from legacy_lora_protocol import (
        LEGACY_OBJECTIVE,
        canonical_json_sha256,
        sha256_file,
        validate_legacy_protocol,
    )


def validate_legacy_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_method: str,
    expected_seed: int,
) -> Dict[str, Any]:
    if metadata.get("objective") != LEGACY_OBJECTIVE:
        raise ValueError("legacy adapter objective mismatch")
    if metadata.get("status") != "complete":
        raise ValueError("legacy adapter status is not complete")
    if int(metadata.get("global_step", -1)) != 300:
        raise ValueError("legacy adapter must finish exactly global_step=300")
    protocol = validate_legacy_protocol(metadata.get("protocol", {}))
    if protocol["method"] != expected_method or protocol["seed"] != expected_seed:
        raise ValueError("legacy adapter method/seed mismatch")
    recorded_protocol_hash = metadata.get("protocol_sha256")
    # Early unit-test metadata may use a placeholder digest; artifact-level
    # validation below always requires the exact digest.
    if not isinstance(recorded_protocol_hash, str) or len(recorded_protocol_hash) != 64:
        raise ValueError("legacy adapter protocol SHA-256 is missing")
    for field in ("adapter_sha256", "frequency_sha256"):
        value = metadata.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"legacy adapter {field} is missing")
    for field in ("data_manifest_sha256", "model_manifest_sha256", "code_sha256"):
        if metadata.get(field) != protocol[field]:
            raise ValueError(f"legacy adapter top-level {field} mismatch")
    if not isinstance(metadata.get("runtime"), dict):
        raise ValueError("legacy adapter runtime identity is missing")
    return dict(metadata)


def validate_adapter_config(config: Mapping[str, Any]) -> None:
    if int(config.get("r", -1)) != 64:
        raise ValueError("adapter rank mismatch")
    if int(config.get("lora_alpha", -1)) != 128:
        raise ValueError("adapter alpha mismatch")
    if abs(float(config.get("lora_dropout", -1.0)) - 0.05) > 1e-12:
        raise ValueError("adapter dropout mismatch")
    if set(config.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj"}:
        raise ValueError("adapter target_modules mismatch")


def validate_artifact(
    adapter_dir: Path,
    *,
    expected_method: str,
    expected_seed: int,
) -> Dict[str, Any]:
    adapter_dir = Path(adapter_dir)
    required = {
        "adapter": adapter_dir / "adapter_model.safetensors",
        "config": adapter_dir / "adapter_config.json",
        "frequency": adapter_dir / "custom_inv_freq.pt",
        "metadata": adapter_dir / "experiment_meta.json",
        "trainer_state": adapter_dir / "trainer_state.json",
        "protocol": adapter_dir / "run_protocol.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"legacy artifact missing: {', '.join(missing)}")
    metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
    validate_legacy_metadata(
        metadata,
        expected_method=expected_method,
        expected_seed=expected_seed,
    )
    protocol = json.loads(required["protocol"].read_text(encoding="utf-8"))
    validate_legacy_protocol(protocol)
    if metadata["protocol"] != protocol:
        raise ValueError("embedded and standalone legacy protocols differ")
    if metadata["protocol_sha256"] != canonical_json_sha256(protocol):
        raise ValueError("legacy protocol digest mismatch")
    trainer_state = json.loads(required["trainer_state"].read_text(encoding="utf-8"))
    if int(trainer_state.get("global_step", -1)) != 300:
        raise ValueError("trainer_state does not prove global_step=300")
    validate_adapter_config(json.loads(required["config"].read_text(encoding="utf-8")))
    if metadata["adapter_sha256"] != sha256_file(required["adapter"]):
        raise ValueError("adapter weight SHA-256 mismatch")
    if metadata["frequency_sha256"] != sha256_file(required["frequency"]):
        raise ValueError("frequency artifact SHA-256 mismatch")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--expected_method", choices=("native_geo", "evq_cosh"), required=True)
    parser.add_argument("--expected_seed", type=int, choices=(42, 43, 44), required=True)
    args = parser.parse_args()
    metadata = validate_artifact(
        args.adapter_dir,
        expected_method=args.expected_method,
        expected_seed=args.expected_seed,
    )
    print(json.dumps({
        "status": "valid",
        "adapter": args.adapter_dir.name,
        "method": metadata["protocol"]["method"],
        "seed": metadata["protocol"]["seed"],
        "global_step": metadata["global_step"],
    }, indent=2))


if __name__ == "__main__":
    main()
