#!/usr/bin/env python3
"""No-GPU R4' dry-run and receipt builder.

The command performs only local JSON, source-hash, asset-binding, host-memory,
and optional Torch capability checks.  It never downloads, loads a checkpoint,
starts a subprocess, allocates a model, or trains.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

from .protocol import (
    ContractError,
    ExecutionConfig,
    GateContract,
    LossWeights,
    NegativeMorphReproduction,
    PhaseCurriculum,
    ProtectedTableConfig,
    QKOnlyProtocol,
    SlowResidualConfig,
    build_protocol_manifest,
    load_json,
    sha256_file,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[3]
CONFIG_PATH = PACKAGE_ROOT / "config.json"
SOURCE_MANIFEST_PATH = PACKAGE_ROOT / "source_manifest.json"


def _status(ok: bool, status: str, **details: Any) -> dict[str, Any]:
    return {"ok": bool(ok), "status": status, **details}


def _source_hash_check(source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = source_manifest.get("sources")
    if not isinstance(rows, list):
        raise ContractError("source manifest sources must be a list")
    checks: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ContractError("source manifest row must be an object")
        relative = row.get("path")
        expected = row.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ContractError("source manifest row needs path and sha256")
        path = REPO_ROOT / relative
        if not path.is_file():
            checks.append(
                {
                    "path": relative,
                    "role": row.get("role"),
                    "status": "MISSING",
                    "ok": False,
                }
            )
            continue
        actual = sha256_file(path)
        checks.append(
            {
                "path": relative,
                "role": row.get("role"),
                "status": "PASS" if actual == expected else "DRIFT",
                "ok": actual == expected,
                "sha256": actual,
            }
        )
    return {
        "status": "PASS" if all(row["ok"] for row in checks) else "BLOCKED",
        "checks": checks,
    }


def _asset_check(bindings: Mapping[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for label in ("checkpoint", "training_view"):
        path_value = bindings.get(f"{label}_path")
        expected = bindings.get(f"{label}_sha256")
        if path_value in (None, ""):
            results[label] = {
                "status": "UNBOUND_EXTERNAL_ASSET",
                "ok": True,
                "provided": False,
            }
            continue
        path = Path(str(path_value))
        if not path.is_file():
            results[label] = {
                "status": "MISSING_EXTERNAL_ASSET",
                "ok": False,
                "provided": True,
            }
            continue
        actual = sha256_file(path)
        results[label] = {
            "status": "PASS" if actual == expected else "DRIFT",
            "ok": actual == expected,
            "provided": True,
            "sha256": actual,
        }
    return results


def _host_memory_check() -> dict[str, Any]:
    if hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return {
                "status": "HOST_RAM_OBSERVED",
                "ok": True,
                "host_ram_gib": pages * page_size / (1024**3),
                "gpu_memory": "UNVERIFIED_NO_GPU_PROBE",
            }
        except (OSError, ValueError):
            pass
    return {
        "status": "HOST_RAM_UNAVAILABLE",
        "ok": True,
        "gpu_memory": "UNVERIFIED_NO_GPU_PROBE",
    }


def _flash_capability_check() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return {
            "status": "UNAVAILABLE_NO_TORCH",
            "ok": True,
            "detail": type(exc).__name__,
            "fallbacks_allowed": False,
        }
    cuda_available = bool(torch.cuda.is_available())
    flash_enabled = bool(
        torch.backends.cuda.flash_sdp_enabled()
        if hasattr(torch.backends, "cuda")
        else False
    )
    memory: dict[str, Any] = {
        "gpu_free_memory_gib": None,
        "gpu_total_memory_gib": None,
    }
    if cuda_available:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        memory = {
            "gpu_free_memory_gib": float(free_bytes) / (1024**3),
            "gpu_total_memory_gib": float(total_bytes) / (1024**3),
        }
    return {
        "status": "GPU_CAPABILITY_QUERIED" if cuda_available else "NO_GPU_VISIBLE",
        "ok": True,
        "cuda_available": cuda_available,
        "flash_sdp_enabled": flash_enabled,
        "fallbacks_allowed": False,
        "training_attempted": False,
        **memory,
    }


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    execution = config.get("execution")
    negative = config.get("negative_morph_reproduction")
    protected = config.get("protected_table")
    residual = config.get("slow_residual")
    matrix = config.get("qk_only_matrix")
    if not all(
        isinstance(value, Mapping)
        for value in (execution, negative, protected, residual, matrix)
    ):
        raise ContractError("config is missing a required contract section")
    if execution.get("default_action") != "dry-run":
        raise ContractError("config default action drift")
    forbidden = (
        execution.get("training_authorized"),
        execution.get("gpu_enabled"),
        execution.get("downloads_allowed"),
        execution.get("network_allowed"),
    )
    if any(bool(value) for value in forbidden):
        raise ContractError("config contains an enabled external action")
    if bool(negative.get("enabled")):
        raise ContractError("negative morph reproduction must remain disabled")
    if bool(protected.get("enabled")) or bool(residual.get("enabled")):
        raise ContractError("new retrofit routes must remain disabled")
    if matrix.get("status") != "PROPOSED_NOT_RUN":
        raise ContractError("Q/K matched matrix must remain proposed")
    return {
        "execution": "PASS_DISABLED",
        "negative_morph": "PASS_DISABLED",
        "protected_table": "PASS_DISABLED",
        "slow_residual": "PASS_DISABLED",
        "qk_matrix": "PASS_PROPOSED_NOT_RUN",
    }


def run_dry_run(
    *,
    config_path: Path = CONFIG_PATH,
    source_manifest_path: Path = SOURCE_MANIFEST_PATH,
) -> dict[str, Any]:
    """Build a deterministic receipt without external side effects."""

    config = load_json(config_path)
    source_manifest = load_json(source_manifest_path)
    config_validation = _validate_config(config)

    # Instantiate the typed contracts as a second schema check.  The values
    # stay disabled; this does not create a model or an optimizer.
    ExecutionConfig().validate()
    NegativeMorphReproduction().validate()
    ProtectedTableConfig().validate()
    SlowResidualConfig().validate()
    qk = QKOnlyProtocol()
    qk.validate()
    LossWeights().validate()
    PhaseCurriculum().validate()
    GateContract().validate()

    source_check = _source_hash_check(source_manifest)
    asset_check = _asset_check(config.get("asset_bindings", {}))
    memory_check = _host_memory_check()
    flash_check = _flash_capability_check()
    all_bound_assets_ok = all(row["ok"] for row in asset_check.values())
    if source_check["status"] != "PASS":
        overall = "BLOCKED_SOURCE_DRIFT"
    elif not all_bound_assets_ok:
        overall = "BLOCKED_ASSET_BINDING"
    else:
        overall = "PREPARED_NO_GPU_EXTERNAL_ASSETS_UNBOUND"

    typed_manifest = build_protocol_manifest(
        source_manifest=source_manifest,
    )
    receipt = {
        "schema_version": 1,
        "method_id": typed_manifest["method_id"],
        "status": overall,
        "config_validation": config_validation,
        "source_hash_check": source_check,
        "asset_check": asset_check,
        "host_memory_check": memory_check,
        "flash_capability_check": flash_check,
        "execution_proof": {
            "training_attempted": False,
            "download_attempted": False,
            "network_access_attempted": False,
            "subprocesses_started": False,
            "model_loaded": False,
            "optimizer_created": False,
        },
        "platform": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "protocol_sha256": typed_manifest["protocol_sha256"],
    }
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    receipt = run_dry_run(
        config_path=args.config,
        source_manifest_path=args.source_manifest,
    )
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
