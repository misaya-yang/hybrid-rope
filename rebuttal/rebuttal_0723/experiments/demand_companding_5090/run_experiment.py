#!/usr/bin/env python3
"""Build the demand-companding manifest; training lives behind a double gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .protocol import (
    SEEDS,
    load_canonical_protocol,
    protocol_fingerprint,
    r1_arm_names,
    r1_training_matrix,
    r2_arm_names,
    r2_training_matrix,
)
from .schedule import (
    DEFAULT_BASE,
    DEFAULT_K,
    DEFAULT_TAU,
    DemandProfile,
    build_schedule_receipts,
    load_r0_profile,
    sha256_file,
)


PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = "demand_companding_5090"


def _hash_package_code() -> str:
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _profile_config(profile: DemandProfile, *, base: float | None, k: int | None) -> tuple[float, int]:
    # R0's ``measurement`` describes the source probe (currently OLMo with
    # head_dim=128/K=64).  It is provenance only; target 151M defaults are
    # deliberately independent and must never be inherited from that source.
    base_value: Any = DEFAULT_BASE if base is None else base
    k_value: Any = DEFAULT_K if k is None else k
    base_f = float(base_value)
    k_i = int(k_value)
    if base_f <= 1.0 or k_i < 2:
        raise ValueError("base must be >1 and K must be >=2")
    return base_f, k_i


def _memory_receipt() -> dict[str, Any]:
    """Record the memory gate without allocating CUDA memory or a model."""

    result: dict[str, Any] = {
        "training_gpu_probe_started": False,
        "required_min_gib": 30.0,
        "status": "PENDING_EXPLICIT_GPU_PREFLIGHT",
        "cuda_available_observed": None,
        "device_name": None,
        "total_memory_gib": None,
        "bf16_required": True,
        "flash_only_required": True,
    }
    try:
        import torch  # type: ignore

        result["cuda_available_observed"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = getattr(props, "total_memory", getattr(props, "total_mem", 0))
            result["device_name"] = str(props.name)
            result["total_memory_gib"] = float(total) / float(2**30)
            if result["total_memory_gib"] < result["required_min_gib"]:
                result["status"] = "STOP_INSUFFICIENT_MEMORY"
    except Exception as exc:  # pragma: no cover - environment-specific
        result["probe_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _data_receipt(canonical: dict[str, Any], data_manifest: str | None) -> dict[str, Any]:
    expected = canonical.get("r0_data_receipts", {}).get("manifest_sha256")
    path = Path(data_manifest).resolve() if data_manifest else None
    observed = sha256_file(path) if path is not None and path.is_file() else None
    if path is None:
        status = "PENDING_DATA_MANIFEST_PATH"
    elif observed != expected:
        status = "STOP_DATA_HASH_MISMATCH"
    else:
        status = "PASS"
    return {
        "status": status,
        "path": str(path) if path is not None else None,
        "expected_sha256": expected,
        "observed_sha256": observed,
        "source": "151.9M exact-range canonical owner new_run_data_receipts.manifest_sha256",
    }


def _checkpoint_receipt(checkpoint_root: str | None) -> dict[str, Any]:
    path = Path(checkpoint_root).resolve() if checkpoint_root else None
    allowed = {".pt", ".pth", ".safetensors", ".bin", ".ckpt"}
    files = []
    if path is not None and path.exists():
        files = sorted(str(item) for item in path.rglob("*") if item.is_file() and item.suffix.lower() in allowed)
    if path is None:
        status = "PENDING_CHECKPOINT_ROOT_PATH"
    elif files:
        status = "STOP_CHECKPOINTS_ALREADY_PRESENT"
    else:
        status = "PASS_EMPTY_BEFORE_TRAINING"
    return {
        "status": status,
        "root": str(path) if path is not None else None,
        "observed_checkpoint_paths": files,
        "checkpoint_hashes": {item: sha256_file(item) for item in files},
        "policy": "no checkpoint may exist before an explicitly authorised run",
    }


def _gate(name: str, status: str, *, blocking_for_training: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "status": status, "blocking_for_training": bool(blocking_for_training), "detail": detail}


def build_manifest(
    r0_json: str | Path,
    *,
    output: str | Path,
    base: float | None = None,
    k: int | None = None,
    tau: float = DEFAULT_TAU,
    data_manifest: str | None = None,
    checkpoint_root: str | None = None,
) -> dict[str, Any]:
    """Build and write a deterministic CPU-only R1'/R2' manifest."""

    profile = load_r0_profile(r0_json)
    canonical = load_canonical_protocol()
    base_f, k_i = _profile_config(profile, base=base, k=k)
    tables = build_schedule_receipts(profile, base=base_f, k=k_i, tau=float(tau), include_r1=True)
    data = _data_receipt(canonical, data_manifest)
    checkpoints = _checkpoint_receipt(checkpoint_root)
    memory = _memory_receipt()
    r2_names = r2_arm_names()
    r1_names = r1_arm_names()
    missing_r2 = sorted(set(r2_names) - set(tables))
    missing_r1 = sorted(set(r1_names) - set(tables))
    if missing_r2 or missing_r1:
        raise AssertionError(f"schedule matrix missing R2={missing_r2}, R1={missing_r1}")

    gates = [
        _gate("r0_profile_hash_and_finiteness", "PASS", blocking_for_training=True, detail=f"R0 JSON and m hash recorded: {profile.source_sha256}"),
        _gate("finite_endpoint_anchored_tables", "PASS", blocking_for_training=True, detail=f"{len(tables)} tables passed strict monotonicity/support assertions"),
        _gate("slow_frequency_content_retained", "PASS", blocking_for_training=True, detail="R1 tables retain positive slow-tail frequencies; no content dimension is deleted"),
        _gate("r1_matrix_locked", "PASS", blocking_for_training=True, detail=f"anchored-tail/mid-only controls x seeds {list(SEEDS)}; all rows are registered NOT_STARTED"),
        _gate("r2_matrix_locked", "PASS", blocking_for_training=True, detail=f"Geo/Cosh/3 lambda arms x seeds {list(SEEDS)}; canonical protocol reused"),
        _gate("data_manifest_hash", data["status"], blocking_for_training=True, detail="canonical 151.9M data manifest must be supplied and hash-match before training"),
        _gate("checkpoint_directory", checkpoints["status"], blocking_for_training=True, detail="checkpoint root must be supplied and empty before training"),
        _gate("gpu_memory_and_kernel_preflight", memory["status"], blocking_for_training=True, detail="explicit GPU preflight must verify >=30 GiB, BF16, and Flash-only attention"),
        _gate("training_not_started", "PASS", blocking_for_training=False, detail="this command only generated schedules and a manifest; no model/training call exists"),
    ]
    blocking = [
        gate
        for gate in gates
        if gate["blocking_for_training"] and gate["status"] != "PASS"
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "status": "DRY_RUN_BLOCKED" if blocking else "DRY_RUN_READY",
        "training_started": False,
        "training_authorized": False,
        "r0": {
            "path": profile.source_path,
            "sha256": profile.source_sha256,
            "m_path": profile.m_path,
            "delta_path": profile.delta_path,
            "delta_was_inferred": profile.delta_was_inferred,
            "delta_raw_range": [profile.raw_delta_min, profile.raw_delta_max],
            "m_count": profile.count,
            "m_sha256": profile.m_sha256,
            "profile_path": profile.profile_path,
            "coordinate_source_path": profile.coordinate_path,
            "metadata": profile.metadata,
        },
        "target_configuration": {
            "base": base_f,
            "K": k_i,
            "head_dim": 2 * k_i,
            "base_source": "explicit --base" if base is not None else "151M target default",
            "K_source": "explicit --K" if k is not None else "151M target default",
            "r0_measurement_not_inherited": True,
            "r0_source_base": profile.metadata.get("base"),
            "r0_source_head_dim": profile.metadata.get("head_dim"),
        },
        "schedule_definition": {
            "base": base_f,
            "K": k_i,
            "tau": float(tau),
            "lambda_values": [0.0, 0.1, 0.3],
            "density": "rho(lambda,delta) proportional to ((1-lambda)*m(delta)+lambda)^(1/3)",
            "direction": "delta_increasing_to_phi_increasing",
            "finite_quantiles": "endpoint-inclusive probabilities k/(K-1)",
            "support": {
                "phi": [0.0, 1.0],
                "omega": [1.0, base_f ** (-(k_i - 1) / k_i)],
                "formula": "omega_k=base^(-((K-1)/K)*phi_k)",
                "log_span_factor": (k_i - 1) / k_i,
            },
        },
        "canonical_151m_protocol": {**canonical, "protocol_fingerprint": protocol_fingerprint(canonical)},
        "r1_controls": {
            "arms": list(r1_names),
            "purpose": "anchored-tail versus mid-only; retain slow/content dimensions",
            "training_matrix": r1_training_matrix(),
        },
        "r2_training_matrix": r2_training_matrix(),
        "schedule_receipts": tables,
        "receipts": {
            "code_sha256": _hash_package_code(),
            "data": data,
            "checkpoints": checkpoints,
            "memory": memory,
            "frequency_hashes": {name: {"phi_sha256": receipt["phi_sha256"], "inv_freq_float32_sha256": receipt["inv_freq_float32_sha256"]} for name, receipt in tables.items()},
        },
        "stop_gates": gates,
        "no_training_policy": {
            "dry_run_only": False,
            "run_5090_train_mode": "double-gated; no training in manifest build",
            "required_before_any_future_run": ["explicit user authorization", "data manifest hash match", "empty checkpoint root", "frequency hashes frozen", "GPU memory/BF16/Flash-only preflight", "first finite training step and runtime receipt"],
        },
    }
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("dry-run", "show-matrix"))
    parser.add_argument("--r0-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("demand_companding_dry_run.json"))
    parser.add_argument("--base", type=float, default=None)
    parser.add_argument("--K", dest="k", type=int, default=None)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--data-manifest", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "show-matrix":
        print(json.dumps({"R1": list(r1_arm_names()), "R2": r2_training_matrix()}, indent=2))
        return 0
    manifest = build_manifest(args.r0_json, output=args.output, base=args.base, k=args.k, tau=args.tau, data_manifest=args.data_manifest, checkpoint_root=args.checkpoint_root)
    print(json.dumps({"status": manifest["status"], "output": str(Path(args.output).resolve()), "training_started": manifest["training_started"], "r2_rows": len(manifest["r2_training_matrix"]), "schedule_count": len(manifest["schedule_receipts"])}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
