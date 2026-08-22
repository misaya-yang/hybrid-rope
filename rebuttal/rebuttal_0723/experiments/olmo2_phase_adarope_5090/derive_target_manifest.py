"""CPU-only derivation of a same-sign mean/RMS moment-matched control."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


METHOD_ID = "olmo2_phase_adarope_target_manifest_derivation_v1"
PHASE_NAMES = ("phase_chord_olmo_r0_lambda_0p1", "phase_chord")
OUTPUT_NAME = "moment_matched_same_sign_control"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float32_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype="<f4")).tobytes()).hexdigest()


def canonical_content_hash(value: dict[str, Any]) -> str:
    content = copy.deepcopy(value)
    content.pop("canonical_content_sha256", None)
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_cpu() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", "-1"):
        raise RuntimeError("target-manifest derivation is CPU-only; CUDA_VISIBLE_DEVICES must be unset or -1")


def _table(record: Any, name: str) -> np.ndarray:
    if not isinstance(record, dict) or "inv_freq" not in record:
        raise ValueError(f"manifest record lacks inv_freq: {name}")
    values = np.asarray(record["inv_freq"], dtype="<f4")
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError(f"invalid frequency table: {name}")
    if not np.all(values[:-1] > values[1:]):
        raise ValueError(f"frequency table is not strictly decreasing: {name}")
    return values.copy()


def _source_tables(manifest: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    native = _table(manifest.get("native"), "native")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError("manifest lacks candidates")
    phase_name = next((name for name in PHASE_NAMES if name in candidates), None)
    if phase_name is None:
        raise ValueError("manifest lacks a phase-chord candidate")
    phase = _table(candidates[phase_name], phase_name)
    if phase.shape != native.shape:
        raise ValueError("native/phase pair count mismatch")
    if not np.array_equal(phase[[0, -1]], native[[0, -1]]):
        raise ValueError("phase candidate does not preserve Native endpoints")
    return native, phase, phase_name


def derive_control(native: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    native = np.asarray(native, dtype="<f4")
    phase = np.asarray(phase, dtype="<f4")
    if native.ndim != 1 or phase.shape != native.shape or native.size < 2:
        raise ValueError("native/phase tables must share a one-dimensional shape")
    if not np.isfinite(native).all() or not np.isfinite(phase).all() or not (native > 0).all() or not (phase > 0).all():
        raise ValueError("native/phase tables must be finite and positive")
    if not np.all(native[:-1] > native[1:]) or not np.all(phase[:-1] > phase[1:]):
        raise ValueError("native/phase tables must be strictly decreasing")
    if not np.array_equal(native[[0, -1]], phase[[0, -1]]):
        raise ValueError("phase table must preserve Native endpoints")
    native64 = native.astype(np.float64)
    phase64 = phase.astype(np.float64)
    phase_displacement = np.log(phase64) - np.log(native64)
    target_rms = float(np.sqrt(np.mean(np.square(phase_displacement))))
    phase_mean = float(np.mean(phase_displacement))
    if not np.isfinite(target_rms) or target_rms <= 0.0 or phase_mean == 0.0:
        raise ValueError("phase direction must have nonzero RMS and signed mean")

    u = np.linspace(0.0, 1.0, native.size, dtype=np.float64)
    basis = np.stack((u * (1.0 - u), u * (1.0 - u) * (2.0 * u - 1.0)), axis=1)
    mean_basis = basis.mean(axis=0)
    gram = (basis.T @ basis) / float(native.size)
    mean_norm_sq = float(mean_basis @ mean_basis)
    if mean_norm_sq <= 0.0:
        raise ValueError("moment basis has zero mean direction")
    # c(t)=c0+t*v preserves the target mean; solve the remaining quadratic
    # moment equation. The positive square-root branch is pre-registered.
    c0 = phase_mean * mean_basis / mean_norm_sq
    v = np.array([-mean_basis[1], mean_basis[0]], dtype=np.float64)
    qa = float(v @ gram @ v)
    qb = float(2.0 * (c0 @ gram @ v))
    qc = float(c0 @ gram @ c0 - target_rms * target_rms)
    discriminant = qb * qb - 4.0 * qa * qc
    if not np.isfinite(discriminant) or discriminant < -1e-12:
        raise ValueError("moment-matched control has no real coefficient solution")
    root = np.sqrt(max(0.0, discriminant))
    t = (-qb + root) / (2.0 * qa)
    coefficients = c0 + t * v
    displacement = basis @ coefficients
    values64 = native64 * np.exp(displacement)
    values64[[0, -1]] = native64[[0, -1]]
    values = values64.astype("<f4")
    values[[0, -1]] = native[[0, -1]]
    displacement = np.log(values.astype(np.float64)) - np.log(native.astype(np.float64))
    realized_rms = float(np.sqrt(np.mean(np.square(displacement))))
    realized_mean = float(np.mean(displacement))
    if not np.array_equal(values[[0, -1]], native[[0, -1]]):
        raise ValueError("control endpoints are not bitwise Native")
    if not np.all(values[:-1] > values[1:]):
        raise ValueError("control table is not strictly decreasing")
    if realized_mean * phase_mean <= 0.0:
        raise ValueError("control and phase signed means disagree")
    mean_abs_error = abs(realized_mean - phase_mean)
    if mean_abs_error > 1e-7:
        raise ValueError(f"mean matching error exceeds 1e-7: {mean_abs_error:.3e}")
    if abs(realized_rms - target_rms) > 1e-7:
        raise ValueError(f"RMS matching error exceeds 1e-7: {abs(realized_rms - target_rms):.3e}")
    return values, {
        "basis": ["u*(1-u)", "u*(1-u)*(2u-1)"],
        "coefficients": coefficients.tolist(),
        "root_branch": "positive_sqrt",
        "quadratic_discriminant": float(discriminant),
        "target_rms_log_displacement": target_rms,
        "realized_rms_log_displacement": realized_rms,
        "rms_abs_error": abs(realized_rms - target_rms),
        "phase_mean_log_displacement": phase_mean,
        "control_mean_log_displacement": realized_mean,
        "mean_abs_error": mean_abs_error,
        "same_signed_mean": True,
        "endpoint_bitwise_native": True,
        "strictly_decreasing": True,
        "inv_freq_float32_sha256": float32_sha256(values),
        "log_displacement": {
            "mean": realized_mean,
            "rms": realized_rms,
            "l1": float(np.sum(np.abs(displacement))),
            "max_abs": float(np.max(np.abs(displacement))),
        },
        "minimum_adjacent_log_gap": float(np.min(-np.diff(np.log(values.astype(np.float64))))),
    }


def derive_manifest(parent_manifest: Path, output: Path, expected_parent_sha: str) -> dict[str, Any]:
    _require_cpu()
    parent_manifest = parent_manifest.resolve()
    output = output.resolve()
    if parent_manifest == output:
        raise ValueError("output must differ from parent manifest")
    actual_parent_sha = sha256_file(parent_manifest)
    if actual_parent_sha != expected_parent_sha:
        raise ValueError("parent manifest SHA mismatch")
    manifest = json.loads(parent_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("parent manifest must be a JSON object")
    native, phase, phase_name = _source_tables(manifest)
    values, validation = derive_control(native, phase)
    derived = copy.deepcopy(manifest)
    parent_content_sha = derived.pop("content_sha256", None)
    candidates = derived.setdefault("candidates", {})
    if not isinstance(candidates, dict) or OUTPUT_NAME in candidates:
        raise ValueError("parent manifest candidates are malformed or already derived")
    candidates[OUTPUT_NAME] = {
        "inv_freq": values.tolist(),
        **validation,
        "derivation": {
            "method_id": METHOD_ID,
            "source_phase_candidate": phase_name,
            "family": "endpoint-zero-two-basis-log-displacement",
            "matching": "mean and RMS log-frequency displacement with same signed mean",
            **validation,
        },
    }
    protocol = derived.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("parent manifest lacks protocol metadata")
    candidate_order = protocol.get("candidate_order")
    if not isinstance(candidate_order, list) or any(not isinstance(name, str) for name in candidate_order):
        raise ValueError("protocol candidate_order is malformed")
    if OUTPUT_NAME in candidate_order:
        raise ValueError("protocol candidate_order already contains derived candidate")
    roles_key = "candidate_roles" if "candidate_roles" in protocol else "roles"
    candidate_roles = protocol.get(roles_key)
    if not isinstance(candidate_roles, dict):
        raise ValueError("protocol candidate roles are malformed")
    if OUTPUT_NAME in candidate_roles:
        raise ValueError("protocol candidate roles already contain derived candidate")
    candidate_order.append(OUTPUT_NAME)
    candidate_roles[OUTPUT_NAME] = (
        "endpoint-zero two-basis log-displacement control matched on f32-realized "
        "mean and RMS with the same signed mean as the phase candidate"
    )
    derived["derivation_receipt"] = {
        "method_id": METHOD_ID,
        "parent_manifest_sha256": actual_parent_sha,
        "source_phase_candidate": phase_name,
        "output_candidate": OUTPUT_NAME,
        "cuda_used": False,
        "parent_content_sha256": parent_content_sha,
        **validation,
    }
    derived["canonical_content_sha256"] = canonical_content_hash(derived)
    if output.exists() or output.with_name(output.name + ".incomplete").exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name + ".incomplete")
    temporary.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return derived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-parent-sha", required=True)
    args = parser.parse_args()
    result = derive_manifest(args.parent_manifest, args.output, args.expected_parent_sha)
    print(json.dumps({"status": "DERIVED", "output": str(args.output.resolve()), "canonical_content_sha256": result["canonical_content_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
