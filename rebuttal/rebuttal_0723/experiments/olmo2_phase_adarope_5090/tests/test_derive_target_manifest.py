from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.derive_target_manifest import (
    OUTPUT_NAME,
    canonical_content_hash,
    derive_control,
    derive_manifest,
)


def _parent(tmp_path: Path) -> Path:
    native = np.geomspace(1.0, 1e-3, 64).astype("<f4")
    phase = native.copy()
    phase[1:-1] = np.exp(np.linspace(np.log(native[1] * 0.9), np.log(native[-2] * 1.1), 62)).astype("<f4")
    # Ensure the phase direction is strictly decreasing and nonzero.
    phase = np.minimum.accumulate(phase)
    phase[0], phase[-1] = native[0], native[-1]
    manifest = {
        "schema_version": 1,
        "native": {"inv_freq": native.tolist()},
        "candidates": {"phase_chord_olmo_r0_lambda_0p1": {"inv_freq": phase.tolist()}},
        "protocol": {
            "candidate_order": ["phase_chord_olmo_r0_lambda_0p1"],
            "candidate_roles": {
                "phase_chord_olmo_r0_lambda_0p1": "phase candidate"
            },
        },
        "provenance": {"source": "frozen-test-parent", "files": {"asset.bin": "abc"}},
        "content_sha256": "parent-content-digest",
    }
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value.astype("<f4")).tobytes()).hexdigest()


def test_control_matches_signed_rms_and_preserves_provenance(tmp_path: Path) -> None:
    parent = _parent(tmp_path)
    output = tmp_path / "derived.json"
    result = derive_manifest(parent, output, _sha(parent))
    assert OUTPUT_NAME in result["candidates"]
    control = np.asarray(result["candidates"][OUTPUT_NAME]["inv_freq"], dtype="<f4")
    native = np.asarray(result["native"]["inv_freq"], dtype="<f4")
    phase = np.asarray(result["candidates"]["phase_chord_olmo_r0_lambda_0p1"]["inv_freq"], dtype="<f4")
    assert np.array_equal(control[[0, -1]], native[[0, -1]])
    assert np.all(control[:-1] > control[1:])
    phase_mean = np.mean(np.log(phase.astype(np.float64)) - np.log(native.astype(np.float64)))
    control_mean = np.mean(np.log(control.astype(np.float64)) - np.log(native.astype(np.float64)))
    assert phase_mean * control_mean > 0
    phase_rms = np.sqrt(np.mean((np.log(phase.astype(np.float64)) - np.log(native.astype(np.float64))) ** 2))
    control_rms = np.sqrt(np.mean((np.log(control.astype(np.float64)) - np.log(native.astype(np.float64))) ** 2))
    assert abs(phase_mean - control_mean) <= 1e-7
    assert abs(phase_rms - control_rms) <= 1e-7
    assert result["candidates"][OUTPUT_NAME]["mean_abs_error"] == abs(phase_mean - control_mean)
    assert result["provenance"] == {"source": "frozen-test-parent", "files": {"asset.bin": "abc"}}
    assert "content_sha256" not in result
    assert result["derivation_receipt"]["parent_content_sha256"] == "parent-content-digest"
    assert result["candidates"][OUTPUT_NAME]["inv_freq_float32_sha256"] == _sha_bytes(control)
    assert result["protocol"]["candidate_order"] == [
        "phase_chord_olmo_r0_lambda_0p1",
        OUTPUT_NAME,
    ]
    assert OUTPUT_NAME in result["protocol"]["candidate_roles"]
    assert result["canonical_content_sha256"] == canonical_content_hash(result)
    assert output.is_file() and not output.with_name(output.name + ".incomplete").exists()


def test_parent_sha_tamper_and_cuda_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = _parent(tmp_path)
    with pytest.raises(ValueError, match="SHA"):
        derive_manifest(parent, tmp_path / "bad.json", "0" * 64)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="CPU-only"):
        derive_manifest(parent, tmp_path / "cuda.json", _sha(parent))


def test_control_builder_rejects_wrong_sign_or_malformed_tables() -> None:
    native = np.geomspace(1.0, 1e-3, 64).astype("<f4")
    with pytest.raises(ValueError):
        derive_control(native, native.copy())
    bad = native.copy()
    bad[10] = bad[9] * 2
    with pytest.raises(ValueError):
        derive_control(native, bad)
