#!/usr/bin/env python3
"""Focused CPU tests for demand-companding schedules and dry-run gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from .run_experiment import build_manifest
from .schedule import (
    LAMBDA_VALUES,
    anchored_tail_phi,
    build_schedule_receipts,
    demand_density,
    load_r0_profile,
    mid_only_phi,
    quantile_phi,
)


def _write_r0(tmp_path: Path, *, m: list[float] | None = None) -> Path:
    path = tmp_path / "r0.json"
    path.write_text(json.dumps({"delta": [0.0, 0.25, 0.5, 0.75, 1.0], "m": m or [0.8, 0.95, 1.1, 1.35, 1.6], "base": 500_000, "K": 32}) + "\n", encoding="utf-8")
    return path


def test_lambda_formula_is_exact_and_positive(tmp_path: Path) -> None:
    profile = load_r0_profile(_write_r0(tmp_path))
    for lambda_value in LAMBDA_VALUES:
        expected = np.cbrt((1.0 - lambda_value) * profile.m + lambda_value)
        expected /= np.sum(0.5 * (expected[:-1] + expected[1:]) * np.diff(profile.delta))
        actual = demand_density(profile, lambda_value)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
        assert np.all(actual > 0.0)


def test_quantiles_keep_delta_to_phi_direction_and_endpoints(tmp_path: Path) -> None:
    profile = load_r0_profile(_write_r0(tmp_path))
    phi = quantile_phi(profile.delta, demand_density(profile, 0.3), 32)
    assert phi[0] == 0.0
    assert phi[-1] == 1.0
    assert np.all(np.diff(phi) > 0.0)
    assert profile.delta[0] < profile.delta[-1]


def test_geo_cosh_and_demand_tables_share_anchored_support(tmp_path: Path) -> None:
    profile = load_r0_profile(_write_r0(tmp_path))
    receipts = build_schedule_receipts(profile, base=500_000, k=32, tau=4.0)
    expected_names = {"geo", "cosh_tau4", "demand_lambda_0", "demand_lambda_0p1", "demand_lambda_0p3"}
    assert expected_names.issubset(receipts)
    for receipt in receipts.values():
        assert receipt["assertions"]["strict_phi_increasing"]
        assert receipt["assertions"]["strict_omega_decreasing"]
        assert receipt["assertions"]["endpoint_anchored"]
        assert receipt["assertions"]["support_anchored"]
        assert receipt["assertions"]["all_frequency_values_positive"]
        assert receipt["support"]["phi_min"] == 0.0
        assert receipt["support"]["phi_max"] == 1.0
        assert receipt["support"]["omega_max"] == 1.0
        assert receipt["support"]["omega_min"] == pytest.approx(
            500_000 ** (-(32 - 1) / 32), abs=1e-12
        )
        assert len(receipt["inv_freq"]) == 32


def test_r1_controls_retain_slow_tail_and_do_not_delete_content(tmp_path: Path) -> None:
    profile = load_r0_profile(_write_r0(tmp_path))
    receipts = build_schedule_receipts(profile, base=500_000, k=32, tau=4.0)
    geo = np.asarray(receipts["geo"]["phi"], dtype=np.float64)
    for lambda_label in ("0", "0p1", "0p3"):
        target = np.asarray(receipts[f"demand_lambda_{lambda_label}"]["phi"], dtype=np.float64)
        tail, tail_meta = anchored_tail_phi(target)
        mid, mid_meta = mid_only_phi(target)
        tail_start = int(tail_meta["retained_slow_tail_start_index"])
        mid_start = int(mid_meta["retained_slow_tail_start_index"])
        np.testing.assert_array_equal(tail[tail_start:], geo[tail_start:])
        np.testing.assert_array_equal(mid[mid_start:], geo[mid_start:])
        assert tail_meta["content_dimensions_removed"] == 0
        assert mid_meta["content_dimensions_removed"] == 0
        assert np.all(np.diff(tail) > 0.0)
        assert np.all(np.diff(mid) > 0.0)


def test_manifest_locks_r2_matrix_and_cpu_only_status(tmp_path: Path) -> None:
    manifest = build_manifest(_write_r0(tmp_path), output=tmp_path / "manifest.json")
    assert manifest["training_started"] is False
    assert manifest["training_authorized"] is False
    assert manifest["status"] in {"DRY_RUN_READY", "DRY_RUN_BLOCKED"}
    assert len(manifest["r2_training_matrix"]) == 15
    assert {row["seed"] for row in manifest["r2_training_matrix"]} == {42, 137, 256}
    assert {row["arm"] for row in manifest["r2_training_matrix"]} == {"geo", "cosh_tau4", "demand_lambda_0", "demand_lambda_0p1", "demand_lambda_0p3"}
    assert manifest["receipts"]["memory"]["training_gpu_probe_started"] is False
    assert manifest["receipts"]["checkpoints"]["status"] == "PENDING_CHECKPOINT_ROOT_PATH"
    assert manifest["stop_gates"][-1]["name"] == "training_not_started"


def test_manifest_records_data_mismatch_and_empty_checkpoint_gate(tmp_path: Path) -> None:
    data = tmp_path / "data_manifest.json"
    data.write_text("{}\n", encoding="utf-8")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    manifest = build_manifest(_write_r0(tmp_path), output=tmp_path / "manifest.json", data_manifest=str(data), checkpoint_root=str(checkpoints))
    gates = {gate["name"]: gate for gate in manifest["stop_gates"]}
    assert gates["data_manifest_hash"]["status"] == "STOP_DATA_HASH_MISMATCH"
    assert gates["checkpoint_directory"]["status"] == "PASS_EMPTY_BEFORE_TRAINING"


def test_nonpositive_r0_m_is_rejected(tmp_path: Path) -> None:
    path = _write_r0(tmp_path, m=[0.8, 0.0, 1.1, 1.3, 1.6])
    with pytest.raises(ValueError, match="strictly positive"):
        load_r0_profile(path)


def test_nested_r0_case_json_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "r0_nested.json"
    path.write_text(json.dumps({"cases": [{"x": [0.0, 0.5, 1.0], "m": [0.8, 1.0, 1.2]}]}) + "\n", encoding="utf-8")
    profile = load_r0_profile(path)
    assert profile.m_path == "cases[0].m"
    assert profile.delta_path == "x"
    assert np.allclose(profile.delta, [0.0, 0.5, 1.0])
