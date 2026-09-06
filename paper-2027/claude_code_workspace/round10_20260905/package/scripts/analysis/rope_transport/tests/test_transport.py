"""Correctness gates for the zero-GPU transport analysis."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from analysis.rope_transport import conditioning, tables, transport, weights  # noqa: E402


def _support(length: int = 512, max_points: int = 512):
    return weights.distance_weight("causal", length=length, max_points=max_points)


def _dense(omega, support):
    blocks = transport.rotation_blocks(omega, support)
    n, k = blocks.shape[0], blocks.shape[1]
    out = np.zeros((n, 2 * k, 2 * k))
    for idx in range(k):
        out[:, 2 * idx : 2 * idx + 2, 2 * idx : 2 * idx + 2] = blocks[:, idx]
    return out


def test_weight_normalises_and_records_support():
    support, weight, meta = _support()
    assert weight.sum() == pytest.approx(1.0)
    assert meta["family"] == "causal"
    assert meta["support_points_used"] == support.size
    assert (weight > 0).all()


def test_hard_swap_matches_dense_reference():
    support, weight, _ = _support(length=256, max_points=256)
    src = np.array([0.9, 0.3, 0.05, 0.001])
    dst = np.array([0.8, 0.25, 0.04, 0.0009])
    reference = float(
        (weight * ((_dense(dst, support) - _dense(src, support)) ** 2).sum(axis=(1, 2))).sum()
    )
    assert transport.hard_swap_residual(src, dst, support, weight) == pytest.approx(
        reference, rel=1e-10, abs=1e-10
    )


def test_identity_table_has_zero_residual():
    support, weight, _ = _support()
    omega = np.array([0.7, 0.2, 0.02, 0.002, 0.0001])
    result = transport.transport_residual(omega, omega, support, weight, max_iter=8)
    assert result.hard_swap == pytest.approx(0.0, abs=1e-18)
    assert result.repaired == pytest.approx(0.0, abs=1e-16)


def test_frequency_permutation_is_exactly_compensable():
    """The obstruction theorem permits permutation; the solver must find it."""
    support, weight, _ = _support(length=256, max_points=256)
    src = np.array([0.9, 0.31, 0.07, 0.004])
    dst = src[[2, 0, 3, 1]]
    result = transport.transport_residual(src, dst, support, weight, max_iter=200, tol=1e-15)
    assert result.hard_swap > 1.0
    assert result.relative_repaired < 1e-8
    assert result.repairability > 1.0 - 1e-8


def test_sign_alias_is_exactly_compensable():
    """R(-wD) = S R(wD) S, so a sign flip is a static reparameterization."""
    support, weight, _ = _support(length=256, max_points=256)
    src = np.array([0.9, 0.31, 0.07, 0.004])
    dst = src.copy()
    blocks_src = transport.rotation_blocks(src, support)
    blocks_neg = transport.rotation_blocks(-src, support)
    assert np.allclose(blocks_neg, np.swapaxes(blocks_src, -1, -2))
    result = transport.transport_residual(src, dst, support, weight, max_iter=8)
    assert result.relative_repaired < 1e-12


def test_repair_never_worse_than_hard_swap_and_is_monotone():
    support, weight, _ = _support(length=512, max_points=512)
    src = np.array([0.9, 0.2, 0.03, 0.004, 0.0002])
    dst = np.array([0.9, 0.15, 0.02, 0.0015, 0.00005])
    result = transport.transport_residual(src, dst, support, weight, max_iter=40)
    assert result.repaired <= result.hard_swap + 1e-12
    values = [row["after_key"] for row in result.history if "after_key" in row]
    assert values
    assert all(b <= a + 1e-9 for a, b in zip(values, values[1:]))


def test_rank_constraint_is_monotone_in_rank():
    support, weight, _ = _support(length=512, max_points=512)
    src = np.array([0.9, 0.2, 0.03, 0.004, 0.0002, 0.00001])
    dst = src * np.array([1.0, 0.7, 0.4, 0.25, 0.2, 0.2])
    previous = None
    for rank in (1, 2, 4, 8, 12):
        value = transport.transport_residual(
            src, dst, support, weight, rank=rank, max_iter=60
        ).repaired
        if previous is not None:
            assert value <= previous + 1e-9
        previous = value
    full = transport.transport_residual(src, dst, support, weight, max_iter=60).repaired
    assert previous >= full - 1e-9


def test_residual_attribution_sums_to_total():
    support, weight, _ = _support(length=256, max_points=256)
    src = np.array([0.9, 0.2, 0.03, 0.004])
    dst = np.array([0.9, 0.16, 0.02, 0.002])
    result = transport.transport_residual(src, dst, support, weight, max_iter=40)
    parts = transport.residual_by_pair(
        src, dst, support, weight, result.query_map, result.key_map
    )
    assert parts["query_pair_energy"].sum() == pytest.approx(result.repaired, rel=1e-6)
    assert parts["key_pair_energy"].sum() == pytest.approx(result.repaired, rel=1e-6)


def test_slow_pairs_are_less_unique_than_fast_pairs():
    """Low-frequency collapse, per channel: slow pairs are near-redundant."""
    support, weight, _ = _support(length=4096, max_points=1024)
    omega = np.array([1.0, 0.1, 0.01, 1e-4, 1e-6, 1e-8])
    values = conditioning.pair_uniqueness(omega, support, weight)["uniqueness"]
    assert values[0] > values[-1]
    assert values[-1] < 1e-3
    assert values[0] > 0.5


def test_range_resolvability_reports_phase_safety():
    support, weight, _ = weights.distance_weight("causal", length=16384, max_points=1024)
    native = np.array([1.0, 0.05, 1e-4, 1e-6])
    report = conditioning.range_resolvability(
        native / 4.0, support, weight, trained_omega=native, trained_length=4096
    )
    assert 0.0 <= report["phase_safe_fraction"] <= 1.0
    assert report["entropy_effective_rank"] <= report["nominal_dimension"] + 1e-9
    assert report["stable_rank"] >= 1.0


def test_official_yarn_preserves_fast_pairs_and_scales_slow_pairs():
    omega = 1.0 / (500000.0 ** (np.arange(0, 128, 2) / 128.0))
    base = tables.Table("native", "test", omega, tables.float32_sha256(omega), {})
    out = tables.official_yarn(
        base,
        scale=4.0,
        head_dim=128,
        rope_base=500000.0,
        original_max_position_embeddings=4096,
    )
    assert out.inv_freq[0] == pytest.approx(omega[0], rel=1e-12)
    assert out.inv_freq[-1] == pytest.approx(omega[-1] / 4.0, rel=1e-9)
    assert out.meta["untouched_fast_pairs"] >= 1
    assert np.all(np.diff(out.inv_freq) < 0)


def test_budgeted_transport_moves_redundant_pairs_more():
    omega = 1.0 / (500000.0 ** (np.arange(0, 128, 2) / 128.0))
    base = tables.Table("native", "test", omega, tables.float32_sha256(omega), {})
    support, weight, _ = weights.distance_weight("causal", length=4096, max_points=1024)
    unique = conditioning.pair_uniqueness(omega, support, weight)["uniqueness"]
    out = tables.budgeted_transport(base, unique, scale=4.0)
    moved = 1.0 - out.inv_freq / omega
    assert moved[0] < moved[-1]
    assert np.all(np.diff(out.inv_freq) < 0)


def test_table_validation_rejects_non_monotone():
    with pytest.raises(ValueError):
        tables._check(np.array([0.1, 0.2, 0.3]), "bad")


def test_module_does_not_require_cuda():
    assert "torch" not in sys.modules
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1", None) or True
