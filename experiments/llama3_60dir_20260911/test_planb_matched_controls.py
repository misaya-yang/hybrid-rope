"""CPU-only tests for Plan B matched controls.

Run from this directory with ``python -m pytest test_planb_matched_controls.py``.
No model, data panel, or CUDA device is required.
"""

import numpy as np
import pytest

import operators as O
import planb_matched_controls as C


def test_registry_is_complete_and_has_plan_b_parent_controls():
    assert len(C.REGISTRY) == 60
    assert set(C.REGISTRY) == {f"D{d:02d}{p}" for d in range(1, 21) for p in "abc"}
    assert C.REGISTRY["D01a"]["parent_ids"] == ["MR", "ResonanceYaRN"]
    assert "D03_GAUGE" in C.REGISTRY["D03a"]["required_controls"]
    assert "DROP" in C.REGISTRY["D05c"]["required_controls"]
    assert C.REGISTRY["D07a"]["construction_status"] == "BLOCKED_UNTIL_MDEV_ACTIVATIONS"
    assert C.REGISTRY["D17a"]["parent_ids"] == ["D16b"]


def test_d03_gauge_is_a_score_level_noop():
    g = O.Geometry.from_native()
    gauge = C._GaugeControl(g, "D03a")
    rng = np.random.default_rng(3)
    q = rng.normal(size=(24, g.head_dim))
    k = rng.normal(size=(24, g.head_dim))
    pos = np.arange(24, dtype=np.float64)
    parent, gauged = gauge.score(q, k, pos, pos)
    np.testing.assert_allclose(parent, gauged, atol=1e-10, rtol=1e-10)


def test_d05_drop_and_sign_are_distinct_controls():
    g = O.Geometry.from_native()
    dc = C.build_dc(g, "D05c")
    drop = C._DropControl(g, "D05c")
    sign = C.build_sign(g, "D04b")
    assert np.all(dc.q_amp(np.array([0.0])) > 0)
    assert np.all(drop.q_amp(np.array([0.0]))[0, drop.mask] == 0)
    assert np.allclose(np.abs(sign.nu()), g.nu_mrpro, atol=0, rtol=0)
    assert np.any(sign.nu() < 0)


def test_d06_area_matches_each_arm_without_changing_outside_t():
    g = O.Geometry.from_native()
    t = C._T(g)
    outside = np.setdiff1d(np.arange(g.K), t)
    for cid in ("D06a", "D06b", "D06c"):
        arm = O.build(cid, g)
        area = C._AreaControl(g, cid)
        m_arm = np.log(g.omega / arm.nu()) / np.log(g.scale)
        m_area = np.log(g.omega / area.nu()) / np.log(g.scale)
        np.testing.assert_allclose(m_area[outside], g.m_mrpro[outside], atol=1e-10)
        np.testing.assert_allclose(m_area.sum(), m_arm.sum(), atol=1e-10)


def test_d13_static_endpoint_matches_schedule():
    g = O.Geometry.from_native()
    p = np.array([float(g.target - 1)])
    for cid in ("D13a", "D13b", "D13c"):
        expected = O.build(cid, g)
        endpoint = C.build_static_endpoint(g, cid)
        want = expected.q_amp(p)[0, 0] * expected.k_amp(p)[0, 0]
        got = endpoint.q_amp(p)[0, 0] * endpoint.k_amp(p)[0, 0]
        assert abs(float(want - got)) < 1e-12


def test_d07_matched_global_refuses_zero_variance():
    g = O.Geometry.from_native()
    q = np.zeros((8, g.head_dim))
    k = np.zeros((8, g.head_dim))
    p = np.arange(8, dtype=np.float64)
    with pytest.raises(ValueError, match="BLOCKED"):
        C.build_matched_global(g, q, k, p, p)


def test_d07_matched_global_uses_same_gain_on_q_and_k():
    g = O.Geometry.from_native()
    rng = np.random.default_rng(4)
    q = rng.normal(size=(16, g.head_dim))
    k = rng.normal(size=(16, g.head_dim))
    p = np.arange(16, dtype=np.float64)
    op, meta = C.build_matched_global(g, q, k, p, p)
    np.testing.assert_allclose(op.q_amp(p), meta["gain_qk"], atol=1e-12)
    np.testing.assert_allclose(op.k_amp(p), meta["gain_qk"], atol=1e-12)


def test_selftest_all_passes():
    report = C.selftest()
    assert report["all_pass"]
    assert report["n_pass"] == report["n_checks"]
