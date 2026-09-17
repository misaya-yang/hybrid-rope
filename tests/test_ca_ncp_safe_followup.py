from __future__ import annotations

import numpy as np

from experiments.ca_ncp_safe_followup_20260917.build_alignments import angle_capped, axis_consensus


def source_alignment() -> dict[str, np.ndarray]:
    layers, groups, size, carrier = 1, 2, 4, 1
    a = np.asarray([[0.0, 0.8]])
    b = np.sqrt(1.0 - a * a)
    v = np.zeros((layers, groups, size), dtype=np.complex128)
    v[0, 0, 2] = 1
    v[0, 1, 3] = 1
    e = np.eye(size, dtype=np.complex128)[:, carrier]
    u = a[..., None] * e + b[..., None] * v
    return {
        "a": a, "b": b, "v_real": v.real, "v_imag": v.imag,
        "u_real": u.real, "u_imag": u.imag,
        "active_indices": np.arange(10, 14), "carrier_local": np.asarray(carrier),
    }


def test_angle_cap_is_exact_and_never_expands_plane():
    source = source_alignment()
    cap = 0.1
    arrays, summary = angle_capped(source, cap)
    angles = np.arctan2(arrays["b"], arrays["a"])
    np.testing.assert_allclose(angles, np.minimum(np.arctan2(source["b"], source["a"]), cap))
    assert summary["angle_cap_radians"] == cap
    assert summary["full_strength_planes"] == 0


def test_axis_consensus_uses_identity_on_source_disagreement(monkeypatch):
    source = source_alignment()
    pg = np.zeros((1, 2, 4, 4), dtype=np.complex128)
    pp = np.zeros_like(pg)
    pg[0, 0, 2, 2] = pp[0, 0, 2, 2] = 3
    pg[0, 1, 2, 2] = 3
    pp[0, 1, 3, 3] = 3

    def fake(_receipt, _root, name):
        return pg if name == "pg19" else pp

    monkeypatch.setattr(
        "experiments.ca_ncp_safe_followup_20260917.build_alignments.source_moments", fake,
    )
    arrays, summary = axis_consensus(source, {"documents": []}, None)
    assert not arrays["identity"][0, 0]
    assert arrays["identity"][0, 1]
    assert summary["source_consensus_planes"] == 1
