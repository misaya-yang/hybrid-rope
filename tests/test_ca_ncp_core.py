from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.ca_ncp_native_20260917.core import (
    apply_group_planes_torch,
    apply_plane,
    build_carrier_table,
    choose_direction,
    complex_to_split_half,
    minimal_plane,
    sample_causal_pairs,
    signed_moment,
    split_half_to_complex,
    tensor_sha256,
    validate_table,
)
from experiments.native_contrastive_proximal_20260915.tables import build_ncp_arrays
from scripts.experiments.cross_audit.tables import native_table


def olmo_geometry():
    native = native_table(128, 500_000).astype(np.float32)
    ncp = build_ncp_arrays(native, native_length=4096)["candidate"]
    return native, ncp, build_carrier_table(native, ncp, 4096)


def random_plane(size=7, carrier=2, seed=11):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=size) + 1j * rng.normal(size=size)
    return minimal_plane(u, carrier)


def test_olmo_carrier_geometry_exact():
    native, ncp, result = olmo_geometry()
    assert result["active_indices"].tolist() == list(range(32, 63))
    assert result["carrier_slot"] == 35
    assert result["carrier_local"] == 3
    assert result["ncp_table_sha256_float32"] == "54b9dd1f73aafc69f7bb5ed1b7b49d49128002371cb378d03ca1fd1d108e0cb7"
    assert result["carrier_table_sha256_float32"] == "10547fcaf8d8d6f0bc3935a839f0bd81734e0fa18855603cda3686c351aa06c7"
    assert result["carrier_table"][0] == native[0]
    assert result["carrier_table"][-1] == native[-1]
    assert result["carrier_table"][35] == np.float32(2 * math.pi / (2.205 * 4096))


def test_validate_table_rejects_wrong_dtype_and_order():
    with pytest.raises(ValueError):
        validate_table(np.asarray([3.0, 2.0, 1.0, 0.5], dtype=np.float64))
    with pytest.raises(ValueError):
        validate_table(np.asarray([3.0, 2.0, 2.0, 0.5], dtype=np.float32))


def test_tensor_hash_is_float32_canonical():
    values = np.asarray([1, 0.5, 0.25, 0.125], dtype=np.float32)
    assert tensor_sha256(values) == tensor_sha256(values.astype("<f4"))


def test_split_half_round_trip():
    values = np.arange(24, dtype=np.float64).reshape(3, 8)
    complex_values = split_half_to_complex(values)
    np.testing.assert_array_equal(complex_to_split_half(complex_values), values)


def test_split_half_rejects_odd_dimension():
    with pytest.raises(ValueError):
        split_half_to_complex(np.ones((2, 7)))


def test_signed_moment_matches_scalar_objective_and_negative_phase_sign():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(17, 5)) + 1j * rng.normal(size=(17, 5))
    k = rng.normal(size=(17, 5)) + 1j * rng.normal(size=(17, 5))
    lag = rng.integers(1, 200, size=17)
    wc, scale = 0.0012, 1 / math.sqrt(128)
    matrix = signed_moment(q, k, lag, wc, scale=scale)
    u = rng.normal(size=5) + 1j * rng.normal(size=5)
    u /= np.linalg.norm(u)
    gamma = np.expm1(-1j * lag * wc)
    delta = scale * np.real(np.conj(q @ u.conj()) * (k @ u.conj()) * gamma).mean()
    assert np.vdot(u, matrix @ u).real == pytest.approx(-delta, abs=1e-12)


def test_signed_moment_is_hermitian():
    q = np.asarray([[1 + 2j, 3 - 1j], [2 - 1j, 1 + 4j]])
    k = np.asarray([[2 + 1j, 1 - 3j], [4 + 2j, -1 + 1j]])
    matrix = signed_moment(q, k, np.asarray([3, 7]), 0.03)
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-14)


def test_signed_moment_weights_are_normalized():
    q = np.asarray([[1 + 0j, 0], [0, 1 + 0j]])
    k = q.copy()
    lag = np.asarray([1, 2])
    a = signed_moment(q, k, lag, 0.1, weights=[1, 3])
    b = signed_moment(q, k, lag, 0.1, weights=[2, 6])
    np.testing.assert_allclose(a, b)


def test_choose_direction_simple_top_eigenvector():
    matrix = np.diag([1.0, 4.0, 2.0]).astype(np.complex128)
    u, receipt = choose_direction(matrix, 1)
    np.testing.assert_allclose(u, [0, 1, 0], atol=1e-14)
    assert receipt["objective"] == pytest.approx(4.0)
    assert not receipt["identity"]


def test_choose_direction_nonpositive_returns_identity():
    u, receipt = choose_direction(-np.eye(4), 2)
    np.testing.assert_array_equal(u, [0, 0, 1, 0])
    assert receipt["identity"]


def test_choose_direction_degenerate_prefers_carrier_projection():
    matrix = np.diag([3.0, 3.0, 1.0]).astype(np.complex128)
    u, receipt = choose_direction(matrix, 1)
    np.testing.assert_allclose(u, [0, 1, 0], atol=1e-14)
    assert receipt["top_cluster_dimension"] == 2


def test_minimal_plane_unitary_and_maps_u_to_carrier():
    plane = random_plane()
    dense = plane.dense()
    e = np.eye(plane.u.size)[:, plane.carrier_local]
    np.testing.assert_allclose(dense.conj().T @ dense, np.eye(plane.u.size), atol=1e-12)
    np.testing.assert_allclose(dense @ plane.u, e, atol=1e-12)
    np.testing.assert_allclose(e.conj() @ dense, plane.u.conj(), atol=1e-12)


def test_minimal_plane_rank_and_frobenius_formula():
    plane = random_plane()
    delta = plane.dense() - np.eye(plane.u.size)
    assert np.linalg.matrix_rank(delta, tol=1e-11) <= 2
    assert np.linalg.norm(delta, "fro") ** 2 == pytest.approx(4 * (1 - plane.a), abs=1e-11)
    assert np.linalg.norm(delta, 2) == pytest.approx(math.sqrt(2 * (1 - plane.a)), abs=1e-11)


def test_apply_plane_matches_dense():
    rng = np.random.default_rng(9)
    plane = random_plane()
    values = rng.normal(size=(4, plane.u.size)) + 1j * rng.normal(size=(4, plane.u.size))
    np.testing.assert_allclose(apply_plane(values, plane), values @ plane.dense().T, atol=1e-12)


def test_plane_preserves_norm_and_inner_product():
    rng = np.random.default_rng(91)
    plane = random_plane()
    q = rng.normal(size=plane.u.size) + 1j * rng.normal(size=plane.u.size)
    k = rng.normal(size=plane.u.size) + 1j * rng.normal(size=plane.u.size)
    uq, uk = apply_plane(q, plane), apply_plane(k, plane)
    assert np.linalg.norm(uq) == pytest.approx(np.linalg.norm(q), abs=1e-12)
    assert np.vdot(uq, uk) == pytest.approx(np.vdot(q, k), abs=1e-12)


def test_frequency_similarity_spectrum_is_preserved():
    plane = random_plane()
    frequencies = np.linspace(0.7, 0.1, plane.u.size)
    diagonal = np.diag(np.exp(-1j * 37 * frequencies))
    transformed = plane.dense().conj().T @ diagonal @ plane.dense()
    np.testing.assert_allclose(
        np.sort_complex(np.linalg.eigvals(transformed)),
        np.sort_complex(np.diag(diagonal)), atol=1e-11,
    )


def test_torch_grouped_map_matches_numpy_and_preserves_inactive_bits():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(17)
    active = np.asarray([1, 3, 4, 6])
    carrier = 1
    planes = [random_plane(len(active), carrier, seed=4), random_plane(len(active), carrier, seed=8)]
    x = torch.tensor(rng.normal(size=(2, 5, 4, 16)), dtype=torch.float32)
    original = x.clone()
    mapped = apply_group_planes_torch(
        x, active, carrier,
        [p.a for p in planes], [p.b for p in planes],
        np.stack([p.v.real for p in planes]), np.stack([p.v.imag for p in planes]),
        [0, 0, 1, 1],
    )
    inactive = sorted(set(range(8)) - set(active))
    for head, group in enumerate([0, 0, 1, 1]):
        z = split_half_to_complex(original[:, :, head].numpy())
        expected = z.copy()
        expected[..., active] = apply_plane(z[..., active], planes[group])
        np.testing.assert_allclose(split_half_to_complex(mapped[:, :, head].numpy()), expected, atol=2e-6)
    assert torch.equal(mapped[..., inactive], original[..., inactive])
    assert torch.equal(mapped[..., [i + 8 for i in inactive]], original[..., [i + 8 for i in inactive]])


def test_torch_identity_is_same_object_and_bit_exact():
    torch = pytest.importorskip("torch")
    x = torch.randn(2, 3, 2, 12, dtype=torch.bfloat16)
    y = apply_group_planes_torch(
        x, [1, 2, 4], 1, [1, 1], [0, 0], np.zeros((2, 3)), np.zeros((2, 3)), [0, 1],
    )
    assert y is x
    assert torch.equal(y, x)


def test_prefill_decode_chunking_is_stateless():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(29)
    plane = random_plane(4, 1)
    x = torch.tensor(rng.normal(size=(1, 13, 1, 12)), dtype=torch.float32)
    kwargs = ([1, 2, 4, 5], 1, [plane.a], [plane.b], [plane.v.real], [plane.v.imag], [0])
    whole = apply_group_planes_torch(x, *kwargs)
    chunks = torch.cat([apply_group_planes_torch(x[:, :7], *kwargs), apply_group_planes_torch(x[:, 7:], *kwargs)], dim=1)
    torch.testing.assert_close(whole, chunks)


def test_gqa_group_average_matrix_linearity():
    rng = np.random.default_rng(31)
    q = rng.normal(size=(3, 19, 4)) + 1j * rng.normal(size=(3, 19, 4))
    k = rng.normal(size=(19, 4)) + 1j * rng.normal(size=(19, 4))
    lag = rng.integers(1, 200, size=19)
    mean_first = signed_moment(q.mean(axis=0), k, lag, 0.02)
    average_matrices = np.mean([signed_moment(head, k, lag, 0.02) for head in q], axis=0)
    np.testing.assert_allclose(mean_first, average_matrices, atol=1e-13)


def test_sample_causal_pairs_deterministic_and_causal():
    valid = np.arange(1, 4096)
    q1, k1 = sample_causal_pairs(valid, 512, 123)
    q2, k2 = sample_causal_pairs(valid, 512, 123)
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_array_equal(k1, k2)
    assert np.all(q1 > k1)


def test_local_displacement_bound_holds():
    plane = random_plane(6, 2)
    frequencies = np.linspace(0.004, 0.0002, 6)
    lag = 91
    diagonal = np.diag(np.exp(-1j * lag * frequencies))
    actual = np.linalg.norm(plane.dense().conj().T @ diagonal @ plane.dense() - diagonal, 2)
    spread = np.max(np.abs(np.exp(-1j * lag * frequencies) - np.exp(-1j * lag * frequencies[2])))
    bound = min(2.0, 2 * np.linalg.norm(plane.dense() - np.eye(6), 2) * spread)
    assert actual <= bound + 1e-12
