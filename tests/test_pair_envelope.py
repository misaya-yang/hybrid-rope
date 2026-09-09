from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.native_sparse_position.pair_envelope import (
    build_pair_envelope, score_pair_envelope, build_quest, score_quest,
)


def exact_max(keys, q):
    keys = torch.as_tensor(keys).float().double()
    q = torch.as_tensor(q).float().double()
    repeats = q.shape[0] // keys.shape[0]
    expanded = keys.repeat_interleave(repeats, dim=0)
    return torch.einsum("hd,hbtd->hbt", q, expanded).amax(dim=-1)


def rotate(x, angles):
    x = torch.as_tensor(x).double()
    angles = torch.as_tensor(angles).double()
    K = angles.numel()
    a, b = x[..., :K], x[..., K:2*K]
    out = x.clone()
    out[..., :K] = a * angles.cos() - b * angles.sin()
    out[..., K:2*K] = a * angles.sin() + b * angles.cos()
    return out.float()


def test_constant_keys_zero_variance_gqa_and_zero_query():
    keys = torch.tensor([[[[1., -2., 3., 4.]]], [[[5., 6., -1., 2.]]]]).expand(2, 3, 64, 4)
    q = torch.tensor([[1., 2., 0., -1.], [0., 0., 0., 0.], [-1., 0., 2., 1.], [2., 1., 1., 0.]])
    cache = build_pair_envelope(keys, K=2)
    assert bool(cache.isotropic.all())
    assert torch.count_nonzero(cache.halfwidth) == 0
    scores, _ = score_pair_envelope(q, cache)
    truth = exact_max(keys, q)
    assert bool((scores.double() >= truth).all())
    torch.testing.assert_close(scores.double(), truth, atol=8e-4, rtol=8e-5)


def test_collinear_pair_extrema_are_exact_up_to_roundoff_padding():
    # Unequal multiplicities make this catch a symmetric-mean-box shortcut.
    t = torch.tensor([-3.] + [2.] * 63)
    keys = (torch.tensor([.3, -1.2]) + t[:, None] * torch.tensor([1., 2.]))[None, None]
    q = torch.tensor([[2., -1.], [1., 2.], [-1., -2.], [.7, -3.]])
    cache = build_pair_envelope(keys, K=1)
    assert not bool(cache.isotropic.any())
    scores, _ = score_pair_envelope(q, cache)
    truth = exact_max(keys, q)
    assert bool((scores.double() >= truth).all())
    torch.testing.assert_close(scores.double(), truth, atol=1e-3, rtol=1e-4)


def test_isotropic_disk_and_axis_sign_are_well_defined():
    points = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    keys = points.repeat(16, 1)[None, None]
    q = torch.tensor([[1., 1.], [-2., .4]])
    cache = build_pair_envelope(keys, K=1)
    assert bool(cache.isotropic.all())
    scores, _ = score_pair_envelope(q, cache)
    expected = torch.linalg.vector_norm(q, dim=-1)[:, None]
    torch.testing.assert_close(scores, expected, atol=1e-4, rtol=1e-4)
    moved, _ = score_pair_envelope(rotate(q, [.39]), build_pair_envelope(rotate(keys, [.39]), K=1))
    torch.testing.assert_close(scores, moved, atol=2e-5, rtol=2e-5)
    anisotropic = build_pair_envelope(keys * torch.tensor([3., .2]), K=1)
    first, _ = score_pair_envelope(q, anisotropic)
    second, _ = score_pair_envelope(q, replace(anisotropic, axis=-anisotropic.axis))
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("pairing", ["native", "random"])
def test_random_upper_bounds_partial_rope_and_bfloat16_keys(pairing):
    rng = np.random.default_rng(123)
    keys = torch.from_numpy(rng.normal(size=(2, 9, 64, 12))).to(torch.bfloat16)
    q = rng.normal(size=(8, 12)).astype(np.float32)
    cache = build_pair_envelope(keys, omega=np.array([1., .1, .01, 0.]), pairing=pairing)
    scores, info = score_pair_envelope(q, cache)
    truth = exact_max(keys, q)
    assert scores.shape == (8, 9)
    assert bool((scores.double() >= truth).all())
    assert info["source_key_bytes"] == keys.numel() * 2
    quest, _ = score_quest(q, build_quest(keys))
    assert bool((quest.double() >= truth).all())


def test_native_rotation_invariance_and_random_pair_control():
    rng = np.random.default_rng(2026)
    keys = rng.normal(size=(2, 5, 64, 10)).astype(np.float32)
    q = rng.normal(size=(6, 10)).astype(np.float32)
    omega = np.array([.8, .13, .03, .0002])
    base, _ = score_pair_envelope(q, build_pair_envelope(keys, omega))
    for offset in (1., 7., 8191.):
        shifted, _ = score_pair_envelope(rotate(q, omega * offset),
            build_pair_envelope(rotate(keys, omega * offset), omega))
        torch.testing.assert_close(base, shifted, atol=1e-4, rtol=2e-5)
    wrong = build_pair_envelope(keys, omega, pairing="random")
    wrong_rotated = build_pair_envelope(rotate(keys, omega * 7), omega, pairing="random")
    before, _ = score_pair_envelope(q, wrong)
    after, _ = score_pair_envelope(rotate(q, omega * 7), wrong_rotated)
    assert float((before - after).abs().max()) > 1e-3


def test_known_quest_gauge_failure_and_rectangle_repair():
    a = torch.tensor([[1., 1.], [-1., -1.]]).repeat(32, 1)
    b = torch.tensor([[.5, -.5]]).repeat(64, 1)
    keys = torch.stack((a, b))[None]
    q = torch.tensor([[1., -1.]])
    selections = []
    for angle in (0., np.pi / 4, np.pi / 7):
        kr, qr = rotate(keys, [angle]), rotate(q, [angle])
        truth = exact_max(kr, qr)
        envelope, _ = score_pair_envelope(qr, build_pair_envelope(kr, K=1))
        quest, _ = score_quest(qr, build_quest(kr))
        assert int(truth.argmax()) == int(envelope.argmax()) == 1
        selections.append(int(quest.argmax()))
    assert selections == [0, 1, 0]
    # A centroid disk is equivariant but cannot fix this thin-line false positive.
    disk_a = torch.linalg.vector_norm(q) * torch.linalg.vector_norm(a, dim=-1).max()
    assert float(disk_a) == pytest.approx(2.)


def test_known_rectangle_worse_than_quest_and_cross_pair_limitation():
    keys = torch.tensor([[[[0., 0.], [2., 0.], [0., 1.]]]])
    q = torch.tensor([[1., 0.]])
    envelope, _ = score_pair_envelope(q, build_pair_envelope(keys, K=1))
    quest, _ = score_quest(q, build_quest(keys))
    assert float(envelope) > float(quest) + .10
    # Pairwise exact supports need not be jointly attainable by one token.
    keys = torch.tensor([[[[1., 1., 0., 0.], [-1., -1., 0., 0.]]]])
    q = torch.tensor([[1., -1., 0., 0.]])
    envelope, _ = score_pair_envelope(q, build_pair_envelope(keys, K=2))
    assert float(exact_max(keys, q)) == 0
    assert float(envelope) >= 2


def test_exact_byte_accounting_and_reproducible_control():
    keys = torch.zeros(2, 7, 64, 8, dtype=torch.bfloat16)
    native = build_pair_envelope(keys, K=3)
    random = build_pair_envelope(keys, K=3, pairing="random")
    repeated = build_pair_envelope(keys.float().numpy(),
                                   K=3, pairing="random")
    assert torch.equal(random.pair_indices, repeated.pair_indices)
    ni, ri = native.byte_info(), random.byte_info()
    assert ni["total_tensor_bytes"] == ri["total_tensor_bytes"]
    expected_descriptor = 2 * 7 * ((6*3 + 2*(8-6))*4 + 3)
    assert ni["descriptor_bytes"] == expected_descriptor
    assert ni["shared_pair_index_bytes"] == 3*2*8
    assert ni["total_tensor_bytes"] == expected_descriptor + 48
    assert build_quest(keys).byte_info()["total_tensor_bytes"] == 2*7*8*2*4
    for name in ("center", "axis", "halfwidth", "nonrot_min", "nonrot_max"):
        assert getattr(native, name).dtype == torch.float32


def test_no_rotary_dimensions_matches_quest():
    generator = torch.Generator().manual_seed(2)
    keys = torch.randn(1, 3, 7, 5, generator=generator)
    q = torch.randn(2, 5, generator=generator)
    pair, _ = score_pair_envelope(q, build_pair_envelope(keys, K=0))
    quest, _ = score_quest(q, build_quest(keys))
    torch.testing.assert_close(pair, quest, rtol=0, atol=0)


def test_input_validation():
    keys = torch.zeros(1, 2, 4, 8)
    with pytest.raises(ValueError, match="disagree"):
        build_pair_envelope(keys, omega=[1., .1], K=3)
    with pytest.raises(ValueError, match="2\\*K"):
        build_pair_envelope(keys, K=5)
    with pytest.raises(ValueError, match="pairing"):
        build_pair_envelope(keys, K=2, pairing="unknown")
    with pytest.raises(ValueError, match="finite"):
        build_pair_envelope(keys + float("nan"), K=2)
    with pytest.raises(ValueError, match="multiple"):
        score_pair_envelope(torch.ones(3, 8), build_pair_envelope(keys.expand(2, -1, -1, -1), K=2))


def test_analytic_2x2_principal_axis_matches_eigh_nondegenerate_and_degenerate():
    from experiments.native_sparse_position.pair_envelope import _principal_axis_2x2
    generator = torch.Generator().manual_seed(918)
    factors = torch.randn(100, 2, 2, generator=generator, dtype=torch.float64)
    cov = factors @ factors.transpose(-1, -2)
    # Include exact and near degeneracy, rank one, both off-diagonal signs,
    # and directions near the atan2 branch cut.
    special = torch.tensor([
        [[0., 0.], [0., 0.]], [[3., 0.], [0., 3.]],
        [[1., 0.], [0., 1. + 1e-12]], [[1., 2.], [2., 4.]],
        [[4., -2.], [-2., 1.]], [[1., 1e-12], [1e-12, 4.]],
        [[1., -1e-12], [-1e-12, 4.]],
    ], dtype=torch.float64)
    cov = torch.cat((cov, special))
    eigenvalues, axis = _principal_axis_2x2(cov)
    expected_values, expected_vectors = torch.linalg.eigh(cov)
    torch.testing.assert_close(eigenvalues, expected_values, rtol=2e-13, atol=2e-14)
    torch.testing.assert_close(axis.square().sum(-1), torch.ones(len(cov), dtype=torch.float64),
                               rtol=2e-14, atol=2e-14)
    torch.testing.assert_close((cov @ axis[..., None]).squeeze(-1),
                               eigenvalues[..., 1, None] * axis, rtol=2e-13, atol=2e-14)
    distinct = eigenvalues[..., 1] - eigenvalues[..., 0] > 1e-10
    agreement = (axis[distinct] * expected_vectors[distinct, :, 1]).sum(-1).abs()
    torch.testing.assert_close(agreement, torch.ones_like(agreement), rtol=2e-13, atol=2e-14)


def test_builder_does_not_call_linalg_eigh(monkeypatch):
    monkeypatch.setattr(torch.linalg, "eigh", lambda *a, **kw: pytest.fail("Unexpected cuSOLVER/eigh dependency"))
    keys = torch.randn(2, 3, 64, 8, generator=torch.Generator().manual_seed(12))
    q = torch.randn(4, 8, generator=torch.Generator().manual_seed(13))
    cache = build_pair_envelope(keys, K=4)
    bound, _ = score_pair_envelope(q, cache)
    assert bool((bound.double() >= exact_max(keys, q)).all())


def test_native_pair_marginals_cannot_identify_full_token_maximum():
    """Same complete pair marginals can encode distinct joint support functions."""
    a=torch.tensor([[1.,1.,0.,0.],[-1.,-1.,0.,0.]]).repeat(32,1)
    b=torch.tensor([[1.,-1.,0.,0.],[-1.,1.,0.,0.]]).repeat(32,1)
    q=torch.tensor([[1.,-1.,0.,0.]])
    ca=build_pair_envelope(a[None,None],K=2)
    cb=build_pair_envelope(b[None,None],K=2)
    assert torch.equal(score_pair_envelope(q,ca)[0],score_pair_envelope(q,cb)[0])
    assert (a@q[0]).max().item()==0
    assert (b@q[0]).max().item()==2
