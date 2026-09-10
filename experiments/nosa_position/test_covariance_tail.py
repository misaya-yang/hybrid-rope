import torch

from .covariance_tail import CovarianceTailSelector, build_covariance_tail_summary, extraction_gains
from .runtime import AttentionSettings, SelectionContext
from .selector_controls import build_summary, summary_logmass
from .tail_pair import build_tail_summary, tail_logmass


def offpair(matrix):
    d = matrix.shape[-1]
    keep = torch.eye(d, dtype=torch.bool)
    for j in range(d // 2):
        keep[j, j + d // 2] = keep[j + d // 2, j] = True
    return matrix.masked_fill(keep, 0)


def covariance(x, w):
    centered = x - (w[..., None] * x).sum(-2, keepdim=True)
    return centered.transpose(-1, -2) @ (w[..., None] * centered)


def test_criterion_matches_bruteforce_real_weighted_covariance():
    torch.manual_seed(27)
    x, cis = torch.randn(2, 3, 7, 8, dtype=torch.float64), torch.randn(2, 3, 7, dtype=torch.float64)
    gains, index, active = extraction_gains(x, cis)
    w = cis.softmax(-1)
    original_loss = offpair(covariance(x, w)).square().sum((-2, -1))
    brute = []
    for i in range(7):
        other = w.clone()
        other[..., i] = 0
        other /= other.sum(-1, keepdim=True)
        error = (1 - w[..., i, None, None]) * offpair(covariance(x, other))
        brute.append(original_loss - error.square().sum((-2, -1)))
    brute = torch.stack(brute, -1)
    torch.testing.assert_close(gains, brute, atol=2e-12, rtol=2e-12)
    assert torch.equal(index, brute.argmax(-1))
    assert torch.equal(active, brute.max(-1).values > 1e-12)


def test_no_extraction_when_no_pair_covariance_is_missing():
    x = torch.tensor([[1., 1., 0., 0.], [1., -1., 0., 0.],
                      [-1., 1., 0., 0.], [-1., -1., 0., 0.]], dtype=torch.float64)[None, None]
    cis = torch.zeros(1, 1, 4, dtype=torch.float64)
    _, _, active = extraction_gains(x, cis)
    assert not bool(active.any())
    q = torch.tensor([[[[.3, -.7, .4, .2]]]], dtype=torch.float64)
    new = build_covariance_tail_summary(x, cis)
    plain = build_summary(x, cis, "pc2")
    torch.testing.assert_close(tail_logmass(q, new), summary_logmass(q, plain, "pc2"), rtol=0, atol=0)
    assert new.nbytes() == build_tail_summary(x, cis).nbytes()


def test_zero_cross_tie_preserves_known_single_pair_rare_key_repair():
    x = torch.zeros(1, 1, 64, 4, dtype=torch.float64)
    x[0, 0, 7, 0] = 10
    cis = torch.zeros(1, 1, 64, dtype=torch.float64)
    _, index, active = extraction_gains(x, cis)
    assert bool(active[0, 0]) and int(index[0, 0]) == 7
    q = torch.tensor([[[[1., 0., 0., 0.]]]], dtype=torch.float64)
    actual = tail_logmass(q, build_covariance_tail_summary(x, cis))
    exact = torch.einsum("hgqd,hbtd->hgqbt", q, x).logsumexp(-1)
    torch.testing.assert_close(actual, exact, atol=1e-12, rtol=1e-12)


def test_mixture_hessian_has_exact_selected_between_component_covariance():
    torch.manual_seed(64)
    x, cis = torch.randn(1, 1, 9, 6, dtype=torch.float64), torch.randn(1, 1, 9, dtype=torch.float64)
    gains, index, active = extraction_gains(x, cis)
    summary = build_covariance_tail_summary(x, cis)
    w = cis.softmax(-1)
    full = covariance(x, w)[0, 0]
    remaining = offpair(full)
    if bool(active[0, 0]):
        i = int(index[0, 0]);r = x[0, 0, i] - (w[..., None] * x).sum(-2)[0, 0]
        between = w[0, 0, i] / (1 - w[0, 0, i]) * r[:, None] * r[None]
        remaining = remaining - offpair(between)
    hessian = torch.autograd.functional.hessian(
        lambda q: tail_logmass(q.reshape(1, 1, 1, -1), summary).sum(), torch.zeros(6, dtype=torch.float64))
    torch.testing.assert_close(hessian, full - remaining, atol=2e-12, rtol=2e-12)
    assert float(remaining.square().sum()) <= float(offpair(full).square().sum()) + 1e-12


def test_native_pair_rotation_preserves_selection_and_gain():
    torch.manual_seed(18)
    x, cis = torch.randn(1, 3, 12, 8, dtype=torch.float64), torch.randn(1, 3, 12, dtype=torch.float64)
    angle = torch.randn(4, dtype=torch.float64)
    left, right = x.chunk(2, -1)
    rotated = torch.cat((left * angle.cos() - right * angle.sin(),
                         left * angle.sin() + right * angle.cos()), -1)
    g1, i1, a1 = extraction_gains(x, cis);g2, i2, a2 = extraction_gains(rotated, cis)
    torch.testing.assert_close(g1, g2, atol=2e-12, rtol=2e-12)
    assert torch.equal(i1, i2) and torch.equal(a1, a2)


def test_causal_runner_and_no_extra_summary_storage():
    torch.manual_seed(7)
    settings = AttentionSettings(kernel_size=4, kernel_stride=2, block_size=4,
        init_blocks=1, local_blocks=0, select_blocks=1, topk=2)
    q, k, v, cis = torch.randn(4, 2, 8), torch.randn(2, 12, 8), torch.randn(2, 12, 8), torch.randn(2, 12)
    pos = torch.tensor([8, 9])
    ctx = SelectionContext(q, k, v, cis, pos, 0, settings)
    before = CovarianceTailSelector().logmass(ctx)
    k2, bias2 = k.clone(), cis.clone();k2[:, 10:] *= 1000;bias2[:, 10:] += 1000
    after = CovarianceTailSelector().logmass(SelectionContext(q, k2, v, bias2, pos, 0, settings))
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    selected = CovarianceTailSelector()(ctx)
    assert selected.shape == (2, 2, 2)
    assert bool((selected <= pos[None, :, None] // 4).all())


def test_complete_tiny_nosa_model_chunking_keeps_the_same_last_logits():
    from .runtime import NosaReferenceForCausalLM
    from .test_runtime import tiny_config, tiny_settings
    torch.set_num_threads(1)
    torch.manual_seed(53)
    model = NosaReferenceForCausalLM(tiny_config(), settings=tiny_settings()).eval()
    model.selector = CovarianceTailSelector()
    ids = torch.randint(4, 41, (1, 51))
    with torch.inference_mode():
        full = model(ids)
        chunked = model.prefill(ids, chunk_size=7)
    torch.testing.assert_close(full.logits[:, -1:], chunked.logits, atol=2e-6, rtol=2e-5)
