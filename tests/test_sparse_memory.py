"""Decision-relevant operator, causality, data and matched-arm checks."""
import random

import numpy as np
import pytest
import torch

from scripts.experiments.sparse_memory.data import one_pair, solve
from scripts.experiments.sparse_memory.model import Config, Model, Compressor, rotate, attention_mask


def test_labels_pair_multisets_and_remote_sources():
    rng = random.Random(928)
    layouts = set()
    for i in range(500):
        rows, ys, meta = one_pair(rng, i)
        assert [solve(x) for x in rows] == ys
        assert ys[0] != ys[1]
        assert sorted(rows[0][:-4]) == sorted(rows[1][:-4])
        assert np.array_equal(rows[0][-4:], rows[1][-4:])
        assert min(131-p for p in meta['target_positions']) > 3*(16-1)
        layouts.add(tuple(meta['slots']))
    assert len(layouts) == 2


def test_rotation_independent_complex_and_inverse():
    torch.manual_seed(123)
    x = torch.randn(2, 3, 12, dtype=torch.float64)
    freq = torch.tensor([1., .17, .011, .0002])
    pos = torch.tensor([[0, 2, 13]])
    actual = rotate(x, pos, freq)
    # Explicit independent complex multiplication, without the implementation.
    source = x.numpy()[..., -8:].reshape(2, 3, 4, 2)
    z = source[..., 0]+1j*source[..., 1]
    expected = z*np.exp(1j*pos.numpy()[..., None]*freq.numpy())
    np.testing.assert_allclose(actual.numpy()[..., -8::2], expected.real, atol=3e-7, rtol=2e-6)
    np.testing.assert_allclose(actual.numpy()[..., -7::2], expected.imag, atol=3e-7, rtol=2e-6)
    torch.testing.assert_close(rotate(actual, -pos, freq), x, atol=5e-7, rtol=2e-6)
    torch.testing.assert_close(actual[..., :-8], x[..., :-8])


def test_gated_operator_reference_and_arm_identity():
    torch.manual_seed(7)
    cfg = Config(width=16, head_dim=12, rotary_dim=8, ratio=4)
    a = Compressor(cfg, 'baseline')
    b = Compressor(cfg, 'tp'); b.load_state_dict(a.state_dict())
    x = torch.randn(2, 8, 16, requires_grad=True)
    u = a.kv(x).reshape(2, 2, 4, 12)
    w = (a.gate(x).reshape_as(u)+a.ape).softmax(2)
    f = u*w
    # Source-by-source independently indexed reduction catches wrong axis/order.
    expected = sum(rotate(f[:, :, j], torch.tensor([[j]]), a.frequency) for j in range(4))
    expected = rotate(a.norm(expected), torch.tensor([[0, 4]]), a.frequency)
    torch.testing.assert_close(b(x), expected)
    assert (b(x)-a(x)).abs().max() > .01
    b(x).square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert b.gate.weight.grad.abs().sum() > 0
    assert b.ape.grad.abs().sum() > 0
    # Degenerate zero-relative-position case must reduce to A.
    cfg.ratio = 1
    torch.manual_seed(8); a = Compressor(cfg, 'baseline')
    torch.manual_seed(8); b = Compressor(cfg, 'tp')
    torch.testing.assert_close(a(x), b(x))


def test_causal_completed_blocks_and_no_future_leak():
    cfg = Config(width=24, heads=2, head_dim=16, rotary_dim=8, layers=2, ratio=4, window=4)
    mask = attention_mask(12, cfg, 'cpu')
    assert not mask[2, 12:].any() and mask[3, 12] and not mask[3, 13]
    assert not mask[0, 1:12].any()
    torch.manual_seed(1)
    model = Model(cfg, 'tp').eval()
    x = torch.randint(0, 64, (2, 12)); changed = x.clone(); changed[:, 7:] = 63
    torch.testing.assert_close(model(x, True)[:, :7], model(changed, True)[:, :7], atol=1e-6, rtol=1e-5)


def test_dense_control_changes_visibility_without_future_or_summary_access():
    cfg = Config(ratio=4, window=4, dense_control=True)
    mask = attention_mask(12, cfg, 'cpu')
    assert mask[11, :12].all() and not mask[:, 12:].any()
    assert not mask[4, 5:12].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU qualification after power-mode change')
def test_cuda_backend_forward_backward_matches_cpu():
    cfg = Config(width=24, heads=2, head_dim=16, rotary_dim=8, layers=2, ratio=4, window=4)
    torch.manual_seed(19)
    model = Model(cfg, 'tp')
    tokens = torch.randint(0, 64, (2, 12))
    expected = model(tokens)
    gpu = Model(cfg, 'tp').cuda(); gpu.load_state_dict(model.state_dict())
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actual = gpu(tokens.cuda())
    torch.testing.assert_close(actual.float().cpu(), expected, atol=.035, rtol=.08)
    actual.float().square().mean().backward()
    assert torch.isfinite(gpu.layers[0].compressor.gate.weight.grad).all()
