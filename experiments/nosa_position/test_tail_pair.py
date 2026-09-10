"""Finite mechanism and causal-contract tests; not LM quality evidence."""
import math

import torch

from .runtime import AttentionSettings, SelectionContext
from .selector_controls import build_summary, summary_logmass
from .tail_pair import TailPairSelector, build_tail_summary, tail_logmass


def test_single_extreme_key_recovers_exact_mass_where_full_second_order_fails():
    keys = torch.zeros(1, 2, 64, 2, dtype=torch.float64)
    keys[0, 0, 0, 0] = 10
    keys[0, 1, :, 0] = 1
    bias = torch.zeros(1, 2, 64, dtype=torch.float64)
    q = torch.tensor([[[[1., 0.]]]], dtype=torch.float64)
    exact = torch.einsum("hgqd,hbtd->hgqbt", q, keys).logsumexp(-1)
    second_order = summary_logmass(q, build_summary(keys, bias, "pc2"), "pc2")
    actual = tail_logmass(q, build_tail_summary(keys, bias))
    torch.testing.assert_close(actual, exact, rtol=1e-12, atol=1e-12)
    assert exact[..., 0] > exact[..., 1]
    assert second_order[..., 0] < second_order[..., 1]


def test_cis_weights_and_zero_query_mass_are_preserved():
    torch.manual_seed(901)
    keys = torch.randn(2, 3, 64, 8, dtype=torch.float64)
    bias = 3 * torch.randn(2, 3, 64, dtype=torch.float64)
    q = torch.zeros(2, 2, 4, 8, dtype=torch.float64)
    actual = tail_logmass(q, build_tail_summary(keys, bias))
    expected = bias.logsumexp(-1)[:, None, None].expand_as(actual)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_future_key_perturbation_does_not_change_visible_queries():
    torch.manual_seed(91)
    settings = AttentionSettings(kernel_size=4, kernel_stride=2, block_size=4,
                                 init_blocks=1, local_blocks=0, select_blocks=1, topk=2)
    q = torch.randn(4, 2, 8)
    k, v = torch.randn(2, 12, 8), torch.randn(2, 12, 8)
    cis = torch.randn(2, 12)
    pos = torch.tensor([8, 9])
    context = SelectionContext(q, k, v, cis, pos, 0, settings)
    first = TailPairSelector().logmass(context)
    k2, cis2 = k.clone(), cis.clone()
    k2[:, 10:] *= 1000
    cis2[:, 10:] += 1000
    other = SelectionContext(q, k2, v, cis2, pos, 0, settings)
    second = TailPairSelector().logmass(other)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    selected = TailPairSelector()(context)
    assert selected.shape == (2, 2, 2)
    assert bool((selected <= pos[None, :, None] // 4).all())


def test_one_tail_is_not_a_universal_exact_summary():
    keys = torch.zeros(1, 1, 64, 4, dtype=torch.float64)
    keys[0, 0, 0, 0] = 20
    keys[0, 0, 1, 1] = 10
    bias = torch.zeros(1, 1, 64, dtype=torch.float64)
    q = torch.tensor([[[[0., 1., 0., 0.]]]], dtype=torch.float64)
    actual = tail_logmass(q, build_tail_summary(keys, bias))
    exact = torch.einsum("hgqd,hbtd->hgqbt", q, keys).logsumexp(-1)
    assert float(exact - actual) > 4
