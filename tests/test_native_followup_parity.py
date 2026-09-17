import math

import numpy as np
import pytest
import torch
from torch import nn

from experiments.native_followup_five_20260917.parity_attention import (
    METHOD_IDS,
    _table_values,
    build_parity_states,
    install,
    project_olmo_qkv,
    single_normalization_attention,
)


def _scores(query, key, scale):
    return torch.matmul(query, key.transpose(-1, -2)) * scale


def _rotary_scores(query, key, positions, frequencies):
    from experiments.native_followup_five_20260917.parity_attention import (
        _frequency_embeddings,
        _rotate_split_half,
    )

    cos, sin = _frequency_embeddings(positions, frequencies, dtype=query.dtype)
    q = _rotate_split_half(query, cos, sin)
    k = _rotate_split_half(key, cos, sin)
    return _scores(q, k, 1.0 / math.sqrt(query.shape[-1]))


@pytest.mark.parametrize("method", ["even_only", "odd_only"])
def test_augmented_dot_product_is_exact_parity_formula(method):
    generator = torch.Generator().manual_seed(20260917)
    query = torch.randn(1, 2, 7, 8, generator=generator, dtype=torch.float64)
    key = torch.randn(1, 2, 7, 8, generator=generator, dtype=torch.float64)
    positions = torch.arange(7)[None]
    native = np.geomspace(1.0, 0.01, 4).astype(np.float32)
    candidate = native.copy()
    candidate[1:-1] *= np.asarray([0.83, 0.91], dtype=np.float32)

    augmented_q, augmented_k = build_parity_states(
        query, key, positions, native, candidate, method=method,
    )
    actual = _scores(augmented_q, augmented_k, 1.0 / math.sqrt(query.shape[-1]))
    native_plus = _rotary_scores(query, key, positions, native)
    native_minus = _rotary_scores(query, key, positions, -native)
    candidate_plus = _rotary_scores(query, key, positions, candidate)
    candidate_minus = _rotary_scores(query, key, positions, -candidate)
    if method == "even_only":
        expected = 0.5 * (candidate_plus + candidate_minus + native_plus - native_minus)
    else:
        expected = 0.5 * (native_plus + native_minus + candidate_plus - candidate_minus)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("method", ["even_only", "odd_only"])
def test_native_candidate_identity_recovers_standard_rope_scores(method):
    generator = torch.Generator().manual_seed(9)
    query = torch.randn(1, 1, 5, 8, generator=generator, dtype=torch.float64)
    key = torch.randn(1, 1, 5, 8, generator=generator, dtype=torch.float64)
    positions = torch.arange(5)[None]
    frequencies = np.geomspace(1.0, 0.001, 4).astype(np.float32)
    augmented_q, augmented_k = build_parity_states(
        query, key, positions, frequencies, frequencies, method=method,
    )
    actual = _scores(augmented_q, augmented_k, 1.0 / math.sqrt(8))
    expected = _rotary_scores(query, key, positions, frequencies)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_single_normalization_uses_one_global_softmax_and_pv(monkeypatch):
    class Module:
        training = False
        scaling = 0.25

    generator = torch.Generator().manual_seed(12)
    query = torch.randn(1, 1, 3, 16, generator=generator)
    key = torch.randn(1, 1, 3, 16, generator=generator)
    value = torch.randn(1, 1, 3, 4, generator=generator)
    real_softmax = torch.softmax
    calls = []

    def counted_softmax(*args, **kwargs):
        calls.append(1)
        return real_softmax(*args, **kwargs)

    monkeypatch.setattr(torch, "softmax", counted_softmax)
    output, weights = single_normalization_attention(Module(), query, key, value, None)
    assert len(calls) == 1
    assert weights is None
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * Module.scaling
    causal = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, torch.finfo(torch.float32).min)
    expected = torch.matmul(real_softmax(scores, dim=-1), value.float()).to(value.dtype)
    torch.testing.assert_close(output, expected)


def test_query_tiling_does_not_partition_key_normalization():
    class Module:
        training = False
        scaling = 0.5

    generator = torch.Generator().manual_seed(13)
    query = torch.randn(1, 2, 5, 8, generator=generator)
    key = torch.randn(1, 2, 5, 8, generator=generator)
    value = torch.randn(1, 2, 5, 4, generator=generator)
    mask = torch.zeros(1, 1, 5, 5)
    mask[:, :, torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)] = float("-inf")
    actual, _ = single_normalization_attention(
        Module(), query, key, value, mask, query_tile_size=2,
    )
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * Module.scaling
    expected = torch.matmul(torch.softmax(scores + mask, dim=-1), value.float()).to(value.dtype)
    torch.testing.assert_close(actual, expected)


def test_olmo_projection_applies_q_and_k_norms_before_reshape():
    class CountingNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return value + 3

    class FakeAttention:
        head_dim = 4
        q_proj = nn.Identity()
        k_proj = nn.Identity()
        v_proj = nn.Identity()
        q_norm = CountingNorm()
        k_norm = CountingNorm()

    hidden = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
    query, key, value = project_olmo_qkv(FakeAttention, hidden)
    assert FakeAttention.q_norm.calls == 1
    assert FakeAttention.k_norm.calls == 1
    torch.testing.assert_close(query.transpose(1, 2).reshape_as(hidden), hidden + 3)
    torch.testing.assert_close(key.transpose(1, 2).reshape_as(hidden), hidden + 3)
    torch.testing.assert_close(value.transpose(1, 2).reshape_as(hidden), hidden)


def test_table_and_method_contracts_are_frozen():
    assert METHOD_IDS == {
        "even_only": "native_followup_A_even_only_v1",
        "odd_only": "native_followup_A_odd_only_control_v1",
    }
    values, gain = _table_values(
        {"values_float32": [1.0, 0.1, 0.01], "gain": 1.0}, name="test",
    )
    assert values.dtype == np.float32
    assert gain == 1.0
    with pytest.raises(ValueError, match="gain"):
        _table_values({"values_float32": [1.0, 0.1], "gain": 1.1}, name="test")
    with pytest.raises(ValueError, match="strictly decreasing"):
        _table_values({"values_float32": [1.0, 1.0], "gain": 1.0}, name="test")


@pytest.mark.parametrize("method", tuple(METHOD_IDS))
def test_identity_install_exposes_method_receipt_without_patching(method):
    model = type("Model", (), {})()
    model.config = type("Config", (), {"model_type": "olmo2"})()
    model.model = type("Backbone", (), {})()
    model.model.layers = []
    model.model.rotary_emb = nn.Identity()
    table = {"values_float32": [1.0, 0.1], "gain": 1.0}
    restore = install(model, table, table, method)
    assert restore.method_id == METHOD_IDS[method]
    restore()
