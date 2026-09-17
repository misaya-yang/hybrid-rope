from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.native_followup_five_20260917.rank_assignment import (
    METHOD_ID,
    _project_qkv,
    install,
    rank_assignment_attention,
    stable_rank_probabilities,
)


def test_native_probability_multiset_and_candidate_order_are_exact():
    native = torch.tensor([[0.1, 2.0, -0.4, 0.8]], dtype=torch.float64)
    candidate = torch.tensor([[3.0, -2.0, 1.0, 0.0]], dtype=torch.float64)
    assigned = stable_rank_probabilities(native, candidate)
    native_probabilities = torch.softmax(native.float(), dim=-1)
    torch.testing.assert_close(
        torch.sort(assigned, descending=True).values,
        torch.sort(native_probabilities, descending=True).values,
    )
    rank = torch.argsort(candidate, descending=True, stable=True)
    ranked_assigned = torch.gather(assigned, -1, rank)
    assert torch.all(ranked_assigned[..., :-1] >= ranked_assigned[..., 1:])


def test_candidate_ties_use_native_score_then_original_key_index():
    native = torch.tensor([[3.0, 5.0, 1.0, 1.0]])
    candidate = torch.tensor([[7.0, 7.0, 2.0, 2.0]])
    assigned = stable_rank_probabilities(native, candidate)
    confidence = torch.sort(torch.softmax(native, dim=-1), descending=True).values
    assert assigned[0, 1] == confidence[0, 0]
    assert assigned[0, 0] == confidence[0, 1]
    # Native tie at keys 2/3 falls through to original absolute key order; the
    # assigned probabilities are equal because their Native scores are equal.
    assert assigned[0, 2] == assigned[0, 3]


def test_masked_keys_receive_zero_and_do_not_change_visible_confidence():
    native = torch.tensor([[0.0, 1.0, 8.0, 9.0]])
    candidate = torch.tensor([[0.0, 2.0, 100.0, 101.0]])
    visible = torch.tensor([[True, True, False, False]])
    assigned = stable_rank_probabilities(native, candidate, visible)
    torch.testing.assert_close(assigned[0, :2], torch.softmax(native[0, :2], dim=-1))
    torch.testing.assert_close(assigned[0, 2:], torch.zeros(2))


def test_identical_ranking_recovers_native_attention_probabilities():
    native = torch.tensor([[0.4, -0.1, 2.0, 0.0]])
    candidate = native * 3 + 7
    assigned = stable_rank_probabilities(native, candidate)
    torch.testing.assert_close(assigned, torch.softmax(native, dim=-1))


def test_rank_attention_sorts_the_complete_key_axis_then_one_pv():
    # Zero frequencies make QK logits ordinary dot products, so the expected
    # complete-key assignment can be computed directly without a model.
    raw_query = torch.tensor([[[[1.0, 0.0], [0.5, 1.0]]]])
    raw_key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    value = torch.tensor([[[[10.0, 0.0], [0.0, 20.0]]]])
    frequency = torch.tensor([1e-12])
    output = rank_assignment_attention(
        raw_query,
        raw_key,
        value,
        frequency,
        frequency,
        1.0,
        query_tile_size=1,
    )
    # Query 0 can only see key 0. Query 1 keeps its Native distribution because
    # candidate and Native ranks are identical.
    expected_second = torch.softmax(torch.tensor([0.5, 1.0]), dim=-1) @ value[0, 0]
    torch.testing.assert_close(output[0, 0, 0], value[0, 0, 0])
    torch.testing.assert_close(output[0, 0, 1], expected_second)


class Scale(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, value):
        return value * self.factor


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 4
        self.q_proj = nn.Identity()
        self.k_proj = nn.Identity()
        self.v_proj = nn.Identity()
        self.q_norm = Scale(2.0)
        self.k_norm = Scale(3.0)
        self.o_proj = nn.Identity()
        self.scaling = 0.5
        self.layer_idx = 0
        self.training = False

    def forward(self, *args, **kwargs):
        return "original"


def fake_model():
    attention = FakeAttention()
    return SimpleNamespace(
        config=SimpleNamespace(
            model_type="olmo2",
            head_dim=4,
            hidden_size=4,
            num_attention_heads=1,
        ),
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]),
    )


def test_projection_uses_qk_norm_before_rope():
    attention = FakeAttention()
    hidden = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    query, key, value = _project_qkv(attention, hidden)
    expected = hidden.view(1, 2, 1, 4).transpose(1, 2)
    torch.testing.assert_close(query, expected * 2)
    torch.testing.assert_close(key, expected * 3)
    torch.testing.assert_close(value, expected)


def test_install_method_id_and_idempotent_restore():
    model = fake_model()
    attention = model.model.layers[0].self_attn
    original = attention.forward
    native = {"values_float32": [1.0, 0.1], "gain": 1.0}
    candidate = {"values_float32": [1.0, 0.05], "gain": 1.0}
    restore = install(model, native, candidate)
    assert restore.method_id == METHOD_ID
    assert restore.identity_fast_path is False
    assert attention._native_rank_assignment_method == "rank_assignment"
    assert attention.forward != original
    restore()
    assert attention.forward == original
    assert not hasattr(attention, "_native_rank_assignment_method")
    restore()


def test_identity_table_uses_stock_fast_path():
    model = fake_model()
    original = model.model.layers[0].self_attn.forward
    table = {"values_float32": [1.0, 0.1], "gain": 1.0}
    restore = install(model, table, table)
    assert restore.method_id == METHOD_ID
    assert restore.identity_fast_path is True
    assert model.model.layers[0].self_attn.forward == original
    restore()


def test_contract_rejects_non_olmo_batch_and_padding():
    table = {"values_float32": [1.0, 0.1], "gain": 1.0}
    model = fake_model()
    model.config.model_type = "qwen2"
    with pytest.raises(ValueError, match="OLMo2-only"):
        install(model, table, table)
    raw = torch.zeros(2, 1, 1, 2)
    with pytest.raises(ValueError, match="batch=1"):
        rank_assignment_attention(raw, raw, raw, torch.tensor([1.0]), torch.tensor([0.5]), 1.0)


def test_cli_is_plan_only():
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.native_followup_five_20260917.rank_assignment"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PLAN_ONLY"
    assert payload["model_loaded"] is False
    assert payload["gpu_execution"] is False
    assert payload["method_id"] == METHOD_ID
