from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from scripts.lib.rope.length_conditioned_budgeted import (
    LengthConditionedRotaryEmbedding,
    RequestBudgetState,
    install_length_conditioned_rope,
    matched_attention_scaling,
    select_observed_session_factor,
)
from scripts.analysis.export_uniqueness_budgeted_tables import (
    EXPECTED_DEFAULT_HASHES,
    build_default_tables,
    float32_sha256,
)


class DummyRotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.tensor([1.0, 0.5]))
        self.original_inv_freq = self.inv_freq.detach().clone()
        self.attention_scaling = 1.0

    def forward(
        self, value: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase = position_ids.float().unsqueeze(-1) * self.inv_freq
        phase = torch.cat((phase, phase), dim=-1)
        return (
            phase.cos() * float(self.attention_scaling),
            phase.sin() * float(self.attention_scaling),
        )


def test_short_branch_is_exact_native_direct_call() -> None:
    native = DummyRotary()
    positions = torch.tensor([[0, 1, 4095]])
    expected = native(torch.empty(1), positions)
    state = RequestBudgetState(reference_length=4096)
    gated = LengthConditionedRotaryEmbedding(
        native,
        long_inv_freq=torch.tensor([0.5, 0.25]),
        long_attention_scaling=1.1,
        long_name="budgeted",
        state=state,
    )
    observed = gated(torch.empty(1), positions)
    assert torch.equal(observed[0], expected[0])
    assert torch.equal(observed[1], expected[1])


def test_long_branch_matches_frozen_table_and_scaling() -> None:
    native = DummyRotary()
    state = RequestBudgetState(reference_length=4096)
    target = torch.tensor([0.5, 0.25])
    gated = LengthConditionedRotaryEmbedding(
        native,
        long_inv_freq=target,
        long_attention_scaling=1.1,
        long_name="budgeted",
        state=state,
    )
    state.force_for_budget(8192)
    positions = torch.tensor([[0, 17, 4095]])
    expected_phase = positions.float().unsqueeze(-1) * target
    expected_phase = torch.cat((expected_phase, expected_phase), dim=-1)
    observed = gated(torch.empty(1), positions)
    assert torch.equal(observed[0], expected_phase.cos() * 1.1)
    assert torch.equal(observed[1], expected_phase.sin() * 1.1)


def test_mixed_batch_dispatches_per_row() -> None:
    native = DummyRotary()
    state = RequestBudgetState(reference_length=4096)
    gated = LengthConditionedRotaryEmbedding(
        native,
        long_inv_freq=torch.tensor([0.5, 0.25]),
        long_attention_scaling=1.1,
        long_name="budgeted",
        state=state,
    )
    positions = torch.tensor([[0, 1], [4096, 4097]])
    observed = gated(torch.empty(1), positions)
    native_out = native(torch.empty(1), positions)
    long_out = gated.long(torch.empty(1), positions)
    assert torch.equal(observed[0][0], native_out[0][0])
    assert torch.equal(observed[1][0], native_out[1][0])
    assert torch.equal(observed[0][1], long_out[0][1])
    assert torch.equal(observed[1][1], long_out[1][1])


def test_cached_crossing_fails_without_bound_request_budget() -> None:
    state = RequestBudgetState(reference_length=4096)
    assert state.update(torch.tensor([[4095]])) == "short"
    with pytest.raises(RuntimeError, match="crossed"):
        state.update(torch.tensor([[4096]]))
    state.clear()
    assert state.force_for_budget(8192) == "long"
    assert state.update(torch.tensor([[0]])) == "long"


def test_install_receipts_zero_parameter_two_branch_contract() -> None:
    model = SimpleNamespace(model=SimpleNamespace(rotary_emb=DummyRotary()))
    state, receipt = install_length_conditioned_rope(
        model,
        long_inv_freq=torch.tensor([0.5, 0.25]),
        long_attention_scaling=matched_attention_scaling(2.0),
        long_name="budgeted_s2_p2",
        reference_length=4096,
        long_context_budget=8192,
    )
    assert isinstance(model.model.rotary_emb, LengthConditionedRotaryEmbedding)
    assert receipt["learned_parameters"] == 0
    assert receipt["short_branch"]["operator"] == "Native rotary module direct call"
    assert state.force_for_budget(4096) == "short"
    assert state.force_for_budget(4097) == "long"
    with pytest.raises(ValueError, match="exceeds"):
        state.force_for_budget(8193)


def test_matched_attention_scaling_values() -> None:
    assert matched_attention_scaling(1.0) == 1.0
    assert matched_attention_scaling(2.0) == pytest.approx(1.0693147180559945)
    assert matched_attention_scaling(4.0) == pytest.approx(1.138629436111989)


def test_observed_session_factor_uses_model_relative_boundaries() -> None:
    assert select_observed_session_factor(
        prefill_tokens=4000,
        max_new_tokens=64,
        native_context_length=4096,
    ) == 1
    assert select_observed_session_factor(
        prefill_tokens=8150,
        max_new_tokens=32,
        native_context_length=4096,
    ) == 2
    assert select_observed_session_factor(
        prefill_tokens=8150,
        max_new_tokens=64,
        native_context_length=4096,
    ) == 4
    assert select_observed_session_factor(
        prefill_tokens=64000,
        max_new_tokens=512,
        native_context_length=32768,
    ) == 2
    assert select_observed_session_factor(
        prefill_tokens=4000,
        max_new_tokens=64,
        native_context_length=4096,
        supported_factors=(1, 4),
    ) == 1
    assert select_observed_session_factor(
        prefill_tokens=4096,
        max_new_tokens=1,
        native_context_length=4096,
        supported_factors=(1, 4),
    ) == 4


def test_observed_session_factor_fails_beyond_frozen_profiles() -> None:
    with pytest.raises(ValueError, match="beyond the largest supported profile"):
        select_observed_session_factor(
            prefill_tokens=16384,
            max_new_tokens=1,
            native_context_length=4096,
        )


def test_frozen_default_tables_rebuild_to_registered_hashes() -> None:
    tables = build_default_tables()
    assert set(tables) == set(EXPECTED_DEFAULT_HASHES)
    for factor, expected in EXPECTED_DEFAULT_HASHES.items():
        assert tables[factor].dtype == "float32"
        assert tables[factor].shape == (64,)
        assert float32_sha256(tables[factor]) == expected
