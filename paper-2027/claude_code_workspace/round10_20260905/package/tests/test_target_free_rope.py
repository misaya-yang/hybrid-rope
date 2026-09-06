from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.analysis.export_uniqueness_budgeted_tables import (
    EXPECTED_DEFAULT_HASHES,
    build_default_tables,
    causal_distance_measure,
    conditional_pair_uniqueness,
    native_endpoint_inv_freq,
    uniqueness_budgeted_table,
)
from scripts.lib.rope.target_free import (
    ModelRoPEProfile,
    TargetFreeRoPE,
    apply_target_free_qk,
    float32_tensor_sha256,
    install_target_free_olmo2,
)


def _profile(
    *,
    native_length: int,
    inv_freq: list[float],
    head_dim: int,
    rotary_dim: int,
    gain: float = 0.1,
    layout: str = "half_split",
) -> ModelRoPEProfile:
    movement = np.linspace(0.0, 1.0, len(inv_freq), dtype=np.float64)
    return ModelRoPEProfile.from_native(
        torch.tensor(inv_freq, dtype=torch.float32),
        native_context_length=native_length,
        native_context_length_source=f"fixture:{native_length}",
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        movement_coefficients=movement,
        native_rope_config={
            "model_type": "fixture",
            "rope_type": "default",
            "theta": 12345.0,
        },
        native_scaling_config={"type": "default", "factor": 1.0},
        gain_coefficient=gain,
        gain_coefficient_source="fixture-only coefficient",
        pair_layout=layout,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("native_length", "inv_freq", "head_dim", "rotary_dim", "layout"),
    [
        (4096, [1.0, 0.17, 0.031], 8, 6, "half_split"),
        (32768, [0.83, 0.21, 0.047, 0.009], 12, 8, "interleaved"),
    ],
)
def test_native_window_phase_is_strictly_consistent(
    native_length: int,
    inv_freq: list[float],
    head_dim: int,
    rotary_dim: int,
    layout: str,
) -> None:
    profile = _profile(
        native_length=native_length,
        inv_freq=inv_freq,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        layout=layout,
    )
    rope = TargetFreeRoPE(profile)
    positions = torch.tensor([0, 1, native_length - 1, native_length])
    expected = positions.to(torch.float64)[:, None] * torch.tensor(
        profile.native_inv_freq, dtype=torch.float64
    )
    assert torch.equal(rope.phase(positions), expected)


def test_boundary_is_continuous_and_uses_profiled_slope() -> None:
    profile = ModelRoPEProfile.from_native(
        [0.5, 0.125],
        native_context_length=17,
        native_context_length_source="fixture config",
        head_dim=8,
        rotary_dim=4,
        movement_coefficients=[0.0, 0.75],
        native_boundary_slope=[0.5, 0.25],
        native_rope_config={"phase": "linear"},
        native_scaling_config={"type": "none"},
        gain_coefficient=0.0,
        gain_coefficient_source="disabled fixture gain",
    )
    rope = TargetFreeRoPE(profile)
    boundary = rope.phase(torch.tensor([17.0]))[0]
    epsilon = rope.phase(torch.tensor([17.0 + 1.0e-7]))[0]
    assert torch.allclose(boundary, torch.tensor([8.5, 2.125], dtype=torch.float64))
    assert torch.allclose(boundary, epsilon, atol=1.0e-6, rtol=0.0)
    expected_next = torch.tensor(
        [8.5 + 0.5 * (1.0 - 0.0), 2.125 + 0.25 * (1.0 - 0.75)],
        dtype=torch.float64,
    )
    assert torch.equal(rope.phase(torch.tensor([18.0]))[0], expected_next)


def test_query_gain_is_query_only_and_key_gain_is_exactly_one() -> None:
    profile = _profile(
        native_length=4,
        inv_freq=[0.6, 0.2],
        head_dim=6,
        rotary_dim=4,
        gain=0.1,
    )
    rope = TargetFreeRoPE(profile)
    positions = torch.tensor([[4, 5]])
    query = torch.ones((1, 1, 2, 6), dtype=torch.float64)
    key = torch.ones_like(query)
    observed_query, observed_key = rope.apply_qk(query, key, positions)
    no_gain_key = rope.rotate(key, positions, query=False)
    no_gain_query = rope.rotate(query, positions, query=False)
    assert torch.equal(observed_key, no_gain_key)
    expected_gain = rope.query_gain(positions).to(dtype=query.dtype).reshape(1, 1, 2, 1)
    assert torch.allclose(
        observed_query[..., : profile.rotary_dim],
        no_gain_query[..., : profile.rotary_dim] * expected_gain,
    )
    assert torch.allclose(
        observed_query[..., 4:],
        query[..., 4:] * expected_gain,
    )
    assert torch.equal(observed_key[..., 4:], key[..., 4:])


def test_external_native_phase_provider_is_required_and_executed() -> None:
    native_phase = torch.arange(10, dtype=torch.float64)[:, None] * torch.tensor(
        [0.7, 0.11], dtype=torch.float64
    )
    profile = ModelRoPEProfile.from_native(
        [0.7, 0.11],
        native_context_length=9,
        native_context_length_source="external fixture",
        head_dim=4,
        movement_coefficients=[0.0, 0.5],
        native_boundary_slope=[0.7, 0.11],
        native_phase=native_phase,
        native_phase_hash_scope="fixture positions 0..9",
        gain_coefficient=0.0,
        gain_coefficient_source="disabled fixture",
    )
    with pytest.raises(ValueError, match="requires a runtime provider"):
        TargetFreeRoPE(profile)

    calls: list[torch.Tensor] = []

    def provider(positions: torch.Tensor) -> torch.Tensor:
        calls.append(positions.detach().clone())
        return torch.stack((positions * 0.7, positions.square() * 0.01), dim=-1)

    rope = TargetFreeRoPE(profile, native_phase_provider=provider)
    observed = rope.phase(torch.tensor([2.0, 9.0, 10.0]))
    assert calls
    assert observed[0, 1] == pytest.approx(0.04)
    assert observed[1, 1] == pytest.approx(0.81)
    assert observed[2, 1] == pytest.approx(0.81 + 0.5 * 0.11)


def test_olmo_installer_applies_gain_to_complete_query_only() -> None:
    class DummyRotary(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("inv_freq", torch.tensor([0.6, 0.2]))
            self.attention_scaling = 1.0

    class DummyAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_norm = torch.nn.Identity()

    class DummyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = DummyAttention()

    class DummyInner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rotary_emb = DummyRotary()
            self.layers = torch.nn.ModuleList([DummyLayer(), DummyLayer()])

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = DummyInner()
            self.config = SimpleNamespace(
                model_type="olmo2",
                hidden_size=4,
                num_attention_heads=1,
                num_key_value_heads=1,
                num_hidden_layers=2,
            )

    profile = _profile(
        native_length=4,
        inv_freq=[0.6, 0.2],
        head_dim=4,
        rotary_dim=4,
        gain=0.1,
    )
    model = DummyModel()
    receipt = install_target_free_olmo2(model, profile)
    positions = torch.tensor([[7]])
    _ = model.model.rotary_emb(torch.ones(1), positions)
    query = torch.ones((1, 1, 4))
    observed = model.model.layers[0].self_attn.q_norm(query)
    expected = model.model.rotary_emb.query_gain(positions).to(query.dtype)[..., None]
    assert torch.allclose(observed, query * expected)
    assert receipt["query_gain_hooks"] == 2
    assert receipt["parameter_count_before"] == receipt["parameter_count_after"]


def test_cached_key_phase_is_position_local_and_stable() -> None:
    profile = _profile(
        native_length=4,
        inv_freq=[0.6, 0.2],
        head_dim=6,
        rotary_dim=4,
        gain=0.1,
    )
    rope = TargetFreeRoPE(profile)
    prefix_positions = torch.tensor([[0, 1, 2, 3]])
    future_positions = torch.tensor([[4, 5, 6]])
    prefix_key = torch.arange(24, dtype=torch.float64).reshape(1, 1, 4, 6)
    prefix_before = rope.rotate(prefix_key, prefix_positions, query=False)
    _ = rope.phase(future_positions)
    prefix_after = rope.rotate(prefix_key, prefix_positions, query=False)
    assert torch.equal(prefix_before, prefix_after)


def test_interface_has_no_target_length_or_request_budget_and_no_parameters() -> None:
    profile = _profile(
        native_length=4,
        inv_freq=[0.6, 0.2],
        head_dim=6,
        rotary_dim=4,
    )
    rope = TargetFreeRoPE(profile)
    assert list(rope.parameters()) == []
    assert "target_length" not in inspect.signature(rope.forward).parameters
    assert "request_budget" not in inspect.signature(rope.forward).parameters
    assert "target_length" not in inspect.signature(apply_target_free_qk).parameters
    assert "request_budget" not in inspect.signature(apply_target_free_qk).parameters
    assert rope.receipt()["requires_target_length"] is False
    assert rope.receipt()["requires_request_budget"] is False


def test_profile_records_model_identity_and_gain_source() -> None:
    profile = _profile(
        native_length=32768,
        inv_freq=[0.9, 0.3],
        head_dim=8,
        rotary_dim=4,
        gain=0.1,
    )
    receipt = profile.as_dict()
    assert receipt["native_context_length"] == 32768
    assert receipt["native_context_length_source"] == "fixture:32768"
    assert receipt["head_dim"] == 8
    assert receipt["rotary_dim"] == 4
    assert receipt["pair_count"] == 2
    assert len(receipt["native_inv_freq_sha256"]) == 64
    assert len(receipt["native_phase_sha256"]) == 64
    assert receipt["gain_coefficient"] == pytest.approx(0.1)
    assert receipt["gain_coefficient_source"] == "fixture-only coefficient"
    assert receipt["learned_parameters"] == 0


def test_current_olmo_movement_and_existing_tables_remain_reproducible() -> None:
    native = native_endpoint_inv_freq()
    support, weight = causal_distance_measure()
    uniqueness = conditional_pair_uniqueness(native, support, weight)
    normalized = (uniqueness - uniqueness.min()) / (
        uniqueness.max() - uniqueness.min()
    )
    movement = (1.0 - normalized) ** 2
    profile = ModelRoPEProfile.from_native(
        native,
        native_context_length=4096,
        native_context_length_source="current OLMo-2 profile config",
        head_dim=native.size * 2,
        rotary_dim=native.size * 2,
        movement_coefficients=movement,
        native_rope_config={
            "model_type": "olmo2",
            "rope_type": "default",
            "rope_theta_source": "downloaded config",
        },
        native_scaling_config={"type": "native", "attention_scaling": 1.0},
        gain_coefficient=0.1,
        gain_coefficient_source="current OLMo experiment profile only",
    )
    assert profile.native_inv_freq_sha256 == (
        "dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34"
    )
    assert float32_tensor_sha256(movement) == (
        "bc02b5248ba034f37eccfaeb38debec2eba5bad859a2a41b9a97c0b8fedcde7c"
    )
    assert profile.effective_boundary_slope_sha256 == (
        "e1b4ece7568b3a5438854f0b87d7b291e95b911ef5f6b2f15dbb6aa2eb670cb1"
    )
    tables = build_default_tables()
    for factor, expected_hash in EXPECTED_DEFAULT_HASHES.items():
        table = uniqueness_budgeted_table(native, uniqueness, factor=factor)
        assert float32_tensor_sha256(table) == expected_hash
        assert float32_tensor_sha256(tables[factor]) == expected_hash
