from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest
from torch import nn
from transformers.cache_utils import DynamicCache

from rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.phase_adarope import (
    PhaseAdaRoPE,
    PhaseAdaRoPEAttention,
    load_phase_adarope_components,
    load_phase_adarope_state,
    phase_adarope_receipt,
    save_phase_adarope_state,
)


class TinyConfig:
    hidden_size = 16
    num_attention_heads = 2
    num_key_value_heads = 2
    _attn_implementation = "eager"


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = TinyConfig()
        self.layer_idx = 0
        self.head_dim = 8
        self.num_key_value_groups = 1
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.k_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)
        self.o_proj = nn.Linear(16, 16, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()


def bank(mode: str = "joint", target_name: str = "phase_chord") -> PhaseAdaRoPE:
    native = torch.logspace(0.0, -3.0, 64)
    target = native.clone()
    target[1] = native[1] * 0.9
    return PhaseAdaRoPE(native, target, target_name=target_name, mode=mode, num_layers=16, num_heads=16, head_dim=128)


def test_native_zero_and_control_table_are_exact() -> None:
    value = bank("scale_only")
    assert torch.equal(value.frequency_table(0), value.native_inv_freq.view(1, -1).expand(16, 64))
    assert torch.equal(value.alpha_projected(), torch.zeros_like(value.alpha))
    assert torch.allclose(value.temperature(0, 4096, l_ref=4096), torch.ones(16))


def test_temperature_is_exact_identity_at_reference_length_after_training() -> None:
    value = bank("scale_only")
    with torch.no_grad():
        value.raw_beta.fill_(8.0)
        value.raw_gamma.fill_(-8.0)
    scale = value.temperature(0, 4096, l_ref=4096)
    assert torch.equal(scale, torch.ones_like(scale))


def test_temperature_is_bounded_and_has_8k_16k_gradients() -> None:
    value = bank("scale_only")
    with torch.no_grad():
        value.raw_beta[:8].fill_(1.0)
        value.raw_beta[8:].fill_(-1.0)
        value.raw_gamma.fill_(0.0)
    scale_8k = value.temperature(0, 8192, l_ref=4096)
    scale_16k = value.temperature(0, 16384, l_ref=4096)
    assert float(scale_8k.detach().min()) >= 0.25 and float(scale_16k.detach().min()) >= 0.25
    assert float(scale_8k.detach().max()) <= 4.0 and float(scale_16k.detach().max()) <= 4.0
    (scale_8k.sum() + scale_16k.sum()).backward()
    assert torch.count_nonzero(value.raw_beta.grad).item() == 16
    assert torch.count_nonzero(value.raw_gamma.grad).item() == 16


def test_alpha_projection_monotonic_endpoints_and_target_direction() -> None:
    value = bank("freq_only")
    with torch.no_grad():
        value.alpha[0, 0] = -1.0
        value.alpha[0, 1] = 1.5
    value.project_parameters()
    realized = value.frequency_table(0)[0]
    assert torch.all((value.alpha_projected() >= 0) & (value.alpha_projected() <= 1))
    assert torch.equal(realized[[0, -1]], value.native_inv_freq[[0, -1]])
    assert torch.all(realized[:-1] > realized[1:])
    assert torch.equal(realized, value.native_inv_freq)
    realized_target = value.frequency_table(0)[1]
    assert torch.equal(realized_target, value.target_inv_freq)


def test_adascale_formula_and_independent_parameter_groups() -> None:
    value = bank("joint")
    with torch.no_grad():
        value.raw_beta[0].fill_(0.5)
        value.raw_gamma[0].fill_(0.5)
    x = torch.log(torch.tensor(8192.0 / 4096.0))
    expected = torch.exp(torch.tensor(0.5) * x.pow(1.0 + torch.nn.functional.softplus(torch.tensor(0.5))))
    assert torch.allclose(value.temperature(0, 8192, l_ref=4096), expected.expand(16))
    groups = value.parameter_groups(frequency_lr=1e-2, temperature_lr=3e-3)
    assert {group["name"] for group in groups} == {"adarope_frequency", "adarope_temperature"}
    assert {group["lr"] for group in groups} == {1e-2, 3e-3}


def test_zero_point_keeps_native_forward_but_frequency_gradient() -> None:
    value = bank("freq_only")
    loss = value.frequency_table(0).sum()
    loss.backward()
    assert value.alpha.grad is not None
    assert torch.isfinite(value.alpha.grad).all()
    assert torch.count_nonzero(value.alpha.grad[0]).item() == 16


def test_temperature_gradient_is_finite_and_nonzero() -> None:
    value = bank("scale_only")
    with torch.no_grad():
        value.raw_beta.fill_(0.5)
    loss = value.temperature(0, 8192, l_ref=4096).sum()
    loss.backward()
    assert value.raw_beta.grad is not None and value.raw_gamma.grad is not None
    assert torch.isfinite(value.raw_beta.grad).all()
    assert torch.isfinite(value.raw_gamma.grad).all()
    assert torch.count_nonzero(value.raw_beta.grad[0]).item() == 16
    assert torch.count_nonzero(value.raw_gamma.grad[0]).item() == 16


def test_scale_only_alpha_zero_matches_native_eager_path() -> None:
    torch.manual_seed(10)
    native = torch.logspace(0.0, -3.0, 64)
    target = native.clone()
    target[1] = native[1] * 0.9
    value = PhaseAdaRoPE(
        native,
        target,
        target_name="phase_chord",
        mode="scale_only",
        num_layers=16,
        num_heads=16,
        head_dim=128,
    )
    attention = TinyAttention()
    attention.config.hidden_size = 2048
    attention.config.num_attention_heads = 16
    attention.config.num_key_value_heads = 16
    attention.head_dim = 128
    attention.q_proj = nn.Linear(2048, 2048, bias=False)
    attention.k_proj = nn.Linear(2048, 2048, bias=False)
    attention.v_proj = nn.Linear(2048, 2048, bias=False)
    attention.o_proj = nn.Linear(2048, 2048, bias=False)
    wrapped = PhaseAdaRoPEAttention(attention, value, strict=True)
    assert wrapped.q_proj is attention.q_proj
    assert wrapped.k_proj is attention.k_proj
    assert wrapped.v_proj is attention.v_proj
    assert wrapped.o_proj is attention.o_proj
    projection_keys = {key for key in attention.state_dict() if key.startswith(("q_proj.", "k_proj.", "v_proj.", "o_proj."))}
    wrapper_projection_keys = {key for key in wrapped.state_dict() if key.startswith(("q_proj.", "k_proj.", "v_proj.", "o_proj."))}
    assert wrapper_projection_keys == projection_keys
    assert not any(key.startswith("native_attention.") for key in wrapped.state_dict())
    hidden = torch.randn(1, 3, 2048)
    positions = torch.arange(3)
    actual, _ = wrapped(hidden, None, None, cache_position=positions, phase_context_budget=4096)
    shape = (1, 3, 16, 128)
    q = attention.q_proj(hidden).view(shape).transpose(1, 2)
    k = attention.k_proj(hidden).view(shape).transpose(1, 2)
    v = attention.v_proj(hidden).view(shape).transpose(1, 2)
    cos, sin = value.cos_sin(0, positions)
    pairs = 64
    q = torch.cat((q[..., :pairs] * cos.unsqueeze(0) - q[..., pairs:] * sin.unsqueeze(0), q[..., :pairs] * sin.unsqueeze(0) + q[..., pairs:] * cos.unsqueeze(0)), dim=-1)
    k = torch.cat((k[..., :pairs] * cos.unsqueeze(0) - k[..., pairs:] * sin.unsqueeze(0), k[..., :pairs] * sin.unsqueeze(0) + k[..., pairs:] * cos.unsqueeze(0)), dim=-1)
    weights = torch.softmax(torch.matmul(q, k.transpose(2, 3)) * attention.scaling, dim=-1)
    expected = attention.o_proj(torch.matmul(weights, v).transpose(1, 2).reshape(1, 3, 2048))
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_fake_attention_dynamic_cache_and_batch_positions() -> None:
    torch.manual_seed(11)
    native = torch.logspace(0.0, -3.0, 64)
    target = native.clone()
    target[1] = native[1] * 0.9
    tiny_bank = PhaseAdaRoPE(
        native,
        target,
        target_name="matched_exponential",
        mode="scale_only",
        num_layers=16,
        num_heads=16,
        head_dim=128,
    )
    # Exercise the registered D128 shape directly with a small hidden fake.
    attention = TinyAttention()
    attention.config.hidden_size = 2048
    attention.config.num_attention_heads = 16
    attention.config.num_key_value_heads = 16
    attention.head_dim = 128
    attention.q_proj = nn.Linear(2048, 2048, bias=False)
    attention.k_proj = nn.Linear(2048, 2048, bias=False)
    attention.v_proj = nn.Linear(2048, 2048, bias=False)
    attention.o_proj = nn.Linear(2048, 2048, bias=False)
    wrapped = PhaseAdaRoPEAttention(attention, tiny_bank, strict=True)
    cache = DynamicCache()
    first, _ = wrapped(torch.randn(1, 2, 2048), None, None, cache, torch.arange(2), phase_context_budget=4096)
    second, _ = wrapped(torch.randn(1, 1, 2048), None, None, cache, torch.tensor([2]), phase_context_budget=4096)
    assert first.shape == (1, 2, 2048)
    assert second.shape == (1, 1, 2048)
    assert cache.get_seq_length(0) == 3
    assert cache.layers[0].keys.shape[-1] == 128


def test_batch_two_broadcast_positions_and_bfloat16_output() -> None:
    native = torch.logspace(0.0, -3.0, 64)
    target = native.clone()
    target[1] = native[1] * 0.9
    value = PhaseAdaRoPE(native, target, target_name="phase_chord", mode="scale_only", num_layers=16, num_heads=16, head_dim=128)
    with torch.no_grad():
        value.raw_beta.fill_(0.5)
        value.raw_gamma.fill_(0.0)
    attention = TinyAttention()
    attention.config.hidden_size = 2048
    attention.config.num_attention_heads = 16
    attention.config.num_key_value_heads = 16
    attention.head_dim = 128
    attention.q_proj = nn.Linear(2048, 2048, bias=False)
    attention.k_proj = nn.Linear(2048, 2048, bias=False)
    attention.v_proj = nn.Linear(2048, 2048, bias=False)
    attention.o_proj = nn.Linear(2048, 2048, bias=False)
    attention = attention.to(dtype=torch.bfloat16)
    wrapped = PhaseAdaRoPEAttention(attention, value, l_ref=2, strict=True)
    hidden = torch.randn(2, 2, 2048, dtype=torch.bfloat16)
    output, _ = wrapped(hidden, None, None, position_ids=torch.tensor([[0, 1]]), phase_context_budget=4096)
    assert output.shape == hidden.shape
    assert output.dtype == torch.bfloat16
    output_specific, _ = wrapped(hidden, None, None, position_ids=torch.tensor([[0, 1], [0, 2]]), phase_context_budget=4096)
    assert output_specific.shape == hidden.shape
    assert output_specific.dtype == torch.bfloat16


def test_dynamic_cache_full_vs_token_decode_parity() -> None:
    torch.manual_seed(12)
    native = torch.logspace(0.0, -3.0, 64)
    target = native.clone()
    target[1] = native[1] * 0.9
    value = PhaseAdaRoPE(native, target, target_name="phase_chord", mode="scale_only", num_layers=16, num_heads=16, head_dim=128)
    with torch.no_grad():
        value.raw_beta.fill_(0.5)
        value.raw_gamma.fill_(0.0)
    attention = TinyAttention()
    attention.config.hidden_size = 2048
    attention.config.num_attention_heads = 16
    attention.config.num_key_value_heads = 16
    attention.head_dim = 128
    attention.q_proj = nn.Linear(2048, 2048, bias=False)
    attention.k_proj = nn.Linear(2048, 2048, bias=False)
    attention.v_proj = nn.Linear(2048, 2048, bias=False)
    attention.o_proj = nn.Linear(2048, 2048, bias=False)
    wrapped = PhaseAdaRoPEAttention(attention, value, l_ref=2, strict=True)
    tokens = torch.randn(1, 3, 2048)
    causal3 = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1).view(1, 1, 3, 3)
    full, _ = wrapped(tokens, None, causal3, position_ids=torch.arange(3), phase_context_budget=8)
    cache = DynamicCache()
    causal2 = torch.triu(torch.full((2, 2), float("-inf")), diagonal=1).view(1, 1, 2, 2)
    _, _ = wrapped(tokens[:, :2], None, causal2, cache, torch.arange(2), phase_context_budget=8)
    last, _ = wrapped(tokens[:, 2:], None, None, cache, torch.tensor([2]), phase_context_budget=8)
    assert torch.allclose(full[:, -1], last[:, 0], atol=1e-5, rtol=1e-5)


def test_dynamic_cache_full_vs_token_decode_parity_batch_two() -> None:
    torch.manual_seed(13)
    native = torch.logspace(0.0, -3.0, 64)
    target = native.clone()
    target[1] = native[1] * 0.9
    value = PhaseAdaRoPE(native, target, target_name="phase_chord", mode="scale_only", num_layers=16, num_heads=16, head_dim=128)
    with torch.no_grad():
        value.raw_beta.fill_(0.5)
        value.raw_gamma.fill_(0.0)
    attention = TinyAttention()
    attention.config.hidden_size = 2048
    attention.config.num_attention_heads = 16
    attention.config.num_key_value_heads = 16
    attention.head_dim = 128
    attention.q_proj = nn.Linear(2048, 2048, bias=False)
    attention.k_proj = nn.Linear(2048, 2048, bias=False)
    attention.v_proj = nn.Linear(2048, 2048, bias=False)
    attention.o_proj = nn.Linear(2048, 2048, bias=False)
    wrapped = PhaseAdaRoPEAttention(attention, value, l_ref=2, strict=True)
    tokens = torch.randn(2, 3, 2048)
    causal3 = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1).view(1, 1, 3, 3)
    full, _ = wrapped(tokens, None, causal3, position_ids=torch.arange(3), phase_context_budget=8)
    cache = DynamicCache()
    causal2 = torch.triu(torch.full((2, 2), float("-inf")), diagonal=1).view(1, 1, 2, 2)
    _, _ = wrapped(tokens[:, :2], None, causal2, cache, torch.arange(2), phase_context_budget=8)
    last, _ = wrapped(tokens[:, 2:], None, None, cache, torch.tensor([2]), phase_context_budget=8)
    assert torch.allclose(full[:, -1], last[:, 0], atol=1e-5, rtol=1e-5)


def test_tiny_hf_olmo2_attention_budget_parity_batch_one_and_two() -> None:
    from transformers import Olmo2Config
    from transformers.models.olmo2.modeling_olmo2 import Olmo2Attention

    config = Olmo2Config(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_hidden_layers=1,
        intermediate_size=4096,
        max_position_embeddings=4096,
        rope_theta=500_000,
    )
    config._attn_implementation = "eager"
    attention = Olmo2Attention(config, layer_idx=0)
    native = torch.logspace(0.0, -3.0, 64)
    target = native.clone()
    target[1] = native[1] * 0.9
    value = PhaseAdaRoPE(native, target, target_name="phase_chord", mode="scale_only")
    with torch.no_grad():
        value.raw_beta.fill_(0.5)
        value.raw_gamma.fill_(0.0)
    wrapped = PhaseAdaRoPEAttention(attention, value, l_ref=2, strict=True).eval()
    for batch in (1, 2):
        torch.manual_seed(20 + batch)
        tokens = torch.randn(batch, 3, 2048)
        full_positions = torch.arange(3).view(1, 3).expand(batch, 3)
        mask3 = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1).view(1, 1, 3, 3)
        full, _ = wrapped(tokens, None, mask3, position_ids=full_positions, phase_context_budget=8)
        cache = DynamicCache()
        mask2 = torch.triu(torch.full((2, 2), float("-inf")), diagonal=1).view(1, 1, 2, 2)
        prefix_positions = torch.arange(2).view(1, 2).expand(batch, 2)
        wrapped(tokens[:, :2], None, mask2, cache, prefix_positions, phase_context_budget=8)
        last_positions = torch.full((batch, 1), 2)
        last, _ = wrapped(tokens[:, 2:], None, None, cache, last_positions, phase_context_budget=8)
        assert torch.allclose(full[:, -1], last[:, 0], atol=1e-5, rtol=1e-5)


def test_phase_context_budget_is_required_and_bounded() -> None:
    value = bank("scale_only")
    attention = TinyAttention()
    attention.config.hidden_size = 2048
    attention.config.num_attention_heads = 16
    attention.config.num_key_value_heads = 16
    attention.head_dim = 128
    attention.q_proj = nn.Linear(2048, 2048, bias=False)
    attention.k_proj = nn.Linear(2048, 2048, bias=False)
    attention.v_proj = nn.Linear(2048, 2048, bias=False)
    attention.o_proj = nn.Linear(2048, 2048, bias=False)
    wrapped = PhaseAdaRoPEAttention(attention, value, strict=True)
    hidden = torch.randn(1, 2, 2048)
    with pytest.raises(ValueError, match="phase_context_budget is required"):
        wrapped(hidden, None, None, position_ids=torch.arange(2))
    with pytest.raises(ValueError, match="must be >="):
        wrapped(hidden, None, None, position_ids=torch.arange(2), phase_context_budget=1)
    with pytest.raises(ValueError, match="exceeds registered maximum"):
        wrapped(hidden, None, None, position_ids=torch.arange(2), phase_context_budget=16_385)
    cache = DynamicCache()
    wrapped(hidden, None, None, cache, torch.arange(2), phase_context_budget=8)
    assert cache._phase_context_budget == 8
    with pytest.raises(ValueError, match="cannot change within a cache"):
        wrapped(hidden[:, :1], None, None, cache, torch.tensor([2]), phase_context_budget=9)


def test_state_roundtrip_and_strict_metadata(tmp_path: Path) -> None:
    original = bank("joint")
    with torch.no_grad():
        original.alpha[0, 0] = 0.5
        original.raw_beta[0, 0] = 0.1
        original.raw_gamma[0, 0] = 0.2
    path = tmp_path / "adarope_state.pt"
    save_phase_adarope_state(path, original, metadata={"arm": "phase_chord"})
    restored = bank("joint")
    metadata = load_phase_adarope_state(path, restored)
    assert metadata["arm"] == "phase_chord"
    assert torch.equal(restored.alpha, original.alpha)
    assert torch.equal(restored.raw_beta, original.raw_beta)
    assert torch.equal(restored.raw_gamma, original.raw_gamma)
    assert phase_adarope_receipt(restored)["target_name"] == "phase_chord"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["state_sha256"]
    assert json.loads(json.dumps(payload["metadata"]))["method_id"] == "olmo2_phase_adarope_headwise_v1"


def test_moment_matched_control_target_identity_roundtrip(tmp_path: Path) -> None:
    source = bank("freq_only", "moment_matched_control")
    with torch.no_grad():
        source.alpha[0, 0] = 0.25
    path = tmp_path / "moment-control.pt"
    save_phase_adarope_state(path, source)
    restored = bank("freq_only", "moment_matched_control")
    metadata = load_phase_adarope_state(path, restored)
    assert metadata["target_name"] == "moment_matched_control"
    assert torch.equal(restored.alpha, source.alpha)
    try:
        load_phase_adarope_state(path, bank("freq_only", "phase_chord"))
    except ValueError as error:
        assert "target_name" in str(error)
    else:
        raise AssertionError("moment control sidecar must not load as phase-chord")


def test_component_loads_are_strict_and_partial(tmp_path: Path) -> None:
    source = bank("freq_only")
    with torch.no_grad():
        source.alpha[0, 0] = 0.4
    frequency_path = tmp_path / "frequency.pt"
    save_phase_adarope_state(frequency_path, source)
    destination = bank("joint")
    with torch.no_grad():
        destination.alpha.fill_(0.2)
        destination.raw_beta.fill_(0.3)
        destination.raw_gamma.fill_(0.1)
    old_beta = destination.raw_beta.detach().clone()
    old_gamma = destination.raw_gamma.detach().clone()
    load_phase_adarope_components(frequency_path, destination, "alpha")
    assert torch.equal(destination.alpha, source.alpha)
    assert torch.equal(destination.raw_beta, old_beta)
    assert torch.equal(destination.raw_gamma, old_gamma)

    temperature_source = bank("scale_only")
    with torch.no_grad():
        temperature_source.raw_beta.fill_(0.2)
        temperature_source.raw_gamma.fill_(0.15)
    temperature_path = tmp_path / "temperature.pt"
    save_phase_adarope_state(temperature_path, temperature_source)
    old_alpha = destination.alpha.detach().clone()
    load_phase_adarope_components(temperature_path, destination, "temperature")
    assert torch.equal(destination.alpha, old_alpha)
    assert torch.equal(destination.raw_beta, temperature_source.raw_beta)
    assert torch.equal(destination.raw_gamma, temperature_source.raw_gamma)

    joint_path = tmp_path / "joint.pt"
    save_phase_adarope_state(joint_path, bank("joint"))
    try:
        load_phase_adarope_components(joint_path, destination, "temperature")
    except ValueError as error:
        assert "scale_only" in str(error)
    else:
        raise AssertionError("joint sidecar must not load as temperature")


def test_component_load_rejects_hash_tamper_and_target_mismatch(tmp_path: Path) -> None:
    source = bank("freq_only")
    path = tmp_path / "tampered.pt"
    save_phase_adarope_state(path, source)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["state"]["alpha"][0, 0] = 0.3
    torch.save(payload, path)
    try:
        load_phase_adarope_components(path, bank("joint"), "alpha")
    except ValueError as error:
        assert "hash" in str(error)
    else:
        raise AssertionError("tampered sidecar must be rejected")

    other_target = torch.logspace(0.0, -3.0, 64)
    other_target[2] *= 0.9
    mismatch = PhaseAdaRoPE(source.native_inv_freq, other_target, target_name="matched_exponential", mode="joint")
    mismatch_path = tmp_path / "mismatch.pt"
    save_phase_adarope_state(mismatch_path, mismatch)
    try:
        load_phase_adarope_components(mismatch_path, bank("joint"), "alpha")
    except ValueError as error:
        assert "target identity" in str(error)
    else:
        raise AssertionError("target-mismatched alpha sidecar must be rejected")

    other_native = torch.logspace(0.0, -2.9, 64)
    other_scale = PhaseAdaRoPE(other_native, other_native.clone(), target_name="matched_exponential", mode="scale_only")
    other_scale_path = tmp_path / "other-native-temperature.pt"
    save_phase_adarope_state(other_scale_path, other_scale)
    try:
        load_phase_adarope_components(other_scale_path, bank("joint"), "temperature")
    except ValueError as error:
        assert "native_inv_freq_sha256" in str(error)
    else:
        raise AssertionError("cross-Native temperature sidecar must be rejected")


def test_modes_have_independent_trainable_scopes() -> None:
    assert [name for name, parameter in bank("control").named_parameters() if parameter.requires_grad] == []
    assert [name for name, parameter in bank("scale_only").named_parameters() if parameter.requires_grad] == ["raw_beta", "raw_gamma"]
    assert [name for name, parameter in bank("freq_only").named_parameters() if parameter.requires_grad] == ["alpha"]
    assert {name for name, parameter in bank("joint").named_parameters() if parameter.requires_grad} == {"alpha", "raw_beta", "raw_gamma"}
