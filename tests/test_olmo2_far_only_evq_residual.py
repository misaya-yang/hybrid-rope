from __future__ import annotations

import copy
from dataclasses import asdict

import torch
import torch.nn.functional as F
from transformers import Olmo2Config, Olmo2ForCausalLM

from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.far_only_evq_residual import (
    METHOD_ID,
    FarOnlyEVQAttention,
    FarOnlyEVQConfig,
    far_only_trainable_named_parameters,
    install_far_only_evq_residual,
    load_far_only_adapter,
    route_for_budget,
    save_far_only_adapter,
    set_far_only_evq_route,
    validate_far_only_installation,
)


def tiny_config() -> Olmo2Config:
    config = Olmo2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=4_096,
        rope_theta=500_000.0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return config


def method_config() -> FarOnlyEVQConfig:
    return FarOnlyEVQConfig(
        projection_rank=4,
        residual_head_dim=32,
        initialization_seed=1234,
    )


def installed_pair() -> tuple[Olmo2ForCausalLM, Olmo2ForCausalLM]:
    torch.manual_seed(7)
    native = Olmo2ForCausalLM(tiny_config()).eval()
    wrapped = copy.deepcopy(native).eval()
    install_far_only_evq_residual(
        wrapped,
        method_config(),
        strict_model_contract=False,
    )
    return native, wrapped


def test_short_route_delegates_to_bitwise_native_path() -> None:
    native, wrapped = installed_pair()
    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    set_far_only_evq_route(wrapped, False)
    with torch.no_grad():
        expected = native(tokens, use_cache=False).logits
        observed = wrapped(tokens, use_cache=False).logits
    assert torch.equal(observed, expected)


def test_active_path_is_native_when_query_gate_is_zero_up_to_roundoff() -> None:
    native, wrapped = installed_pair()
    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    positions = torch.arange(tokens.shape[1])[None, :]
    set_far_only_evq_route(wrapped, True)
    with torch.no_grad():
        expected = native(
            tokens,
            position_ids=positions,
            use_cache=False,
        ).logits
        observed = wrapped(
            tokens,
            position_ids=positions,
            use_cache=False,
        ).logits
    torch.testing.assert_close(observed, expected, atol=2e-5, rtol=2e-5)


def test_far_positions_change_logits_and_train_residual() -> None:
    native, wrapped = installed_pair()
    wrapped.train()
    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    positions = torch.tensor(
        [[0, 1, 2, 4_096, 4_097]], dtype=torch.long
    )
    set_far_only_evq_route(wrapped, True)
    expected = native(
        tokens,
        position_ids=positions,
        use_cache=False,
    ).logits
    observed = wrapped(
        tokens,
        position_ids=positions,
        use_cache=False,
    ).logits
    assert not torch.equal(observed, expected)
    loss = F.cross_entropy(
        observed[:, -1, :],
        torch.tensor([6], dtype=torch.long),
    )
    loss.backward()
    named = far_only_trainable_named_parameters(wrapped)
    nonzero = {
        name
        for name, parameter in named
        if parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad))
    }
    assert any(name.endswith(".residual_q.b") for name in nonzero)
    assert any(name.endswith(".residual_k.b") for name in nonzero)
    assert any(name.endswith(".raw_logit_gain") for name in nonzero)


def test_active_dynamic_cache_uses_augmented_width() -> None:
    _, wrapped = installed_pair()
    wrapped.eval()
    set_far_only_evq_route(wrapped, True)
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    positions = torch.tensor(
        [[4_094, 4_095, 4_096, 4_097]], dtype=torch.long
    )
    with torch.no_grad():
        first = wrapped(
            prompt,
            position_ids=positions,
            use_cache=True,
            return_dict=True,
        )
        assert first.past_key_values.get_seq_length() == 4
        assert first.past_key_values.layers[0].keys.shape[-1] == 64
        assert first.past_key_values.layers[0].values.shape[-1] == 64
        second = wrapped(
            torch.tensor([[5]], dtype=torch.long),
            position_ids=torch.tensor([[4_098]], dtype=torch.long),
            cache_position=torch.tensor([4], dtype=torch.long),
            past_key_values=first.past_key_values,
            use_cache=True,
            return_dict=True,
        )
    assert second.past_key_values.get_seq_length() == 5
    assert torch.isfinite(second.logits).all()


def test_active_path_runs_through_transformers_sdpa_interface() -> None:
    _, wrapped = installed_pair()
    wrapped.config._attn_implementation = "sdpa"
    wrapped.eval()
    set_far_only_evq_route(wrapped, True)
    with torch.no_grad():
        output = wrapped(
            torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            position_ids=torch.tensor(
                [[4_096, 4_097, 4_098, 4_099]], dtype=torch.long
            ),
            use_cache=False,
            return_dict=True,
        )
    assert torch.isfinite(output.logits).all()


def test_route_cannot_change_after_cache_prefill() -> None:
    _, wrapped = installed_pair()
    wrapped.eval()
    set_far_only_evq_route(wrapped, True)
    with torch.no_grad():
        first = wrapped(
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            position_ids=torch.tensor(
                [[4_096, 4_097, 4_098]], dtype=torch.long
            ),
            use_cache=True,
            return_dict=True,
        )
    set_far_only_evq_route(wrapped, False)
    try:
        wrapped(
            torch.tensor([[4]], dtype=torch.long),
            position_ids=torch.tensor([[4_099]], dtype=torch.long),
            cache_position=torch.tensor([3], dtype=torch.long),
            past_key_values=first.past_key_values,
            use_cache=True,
            return_dict=True,
        )
    except RuntimeError as error:
        assert "augmented cache" in str(error)
    else:
        raise AssertionError("route change accepted an incompatible cache")


def test_route_for_budget_has_strict_4k_boundary() -> None:
    _, wrapped = installed_pair()
    assert route_for_budget(wrapped, 4_096) is False
    assert all(
        not module.route_enabled
        for module in wrapped.modules()
        if isinstance(module, FarOnlyEVQAttention)
    )
    assert route_for_budget(wrapped, 4_097) is True
    assert all(
        module.route_enabled
        for module in wrapped.modules()
        if isinstance(module, FarOnlyEVQAttention)
    )


def test_frequency_validation_fails_closed() -> None:
    _, wrapped = installed_pair()
    residual = next(
        module
        for module in wrapped.modules()
        if isinstance(module, FarOnlyEVQAttention)
    )
    with torch.no_grad():
        residual.evq_inv_freq[1].add_(1.0)
    try:
        validate_far_only_installation(
            wrapped, strict_model_contract=False
        )
    except RuntimeError as error:
        assert "frequency drift" in str(error)
    else:
        raise AssertionError("tampered EVQ frequency was accepted")


def test_adapter_round_trip(tmp_path) -> None:
    _, source = installed_pair()
    with torch.no_grad():
        for _, parameter in far_only_trainable_named_parameters(source):
            parameter.add_(0.125)
    metadata = {
        "method_id": METHOD_ID,
        "method_config": asdict(method_config()),
        "base_checkpoint_sha256": "test-checkpoint",
    }
    path = tmp_path / "adapter.pt"
    digest = save_far_only_adapter(path, source, metadata=metadata)
    assert len(digest) == 64

    _, target = installed_pair()
    loaded = load_far_only_adapter(
        path,
        target,
        expected_checkpoint_sha256="test-checkpoint",
    )
    assert loaded == metadata
    source_state = dict(far_only_trainable_named_parameters(source))
    target_state = dict(far_only_trainable_named_parameters(target))
    assert set(source_state) == set(target_state)
    for name in source_state:
        assert torch.equal(source_state[name], target_state[name])
