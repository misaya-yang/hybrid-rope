from __future__ import annotations

import copy
from dataclasses import asdict
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import Olmo2Config, Olmo2ForCausalLM

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    install_adaptation,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.far_only_evq_residual import (
    METHOD_ID,
    FarPassChordAttention,
    FarPassChordConfig,
    far_pass_frequency_receipt,
    far_pass_inv_freq,
    far_only_trainable_named_parameters,
    install_far_pass_chord_residual,
    load_far_only_adapter,
    route_for_budget,
    save_far_only_adapter,
    set_far_pass_chord_route,
    validate_far_only_installation,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_4k_far_only_evq_residual import (
    expected_shape,
    supervised_loss,
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


def method_config() -> FarPassChordConfig:
    return FarPassChordConfig(
        projection_rank=4,
        residual_pairs=8,
        initialization_seed=1234,
    )


def content_method_config() -> FarPassChordConfig:
    return FarPassChordConfig(
        projection_rank=4,
        residual_pairs=8,
        initialization_seed=1234,
        content_value_dim=32,
        content_projection_rank=4,
        initial_content_gain=0.1,
    )


def installed_pair() -> tuple[Olmo2ForCausalLM, Olmo2ForCausalLM]:
    torch.manual_seed(7)
    native = Olmo2ForCausalLM(tiny_config()).eval()
    wrapped = copy.deepcopy(native).eval()
    install_far_pass_chord_residual(
        wrapped,
        method_config(),
        strict_model_contract=False,
    )
    return native, wrapped


def test_short_route_delegates_to_bitwise_native_path() -> None:
    native, wrapped = installed_pair()
    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    set_far_pass_chord_route(wrapped, False)
    with torch.no_grad():
        expected = native(tokens, use_cache=False).logits
        observed = wrapped(tokens, use_cache=False).logits
    assert torch.equal(observed, expected)


def test_short_route_preserves_a_loaded_parent_lora_path() -> None:
    torch.manual_seed(17)
    parent = Olmo2ForCausalLM(tiny_config()).eval()
    readout = install_adaptation(
        parent, "qk_answer", rank=4, alpha=8.0
    )
    assert readout is None
    with torch.no_grad():
        for name, parameter in parent.named_parameters():
            if name.endswith(".b") and parameter.requires_grad:
                parameter.normal_(mean=0.0, std=0.02)
    wrapped = copy.deepcopy(parent).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        expected = parent(tokens, use_cache=False).logits
    install_far_pass_chord_residual(
        wrapped,
        method_config(),
        strict_model_contract=False,
    )
    set_far_pass_chord_route(wrapped, False)
    with torch.no_grad():
        observed = wrapped(tokens, use_cache=False).logits
    assert torch.equal(observed, expected)


def test_active_path_is_native_when_query_gate_is_zero_up_to_roundoff() -> None:
    native, wrapped = installed_pair()
    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    positions = torch.arange(tokens.shape[1])[None, :]
    set_far_pass_chord_route(wrapped, True)
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
    set_far_pass_chord_route(wrapped, True)
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


def test_content_transport_uses_added_values_and_trains_v_o() -> None:
    torch.manual_seed(41)
    native = Olmo2ForCausalLM(tiny_config()).eval()
    wrapped = copy.deepcopy(native).train()
    receipt = install_far_pass_chord_residual(
        wrapped,
        content_method_config(),
        strict_model_contract=False,
    )
    assert receipt["content_transport"] == (
        "learned_augmented_value_and_long_query_output"
    )
    set_far_pass_chord_route(wrapped, True)
    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    positions = torch.tensor(
        [[0, 1, 2, 4_096, 4_097]], dtype=torch.long
    )
    logits = wrapped(
        tokens,
        position_ids=positions,
        use_cache=False,
    ).logits
    loss = F.cross_entropy(
        logits[:, -1, :], torch.tensor([6], dtype=torch.long)
    )
    loss.backward()
    nonzero = {
        name
        for name, parameter in far_only_trainable_named_parameters(wrapped)
        if parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad))
    }
    assert any(name.endswith(".residual_v.b") for name in nonzero)
    assert any(name.endswith(".residual_o.b") for name in nonzero)
    assert any(name.endswith(".raw_content_gain") for name in nonzero)


def test_weighted_loss_balances_first_retrieval_token_and_rest() -> None:
    class CrossEntropy(torch.nn.Module):
        def forward(
            self,
            weight: torch.Tensor,
            hidden: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
            return F.cross_entropy(F.linear(hidden, weight), labels)

    torch.manual_seed(43)
    hidden = torch.randn(2, 4, 3)
    weight = torch.randn(7, 3)
    labels = torch.tensor(
        [[-100, 2, 3, 4], [-100, -100, 5, 6]], dtype=torch.long
    )
    observed = supervised_loss(
        loss_module=CrossEntropy(),
        lm_head_weight=weight,
        hidden=hidden,
        labels=labels,
        first_token_weight=0.5,
    )
    first_logits = torch.stack((hidden[0, 1], hidden[1, 2])) @ weight.T
    rest_logits = torch.stack((hidden[0, 2], hidden[0, 3], hidden[1, 3])) @ weight.T
    expected = 0.5 * F.cross_entropy(
        first_logits, torch.tensor([2, 5])
    ) + 0.5 * F.cross_entropy(
        rest_logits, torch.tensor([3, 4, 6])
    )
    torch.testing.assert_close(observed, expected)


def test_active_dynamic_cache_uses_augmented_width() -> None:
    _, wrapped = installed_pair()
    wrapped.eval()
    set_far_pass_chord_route(wrapped, True)
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
    set_far_pass_chord_route(wrapped, True)
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
    set_far_pass_chord_route(wrapped, True)
    with torch.no_grad():
        first = wrapped(
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            position_ids=torch.tensor(
                [[4_096, 4_097, 4_098]], dtype=torch.long
            ),
            use_cache=True,
            return_dict=True,
        )
    set_far_pass_chord_route(wrapped, False)
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
        if isinstance(module, FarPassChordAttention)
    )
    assert route_for_budget(wrapped, 4_097) is True
    assert all(
        module.route_enabled
        for module in wrapped.modules()
        if isinstance(module, FarPassChordAttention)
    )


def test_registered_training_shapes_preserve_global_batch_eight() -> None:
    phase = expected_shape(SimpleNamespace(training_mode="phase_gap_4k"))
    continuous = expected_shape(
        SimpleNamespace(training_mode="continuous_8k")
    )
    assert phase == (4, 2)
    assert continuous == (2, 4)
    assert phase[0] * phase[1] == continuous[0] * continuous[1] == 8


def test_frequency_validation_fails_closed() -> None:
    _, wrapped = installed_pair()
    residual = next(
        module
        for module in wrapped.modules()
        if isinstance(module, FarPassChordAttention)
    )
    with torch.no_grad():
        residual.chord_inv_freq[1].add_(1.0)
    try:
        validate_far_only_installation(
            wrapped, strict_model_contract=False
        )
    except RuntimeError as error:
        assert "frequency drift" in str(error)
    else:
        raise AssertionError("tampered chord frequency was accepted")


def test_registered_far_pass_band_has_signal_and_selectivity() -> None:
    receipt = far_pass_frequency_receipt()
    assert receipt["pairs"] == 8
    assert receipt["augmented_head_dimension"] == 160
    assert receipt["maximum_phase_at_16k"] <= torch.pi + 1e-6
    assert 0.03 < receipt["mean_near_chord_0_4k"] < 0.04
    assert 0.50 < receipt["mean_far_chord_4k_16k"] < 0.55
    assert receipt["far_to_near_ratio"] > 15.0
    assert torch.equal(
        far_pass_inv_freq(),
        torch.tensor(receipt["inv_freq"], dtype=torch.float32),
    )


def test_I_minus_R_residual_is_zero_at_equal_positions() -> None:
    torch.manual_seed(29)
    query = torch.randn(8, 2, dtype=torch.float64)
    key = torch.randn(8, 2, dtype=torch.float64)
    theta = far_pass_inv_freq(dtype=torch.float64) * 7_777.0
    cosine = theta.cos()
    sine = theta.sin()

    def rotate(value: torch.Tensor) -> torch.Tensor:
        first, second = value[:, 0], value[:, 1]
        return torch.stack(
            (
                first * cosine - second * sine,
                first * sine + second * cosine,
            ),
            dim=-1,
        )

    residual = (query * key).sum() - (rotate(query) * rotate(key)).sum()
    assert abs(float(residual)) < 1e-12


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
