from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import Olmo2Config, Olmo2ForCausalLM
from transformers.models.olmo2.modeling_olmo2 import (
    Olmo2RotaryEmbedding,
)

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    endpoint_geo_inv_freq,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    TrainingBackbone,
    install_adaptation,
    trainable_named_parameters,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.olmo2_length_gated_method import (
    EVQ_FREQUENCY_SHA256,
    LENGTH_GATED_FREQUENCY_NAME,
    LengthGatedEOSVocabRowHead,
    LengthGatedLoRALinear,
    LengthGatedRotaryEmbedding,
    LengthModeState,
    eos_head_trainable_named_parameters,
    freeze_length_gated_qkvo_adapter,
    install_length_gated_eos_vocab_row_head,
    install_length_gated_qkvo,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    LoRALinear,
)


def _olmo_config() -> Olmo2Config:
    return Olmo2Config(
        hidden_size=2_048,
        intermediate_size=8_192,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=128,
        max_position_embeddings=4_096,
        rope_theta=500_000.0,
        vocab_size=100_352,
        tie_word_embeddings=False,
    )


def test_length_gated_rotary_is_exact_on_both_uniform_branches() -> None:
    native = Olmo2RotaryEmbedding(_olmo_config())
    state = LengthModeState(short_context_limit=4_096)
    gated = LengthGatedRotaryEmbedding(native, state)
    hidden = torch.randn(1, 4, 2_048)

    short_positions = torch.tensor([[0, 1, 4_094, 4_095]])
    expected_short = native(hidden, short_positions)
    observed_short = gated(hidden, short_positions)
    assert state.mode == "short"
    assert torch.equal(observed_short[0], expected_short[0])
    assert torch.equal(observed_short[1], expected_short[1])

    long_positions = torch.tensor([[0, 4_096, 8_191, 16_383]])
    expected_long = gated.evq(hidden, long_positions)
    observed_long = gated(hidden, long_positions)
    assert state.mode == "long"
    assert torch.equal(observed_long[0], expected_long[0])
    assert torch.equal(observed_long[1], expected_long[1])

    receipt = gated.receipt()
    assert receipt["active_frequency"] == LENGTH_GATED_FREQUENCY_NAME
    assert (
        receipt["long_branch"]["frequency_sha256_float32"]
        == EVQ_FREQUENCY_SHA256
    )


def test_length_gated_rotary_dispatches_mixed_rows_independently() -> None:
    native = Olmo2RotaryEmbedding(_olmo_config())
    state = LengthModeState(short_context_limit=4_096)
    gated = LengthGatedRotaryEmbedding(native, state)
    hidden = torch.randn(2, 3, 2_048)
    positions = torch.tensor(
        [
            [0, 2_048, 4_095],
            [0, 4_096, 8_192],
        ]
    )

    native_values = native(hidden, positions)
    evq_values = gated.evq(hidden, positions)
    observed = gated(hidden, positions)

    assert state.mode == "mixed"
    assert torch.equal(observed[0][0], native_values[0][0])
    assert torch.equal(observed[1][0], native_values[1][0])
    assert torch.equal(observed[0][1], evq_values[0][1])
    assert torch.equal(observed[1][1], evq_values[1][1])


def test_cached_short_to_long_transition_fails_without_forced_budget() -> None:
    state = LengthModeState(short_context_limit=4_096)
    state.update(torch.arange(4_096).unsqueeze(0))
    assert state.mode == "short"

    with pytest.raises(RuntimeError, match="cached generation crossed"):
        state.update(torch.tensor([[4_096]]))

    state.force("long")
    assert state.update(torch.tensor([[4_096]])) == "long"


def test_length_gated_lora_returns_base_directly_in_short_mode() -> None:
    base = nn.Linear(4, 4, bias=False)
    state = LengthModeState(short_context_limit=4_096)
    gated = LengthGatedLoRALinear(
        base,
        rank=2,
        alpha=4.0,
        state=state,
    )
    with torch.no_grad():
        gated.a.fill_(0.5)
        gated.b.fill_(0.25)
    value = torch.randn(2, 3, 4)

    state.force("short")
    expected = base(value)
    observed = gated(value)
    assert torch.equal(observed, expected)

    state.force("long")
    long_output = gated(value)
    assert not torch.equal(long_output, expected)


def test_length_gated_long_mode_matches_ordinary_lora() -> None:
    base = nn.Linear(4, 4, bias=False)
    state = LengthModeState(short_context_limit=4_096)
    gated = LengthGatedLoRALinear(
        copy.deepcopy(base),
        rank=2,
        alpha=4.0,
        state=state,
    )
    ordinary = LoRALinear(
        copy.deepcopy(base),
        rank=2,
        alpha=4.0,
    )
    with torch.no_grad():
        ordinary.a.copy_(gated.a)
        ordinary.b.copy_(torch.randn_like(ordinary.b))
        gated.b.copy_(ordinary.b)
    value = torch.randn(2, 3, 4)

    state.force("long")
    assert torch.equal(gated(value), ordinary(value))


def test_length_gated_eos_head_changes_only_eos_in_long_mode() -> None:
    base = nn.Linear(4, 7, bias=False)
    state = LengthModeState(short_context_limit=4_096)
    head = LengthGatedEOSVocabRowHead(
        base,
        rank=1,
        alpha=1.0,
        eos_token_id=3,
        state=state,
    )
    with torch.no_grad():
        head.delta_weight.fill_(0.5)
        head.eos_bias.fill_(0.75)
    value = torch.randn(2, 3, 4)
    expected = base(value)

    state.force("short")
    assert torch.equal(head(value), expected)

    state.force("long")
    observed = head(value)
    non_eos = [0, 1, 2, 4, 5, 6]
    assert torch.equal(observed[..., non_eos], expected[..., non_eos])
    assert not torch.equal(observed[..., 3], expected[..., 3])


def test_collapsed_eos_row_has_nonzero_first_step_gradient() -> None:
    base = nn.Linear(4, 7, bias=False)
    with torch.no_grad():
        base.weight.zero_()
    state = LengthModeState(short_context_limit=4_096)
    state.force("long")
    head = LengthGatedEOSVocabRowHead(
        base,
        rank=1,
        alpha=1.0,
        eos_token_id=3,
        state=state,
    )
    value = torch.ones(2, 3, 4)

    loss = -head(value).log_softmax(dim=-1)[..., 3].mean()
    loss.backward()

    assert head.delta_weight.grad is not None
    assert torch.count_nonzero(head.delta_weight.grad).item() == 4
    assert head.eos_bias.grad is not None
    assert head.eos_bias.grad.item() != 0.0
    assert base.weight.grad is None


def test_forced_long_path_is_fullgraph_compile_compatible() -> None:
    state = LengthModeState(short_context_limit=4_096)
    state.force("long")

    class CompiledPath(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rotary = LengthGatedRotaryEmbedding(
                Olmo2RotaryEmbedding(_olmo_config()),
                state,
            )
            self.linear = LengthGatedLoRALinear(
                nn.Linear(4, 4, bias=False),
                rank=2,
                alpha=4.0,
                state=state,
            )

        def forward(
            self,
            value: torch.Tensor,
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            cos, sin = self.rotary(value, positions)
            return self.linear(value), cos, sin

    module = CompiledPath().eval()
    value = torch.randn(1, 4, 4)
    positions = torch.tensor([[0, 4_096, 8_192, 16_383]])
    expected = module(value, positions)
    compiled = torch.compile(
        module,
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )
    observed = compiled(value, positions)

    assert all(
        torch.equal(expected_value, observed_value)
        for expected_value, observed_value in zip(expected, observed)
    )


def test_real_olmo_forward_and_kv_cache_match_branch_references() -> None:
    config = Olmo2Config(
        vocab_size=257,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=4_096,
        rope_theta=500_000.0,
        tie_word_embeddings=False,
        use_cache=True,
        attention_dropout=0.0,
    )
    torch.manual_seed(7)
    native = Olmo2ForCausalLM(config).eval()
    gated = copy.deepcopy(native).eval()
    full_evq_lora = copy.deepcopy(native).eval()
    state = LengthModeState(short_context_limit=4_096)
    gated.model.rotary_emb = LengthGatedRotaryEmbedding(
        gated.model.rotary_emb,
        state,
    )
    full_evq_lora.model.rotary_emb = copy.deepcopy(
        gated.model.rotary_emb.evq
    )
    for gated_layer, reference_layer in zip(
        gated.model.layers,
        full_evq_lora.model.layers,
    ):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            gated_wrapper = LengthGatedLoRALinear(
                getattr(gated_layer.self_attn, name),
                rank=4,
                alpha=8.0,
                state=state,
            )
            reference_wrapper = LoRALinear(
                getattr(reference_layer.self_attn, name),
                rank=4,
                alpha=8.0,
            )
            with torch.no_grad():
                gated_wrapper.a.copy_(
                    torch.randn_like(gated_wrapper.a)
                )
                gated_wrapper.b.copy_(
                    torch.randn_like(gated_wrapper.b)
                )
                reference_wrapper.a.copy_(gated_wrapper.a)
                reference_wrapper.b.copy_(gated_wrapper.b)
            setattr(gated_layer.self_attn, name, gated_wrapper)
            setattr(
                reference_layer.self_attn,
                name,
                reference_wrapper,
            )

    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    state.force("short")
    native_logits = native(
        input_ids=input_ids,
        use_cache=False,
    ).logits
    short_logits = gated(
        input_ids=input_ids,
        use_cache=False,
    ).logits
    assert torch.equal(short_logits, native_logits)

    state.force("long")
    long_logits = gated(
        input_ids=input_ids,
        use_cache=False,
    ).logits
    reference_logits = full_evq_lora(
        input_ids=input_ids,
        use_cache=False,
    ).logits
    assert torch.equal(long_logits, reference_logits)

    state.force("long")
    gated_prefill = gated(
        input_ids=input_ids[:, :6],
        use_cache=True,
        return_dict=True,
    )
    reference_prefill = full_evq_lora(
        input_ids=input_ids[:, :6],
        use_cache=True,
        return_dict=True,
    )
    assert torch.equal(
        gated_prefill.logits,
        reference_prefill.logits,
    )
    gated_decode = gated(
        input_ids=input_ids[:, 6:7],
        past_key_values=gated_prefill.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    reference_decode = full_evq_lora(
        input_ids=input_ids[:, 6:7],
        past_key_values=reference_prefill.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    assert torch.equal(gated_decode.logits, reference_decode.logits)

    virtual_positions = torch.tensor(
        [[0, 1, 2, 3, 4_096, 8_192, 12_288, 16_383]]
    )
    state.force("long")
    training_backbone = TrainingBackbone(gated.model).eval()
    expected_hidden = training_backbone(
        input_ids,
        virtual_positions,
    )
    compiled_backbone = torch.compile(
        training_backbone,
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )
    observed_hidden = compiled_backbone(
        input_ids,
        virtual_positions,
    )
    assert torch.equal(observed_hidden, expected_hidden)

    freeze_length_gated_qkvo_adapter(gated)
    eos_head, _ = install_length_gated_eos_vocab_row_head(
        gated,
        state,
        rank=1,
        alpha=1.0,
        eos_token_id=2,
    )
    with torch.no_grad():
        eos_head.delta_weight.fill_(0.5)
        eos_head.eos_bias.fill_(1.0)
    state.force("long")
    eos_adapted_long = gated(
        input_ids=input_ids,
        use_cache=False,
    ).logits
    non_eos = [
        token_id
        for token_id in range(config.vocab_size)
        if token_id != eos_head.eos_token_id
    ]
    assert torch.equal(
        eos_adapted_long[..., non_eos],
        reference_logits[..., non_eos],
    )
    assert not torch.equal(
        eos_adapted_long[..., eos_head.eos_token_id],
        reference_logits[..., eos_head.eos_token_id],
    )

    state.force("short")
    native_short_prefill = native(
        input_ids=input_ids[:, :6],
        use_cache=True,
        return_dict=True,
    )
    gated_short_prefill = gated(
        input_ids=input_ids[:, :6],
        use_cache=True,
        return_dict=True,
    )
    assert torch.equal(
        gated_short_prefill.logits,
        native_short_prefill.logits,
    )
    native_cache = native_short_prefill.past_key_values.to_legacy_cache()
    gated_cache = gated_short_prefill.past_key_values.to_legacy_cache()
    assert len(native_cache) == len(gated_cache)
    for native_layer, gated_layer in zip(native_cache, gated_cache):
        assert torch.equal(gated_layer[0], native_layer[0])
        assert torch.equal(gated_layer[1], native_layer[1])
    native_short_decode = native(
        input_ids=input_ids[:, 6:7],
        past_key_values=native_short_prefill.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    gated_short_decode = gated(
        input_ids=input_ids[:, 6:7],
        past_key_values=gated_short_prefill.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    assert torch.equal(gated_short_decode.logits, native_short_decode.logits)


class _FakeRotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq",
            endpoint_geo_inv_freq().clone(),
            persistent=False,
        )
        self.original_inv_freq = self.inv_freq

    def forward(
        self,
        value: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (*position_ids.shape, 128)
        return (
            torch.ones(shape, dtype=value.dtype, device=value.device),
            torch.zeros(shape, dtype=value.dtype, device=value.device),
        )


def _fake_model() -> nn.Module:
    model = nn.Module()
    model.config = SimpleNamespace(
        model_type="olmo2",
        hidden_size=2_048,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=16,
    )
    model.model = nn.Module()
    model.model.rotary_emb = _FakeRotary()
    model.model.layers = nn.ModuleList()
    model.lm_head = nn.Linear(4, 11, bias=False)
    for _ in range(16):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(
                layer.self_attn,
                name,
                nn.Linear(4, 4, bias=False),
            )
        model.model.layers.append(layer)
    return model


def test_gated_installer_is_parent_adapter_state_compatible() -> None:
    ordinary_model = _fake_model()
    gated_model = _fake_model()
    install_adaptation(
        ordinary_model,
        "qkvo_answer",
        rank=2,
        alpha=4.0,
    )
    state, receipt = install_length_gated_qkvo(
        gated_model,
        rank=2,
        alpha=4.0,
    )
    ordinary = dict(trainable_named_parameters(ordinary_model, None))
    gated = dict(trainable_named_parameters(gated_model, None))

    assert set(gated) == set(ordinary)
    assert {
        name: tuple(value.shape) for name, value in gated.items()
    } == {
        name: tuple(value.shape) for name, value in ordinary.items()
    }
    assert receipt["adaptation"] == "length_gated_qkvo_answer"
    assert state.forced_mode is None


def test_parent_child_chain_freezes_qkvo_and_trains_only_eos_head() -> None:
    model = _fake_model()
    state, _ = install_length_gated_qkvo(
        model,
        rank=2,
        alpha=4.0,
    )
    freeze_receipt = freeze_length_gated_qkvo_adapter(model)
    eos_head, head_receipt = install_length_gated_eos_vocab_row_head(
        model,
        state,
        rank=1,
        alpha=1.0,
        eos_token_id=2,
    )

    trainable = dict(trainable_named_parameters(model, None))
    assert set(trainable) == {
        "model.lm_head.delta_weight",
        "model.lm_head.eos_bias",
    }
    assert all(parameter.requires_grad for parameter in trainable.values())
    assert freeze_receipt["qkvo_lora_modules"] == 64
    assert freeze_receipt["parameter_tensors"] == 128
    assert freeze_receipt["trainable_after_freeze"] is False
    assert head_receipt["modified_vocab_rows"] == [2]
    assert eos_head.base.weight.requires_grad is False

    child = dict(eos_head_trainable_named_parameters(model))
    assert child == trainable
