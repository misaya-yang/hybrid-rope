from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    endpoint_evq_inv_freq,
    endpoint_geo_inv_freq,
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    evaluate_instruct_ruler_transfer as transfer,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    evaluate_instruct_ruler_screen as screen,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_4k_natural_multiquery_hybrid_screen import (
    parse_custom_head_sets,
)


class _Rotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq",
            endpoint_geo_inv_freq().clone(),
            persistent=False,
        )


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(rotary_emb=_Rotary()),
        config=SimpleNamespace(num_attention_heads=16),
    )


@pytest.mark.parametrize(
    ("name", "native_indices"),
    (
        ("hybrid_native_low8", tuple(range(56, 64))),
        ("hybrid_native_low16", tuple(range(48, 64))),
        ("hybrid_native_low32", tuple(range(32, 64))),
        ("hybrid_native_high16", tuple(range(0, 16))),
        (
            "hybrid_native_ends16",
            tuple(range(0, 8)) + tuple(range(56, 64)),
        ),
        ("hybrid_evq_low4", tuple(range(0, 60))),
        ("hybrid_evq_low8", tuple(range(0, 56))),
        ("hybrid_evq_high8", tuple(range(8, 64))),
        (
            "hybrid_evq_mid8",
            tuple(range(0, 28)) + tuple(range(36, 64)),
        ),
    ),
)
def test_pair_hybrid_preserves_the_declared_native_indices(
    name: str,
    native_indices: tuple[int, ...],
) -> None:
    model = _model()
    native = endpoint_geo_inv_freq()
    evq = endpoint_evq_inv_freq()
    receipt = screen.apply_screen_frequency(model, name)
    actual = model.model.rotary_emb.inv_freq.detach().cpu()
    expected = evq.clone()
    expected[list(native_indices)] = native[list(native_indices)]

    assert torch.equal(actual, expected)
    assert receipt["active_sha256_float32"] == tensor_sha256(expected)
    assert receipt["active_sha256_float32"] not in {
        tensor_sha256(native),
        tensor_sha256(evq),
    }
    assert receipt["hybrid_native_pair_indices"] == list(native_indices)


@pytest.mark.parametrize(
    ("name", "weight"),
    (
        ("hybrid_blend10", 0.10),
        ("hybrid_blend25", 0.25),
        ("hybrid_blend_evq_0p1pct", 0.001),
        ("hybrid_blend_evq_0p5pct", 0.005),
        ("hybrid_blend_evq_1pct", 0.01),
        ("hybrid_blend_evq_2pct", 0.02),
        ("hybrid_blend_evq_5pct", 0.05),
    ),
)
def test_log_frequency_blend_is_not_silently_full_evq(
    name: str,
    weight: float,
) -> None:
    model = _model()
    native = endpoint_geo_inv_freq()
    evq = endpoint_evq_inv_freq()
    receipt = screen.apply_screen_frequency(model, name)
    actual = model.model.rotary_emb.inv_freq.detach().cpu()
    expected = torch.exp(
        (1.0 - weight) * torch.log(native)
        + weight * torch.log(evq)
    )

    assert torch.equal(actual, expected)
    assert receipt["active_sha256_float32"] == tensor_sha256(expected)
    assert receipt["active_sha256_float32"] not in {
        tensor_sha256(native),
        tensor_sha256(evq),
    }


def test_head_hybrid_contains_both_native_and_evq_tables() -> None:
    model = _model()
    native = endpoint_geo_inv_freq()
    evq = endpoint_evq_inv_freq()
    original_apply = screen.modeling_olmo2.apply_rotary_pos_emb
    try:
        receipt = screen.apply_screen_frequency(
            model,
            "hybrid_heads_custom",
            custom_evq_head_indices=(0, 15),
        )
        actual = model.model.rotary_emb.inv_freq_by_head.detach().cpu()
    finally:
        screen.modeling_olmo2.apply_rotary_pos_emb = original_apply

    assert actual.shape == (16, 64)
    assert torch.equal(actual[0], evq)
    assert torch.equal(actual[15], evq)
    assert torch.equal(actual[1:15], native.repeat(14, 1))
    assert receipt["hybrid_evq_head_indices"] == [0, 15]
    assert receipt["hybrid_native_head_count"] == 14


def test_custom_head_sets_are_canonicalized() -> None:
    assert parse_custom_head_sets(("15", "7,2,7")) == [
        ("hybrid_heads_custom__15", (15,)),
        ("hybrid_heads_custom__2_7", (2, 7)),
    ]


@pytest.mark.parametrize(
    "value",
    ("", "16", "-1", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"),
)
def test_custom_head_sets_reject_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_custom_head_sets((value,))


def test_evq_official_yarn_reuses_exact_index_scaler() -> None:
    native = endpoint_geo_inv_freq()
    evq = endpoint_evq_inv_freq()
    scaler = torch.linspace(0.5, 1.0, native.numel())
    official_native_yarn = (native * scaler).clone()
    rotary = SimpleNamespace(
        inv_freq=official_native_yarn.clone(),
        original_inv_freq=official_native_yarn.clone(),
        attention_scaling=1.2,
    )
    model = SimpleNamespace(model=SimpleNamespace(rotary_emb=rotary))
    config = SimpleNamespace(rope_scaling={"factor": 4.0})
    receipt = transfer.apply_evq_official_yarn(
        model,
        config,
        {
            "active_frequency": "official_transformers_yarn",
            "active_sha256_float32": tensor_sha256(
                official_native_yarn
            ),
            "attention_scaling": 1.2,
        },
    )

    realized_scaler = official_native_yarn / native
    expected = evq * realized_scaler
    assert torch.allclose(rotary.inv_freq, expected)
    assert torch.equal(rotary.original_inv_freq, rotary.inv_freq)
    assert receipt["active_sha256_float32"] == tensor_sha256(expected)
    assert receipt["per_index_scaler_sha256_float32"] == tensor_sha256(
        realized_scaler
    )


@pytest.mark.parametrize(
    ("substrate", "factor"),
    (("native", 2.0), ("evq", 4.0)),
)
def test_repo_fixed_ramp_uses_canonical_legacy_scaler(
    substrate: str,
    factor: float,
) -> None:
    model = _model()
    base = (
        endpoint_geo_inv_freq()
        if substrate == "native"
        else endpoint_evq_inv_freq()
    )
    expected, expected_attention_scaling, expected_meta = (
        transfer.repo_fixed_ramp_inv_freq(base, scale=factor)
    )
    receipt = transfer.apply_repo_fixed_ramp(
        model,
        substrate=substrate,
        factor=factor,
    )
    actual = model.model.rotary_emb.inv_freq.detach().cpu()

    assert torch.equal(actual, expected.float())
    assert expected_attention_scaling == 1.0
    assert receipt["active_sha256_float32"] == tensor_sha256(
        expected.float()
    )
    assert receipt["mode"] == expected_meta["mode"]
    assert receipt["attention_scaling"] == 1.0
