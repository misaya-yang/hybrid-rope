from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.native_followup_five_20260917.mass_projection import (
    METHOD_IDS,
    _project_qkv,
    install,
    mass_projection_logits,
    mass_raw_logits,
)


def probabilities(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / numerator.sum(axis=-1, keepdims=True)


def test_mass_projection_preserves_every_local_probability_and_far_mass():
    native = np.array([[0.3, -0.7, 1.2, 0.4, -1.0], [-0.2, 0.9, -0.1, 1.4, 0.2]])
    candidate = np.array([[1.8, 0.1, -0.5, 2.2, 0.7], [0.6, -0.3, 1.1, -0.8, 2.0]])
    distance = np.array([[0, 1, 3, 5, 8], [0, 2, 4, 6, 9]])
    projected = mass_projection_logits(native, candidate, distance, local_radius=2)
    p0, p = probabilities(native), probabilities(projected)
    local = distance <= 2
    far = ~local
    np.testing.assert_allclose(p[local], p0[local], atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose((p * far).sum(-1), (p0 * far).sum(-1), atol=2e-15, rtol=2e-15)
    for row in range(native.shape[0]):
        np.testing.assert_allclose(
            p[row, far[row]] / p[row, far[row]].sum(),
            probabilities(candidate[row, far[row]]),
            atol=2e-15,
            rtol=2e-15,
        )


def test_mass_raw_is_piecewise_logits_under_one_common_softmax():
    native = np.array([0.0, 1.0, 2.0, 3.0])
    candidate = np.array([9.0, 8.0, -1.0, -2.0])
    distance = np.array([0, 1, 2, 3])
    got = mass_raw_logits(native, candidate, distance, local_radius=1)
    np.testing.assert_array_equal(got, np.array([0.0, 1.0, -1.0, -2.0]))
    # Unlike B, the common denominator is allowed to change local probabilities.
    assert not np.allclose(probabilities(got)[:2], probabilities(native)[:2])


def test_no_far_keys_and_identical_tables_are_exact_identities():
    native = np.array([[0.1, -0.4, 2.0]])
    distance = np.array([[0, 1, 2]])
    np.testing.assert_array_equal(
        mass_projection_logits(native, native + 5, distance, local_radius=2), native
    )
    np.testing.assert_allclose(
        mass_projection_logits(native, native, distance, local_radius=0), native, atol=1e-15
    )
    np.testing.assert_array_equal(mass_raw_logits(native, native, distance, 0), native)


def test_invalid_future_key_is_rejected():
    with pytest.raises(ValueError, match="future"):
        mass_projection_logits([1.0, 2.0], [3.0, 4.0], [0, -1], 1)


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

    def forward(self, *args, **kwargs):
        return "original"


def fake_model():
    attention = FakeAttention()
    return SimpleNamespace(
        config=SimpleNamespace(model_type="olmo2", head_dim=4, max_position_embeddings=32),
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]),
    )


def test_olmo_projection_applies_qk_norm_before_head_reshape():
    attention = FakeAttention()
    hidden = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    query, key, value = _project_qkv(attention, hidden)
    expected = hidden.view(1, 2, 1, 4).transpose(1, 2)
    torch.testing.assert_close(query, expected * 2)
    torch.testing.assert_close(key, expected * 3)
    torch.testing.assert_close(value, expected)


@pytest.mark.parametrize("method", tuple(METHOD_IDS))
def test_install_exposes_identity_and_idempotent_restore(method):
    model = fake_model()
    attention = model.model.layers[0].self_attn
    original = attention.forward
    table = {"values_float32": [1.0, 0.1], "gain": 1.0}
    restore = install(model, table, table, method)
    assert attention._native_mass_projection_method == method
    assert restore.method_id == METHOD_IDS[method]
    assert restore.local_radius == 2
    assert attention.forward != original
    restore()
    assert attention.forward == original
    assert not hasattr(attention, "_native_mass_projection_method")
    restore()


def test_install_rejects_gain_or_geometry_drift():
    good = {"values_float32": [1.0, 0.1], "gain": 1.0}
    with pytest.raises(ValueError, match="gain=1"):
        install(fake_model(), good, {**good, "gain": 1.1}, "mass_projection")
    with pytest.raises(ValueError, match="geometry"):
        install(fake_model(), good, {"values_float32": [1.0], "gain": 1.0}, "mass_projection")


def test_module_cli_is_plan_only():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.native_followup_five_20260917.mass_projection",
            "--method",
            "mass_raw",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "PLAN_ONLY"
    assert result["gpu_execution"] is False
    assert result["model_loaded"] is False
