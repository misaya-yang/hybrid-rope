import numpy as np
import torch
import torch.nn as nn

from experiments.olmo_recovery_20260912.native_relative_allocation import (
    NativeRelativeAllocation,
    install_native_relative_allocation,
)


class DummyRotary(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("inv_freq", torch.tensor([1.0, 0.5, 0.25, 0.125, 0.0625]))
        self.attention_scaling = 1.0
        self.rope_type = "default"


def exponents():
    return np.asarray([0.0, 0.0, 0.25, 0.75, 1.0])


def test_three_band_exponents_and_gain_are_exact_at_initialization():
    module = NativeRelativeAllocation(
        DummyRotary(), low=1, high=4, scale=4.0,
        initial_exponents=exponents(), initial_gain=1.2,
    )
    np.testing.assert_allclose(module.exponents().detach().numpy(), exponents(), atol=2e-8)
    expected = DummyRotary().inv_freq.numpy() * np.power(4.0, -exponents())
    np.testing.assert_allclose(module.realized_inv_freq().detach().numpy(), expected, rtol=2e-7)
    assert float(module.attention_scaling.detach()) == np.float32(1.2)


def test_frequency_and_gain_receive_gradients_but_model_weights_do_not():
    model = nn.Module()
    model.weight = nn.Parameter(torch.ones(()))
    model.model = nn.Module()
    model.model.rotary_emb = DummyRotary()
    module = install_native_relative_allocation(
        model, low=1, high=4, scale=4.0,
        initial_exponents=exponents(), initial_gain=1.2,
    )
    cos, sin = module(torch.empty(1), torch.tensor([[0, 7, 31]]))
    (cos.sum() + sin.sum()).backward()
    assert module.increment_logits.grad is not None and torch.isfinite(module.increment_logits.grad).all()
    assert module.log_gain.grad is not None and torch.isfinite(module.log_gain.grad)
    assert model.weight.grad is None and not model.weight.requires_grad


def test_set_state_removes_softmax_translation_null_direction():
    module = NativeRelativeAllocation(
        DummyRotary(), low=1, high=4, scale=4.0,
        initial_exponents=exponents(), initial_gain=1.2,
    )
    module.set_state_(torch.tensor([4.0, 5.0, 6.0]), np.log(1.1))
    assert float(module.increment_logits.detach().sum()) == 0.0
    assert np.isclose(float(module.attention_scaling.detach()), 1.1)


def test_exact_frozen_fp32_initial_table_uses_straight_through_gradient():
    exact = (DummyRotary().inv_freq.numpy() * np.power(4.0, -exponents())).astype(np.float32)
    exact[2] = np.nextafter(exact[2], np.float32(np.inf))
    module = NativeRelativeAllocation(
        DummyRotary(), low=1, high=4, scale=4.0,
        initial_exponents=exponents(), initial_gain=1.2, initial_inv_freq=exact,
    )
    np.testing.assert_array_equal(module.realized_inv_freq().detach().numpy(), exact)
    module.realized_inv_freq().sum().backward()
    assert module.increment_logits.grad is not None
