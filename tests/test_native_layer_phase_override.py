from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.native_enhancement_oral_20260915.layer_phase_override import (
    install_layer_phase_override,
    parse_layer_range,
    remove_hooks,
)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None):
        self.seen = position_embeddings
        return hidden_states


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.model = SimpleNamespace(layers=[Block() for _ in range(4)])
        self.config = SimpleNamespace(hidden_size=8, num_attention_heads=1)


def test_layer_range_is_exclusive_and_bounded():
    assert parse_layer_range("2:4", num_hidden_layers=4) == (2, 3)
    with pytest.raises(ValueError):
        parse_layer_range("4:5", num_hidden_layers=4)


def test_override_changes_only_selected_attention_position_embeddings():
    model = Model()
    table = np.array([1.0, 0.5, 0.2, 0.1], dtype=np.float32)
    handles = install_layer_phase_override(model, layers=(2, 3), values_float32=table, gain=1.0)
    hidden = torch.zeros((1, 3, 8))
    original = (torch.zeros((1, 3, 8)), torch.zeros((1, 3, 8)))
    for block in model.model.layers:
        block.self_attn(hidden, original, None, None)
    assert model.model.layers[0].self_attn.seen is original
    assert model.model.layers[1].self_attn.seen is original
    cos, sin = model.model.layers[2].self_attn.seen
    torch.testing.assert_close(cos[:, 0], torch.ones((1, 8)))
    assert not torch.equal(cos[:, 1], original[0][:, 1])
    assert not torch.equal(sin[:, 1], original[1][:, 1])
    remove_hooks(handles)
