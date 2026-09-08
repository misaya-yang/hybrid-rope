from types import SimpleNamespace

import torch

from scripts.experiments.olmo_fast_screen.cross_cache import rotated_key, trim_prefix


def test_rotation_matches_independent_complex_pairs():
    raw = torch.tensor([[[[1., 2., 3., 4.], [5., 6., 7., 8.]]]])
    phase = torch.tensor([[[.3, .7], [1.2, -.2]]])
    angles = torch.cat((phase, phase), dim=-1)
    actual = rotated_key(raw, angles.cos(), angles.sin())
    pairs = torch.complex(raw[..., :2], raw[..., 2:])
    expected = pairs * torch.exp(1j * phase.unsqueeze(1))
    assert torch.allclose(actual, torch.cat((expected.real, expected.imag), dim=-1))


def test_trim_removes_query_and_generated_cache_without_changing_prefix_values():
    DynamicLayer = type('DynamicLayer', (), {})
    layer = DynamicLayer()
    layer.keys = torch.arange(24).view(1, 2, 6, 2)
    layer.values = layer.keys + 100
    expected = layer.values[..., :3, :].clone()
    cache = SimpleNamespace(layers=[layer], get_seq_length=lambda: layer.keys.shape[-2])
    trim_prefix(cache, 3)
    assert torch.equal(layer.values, expected)
    assert cache.get_seq_length() == 3


def test_fp32_phases_cast_only_after_rotation_for_bf16_keys():
    raw = torch.tensor([[[[1.2, -2.3, 3.4, 4.5]]]], dtype=torch.bfloat16)
    phase = torch.tensor([[[.31, .67]]], dtype=torch.float32)
    angle = torch.cat((phase,phase),dim=-1)
    actual = rotated_key(raw,angle.cos(),angle.sin())
    pair = torch.complex(raw.float()[...,:2],raw.float()[...,2:])
    rotated = pair*torch.complex(phase.cos(),phase.sin()).unsqueeze(1)
    expected = torch.cat((rotated.real,rotated.imag),dim=-1).to(torch.bfloat16)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual,expected)
