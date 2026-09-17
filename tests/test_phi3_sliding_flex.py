import torch

from experiments.olmo_recovery_20260912.runtime import (
    phi3_pad_attention_head,
    phi3_sliding_visibility,
)


def test_phi3_sliding_visibility_matches_strict_window_boundary():
    observed = phi3_sliding_visibility(3, 5, 3)
    expected = torch.tensor([
        [True, True, True, False, False],
        [False, True, True, True, False],
        [False, False, True, True, True],
    ])
    torch.testing.assert_close(observed, expected)


def test_phi3_sliding_visibility_rejects_invalid_geometry():
    for values in ((0, 5, 3), (6, 5, 3), (3, 5, 0)):
        try:
            phi3_sliding_visibility(*values)
        except ValueError:
            pass
        else:
            raise AssertionError(values)


def test_phi3_head_dim_96_is_zero_padded_to_128():
    tensors = [torch.randn(1, 2, 3, 96) for _ in range(3)]
    query, key, value, output_dim = phi3_pad_attention_head(*tensors)
    assert output_dim == 96
    assert query.shape[-1] == key.shape[-1] == value.shape[-1] == 128
    for original, padded in zip(tensors, (query, key, value)):
        torch.testing.assert_close(padded[..., :96], original)
        assert torch.count_nonzero(padded[..., 96:]) == 0
