import torch

from experiments.olmo_recovery_20260912.runtime import phi3_sliding_visibility


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
