#!/usr/bin/env python3

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.frequency_causal_spectrum_5090 import (
    phase_utility,
    spearman,
    swap_pairs,
)


def test_swap_preserves_frequency_multiset() -> None:
    value = torch.arange(8, dtype=torch.float32)
    swapped = swap_pairs(value, [1, 2], [5, 6])
    assert torch.equal(torch.sort(swapped).values, value)
    assert swapped.tolist() == [0, 5, 6, 3, 4, 1, 2, 7]


def test_phase_utility_and_spearman() -> None:
    utility = phase_utility(torch.tensor([1.0, 0.1, 0.01]), 256)
    assert utility.shape == (3,)
    assert np.isfinite(utility).all()
    assert spearman([1, 2, 3], [10, 20, 30]) == 1.0
    assert spearman([1, 2, 3], [30, 20, 10]) == -1.0
