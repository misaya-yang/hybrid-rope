#!/usr/bin/env python3

import torch

from rebuttal.rebuttal_0723.experiments.frequency_band_usage_5090 import (
    apply_nope,
    intervention_windows,
    predicted_band_indices,
    runtime_inv_freq,
)


def test_native_base256_band_prediction() -> None:
    index = torch.arange(32, dtype=torch.float64)
    inv = torch.pow(256.0, -index / 32.0).float()
    assert predicted_band_indices(inv) == {"3.657": 25, "4.493": 23}


def test_nope_is_exact_and_controls_do_not_overlap() -> None:
    inv = torch.linspace(1.0, 0.1, 32)
    windows = intervention_windows(30)
    assert all(len(value) == 3 for value in windows.values())
    assert set(windows["band_nope"]).isdisjoint(windows["adjacent_nope"])
    assert set(windows["band_nope"]).isdisjoint(windows["far_nope"])
    masked = apply_nope(inv, windows["band_nope"])
    assert torch.count_nonzero(masked == 0).item() == 3
    keep = [index for index in range(32) if index not in windows["band_nope"]]
    assert torch.equal(masked[keep], inv[keep])


def test_runtime_frequency_uses_immutable_training_table() -> None:
    training = torch.linspace(1.0, 0.1, 32)
    assert torch.equal(runtime_inv_freq(training, "paper_geo_base500k", 8_192), training)
    assert torch.equal(runtime_inv_freq(training, "fmrope_base256", 256), training)
    assert not torch.equal(runtime_inv_freq(training, "fmrope_base256", 8_192), training)
