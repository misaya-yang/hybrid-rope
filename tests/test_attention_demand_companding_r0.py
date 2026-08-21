import math
from argparse import Namespace

import numpy as np
import torch

from scripts.analysis.attention_demand_companding_r0 import (
    collect_hf_olmo2,
    companded_density,
    density_from_distance,
    distance_phi,
    distortion,
    headroom,
    midpoint_quantiles,
    peak_locations,
)


def test_companding_identities_and_direction():
    uniform = np.ones(64)
    h_uniform, d_uniform = headroom(uniform)
    assert math.isclose(h_uniform, 0.0, abs_tol=1e-12)
    assert math.isclose(d_uniform, 1.0, abs_tol=1e-12)

    demand = np.zeros(64)
    demand[:8] = 8.0
    h_value, d_star = headroom(demand)
    rho = companded_density(demand, 0.1)
    quantiles = midpoint_quantiles(rho, 32)

    assert 0.0 < h_value < 1.0
    assert math.isclose(d_star, 1.0 - h_value, abs_tol=1e-12)
    assert np.all(np.diff(quantiles) > 0)
    assert distortion(demand, rho) < distortion(demand, np.ones_like(demand))

    distance = np.arange(512)
    endpoint = distance_phi(distance, 512, 500_000.0, "endpoint_log")
    physical = distance_phi(distance, 512, 500_000.0, "physical_1rad")
    assert np.all(np.diff(endpoint) >= 0)
    assert np.all(np.diff(physical) >= 0)
    assert endpoint[-1] == 1.0
    assert physical[-1] < 1.0

    mass = np.ones(512)
    _, density, probability = density_from_distance(
        mass, 512, 500_000.0, 64, "endpoint_log", include_self=False
    )
    assert math.isclose(np.mean(density), 1.0, abs_tol=1e-12)
    assert math.isclose(probability.sum(), 1.0, abs_tol=1e-12)

    bimodal = np.exp(-((np.arange(64) - 10) / 2) ** 2)
    bimodal += 0.8 * np.exp(-((np.arange(64) - 50) / 3) ** 2)
    peaks = peak_locations(bimodal)
    assert len(peaks) == 2
    assert peaks[0] < 20 and peaks[1] > 40


def test_tiny_olmo2_streaming_backend(tmp_path):
    from transformers import Olmo2Config, Olmo2ForCausalLM

    model_dir = tmp_path / "model"
    config = Olmo2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_theta=500_000.0,
    )
    Olmo2ForCausalLM(config).save_pretrained(model_dir)
    token_path = tmp_path / "tokens.pt"
    torch.save(torch.arange(64, dtype=torch.long) % config.vocab_size, token_path)
    args = Namespace(
        device="cpu",
        allow_slow_device=True,
        tokens=token_path,
        length=32,
        windows=2,
        queries=4,
        batch_size=1,
        model=model_dir,
        model_revision="tiny-test",
    )
    metadata, arrays = collect_hf_olmo2(args)
    assert metadata["layers"] == 2
    assert metadata["heads"] == 4
    assert metadata["attention_mass_conservation_max_abs"] < 1e-4
    assert arrays["mass"].shape == (2, 4, 32)
    assert arrays["window_global_mass"].shape == (2, 32)
