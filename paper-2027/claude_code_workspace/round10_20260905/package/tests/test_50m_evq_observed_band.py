import math

import torch

from rebuttal.rebuttal_0723.experiments.run_50m_native_vs_evq_lerope import (
    Protocol,
    SelectiveRotaryEmbedding,
    evq_inv_freq,
    native_inv_freq,
)


def test_observed_band_partition_is_contiguous_and_protocol_derived() -> None:
    protocol = Protocol()
    rope = SelectiveRotaryEmbedding(protocol, "evq_observed_lerope")
    expected = (
        protocol.train_length
        * evq_inv_freq(protocol)
        / (2.0 * math.pi)
        >= 1.0
    )
    assert torch.equal(rope.learnable_mask.bool(), expected)
    assert torch.nonzero(expected, as_tuple=False).flatten().tolist() == list(
        range(22)
    )
    assert torch.nonzero(~expected, as_tuple=False).flatten().tolist() == list(
        range(22, 32)
    )


def test_observed_band_arm_starts_at_exact_evq_grid() -> None:
    protocol = Protocol()
    rope = SelectiveRotaryEmbedding(protocol, "evq_observed_lerope")
    assert torch.equal(rope.base_inv_freq, evq_inv_freq(protocol))
    assert torch.equal(rope.current_inv_freq(), evq_inv_freq(protocol))
    assert torch.all(
        rope.current_inv_freq()[:-1] > rope.current_inv_freq()[1:]
    )


def test_fixed_and_full_learned_control_masks() -> None:
    protocol = Protocol()
    expected = {
        "native": 0,
        "evq_fixed": 0,
        "native_lerope": 32,
        "evq_lerope": 32,
        "evq_observed_lerope": 22,
    }
    for arm, learned_count in expected.items():
        rope = SelectiveRotaryEmbedding(protocol, arm)
        assert int(rope.learnable_mask.sum()) == learned_count
        assert rope.log_frequency_scale.requires_grad == (
            learned_count > 0
        )


def test_native_and_evq_control_grids_are_distinct_and_monotone() -> None:
    protocol = Protocol()
    native = native_inv_freq(protocol)
    evq = evq_inv_freq(protocol)
    assert not torch.equal(native, evq)
    assert torch.all(native[:-1] > native[1:])
    assert torch.all(evq[:-1] > evq[1:])
