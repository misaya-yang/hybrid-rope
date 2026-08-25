from __future__ import annotations

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_allocation_oracle_5090.oracle import (
    NATIVE_LENGTH,
    deterministic_offsets,
    position_ids_for_offsets,
    protocol,
)
from scripts.lib.rope.fixed_support_z import FixedSupportZRotaryEmbedding


def native_table() -> torch.Tensor:
    return 1.0 / (
        500_000.0
        ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128.0)
    )


def test_allocation_starts_exactly_native_and_stays_ordered() -> None:
    native = native_table()
    module = FixedSupportZRotaryEmbedding(native)
    assert torch.equal(module.realized_inv_freq(), native)
    with torch.no_grad():
        module.gap_delta_logits.copy_(torch.linspace(-1.0, 1.0, 63))
    module.project_()
    active = module.realized_inv_freq()
    assert torch.equal(active[[0, -1]], native[[0, -1]])
    assert torch.all(active[:-1] > active[1:])


def test_allocation_realization_is_fullgraph_compilable() -> None:
    module = FixedSupportZRotaryEmbedding(native_table())
    compiled = torch.compile(module.realized_inv_freq, backend="eager", fullgraph=True)
    assert torch.equal(compiled(), module.native_inv_freq)


def test_offset_schedule_is_deterministic_and_covers_registered_shells() -> None:
    first = deterministic_offsets(seed=7, step=11, accumulation=1)
    second = deterministic_offsets(seed=7, step=11, accumulation=1)
    assert np.array_equal(first, second)
    assert sorted(first.tolist())[0] == 0
    assert any(1 <= value <= NATIVE_LENGTH for value in first)
    assert any(NATIVE_LENGTH < value <= 3 * NATIVE_LENGTH for value in first)
    assert any(3 * NATIVE_LENGTH < value <= 15 * NATIVE_LENGTH for value in first)


def test_position_gap_changes_only_query_suffix() -> None:
    starts = np.asarray([5, 7], dtype=np.int64)
    offsets = np.asarray([10, 20], dtype=np.int64)
    positions = position_ids_for_offsets(
        query_starts=starts,
        offsets=offsets,
        sequence_length=16,
    )
    base = np.arange(16)
    assert np.array_equal(positions[0, :5], base[:5])
    assert np.array_equal(positions[0, 5:], base[5:] + 10)
    assert np.array_equal(positions[1, :7], base[:7])
    assert np.array_equal(positions[1, 7:], base[7:] + 20)


def test_protocol_is_one_shared_table_without_target_length() -> None:
    value = protocol(steps=300, seed=1)
    assert value["target_length_used_by_construction"] is False
    assert value["qk_lora"]["rank"] == 64
    assert value["family_pattern"] == ["phase", "phase", "natural"]
