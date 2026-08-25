#!/usr/bin/env python3
"""
Focused unit tests for scripts/lib/rope/fixed_support_z.py.

The module is the frequency-table authority for the frozen-checkpoint
fixed-support allocation; its at-native decision lives in tensor space so the
realization stays fullgraph-compilable.  These tests pin that contract so the
default `pytest tests/` route and the CI smoke job exercise it:

  - at-native branch: exact Native table with a nonzero straight-through grad
  - computed branch: exact Native endpoints, strict ordering, interior shift
  - fullgraph torch.compile of realized_inv_freq in both branches
"""

from __future__ import annotations

import os
import sys

import torch

# Add project root to path so we can import scripts.lib.rope
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.lib.rope.fixed_support_z import FixedSupportZRotaryEmbedding


def native_table() -> torch.Tensor:
    return 1.0 / (
        500_000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128.0)
    )


def make_module() -> FixedSupportZRotaryEmbedding:
    return FixedSupportZRotaryEmbedding(native_table())


def test_at_native_branch_returns_exact_native_with_straight_through_grad() -> None:
    module = make_module()
    realized = module.realized_inv_freq()
    assert torch.equal(realized, module.native_inv_freq)
    # The released tensor is exact, but the first optimizer step must still
    # see a nonzero gradient through the gap logits.
    loss = realized.sum()
    loss.backward()
    assert module.gap_delta_logits.grad is not None
    assert bool((module.gap_delta_logits.grad != 0.0).any())


def test_computed_branch_keeps_exact_endpoints_and_ordering() -> None:
    module = make_module()
    module.set_gap_delta_(torch.linspace(-1.0, 1.0, module.pair_count - 1))
    active = module.realized_inv_freq()
    native = module.native_inv_freq
    assert torch.equal(active[[0, -1]], native[[0, -1]])
    assert bool(torch.all(active[:-1] > active[1:]))
    assert not torch.equal(active, native)


def test_realization_is_fullgraph_compilable_in_both_branches() -> None:
    module = make_module()
    compiled = torch.compile(module.realized_inv_freq, backend="eager", fullgraph=True)
    assert torch.equal(compiled(), module.native_inv_freq)
    module.set_gap_delta_(torch.linspace(-1.0, 1.0, module.pair_count - 1))
    assert torch.equal(compiled(), module.realized_inv_freq())


def test_forward_shapes_and_batch_broadcast() -> None:
    module = make_module()
    value = torch.zeros(2, 8, 4)
    position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
    cos, sin = module.forward(value, position_ids)
    assert cos.shape == (2, 8, 2 * module.pair_count)
    assert sin.shape == cos.shape
    single, _ = module.forward(value, torch.arange(8))
    assert torch.equal(single[0], cos[0])
