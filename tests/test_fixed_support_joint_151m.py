from __future__ import annotations

import torch

from experiments.fixed_support_joint_151m_20260912.learnable_rope import (
    JointFixedSupportRotaryEmbedding,
)
from experiments.fixed_support_joint_151m_20260912.protocol import (
    ARMS,
    SPEC,
    SUPPORTS,
    cosh_table,
    geo_table,
    learning_rate,
    table_for,
)


def test_protocol_is_exact_one_billion_trajectory_over_two_frozen_epochs() -> None:
    assert SPEC.rows_per_epoch == 244_128
    assert SPEC.optimizer_steps == 15_258
    assert SPEC.midpoint_step == 7_629
    assert SPEC.train_tokens == 999_948_288
    assert learning_rate(0) > 0.0
    assert learning_rate(SPEC.optimizer_steps - 1) == SPEC.min_learning_rate


def test_fixed_tables_share_support_and_are_distinct() -> None:
    for support in SUPPORTS:
        geo = geo_table(support)
        cosh = cosh_table(support)
        assert geo.shape == cosh.shape == (32,)
        assert torch.equal(geo[[0, -1]], cosh[[0, -1]])
        assert bool(torch.all(geo[:-1] > geo[1:]))
        assert bool(torch.all(cosh[:-1] > cosh[1:]))
        assert not torch.equal(geo, cosh)
        assert torch.equal(table_for("full_z", support), geo)
    assert set(ARMS) == {"geo", "cosh", "full_z"}


def test_full_z_exact_geo_initialization_gradient_and_update() -> None:
    initial = geo_table(500_000)
    rope = JointFixedSupportRotaryEmbedding(initial)
    assert torch.equal(rope.realized_inv_freq(), initial)
    loss = (rope.realized_inv_freq() * torch.arange(32.0)).sum()
    loss.backward()
    assert rope.gap_delta_logits.grad is not None
    assert bool((rope.gap_delta_logits.grad != 0.0).any())
    optimizer = torch.optim.AdamW(
        [rope.gap_delta_logits], lr=6e-4, weight_decay=0.0
    )
    optimizer.step()
    rope.fix_gauge_()
    active = rope.realized_inv_freq()
    assert torch.equal(active[[0, -1]], initial[[0, -1]])
    assert bool(torch.all(active[:-1] > active[1:]))
    assert not torch.equal(active, initial)
    assert abs(float(rope.gap_delta_logits.detach().mean())) < 1e-8


def test_full_z_forward_keeps_autograd_graph() -> None:
    rope = JointFixedSupportRotaryEmbedding(geo_table(2_048))
    cos, sin = rope(32)
    objective = cos[1:].sum() + sin[1:].sum()
    objective.backward()
    assert rope.gap_delta_logits.grad is not None
    assert bool(torch.isfinite(rope.gap_delta_logits.grad).all())
    assert float(torch.linalg.vector_norm(rope.gap_delta_logits.grad)) > 0.0
