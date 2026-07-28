from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    endpoint_evq_inv_freq,
    endpoint_geo_inv_freq,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.native_protected_evq import (
    ATTENTION_HEADS,
    LAYERS,
    PAIR_COUNT,
    exact_pair_ablation_forward_kl,
    native_protected_evq_inv_freq,
    qk_unprotected_output_mask,
    select_protected_pairs,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_4k_native_protected_evq import (
    protocol_from_args,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    LoRALinear,
)


def test_exact_pair_ablation_kl_matches_brute_force() -> None:
    torch.manual_seed(41)
    query = torch.randn(1, 2, 7, 128)
    key = torch.randn(1, 2, 7, 128)
    positions = torch.tensor([2, 4, 6])
    actual = exact_pair_ablation_forward_kl(
        query,
        key,
        query_positions=positions,
        pair_chunk_size=7,
    )
    scale = 128**-0.5
    q_rows = query[:, :, positions, :]
    logits = q_rows @ key.transpose(-1, -2) * scale
    valid = torch.arange(7)[None, :] <= positions[:, None]
    logits = logits.masked_fill(~valid[None, None, :, :], float("-inf"))
    base_log_prob = torch.log_softmax(logits, dim=-1)
    base_prob = base_log_prob.exp()
    expected = []
    for pair in range(PAIR_COUNT):
        contribution = (
            q_rows[..., pair, None] * key[:, :, None, :, pair]
            + q_rows[..., pair + PAIR_COUNT, None]
            * key[:, :, None, :, pair + PAIR_COUNT]
        ) * scale
        removed = (logits - contribution).masked_fill(
            ~valid[None, None, :, :], float("-inf")
        )
        removed_log_prob = torch.log_softmax(removed, dim=-1)
        difference = torch.where(
            valid[None, None, :, :],
            base_log_prob - removed_log_prob,
            torch.zeros_like(base_log_prob),
        )
        expected.append((base_prob * difference).sum(dim=-1))
    reference = torch.stack(expected, dim=-1)
    assert torch.allclose(actual, reference, atol=2e-5, rtol=2e-5)


def test_stable_concentrated_importance_passes() -> None:
    rng = np.random.default_rng(7)
    values = rng.uniform(
        0.0001,
        0.0002,
        size=(16, LAYERS, ATTENTION_HEADS, PAIR_COUNT),
    )
    values[..., :8] += 1.0
    result = select_protected_pairs(values)
    assert result["passed"] is True
    assert set(result["protected_pair_indices"]).issubset(set(range(8)))
    assert 0 not in result["protected_pair_indices"]
    assert result["already_identical_native_evq_pair_indices"] == [0]
    assert len(result["protected_pair_indices"]) == 6
    assert result["split_score_cosine"] >= 0.98
    assert result["minimum_per_layer_protected_importance_mass"] >= 0.5


def test_diffuse_importance_fails_before_training() -> None:
    values = np.ones((16, LAYERS, ATTENTION_HEADS, PAIR_COUNT))
    result = select_protected_pairs(values)
    assert result["passed"] is False
    assert result["gates"]["aggregate_concentration"] is False
    assert len(result["protected_pair_indices"]) == 16


def test_hybrid_uses_exact_native_and_evq_entries() -> None:
    protected = (2, 7, 19, 41)
    hybrid = native_protected_evq_inv_freq(protected)
    native = endpoint_geo_inv_freq()
    evq = endpoint_evq_inv_freq()
    protected_mask = torch.zeros(PAIR_COUNT, dtype=torch.bool)
    protected_mask[list(protected)] = True
    assert torch.equal(hybrid[protected_mask], native[protected_mask])
    assert torch.equal(hybrid[~protected_mask], evq[~protected_mask])


def test_empty_protected_set_is_exact_full_evq_control() -> None:
    assert torch.equal(
        native_protected_evq_inv_freq(()), endpoint_evq_inv_freq()
    )


def test_qk_mask_disables_both_coordinates_of_protected_pairs() -> None:
    protected = (0, 17, 63)
    config = SimpleNamespace(
        hidden_size=2_048,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=128,
    )
    mask = qk_unprotected_output_mask(config, protected)
    assert int(mask.sum()) == 16 * 2 * (64 - len(protected))
    for head in range(16):
        offset = head * 128
        for pair in protected:
            assert float(mask[offset + pair]) == 0.0
            assert float(mask[offset + 64 + pair]) == 0.0


def test_lora_output_mask_is_exact_forbidden_coordinate_gate() -> None:
    torch.manual_seed(9)
    base = nn.Linear(8, 8, bias=False)
    mask = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.float32)
    module = LoRALinear(base, rank=4, alpha=8.0, output_mask=mask)
    with torch.no_grad():
        module.b.fill_(0.5)
    values = torch.randn(3, 8)
    update = module(values) - base(values)
    assert torch.equal(update[:, mask == 0], torch.zeros(3, 4))


def test_matched_full_evq_control_changes_only_protection(
    tmp_path,
) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text("{}\n", encoding="utf-8")
    base = {
        "steps": 144,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "rank": 512,
        "alpha": 1024.0,
        "learning_rate": 2e-5,
        "warmup_steps": 4,
        "minimum_lr_ratio": 0.9,
        "maximum_gradient_norm": 5.0,
        "attention_weight": 1.0,
        "context_weight": 1.0,
        "seed": 20_260_805,
    }
    protected = protocol_from_args(
        SimpleNamespace(**base, arm="protected"),
        selection_receipt=selection,
        protected_pairs=(3, 11),
    )
    control = protocol_from_args(
        SimpleNamespace(**base, arm="full-evq-control"),
        selection_receipt=selection,
        protected_pairs=(3, 11),
    )
    assert protected["diagnostic_selected_pair_indices"] == [3, 11]
    assert control["diagnostic_selected_pair_indices"] == [3, 11]
    assert protected["protected_native_pair_indices"] == [3, 11]
    assert control["protected_native_pair_indices"] == []
    assert control["frequency_sha256_float32"] != protected[
        "frequency_sha256_float32"
    ]
