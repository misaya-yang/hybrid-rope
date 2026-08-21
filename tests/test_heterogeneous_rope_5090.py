from __future__ import annotations

import json

import pytest
import torch

from rebuttal.rebuttal_0723.experiments.heterogeneous_rope_5090.analyze_results import (
    analyze_payload,
    classify_profile,
)
from rebuttal.rebuttal_0723.experiments.heterogeneous_rope_5090.dry_run_matrix import (
    build_dry_run_matrix,
)
from rebuttal.rebuttal_0723.experiments.heterogeneous_rope_5090.preflight import (
    run_preflight,
)
from rebuttal.rebuttal_0723.experiments.heterogeneous_rope_5090.protocol import (
    build_layer_inv_freqs,
    load_r0_json,
    plan_from_values,
)


def _tiny_config(attention_type: str = "mha") -> dict:
    config = {
        "vocab_size": 97,
        "hidden_size": 32,
        "num_layers": 3,
        "num_heads": 2,
        "head_dim": 16,
        "intermediate_size": 64,
        "max_position_embeddings": 32,
        "seq_len": 16,
        "batch_size": 2,
        "attn_type": attention_type,
    }
    if attention_type == "mla":
        config.update(
            {
                "d_rope": 8,
                "d_nope": 8,
                "v_head_dim": 16,
                "kv_lora_rank": 8,
            }
        )
    elif attention_type == "gqa":
        config["n_kv_heads"] = 1
    return config


def test_r0_parser_accepts_per_layer_m_and_tau(tmp_path) -> None:
    path = tmp_path / "r0.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "num_layers": 3,
                "attention_type": "mha",
                "head_dim": 16,
                "base": 10_000,
                "train_length": 64,
                "effective_dim": 8,
                "layers": [
                    {"layer": 0, "tau": 0.0},
                    {"layer": 1, "m": 2.0},
                    {"layer": 2, "tau": 1.0},
                ],
            }
        )
    )
    plan = load_r0_json(path)
    assert plan.num_layers == 3
    assert plan.rope_dim == 16
    assert plan.layers[1].multiplier_m == pytest.approx(2.0)
    # tau = m * d_eff / sqrt(L_train) = 2 * 8 / 8.
    assert plan.layers[1].tau == pytest.approx(2.0)
    assert plan.taus == pytest.approx((0.0, 2.0, 1.0))


def test_direct_layer_frequency_rows_are_realized_and_hashed() -> None:
    plan = plan_from_values(
        num_layers=2,
        rope_dim=16,
        inv_freq=[
            [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125],
            [0.9, 0.45, 0.225, 0.1125, 0.05625, 0.028125, 0.0140625, 0.00703125],
        ],
    )
    rows = build_layer_inv_freqs(plan)
    assert len(rows) == 2
    assert rows[0].dtype == torch.float32
    assert rows[0].shape == (8,)
    assert not torch.equal(rows[0], rows[1])


def test_r0_parser_accepts_per_layer_wrapper_and_shared_direct_table(tmp_path) -> None:
    path = tmp_path / "r0_shared.json"
    direct = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
    path.write_text(
        json.dumps(
            {
                "num_layers": 3,
                "rope_dim": 16,
                "per_layer": {"tau": [0.1, 0.2, 0.3]},
            }
        )
    )
    assert load_r0_json(path).taus == pytest.approx((0.1, 0.2, 0.3))
    path.write_text(
        json.dumps(
            {
                "num_layers": 3,
                "rope_dim": 16,
                "shared_inv_freq": direct,
            }
        )
    )
    plan = load_r0_json(path)
    assert all(item.inv_freq == tuple(direct) for item in plan.layers)


def test_mha_shared_tau_forward_parity_and_parameter_contract() -> None:
    plan = plan_from_values(num_layers=3, rope_dim=16, tau=[0.7, 0.7, 0.7])
    receipt = run_preflight(plan, _tiny_config("mha"), seed=7, parity_seq_len=6)
    assert receipt["status"] == "READY_FOR_AUTHORIZED_GATE"
    assert receipt["training_started"] is False
    assert receipt["layerwise_parameter_contract"]["status"] == "PASS"
    assert receipt["forward_parity"]["exact_equal"] is True
    assert receipt["forward_parity"]["max_abs_diff"] == 0.0
    assert receipt["layerwise_installation"]["shared_object_before"] is True
    assert receipt["layerwise_installation"]["unique_objects_after"] == 3
    assert receipt["layerwise_installation"]["attention_kernel_unchanged"] is True
    assert len(receipt["realized_frequency"]["unique_hashes"]) == 1


def test_mla_heterogeneous_tables_have_per_layer_hashes_and_no_parity_claim() -> None:
    plan = plan_from_values(
        num_layers=3,
        rope_dim=8,
        tau=[0.0, 0.7, 1.4],
        attention_type="mla",
        num_heads=2,
        head_dim=16,
        d_rope=8,
        d_nope=8,
    )
    receipt = run_preflight(plan, _tiny_config("mla"), seed=11, parity_seq_len=5)
    assert receipt["status"] == "READY_FOR_AUTHORIZED_GATE"
    assert receipt["forward_parity"]["status"] == "NOT_APPLICABLE_HETEROGENEOUS"
    assert len(receipt["realized_frequency"]["unique_hashes"]) == 3
    assert receipt["per_head_feasibility_gate"]["status"] == "GATED_NOT_IMPLEMENTED"
    assert receipt["per_head_feasibility_gate"]["training_allowed"] is False


def test_dry_run_matrix_has_no_metrics_or_training_authorization() -> None:
    plan = plan_from_values(num_layers=3, rope_dim=16, tau=[0.1, 0.5, 0.1])
    matrix = build_dry_run_matrix(
        plan,
        candidate_profiles={"bimodal": {"kind": "tau", "values": [0.1, 0.8, 0.1]}},
        shared_tau_values=[0.0],
    )
    assert matrix["status"] == "DRY_RUN_ONLY"
    assert matrix["training_started"] is False
    assert matrix["training_authorized"] is False
    assert all(row["metrics"] is None for row in matrix["arms"])
    assert any(row["arm_id"] == "bimodal" for row in matrix["arms"])


def test_direct_inv_freq_plan_is_present_in_dry_run_matrix() -> None:
    plan = plan_from_values(
        num_layers=2,
        rope_dim=8,
        inv_freq=[
            [1.0, 0.5, 0.25, 0.125],
            [0.9, 0.45, 0.225, 0.1125],
        ],
    )
    matrix = build_dry_run_matrix(plan)
    assert matrix["arms"][0]["arm_id"] == "configured_r0"
    assert matrix["arms"][0]["layer_tau"] is None
    assert len(matrix["arms"][0]["layer_inv_freq_sha256_raw"]) == 2


def test_bimodal_screen_and_pareto_frontier_are_descriptive() -> None:
    profile = classify_profile([0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    assert profile["label"] == "bimodal"
    result = analyze_payload(
        {
            "results": [
                {
                    "arm_id": "bimodal",
                    "layer_tau": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                    "metrics": {"in_window": 1.0, "ood": 3.0},
                },
                {
                    "arm_id": "monotonic",
                    "layer_tau": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "metrics": {"in_window": 2.0, "ood": 2.0},
                },
                {
                    "arm_id": "ood_best",
                    "layer_tau": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                    "metrics": {"in_window": 3.0, "ood": 1.0},
                },
            ]
        }
    )
    assert result["status"] == "SCREENING_ONLY"
    assert result["pareto_frontier_indices"] == [0, 1, 2]
    assert result["bimodal_frontier_count"] == 1
    assert result["screening_decision"] == "BIMODAL_FRONTIER_SIGNAL"
    assert result["rows"][0]["profile"]["label"] == "bimodal"
