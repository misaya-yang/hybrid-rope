import math

import numpy as np
import pytest
import torch

from experiments.checkpoint_attention_replay_20260913.capture_io import (
    load_capture,
    save_capture,
)
from experiments.checkpoint_attention_replay_20260913.capture_checkpoint import (
    annotated_queries,
    balanced_rows,
    head_layout,
    query_positions,
)
from experiments.checkpoint_attention_replay_20260913.core import (
    ReplayCapture,
    evaluate_replay_objective,
    finite_rho_grid,
    increments_to_exponents,
)
from experiments.checkpoint_attention_replay_20260913.solver import (
    project_simplex,
    solve_increment_qp,
)
from scripts.analysis.native_attention_kl import native_attention_kl
from experiments.checkpoint_attention_replay_20260913.signed_phase_response import (
    analyze_capture,
)


def fixture(group="layer0"):
    rng = np.random.default_rng(903)
    return ReplayCapture(
        q=rng.normal(size=(4, 2, 8)).astype(np.float32),
        k=rng.normal(size=(2, 7, 8)).astype(np.float32),
        query_positions=np.array([3, 6], dtype=np.int64),
        native_inv_freq=np.array([1.0, 0.3, 0.08, 0.02], dtype=np.float32),
        attention_scale=1 / math.sqrt(8),
        reference_gain=1.13,
        group=group,
        row_id="synthetic-row",
        layer=0,
    )


def rotate(values, positions, inv):
    phase = np.asarray(positions)[..., None] * inv
    cosine, sine = np.cos(phase), np.sin(phase)
    left, right = np.split(values, 2, axis=-1)
    return np.concatenate((left * cosine - right * sine, right * cosine + left * sine), axis=-1)


def log_softmax(values):
    shifted = values - np.max(values)
    return shifted - np.log(np.exp(shifted).sum())


def brute_scaled_kl(capture, active, gain, rho):
    repeated = np.repeat(capture.k, capture.q.shape[0] // capture.k.shape[0], axis=0)
    result = np.zeros(capture.q.shape[:2])
    for head in range(capture.q.shape[0]):
        for query_index, position in enumerate(capture.query_positions):
            keys = repeated[head, : position + 1]
            key_positions = np.arange(position + 1)
            reference = (
                capture.reference_gain * rotate(keys, key_positions, capture.native_inv_freq)
            ) @ (
                capture.reference_gain
                * rotate(capture.q[head, query_index], position, capture.native_inv_freq)
            ) * capture.attention_scale
            candidate = (
                gain * rotate(keys, rho * key_positions, active)
            ) @ (
                gain * rotate(capture.q[head, query_index], rho * position, active)
            ) * capture.attention_scale
            log_reference, log_candidate = log_softmax(reference), log_softmax(candidate)
            result[head, query_index] = np.exp(log_reference) @ (log_reference - log_candidate)
    return result


def test_scaled_replay_matches_split_half_causal_gqa_and_gain_squared():
    capture = fixture()
    active = capture.native_inv_freq * np.array([1.0, 0.8, 0.6, 0.4])
    actual = native_attention_kl(
        capture.q,
        capture.k,
        capture.query_positions,
        capture.native_inv_freq,
        active,
        gain=1.21,
        native_gain=capture.reference_gain,
        attention_scale=capture.attention_scale,
        position_scale=1.7,
    )
    expected = brute_scaled_kl(capture, active, 1.21, 1.7)
    np.testing.assert_allclose(actual["kl"], expected, atol=3e-14, rtol=2e-12)
    assert actual["gain_logit_multiplier"] == pytest.approx(1.21 ** 2)
    assert actual["native_gain_logit_multiplier"] == pytest.approx(1.13 ** 2)


def test_native_zero_point_and_group_constraints_are_exact():
    captures = [fixture("early"), fixture("late")]
    increments = np.zeros(3)
    result = evaluate_replay_objective(
        captures,
        increments,
        scale=4.0,
        gain=1.13,
        rhos=[1.0],
        native_mean_limit=0.0,
        native_group_cvar_limit=0.0,
    )
    assert result["finite_grid_worst_mean_kl"] == 0.0
    assert result["native_constraints"]["holds"] is True
    assert set(result["risk_by_rho"]["1"]["groups"]) == {"early", "late"}
    assert result["profile"]["zero_increment_indices"] == [0, 1, 2]


def test_rho_grid_and_zero_increment_profile_contract():
    np.testing.assert_allclose(finite_rho_grid(4.0), [1, math.sqrt(2), 2, 2 * math.sqrt(2), 4])
    np.testing.assert_equal(increments_to_exponents([0.0, 0.4, 0.0]), [0.0, 0.0, 0.4, 0.4])
    with pytest.raises(ValueError):
        increments_to_exponents([0.8, 0.3])


def test_capture_receipt_round_trip(tmp_path):
    capture = fixture()
    receipt = save_capture(tmp_path / "capture", capture)
    restored = load_capture(tmp_path / "capture", mmap_mode="r")
    assert receipt["causal_lag_sign"] == "query_position - key_position"
    assert receipt["rotary_layout"] == "split_half"
    assert receipt["gqa_query_heads_per_kv_head"] == 2
    np.testing.assert_equal(restored.q, capture.q)
    np.testing.assert_equal(restored.k, capture.k)
    q_path = tmp_path / "capture/q.npy"
    payload = bytearray(q_path.read_bytes())
    payload[-1] ^= 1
    q_path.write_bytes(payload)
    with pytest.raises(ValueError, match="hash differs"):
        load_capture(tmp_path / "capture")


def test_qkv_evidence_capture_round_trip_and_signed_response(tmp_path):
    base = fixture()
    q = torch.tensor(base.q).to(torch.bfloat16).float().numpy()
    k = torch.tensor(base.k).to(torch.bfloat16).float().numpy()
    v = torch.arange(2 * 7 * 5, dtype=torch.float32).reshape(2, 7, 5).to(torch.bfloat16).float().numpy() / 8
    capture = ReplayCapture(
        q=q, k=k,
        **{name: getattr(base, name) for name in (
            "query_positions", "native_inv_freq", "attention_scale",
            "reference_gain", "group", "row_id", "layer",
        )},
        v=v,
        query_roles=("relation_write", "final_readout"),
        evidence_key_positions=((1, 2), (1, 4)),
    )
    receipt = save_capture(tmp_path / "qkv", capture)
    assert receipt["status"] == "CHECKPOINT_ATTENTION_QKV_CAPTURE_V2"
    assert receipt["arrays"]["k"]["encoding"] == "bfloat16_bits"
    restored = load_capture(tmp_path / "qkv", mmap_mode="r")
    np.testing.assert_equal(restored.v, capture.v)
    assert restored.query_roles == capture.query_roles
    assert restored.evidence_key_positions == capture.evidence_key_positions
    result = analyze_capture(restored, capture.native_inv_freq.copy(), gain=1.13)
    assert len(result["observations"]) == capture.q.shape[0] * capture.q.shape[1]
    assert max(abs(row["delta_margin"]) for row in result["observations"]) < 1e-12
    assert max(row["delta_output_l2"] for row in result["observations"]) < 1e-12
    np.testing.assert_allclose(result["fmr_linear_contrast_mean"], 0, atol=1e-12)


def test_signed_margin_gradient_matches_finite_difference():
    rng = np.random.default_rng(811)
    capture = ReplayCapture(
        q=rng.normal(size=(1, 1, 6)).astype(np.float32),
        k=rng.normal(size=(1, 5, 6)).astype(np.float32),
        v=rng.normal(size=(1, 5, 4)).astype(np.float32),
        query_positions=np.array([4], dtype=np.int64),
        native_inv_freq=np.array([0.9, 0.2, 0.04], dtype=np.float32),
        attention_scale=1 / math.sqrt(6), reference_gain=1.0,
        group="gradient", row_id="gradient", layer=0,
        query_roles=("final_readout",), evidence_key_positions=((1, 3),),
    )
    native = capture.native_inv_freq.astype(np.float64)
    direction = np.array([0.3, -0.2, 0.1])
    step = 1e-6
    plus = analyze_capture(capture, native * np.exp(-step * direction), gain=1.0)
    minus = analyze_capture(capture, native * np.exp(step * direction), gain=1.0)
    finite = (
        plus["observations"][0]["candidate_margin"]
        - minus["observations"][0]["candidate_margin"]
    ) / (2 * step)
    base = analyze_capture(capture, native, gain=1.0)
    assert finite == pytest.approx(float(base["b_mean"] @ direction), rel=2e-6, abs=2e-8)


def test_closed_simplex_qp_can_create_exact_zero_increments():
    initial = np.full(4, 0.25)
    result = solve_increment_qp(
        initial,
        [0.0, 0.0, -2.0, 0.0],
        np.eye(4),
        tail_depth=1.0,
    )
    np.testing.assert_allclose(result["increments"], [0, 0, 1, 0], atol=1e-10)
    assert result["active_increment_indices"] == [2]
    assert result["zero_increment_indices"] == [0, 1, 3]
    assert result["status"] == "INCREMENT_QP_COMPLETE"
    np.testing.assert_allclose(project_simplex([1, -1, 3], 0.0), 0.0)


def test_finite_difference_matches_autograd_for_dilated_replay():
    rng = np.random.default_rng(44)
    q = rng.normal(size=(1, 1, 4))
    k = rng.normal(size=(1, 5, 4))
    position = np.array([4])
    native = np.array([0.7, 0.09])
    scale, rho, gain = 4.0, 1.6, 1.08

    def numpy_loss(value):
        active = native * np.power(scale, -np.array([0.0, value]))
        return native_attention_kl(
            q, k, position, native, active,
            gain=gain, position_scale=rho,
        )["mean_kl"]

    value, step = 0.35, 1e-6
    finite_difference = (numpy_loss(value + step) - numpy_loss(value - step)) / (2 * step)

    parameter = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    tq, tk = torch.tensor(q[0, 0]), torch.tensor(k[0])
    tinv = torch.tensor(native)
    relative = torch.tensor(position[0] - np.arange(position[0] + 1), dtype=torch.float64)
    a = tq[:2] * tk[:, :2] + tq[2:] * tk[:, 2:]
    b = tq[:2] * tk[:, 2:] - tq[2:] * tk[:, :2]
    native_phase = relative[:, None] * tinv[None, :]
    active_inv = tinv * torch.pow(torch.tensor(scale), -torch.stack((parameter * 0, parameter)))
    active_phase = rho * relative[:, None] * active_inv[None, :]
    factor = 1 / math.sqrt(4)
    native_logits = factor * (a * native_phase.cos() + b * native_phase.sin()).sum(-1)
    active_logits = gain ** 2 * factor * (a * active_phase.cos() + b * active_phase.sin()).sum(-1)
    logp = native_logits.log_softmax(-1)
    autograd_loss = (logp.exp() * (logp - active_logits.log_softmax(-1))).sum()
    autograd_loss.backward()
    assert finite_difference == pytest.approx(float(parameter.grad), rel=2e-6, abs=2e-8)


def test_capture_row_and_query_selection_are_deterministic_and_task_balanced():
    rows = [
        {"task": "b", "row_id": "b0"},
        {"task": "a", "row_id": "a0"},
        {"task": "a", "row_id": "a1"},
        {"task": "b", "row_id": "b1"},
    ]
    assert [row["row_id"] for row in balanced_rows(rows, 3)] == ["a0", "b0", "a1"]
    positions = query_positions(100, 4)
    assert positions.tolist() == [20, 46, 73, 99]
    positions, roles, evidence = annotated_queries({"capture_queries": [
        {"role": "write", "position": 3, "evidence_token_indices": [0, 1]},
        {"role": "read", "position": 6, "evidence_token_indices": [1, 4]},
    ]})
    np.testing.assert_equal(positions, [3, 6])
    assert roles == ("write", "read")
    assert evidence == ((0, 1), (1, 4))


def test_capture_head_layout_accepts_projection_and_both_normalized_layouts():
    projection = torch.arange(2 * 3 * 4).reshape(1, 2, 12)
    expected = projection[0].reshape(2, 3, 4)
    torch.testing.assert_close(head_layout(projection, tokens=2, heads=3, head_dim=4), expected)
    torch.testing.assert_close(
        head_layout(expected.unsqueeze(0), tokens=2, heads=3, head_dim=4), expected,
    )
    torch.testing.assert_close(
        head_layout(expected.permute(1, 0, 2).unsqueeze(0), tokens=2, heads=3, head_dim=4),
        expected,
    )
