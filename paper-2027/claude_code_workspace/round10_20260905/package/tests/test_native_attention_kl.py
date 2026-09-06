"""Small numerical fixtures only; no model/hook/GPU or real result inputs."""

import numpy as np
import pytest

from scripts.analysis.native_attention_kl import native_attention_kl


def fixture():
    rng = np.random.default_rng(217)
    q, k = rng.normal(size=(4, 3, 8)), rng.normal(size=(2, 7, 8))
    return q, k, np.array([0, 3, 6]), np.array([1., .3, .09, .02])


def rotate(values, positions, inv):
    phase = np.asarray(positions)[..., None] * inv
    cosine, sine = np.cos(phase), np.sin(phase)
    left, right = np.split(values, 2, axis=-1)
    return np.concatenate((left * cosine - right * sine, right * cosine + left * sine), axis=-1)


def log_softmax(values):
    shifted = values - np.max(values)
    return shifted - np.log(np.exp(shifted).sum())


def brute_kl(q, k, positions, native, active, gain, scale):
    result = np.zeros(q.shape[:2])
    repeated_keys = np.repeat(k, q.shape[0] // k.shape[0], axis=0)
    for head in range(q.shape[0]):
        for qi, position in enumerate(positions):
            prefix = repeated_keys[head, :position + 1]
            key_positions = np.arange(position + 1)
            native_logits = rotate(prefix, key_positions, native) @ rotate(q[head, qi], position, native) * scale
            candidate_logits = ((gain * rotate(prefix, key_positions, active)) @
                                (gain * rotate(q[head, qi], position, active))) * scale
            logp, logc = log_softmax(native_logits), log_softmax(candidate_logits)
            result[head, qi] = np.exp(logp) @ (logp - logc)
    return result


def test_native_identity_is_strict_zero_and_float64_without_mutation():
    q, k, positions, inv = fixture()
    q, k = q.astype(np.float32), k.astype(np.float32)
    before_q, before_k = q.copy(), k.copy()
    output = native_attention_kl(q, k, positions, inv, inv.copy())
    for value in output.values():
        if isinstance(value, np.ndarray):
            assert value.dtype == np.float64
            assert np.count_nonzero(value) == 0
    assert output["mean_kl"] == 0
    assert np.array_equal(q, before_q) and np.array_equal(k, before_k)


def test_relative_phase_matches_explicit_split_half_rotation_and_gqa():
    q, k, positions, inv = fixture()
    active = inv * np.array([1, .8, .7, .5])
    output = native_attention_kl(q, k, positions, inv, active, gain=1.13, attention_scale=.37)
    expected = brute_kl(q, k, positions, inv, active, 1.13, .37)
    np.testing.assert_allclose(output["kl"], expected, atol=2e-14, rtol=1e-12)
    assert output["mean_kl"] == pytest.approx(expected.mean())
    assert output["slot_logit_delta_mean"].shape == (4, 3, 4)
    assert np.all(output["kl"][:, 0] == 0)  # One causal key always has probability one.


def test_future_keys_do_not_change_any_output():
    q, k, positions, inv = fixture()
    positions = np.array([0, 1, 3])
    before = native_attention_kl(q, k, positions, inv, inv * .5, gain=1.1)
    changed = k.copy()
    changed[:, 4:] = 1e100
    after = native_attention_kl(q, changed, positions, inv, inv * .5, gain=1.1)
    for key in before:
        np.testing.assert_equal(before[key], after[key])


def test_gain_only_has_squared_logit_gain_and_no_missing_component():
    q, k, positions, inv = fixture()
    output = native_attention_kl(q, k, positions, inv, inv, gain=1.4)
    expected = brute_kl(q, k, positions, inv, inv, 1.4, 1 / np.sqrt(8))
    np.testing.assert_allclose(output["kl"], expected, atol=2e-14)
    np.testing.assert_allclose(output["gain_logit_delta_mean"], output["slot_logit_delta_mean"].sum(axis=-1), atol=1e-14)
    np.testing.assert_allclose(output["gain_logit_delta_variance"], output["total_logit_delta_variance"], atol=1e-13)
    assert output["mean_kl"] > 0
    assert output["gain_logit_multiplier"] == pytest.approx(1.96)


def test_native_p_weighted_slot_moments_and_covariance_identity():
    q, k, positions, inv = fixture()
    active, gain, scale = inv * .6, 1.2, .4
    output = native_attention_kl(q, k, positions, inv, active, gain=gain, attention_scale=scale)
    head, qi, position = 2, 1, int(positions[1])
    keys = k[1, :position + 1]
    native_q, native_k = rotate(q[head, qi], position, inv), rotate(keys, np.arange(position + 1), inv)
    active_q, active_k = gain * rotate(q[head, qi], position, active), gain * rotate(keys, np.arange(position + 1), active)
    native_slots = scale * (native_q[:4] * native_k[:, :4] + native_q[4:] * native_k[:, 4:])
    active_slots = scale * (active_q[:4] * active_k[:, :4] + active_q[4:] * active_k[:, 4:])
    probabilities = np.exp(log_softmax(native_slots.sum(axis=1)))
    delta = active_slots - native_slots
    mean = probabilities @ delta
    centered = delta - mean
    covariance = (centered * probabilities[:, None]).T @ centered
    np.testing.assert_allclose(output["slot_logit_delta_mean"][head, qi], mean, atol=2e-14)
    np.testing.assert_allclose(output["slot_logit_delta_variance"][head, qi], np.diag(covariance), atol=2e-14)
    assert output["total_logit_delta_variance"][head, qi] == pytest.approx(covariance.sum())
    assert output["off_diagonal_cancellation"][head, qi] == pytest.approx(covariance.sum() - np.trace(covariance))


def test_cancelling_slots_can_have_positive_variance_but_zero_total_kl():
    inv = np.array([.1, .2])
    positions, relative = np.array([3]), np.array([3, 2, 1, 0])
    x = np.array([-1., 0., 1., 2.])
    k = np.zeros((1, 4, 4))
    k[0, :, 0] = x / np.cos(relative * inv[0])
    k[0, :, 1] = -x / np.cos(relative * inv[1])
    output = native_attention_kl(np.array([[[1., 1., 0., 0.]]]), k, positions, inv, inv, gain=1.2)
    assert output["mean_kl"] < 1e-14
    assert output["total_logit_delta_variance"][0, 0] < 1e-25
    assert np.all(output["slot_logit_delta_variance"][0, 0] > 0)
    assert output["off_diagonal_cancellation"][0, 0] < 0


def test_large_logits_remain_finite_without_logging_zero_probabilities():
    q, k, positions, inv = fixture()
    output = native_attention_kl(q * 100, k * 100, positions, inv, inv * .3, gain=1.1)
    assert np.isfinite(output["kl"]).all()
    assert np.all(output["kl"] >= 0)


@pytest.mark.parametrize("bad", ["heads", "odd", "positions", "float_positions", "negative_position", "frequency", "nan", "complex", "gain", "scale"])
def test_invalid_inputs_fail_closed(bad):
    q, k, positions, inv = fixture()
    active, options = inv.copy(), {}
    if bad == "heads":
        q = q[:3]
    elif bad == "odd":
        q, k = q[..., :-1], k[..., :-1]
    elif bad == "positions":
        positions[-1] = 7
    elif bad == "float_positions":
        positions = positions.astype(float)
    elif bad == "negative_position":
        positions[0] = -1
    elif bad == "frequency":
        active[0] = 0
    elif bad == "nan":
        q[0, 0, 0] = np.nan
    elif bad == "complex":
        q = q.astype(complex) + 1j
    elif bad == "gain":
        options["gain"] = np.inf
    else:
        options["attention_scale"] = 0
    with pytest.raises(ValueError):
        native_attention_kl(q, k, positions, inv, active, **options)
