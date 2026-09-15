import numpy as np
import pytest

from experiments.native_contrastive_proximal_20260915.tables import (
    EXPECTED_OLMO_TABLE_SHA256,
    build_audit,
    build_ncp_arrays,
    tensor_sha256,
)
from scripts.experiments.cross_audit.tables import native_table


def olmo_native() -> np.ndarray:
    return native_table(128, 500_000.0).astype(np.float32)


def test_ncp_reproduces_the_frozen_olmo_table_and_reference_risk():
    result = build_ncp_arrays(olmo_native(), native_length=4096)
    assert tensor_sha256(result["candidate"]) == EXPECTED_OLMO_TABLE_SHA256
    assert result["changed_indices"].size == 46
    assert int(np.argmax(result["log_shifts"])) == 33
    assert np.max(1.0 - result["candidate"] / result["native"]) == pytest.approx(
        0.167164, abs=1e-6,
    )
    assert np.mean(result["reference_risk_native"]) == pytest.approx(0.614081711, abs=2e-9)
    assert np.mean(result["reference_risk_candidate"]) == pytest.approx(0.609641282, abs=2e-9)


def test_ncp_preserves_native_support_gain_contract_and_gap():
    result = build_ncp_arrays(olmo_native(), native_length=4096)
    assert np.array_equal(result["candidate"][[0, -1]], result["native"][[0, -1]])
    assert np.all(result["candidate"] <= result["native"])
    assert np.all(result["candidate"][:-1] > result["candidate"][1:])
    assert result["minimum_log_gap_ratio"] == pytest.approx(0.768076, abs=2e-6)
    assert all(result["checks"].values())
    audit = build_audit(result, model_id="olmo2_1b", config_sha256="a" * 64)
    assert audit["public_inputs_only"] is True
    assert audit["model_execution"] is False
    assert audit["checks"]["frozen_olmo_table_identity"] is True
    assert "does not establish" in audit["claim_boundary"]


def test_ncp_rejects_invalid_native_input():
    with pytest.raises(ValueError, match="strictly decreasing"):
        build_ncp_arrays([1.0, 0.5, 0.5, 0.1], native_length=4096)
    with pytest.raises(ValueError, match="native_length"):
        build_ncp_arrays([1.0, 0.5, 0.25, 0.1], native_length=1)

