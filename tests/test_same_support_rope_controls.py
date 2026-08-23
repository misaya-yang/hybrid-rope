from __future__ import annotations

import numpy as np
import torch

from scripts.analysis.export_uniqueness_budgeted_tables import (
    EXPECTED_DEFAULT_HASHES,
    native_endpoint_inv_freq,
)
from scripts.analysis.rope_transport.same_support_controls import (
    build_same_support_controls,
)


def _qwen_native() -> np.ndarray:
    indices = torch.arange(0, 128, 2, dtype=torch.float32)
    values = 1.0 / torch.pow(
        torch.tensor(1_000_000.0, dtype=torch.float32),
        indices / 128.0,
    )
    return values.numpy().astype(np.float64)


def test_olmo_phase_resolved_control_preserves_registered_s4() -> None:
    native = native_endpoint_inv_freq()
    controls = build_same_support_controls(native, native_context_length=4096)
    table, receipt = controls["converged_budgeted_s4"]
    assert receipt["support_points"] == 2048
    assert receipt["active_sha256_float32"] == EXPECTED_DEFAULT_HASHES[4.0]
    assert table[0] == np.float32(native[0])
    assert table[-1] == np.float32(np.float32(native[-1]) / 4.0)


def test_qwen_controls_are_same_support_monotone_and_label_free() -> None:
    native = _qwen_native()
    controls = build_same_support_controls(native, native_context_length=32768)
    hashes = set()
    for table, receipt in controls.values():
        assert table.dtype == np.dtype("float32")
        assert np.all(table[:-1] > table[1:])
        assert table[0] == np.float32(native[0])
        assert table[-1] == np.float32(np.float32(native[-1]) / 4.0)
        hashes.add(receipt["active_sha256_float32"])
    assert len(hashes) == 3
    corrected = controls["converged_budgeted_s4"][1]
    assert corrected["support_points"] == 16384
    assert corrected["order_crossing_indices"] == []
    ramp = controls["nearest_yarn_ramp_s4"][1]
    assert ramp["selection_uses_task_labels"] is False
    assert ramp["movement_mse_to_uniqueness"] < 0.001
