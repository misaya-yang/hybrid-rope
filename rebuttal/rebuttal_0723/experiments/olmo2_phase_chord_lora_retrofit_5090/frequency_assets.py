"""Strict loader for frozen phase-chord frequency assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED = {
    "Native": "dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34",
    "anchored_evq_cosh_tau_2": "9e82c83312b6f5f44c63f6ad4fea8fbdd1256a64b3bc853ae8e52c5538a4d02d",
    "phase_chord_olmo_r0_lambda_0p1": "4d985cce3c47506079119d9d0454d02a49d753238bce86ea4bf0f0b2e398e931",
}
PAIR_COUNT = 64


def float32_sha256(values: Any) -> str:
    array = np.asarray(values, dtype="<f4")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_frequency_assets(manifest_path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(manifest_path.resolve().read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    native_record = payload.get("native")
    if not isinstance(candidates, dict) or not isinstance(native_record, dict):
        raise ValueError("target manifest lacks native/candidate tables")
    native_values = native_record.get("inv_freq")
    if float32_sha256(native_values) != EXPECTED["Native"]:
        raise ValueError("Native frequency hash drift")
    native = np.asarray(native_values, dtype="<f4")
    if native.shape != (PAIR_COUNT,) or not np.isfinite(native).all() or not (native > 0).all():
        raise ValueError("Native frequency shape/positivity drift")
    if not np.all(native[:-1] > native[1:]):
        raise ValueError("Native frequencies are not strictly decreasing")
    result = {"Native": native.copy()}
    for name in ("anchored_evq_cosh_tau_2", "phase_chord_olmo_r0_lambda_0p1"):
        record = candidates.get(name)
        if not isinstance(record, dict) or "inv_freq" not in record:
            raise ValueError(f"missing frozen frequency asset: {name}")
        values = np.asarray(record["inv_freq"], dtype="<f4")
        if values.shape != (PAIR_COUNT,) or not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError(f"{name} frequency shape/positivity drift")
        if not np.all(values[:-1] > values[1:]):
            raise ValueError(f"{name} frequencies are not strictly decreasing")
        if not np.array_equal(values[[0, -1]], native[[0, -1]]):
            raise ValueError(f"{name} endpoint drift")
        if float32_sha256(values) != EXPECTED[name]:
            raise ValueError(f"{name} frequency hash drift")
        result[name] = values.copy()
    if set(candidates) - set(result):
        # Extra manifest candidates are harmless only if they are not selected;
        # the loader never reconstructs or approximates them.
        pass
    return result
