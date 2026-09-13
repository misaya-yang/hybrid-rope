import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_DIR = Path(__file__).parents[1] / "experiments" / "llama3_60dir_20260911"


def load_runner(monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE_DIR))
    sys.modules.pop("llama_runner", None)
    return importlib.import_module("llama_runner")


def test_static_table_compiles_exact_values_and_gain(monkeypatch):
    runner = load_runner(monkeypatch)
    geometry = runner.O.Geometry.from_native(window=8192, theta=500000.0, scale=8.0)
    values = (geometry.omega * np.power(8.0, -np.linspace(0.0, 1.0, 64))).astype(np.float32)
    op = runner.build_static_table_operator(
        {"table": {"values_float32": values.tolist(), "gain": 1.17}}, geometry, "Candidate"
    )
    assert op.nu() == pytest.approx(values)
    assert op.q_amp(np.array([0.0]))[0, 0] == pytest.approx(1.17)
    assert op.scope == "frequency"


def test_static_table_rejects_bad_order_or_label_collision(monkeypatch):
    runner = load_runner(monkeypatch)
    geometry = runner.O.Geometry.from_native(window=8192, theta=500000.0, scale=8.0)
    bad = geometry.omega.astype(np.float32)
    bad[2] = bad[1]
    with pytest.raises(ValueError, match="invalid explicit"):
        runner.build_static_table_operator(
            {"values_float32": bad.tolist(), "gain": 1.1}, geometry, "Candidate"
        )
    with pytest.raises(ValueError, match="collides"):
        runner.build_static_table_operator(
            {"values_float32": geometry.omega.tolist(), "gain": 1.1}, geometry, "BM"
        )
