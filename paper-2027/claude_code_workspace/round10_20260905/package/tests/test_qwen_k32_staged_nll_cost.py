"""CPU-only staged packed-NLL cost gate tests."""

import pytest

from scripts.analysis import estimate_qwen_k32_staged_nll_cost as cost


def canary():
    return {"status": cost.STATUS, "metrics_exposed": False, "rows": 2,
            "batch_seconds": {"32768": 1.0, "65536": 3.0},
            "elapsed_seconds": 4.0, "process_elapsed_seconds": 14.0}


def test_cost_gate_admits_complete_buffered_panel():
    report = cost.estimate(canary(), rmb_per_hour=2, quantum_seconds=60,
                           budget_rmb=8, buffer_fraction=.2)
    assert report["decision"] == "RUN"
    assert report["estimated_primary_seconds"] == pytest.approx(266)
    assert report["estimated_baseline_seconds"] == pytest.approx(138)


def test_cost_gate_rejects_over_budget_panel():
    report = cost.estimate(canary(), rmb_per_hour=100, quantum_seconds=60,
                           budget_rmb=8, buffer_fraction=.2)
    assert report["decision"] == "DO_NOT_RUN"


@pytest.mark.parametrize("mutation", ["metrics", "rows", "time", "quantum", "buffer"])
def test_cost_gate_fails_closed(mutation):
    result = canary()
    kwargs = {"rmb_per_hour": 2, "quantum_seconds": 60,
              "budget_rmb": 8, "buffer_fraction": .2}
    if mutation == "metrics": result["metrics_exposed"] = True
    elif mutation == "rows": result["rows"] = 1
    elif mutation == "time": result["batch_seconds"]["65536"] = float("nan")
    elif mutation == "quantum": kwargs["quantum_seconds"] = 0
    elif mutation == "buffer": kwargs["buffer_fraction"] = 1
    with pytest.raises(ValueError):
        cost.estimate(result, **kwargs)
