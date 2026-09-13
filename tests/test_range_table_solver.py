import numpy as np

from experiments.olmo_recovery_20260912.solve_range_table import (
    combine_full_values_with_stochastic_gradients,
    feasibility_objective,
    lexicographic_objective,
    linear_epigraph_step,
    linear_feasibility_step,
    no_boundary_spike,
    round_rows,
)


def test_round_rows_cycles_two_rows_per_cell():
    rows = [
        {"row_id": f"r_{task}_{index}", "task": task, "length_cap": cap}
        for cap in (4096, 8192)
        for task in ("a", "b")
        for index in range(8)
    ]
    first = round_rows(rows, 0)
    second = round_rows(rows, 1)
    assert len(first) == len(second) == 8
    assert {row["row_id"] for row in first}.isdisjoint(row["row_id"] for row in second)


def test_linear_epigraph_moves_against_the_worst_group():
    values = {"worst": 1.0, "other": 0.0}
    gradients = {"worst": np.asarray([1.0, -1.0, 0.0]), "other": np.asarray([0.0, 0.0, 0.0])}
    jacobian = np.asarray([[1.0, -1.0]])
    step = linear_epigraph_step(values, gradients, jacobian, [])
    assert step[0] < 0.0
    assert step[1] > 0.0
    assert abs(step[0]) <= 0.03 + 1e-12
    assert abs(step[2]) <= 0.02 + 1e-12


def test_lexicographic_objective_is_worst_then_mean():
    assert lexicographic_objective({"a": 0.5, "b": 0.1}) == (0.5, 0.3)


def test_linear_feasibility_step_reduces_violation_and_keeps_hard_constraint():
    boundary = [
        (0.6, np.array([1.0, -1.0, 0.0]), 0.1),
        (0.2, np.array([-0.2, 0.2, 0.0]), 0.1),
    ]
    kl = [(0.0, np.array([0.0, 0.0, 1.0]), 0.0)]
    jacobian = np.array([[1.0, -1.0]])
    step = linear_feasibility_step(boundary, kl, jacobian)
    assert step[0] < 0.0
    assert step[1] > 0.0
    assert step[2] <= 0.0
    assert abs((jacobian @ step[:2]).item()) <= 0.030001
    assert feasibility_objective([(0.6, None, 0.1), (0.2, None, 0.1)]) == (0.5, 0.3)


def test_feasibility_restoration_rejects_a_new_constraint_spike():
    current = [(0.6, None, 0.1), (0.0, None, 0.1)]
    assert no_boundary_spike(current, [(0.4, None, 0.1), (0.04, None, 0.1)])
    assert not no_boundary_spike(current, [(0.4, None, 0.1), (0.06, None, 0.1)])


def test_full_values_are_combined_with_rotating_gradients():
    full = {
        "values": {"task/4096/a": 0.2},
        "cf_regrets": {"cf/4096/a": 0.3},
        "native_kl_gradients": {"0": None},
    }
    directional = {
        "gradients": {
            "task/4096/a": np.ones(3),
            "cf/4096/a": np.ones(3) * 2,
        },
        "native_kl_gradients": {"0": np.ones(3)},
        "cf_seed": 7,
        "selected_row_ids": ["row"],
    }
    combined = combine_full_values_with_stochastic_gradients(full, directional)
    assert combined["values"] is full["values"]
    assert combined["gradients"] is directional["gradients"]
    assert combined["cf_seed"] == 7
