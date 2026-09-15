from __future__ import annotations

from fractions import Fraction

import numpy as np

from experiments.fixed_rope_three_interfaces_20260913.matched_naturalqa_report import (
    TASKS,
    bootstrap,
    health,
    pooled_cluster_bootstrap,
)
from experiments.iclr2027_three_track_sprint_20260915.verify_sprint_math import (
    check_n,
    check_quadratic_identity,
)
from experiments.iclr2027_three_track_sprint_20260915.verify_theory_deepening import CHECKS


def test_exact_sprint_relations_include_small_n_degeneracy():
    assert check_n(1)["same_for_n_le_2"]
    assert check_n(2)["same_for_n_le_2"]
    assert not check_n(17)["same_for_n_le_2"]
    check_quadratic_identity(17)


def synthetic_natural_rows():
    panel = {}
    baseline = {}
    candidate = {}
    for task_index, task in enumerate(TASKS):
        for row_index in range(3):
            row_id = f"{task}-{row_index}"
            cluster = f"{task}-document-{row_index // 2}"
            panel[row_id] = {"task": task, "document_cluster_id": cluster}
            common = {
                "task": task,
                "ended_eos": True,
                "hit_cap": False,
                "empty": False,
            }
            baseline[row_id] = {**common, "whole_response_f1": 0.25}
            candidate[row_id] = {**common, "whole_response_f1": 0.5}
    return panel, candidate, baseline


def test_naturalqa_bootstraps_preserve_constant_clustered_delta():
    panel, candidate, baseline = synthetic_natural_rows()
    ids = list(panel)
    task_equal = bootstrap(panel, candidate, baseline, ids, draws=200, seed=7)
    question_equal = pooled_cluster_bootstrap(panel, candidate, baseline, ids, draws=200, seed=8)
    assert task_equal["bootstrap_mean"] == 0.25
    assert task_equal["ci95"] == [0.25, 0.25]
    assert question_equal["estimate"] == 0.25
    assert question_equal["ci95"] == [0.25, 0.25]


def test_naturalqa_health_is_split_by_task_without_dropping_rows():
    panel, candidate, _ = synthetic_natural_rows()
    ids = list(panel)
    result = health(candidate, ids)
    assert {key: result[key] for key in ("rows", "ended_eos", "hit_cap", "empty")} == {
        "rows": 15, "ended_eos": 15, "hit_cap": 0, "empty": 0,
    }
    assert all(result["by_task"][task]["rows"] == 3 for task in TASKS)


def test_theory_deepening_cpu_checks_cover_all_fifteen_classes():
    checks = [function() for function in CHECKS]
    assert len(checks) == 15
    assert all(check["status"] == "PASS" for check in checks)
