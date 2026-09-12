#!/usr/bin/env python3
"""Small CPU tests for E2 table identities, whole-response F1, and macro weighting."""

from __future__ import annotations

import unittest

from experiments.rope_fast_5090_20260912.e2_prepare import HISTORICAL_TABLE_SHA256, build_tables
from experiments.rope_fast_5090_20260912.e2_score import TASKS, holm, macro
from scripts.eval.longbench_metrics import qa_f1_score


class E2Tests(unittest.TestCase):
    def test_reused_table_identities_and_gain_controls(self):
        tables = build_tables()
        self.assertEqual(tables["native_g1"]["tensor_sha256"], HISTORICAL_TABLE_SHA256["Native"])
        self.assertEqual(tables["mrpro_g4"]["tensor_sha256"], HISTORICAL_TABLE_SHA256["MrPro"])
        self.assertEqual(tables["bm_g4"]["tensor_sha256"], HISTORICAL_TABLE_SHA256["MrProBM"])
        self.assertEqual(tables["bm_g4"]["tensor_sha256"], tables["bm_g1"]["tensor_sha256"])
        self.assertNotEqual(tables["bm_g4"]["gain"], tables["bm_g1"]["gain"])

    def test_f1_scores_whole_response_without_substring_shortcut(self):
        refs = ["PewDiePie"]
        self.assertEqual(qa_f1_score("PewDiePie", refs), 1.0)
        self.assertLess(qa_f1_score("The answer is PewDiePie", refs), 1.0)

    def test_task_equal_macro_does_not_row_weight_tasks(self):
        data, ids = {}, []
        for task_index, task in enumerate(TASKS):
            count = 10 if task_index == 0 else 1
            for index in range(count):
                row_id = f"{task}_{index}"
                data[row_id] = {"task": task, "whole_response_f1": float(task_index == 0)}
                ids.append(row_id)
        value, by_task = macro(data, ids)
        self.assertEqual(value, 1 / 5)
        self.assertEqual(by_task[TASKS[0]], 1.0)

    def test_holm_is_step_down_and_never_reduces_p(self):
        adjusted = holm({"a": 0.01, "b": 0.04})
        self.assertEqual(adjusted, {"a": 0.02, "b": 0.04})


if __name__ == "__main__":
    unittest.main()
