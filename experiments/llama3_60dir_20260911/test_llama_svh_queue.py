"""Offline contract tests for the post-P_L0 Llama S queue.

These tests never contact SSH, import torch, or launch the runner.
"""

import json
from pathlib import Path
import sys
import os
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama_svh_queue as Q


HERE = Path(__file__).resolve().parent


class SQueueContractTests(unittest.TestCase):
    def test_plan_is_ordered_and_complete_for_corrected_run_ledger(self):
        units = Q.build_units(HERE)
        Q.validate_units(units)
        ids = [u["unit_id"] for u in units]
        # Matched controls are inserted before the join, so use semantic checks
        # rather than a brittle fixed join index.
        self.assertEqual(ids[:6], [
            "S_NATIVE", "S_MR", "S_YARN", "S_BM", "S_UNI",
            "S_RESONANCE_YARN"])
        join_index = ids.index("S_L0_CONTROLS")
        self.assertEqual(join_index, len(Q.CONTROL_ORDER))
        module_ids = ids[join_index + 1:join_index + 1 + len(Q.CONTROLLED_MODULE_ORDER)]
        self.assertEqual(module_ids, [f"S_{x}" for x in Q.CONTROLLED_MODULE_ORDER])
        candidate_ids = ids[join_index + 1 + len(Q.CONTROLLED_MODULE_ORDER):]
        self.assertEqual(candidate_ids, [
            "S_D01a", "S_D03a", "S_D04a", "S_D05a", "S_D06a", "S_D07a",
            "S_D13a", "S_D03b", "S_D05b", "S_D06b", "S_D07b", "S_D13b",
            "S_D05c", "S_D06c", "S_D07c", "S_D13c"])
        self.assertTrue(all(u["status"] == "PENDING" for u in units))
        d01 = next(u for u in units if u["unit_id"] == "S_D01a")
        self.assertEqual(d01["parent_controls"], ["MR", "ResonanceYaRN"])
        d07 = next(u for u in units if u["unit_id"] == "S_D07a")
        self.assertIn("M-dev", d07["interpretation_limit"])

    def test_plan_never_contains_old_model_commands(self):
        units = Q.build_units(HERE)
        blob = json.dumps(units, ensure_ascii=False).lower()
        self.assertNotIn("olmo", blob)
        self.assertNotIn("qwen", blob)

    def test_atomic_state_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            Q.atomic_json(path, {"status": "RUNNING", "n": 1})
            self.assertEqual(Q.read_json(path)["status"], "RUNNING")
            self.assertFalse(list(Path(td).glob("*.tmp")))

    def test_model_identity_rejects_wrong_architecture(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "Meta-Llama-3-8B-Instruct"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"model_type": "qwen"}))
            with self.assertRaises(RuntimeError):
                Q.model_identity(model)

    def test_failed_unit_consumes_exactly_one_retry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path = root / "state.json"
            unit = Q.make_unit("S_X", "control", "X", ["P_L0"], "P_L0",
                               "test", "test", "retry once", None)
            state = {"units": {"S_X": unit}}
            Q.atomic_json(state_path, state)
            command = ["/bin/sh", "-c", "exit 7"]
            self.assertFalse(Q.run_unit(root, state, state_path, unit, command,
                                        os.environ.copy()))
            self.assertEqual(state["units"]["S_X"]["status"],
                             "BLOCKED_ENGINEERING")
            log = (root / "logs" / "S_X.log").read_text()
            self.assertEqual(log.count("=== attempt"), 2)


if __name__ == "__main__":
    unittest.main()
