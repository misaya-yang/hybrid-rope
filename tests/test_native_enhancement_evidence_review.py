import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.native_enhancement_oral_20260915.evidence_review import paired, raw


def row(i, task, score):
    return dict(row_id=str(i), task=task, length_cap=4096, prompt_sha256=str(i),
                input_tokens=4000, references=["answer"], max_new_tokens=32,
                correct=score, generated_ids=[1], ended_eos=True)


class ExistingEvidenceTests(unittest.TestCase):
    def test_historical_blocks_may_reuse_display_row_id(self):
        a, b = row(1, "a", 1), row(1, "a", 0)
        b["prompt_sha256"] = "different_prompt"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
            self.assertEqual(len(raw([path])), 2)
            path.write_text(json.dumps(a) + "\n" + json.dumps(a) + "\n")
            with self.assertRaises(ValueError):
                raw([path])

    def test_macro_is_task_equal_with_unequal_counts(self):
        left = {str(i): row(i, "a" if i < 3 else "b", 1 if i < 3 else 0) for i in range(4)}
        right = {k: dict(v, correct=0) for k, v in left.items()}
        result = paired(left, right, score_field="correct")
        self.assertEqual(result["by_length"]["4096"]["delta"], .5)

    def test_no_silent_intersection_or_identity_drift(self):
        a = {"1": row(1, "a", 1)}
        b = copy.deepcopy(a)
        b["2"] = row(2, "a", 0)
        with self.assertRaises(ValueError):
            paired(a, b, score_field="correct")
        b = copy.deepcopy(a)
        b["1"]["references"] = ["other"]
        with self.assertRaises(ValueError):
            paired(a, b, score_field="correct")

    def test_recorded_nan_score_is_rejected(self):
        a = {"1": row(1, "a", float("nan"))}
        with self.assertRaises(ValueError):
            paired(a, a, score_field="correct")
