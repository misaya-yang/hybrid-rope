import copy
import unittest

from experiments.native_enhancement_oral_20260915.prepare import (
    audit_group, follow, make_world, prepare_group,
)


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        result = {"input_ids": [ord(x) for x in text]}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


class PreparationTests(unittest.TestCase):
    def test_counterfactuals_are_valid_and_fit(self):
        for task in ("native_binding", "native_chain"):
            rows = prepare_group(CharacterTokenizer(), task, 1024, 0)
            audit_group(rows)
            self.assertEqual(len({r["prompt_sha256"] for r in rows}), 4)
            self.assertTrue(all(0.9 * 1024 <= r["input_tokens"] <= 992 for r in rows))
            self.assertTrue(all(r["evidence_positions"] for r in rows))

    def test_answer_and_context_corruption_rejected(self):
        rows = prepare_group(CharacterTokenizer(), "native_chain", 1024, 1)
        corrupt = copy.deepcopy(rows)
        corrupt[0]["references"] = ["WRONG"]
        with self.assertRaises(ValueError):
            audit_group(corrupt)
        corrupt = copy.deepcopy(rows)
        corrupt[1]["context_id"] = "changed"
        with self.assertRaises(ValueError):
            audit_group(corrupt)

    def test_rewiring_swaps_answers_with_identical_nodes(self):
        world = make_world("native_chain", 8)
        for i, query in enumerate(world["queries"]):
            self.assertNotEqual(follow(world["base"], query, 3), follow(world["rewired"], query, 3))
            self.assertEqual(follow(world["base"], query, 3), follow(world["rewired"], world["queries"][1-i], 3))


if __name__ == "__main__":
    unittest.main()
