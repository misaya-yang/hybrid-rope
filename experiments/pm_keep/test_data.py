"""Check leakage, BPE boundary preservation, source disjointness and answer labels."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest

from experiments.pm_keep import prepare


class PMDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.root = Path(__file__).resolve().parents[2]
        path = cls.root / "results/pm_keep_20260909/source/tokenizer"
        if not path.exists():
            raise unittest.SkipTest("PM tokenizer artifact is not available")
        cls.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    def test_complete_chat_bpe_boundary(self):
        for context in ("The key is amber.", "文档：第一次记录甲，第二次记录乙。", "prefix ends café"):
            question = "Which record was first?"
            row = prepare.render_split(self.tokenizer, context, question, "Only give the answer.")
            self.assertEqual(row["prefix_ids"] + row["suffix_ids"], row["prompt_ids"])
            self.assertEqual(self.tokenizer.encode(row["full_prompt"], add_special_tokens=False), row["prompt_ids"])
            self.assertNotIn(question, row["prefix_text"])
            self.assertIn(context, row["prefix_text"])
            self.assertLessEqual(row["last_prefix_offset_end"], row["boundary_char"])

    def test_preexisting_question_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "leaked"):
            prepare.render_split(self.tokenizer, "Quoted question: What is X?", "What is X?", "Answer.")

    def test_different_future_questions_leave_prefix_identical(self):
        first = prepare.render_split(self.tokenizer, "Record key=alpha; value=bravo.",
                                     "What is the first value?", "Answer only.")
        second = prepare.render_split(self.tokenizer, "Record key=alpha; value=bravo.",
                                      "How many records mention delta?", "Answer only.")
        self.assertEqual(first["prefix_ids"], second["prefix_ids"])
        self.assertEqual(first["prefix_length"], second["prefix_length"])

    def test_synthetic_stratification_and_expected_values(self):
        depths = [prepare.synthetic_material("single_kv", "dev", i)["positions"][0] for i in range(4)]
        self.assertEqual(depths, [0.1, 0.35, 0.6, 0.85])
        row = prepare.make_synthetic(self.tokenizer, "multi_kv_order", "dev", 1, 2048)
        self.assertTrue(1948 <= row["prefix_tokens"] <= 2048)
        prepare.validate([row], self.tokenizer)
        bad = copy.deepcopy(row)
        bad["expected"] = "wrong"
        with self.assertRaisesRegex(ValueError, "mapping"):
            prepare.validate([bad], self.tokenizer)

    def test_document_titles_are_case_normalized(self):
        self.assertEqual(prepare.constituent_ids("Passage 1:\nA Famous Place\nText"),
                         prepare.constituent_ids("Passage 9:\na famous PLACE\nDifferent question context"))

    def test_shared_development_document_is_excluded_from_test(self):
        def candidate(task, n, docs):
            return {"task": task, "context_sha256": f"{task}{n}", "constituent_doc_ids": docs}
        candidates = {
            "hotpotqa": [candidate("hotpotqa", 0, ["a"]), candidate("hotpotqa", 1, ["b"]), candidate("hotpotqa", 2, ["c"])],
            "2wikimqa": [candidate("2wikimqa", 0, ["d"]), candidate("2wikimqa", 1, ["a"]), candidate("2wikimqa", 2, ["e"])],
        }
        rows = prepare.select_natural(candidates, 1, 1)
        test = [r for r in rows if r["split"] == "test"]
        self.assertEqual(len(test), 2)
        self.assertFalse({"a", "d"} & {d for r in test for d in r["constituent_doc_ids"]})

    def test_smoke_or_frozen_rows_validate(self):
        data = self.root / "results/pm_keep_20260909/data/rows.jsonl"
        if not data.exists():
            data = self.root / "results/pm_keep_20260909/smoke_data/rows.jsonl"
        if not data.exists():
            self.skipTest("no PM data artifact yet")
        rows = prepare.read_jsonl(data)
        result = prepare.validate(rows, self.tokenizer)
        self.assertTrue(result["complete_context_preserved"])
        self.assertTrue(all(r["prefix_tokens"] >= 1040 for r in rows))
        if len(rows) == 192:
            for task in prepare.TASKS:
                self.assertEqual(sum(r["task"] == task and r["split"] == "dev" for r in rows), 16)
                self.assertEqual(sum(r["task"] == task and r["split"] == "test" for r in rows), 32)

    def test_frozen_natural_context_question_and_answers_match_raw(self):
        path = self.root / "results/pm_keep_20260909/data/rows.jsonl"
        if not path.exists():
            self.skipTest("complete natural data is not yet frozen")
        sources = {task: prepare.read_jsonl(self.root / f"results/pm_keep_20260909/source/longbench/{task}.jsonl")
                   for task in ("hotpotqa", "2wikimqa")}
        for row in prepare.read_jsonl(path):
            if row["task"] in sources:
                original = sources[row["task"]][row["source"]["original_row_index"]]
                self.assertEqual(row["raw_context"], original["context"])
                self.assertEqual(row["raw_question"], original["input"])
                self.assertEqual(row["references"], original["answers"])
                self.assertEqual(row["doc_id"], original["_id"])


if __name__ == "__main__":
    unittest.main()
