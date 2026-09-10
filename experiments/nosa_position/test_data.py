"""Decision-relevant checks for frozen data provenance and counterfactual labels."""
from __future__ import annotations

import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import re
import unittest


SPEC = importlib.util.spec_from_file_location("nosa_prepare", Path(__file__).with_name("prepare.py"))
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def row_for(material, split, content_swap, query_swap):
    # Test the logical record schema independently of tokenizer infrastructure.
    text, expected, records = prepare.counterfactual_user(
        material, (0, 3)[query_swap], content_swap, 20)
    return dict(row_id=f"{split}_c{content_swap}q{query_swap}", family_id=f"family_{split}",
                task="repeat_key_first_latest", split=split, length_cap=8192,
                seed=material["seed"], material_cluster_id=f"{split}:{material['seed']}",
                prompt_ids=[1, 2, 3], input_tokens=3, max_new_tokens=32,
                prompt=text, prompt_sha256=prepare.text_sha(text), expected=expected,
                references=[expected], score_contract=prepare.EXACT_CONTRACT,
                records=records, query_key=material["key"], query_ordinal=(1, 4)[query_swap],
                content_swap=content_swap, query_swap=query_swap, noise_sentence_count=20)


def fourway(split="dev"):
    material = prepare.family_material(split, 0)
    return [row_for(material, split, c, q) for c in (0, 1) for q in (0, 1)]


class DataContractTests(unittest.TestCase):
    def test_frozen_public_prompts_and_references_preserve_upstream_source(self):
        root = Path(__file__).resolve().parents[2]
        frozen = root / "results/nosa_position_20260909/data"
        if not (frozen / "manifest.json").exists():
            self.skipTest("local frozen data has not been prepared")
        manifest = json.loads((frozen / "manifest.json").read_text())
        self.assertEqual(hashlib.sha256((frozen / "rows.jsonl").read_bytes()).hexdigest(),
                         manifest["rows_sha256"])
        rows = prepare.load_jsonl(frozen / "rows.jsonl")
        sources = {}
        for call in manifest["generation_calls"]:
            source = frozen / call["source_jsonl"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), call["source_sha256"])
            sources[call["source_jsonl"]] = prepare.load_jsonl(source)
        for row in rows:
            if row["score_contract"] == prepare.PUBLIC_CONTRACT:
                original = sources[row["source"]["path"]][row["source"]["line"] - 1]
                self.assertEqual(row["prompt"], original["input"] + original["answer_prefix"])
                self.assertEqual(row["references"], original["outputs"])
                self.assertTrue(all(ref in row["prompt"] for ref in row["references"]))

    def test_content_and_query_swaps_change_answer_independently(self):
        material = prepare.family_material("test", 3)
        for pair in ((0, 3), (1, 2)):
            cells = {}
            for c in (0, 1):
                for q in (0, 1):
                    text, expected, _ = prepare.counterfactual_user(material, pair[q], c, 11)
                    parsed = re.findall(r"Record key=([a-z]+); value=([a-z]+)\.", text)
                    self.assertEqual(len(parsed), 12)
                    found = [v for k, v in parsed if k == material["key"]]
                    self.assertEqual(len(found), 4)
                    self.assertEqual(expected, found[pair[q]])
                    cells[c, q] = expected
            self.assertEqual(cells[0, 0], cells[1, 1])
            self.assertEqual(cells[0, 1], cells[1, 0])
            self.assertNotEqual(cells[0, 0], cells[0, 1])

    def test_split_content_is_disjoint(self):
        content = {}
        for split in ("dev", "test"):
            content[split] = {record[field]
                              for i in range(32)
                              for record in prepare.family_material(split, i)["records"]
                              for field in ("key", "value")}
        self.assertFalse(content["dev"] & content["test"])

    def test_complete_families_validate(self):
        result = prepare.validate_rows(fourway("dev") + fourway("test"))
        self.assertEqual(result["factorial_families_per_length"], 2)

    def test_answer_remapping_is_not_silent(self):
        rows = fourway()
        rows[0]["expected"] = rows[1]["expected"]
        rows[0]["references"] = [rows[0]["expected"]]
        with self.assertRaisesRegex(ValueError, "answer mapping"):
            prepare.validate_rows(rows)

    def test_history_truncation_is_rejected_even_with_rehashed_prompt(self):
        rows = fourway()
        row = rows[0]
        row["prompt"] = re.sub(r"Record key=[a-z]+; value=[a-z]+\.\n", "", row["prompt"], count=1)
        row["prompt_sha256"] = prepare.text_sha(row["prompt"])
        with self.assertRaisesRegex(ValueError, "history"):
            prepare.validate_rows(rows)

    def test_missing_fourth_cell_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            prepare.validate_rows(fourway()[:-1])

    def test_public_recall_cannot_be_presented_as_exact(self):
        row = copy.deepcopy(fourway()[0])
        row.update(task="niah_single_1", score_contract=prepare.PUBLIC_CONTRACT)
        with self.assertRaisesRegex(ValueError, "mislabeled as exact"):
            prepare.validate_rows([row])

    def test_prompt_budget_is_checked(self):
        rows = fourway()
        rows[0]["length_cap"] = 4
        with self.assertRaisesRegex(ValueError, "budget"):
            prepare.validate_rows(rows)


if __name__ == "__main__":
    unittest.main()
