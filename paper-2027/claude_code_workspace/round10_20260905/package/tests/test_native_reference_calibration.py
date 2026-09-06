"""Standard-library, fake-tokenizer data-contract checks; not model evidence."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.data import prepare_native_reference_calibration as builder


class CharacterTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    chat_template = "fake unit-test template, not a real checkpoint"

    def encode(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(char) + 10 for char in text]
        return ids[:max_length] if truncation else ids

    def decode(self, ids, **kwargs):
        return "".join(chr(i - 10) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert len(messages) == 1 and messages[0]["role"] == "user"
        assert not tokenize and add_generation_prompt
        return "<bos><user>" + messages[0]["content"] + "<assistant>"


class TestNativeReferenceCalibration(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_fixed_protocol_and_counterbalanced_disjoint_blueprints(self):
        self.assertEqual(builder.GRID, (1024, 2048, 4096, 8192))
        self.assertEqual(builder.NATURAL_COUNTS, {"calibration": 32, "confirmation": 64})
        cal = builder.make_blueprints("calibration")
        confirm = builder.make_blueprints("confirmation")
        self.assertEqual(cal, builder.make_blueprints("calibration"))
        self.assertEqual((len(cal), len(confirm)), (64, 128))
        self.assertFalse({b["blueprint_sha256"] for b in cal} &
                         {b["blueprint_sha256"] for b in confirm})
        for blueprints, per_cell in ((cal, 8), (confirm, 16)):
            cells = Counter((b["depth_stratum"], b["query_reversed"]) for b in blueprints)
            self.assertEqual(set(cells.values()), {per_cell})
            self.assertEqual(len(cells), 8)
            for blueprint in blueprints:
                self.assertEqual(len({f["key"] for f in blueprint["facts"]}), 8)
                self.assertEqual(len({f["code"] for f in blueprint["facts"]}), 8)
                self.assertRegex(builder.expected_text(blueprint), r"^\d{6}, \d{6}\.$")

    def test_document_selection_order_exclusions_dedup_and_disjointness(self):
        excluded_text = "excluded" * 1100
        excluded = {hashlib.sha256(excluded_text.encode()).hexdigest()}
        first = "document-first-" * 700
        rows = [(19999, "too early" * 1000), (20000, excluded_text),
                (20001, "short"), (20002, first), (20003, first)]
        rows.extend((20004 + i, f"document-{i:03d}-" * 700) for i in range(95))
        splits = builder.select_documents(rows, self.tokenizer, excluded, 20000)
        self.assertEqual(len(splits["calibration"]), 32)
        self.assertEqual(len(splits["confirmation"]), 64)
        self.assertEqual(splits["calibration"][0]["source_row"], 20002)
        hashes = [d["source_text_sha256"] for docs in splits.values() for d in docs]
        self.assertEqual(len(set(hashes)), 96)
        self.assertFalse(set(hashes) & excluded)
        with self.assertRaisesRegex(ValueError, "found 0"):
            builder.select_documents([], self.tokenizer, set(), 20000)

    def test_nested_suffix_has_identical_targets_and_no_appended_eos(self):
        doc = {"document_ids": list(range(100, 8292)), "source_row": 20000,
               "source_text_sha256": "a" * 64}
        rows = list(builder.natural_rows(doc, "calibration", 0, 1))
        expected = doc["document_ids"][:8191][-256:]
        for row in rows:
            self.assertEqual(len(row["input_ids"]), row["length"])
            self.assertEqual(row["input_ids"][0], 1)
            self.assertEqual(row["input_ids"][row["target_start"]:], expected)
            self.assertEqual(row["target_tokens"], 256)
            self.assertEqual(row["prompt_ids_sha256"], builder.ids_hash(row["input_ids"]))
        self.assertEqual(len({r["target_ids_sha256"] for r in rows}), 1)

    def test_capability_pairing_exact_format_and_filler_only_length_fit(self):
        blueprint = builder.make_blueprints("calibration")[0]
        rows = list(builder.capability_rows(blueprint, "calibration", self.tokenizer))
        self.assertEqual([r["length"] for r in rows], [0, 1024, 2048, 4096, 8192])
        self.assertEqual(rows[0]["variant"], "compact")
        for row in rows:
            text = self.tokenizer.decode(row["input_ids"])
            self.assertTrue(text.endswith("<assistant>" + builder.ASSISTANT_PREFIX))
            self.assertEqual(text.count("Record "), 8)
            for fact in blueprint["facts"]:
                self.assertEqual(text.count(f"has code {fact['code']}."), 1)
            self.assertEqual(row["facts"], blueprint["facts"])
            self.assertEqual(row["query_indices"], blueprint["query_indices"])
            self.assertEqual(row["expected_text"], builder.expected_text(blueprint))
            self.assertEqual(row["terminal_ids"], [2])
            if row["length"]:
                self.assertEqual(row["prompt_token_budget"], row["length"] - 48)
                self.assertGreaterEqual(row["prompt_underage"], 0)
                self.assertLessEqual(row["prompt_underage"], 16)
                self.assertEqual(row["prompt_underage"], 0)
                self.assertEqual(len(row["input_ids"]) + row["prompt_underage"], row["length"] - 48)

    def test_full_output_fixtures_reject_proxy_success(self):
        fixtures = builder.scorer_fixtures(self.tokenizer)
        for case in fixtures["cases"]:
            with self.subTest(case=case["name"]):
                ids = case["generated_ids"]
                eos = fixtures["terminal_ids"][0]
                exact = (bool(ids) and ids[-1] == eos and eos not in ids[:-1] and
                         self.tokenizer.decode(ids[:-1]).strip() == fixtures["expected_text"])
                self.assertEqual(exact, case["expected_pass"])

    def test_exclusions_consume_identity_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.jsonl"
            path.write_text(json.dumps({"source_text_sha256": "a" * 64, "score": -999}) +
                            "\n" + json.dumps({"family": "capability", "score": 1}) + "\n")
            excluded, receipts = builder.load_exclusions([path])
            self.assertEqual(excluded, {"a" * 64})
            self.assertNotIn("score", json.dumps(receipts))

    def test_prepare_complete_hashed_files_with_small_mock_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text('{"model_type":"gemma"}')
            (checkpoint / "tokenizer.json").write_text('{}')
            source = root / "source.parquet"
            source.write_bytes(b"fake source; parquet reader mocked")
            args = argparse.Namespace(source=source, checkpoint=checkpoint, output=root / "data",
                                      start_row=20000, exclude_rows_jsonl=[])
            texts = [(20000, "first long document " * 600),
                     (20001, "second long document " * 600)]
            with patch.object(builder, "NATURAL_COUNTS", {"calibration": 1, "confirmation": 1}), \
                 patch.object(builder, "CAPABILITY_COUNTS", {"calibration": 1, "confirmation": 1}), \
                 patch.object(builder, "load_tokenizer", return_value=self.tokenizer), \
                 patch.object(builder, "iter_source_texts", return_value=iter(texts)):
                manifest = builder.prepare(args)
            self.assertEqual(manifest["model_evaluation_status"], "NOT_RUN")
            self.assertFalse(manifest["L_ref_selected"])
            self.assertTrue((args.output / "manifest.json").is_file())
            for split in ("calibration", "confirmation"):
                receipt = manifest["files"][split]
                path = args.output / receipt["path"]
                self.assertEqual(receipt["sha256"], builder.sha256_file(path))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(len(rows), 9)
                self.assertEqual({r["split"] for r in rows}, {split})
            with self.assertRaises(FileExistsError):
                builder.prepare(args)


if __name__ == "__main__":
    unittest.main()
