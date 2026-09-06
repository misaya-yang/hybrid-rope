"""Single fixed-probe repair contracts; fake tokenizer, no model/GPU evidence."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from scripts.data import prepare_native_single_code_calibration as builder


class CharacterTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    chat_template = "unit-test template only"

    def encode(self, text, **kwargs):
        return [ord(char) + 10 for char in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i - 10) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert len(messages) == 1 and messages[0]["role"] == "user"
        assert not tokenize and add_generation_prompt
        return "<bos><user>" + messages[0]["content"] + "<assistant>"


class TestNativeSingleCodeCalibration(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_fresh_deterministic_balanced_single_fact_blueprints(self):
        self.assertEqual(builder.SEEDS, {"calibration": 202609021, "confirmation": 202609022})
        all_hashes = []
        for split, count in (("calibration", 64), ("confirmation", 128)):
            rows = builder.make_blueprints(split)
            self.assertEqual(rows, builder.make_blueprints(split))
            self.assertEqual(len(rows), count)
            self.assertEqual(Counter(b["query_index"] for b in rows), {i: count // 8 for i in range(8)})
            self.assertEqual(Counter(b["depth_stratum"] for b in rows), {i: count // 4 for i in range(4)})
            for blueprint in rows:
                self.assertEqual(len({f["key"] for f in blueprint["facts"]}), 8)
                self.assertEqual(len({f["code"] for f in blueprint["facts"]}), 8)
                self.assertRegex(builder.expected_text(blueprint), r"^\d{6}\.$")
                all_hashes.append(blueprint["blueprint_sha256"])
        self.assertEqual(len(set(all_hashes)), 192)
        old_hashes = {b["blueprint_sha256"] for b in builder.common.make_blueprints("calibration")}
        self.assertFalse(old_hashes & set(all_hashes))

    def test_single_prompt_fixed_facts_pairing_and_filler_only_fitting(self):
        blueprint = builder.make_blueprints("calibration")[7]
        rows = list(builder.capability_rows(blueprint, "calibration", self.tokenizer))
        self.assertEqual([r["length"] for r in rows], [0, 1024, 2048, 4096, 8192])
        self.assertEqual(rows[0]["variant"], "compact")
        for row in rows:
            text = self.tokenizer.decode(row["input_ids"])
            self.assertTrue(text.endswith("<assistant>The code is "))
            self.assertNotIn("The two codes are", text)
            self.assertEqual(text.count("Record "), 8)
            self.assertEqual(row["facts"], blueprint["facts"])
            self.assertEqual(row["query_index"], 7)
            self.assertEqual(row["depth_stratum"], 3)
            self.assertEqual(row["expected_text"], blueprint["facts"][7]["code"] + ".")
            self.assertEqual(row["generation_budget"], 48)
            self.assertEqual(row["terminal_ids"], [2])
            self.assertEqual(row["prompt_ids_sha256"], builder.common.ids_hash(row["input_ids"]))
            if row["length"]:
                self.assertEqual(len(row["input_ids"]), row["length"] - 48)
                self.assertEqual(row["prompt_underage"], 0)
            for fact in blueprint["facts"]:
                self.assertEqual(text.count(f"has code {fact['code']}."), 1)

    def test_exact_full_output_fixtures_allow_only_outer_whitespace(self):
        fixtures = builder.scorer_fixtures(self.tokenizer)
        self.assertEqual(fixtures["expected_text"], "123456.")
        for case in fixtures["cases"]:
            with self.subTest(case=case["name"]):
                ids, eos = case["generated_ids"], fixtures["terminal_ids"][0]
                passed = (bool(ids) and ids[-1] == eos and eos not in ids[:-1] and
                          self.tokenizer.decode(ids[:-1]).strip() == fixtures["expected_text"])
                self.assertEqual(passed, case["expected_pass"])

    def write_parent(self, root):
        manifest = {"status": builder.READY, "grid": list(builder.GRID),
                    "bos_token_id": 1, "files": {}}
        originals = {}
        for split, count in builder.NATURAL_COUNTS.items():
            rows = []
            for index in range(count):
                document = {"document_ids": list(range(100, 8292)),
                            "source_row": index + (0 if split == "calibration" else 100),
                            "source_text_sha256": builder.common.canonical_hash([split, index])}
                rows.extend(builder.common.natural_rows(document, split, index, 1))
            originals[split] = rows
            path = root / f"{split}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest["files"][split] = {"path": path.name, "sha256": builder.common.sha256_file(path),
                                        "rows": len(rows)}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return manifest, originals

    def test_parent_input_hashes_and_exact_natural_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, originals = self.write_parent(root)
            parent, loaded, receipt = builder.load_parent_inputs(root)
            self.assertEqual(loaded, originals)
            self.assertEqual(parent, manifest)
            self.assertEqual(receipt["manifest_sha256"], builder.common.sha256_file(root / "manifest.json"))
            for split in builder.SEEDS:
                self.assertEqual(receipt["files"][split]["natural_rows_canonical_sha256"],
                                 builder.common.canonical_hash(originals[split]))
            path = root / "calibration.jsonl"
            with path.open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "hash drift"):
                builder.load_parent_inputs(root)

    def test_repair_cannot_chain_or_accept_output_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, originals = self.write_parent(root)
            manifest["measurement_protocol"] = builder.PROTOCOL
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "only one repair"):
                builder.load_parent_inputs(root)
            del manifest["measurement_protocol"]
            originals["calibration"][0]["nll"] = 1.0
            path = root / "calibration.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in originals["calibration"]))
            manifest["files"]["calibration"]["sha256"] = builder.common.sha256_file(path)
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "pure V1 input schema"):
                builder.load_parent_inputs(root)

    def test_original_module_is_not_monkeypatched(self):
        original = (builder.common.ASSISTANT_PREFIX, builder.common.SEEDS.copy(),
                    builder.common.render_user, builder.common.chat_ids, builder.common.fit_prompt)
        blueprint = builder.make_blueprints("calibration")[0]
        list(builder.capability_rows(blueprint, "calibration", self.tokenizer))
        self.assertEqual(original, (builder.common.ASSISTANT_PREFIX, builder.common.SEEDS,
                                   builder.common.render_user, builder.common.chat_ids,
                                   builder.common.fit_prompt))


if __name__ == "__main__":
    unittest.main()
