import json
from pathlib import Path
import tempfile
import unittest

from experiments.position_overnight.reuse import append_missing, candidate_rows


class CandidateReuseTests(unittest.TestCase):
    def test_pc2_reuses_only_selected_candidate_and_preserves_raw_tokens(self):
        contract = dict(model="model", backend="native", split="dev", dtype="bf16", topk=64,
                        select_blocks=16, chunk_size=128, attention_query_chunk_size=64,
                        data_sha256="data", generation="greedy", eos_ids=[9],
                        source_hashes={"runtime.py": "runtime", "selector_controls.py": "selector"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "contract.json").write_text(json.dumps(contract))
            rows = [dict(row_id=rid, selector=arm, generated_token_ids=[1, 9], prefill_seconds=17)
                    for rid, arm in [("dev0", "pc2"), ("dev0", "native"), ("test0", "pc2")]]
            (root / "generations.jsonl").write_text("\n".join(map(json.dumps, rows)))
            copied = candidate_rows(root, contract, {"dev0"}, ["pc2", "native"], kind="pc2")
            self.assertEqual(len(copied), 1)
            self.assertEqual(copied[0]["generated_token_ids"], [1, 9])
            self.assertEqual(copied[0]["prefill_seconds"], 17)
            out = root / "new.jsonl"
            self.assertEqual(append_missing(out, copied, "selector"), 1)
            self.assertEqual(append_missing(out, copied, "selector"), 0)
            for changed in [{**contract, "topk": 48}, {**contract, "data_sha256": "new-data"},
                            {**contract, "source_hashes": {"runtime.py": "changed", "selector_controls.py": "selector"}}]:
                with self.assertRaises(ValueError):
                    candidate_rows(root, changed, {"dev0"}, ["pc2"], kind="pc2")

    def test_pm_rejects_same_row_id_with_changed_input_or_candidate(self):
        contract = dict(model={}, backend="native", dtype="bf16", config={"horizon": 512},
                        decode="greedy", baseline_version="v1", task_protocol="prefix-only",
                        sources={"adapter.py": "adapter", "ops.py": "ops"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(contract))
            row = dict(row_id="dev0", arm="P", generated_token_ids=[1, 9], baseline_cache_key="old-input")
            (root / "per_example.jsonl").write_text(json.dumps(row))
            for target, keys in [(contract, {("dev0", "P"): "new-input"}),
                                 ({**contract, "config": {"horizon": 128}}, {("dev0", "P"): "old-input"})]:
                with self.assertRaises(ValueError):
                    candidate_rows(root, target, {"dev0"}, ["P"], kind="pm", row_keys=keys)
            result = candidate_rows(root, contract, {"dev0"}, ["P"], kind="pm",
                                    row_keys={("dev0", "P"): "old-input"})
            self.assertEqual(result[0]["generated_token_ids"], [1, 9])

    def test_conflicting_existing_generation_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "rows.jsonl"
            original = dict(row_id="dev0", arm="P", generated_token_ids=[1, 9])
            append_missing(out, [original], "arm")
            with self.assertRaises(ValueError):
                append_missing(out, [{**original, "generated_token_ids": [2, 9]}], "arm")
            self.assertEqual(json.loads(out.read_text()), original)


if __name__ == "__main__":
    unittest.main()
