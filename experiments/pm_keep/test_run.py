import json
import tempfile
from dataclasses import replace
from pathlib import Path
import unittest

from experiments.pm_keep.adapter import AdapterConfig
from experiments.pm_keep.run import baseline_key, records


class ResultReuseTests(unittest.TestCase):
    def test_fixed_ea_reused_when_only_candidate_horizon_and_proxy_change(self):
        row = {"prompt_ids": [1, 2, 3], "prefix_length": 2, "max_new_tokens": 64,
               "score_contract": "literal_full_string_plus_terminal_eos_v1", "expected": "answer", "references": ["answer"]}
        cfg = AdapterConfig()
        changed = replace(cfg, horizon=128, query_policy="recent_prefix", seed=7)
        self.assertEqual(baseline_key({}, row, "E", cfg, "bfloat16"), baseline_key({}, row, "E", changed, "bfloat16"))
        self.assertNotEqual(baseline_key({}, row, "E", cfg, "bfloat16"), baseline_key({}, row, "E", replace(cfg, keep_fraction=.5), "bfloat16"))
        self.assertEqual(baseline_key({}, row, "F", cfg, "bfloat16"), baseline_key({}, row, "F", replace(cfg, keep_fraction=.5), "bfloat16"))

    def test_partial_final_write_is_preserved_for_resuming(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_bytes(b'{"row_id":1}\n{"row_')
            self.assertEqual(records(path, recover_tail=True), [{"row_id": 1}])
            self.assertEqual(path.read_bytes(), b'{"row_id":1}\n')
            self.assertEqual(path.with_suffix(".interrupted_tail").read_bytes(), b'{"row_')


if __name__ == "__main__":
    unittest.main()
