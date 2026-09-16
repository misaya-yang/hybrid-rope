"""CPU tests for identity-preserving RULER scheduling and work accounting."""
import copy
import unittest

from experiments.native_enhancement_oral_20260915.throughput import (
    batch_plan, generation_metadata, kv_bytes_per_token, panel_metadata, plan_costs, profile,
)


class ThroughputTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(row_id=str(i), task="vt", input_tokens=n,
                         max_new_tokens=16, length_cap=128)
                     for i, n in enumerate((99, 100, 100, 101))]
        self.panel = panel_metadata(iter(self.rows))
        self.outputs = [dict(row_id=str(i), task="vt", length_cap=128, input_tokens=n,
                             generated_ids=[1] * count, ended_eos=count < 16, hit_cap=count == 16)
                        for i, (n, count) in enumerate(zip((99, 100, 100, 101), (4, 6, 16, 8)))]
        self.generated, _ = generation_metadata(iter(self.outputs), self.panel)

    def test_unpadded_only_groups_exact_lengths_and_preserves_all_rows(self):
        batches = batch_plan(self.panel, batch_size=2, max_kv_tokens=256)
        self.assertEqual(list(map(len, batches)), [1, 2, 1])
        self.assertEqual(plan_costs(batches, self.panel, self.generated)["prefill_square_cost_ratio"], 1)

    def test_left_padding_is_bounded_and_costs_include_early_eos_waste(self):
        batches = batch_plan(self.panel, batch_size=2, max_kv_tokens=256, left_pad=True)
        self.assertEqual(list(map(len, batches)), [2, 2])
        costs = plan_costs(batches, self.panel, self.generated)
        self.assertGreater(costs["prefill_square_cost_ratio"], 1)
        self.assertEqual(costs["observed_synchronous_decode_token_ratio"], 44 / 34)
        self.assertEqual(costs["max_reserved_kv_tokens"], 234)

    def test_budgets_and_physical_caps_never_mix(self):
        panel = copy.deepcopy(self.panel)
        panel["1"]["max_new_tokens"] = 15
        panel["2"]["length_cap"] = 256
        batches = batch_plan(panel, batch_size=4, max_kv_tokens=1024, left_pad=True)
        for batch in batches:
            self.assertEqual(len({panel[k]["max_new_tokens"] for k in batch}), 1)
            self.assertEqual(len({panel[k]["length_cap"] for k in batch}), 1)

    def test_task_buckets_use_input_task_only(self):
        panel = copy.deepcopy(self.panel)
        panel["1"]["task"] = "qa_1"
        batches = batch_plan(panel, batch_size=4, max_kv_tokens=1024,
                             left_pad=True, group_by_task=True)
        self.assertTrue(all(len({panel[k]["task"] for k in batch}) == 1 for batch in batches))

    def test_token_ceiling_can_force_batch_one_and_reject_impossible_row(self):
        batches = batch_plan(self.panel, batch_size=2, max_kv_tokens=128, left_pad=True)
        self.assertTrue(all(len(b) == 1 for b in batches))
        with self.assertRaises(ValueError):
            batch_plan(self.panel, batch_size=2, max_kv_tokens=100)

    def test_duplicate_missing_and_misidentified_outputs_fail(self):
        for rows in (self.outputs[:-1], self.outputs + [self.outputs[0]]):
            with self.assertRaises(ValueError):
                generation_metadata(rows, self.panel)
        corrupt = copy.deepcopy(self.outputs)
        corrupt[0]["task"] = "qa_1"
        with self.assertRaises(ValueError):
            generation_metadata(corrupt, self.panel)

    def test_bad_prompt_budget_and_completion_receipts_fail(self):
        for update in ({"input_tokens": 129}, {"max_new_tokens": 0}, {"prompt_ids": [1]}):
            row = dict(self.rows[0], **update)
            with self.assertRaises(ValueError):
                panel_metadata([row])
        corrupt = copy.deepcopy(self.outputs)
        corrupt[0]["hit_cap"] = True
        with self.assertRaises(ValueError):
            generation_metadata(corrupt, self.panel)

    def test_costs_reject_omitted_rows_and_report_real_output_counts(self):
        with self.assertRaises(ValueError):
            plan_costs([["0"]], self.panel, self.generated)
        overall = profile(self.panel, self.generated)["overall"]
        self.assertEqual(overall["total_generated_tokens"], 34)
        self.assertEqual(overall["total_reserved_output_tokens"], 64)
        self.assertEqual(overall["hit_cap_rows"], 1)

    def test_llama_gqa_kv_uses_kv_heads_not_query_heads(self):
        self.assertEqual(kv_bytes_per_token(dict(hidden_size=4096, num_attention_heads=32,
                                                num_key_value_heads=8, num_hidden_layers=32)), 131072)


if __name__ == "__main__":
    unittest.main()
