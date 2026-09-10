"""CPU local-causality harness tests, not evidence of a pretrained repair."""
import contextlib
import fcntl
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .causal_probe import (exact_group_block_mass, fixed_state_diagnostics, main,
                           make_swap_plan, run_probe)
from .runtime import AttentionSettings, NosaReferenceForCausalLM, SelectionContext


def small_settings():
    return AttentionSettings(kernel_size=2, kernel_stride=1, block_size=2,
                             init_blocks=1, local_blocks=0, select_blocks=1,
                             topk=3, attention_query_chunk_size=2)


def fixed_context():
    levels = torch.tensor([.2, 5., 1., -2., -5., 0.])
    keys = torch.zeros(1, 12, 2)
    keys[0, :, 0] = levels.repeat_interleave(2)
    values = torch.zeros_like(keys)
    values[:, 2:4] = 1
    query = torch.tensor([math.sqrt(2), 0.]).view(1, 1, 2).repeat(2, 1, 1)
    return SelectionContext(query, keys, values, torch.zeros(1, 12),
                            torch.tensor([11]), 0, small_settings())


class SwapTests(unittest.TestCase):
    def test_exact_mass_matches_per_head_softmax_including_cis(self):
        context = fixed_context()
        context.cis[0, 2:4] = -.7
        actual = exact_group_block_mass(context)
        expected = torch.tensor([.2, 4.3, 1., -2., -5., 0.]).softmax(0)[None]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(-1), torch.ones(1))

    def test_equal_budget_protection_and_matched_low_mass_sham(self):
        context = fixed_context()
        control = torch.tensor([[[0, 2, 5]]])
        copies = [t.clone() for t in (context.q, context.k, context.v, context.cis, control)]
        plan = make_swap_plan(context, control, max_swaps=1)
        torch.testing.assert_close(plan.selected["repair"], torch.tensor([[[0, 1, 5]]]))
        torch.testing.assert_close(plan.selected["sham"], torch.tensor([[[0, 4, 5]]]))
        report = plan.report()
        self.assertEqual(report["status"], "intervention")
        head = report["per_kv_head"][0]
        self.assertEqual(head["swap_count"], 1)
        self.assertEqual(head["mandatory_blocks"], [0, 5])
        self.assertGreater(head["repair_added_mass"], head["removed_mass"])
        self.assertLess(head["sham_added_mass"], head["removed_mass"])
        for selection in plan.selected.values():
            self.assertEqual(selection.shape, control.shape)
            self.assertEqual(len(set(selection.flatten().tolist())), 3)
            self.assertTrue({0, 5}.issubset(selection.flatten().tolist()))
        for original, copied in zip((context.q, context.k, context.v, context.cis, control), copies):
            torch.testing.assert_close(original, copied, atol=0, rtol=0)

    def test_fixed_state_output_error_and_mass_are_recorded(self):
        context = fixed_context()
        plan = make_swap_plan(context, torch.tensor([[[0, 2, 5]]]))
        metrics = fixed_state_diagnostics(context, plan)
        self.assertGreater(metrics["repair"]["exact_retained_mass_mean"],
                           metrics["control"]["exact_retained_mass_mean"])
        self.assertLess(metrics["repair"]["attention_output_l2_error"],
                        metrics["control"]["attention_output_l2_error"])
        self.assertLess(metrics["sham"]["exact_retained_mass_mean"],
                        metrics["control"]["exact_retained_mass_mean"])

    def test_no_swap_is_explicit_and_does_not_change_support(self):
        context = fixed_context()
        # Preserve even an unsorted control exactly when no intervention occurs.
        control = torch.tensor([[[5, 0, 2]]])
        plan = make_swap_plan(context, control, max_swaps=0)
        self.assertEqual(plan.report()["status"], "no_intervention")
        for selected in plan.selected.values():
            torch.testing.assert_close(selected, control, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "mandatory"):
            make_swap_plan(context, torch.tensor([[[1, 2, 5]]]))


class TokenizerStub:
    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


def tiny_model():
    torch.manual_seed(213)
    config = dict(vocab_size=31, hidden_size=16, intermediate_size=24,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                  num_hidden_layers=2, rms_norm_eps=1e-6, rope_theta=10000.,
                  eos_token_id=2, tie_word_embeddings=False)
    return NosaReferenceForCausalLM(config, settings=small_settings()).eval()


def tiny_row():
    return dict(row_id="dev_probe", task="tiny", split="dev", prompt_ids=list(range(4, 27)),
                max_new_tokens=3, length_cap=32, references=["7"],
                score_contract="ruler_official_string_match_all_v1")


class CompleteModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shared_prefill_once_unmodified_and_three_full_generation_paths(self):
        for candidate in ("native", "pc2"):
            model, row = tiny_model(), tiny_row()
            original_selector = model.selector
            original_prefill = model.prefill
            captured = []
            def observe_prefill(*args, **kwargs):
                output = original_prefill(*args, **kwargs)
                captured.append((output.past_key_values,
                    [(entry.k.clone(), entry.v.clone(), entry.cis.clone())
                     for entry in output.past_key_values]))
                return output
            model.prefill = observe_prefill
            result = run_probe(model, row, TokenizerStub(), set(), layer_idx=1,
                               candidate=candidate, max_swaps=0, chunk_size=5)
            self.assertEqual(len(captured), 1)
            self.assertEqual(result["shared_prefill_tokens"], len(row["prompt_ids"]) - 1)
            self.assertEqual(result["swap_plan"]["status"], "no_intervention")
            cache, snapshot = captured[0]
            for entry, old in zip(cache, snapshot):
                for value, original in zip((entry.k, entry.v, entry.cis), old):
                    torch.testing.assert_close(value, original, atol=0, rtol=0)
            generations = result["generations"]
            self.assertEqual(generations["repair"]["generated_token_ids"], generations["control"]["generated_token_ids"])
            self.assertEqual(generations["sham"]["generated_token_ids"], generations["control"]["generated_token_ids"])
            for generation in generations.values():
                self.assertEqual(generation["generated_tokens"], 3)
                self.assertEqual(generation["single_point_hits"], 1)
                self.assertFalse(generation["ended_with_eos"])
                self.assertIn("official_recall", generation)
            self.assertIs(model.selector, original_selector)

    def test_real_eos_token_terminates_complete_generation(self):
        model, row = tiny_model(), tiny_row()
        row.update(score_contract="literal_full_string_plus_terminal_eos_v1", expected="")
        # Every possible first raw token is an EOS for this stop-path unit test.
        result = run_probe(model, row, TokenizerStub(), set(range(31)), layer_idx=0,
                           candidate="pc2", max_swaps=1, chunk_size=5)
        for generation in result["generations"].values():
            self.assertEqual(generation["generated_tokens"], 1)
            self.assertTrue(generation["exact_plus_eos"])

    def test_nonzero_intervention_reaches_free_generation(self):
        result = run_probe(tiny_model(), tiny_row(), TokenizerStub(), set(), layer_idx=0,
                           candidate="pc2", max_swaps=1, chunk_size=5)
        self.assertEqual(result["swap_plan"]["status"], "intervention")
        self.assertGreater(result["swap_plan"]["total_head_block_swaps"], 0)
        self.assertTrue(result["pre_intervention_q_current_kv_cis_and_candidate_support_equal"])
        for generation in result["generations"].values():
            self.assertEqual(generation["generated_tokens"], 3)
            self.assertEqual(generation["single_point_hits"], 1)
        # An actual support change propagates through the complete tiny model;
        # this assertion does not claim either continuation is a correct answer.
        self.assertNotEqual(result["generations"]["control"]["generated_token_ids"],
                            result["generations"]["repair"]["generated_token_ids"])

    def test_test_split_is_rejected(self):
        row = tiny_row()
        row["split"] = "test"
        with self.assertRaisesRegex(ValueError, "DEV"):
            run_probe(tiny_model(), row, TokenizerStub(), set(), layer_idx=0)


class CliTests(unittest.TestCase):
    def test_default_dry_run_does_not_load_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "rows.jsonl"
            data.write_text(json.dumps(tiny_row()) + "\n")
            argv = ["--root", directory, "--data", str(data), "--row-id", "dev_probe",
                    "--layer", "0", "--output", str(root / "output")]
            with patch.object(NosaReferenceForCausalLM, "from_pretrained") as load:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = main(argv)
                load.assert_not_called()
            self.assertEqual(result["status"], "DRY_RUN")
            self.assertFalse((root / "output").exists())
            self.assertFalse((root / "queue.lock").exists())

    def test_busy_shared_lock_prevents_any_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "rows.jsonl"
            data.write_text(json.dumps(tiny_row()) + "\n")
            argv = ["--root", directory, "--data", str(data), "--row-id", "dev_probe",
                    "--layer", "0", "--output", str(root / "output"), "--execute", "--device", "cpu"]
            with (root / "queue.lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(NosaReferenceForCausalLM, "from_pretrained") as load:
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(RuntimeError, "queue is busy"):
                            main(argv)
                    load.assert_not_called()
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
