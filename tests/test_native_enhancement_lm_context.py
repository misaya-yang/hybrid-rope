import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.native_enhancement_oral_20260915.lm_context import (
    analyze_four_conditions, build_context_pair, build_manifest, cpu_canary,
    target_nll_from_logits,
)


class ContextAlignmentTests(unittest.TestCase):
    def test_native_4096_and_recent_767_predict_exact_same_256_tokens(self):
        pair = build_context_pair(np.arange(4097), native_length=4096)
        self.assertEqual(len(pair["full"]["input_ids"]), 4096)
        self.assertEqual(len(pair["recent"]["input_ids"]), 767)
        np.testing.assert_array_equal(pair["full"]["target_ids"], np.arange(3841, 4097))
        np.testing.assert_array_equal(pair["recent"]["target_ids"], pair["full"]["target_ids"])
        np.testing.assert_array_equal(pair["full"]["loss_positions"], np.arange(3840, 4096))
        np.testing.assert_array_equal(pair["recent"]["loss_positions"], np.arange(511, 767))
        self.assertEqual(pair["recent"]["input_ids"][511], 3840)
        self.assertEqual(pair["recent"]["input_ids"][-1], 4095)

    def test_offsets_and_relative_distances(self):
        pair = build_context_pair(np.arange(50), native_length=16, recent_history=4,
                                  target_tokens=3, window_start=10)
        np.testing.assert_array_equal(pair["full"]["target_ids"], [24, 25, 26])
        self.assertEqual(pair["recent"]["source_input_start"], 20)
        self.assertEqual(pair["recent"]["source_target_start"], 24)
        n = len(pair["recent"]["input_ids"])
        a, b = pair["full"]["position_ids"][-n:], pair["recent"]["position_ids"]
        np.testing.assert_array_equal(a[:, None] - a, b[:, None] - b)
        np.testing.assert_array_equal(pair["recent"]["input_ids"], pair["full"]["input_ids"][-n:])
        self.assertEqual(b[0], 0)

    def test_oracle_catches_off_by_one_scoring(self):
        pair = build_context_pair(np.arange(20), native_length=16, recent_history=4, target_tokens=3)
        context = pair["recent"]
        logits = np.zeros((len(context["input_ids"]), 20))
        logits[np.arange(len(logits)), context["input_ids"] + 1] = 9.0
        correct = target_nll_from_logits(logits, context)["nll"]
        wrong = dict(context, loss_positions=context["loss_positions"] - 1)
        self.assertAlmostEqual(target_nll_from_logits(logits, wrong)["nll"] - correct, 9.0)
        self.assertLess(correct, 0.01)

    def test_missing_final_token_and_noninteger_tokens_rejected(self):
        for tokens in (np.arange(16), np.arange(17, dtype=float)):
            with self.assertRaises(ValueError):
                build_context_pair(tokens, native_length=16, recent_history=4, target_tokens=3)
        with self.assertRaises(ValueError):
            build_context_pair(np.arange(20), native_length=6, recent_history=4, target_tokens=3)

    def test_canary_is_explicitly_cpu_only(self):
        result = cpu_canary()
        self.assertEqual(result["status"], "CPU_CANARY_PASS")
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["gpu_used"])
        self.assertEqual(result["target_ids"], [14, 15, 16])


class ManifestAndAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "tokens.npy"
        np.save(self.path, np.tile(np.arange(20), (3, 1)), allow_pickle=False)

    def manifest(self, **kwargs):
        return build_manifest(self.path, native_length=16, recent_history=4, target_tokens=3,
                              split=kwargs.pop("split", "development"), **kwargs)

    @staticmethod
    def scores(manifest, deltas=(-1.0, -3.0, 6.0)):
        rows = []
        for sample, delta in zip(manifest["samples"], deltas):
            for arm, context, value in (("native", "full", 10.0), ("native", "recent", 14.0),
                                        ("ncp", "full", 10.0 + delta), ("ncp", "recent", 15.0)):
                rows.append({"pair_id": sample["pair_id"], "target_sha256": sample["target_sha256"],
                             "arm": arm, "context": context, "nll_sum": value * 3, "target_count": 3})
        return rows

    def test_no_source_ids_does_not_claim_independent_documents(self):
        manifest = self.manifest()
        self.assertIsNone(manifest["documents"])
        self.assertFalse(manifest["fresh_slice_is_independent_document"])
        report = analyze_four_conditions(manifest, self.scores(manifest), draws=10)
        self.assertIsNone(report["documents"])
        self.assertEqual(report["ci_method"], "unavailable")
        self.assertIsNone(report["metrics"]["delta_full"]["ci95"])

    def test_confirmation_requires_explicit_verified_exclusions(self):
        for kwargs in ({}, {"document_ids": ["a", "b", "c"]},
                       {"document_ids": ["a", "b", "c"], "excluded_document_ids": []}):
            with self.assertRaises(ValueError):
                self.manifest(split="confirmation", **kwargs)
        manifest = self.manifest(split="confirmation", document_ids=["a", "b", "c"],
                                 excluded_document_ids=["old"], exclusions_verified=True)
        self.assertEqual(manifest["documents"], 3)
        with self.assertRaises(ValueError):
            self.manifest(split="confirmation", document_ids=["a", "b", "c"],
                          excluded_document_ids=["a"], exclusions_verified=True)

    def test_split_and_document_id_errors(self):
        with self.assertRaises(ValueError):
            self.manifest(split="fresh")
        with self.assertRaises(ValueError):
            self.manifest(document_ids=["a"])
        with self.assertRaises(ValueError):
            self.manifest(document_ids=["a", "", "b"])

    def test_repeated_slices_are_averaged_within_document_before_bootstrap(self):
        manifest = self.manifest(document_ids=["a", "a", "b"])
        report = analyze_four_conditions(manifest, self.scores(manifest), draws=1000)
        self.assertEqual(report["pairs"], 3)
        self.assertEqual(report["documents"], 2)
        # a: mean(-1,-3)=-2; b: +6. Document-equal delta = 2, not row mean 2/3.
        self.assertEqual(report["metrics"]["delta_full"]["estimate"], 2.0)
        self.assertEqual(report["metrics"]["delta_recent"]["estimate"], 1.0)
        self.assertEqual(report["metrics"]["delta_use"]["estimate"], -1.0)
        self.assertEqual(report["metrics"]["use_native"]["estimate"], 4.0)
        self.assertEqual(report["metrics"]["use_ncp"]["estimate"], 3.0)

    def test_point_estimate_does_not_depend_on_bootstrap_draws_or_seed(self):
        manifest = self.manifest(document_ids=["a", "b", "c"])
        a = analyze_four_conditions(manifest, self.scores(manifest), draws=1, seed=1)
        b = analyze_four_conditions(manifest, self.scores(manifest), draws=100, seed=999)
        self.assertAlmostEqual(a["metrics"]["delta_full"]["estimate"], 2 / 3)
        self.assertEqual(a["metrics"]["delta_full"]["estimate"], b["metrics"]["delta_full"]["estimate"])

    def test_unpaired_duplicate_corrupt_target_and_nonfinite_scores_rejected(self):
        manifest = self.manifest(document_ids=["a", "b", "c"])
        good = self.scores(manifest)
        bad_cases = [good[:-1], good + [good[0]]]
        for field, value in (("target_sha256", "wrong"), ("target_count", 2), ("nll_sum", float("nan"))):
            bad = copy.deepcopy(good)
            bad[0][field] = value
            bad_cases.append(bad)
        for bad in bad_cases:
            with self.assertRaises(ValueError):
                analyze_four_conditions(manifest, bad)


if __name__ == "__main__":
    unittest.main()
