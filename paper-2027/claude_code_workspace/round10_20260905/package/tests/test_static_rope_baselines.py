"""NumPy + standard-library CPU checks; no torch/transformers required."""

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.analysis.export_static_rope_baselines import (
    bind_reference, build_tables, compare_tables, export, load_inputs,
    resolve_native_initializer, sha256_file, tensor_hash, verify_transformers,
)


class TestStaticRopeBaselines(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.json"
        self.native_path = self.root / "native.npy"
        self.raw = {"model_type": "llama", "head_dim": 64, "rope_theta": 500000.0,
                    "max_position_embeddings": 2048, "rope_scaling": None}
        self.native = (500000.0 ** (-np.arange(32) / 32)).astype(np.float32)
        self.save_inputs()

    def save_inputs(self):
        self.config.write_text(json.dumps(self.raw))
        np.save(self.native_path, self.native)

    def args(self, output="out", verify=False):
        return argparse.Namespace(config=self.config, native_inv=self.native_path,
                                  factor=8.0, verify_transformers=verify,
                                  output=self.root / output)

    def test_fixed_equations_gain_endpoints_and_rounding(self):
        identity, native = load_inputs(self.config, self.native_path)
        tables, meta = build_tables(identity, native, 8.0)
        yarn = meta["official_equation_yarn"]
        self.assertEqual((yarn["low"], yarn["high"]), (5, 15))
        self.assertEqual(yarn["attention_scaling"], 1 + 0.1 * math.log(8))
        ramp = np.clip((np.arange(32) - 5) / 10, 0, 1)
        expected = (native.astype(np.float64) * (ramp / 8 + 1 - ramp)).astype(np.float32)
        np.testing.assert_array_equal(tables["official_equation_yarn"], expected)
        ntk = tables["static_ntk"]
        self.assertEqual(ntk[0], native[0])
        self.assertEqual(ntk[-1], np.float32(float(native[-1]) / 8))
        self.assertEqual(meta["static_ntk"]["base_multiplier"], 8 ** (64 / 62))
        self.assertEqual(meta["static_ntk"]["attention_scaling"], 1.0)
        for table in tables.values():
            self.assertEqual(table.dtype, np.float32)
            self.assertTrue(np.all(table[:-1] > table[1:]))

    def test_deterministic_manifest_and_hashes(self):
        first = export(self.args())
        second = export(self.args("other"))
        self.assertEqual(first, second)
        self.assertFalse(first["search_performed"])
        self.assertEqual(first["transformers_verification"]["status"], "NOT_RUN")
        self.assertEqual(first["checkpoint"]["config_sha256"], sha256_file(self.config))
        self.assertEqual(first["checkpoint"]["native_tensor_file_sha256"], sha256_file(self.native_path))
        for row in first["tables"].values():
            path = self.root / "out" / row["path"]
            self.assertEqual(row["file_sha256"], sha256_file(path))
            self.assertEqual(row["tensor_sha256"], tensor_hash(np.load(path)))
        with self.assertRaises(FileExistsError):
            export(self.args())

    def test_reject_scaled_partial_and_mismatched_native(self):
        for change in (
            {"rope_scaling": {"rope_type": "linear", "factor": 2}},
            {"partial_rotary_factor": 0.5},
            {"rope_parameters": {"rope_type": "yarn"}},
            {"rope_parameters": {"partial_rotary_factor": 0.5}},
            {"rope_parameters": {"factor": 2}},
            {"head_dim": 2}, {"rope_theta": float("nan")},
        ):
            with self.subTest(change=change):
                self.config.write_text(json.dumps({**self.raw, **change}))
                with self.assertRaises(ValueError):
                    load_inputs(self.config, self.native_path)
        self.save_inputs()
        for native in (self.native.astype(np.float64), self.native[:-1], self.native * 0.9,
                       self.native[::-1], np.full(32, np.nan, dtype=np.float32)):
            with self.subTest(shape=native.shape, dtype=native.dtype):
                np.save(self.native_path, native)
                with self.assertRaises(ValueError):
                    load_inputs(self.config, self.native_path)

    def test_invalid_factor(self):
        identity, native = load_inputs(self.config, self.native_path)
        for factor in (0, 1, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                build_tables(identity, native, factor)

    def test_full_head_grids_preserve_static_ntk_endpoints(self):
        for dim in (128, 256):
            with self.subTest(dim=dim):
                self.raw.update(head_dim=dim, rope_theta=10000.0,
                                max_position_embeddings=8192)
                self.native = (10000.0 ** (-np.arange(dim // 2) / (dim // 2))).astype(np.float32)
                self.save_inputs()
                identity, native = load_inputs(self.config, self.native_path)
                tables, _ = build_tables(identity, native, 2.0)
                self.assertEqual(identity["K"], dim // 2)
                self.assertEqual(tables["static_ntk"][-1], native[-1] / 2)
                self.assertTrue(np.all(tables["official_equation_yarn"][:-1] >
                                       tables["official_equation_yarn"][1:]))

    def test_tolerance_is_not_exact_hash_parity(self):
        shifted = np.nextafter(self.native, np.float32(np.inf))
        receipt = compare_tables(shifted, self.native)
        self.assertTrue(receipt["passed"])
        self.assertFalse(receipt["exact_tensor_hash_match"])
        with self.assertRaisesRegex(ValueError, "parity failed"):
            compare_tables(self.native * 0.99, self.native)

    def test_verified_export_uses_actual_hf_tensor_mock_not_runtime_evidence(self):
        identity, native = load_inputs(self.config, self.native_path)
        tables, _ = build_tables(identity, native, 8.0)
        actual = np.nextafter(tables["official_equation_yarn"], np.float32(np.inf))
        with patch("scripts.analysis.export_static_rope_baselines.verify_transformers",
                   return_value=(actual, {"passed": True, "test_mock": True})):
            receipt = export(self.args(verify=True))
        row = receipt["tables"]["official_equation_yarn"]
        self.assertEqual(row["tensor_sha256"], tensor_hash(actual))
        np.testing.assert_array_equal(np.load(self.root / "out" / row["path"]), actual)

    def test_verification_failure_writes_no_artifacts(self):
        with patch("scripts.analysis.export_static_rope_baselines.verify_transformers",
                   side_effect=ValueError("HF initializer parity failed")):
            with self.assertRaisesRegex(ValueError, "parity failed"):
                export(self.args(verify=True))
        self.assertFalse((self.root / "out").exists())

    def reference_receipt(self, **changes):
        receipt = {
            "status": "NATIVE_REFERENCE_CONFIRMED", "reference_length": 1024,
            "checkpoint_weight_sha256": "a" * 64,
            "config_sha256": sha256_file(self.config),
            "native_sha256_float32": tensor_hash(self.native),
            "confirmation_decision_sha256": "b" * 64,
            "data_manifest_sha256": "c" * 64,
        }
        receipt.update(changes)
        path = self.root / "reference.json"
        path.write_text(json.dumps(receipt))
        return path

    def test_confirmed_reference_changes_yarn_ramp_not_native_or_ntk(self):
        identity, native = load_inputs(self.config, self.native_path)
        old_tables, old_meta = build_tables(identity, native, 2.0)
        args = self.args()
        args.reference_receipt = self.reference_receipt()
        args.target_length = 2048
        args.factor = None
        config_hash = sha256_file(self.config)
        native_hash = sha256_file(self.native_path)
        receipt = export(args)
        self.assertEqual(receipt["checkpoint"]["native_length"], 2048)
        self.assertEqual(receipt["checkpoint"]["L"], 2048)
        self.assertEqual(receipt["reference_length"], 1024)
        self.assertEqual(receipt["factor"], 2.0)
        self.assertEqual(receipt["target_length"], 2048)
        self.assertEqual(receipt["reference_receipt"]["receipt_sha256"], sha256_file(args.reference_receipt))
        self.assertTrue(receipt["benchmark_scores_used"])
        self.assertFalse(receipt["long_benchmark_scores_used"])
        yarn = receipt["tables"]["official_equation_yarn"]
        self.assertEqual(yarn["original_max_position_embeddings"], 1024)
        self.assertNotEqual((yarn["low"], yarn["high"]),
                            (old_meta["official_equation_yarn"]["low"], old_meta["official_equation_yarn"]["high"]))
        ntk_path = args.output / receipt["tables"]["static_ntk"]["path"]
        np.testing.assert_array_equal(np.load(ntk_path), old_tables["static_ntk"])
        self.assertEqual(sha256_file(self.config), config_hash)
        self.assertEqual(sha256_file(self.native_path), native_hash)

    def test_reference_rejects_unconfirmed_lengths_hashes_and_missing_evidence(self):
        for change in (
            {"status": "PROVISIONAL"}, {"reference_length": 0},
            {"reference_length": 768}, {"reference_length": 4096},
            {"reference_length": True}, {"reference_length": 1024.0},
            {"config_sha256": "d" * 64}, {"native_sha256_float32": "d" * 64},
            {"checkpoint_weight_sha256": None}, {"confirmation_decision_sha256": ""},
            {"data_manifest_sha256": "not a hash"},
        ):
            with self.subTest(change=change):
                args = self.args()
                args.reference_receipt = self.reference_receipt(**change)
                args.target_length, args.factor = 2048, 2.0
                with self.assertRaises(ValueError):
                    export(args)
                self.assertFalse(args.output.exists())

    def test_reference_target_and_factor_are_bound_exactly(self):
        identity, _ = load_inputs(self.config, self.native_path)
        path = self.reference_receipt()
        for target, factor in ((None, 2), (1024, 1), (2048, 4),
                               (2048, np.nextafter(2.0, 3.0)), (True, None)):
            with self.subTest(target=target, factor=factor):
                with self.assertRaises(ValueError):
                    bind_reference(identity, path, target, factor)
        factor, _ = bind_reference(identity, path, 2048, 2.0)
        self.assertEqual(factor, 2.0)
        with self.assertRaises(ValueError):
            bind_reference(identity, None, 2048, 2.0)
        with self.assertRaises(ValueError):
            bind_reference(identity, None, None, None)

    def test_native_initializer_legacy_and_fail_closed_unknown_api(self):
        legacy = object()
        with patch("scripts.analysis.export_static_rope_baselines.importlib.import_module") as importer:
            self.assertIs(resolve_native_initializer({"default": legacy}, "gemma"), legacy)
            importer.assert_not_called()
            with self.assertRaisesRegex(ValueError, "unverified"):
                resolve_native_initializer({}, "unknown")
            importer.return_value = SimpleNamespace(GemmaRotaryEmbedding=object)
            with self.assertRaisesRegex(ValueError, "no verified"):
                resolve_native_initializer({}, "gemma")

    def test_hf5_model_static_initializer_and_reference_keep_real_config(self):
        self.raw["model_type"] = "gemma"
        self.save_inputs()
        identity, native = load_inputs(self.config, self.native_path)
        identity["reference_length"] = 1024
        tables, meta = build_tables(identity, native, 2.0)
        equation = tables["official_equation_yarn"]
        gain = meta["official_equation_yarn"]["attention_scaling"]
        observed = {}

        class Tensor:
            def __init__(self, array):
                self.array = array
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.array

        class GemmaRotaryEmbedding:
            def __init__(self, *args, **kwargs):
                raise AssertionError("no rotary/full model instance may be constructed")

            @staticmethod
            def compute_default_rope_parameters(config, device):
                observed["native"] = copy.deepcopy(config)
                return Tensor(native), 1.0

        def yarn(config, device):
            observed["yarn"] = copy.deepcopy(config)
            return Tensor(equation), gain

        class AutoConfig:
            @staticmethod
            def for_model(model_type, **raw):
                return SimpleNamespace(model_type=model_type, **copy.deepcopy(raw))

        torch = ModuleType("torch")
        torch.__version__, torch.device = "mock-cpu", lambda value: value
        transformers = ModuleType("transformers")
        transformers.__version__, transformers.AutoConfig = "5.15.1-mock", AutoConfig
        rope_utils = ModuleType("transformers.modeling_rope_utils")
        rope_utils.ROPE_INIT_FUNCTIONS = {"yarn": yarn}
        modules = {"torch": torch, "transformers": transformers,
                   "transformers.modeling_rope_utils": rope_utils}
        model_module = SimpleNamespace(GemmaRotaryEmbedding=GemmaRotaryEmbedding)
        config_hash = sha256_file(self.config)
        with patch.dict(sys.modules, modules), patch(
            "scripts.analysis.export_static_rope_baselines.importlib.import_module",
            return_value=model_module,
        ) as importer:
            actual, receipt = verify_transformers(
                self.config, native, equation, 2.0, gain, reference_length=1024,
            )
        importer.assert_called_once_with("transformers.models.gemma.modeling_gemma")
        np.testing.assert_array_equal(actual, equation)
        self.assertTrue(receipt["native_parity"]["exact_tensor_hash_match"])
        self.assertIn("compute_default_rope_parameters", receipt["native_initializer"])
        self.assertEqual(observed["native"].max_position_embeddings, 2048)
        self.assertIsNone(observed["native"].rope_scaling)
        self.assertEqual(observed["yarn"].max_position_embeddings, 2048)
        self.assertEqual(observed["yarn"].rope_parameters["original_max_position_embeddings"], 1024)
        self.assertEqual(observed["yarn"].rope_scaling["original_max_position_embeddings"], 1024)
        self.assertEqual(sha256_file(self.config), config_hash)


if __name__ == "__main__":
    unittest.main()
