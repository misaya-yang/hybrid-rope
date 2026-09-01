"""NumPy + standard-library CPU checks; no torch/transformers required."""

import argparse
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.analysis.export_static_rope_baselines import (
    build_tables, compare_tables, export, load_inputs, sha256_file, tensor_hash,
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


if __name__ == "__main__":
    unittest.main()
