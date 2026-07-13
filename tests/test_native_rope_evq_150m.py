#!/usr/bin/env python3
"""Regression gates for the native-RoPE versus endpoint-EVQ 150M pilot."""

from __future__ import annotations

import math
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from experiments.native_rope_evq_150m.prepare_data import (
    assert_independent_validation,
    compile_passkey_cache,
    validate_data_manifest,
)
from experiments.native_rope_evq_150m.protocol import (
    LEGACY_PASSKEY_HASH_MULTIPLIER,
    SPEC,
    estimate_parameter_count,
    get_arm_inv_freq,
    legacy_passkey_indices,
)
from experiments.native_rope_evq_150m.train import (
    FrozenMixedDataset,
    build_model,
    deterministic_row_order,
    learning_rate_for_step,
    trainable_state_sha256,
    validate_cuda_runtime,
)
from scripts.lib.rope.schedules import geometric_inv_freq


class TestNativeRopeEvq150MProtocol(unittest.TestCase):
    def test_model_and_training_counts_are_exact(self):
        self.assertEqual(estimate_parameter_count(), 151_898_880)
        self.assertEqual(SPEC.train_rows, 244_140)
        self.assertEqual(SPEC.train_tokens, 499_998_720)
        self.assertEqual(SPEC.batch_size, 60)
        self.assertEqual(SPEC.optimizer_steps, 4_069)
        self.assertEqual(SPEC.seq_len, 2_048)

    def test_legacy_passkey_selection_matches_previous_dataset(self):
        indices = legacy_passkey_indices(SPEC.train_rows)
        expected = tuple(
            i
            for i in range(SPEC.train_rows)
            if random.Random(i * LEGACY_PASSKEY_HASH_MULTIPLIER + 1).random()
            < SPEC.passkey_mix_ratio
        )
        self.assertEqual(indices, expected)
        self.assertEqual(len(indices), 4_926)
        self.assertEqual(len(indices) * SPEC.seq_len, 10_088_448)

    def test_native_arm_is_standard_endpoint_rope(self):
        actual = get_arm_inv_freq("native_rope")
        expected = geometric_inv_freq(
            head_dim=SPEC.head_dim,
            base=SPEC.rope_base,
            dtype=torch.float64,
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(float(actual[0]), 1.0)

    def test_endpoint_evq_is_finite_distinct_and_tau_zero_recovers_native(self):
        evq = get_arm_inv_freq("endpoint_evq_tau1p5")
        native = get_arm_inv_freq("native_rope")
        evq_zero = get_arm_inv_freq("endpoint_evq_tau1p5", tau_override=0.0)
        self.assertTrue(torch.isfinite(evq).all())
        self.assertFalse(torch.allclose(evq, native, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(evq_zero, native, atol=1e-12, rtol=1e-12))
        self.assertAlmostEqual(SPEC.evq_tau, 1.5)
        self.assertAlmostEqual(
            SPEC.head_dim / math.sqrt(SPEC.seq_len), math.sqrt(2.0), places=12
        )

    def test_unknown_arm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown arm"):
            get_arm_inv_freq("geo")


class _TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [1_000 + (ord(ch) % 200) for ch in str(text)]


class TestNativeRopeEvq150MData(unittest.TestCase):
    def test_compile_passkey_cache_uses_legacy_selector_and_train_row_filler(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = np.arange(20 * 64, dtype=np.int64).reshape(20, 64)
            train_path = root / "train.npy"
            np.save(train_path, train)
            result = compile_passkey_cache(
                train_path=train_path,
                tokenizer=_TinyTokenizer(),
                output_dir=root / "prepared",
                seq_len=64,
                ratio=0.25,
            )
            expected = legacy_passkey_indices(20, ratio=0.25)
            selected = np.load(result["indices_path"])
            samples = np.load(result["passkey_path"], mmap_mode="r")
            self.assertEqual(tuple(selected.tolist()), expected)
            self.assertEqual(samples.shape, (len(expected), 64))
            self.assertEqual(samples.dtype, np.int64)
            # The old helper is deterministic; recompilation must be byte-identical.
            result2 = compile_passkey_cache(
                train_path=train_path,
                tokenizer=_TinyTokenizer(),
                output_dir=root / "prepared_again",
                seq_len=64,
                ratio=0.25,
            )
            self.assertEqual(result["passkey_sha256"], result2["passkey_sha256"])

    def test_identical_validation_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = np.arange(128, dtype=np.int64).reshape(2, 64)
            np.save(root / "train.npy", train)
            np.save(root / "val.npy", train.reshape(-1)[:80])
            with self.assertRaisesRegex(ValueError, "overlaps the training prefix"):
                assert_independent_validation(root / "train.npy", root / "val.npy")

    def test_independent_validation_prefix_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "train.npy", np.arange(128, dtype=np.int64).reshape(2, 64))
            np.save(root / "val.npy", np.arange(1_000, 1_080, dtype=np.int64))
            checked = assert_independent_validation(root / "train.npy", root / "val.npy")
            self.assertEqual(checked, 80)

    def test_manifest_rejects_legacy_validation_and_reused_source_shard(self):
        base = {
            "schema_version": 1,
            "train": {"sha256": SPEC.train_npy_sha256},
            "passkey": {"selector": "legacy_hash_v1", "ratio": 0.02},
            "validation": {
                "sha256": SPEC.forbidden_leaked_val_sha256,
                "source_shard": "004_00000.parquet",
            },
            "train_source_shards": ["000_00000.parquet"],
        }
        with self.assertRaisesRegex(ValueError, "forbidden leaked validation"):
            validate_data_manifest(base, check_files=False)
        base["validation"]["sha256"] = "a" * 64
        base["validation"]["source_shard"] = "000_00000.parquet"
        with self.assertRaisesRegex(ValueError, "also listed as a training source"):
            validate_data_manifest(base, check_files=False)


class TestNativeRopeEvq150MTraining(unittest.TestCase):
    def test_row_order_is_a_seeded_shared_permutation(self):
        first = deterministic_row_order(100, seed=42)
        second = deterministic_row_order(100, seed=42)
        other = deterministic_row_order(100, seed=43)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        self.assertEqual(sorted(first.tolist()), list(range(100)))

    def test_memmap_dataset_substitutes_only_registered_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = np.arange(8 * 16, dtype=np.int64).reshape(8, 16)
            passkeys = np.full((2, 16), 9_999, dtype=np.int64)
            np.save(root / "train.npy", train)
            np.save(root / "passkeys.npy", passkeys)
            np.save(root / "indices.npy", np.asarray([2, 6], dtype=np.int64))
            dataset = FrozenMixedDataset(
                train_path=root / "train.npy",
                passkey_path=root / "passkeys.npy",
                indices_path=root / "indices.npy",
                expected_rows=8,
                seq_len=16,
            )
            self.assertTrue(torch.equal(dataset[0], torch.from_numpy(train[0])))
            self.assertTrue(torch.equal(dataset[2], torch.full((16,), 9_999)))
            self.assertTrue(torch.equal(dataset[6], torch.full((16,), 9_999)))

    def test_trainable_initialization_is_identical_across_frequency_arms(self):
        tiny = replace(
            SPEC,
            vocab_size=128,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            head_dim=8,
            intermediate_size=64,
            seq_len=64,
            train_tokens_requested=1_024,
            batch_size=4,
        )
        native = build_model("native_rope", spec=tiny, seed=42)
        evq = build_model("endpoint_evq_tau1p5", spec=tiny, seed=42)
        self.assertEqual(
            trainable_state_sha256(native), trainable_state_sha256(evq)
        )
        self.assertFalse(
            torch.equal(
                native.blocks[0].attn.rope.inv_freq,
                evq.blocks[0].attn.rope.inv_freq,
            )
        )

    def test_learning_rate_matches_registered_warmup_and_floor(self):
        self.assertEqual(learning_rate_for_step(0, SPEC), 0.0)
        self.assertAlmostEqual(
            learning_rate_for_step(SPEC.warmup_steps, SPEC), SPEC.learning_rate
        )
        self.assertAlmostEqual(
            learning_rate_for_step(SPEC.optimizer_steps - 1, SPEC),
            SPEC.min_learning_rate,
            delta=1e-8,
        )

    def test_cuda_runtime_gate_rejects_cpu_mode(self):
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA is required"):
                validate_cuda_runtime()


if __name__ == "__main__":
    unittest.main()
