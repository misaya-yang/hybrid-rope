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
import torch.nn.functional as F

from experiments.native_rope_evq_150m import evaluate as evaluate_module
from experiments.native_rope_evq_150m.model import (
    GPT as ExperimentGPT,
    RotaryEmbedding,
    apply_rope,
)
from experiments.native_rope_evq_150m.evaluate import (
    aggregate_nll,
    apply_registered_operator,
    causal_nll_metrics,
    fixed_validation_offsets,
    load_weights_only_checkpoint,
    target_yarn_factor,
)
from experiments.native_rope_evq_150m.prepare_data import (
    assert_independent_validation,
    compile_passkey_cache,
    validate_data_manifest,
)
from experiments.native_rope_evq_150m.protocol import (
    ARMS,
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
    meta_parameter_count,
    registered_inv_freq_sha256,
    tensor_sha256,
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
        self.assertEqual(SPEC.micro_batch_size, 12)
        self.assertEqual(SPEC.grad_accum_steps, 5)
        self.assertEqual(SPEC.micro_steps, 20_345)
        self.assertEqual(SPEC.optimizer_steps, 4_069)
        self.assertEqual(SPEC.seq_len, 2_048)

    def test_legacy_passkey_selection_matches_previous_dataset(self):
        self.assertEqual(SPEC.passkey_mix_ratio, 0.02)
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
        # The historical Primary-I run used 10% of 100M tokens.  The new
        # 500M run keeps that approximately 10M-token absolute task budget
        # instead of multiplying synthetic exposure by five.
        self.assertLess(
            abs(len(indices) * SPEC.seq_len - 10_000_000) / 10_000_000,
            0.01,
        )

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

    def test_repo_fixed_ramp_is_registered_as_a_separate_operator(self):
        native = get_arm_inv_freq("native_rope")
        fixed, mscale, meta = apply_registered_operator(
            native,
            arm="native_rope",
            operator="repo_fixed_ramp",
            length=8_192,
        )
        official, official_mscale, official_meta = apply_registered_operator(
            native,
            arm="native_rope",
            operator="yarn",
            length=8_192,
        )
        self.assertEqual(mscale, 1.0)
        self.assertGreater(official_mscale, 1.0)
        self.assertEqual(meta["mode"], "repo_fixed_ramp")
        self.assertEqual(official_meta["mode"], "official_yarn_native")
        self.assertFalse(torch.allclose(fixed, official))


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
    def test_legacy_torch_version_metadata_loads_weights_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save(
                {
                    "model": {"weight": torch.arange(4)},
                    "metadata": {"torch": torch.__version__},
                },
                path,
            )
            loaded = load_weights_only_checkpoint(path)
            self.assertTrue(
                torch.equal(loaded["model"]["weight"], torch.arange(4))
            )

    def test_rope_preserves_bf16_activation_dtype(self):
        x = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
        cos = torch.randn(4, 8, dtype=torch.float32)
        sin = torch.randn(4, 8, dtype=torch.float32)
        actual = apply_rope(x, cos[None, None], sin[None, None])
        self.assertEqual(actual.dtype, torch.bfloat16)

    def test_rotary_attention_scaling_is_explicit_and_resettable(self):
        rope = RotaryEmbedding(8, 16, torch.ones(4))
        raw_cos, raw_sin = rope(8)
        rope.attention_scaling = 1.2
        scaled_cos, scaled_sin = rope(8)
        self.assertTrue(torch.allclose(scaled_cos, raw_cos * 1.2))
        self.assertTrue(torch.allclose(scaled_sin, raw_sin * 1.2))
        rope.attention_scaling = 1.0
        reset_cos, reset_sin = rope(8)
        self.assertTrue(torch.equal(reset_cos, raw_cos))
        self.assertTrue(torch.equal(reset_sin, raw_sin))

    def test_row_order_is_a_seeded_shared_permutation(self):
        first = deterministic_row_order(100, seed=42)
        second = deterministic_row_order(100, seed=42)
        other = deterministic_row_order(100, seed=43)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        self.assertEqual(sorted(first.tolist()), list(range(100)))

    def test_registered_full_model_parameter_count_on_meta_device(self):
        self.assertEqual(meta_parameter_count("native_rope"), 151_898_880)
        self.assertEqual(meta_parameter_count("endpoint_evq_tau1p5"), 151_898_880)

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
            micro_batch_size=4,
        )
        native = build_model("native_rope", spec=tiny, seed=42)
        evq = build_model("endpoint_evq_tau1p5", spec=tiny, seed=42)
        self.assertIsInstance(native, ExperimentGPT)
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

    def test_frequency_metadata_hashes_the_registered_float64_schedule(self):
        for arm in ARMS:
            self.assertEqual(
                registered_inv_freq_sha256(arm),
                tensor_sha256(get_arm_inv_freq(arm)),
            )

    def test_cuda_runtime_gate_rejects_cpu_mode(self):
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA is required"):
                validate_cuda_runtime()


class TestNativeRopeEvq150MEvaluation(unittest.TestCase):
    def test_target_matched_yarn_factors(self):
        self.assertEqual(
            [target_yarn_factor(length) for length in (2048, 4096, 8192, 16384)],
            [1.0, 2.0, 4.0, 8.0],
        )
        with self.assertRaisesRegex(ValueError, "registered evaluation length"):
            target_yarn_factor(3072)

    def test_native_operator_is_official_and_evq_operator_is_derived(self):
        native = get_arm_inv_freq("native_rope")
        evq = get_arm_inv_freq("endpoint_evq_tau1p5")
        _, _, native_meta = apply_registered_operator(
            native, arm="native_rope", operator="yarn", length=8192
        )
        _, _, evq_meta = apply_registered_operator(
            evq, arm="endpoint_evq_tau1p5", operator="yarn", length=8192
        )
        self.assertEqual(native_meta["mode"], "official_yarn_native")
        self.assertIn("official YaRN", native_meta["public_label"])
        self.assertEqual(evq_meta["mode"], "yarn_derived_virtual_dim")
        self.assertIn("YaRN-derived", evq_meta["public_label"])

    def test_raw_operator_preserves_substrate(self):
        inv = get_arm_inv_freq("endpoint_evq_tau1p5")
        actual, mscale, meta = apply_registered_operator(
            inv, arm="endpoint_evq_tau1p5", operator="raw", length=16384
        )
        self.assertTrue(torch.equal(actual, inv))
        self.assertEqual(mscale, 1.0)
        self.assertEqual(meta["mode"], "raw_substrate")

    def test_yarn_components_are_independently_switchable(self):
        for arm in ARMS:
            base = get_arm_inv_freq(arm)
            raw, raw_mscale, _ = apply_registered_operator(
                base,
                arm=arm,
                operator="raw",
                length=16_384,
                use_frequency_transform=False,
                use_attention_scaling=False,
            )
            freq, freq_mscale, _ = apply_registered_operator(
                base,
                arm=arm,
                operator="freq_only",
                length=16_384,
                use_frequency_transform=True,
                use_attention_scaling=False,
            )
            mscale, mscale_value, _ = apply_registered_operator(
                base,
                arm=arm,
                operator="mscale_only",
                length=16_384,
                use_frequency_transform=False,
                use_attention_scaling=True,
            )
            full, full_mscale, full_meta = apply_registered_operator(
                base,
                arm=arm,
                operator="full",
                length=16_384,
                use_frequency_transform=True,
                use_attention_scaling=True,
            )
            self.assertTrue(torch.equal(raw, base))
            self.assertEqual(raw_mscale, 1.0)
            self.assertTrue(torch.equal(mscale, base))
            self.assertGreater(mscale_value, 1.0)
            self.assertTrue(torch.equal(freq, full))
            self.assertEqual(freq_mscale, 1.0)
            self.assertEqual(mscale_value, full_mscale)
            self.assertEqual(full_meta["use_frequency_transform"], True)
            self.assertEqual(full_meta["use_attention_scaling"], True)

    def test_checkpoint_identity_rejects_changed_bytes(self):
        self.assertTrue(hasattr(evaluate_module, "validate_checkpoint_identity"))
        validate_checkpoint_identity = evaluate_module.validate_checkpoint_identity
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arm = "native_rope"
            np.save(root / "inv_freq.npy", get_arm_inv_freq(arm).numpy())
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"registered checkpoint")
            checkpoint_sha = __import__("hashlib").sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            metadata = {
                "arm": arm,
                "seed": 42,
                "checkpoint_sha256": checkpoint_sha,
                "inv_freq_sha256": tensor_sha256(get_arm_inv_freq(arm)),
                "initial_trainable_sha256": "a" * 64,
                "row_order_sha256": "b" * 64,
                "data_manifest_sha256": "c" * 64,
            }
            (root / "train_meta.json").write_text(__import__("json").dumps(metadata))
            identity = validate_checkpoint_identity(root, arm, metadata)
            self.assertEqual(identity["checkpoint_sha256"], checkpoint_sha)
            checkpoint.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint sha256 mismatch"):
                validate_checkpoint_identity(root, arm, metadata)

    def test_ablation_attribution_uses_registered_sign_convention(self):
        from experiments.native_rope_evq_150m import evaluate_yarn_ablation

        self.assertTrue(hasattr(evaluate_yarn_ablation, "attribution_row"))
        row = evaluate_yarn_ablation.attribution_row(
            native_raw=5.0,
            evq_raw=4.0,
            native_operator=3.0,
            evq_operator=2.5,
        )
        self.assertEqual(row["substrate_gap"], 0.5)
        self.assertEqual(row["interaction"], 0.5)

    def test_offsets_are_shared_deterministic_and_valid(self):
        first = fixed_validation_offsets(5_000_000, 16_384, chunks=8, seed=9999)
        second = fixed_validation_offsets(5_000_000, 16_384, chunks=8, seed=9999)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(len(set(first)), 8)
        self.assertTrue(all(0 <= offset <= 5_000_000 - 16_384 for offset in first))

    def test_nll_aggregation_reports_ppl_without_rounding_inputs(self):
        result = aggregate_nll([2.0, 3.0, 4.0])
        self.assertEqual(result["sample_count"], 3)
        self.assertAlmostEqual(result["mean_nll"], 3.0)
        self.assertAlmostEqual(result["ppl"], math.exp(3.0))

    def test_causal_nll_metrics_separates_full_context_and_last_tokens(self):
        logits = torch.tensor(
            [
                [
                    [8.0, 0.0],
                    [8.0, 0.0],
                    [0.0, 8.0],
                    [0.0, 8.0],
                ]
            ]
        )
        targets = torch.tensor([[0, 0, 0, 0]])
        full, tail = causal_nll_metrics(logits, targets, tail_tokens=2)
        self.assertGreater(tail, full)
        self.assertAlmostEqual(
            full,
            float(F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))),
        )

    def test_launcher_has_prepare_preflight_run_and_no_private_default_paths(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "native_rope_evq_150m"
            / "run_seed42.sh"
        )
        self.assertTrue(launcher.is_file())
        text = launcher.read_text()
        for mode in ("prepare)", "preflight)", "run)"):
            self.assertIn(mode, text)
        self.assertIn("TORCHINDUCTOR_CACHE_DIR", text)
        self.assertIn("endpoint_evq_tau1p5", text)
        self.assertIn('"$PACKAGE_DIR/model.py"', text)
        self.assertNotIn("/root/autodl-tmp", text)
        self.assertNotIn("seetacloud", text)


if __name__ == "__main__":
    unittest.main()
