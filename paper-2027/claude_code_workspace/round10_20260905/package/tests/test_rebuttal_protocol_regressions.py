#!/usr/bin/env python3
"""Regression tests for protocol bugs found during rebuttal preparation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "lora_evq_v2"))

from experiments.lora_evq_v2 import (
    eval_positional_ppl,
    eval_ruler,
    train_evq_lora,
    validate_checkpoint_artifact,
)
from scripts.lib.rope.schedules import geometric_inv_freq
from scripts.text_eval import llama3_continued_pretrain


class LoraFrequencyProtocolTests(unittest.TestCase):
    def test_training_arguments_keyword_tracks_installed_api(self):
        class ModernTrainingArguments:
            def __init__(self, *, eval_strategy="no"):
                pass

        class LegacyTrainingArguments:
            def __init__(self, *, evaluation_strategy="no"):
                pass

        self.assertEqual(
            train_evq_lora.evaluation_strategy_kwargs(ModernTrainingArguments),
            {"eval_strategy": "no"},
        )
        self.assertEqual(
            train_evq_lora.evaluation_strategy_kwargs(LegacyTrainingArguments),
            {"evaluation_strategy": "no"},
        )

    def test_native_geo_uses_standard_endpoint_schedule(self):
        actual, metadata = train_evq_lora.build_training_inv_freq(
            rope_method="native_geo",
            head_dim=128,
            base=500_000.0,
            tau=0.0,
        )
        expected = geometric_inv_freq(128, 500_000.0)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(metadata["method"], "native_geo")
        self.assertFalse(metadata["midpoint"])

    def test_tau_zero_midpoint_evq_is_not_native_geo(self):
        midpoint, _ = train_evq_lora.build_training_inv_freq(
            rope_method="evq_cosh",
            head_dim=128,
            base=500_000.0,
            tau=0.0,
        )
        native, _ = train_evq_lora.build_training_inv_freq(
            rope_method="native_geo",
            head_dim=128,
            base=500_000.0,
            tau=0.0,
        )

        self.assertFalse(torch.allclose(midpoint, native))

    def test_geo_evaluation_reuses_saved_training_frequencies(self):
        with tempfile.TemporaryDirectory() as td:
            adapter = Path(td)
            freq_path = adapter / "custom_inv_freq.pt"
            torch.save({"inv_freq": torch.ones(64), "method": "native_geo"}, freq_path)

            self.assertEqual(
                eval_positional_ppl.resolve_custom_inv_freq_path(adapter, "geo"),
                freq_path,
            )
            self.assertEqual(
                eval_positional_ppl.resolve_custom_inv_freq_path(adapter, "evq"),
                freq_path,
            )
            self.assertIsNone(
                eval_positional_ppl.resolve_custom_inv_freq_path(adapter, "yarn")
            )

    def test_geo_evaluation_fails_when_training_frequencies_are_missing(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(FileNotFoundError, "custom_inv_freq"):
                eval_positional_ppl.resolve_custom_inv_freq_path(td, "geo")

    def test_frequency_artifact_rejects_method_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "custom_inv_freq.pt"
            torch.save(
                {
                    "inv_freq": torch.ones(64),
                    "method": "native_geo",
                    "head_dim": 128,
                    "base": 500_000.0,
                    "midpoint": False,
                },
                path,
            )
            with self.assertRaisesRegex(RuntimeError, "method mismatch"):
                train_evq_lora.load_frequency_artifact(
                    path,
                    expected_method="evq_cosh",
                )

    def test_frequency_provenance_never_contains_absolute_path(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "custom_inv_freq.pt"
            torch.save(
                {
                    "inv_freq": torch.ones(64),
                    "method": "evq_cosh",
                    "head_dim": 128,
                    "base": 500_000.0,
                    "tau": 1.414,
                    "midpoint": True,
                },
                path,
            )
            _, _, provenance = train_evq_lora.load_frequency_artifact(
                path,
                expected_method="evq_cosh",
            )
            serialized = json.dumps(provenance)
            self.assertNotIn(td, serialized)
            self.assertEqual(provenance["artifact"], "custom_inv_freq.pt")
            self.assertEqual(len(provenance["sha256"]), 64)

    def test_frequency_verification_checks_every_rotary_module(self):
        class Rope(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.register_buffer("inv_freq", torch.full((4,), value))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = Rope(1.0)
                self.second = Rope(2.0)

        with self.assertRaisesRegex(RuntimeError, "second"):
            train_evq_lora.verify_model_inv_freq(Model(), torch.ones(4))

    def test_model_geometry_is_inferred_instead_of_hard_coded(self):
        config = SimpleNamespace(
            hidden_size=2048,
            num_attention_heads=32,
            rope_theta=500_000.0,
        )
        geometry = train_evq_lora.resolve_model_rope_geometry(config)
        self.assertEqual(geometry.head_dim, 64)
        self.assertEqual(geometry.rope_base, 500_000.0)

    def test_local_model_identifier_is_sanitized_for_public_metadata(self):
        local_model = str(Path.cwd() / "models" / "Meta-Llama-3-8B-Instruct")
        self.assertEqual(
            train_evq_lora.public_model_identifier(local_model),
            "Meta-Llama-3-8B-Instruct",
        )
        self.assertEqual(
            train_evq_lora.public_model_identifier(
                "meta-llama/Meta-Llama-3-8B-Instruct"
            ),
            "meta-llama/Meta-Llama-3-8B-Instruct",
        )

    def test_geo_launchers_request_native_geo_explicitly(self):
        launchers = [
            ROOT / "experiments/lora_evq_v2/run_pe_comparison.sh",
            ROOT / "scripts/2026-04/01a_lora_train_geo_s42.sh",
            ROOT / "scripts/2026-04/01b_lora_train_geo_s43.sh",
            ROOT / "scripts/2026-04/01c_lora_train_geo_s44.sh",
        ]
        for launcher in launchers:
            text = launcher.read_text(encoding="utf-8")
            self.assertIn("--rope_method native_geo", text, launcher)


class RulerOutputProtocolTests(unittest.TestCase):
    def test_adapter_evaluation_requires_frequency_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(FileNotFoundError, "custom_inv_freq"):
                eval_ruler.resolve_required_inv_freq_path(td, base_only=False)

    def test_ruler_frequency_verification_accepts_exact_native_geo(self):
        expected = geometric_inv_freq(128, 500_000.0)
        max_error = eval_ruler.verify_loaded_inv_freq(expected.clone(), expected)
        self.assertEqual(max_error, 0.0)

    def test_ruler_frequency_verification_rejects_mismatch(self):
        expected = geometric_inv_freq(128, 500_000.0)
        with self.assertRaisesRegex(RuntimeError, "frequency mismatch"):
            eval_ruler.verify_loaded_inv_freq(expected + 1e-3, expected)

    def test_variant_is_unique_and_filename_safe(self):
        label = eval_ruler.resolve_variant_label(
            base_only=False,
            requested="geo_s42/stage2",
            adapter_dir="/tmp/checkpoints/ignored",
        )
        self.assertEqual(label, "geo_s42_stage2")
        self.assertEqual(eval_ruler.ruler_output_name(label), "ruler_geo_s42_stage2.json")

    def test_april_wrapper_passes_variant_label(self):
        wrapper = (ROOT / "scripts/2026-04/04_lora_eval_ruler.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('--variant "${LABEL}"', wrapper)


class ContinuedPretrainGeometryTests(unittest.TestCase):
    def test_geometry_is_inferred_for_llama_8b(self):
        config = SimpleNamespace(
            hidden_size=4096,
            num_attention_heads=32,
            rope_theta=500_000.0,
            max_position_embeddings=8192,
        )
        geometry = llama3_continued_pretrain.resolve_rope_geometry(config)

        self.assertEqual(geometry.head_dim, 128)
        self.assertEqual(geometry.n_freqs, 64)
        self.assertEqual(geometry.rope_theta, 500_000.0)
        self.assertEqual(geometry.model_max_position_embeddings, 8192)

    def test_geometry_is_inferred_for_llama_1b(self):
        config = SimpleNamespace(
            hidden_size=2048,
            num_attention_heads=32,
            rope_theta=500_000.0,
            max_position_embeddings=131072,
        )
        geometry = llama3_continued_pretrain.resolve_rope_geometry(config)

        self.assertEqual(geometry.head_dim, 64)
        self.assertEqual(geometry.n_freqs, 32)

    def test_patch_rejects_frequency_shape_mismatch(self):
        class Rope(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("inv_freq", torch.ones(64))

        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.rotary_emb = Rope()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Inner()

        with self.assertRaisesRegex(RuntimeError, "frequency count mismatch"):
            llama3_continued_pretrain.patch_inv_freq(Model(), torch.ones(32))

    def test_packed_eval_tensor_must_match_requested_length(self):
        with self.assertRaisesRegex(ValueError, "test L=16384"):
            llama3_continued_pretrain.validate_packed_tensor(
                torch.ones((2, 8192), dtype=torch.long),
                expected_seq_len=16384,
                label="test L=16384",
            )

    def test_eval_scaling_cannot_silently_patch_zero_modules(self):
        with self.assertRaisesRegex(RuntimeError, "No rotary"):
            llama3_continued_pretrain.require_patched_rotary_modules(
                0,
                "PPL YaRN scaling",
            )


class ExistingCheckpointProtocolTests(unittest.TestCase):
    def test_comparison_launcher_validates_existing_checkpoints_before_skip(self):
        launcher = (ROOT / "experiments/lora_evq_v2/run_pe_comparison.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("validate_checkpoint_artifact.py", launcher)

    def test_checkpoint_validator_rejects_stale_method_label(self):
        with tempfile.TemporaryDirectory() as td:
            checkpoint = Path(td)
            (checkpoint / "adapter_model.safetensors").touch()
            (checkpoint / "experiment_meta.json").write_text(
                json.dumps({"rope_method": "evq_cosh"}),
                encoding="utf-8",
            )
            torch.save(
                {
                    "inv_freq": torch.ones(4),
                    "method": "evq_cosh",
                    "head_dim": 8,
                    "base": 500_000.0,
                    "tau": 1.414,
                    "midpoint": True,
                },
                checkpoint / "custom_inv_freq.pt",
            )

            with self.assertRaisesRegex(RuntimeError, "method mismatch"):
                validate_checkpoint_artifact.validate_checkpoint_artifact(
                    checkpoint,
                    expected_method="native_geo",
                )

            result = validate_checkpoint_artifact.validate_checkpoint_artifact(
                checkpoint,
                expected_method="evq_cosh",
            )
            self.assertEqual(result["checkpoint"], checkpoint.name)
            self.assertNotIn(td, json.dumps(result))


if __name__ == "__main__":
    unittest.main()
