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

from experiments.lora_evq_v2 import eval_positional_ppl, eval_ruler, train_evq_lora
from scripts.lib.rope.schedules import geometric_inv_freq
from scripts.text_eval import llama3_continued_pretrain


class LoraFrequencyProtocolTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
