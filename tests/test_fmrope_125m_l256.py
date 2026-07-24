#!/usr/bin/env python3
"""Regression tests for the matched FMRoPE / EVQ rebuttal experiment."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256.prepare import (
    choose_eval_anchors,
    download_verified_file,
    sha256_token_prefix,
    tokenize_batches_to_npy,
    token_prefix_bounds,
)
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256.protocol import (
    ARMS,
    ARM_CONDITIONS,
    SPEC,
    estimate_parameter_count,
    learning_rate_for_step,
    runtime_frequency,
    training_inv_freq,
)
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256.run_experiment import (
    CausalLanguageModelLoss,
    FlatPrefixDataset,
    _load_checkpoint,
    _model_inv_freq,
    _save_checkpoint,
    build_eval_windows,
    build_model,
    code_fingerprint,
    deterministic_row_order,
    meta_parameter_count,
    set_runtime_rope,
    summarize_records,
    trainable_state_sha256,
)
from rebuttal.rebuttal_0723.experiments.geo_rope_contract import (
    EVQ_COSH,
    HISTORICAL_PAPER_GEO_SHA256_FLOAT32,
    PAPER_GEO,
    build_training_inv_freq,
    paper_to_std_ratios,
    std_geo_inv_freq,
)


class TestProtocol(unittest.TestCase):
    def test_registered_budget_and_exact_parameter_count(self):
        self.assertEqual(estimate_parameter_count(), 151_898_880)
        self.assertEqual(SPEC.optimizer_steps, 1_525)
        self.assertEqual(SPEC.train_rows, 390_400)
        self.assertEqual(SPEC.train_tokens, 99_942_400)
        self.assertEqual(SPEC.prediction_tokens, 99_552_000)
        self.assertEqual(SPEC.grad_accum_steps, 4)
        self.assertEqual(SPEC.micro_steps, 6_100)
        self.assertEqual(SPEC.eval_lengths, (256, 512, 1_024, 2_048))

    def test_all_arms_have_the_exact_model_size_on_meta(self):
        for arm in ARMS:
            self.assertEqual(meta_parameter_count(arm), 151_898_880)

    def test_training_schedules_match_registered_definitions(self):
        self.assertTrue(
            torch.equal(
                training_inv_freq("paper_geo_base500k"),
                build_training_inv_freq(
                    PAPER_GEO,
                    head_dim=64,
                    base=500_000.0,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                training_inv_freq("fmrope_base256"),
                std_geo_inv_freq(64, 256.0),
            )
        )
        self.assertTrue(
            torch.equal(
                training_inv_freq(
                    "evq_cosh_tau4_paper_grid_base500k"
                ),
                build_training_inv_freq(
                    EVQ_COSH,
                    head_dim=64,
                    base=500_000.0,
                    tau=4.0,
                ),
            )
        )
        self.assertEqual(
            hashlib.sha256(
                training_inv_freq("paper_geo_base500k").numpy().tobytes()
            ).hexdigest(),
            HISTORICAL_PAPER_GEO_SHA256_FLOAT32,
        )

    def test_fmrope_target_base_is_exactly_target_length(self):
        train_inv = training_inv_freq("fmrope_base256")
        for length in SPEC.eval_lengths:
            fixed, fixed_scale, fixed_meta = runtime_frequency(
                "fmrope_base256", "fixed_train_base", length
            )
            target, target_scale, target_meta = runtime_frequency(
                "fmrope_base256", "target_matched_base", length
            )
            self.assertTrue(torch.equal(fixed, train_inv))
            self.assertTrue(
                torch.equal(target, std_geo_inv_freq(64, float(length)))
            )
            self.assertEqual(fixed_scale, 1.0)
            self.assertEqual(target_scale, 1.0)
            self.assertEqual(fixed_meta["inference_base"], 256.0)
            self.assertEqual(target_meta["inference_base"], float(length))
        at_train, _, _ = runtime_frequency(
            "fmrope_base256", "target_matched_base", 256
        )
        self.assertTrue(torch.equal(at_train, train_inv))

    def test_larger_base_slows_all_but_first_frequency(self):
        base_256 = std_geo_inv_freq(64, 256.0)
        base_2048 = std_geo_inv_freq(64, 2_048.0)
        self.assertEqual(float(base_256[0]), 1.0)
        self.assertEqual(float(base_2048[0]), 1.0)
        self.assertTrue(torch.all(base_2048[1:] < base_256[1:]))

    def test_yarn_cell_is_identity_in_domain_and_official_outside(self):
        raw, _, _ = runtime_frequency(
            "paper_geo_base500k", "raw", 256
        )
        in_domain, in_scale, in_meta = runtime_frequency(
            "paper_geo_base500k", "yarn_derived_inference_only", 256
        )
        extrap, extrap_scale, extrap_meta = runtime_frequency(
            "paper_geo_base500k",
            "yarn_derived_inference_only",
            2_048,
        )
        self.assertTrue(torch.equal(raw, in_domain))
        self.assertEqual(in_scale, 1.0)
        self.assertEqual(in_meta["mode"], "identity")
        self.assertFalse(torch.equal(raw, extrap))
        self.assertGreater(extrap_scale, 1.0)
        self.assertEqual(extrap_meta["mode"], "yarn_derived_virtual_dim")

    def test_unregistered_cross_product_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not registered"):
            runtime_frequency(
                "evq_cosh_tau4_paper_grid_base500k",
                "target_matched_base",
                512,
            )
        self.assertEqual(
            set(ARM_CONDITIONS),
            set(ARMS),
        )

    def test_learning_rate_warmup_and_floor(self):
        self.assertAlmostEqual(
            learning_rate_for_step(0),
            SPEC.learning_rate / SPEC.warmup_steps,
        )
        self.assertAlmostEqual(
            learning_rate_for_step(SPEC.warmup_steps - 1),
            SPEC.learning_rate,
        )
        self.assertAlmostEqual(
            learning_rate_for_step(SPEC.optimizer_steps - 1),
            SPEC.min_learning_rate,
        )

    def test_paper_geo_ratio_and_checked_in_manifest(self):
        ratio = paper_to_std_ratios(64, 500_000.0)
        self.assertAlmostEqual(
            ratio["paper_over_std_frequency"],
            0.8146172338565447,
        )
        self.assertAlmostEqual(
            ratio["paper_over_std_wavelength"],
            1.2275703955658044,
        )
        manifest = json.loads(
            (
                Path(__file__).parents[1]
                / "rebuttal/rebuttal_0723/theory_results/FREQUENCY_DEFINITION_MANIFEST.json"
            ).read_text()
        )
        schedules = {
            "Std-Geo": std_geo_inv_freq(64, 500_000.0),
            "Paper-Geo": training_inv_freq("paper_geo_base500k"),
            "EVQ-Cosh": training_inv_freq(
                "evq_cosh_tau4_paper_grid_base500k"
            ),
        }
        for name, inv in schedules.items():
            record = manifest["methods"][name]
            if name == "EVQ-Cosh":
                record = record["receipts"]["fmrope_l256_tau4"]
            self.assertEqual(
                record["sha256"],
                hashlib.sha256(inv.numpy().tobytes()).hexdigest(),
            )
            self.assertEqual(record["first_channels"], inv[:4].tolist())
            self.assertEqual(record["last_channels"], inv[-4:].tolist())
        self.assertEqual(
            manifest["methods"]["Paper-Geo"]["sha256"],
            HISTORICAL_PAPER_GEO_SHA256_FLOAT32,
        )
        for receipt_name, tau in (
            ("submitted_primary_ii_tau5", 5.0),
            ("shape_l128_rule_tau5p656854", 64.0 / math.sqrt(128.0)),
        ):
            inv = build_training_inv_freq(
                EVQ_COSH,
                head_dim=64,
                base=500_000.0,
                tau=tau,
            )
            record = manifest["methods"]["EVQ-Cosh"]["receipts"][
                receipt_name
            ]
            self.assertEqual(
                record["sha256"],
                hashlib.sha256(inv.numpy().tobytes()).hexdigest(),
            )
            self.assertEqual(record["first_channels"], inv[:4].tolist())
            self.assertEqual(record["last_channels"], inv[-4:].tolist())


class TestDataAndInitialization(unittest.TestCase):
    def test_checkpoint_roundtrip_keeps_saved_inv_freq_authoritative(self):
        tiny = replace(
            SPEC,
            vocab_size=128,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            head_dim=8,
            intermediate_size=64,
            train_length=64,
            requested_train_tokens=1_024,
            global_batch_size=4,
            micro_batch_size=4,
            eval_lengths=(64, 128),
        )
        arm = "paper_geo_base500k"
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            run_dir = work / "runs" / arm
            run_dir.mkdir(parents=True)
            model = build_model(arm, spec=tiny, seed=42)
            expected = _model_inv_freq(model)
            metadata = {
                "arm": arm,
                "protocol_sha256": tiny.fingerprint(),
                "code_sha256": code_fingerprint(),
                "training_inv_freq_sha256": hashlib.sha256(
                    expected.numpy().tobytes()
                ).hexdigest(),
            }
            checkpoint = run_dir / "model.pt"
            metadata["checkpoint_sha256"] = _save_checkpoint(
                checkpoint, model, metadata
            )
            (run_dir / "train_meta.json").write_text(
                json.dumps(metadata)
            )
            np.save(run_dir / "inv_freq.npy", expected.numpy())

            def wrong_constructor(
                selected_arm: str,
                *,
                spec=SPEC,
                seed: int = 42,
            ):
                del selected_arm, seed
                from experiments.native_rope_evq_150m.model import GPT

                return GPT(
                    spec.model_config(),
                    std_geo_inv_freq(spec.head_dim, spec.geo_base),
                )

            with mock.patch(
                "rebuttal.rebuttal_0723.experiments.fmrope_125m_l256."
                "run_experiment.build_model",
                side_effect=wrong_constructor,
            ):
                loaded, _ = _load_checkpoint(work, arm, spec=tiny)
            self.assertTrue(torch.equal(_model_inv_freq(loaded), expected))

    def test_preexisting_download_is_accepted_only_by_size_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.bin"
            payload = b"pinned-source-bytes"
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            receipt = download_verified_file(
                destination=path,
                url="https://example.invalid/source.bin",
                expected_sha256=digest,
                expected_size=len(payload),
            )
            self.assertEqual(receipt["sha256"], digest)
            self.assertIn("preexisting cache", receipt["download_transport"])
            with self.assertRaisesRegex(ValueError, "wrong SHA-256"):
                download_verified_file(
                    destination=path,
                    url="https://example.invalid/source.bin",
                    expected_sha256="0" * 64,
                    expected_size=len(payload),
                )

    def test_streaming_tokenizer_writes_exact_prefix_and_shape(self):
        class FakeTokenizer:
            def __call__(self, texts, **kwargs):
                self.kwargs = kwargs
                return {
                    "input_ids": [
                        [ord(character) % 31 for character in text]
                        for text in texts
                    ]
                }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.npy"
            tokenizer = FakeTokenizer()
            metadata = tokenize_batches_to_npy(
                text_batches=[["abc", "de"], ["fghi"]],
                tokenizer=tokenizer,
                output_path=path,
                token_count=8,
                seq_len=4,
                progress_tokens=100,
            )
            saved = np.load(path, allow_pickle=False)
            expected = np.asarray(
                [ord(character) % 31 for character in "abcdefgh"],
                dtype=np.int64,
            ).reshape(2, 4)
            self.assertTrue(np.array_equal(saved, expected))
            self.assertEqual(metadata["documents_consumed"], 3)
            self.assertEqual(metadata["last_document_tokens_used"], 3)
            self.assertFalse(tokenizer.kwargs["add_special_tokens"])
            self.assertFalse(tokenizer.kwargs["truncation"])

    def test_prefix_dataset_rechunks_without_copying_a_new_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npy"
            source = np.arange(8 * 16, dtype=np.int64).reshape(8, 16)
            np.save(path, source)
            dataset = FlatPrefixDataset(path, rows=16, seq_len=8)
            self.assertEqual(len(dataset), 16)
            expected = torch.from_numpy(source.reshape(-1)[24:32].copy())
            self.assertTrue(torch.equal(dataset[3], expected))

    def test_prefix_hash_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npy"
            source = np.arange(100, dtype=np.int64).reshape(10, 10)
            np.save(path, source)
            expected = hashlib.sha256(
                np.ascontiguousarray(source.reshape(-1)[:37]).tobytes()
            ).hexdigest()
            self.assertEqual(sha256_token_prefix(path, 37), expected)
            self.assertEqual(token_prefix_bounds(path, 37), (0, 36))

    def test_eval_anchors_are_frozen_unique_and_valid(self):
        first = choose_eval_anchors(
            100_000, count=16, max_length=2_048, seed=123
        )
        second = choose_eval_anchors(
            100_000, count=16, max_length=2_048, seed=123
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(set(first.tolist())), 16)
        self.assertGreaterEqual(int(first.min()), 2_048)
        self.assertLess(int(first.max()), 100_000)
        self.assertTrue(np.all(np.diff(first) >= 2_048))

    def test_eval_windows_match_historical_l_token_convention(self):
        validation = np.arange(10_000, dtype=np.int64)
        anchors = np.asarray([4_000, 8_000], dtype=np.int64)
        short_inputs, short_targets = build_eval_windows(
            validation, anchors, length=256
        )
        long_inputs, long_targets = build_eval_windows(
            validation, anchors, length=2_048
        )
        self.assertEqual(short_inputs.shape, (2, 255))
        self.assertEqual(short_targets.shape, (2, 255))
        self.assertEqual(long_inputs.shape, (2, 2_047))
        self.assertEqual(long_targets.shape, (2, 2_047))
        self.assertTrue(
            np.array_equal(short_inputs[0], validation[3_744:3_999])
        )
        self.assertTrue(
            np.array_equal(short_targets[0], validation[3_745:4_000])
        )
        self.assertTrue(
            np.array_equal(
                short_targets[:, -SPEC.eval_tail_tokens :],
                long_targets[:, -SPEC.eval_tail_tokens :],
            )
        )

    def test_row_order_is_shared_and_deterministic(self):
        first = deterministic_row_order(100, seed=42)
        second = deterministic_row_order(100, seed=42)
        other = deterministic_row_order(100, seed=43)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        self.assertEqual(sorted(first.tolist()), list(range(100)))

    def test_trainable_initialization_is_identical_across_arms(self):
        tiny = replace(
            SPEC,
            vocab_size=128,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            head_dim=8,
            intermediate_size=64,
            train_length=64,
            requested_train_tokens=1_024,
            global_batch_size=4,
            micro_batch_size=4,
            fmrope_train_base=64.0,
            eval_lengths=(64, 128, 256, 512),
        )
        hashes = []
        for arm in ARMS:
            model = build_model(arm, spec=tiny, seed=42)
            hashes.append(trainable_state_sha256(model))
        self.assertEqual(len(set(hashes)), 1)

    def test_runtime_schedule_is_installed_in_shared_rope(self):
        tiny = replace(
            SPEC,
            vocab_size=128,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            head_dim=8,
            intermediate_size=64,
            train_length=64,
            requested_train_tokens=1_024,
            global_batch_size=4,
            micro_batch_size=4,
            fmrope_train_base=64.0,
            eval_lengths=(64, 128, 256, 512),
        )
        model = build_model("fmrope_base256", spec=tiny)
        new_inv = std_geo_inv_freq(8, 512.0)
        set_runtime_rope(model, new_inv, length=512, mscale=1.0)
        rope = model.blocks[0].attn.rope
        self.assertTrue(torch.equal(rope.inv_freq, new_inv))
        self.assertTrue(all(block.attn.rope is rope for block in model.blocks))
        self.assertEqual(rope._max, 512)
        token_ids = torch.randint(0, 128, (1, 96))
        logits = model(token_ids)
        self.assertEqual(tuple(logits.shape), (1, 96, 128))
        self.assertTrue(torch.isfinite(logits).all())

    def test_all_arms_receive_nonzero_schedule_dependent_gradients(self):
        tiny = replace(
            SPEC,
            vocab_size=128,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            head_dim=8,
            intermediate_size=64,
            train_length=64,
            requested_train_tokens=1_024,
            global_batch_size=4,
            micro_batch_size=4,
            fmrope_train_base=64.0,
            eval_lengths=(64, 128, 256, 512),
        )
        generator = torch.Generator().manual_seed(123)
        batch = torch.randint(0, 128, (2, 64), generator=generator)
        gradients = []
        for arm in ARMS:
            model = build_model(arm, spec=tiny, seed=42)
            loss = CausalLanguageModelLoss(model)(batch)
            loss.backward()
            gradient = (
                model.blocks[0]
                .attention.qkv.weight.grad.detach()
                .clone()
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.norm()), 0.0)
            gradients.append(gradient)
        for left in range(len(gradients)):
            for right in range(left + 1, len(gradients)):
                self.assertFalse(
                    torch.equal(gradients[left], gradients[right])
                )


class TestSummary(unittest.TestCase):
    @staticmethod
    def _complete_records():
        values = {
            ("paper_geo_base500k", "raw"): 2.0,
            (
                "paper_geo_base500k",
                "yarn_derived_inference_only",
            ): 1.9,
            ("fmrope_base256", "fixed_train_base"): 2.2,
            ("fmrope_base256", "target_matched_base"): 1.7,
            ("evq_cosh_tau4_paper_grid_base500k", "raw"): 1.5,
        }
        records = []
        for (arm, condition), base_nll in values.items():
            for length in SPEC.eval_lengths:
                length_nll = base_nll
                if (
                    length == SPEC.train_length
                    and arm == "paper_geo_base500k"
                    and condition == "yarn_derived_inference_only"
                ):
                    length_nll = values[("paper_geo_base500k", "raw")]
                if (
                    length == SPEC.train_length
                    and arm == "fmrope_base256"
                    and condition == "target_matched_base"
                ):
                    length_nll = values[
                        ("fmrope_base256", "fixed_train_base")
                    ]
                for anchor in (3_000, 4_000):
                    records.append(
                        {
                            "arm": arm,
                            "condition": condition,
                            "length": length,
                            "anchor": anchor,
                            "full_nll": length_nll + 0.1,
                            "tail_nll": length_nll,
                            "tail_target_sha256": "a" * 64,
                        }
                    )
        return records

    def test_paired_summary_has_expected_sign(self):
        records = self._complete_records()
        summary, markdown = summarize_records(records)
        delta = summary["paired"]["evq_raw_minus_fmrope_target"]["512"][
            "mean_left_minus_right_tail_nll"
        ]
        self.assertAlmostEqual(delta, -0.2)
        self.assertIn("Negative means", markdown)
        self.assertIn("Single seed (42)", markdown)
        self.assertEqual(
            summary["in_domain_identity_controls"],
            "PASS",
        )

    def test_summary_rejects_mismatched_paired_targets(self):
        records = self._complete_records()
        records[-1]["tail_target_sha256"] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "tail targets differ"):
            summarize_records(records)

    def test_summary_rejects_failed_identity_control(self):
        records = self._complete_records()
        for row in records:
            if (
                row["arm"] == "paper_geo_base500k"
                and row["condition"] == "yarn_derived_inference_only"
                and row["length"] == SPEC.train_length
            ):
                row["tail_nll"] += 0.01
                break
        with self.assertRaisesRegex(RuntimeError, "identity control failed"):
            summarize_records(records)


if __name__ == "__main__":
    unittest.main()
