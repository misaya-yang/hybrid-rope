import json
import inspect
import math
import re
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from experiments.lora_evq_v2 import legacy_lora_protocol
from experiments.lora_evq_v2.legacy_lora_protocol import (
    LEGACY_METHODS,
    LEGACY_RUNTIME_PACKAGES,
    LEGACY_SEEDS,
    OFFICIAL_LONGALIGN_RAW_SHA256,
    PAPER_LONGALPACA_PROVENANCE_STATUS,
    PAPER_LONGALPACA_RAW_SHA256,
    PAPER_LONGALPACA_REVISION,
    PAPER_LONGALPACA_SOURCE,
    canonical_training_protocol,
    legacy_eval_filename,
    legacy_run_name,
    paired_metric_summary,
    validate_complete_matrix,
    validate_legacy_protocol,
    validate_source_receipt,
    validate_training_source_receipt,
)
from experiments.lora_evq_v2.prepare_legacy_longalpaca_data import (
    convert_longalpaca_records_to_legacy_jsonl,
)
from experiments.lora_evq_v2.prepare_legacy_longalign_data import (
    compact_tokenized_row,
    split_legacy_tokenized,
    tokenize_legacy_rows,
)
from experiments.lora_evq_v2.validate_legacy_lora_artifact import (
    validate_legacy_metadata,
)
from experiments.lora_evq_v2.train_evq_lora import (
    PACKED_FREE_CAUSAL_SDPA_BACKEND,
    LoraRopeGeometry,
    PaddingCollator,
    configure_packed_free_causal_sdpa,
    packed_free_causal_sdpa_forward,
    resolve_legacy_resume_checkpoint,
    training_dataset_identifier,
    validate_legacy_model_geometry,
    validate_strict_legacy_args,
)
from experiments.lora_evq_v2.eval_legacy_lora_matched import variant_spec
from experiments.lora_evq_v2.summarize_legacy_lora_matched import summarize_records
from experiments.lora_evq_v2.summarize_legacy_geo_control import (
    summarize_geo_control_records,
)


class FakeTokenizer:
    name_or_path = "fake-llama-tokenizer"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.last_messages = messages
        return " ".join(str(item.get("content", "")) for item in messages)

    def __call__(self, text, **kwargs):
        ids = [len(piece) + 1 for piece in text.split()]
        max_length = int(kwargs["max_length"])
        ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class LegacyProtocolTests(unittest.TestCase):
    def test_source_receipt_requires_pinned_identity_and_hash(self):
        receipt = {
            "source_id": "zai-org/LongAlign-10k",
            "revision": "12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc",
            "split": "train",
            "filename": "long.jsonl",
            "raw_sha256": OFFICIAL_LONGALIGN_RAW_SHA256,
        }
        self.assertEqual(validate_source_receipt(receipt), receipt)
        for key in ("revision", "raw_sha256"):
            invalid = dict(receipt)
            invalid[key] = "unknown"
            with self.assertRaises(ValueError):
                validate_source_receipt(invalid)

    def test_paper_longalpaca_receipt_is_hash_pinned_and_explicitly_best_effort(self):
        receipt = {
            "source_id": PAPER_LONGALPACA_SOURCE,
            "revision": PAPER_LONGALPACA_REVISION,
            "split": "train",
            "filename": "LongAlpaca-12k_raw.json",
            "raw_sha256": PAPER_LONGALPACA_RAW_SHA256,
            "provenance_status": PAPER_LONGALPACA_PROVENANCE_STATUS,
        }
        self.assertEqual(validate_training_source_receipt(receipt), receipt)
        for key in ("raw_sha256", "provenance_status"):
            invalid = dict(receipt)
            invalid[key] = "unknown"
            with self.assertRaises(ValueError):
                validate_training_source_receipt(invalid)

    def test_paper_specific_receipt_rejects_official_longalign(self):
        self.assertTrue(
            hasattr(legacy_lora_protocol, "validate_paper_longalpaca_receipt"),
            "paper launcher needs a LongAlpaca-only receipt validator",
        )
        official_longalign = {
            "source_id": "zai-org/LongAlign-10k",
            "revision": "12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc",
            "split": "train",
            "filename": "long.jsonl",
            "raw_sha256": OFFICIAL_LONGALIGN_RAW_SHA256,
        }
        with self.assertRaises(ValueError):
            legacy_lora_protocol.validate_paper_longalpaca_receipt(official_longalign)

    def test_locked_runtime_rejects_package_drift(self):
        self.assertTrue(hasattr(legacy_lora_protocol, "validate_legacy_runtime_packages"))
        locked = {
            "torch": "2.8.0+cu128",
            "transformers": "4.57.6",
            "peft": "0.17.1",
            "accelerate": "1.10.1",
            "datasets": "4.5.0",
            "triton": "3.4.0",
        }
        self.assertEqual(
            legacy_lora_protocol.validate_legacy_runtime_packages(locked),
            locked,
        )
        drifted = dict(locked, transformers="5.13.0")
        with self.assertRaises(ValueError):
            legacy_lora_protocol.validate_legacy_runtime_packages(drifted)

    def test_longalpaca_eval_names_do_not_alias_longalign_results(self):
        self.assertEqual(
            legacy_eval_filename("geo_longalpaca_s42"),
            "eval_geo_longalpaca_s42.json",
        )
        self.assertEqual(
            variant_spec("geo_longalpaca_s42"),
            {"method": "native_geo", "seed": 42, "requires_adapter": True},
        )

    def test_canonical_protocol_is_exact_and_rejects_drift(self):
        protocol = canonical_training_protocol(
            method="evq_cosh",
            seed=43,
            data_manifest_sha256="b" * 64,
            model_manifest_sha256="c" * 64,
            code_sha256="f" * 64,
        )
        self.assertEqual(validate_legacy_protocol(protocol), protocol)
        self.assertEqual(protocol["lora_targets"], ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.assertEqual(protocol["max_steps"], 300)
        self.assertEqual(protocol["micro_batch_size"], 2)
        self.assertEqual(protocol["gradient_accumulation_steps"], 4)
        self.assertEqual(protocol["effective_batch_size"], 8)
        self.assertEqual(protocol["split_seed"], 42)

        for key, bad_value in (
            ("max_steps", 299),
            ("lora_targets", ["q_proj", "k_proj"]),
            ("lora_dropout", 0.0),
            ("learning_rate", 2e-5),
        ):
            invalid = dict(protocol)
            invalid[key] = bad_value
            with self.assertRaises(ValueError):
                validate_legacy_protocol(invalid)

    def test_fresh_six_arm_matrix_and_unique_eval_names(self):
        records = []
        for method in LEGACY_METHODS:
            for seed in LEGACY_SEEDS:
                records.append({
                    "method": method,
                    "seed": seed,
                    "data_manifest_sha256": "d" * 64,
                    "model_manifest_sha256": "e" * 64,
                    "code_sha256": "f" * 64,
                    "status": "complete",
                })
                self.assertIn(str(seed), legacy_run_name(method, seed))
        validated = validate_complete_matrix(records)
        self.assertEqual(len(validated), 6)
        with self.assertRaises(ValueError):
            validate_complete_matrix(records[:-1])

        variants = ["base_geo", "base_evq_tau1414"] + [
            legacy_run_name(method, seed)
            for method in LEGACY_METHODS
            for seed in LEGACY_SEEDS
        ]
        filenames = [legacy_eval_filename(name) for name in variants]
        self.assertEqual(len(filenames), len(set(filenames)))

    def test_paired_summary_uses_sample_standard_deviation(self):
        summary = paired_metric_summary(
            geo_by_seed={42: 10.0, 43: 12.0, 44: 14.0},
            evq_by_seed={42: 8.0, 43: 9.0, 44: 10.0},
        )
        self.assertEqual(summary["paired_delta_evq_minus_geo"]["values"], [-2.0, -3.0, -4.0])
        self.assertTrue(math.isclose(summary["geo"]["sample_std"], 2.0))
        self.assertTrue(math.isclose(summary["paired_delta_evq_minus_geo"]["sample_std"], 1.0))
        with self.assertRaises(ValueError):
            paired_metric_summary({42: 1.0}, {43: 1.0})

    def test_right_padded_collator_omits_redundant_attention_mask(self):
        collator = PaddingCollator(pad_token_id=128001, omit_attention_mask=True)
        batch = collator([
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [4, 5]},
        ])

        self.assertEqual(set(batch), {"input_ids", "labels"})
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3], [4, 5, 128001]])
        self.assertEqual(batch["labels"].tolist(), [[1, 2, 3], [4, 5, -100]])

    def test_mask_free_collator_rejects_noncontiguous_source_masks(self):
        collator = PaddingCollator(pad_token_id=0, omit_attention_mask=True)
        for source_mask in ([1, 0, 1], [1, 1, 0]):
            with self.subTest(source_mask=source_mask):
                with self.assertRaisesRegex(ValueError, "right-padded full-token"):
                    collator([
                        {
                            "input_ids": [1, 2, 3],
                            "attention_mask": source_mask,
                            "labels": [1, 2, 3],
                        }
                    ])

    def test_custom_attention_backend_skips_transformers_mask_materialization(self):
        from transformers import AttentionInterface
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

        model = type("Model", (), {"config": type("Config", (), {})()})()
        backend = configure_packed_free_causal_sdpa(model)

        self.assertEqual(backend, PACKED_FREE_CAUSAL_SDPA_BACKEND)
        self.assertEqual(model.config._attn_implementation, backend)
        self.assertIn(backend, AttentionInterface._global_mapping)
        self.assertNotIn(backend, ALL_MASK_ATTENTION_FUNCTIONS._global_mapping)

    def test_packed_free_backend_preserves_kv_heads_and_enables_gqa(self):
        from unittest.mock import patch

        module = type("Attention", (), {"num_key_value_groups": 4})()
        query = torch.randn(2, 32, 5, 8)
        key = torch.randn(2, 8, 5, 8)
        value = torch.randn(2, 8, 5, 8)
        captured = {}

        def fake_sdpa(actual_query, actual_key, actual_value, **kwargs):
            captured.update({
                "query_shape": tuple(actual_query.shape),
                "key_shape": tuple(actual_key.shape),
                "value_shape": tuple(actual_value.shape),
                **kwargs,
            })
            return torch.zeros_like(actual_query)

        with patch(
            "torch.nn.functional.scaled_dot_product_attention",
            side_effect=fake_sdpa,
        ):
            output, weights = packed_free_causal_sdpa_forward(
                module,
                query,
                key,
                value,
                attention_mask=None,
            )

        self.assertEqual(captured["query_shape"], (2, 32, 5, 8))
        self.assertEqual(captured["key_shape"], (2, 8, 5, 8))
        self.assertEqual(captured["value_shape"], (2, 8, 5, 8))
        self.assertIsNone(captured["attn_mask"])
        self.assertTrue(captured["is_causal"])
        self.assertTrue(captured["enable_gqa"])
        self.assertEqual(output.shape, (2, 5, 32, 8))
        self.assertIsNone(weights)

    def test_right_padding_mask_is_redundant_for_valid_causal_queries(self):
        torch.manual_seed(7)
        query = torch.randn(2, 4, 5, 8)
        key = torch.randn(2, 2, 5, 8)
        value = torch.randn(2, 2, 5, 8)
        lengths = (3, 5)
        causal = torch.tril(torch.ones(5, 5, dtype=torch.bool))
        mask = torch.zeros(2, 1, 5, 5, dtype=torch.bool)
        for row, length in enumerate(lengths):
            mask[row, 0] = causal & (torch.arange(5)[None, :] < length)
        repeated_key = key.repeat_interleave(2, dim=1)
        repeated_value = value.repeat_interleave(2, dim=1)
        masked = torch.nn.functional.scaled_dot_product_attention(
            query, repeated_key, repeated_value, attn_mask=mask
        )
        mask_free = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True, enable_gqa=True
        )

        for row, length in enumerate(lengths):
            self.assertTrue(
                torch.allclose(
                    masked[row, :, :length],
                    mask_free[row, :, :length],
                    rtol=1e-5,
                    atol=1e-6,
                )
            )

    def test_remaining_seed_launcher_trains_only_evq_seeds_43_and_44(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts/2026-07/07_lora_longalpaca_evq_remaining_seeds.sh"
        ).read_text(encoding="utf-8")
        lower = launcher.lower()
        self.assertNotIn("longalign", lower)
        self.assertNotIn("--seed 42", launcher)
        self.assertNotIn("--rope_method native_geo", launcher)
        self.assertNotIn("for method in", launcher)
        for required in (
            "geo_longalpaca_s42",
            "evq_longalpaca_tau1414_s${seed}",
            "--rope_method evq_cosh",
            "for seed in 43 44",
            'case "$seed" in',
            "43|44",
            "TORCHINDUCTOR_CACHE_DIR",
            "--packed_free_causal_sdpa",
        ):
            self.assertIn(required, launcher)


class FrozenDataTests(unittest.TestCase):
    def test_longalpaca_conversion_reproduces_legacy_messages_jsonl(self):
        rows = [
            {"instruction": "Question", "input": "Context", "output": "Answer"},
            {"messages": [{"role": "user", "content": "Already normalized"}]},
            {"unsupported": True},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "legacy.jsonl"
            stats = convert_longalpaca_records_to_legacy_jsonl(rows, output)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(stats, {"source_rows_seen": 3, "converted_rows": 2, "unsupported_rows": 1})
        self.assertEqual(
            records[0],
            {"messages": [
                {"role": "user", "content": "Question\n\nContext"},
                {"role": "assistant", "content": "Answer"},
            ]},
        )
        self.assertEqual(records[1]["messages"][0]["content"], "Already normalized")

    def test_strict_metadata_uses_manifest_source_not_legacy_default_name(self):
        args = Namespace(dataset_name="THUDM/LongAlign-10k", local_data_path=None)
        manifest = {"source": {"source_id": PAPER_LONGALPACA_SOURCE}}
        self.assertEqual(
            training_dataset_identifier(args, manifest),
            PAPER_LONGALPACA_SOURCE,
        )

    def test_first_n_accepted_rows_are_tokenized_and_split_deterministically(self):
        rows = [
            {"messages": [{"role": "user", "content": "one two three"}]},
            {"ignored": "not a supported row"},
            {"instruction": "four five", "output": "six"},
            {"messages": [{"role": "user", "content": "must not be selected"}]},
        ]
        tokenized, stats = tokenize_legacy_rows(
            rows,
            FakeTokenizer(),
            max_samples=2,
            max_seq_len=8,
            min_tokens=2,
        )
        self.assertEqual(tokenized["offsets"].numel() - 1, 2)
        self.assertEqual(stats["accepted_source_rows"], 2)
        self.assertEqual(stats["unsupported_source_rows"], 1)
        self.assertEqual(compact_tokenized_row(tokenized, 0), [4, 4, 6])

        first = split_legacy_tokenized(tokenized, val_ratio=0.5, split_seed=42)
        second = split_legacy_tokenized(tokenized, val_ratio=0.5, split_seed=42)
        self.assertTrue(torch.equal(first["train_indices"], second["train_indices"]))
        self.assertTrue(torch.equal(first["validation_indices"], second["validation_indices"]))
        self.assertEqual(first["train_indices"].numel(), 1)
        self.assertEqual(first["validation_indices"].numel(), 1)

    def test_strict_tensor_validation_rejects_corrupt_prepared_data(self):
        from experiments.lora_evq_v2 import train_evq_lora

        self.assertTrue(hasattr(train_evq_lora, "validate_prepared_training_data"))
        manifest = {
            "statistics": {
                "tokenized_rows": 3,
                "train_rows": 2,
                "validation_rows": 1,
                "minimum_length": 64,
                "maximum_length": 64,
            }
        }
        valid = {
            "tokens": torch.arange(192, dtype=torch.int32),
            "offsets": torch.tensor([0, 64, 128, 192], dtype=torch.int64),
            "train_indices": torch.tensor([0, 2], dtype=torch.int32),
            "validation_indices": torch.tensor([1], dtype=torch.int32),
        }
        train_evq_lora.validate_prepared_training_data(valid, manifest, vocab_size=256)
        corruptions = []
        bad_offsets = {key: value.clone() for key, value in valid.items()}
        bad_offsets["offsets"][2] = 1
        corruptions.append(bad_offsets)
        overlapping = {key: value.clone() for key, value in valid.items()}
        overlapping["validation_indices"][0] = 2
        corruptions.append(overlapping)
        out_of_vocab = {key: value.clone() for key, value in valid.items()}
        out_of_vocab["tokens"][0] = 256
        corruptions.append(out_of_vocab)
        for corrupted in corruptions:
            with self.subTest(corrupted=corrupted):
                with self.assertRaises(ValueError):
                    train_evq_lora.validate_prepared_training_data(
                        corrupted,
                        manifest,
                        vocab_size=256,
                    )


class LegacyArtifactTests(unittest.TestCase):
    def test_final_metadata_requires_step_300_and_exact_protocol(self):
        protocol = canonical_training_protocol(
            method="native_geo",
            seed=42,
            data_manifest_sha256="1" * 64,
            model_manifest_sha256="2" * 64,
            code_sha256="f" * 64,
        )
        metadata = {
            "objective": "legacy_longalign_full_token_causal_lm_v2",
            "status": "complete",
            "global_step": 300,
            "protocol": protocol,
            "protocol_sha256": "3" * 64,
            "adapter_sha256": "4" * 64,
            "frequency_sha256": "5" * 64,
            "data_manifest_sha256": "1" * 64,
            "model_manifest_sha256": "2" * 64,
            "code_sha256": "f" * 64,
            "runtime": {"python": "test", "packages": dict(LEGACY_RUNTIME_PACKAGES)},
        }
        validate_legacy_metadata(metadata, expected_method="native_geo", expected_seed=42)
        metadata["global_step"] = 299
        with self.assertRaises(ValueError):
            validate_legacy_metadata(metadata, expected_method="native_geo", expected_seed=42)

    def test_artifact_validator_binds_expected_data_manifest(self):
        self.assertIn(
            "expected_data_manifest_sha256",
            inspect.signature(validate_legacy_metadata).parameters,
        )
        protocol = canonical_training_protocol(
            method="native_geo",
            seed=42,
            data_manifest_sha256="1" * 64,
            model_manifest_sha256="2" * 64,
            code_sha256="f" * 64,
        )
        metadata = {
            "objective": "legacy_longalign_full_token_causal_lm_v2",
            "status": "complete",
            "global_step": 300,
            "protocol": protocol,
            "protocol_sha256": "3" * 64,
            "adapter_sha256": "4" * 64,
            "frequency_sha256": "5" * 64,
            "data_manifest_sha256": "1" * 64,
            "model_manifest_sha256": "2" * 64,
            "code_sha256": "f" * 64,
            "runtime": {"python": "test", "packages": dict(LEGACY_RUNTIME_PACKAGES)},
        }
        with self.assertRaises(ValueError):
            validate_legacy_metadata(
                metadata,
                expected_method="native_geo",
                expected_seed=42,
                expected_data_manifest_sha256="9" * 64,
            )

    def test_artifact_validator_rejects_runtime_package_drift(self):
        protocol = canonical_training_protocol(
            method="native_geo",
            seed=42,
            data_manifest_sha256="1" * 64,
            model_manifest_sha256="2" * 64,
            code_sha256="f" * 64,
        )
        metadata = {
            "objective": "legacy_longalign_full_token_causal_lm_v2",
            "status": "complete",
            "global_step": 300,
            "protocol": protocol,
            "protocol_sha256": "3" * 64,
            "adapter_sha256": "4" * 64,
            "frequency_sha256": "5" * 64,
            "data_manifest_sha256": "1" * 64,
            "model_manifest_sha256": "2" * 64,
            "code_sha256": "f" * 64,
            "runtime": {"packages": dict(LEGACY_RUNTIME_PACKAGES)},
        }
        metadata["runtime"]["packages"]["transformers"] = "5.13.0"
        with self.assertRaises(ValueError):
            validate_legacy_metadata(metadata, expected_method="native_geo", expected_seed=42)

    def test_strict_args_reject_scientific_drift(self):
        args = Namespace(
            rope_method="evq_cosh",
            tau=1.414,
            seed=42,
            max_seq_len=8192,
            max_samples=8000,
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            lora_targets="q_proj,k_proj,v_proj,o_proj",
            max_steps=300,
            per_device_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            warmup_steps=60,
            weight_decay=0.01,
            max_grad_norm=1.0,
            save_steps=100,
            bf16=True,
            load_in_4bit=False,
            compile=True,
            compile_mode="default",
        )
        validate_strict_legacy_args(args)
        args.per_device_batch_size = 8
        with self.assertRaises(ValueError):
            validate_strict_legacy_args(args)

    def test_auto_resume_selects_latest_valid_recovery_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for step in (100, 200):
                checkpoint = output / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps({"global_step": step}), encoding="utf-8"
                )
                for name in (
                    "adapter_model.safetensors",
                    "adapter_config.json",
                    "optimizer.pt",
                    "scheduler.pt",
                    "rng_state.pth",
                    "training_args.bin",
                ):
                    (checkpoint / name).write_bytes(b"test")
            (output / "checkpoint-junk").mkdir()
            self.assertEqual(
                resolve_legacy_resume_checkpoint(output, "auto"),
                output / "checkpoint-200",
            )
            self.assertIsNone(resolve_legacy_resume_checkpoint(output, "none"))
            (output / "checkpoint-200" / "optimizer.pt").unlink()
            self.assertEqual(
                resolve_legacy_resume_checkpoint(output, "auto"),
                output / "checkpoint-100",
            )

    def test_strict_resume_rejects_checkpoint_protocol_drift(self):
        from experiments.lora_evq_v2 import train_evq_lora

        self.assertIn(
            "expected_protocol",
            inspect.signature(resolve_legacy_resume_checkpoint).parameters,
        )
        self.assertTrue(hasattr(train_evq_lora, "write_legacy_checkpoint_receipt"))
        protocol = canonical_training_protocol(
            method="native_geo",
            seed=42,
            data_manifest_sha256="1" * 64,
            model_manifest_sha256="2" * 64,
            code_sha256="f" * 64,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            checkpoint = output / "checkpoint-100"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"global_step": 100}), encoding="utf-8"
            )
            for name in (
                "adapter_model.safetensors",
                "adapter_config.json",
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "training_args.bin",
            ):
                (checkpoint / name).write_bytes(b"test")
            train_evq_lora.write_legacy_checkpoint_receipt(
                checkpoint,
                protocol,
                LEGACY_RUNTIME_PACKAGES,
            )
            self.assertEqual(
                resolve_legacy_resume_checkpoint(
                    output,
                    "auto",
                    expected_protocol=protocol,
                    expected_runtime_packages=LEGACY_RUNTIME_PACKAGES,
                ),
                checkpoint,
            )
            (checkpoint / "optimizer.pt").write_bytes(b"tampered")
            with self.assertRaises(RuntimeError):
                resolve_legacy_resume_checkpoint(
                    output,
                    "auto",
                    expected_protocol=protocol,
                    expected_runtime_packages=LEGACY_RUNTIME_PACKAGES,
                )
            (checkpoint / "optimizer.pt").write_bytes(b"test")
            train_evq_lora.write_legacy_checkpoint_receipt(
                checkpoint,
                protocol,
                LEGACY_RUNTIME_PACKAGES,
            )
            receipt = checkpoint / "checkpoint_receipt.json"
            record = json.loads(receipt.read_text(encoding="utf-8"))
            record["code_sha256"] = "0" * 64
            receipt.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                resolve_legacy_resume_checkpoint(
                    output,
                    "auto",
                    expected_protocol=protocol,
                    expected_runtime_packages=LEGACY_RUNTIME_PACKAGES,
                )

    def test_strict_geometry_rejects_model_drift(self):
        config = Namespace(
            model_type="llama",
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=128256,
            max_position_embeddings=8192,
            rope_scaling=None,
        )
        validate_legacy_model_geometry(config, LoraRopeGeometry(128, 500000.0))
        config.num_key_value_heads = 4
        with self.assertRaises(ValueError):
            validate_legacy_model_geometry(config, LoraRopeGeometry(128, 500000.0))

    def test_strict_training_uses_explicit_single_gpu_map(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "experiments/lora_evq_v2/train_evq_lora.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"device_map": {"": 0} if args.strict_legacy_protocol else "auto"', source)
        self.assertIn("CPU or disk offload", source)


class LegacyEvaluationTests(unittest.TestCase):
    def test_variant_dispatch_distinguishes_static_baselines_and_adapters(self):
        self.assertEqual(
            variant_spec("base_geo"),
            {"method": "native_geo", "seed": None, "requires_adapter": False},
        )
        self.assertEqual(
            variant_spec("base_evq_tau1414"),
            {"method": "evq_cosh", "seed": None, "requires_adapter": False},
        )
        self.assertEqual(
            variant_spec("geo_longalign_s43"),
            {"method": "native_geo", "seed": 43, "requires_adapter": True},
        )
        with self.assertRaises(ValueError):
            variant_spec("historical_evq_seed42")

    def test_longalpaca_evaluation_binds_adapter_to_longalpaca_manifest(self):
        from experiments.lora_evq_v2 import eval_legacy_lora_matched

        self.assertTrue(
            hasattr(eval_legacy_lora_matched, "validate_variant_training_manifest")
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps({"source": {
                "source_id": "zai-org/LongAlign-10k",
                "revision": "12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc",
                "split": "train",
                "filename": "long.jsonl",
                "raw_sha256": OFFICIAL_LONGALIGN_RAW_SHA256,
            }}), encoding="utf-8")
            with self.assertRaises(ValueError):
                eval_legacy_lora_matched.validate_variant_training_manifest(
                    "geo_longalpaca_s42",
                    path,
                    {"data_manifest_sha256": "0" * 64},
                )

    def test_longalpaca_result_validation_binds_evaluator_code(self):
        from experiments.lora_evq_v2 import eval_legacy_lora_matched

        self.assertTrue(
            hasattr(eval_legacy_lora_matched, "legacy_evaluation_code_sha256")
        )
        record = self._eval_record("base_geo_longalpaca", None, None, 10.0)
        record["method"] = "native_geo"
        record["frequency_provenance"]["method"] = "native_geo"
        record["evaluation_code_sha256"] = "0" * 64
        record["evaluation_runtime_packages"] = dict(LEGACY_RUNTIME_PACKAGES)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "eval_base_geo_longalpaca.json"
            output.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(ValueError):
                eval_legacy_lora_matched._validate_existing_result(
                    output,
                    variant="base_geo_longalpaca",
                    spec={"method": "native_geo", "seed": None},
                    model_manifest_sha256="7" * 64,
                    eval_manifest_sha256="6" * 64,
                    adapter_meta=None,
                )

    def test_longalpaca_result_validation_binds_evaluator_runtime(self):
        from experiments.lora_evq_v2 import eval_legacy_lora_matched

        record = self._eval_record("base_geo_longalpaca", None, None, 10.0)
        record["method"] = "native_geo"
        record["frequency_provenance"]["method"] = "native_geo"
        record["evaluation_code_sha256"] = (
            eval_legacy_lora_matched.legacy_evaluation_code_sha256()
        )
        record["evaluation_runtime_packages"] = dict(
            LEGACY_RUNTIME_PACKAGES,
            transformers="5.13.0",
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "eval_base_geo_longalpaca.json"
            output.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(ValueError):
                eval_legacy_lora_matched._validate_existing_result(
                    output,
                    variant="base_geo_longalpaca",
                    spec={"method": "native_geo", "seed": None},
                    model_manifest_sha256="7" * 64,
                    eval_manifest_sha256="6" * 64,
                    adapter_meta=None,
                )

    def test_summary_requires_all_variants_and_shared_provenance(self):
        records = []
        for variant in ("base_geo", "base_evq_tau1414"):
            records.append(self._eval_record(variant, None, None, 10.0))
        for method in LEGACY_METHODS:
            for seed in LEGACY_SEEDS:
                value = 10.0 + seed / 100.0 + (0.0 if method == "native_geo" else -1.0)
                records.append(self._eval_record(legacy_run_name(method, seed), method, seed, value))
        summary = summarize_records(records)
        self.assertEqual(summary["seed_scope"], [42, 43, 44])
        self.assertIn("paired_delta_evq_minus_geo", summary["metrics"]["ppl@8K"])
        with self.assertRaises(ValueError):
            summarize_records(records[:-1])
        mismatched = [dict(record) for record in records]
        mismatched[-1]["eval_manifest_sha256"] = "9" * 64
        with self.assertRaises(ValueError):
            summarize_records(mismatched)
        non_finite = [json.loads(json.dumps(record)) for record in records]
        non_finite[-1]["ppl"]["32K"]["ppl"] = float("nan")
        with self.assertRaises(ValueError):
            summarize_records(non_finite)

    def test_geo_control_summary_isolates_finetuning_drift(self):
        base = self._eval_record("base_geo", None, None, 10.0)
        geo = self._eval_record("geo_longalign_s42", "native_geo", 42, 13.0)
        summary = summarize_geo_control_records([base, geo])
        self.assertEqual(summary["seed"], 42)
        self.assertEqual(summary["comparison"], "Geo+LoRA-s42 minus Base-Geo")
        self.assertTrue(math.isclose(summary["metrics"]["ppl@8K"]["drift_pct"], 30.0))
        self.assertEqual(summary["metrics"]["ppl@8K"]["base_geo"], 10.0)
        self.assertEqual(summary["metrics"]["ppl@8K"]["geo_lora"], 13.0)
        mismatched = json.loads(json.dumps(geo))
        mismatched["eval_manifest_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            summarize_geo_control_records([base, mismatched])

    def test_geo_control_launcher_cannot_start_evq_or_other_seeds(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts/2026-07/03_lora_longalign_matched_multiseed.sh"
        ).read_text(encoding="utf-8")
        match = re.search(r"run_geo_control\(\) \{(?P<body>.*?)\n\}", script, re.S)
        self.assertIsNotNone(match)
        body = match.group("body")
        self.assertIn("preflight", body)
        self.assertIn("eval_variant base_geo", body)
        self.assertIn("train_arm native_geo 42", body)
        self.assertIn("eval_variant geo_longalign_s42 native_geo 42", body)
        self.assertNotIn("evq_cosh", body)
        self.assertNotRegex(body, r"\b43\b|\b44\b")

    def test_longalpaca_shared_launcher_keeps_seed42_and_method_gates(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts/2026-07/04_lora_longalpaca_paper_geo_s42.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("train_arm", script)
        self.assertIn('ROPE_METHOD="${EVQ_PAPER_ROPE_METHOD:-native_geo}"', script)
        self.assertIn('--rope_method "$ROPE_METHOD"', script)
        self.assertIn("--seed 42", script)
        self.assertIn("geo_longalpaca_s42", script)
        self.assertIn("evq_longalpaca_tau1414_s42", script)
        self.assertIn("EVQ_PAPER_UNLOCK_GEO42", script)
        self.assertIn("EVQ_PAPER_UNLOCK_EVQ42", script)
        self.assertIn("checkpoint-300", script)
        self.assertIn("checkpoint-200", script)
        self.assertNotRegex(script, r"--seed (43|44)")

        wrapper = (
            Path(__file__).resolve().parents[1]
            / "scripts/2026-07/05_lora_longalpaca_paper_evq_s42.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("export EVQ_PAPER_ROPE_METHOD=evq_cosh", wrapper)
        self.assertIn('preflight|train', wrapper)
        self.assertNotIn("baseline|eval", wrapper)
        self.assertIn('exec "$SCRIPT_DIR/04_lora_longalpaca_paper_geo_s42.sh"', wrapper)

    def test_longalpaca_launcher_runs_exact_dry_run_before_gpu_allocation(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts/2026-07/04_lora_longalpaca_paper_geo_s42.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("validate_paper_longalpaca_receipt", script)
        self.assertIn("verify_hashes=True", script)
        self.assertIn("--dry_run", script)

    def test_longalpaca_launcher_prevents_duplicate_training_and_records_telemetry(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts/2026-07/04_lora_longalpaca_paper_geo_s42.sh"
        ).read_text(encoding="utf-8")
        train_arm = re.search(r"train_arm\(\) \{(?P<body>.*?)\n\}", script, re.S)
        self.assertIsNotNone(train_arm)
        body = train_arm.group("body")
        self.assertIn("flock", body)
        self.assertIn("--expected_data_manifest_sha256", body)
        self.assertIn("--loop-ms", body)
        self.assertRegex(body, r"validated final adapter|validated completed adapter")

    def test_longalpaca_launcher_fail_closes_compile_gpu_lease_and_telemetry(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts/2026-07/04_lora_longalpaca_paper_geo_s42.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("TORCHDYNAMO_DISABLE", script)
        self.assertIn("torch._dynamo.config.suppress_errors", script)
        self.assertGreaterEqual(script.count('"$GPU_LOCK_FILE"'), 2)
        self.assertNotIn('"$LOCK_DIR/gpu.lock"', script)
        self.assertIn("invocation_id", script)
        self.assertIn("telemetry monitor produced no samples", script)
        self.assertIn('"model_dir": str(model_dir.resolve())', script)
        self.assertIn("EVQ_GLOBAL_GPU_LOCK_FILE", script)
        self.assertIn("training_pid", script)
        self.assertIn("model_inventory", script)

    @staticmethod
    def _eval_record(variant, method, seed, ppl):
        return {
            "format_version": 1,
            "variant": variant,
            "method": method if method else ("native_geo" if variant == "base_geo" else "evq_cosh"),
            "seed": seed,
            "eval_manifest_sha256": "6" * 64,
            "model_manifest_sha256": "7" * 64,
            "adapter_sha256": "a" * 64 if seed is not None else None,
            "training_data_manifest_sha256": "8" * 64 if seed is not None else None,
            "training_code_sha256": "9" * 64 if seed is not None else None,
            "frequency_provenance": {
                "method": method if method else ("native_geo" if variant == "base_geo" else "evq_cosh")
            },
            "ppl": {
                key: {
                    "ppl": ppl + offset,
                    "nll": math.log(ppl + offset),
                    "chunks": 5,
                    "per_chunk_nll": [math.log(ppl + offset)] * 5,
                    "scored_tokens": 5 * (context - 1),
                }
                for key, offset, context in (
                    ("8K", 0.0, 8192),
                    ("16K", 1.0, 16384),
                    ("32K", 2.0, 32768),
                )
            },
        }


if __name__ == "__main__":
    unittest.main()
