import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from experiments.lora_evq_v2.legacy_lora_protocol import (
    LEGACY_METHODS,
    LEGACY_SEEDS,
    OFFICIAL_LONGALIGN_RAW_SHA256,
    canonical_training_protocol,
    legacy_eval_filename,
    legacy_run_name,
    paired_metric_summary,
    validate_complete_matrix,
    validate_legacy_protocol,
    validate_source_receipt,
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
    LoraRopeGeometry,
    resolve_legacy_resume_checkpoint,
    validate_legacy_model_geometry,
    validate_strict_legacy_args,
)
from experiments.lora_evq_v2.eval_legacy_lora_matched import variant_spec
from experiments.lora_evq_v2.summarize_legacy_lora_matched import summarize_records


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


class FrozenDataTests(unittest.TestCase):
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
            "runtime": {"python": "test"},
        }
        validate_legacy_metadata(metadata, expected_method="native_geo", expected_seed=42)
        metadata["global_step"] = 299
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
