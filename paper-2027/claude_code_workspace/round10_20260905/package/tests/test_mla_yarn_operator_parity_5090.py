#!/usr/bin/env python3
"""CPU-only gates for the MLA shared-operator parity preparation."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # CPU-only repository checks may omit torch.
    torch = None

from rebuttal.rebuttal_0723.experiments.mla_yarn_operator_parity_5090 import (
    prepare,
)
from rebuttal.rebuttal_0723.experiments.mla_yarn_operator_parity_5090.protocol import (
    BASE_TRAINING_PROTOCOL_SHA256,
    FREQUENCY_PAIRS,
    OPERATORS,
    PRIMARY_LENGTHS,
    SPEC,
    TRAINING_ARMS,
    operators_for_stage,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestOperatorParityProtocol(unittest.TestCase):
    def test_frozen_factorial(self):
        self.assertEqual(TRAINING_ARMS, ("native_geo", "evq_cosh"))
        self.assertEqual(FREQUENCY_PAIRS, (8, 32))
        self.assertEqual(PRIMARY_LENGTHS, (16_384, 32_768))
        self.assertEqual(
            OPERATORS,
            (
                "raw",
                "position_interpolation",
                "shared_index_freq_only",
                "mscale_only",
                "shared_index_full",
                "virtual_coordinate_full",
            ),
        )
        self.assertEqual(len(SPEC.fingerprint()), 64)
        self.assertEqual(len(BASE_TRAINING_PROTOCOL_SHA256), 64)
        self.assertEqual(
            operators_for_stage("200m"),
            ("raw", "shared_index_full"),
        )
        self.assertEqual(operators_for_stage("300m"), OPERATORS)

    def test_fingerprint_is_stable(self):
        self.assertEqual(SPEC.fingerprint(), SPEC.fingerprint())


class TestFreshAnchors(unittest.TestCase):
    def test_deterministic_and_disjoint_from_previous(self):
        previous = [100, 300, 500]
        kwargs = {
            "validation_tokens": 4_000,
            "previous_endpoints": previous,
            "count": 10,
            "max_length": 100,
            "seed": 42,
        }
        first = prepare.choose_fresh_disjoint_anchor_endpoints(**kwargs)
        second = prepare.choose_fresh_disjoint_anchor_endpoints(**kwargs)
        self.assertTrue(np.array_equal(first, second))
        windows = sorted(
            [
                *prepare._windows(previous, 100),
                *prepare._windows(first.tolist(), 100),
            ]
        )
        self.assertTrue(
            all(
                right[0] >= left[1]
                for left, right in zip(windows, windows[1:])
            )
        )

    def test_fails_when_not_enough_fresh_windows(self):
        with self.assertRaisesRegex(ValueError, "only 1 fresh"):
            prepare.choose_fresh_disjoint_anchor_endpoints(
                300,
                previous_endpoints=(100, 200),
                count=2,
                max_length=100,
                seed=42,
            )

    def test_build_manifest_uses_fresh_hardlinked_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "train.npy"
            validation_path = root / "validation.npy"
            old_selection_path = root / "old_selection.npy"
            old_test_path = root / "old_test.npy"
            np.save(train_path, np.arange(1_000, dtype=np.int64))
            np.save(validation_path, np.arange(10_000, dtype=np.int64))
            np.save(
                old_selection_path,
                np.asarray([100, 300], dtype=np.int64),
            )
            np.save(
                old_test_path,
                np.asarray([500, 700, 900, 1_100], dtype=np.int64),
            )
            source = {
                "schema_version": 1,
                "protocol_sha256": BASE_TRAINING_PROTOCOL_SHA256,
                "train": {
                    "path": str(train_path),
                    "tokens_available": 1_000,
                    "tokens_used": 600,
                    "rows_used": 6,
                    "sha256": _sha(train_path),
                    "token_prefix_sha256": prepare.sha256_token_prefix(
                        train_path, 600
                    ),
                    "token_min": 0,
                    "token_max": 999,
                },
                "validation": {
                    "path": str(validation_path),
                    "tokens": 10_000,
                    "sha256": _sha(validation_path),
                    "token_min": 0,
                    "token_max": 9_999,
                },
                "selection_anchors": {
                    "path": str(old_selection_path),
                    "count": 2,
                    "values": [100, 300],
                    "sha256": _sha(old_selection_path),
                },
                "test_anchors": {
                    "path": str(old_test_path),
                    "count": 4,
                    "values": [500, 700, 900, 1_100],
                    "sha256": _sha(old_test_path),
                },
            }
            source_path = root / "source_manifest.json"
            source_path.write_text(
                json.dumps(source, indent=2, sort_keys=True) + "\n"
            )
            fake_spec = SimpleNamespace(
                train_tokens=600,
                selection_anchor_count=2,
                test_anchor_count=4,
                eval_lengths=(25, 50, 75, 100),
                eval_batch_size_by_length={
                    25: 4,
                    50: 2,
                    75: 1,
                    100: 1,
                },
                anchor_seed=99,
                fingerprint=lambda: "f" * 64,
            )
            output = root / "fresh"
            with mock.patch.object(prepare, "SPEC", fake_spec):
                manifest = prepare.build_manifest(source_path, output)
                prepare.validate_manifest(
                    manifest,
                    source_manifest=source_path,
                    check_tensor_hashes=True,
                )
            self.assertTrue(
                Path(manifest["train"]["path"]).samefile(train_path)
            )
            self.assertTrue(
                Path(manifest["validation"]["path"]).samefile(
                    validation_path
                )
            )
            self.assertEqual(
                manifest["operator_parity"]["protocol_sha256"], "f" * 64
            )
            old = source["selection_anchors"]["values"] + source[
                "test_anchors"
            ]["values"]
            fresh = manifest["selection_anchors"]["values"] + manifest[
                "test_anchors"
            ]["values"]
            windows = sorted(
                [
                    *prepare._windows(old, 100),
                    *prepare._windows(fresh, 100),
                ]
            )
            self.assertTrue(
                all(
                    right[0] >= left[1]
                    for left, right in zip(windows, windows[1:])
                )
            )


class TestArtifactStatus(unittest.TestCase):
    def test_status_moves_from_offline_to_ready_running_and_stop(self):
        from rebuttal.rebuttal_0723.experiments.mla_yarn_operator_parity_5090.status import (
            build_status,
        )

        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            self.assertEqual(
                build_status(work)["phase"], "OFFLINE_NOT_READY"
            )
            (work / "ready_receipt.json").write_text(
                '{"status":"READY"}\n'
            )
            (work / "operator_parity_ready.json").write_text(
                '{"status":"READY"}\n'
            )
            self.assertEqual(
                build_status(work)["phase"], "READY_FOR_GATE"
            )
            run = work / "runs/k8/native_geo/seed42"
            run.mkdir(parents=True)
            (run / "metadata.json").write_text("{}\n")
            (run / "train_log.jsonl").write_text(
                json.dumps(
                    {
                        "step": 100,
                        "optimizer_steps": 2_288,
                        "tokens_seen": 13_107_200,
                        "loss": 4.2,
                        "tokens_per_second": 400_000.0,
                    }
                )
                + "\n"
            )
            running = build_status(work)
            self.assertEqual(running["phase"], "GATE_RUNNING")
            self.assertGreater(
                running["training"]["active"][0]["eta_seconds"], 0
            )
            (work / "operator_parity_gate.json").write_text(
                '{"status":"STOP"}\n'
            )
            stopped = build_status(work)
            self.assertEqual(stopped["phase"], "GATE_STOP")
            self.assertTrue(stopped["terminal"])

    def test_stop_report_keeps_negative_decision_and_provenance(self):
        from rebuttal.rebuttal_0723.experiments.mla_yarn_operator_parity_5090.report import (
            render_markdown,
        )

        gate = {
            "status": "STOP",
            "decision": "do_not_expand",
            "protocol_sha256": "p" * 64,
            "evaluation_code_sha256": "c" * 64,
            "criteria": {"registered_effect": False},
            "rows": TestRuntimeOperatorIdentity._gate_rows(
                "300m", collapse_k32=True
            ),
        }
        text = render_markdown(
            gate=gate,
            ready={"status": "READY", "launcher_sha256": "l" * 64},
            summary=None,
            provenance={
                "gate_sha256": "g" * 64,
                "ready_sha256": "r" * 64,
                "summary_sha256": "not-read",
                "report_code_sha256": "x" * 64,
            },
        )
        self.assertIn("STOP after the registered seed-42", text)
        self.assertIn("test split was not read", text)
        self.assertIn("do_not_expand", text)
        self.assertIn("g" * 64, text)


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class TestRuntimeOperatorIdentity(unittest.TestCase):
    def test_anchor_effects_are_paired_and_retained(self):
        from rebuttal.rebuttal_0723.experiments.mla_yarn_operator_parity_5090.run_experiment import (
            _anchor_effect_stats,
            _paired_tail_differences,
        )

        native = {
            "rows": [
                {
                    "length": 16,
                    "anchor_index": 0,
                    "anchor_endpoint": 100,
                    "tail_nll": 4.2,
                    "tail_tokens": 8,
                },
                {
                    "length": 16,
                    "anchor_index": 1,
                    "anchor_endpoint": 200,
                    "tail_nll": 4.0,
                    "tail_tokens": 8,
                },
            ]
        }
        evq = {
            "rows": [
                {
                    "length": 16,
                    "anchor_index": 0,
                    "anchor_endpoint": 100,
                    "tail_nll": 4.1,
                    "tail_tokens": 8,
                },
                {
                    "length": 16,
                    "anchor_index": 1,
                    "anchor_endpoint": 200,
                    "tail_nll": 3.8,
                    "tail_tokens": 8,
                },
            ]
        }
        paired = _paired_tail_differences(native, evq, length=16)
        stats = _anchor_effect_stats(paired)
        self.assertEqual(stats["anchors"], 2)
        self.assertAlmostEqual(stats["mean"], 0.15)
        self.assertEqual(
            [row["anchor_endpoint"] for row in stats["values"]],
            [100, 200],
        )
        evq["rows"][1]["anchor_endpoint"] = 201
        with self.assertRaisesRegex(ValueError, "do not align"):
            _paired_tail_differences(native, evq, length=16)

    def test_batched_nll_preserves_independent_window_values(self):
        from rebuttal.rebuttal_0723.experiments.mla_scarcity_5090.run_experiment import (
            per_sequence_nll,
        )

        generator = torch.Generator().manual_seed(42)
        logits = torch.randn(3, 7, 11, generator=generator)
        targets = torch.randint(0, 11, (3, 7), generator=generator)
        batched_full, batched_tail, tail = per_sequence_nll(
            logits, targets, tail_tokens=4
        )
        self.assertEqual(tail, 4)
        for index in range(3):
            single_full, single_tail, single_count = per_sequence_nll(
                logits[index : index + 1],
                targets[index : index + 1],
                tail_tokens=4,
            )
            self.assertEqual(single_count, tail)
            self.assertTrue(
                torch.equal(batched_full[index], single_full[0])
            )
            self.assertTrue(
                torch.equal(batched_tail[index], single_tail[0])
            )

    def test_native_checkpoint_has_one_ulp_official_output_parity(self):
        from rebuttal.rebuttal_0723.experiments.mla_scarcity_5090.protocol import (
            training_inv_freq,
        )
        from rebuttal.rebuttal_0723.experiments.mla_yarn_operator_parity_5090.run_experiment import (
            runtime_operator,
        )
        from scripts.lib.rope.official_yarn import (
            official_yarn_on_native_grid,
        )

        for pairs in FREQUENCY_PAIRS:
            trained, _ = training_inv_freq("native_geo", pairs)
            for length in PRIMARY_LENGTHS:
                actual, actual_mscale, metadata = runtime_operator(
                    trained,
                    frequency_pairs=pairs,
                    arm="native_geo",
                    length=length,
                    operator="shared_index_full",
                )
                expected, expected_mscale, _ = (
                    official_yarn_on_native_grid(
                        head_dim=2 * pairs,
                        base=SPEC.base,
                        scale=length / SPEC.train_length,
                        original_max_position_embeddings=SPEC.train_length,
                        beta_fast=SPEC.beta_fast,
                        beta_slow=SPEC.beta_slow,
                    )
                )
                left = actual[:pairs].float().contiguous().view(torch.int32)
                right = expected.float().contiguous().view(torch.int32)
                self.assertLessEqual(
                    int(
                        (
                            left.to(torch.int64) - right.to(torch.int64)
                        )
                        .abs()
                        .max()
                    ),
                    1,
                )
                self.assertEqual(actual_mscale, expected_mscale)
                self.assertTrue(metadata["official_on_input"])
                self.assertTrue(
                    metadata[
                        "official_runtime_output_parity_within_one_fp32_ulp"
                    ]
                )

    def test_arm_identity_is_not_inferred_from_close_values(self):
        from rebuttal.rebuttal_0723.experiments.mla_scarcity_5090.protocol import (
            training_inv_freq,
        )
        from rebuttal.rebuttal_0723.experiments.mla_yarn_operator_parity_5090.run_experiment import (
            runtime_operator,
        )

        trained, _ = training_inv_freq("native_geo", 8)
        altered = trained.clone()
        altered[0] = torch.nextafter(
            altered[0], torch.tensor(0.0, dtype=altered.dtype)
        )
        with self.assertRaisesRegex(
            ValueError, "does not match registered arm"
        ):
            runtime_operator(
                altered,
                frequency_pairs=8,
                arm="native_geo",
                length=16_384,
                operator="shared_index_full",
            )

    @staticmethod
    def _gate_rows(stage: str, *, collapse_k32: bool) -> list[dict]:
        rows = []
        for length in SPEC.eval_lengths:
            primary = length in PRIMARY_LENGTHS
            k8_advantage = 0.10 if primary else -0.01
            k8_interaction = 0.10 if primary else 0.0
            k32_interaction = 0.02 if primary else 0.0
            k32_cost = (
                0.06
                if collapse_k32 and length == PRIMARY_LENGTHS[0]
                else 0.0
            )

            def record(
                advantage: float,
                interaction: float,
                native_cost: float,
            ) -> dict:
                return {
                    "operators": {
                        "shared_index_full": {
                            "native_nll": 4.0 + native_cost,
                            "evq_nll": 4.0 + native_cost - advantage,
                            "evq_advantage": advantage,
                        },
                        "raw": {
                            "native_nll": 4.0,
                            "evq_nll": 4.0 - (advantage - interaction),
                            "evq_advantage": advantage - interaction,
                        },
                    },
                    "shared_interaction": interaction,
                    "native_shared_minus_raw": native_cost,
                }

            rows.append(
                {
                    "seed": 42,
                    "stage": stage,
                    "length": length,
                    "k8": record(k8_advantage, k8_interaction, 0.0),
                    "k32": record(-0.01, k32_interaction, k32_cost),
                    "scarcity_interaction": (
                        k8_interaction - k32_interaction
                    ),
                }
            )
        return rows

    def test_gate_rejects_collapse_in_either_native_budget(self):
        from rebuttal.rebuttal_0723.experiments.mla_yarn_operator_parity_5090 import (
            run_experiment,
        )

        for collapse, expected in ((False, "PASS"), (True, "STOP")):
            with self.subTest(collapse_k32=collapse):
                with tempfile.TemporaryDirectory() as temporary:
                    work = Path(temporary)
                    (work / "operator_parity_ready.json").write_text("{}\n")

                    def rows_for_call(
                        _work_dir,
                        *,
                        split,
                        stage,
                        seeds,
                    ):
                        self.assertEqual(split, "selection")
                        self.assertEqual(seeds, (42,))
                        return self._gate_rows(
                            stage, collapse_k32=collapse
                        )

                    with mock.patch.object(
                        run_experiment,
                        "_validate_parity_ready_record",
                        return_value={"official_native_parity": {}},
                    ), mock.patch.object(
                        run_experiment,
                        "effect_rows",
                        side_effect=rows_for_call,
                    ):
                        with contextlib.redirect_stdout(io.StringIO()):
                            result = run_experiment.gate(work)
                    self.assertEqual(result["status"], expected)
                    if collapse:
                        self.assertFalse(
                            result["criteria"][
                                "native_scaler_cost_lte_0p05_for_k8_k32_at_16k_32k"
                            ]
                        )

    def test_underlying_trainer_fails_closed_before_pass_gate(self):
        from rebuttal.rebuttal_0723.experiments.mla_scarcity_5090.run_experiment import (
            enforce_nested_operator_parity_phase,
        )

        manifest = {
            "operator_parity": {
                "protocol_sha256": SPEC.fingerprint(),
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            parity_ready = work / "operator_parity_ready.json"
            parity_ready.write_text(
                json.dumps(
                    {
                        "status": "READY",
                        "protocol_sha256": SPEC.fingerprint(),
                        "evaluation_code_sha256": "e" * 64,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            enforce_nested_operator_parity_phase(
                manifest, work, seed=42, split="selection"
            )
            with self.assertRaisesRegex(RuntimeError, "requires a PASS"):
                enforce_nested_operator_parity_phase(
                    manifest, work, seed=43
                )
            with self.assertRaisesRegex(RuntimeError, "requires a PASS"):
                enforce_nested_operator_parity_phase(
                    manifest, work, seed=42, split="test"
                )
            (work / "operator_parity_gate.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "protocol_sha256": SPEC.fingerprint(),
                        "selection_split_only": True,
                        "test_split_read": False,
                        "evaluation_code_sha256": "e" * 64,
                        "ready_receipt_sha256": _sha(parity_ready),
                    }
                )
            )
            enforce_nested_operator_parity_phase(
                manifest, work, seed=43
            )
            enforce_nested_operator_parity_phase(
                manifest, work, seed=42, split="test"
            )
            with self.assertRaisesRegex(
                RuntimeError, "selection is restricted"
            ):
                enforce_nested_operator_parity_phase(
                    manifest, work, seed=43, split="selection"
                )

    @staticmethod
    def _summary_rows(stage: str) -> list[dict]:
        rows = []
        for seed in (42, 43, 88):
            for length in SPEC.eval_lengths:
                primary = length in PRIMARY_LENGTHS

                def record(pairs: int) -> dict:
                    shared_advantage = (
                        (0.10 if pairs == 8 else 0.02)
                        if primary
                        else -0.01
                    )
                    raw_advantage = 0.0 if primary else -0.01
                    operators = {}
                    for operator in OPERATORS:
                        advantage = (
                            shared_advantage
                            if operator == "shared_index_full"
                            else raw_advantage
                        )
                        operators[operator] = {
                            "native_nll": 4.0,
                            "evq_nll": 4.0 - advantage,
                            "evq_advantage": advantage,
                        }
                    return {
                        "operators": operators,
                        "shared_interaction": (
                            shared_advantage - raw_advantage
                        ),
                        "native_shared_minus_raw": 0.0,
                    }

                k8 = record(8)
                k32 = record(32)
                rows.append(
                    {
                        "seed": seed,
                        "stage": stage,
                        "length": length,
                        "k8": k8,
                        "k32": k32,
                        "scarcity_interaction": (
                            k8["shared_interaction"]
                            - k32["shared_interaction"]
                        ),
                    }
                )
        return rows

    def test_summary_requires_all_seeds_and_builds_claim_gate(self):
        from rebuttal.rebuttal_0723.experiments.mla_yarn_operator_parity_5090 import (
            run_experiment,
        )

        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / "operator_parity_gate.json").write_text("{}\n")

            def rows_for_call(
                _work_dir,
                *,
                split,
                stage,
                seeds,
            ):
                self.assertEqual(split, "test")
                self.assertEqual(seeds, (42, 43, 88))
                return self._summary_rows(stage)

            with mock.patch.object(
                run_experiment,
                "validate_gate",
                return_value={"status": "PASS"},
            ), mock.patch.object(
                run_experiment,
                "effect_rows",
                side_effect=rows_for_call,
            ), contextlib.redirect_stdout(io.StringIO()):
                result = run_experiment.summarize(work)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(
                result["claim_gate"],
                "SUPPORTS_SHARED_OPERATOR_SCARCITY_INTERACTION",
            )
            self.assertEqual(
                result["precision_grade"],
                "SEED_LEVEL_CI95_EXCLUDES_ZERO",
            )
            self.assertEqual(len(result["per_seed_primary"]), 3)
            self.assertEqual(len(result["aggregates"]), 8)
            from rebuttal.rebuttal_0723.experiments.mla_yarn_operator_parity_5090.report import (
                render_markdown,
            )

            report = render_markdown(
                gate={"status": "PASS"},
                ready={
                    "status": "READY",
                    "launcher_sha256": "l" * 64,
                },
                summary=result,
                provenance={
                    "gate_sha256": "g" * 64,
                    "ready_sha256": "r" * 64,
                    "summary_sha256": "s" * 64,
                    "report_code_sha256": "x" * 64,
                },
            )
            self.assertIn(
                "SUPPORTS_SHARED_OPERATOR_SCARCITY_INTERACTION",
                report,
            )
            self.assertIn("300M operator decomposition", report)
            self.assertIn("Per-seed primary checks", report)
            self.assertIn("Seed-level precision checks", report)


if __name__ == "__main__":
    unittest.main()
