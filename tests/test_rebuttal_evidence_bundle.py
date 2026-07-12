#!/usr/bin/env python3
"""Contract tests for the portable July rebuttal evidence bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import (
    build_rebuttal_evidence_bundle,
    package_supplement,
    validate_rebuttal_evidence_bundle,
)
from scripts.core_text_phases import export_phase16_manifest


ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data" / "curated"

EXPECTED_JSON = {
    "learnable_tau_128tok_evidence.json": "report-backed",
    "mla_channel_count_125m_pilot.json": "report-backed",
    "primary1_evq_yarn_10pct_raw.json": "raw-json-backed",
    "primary2_l128_fixed_tau5_3seed.json": "raw-json-backed",
    "phase11_l256_3seed_recovered.json": "raw-json-backed",
    "phase11b_125m_l256_3seed.json": "raw-json-backed",
    "phase16_99run_manifest.meta.json": "sanitized-run-manifest",
    "quality_454m_full_eval.json": "raw-json-backed",
    "table18_mla_3seed_aggregate.json": "raw-json-backed",
    "text_base_10k_500k_pilot.json": "raw-json-backed",
}

PUBLIC_BUNDLE_FILES = [
    *(CURATED / name for name in EXPECTED_JSON),
    CURATED / "eval_3seeds_full_results.json",
    CURATED / "phase16_99run_manifest.csv",
]


def load_json(name: str) -> dict:
    return json.loads((CURATED / name).read_text(encoding="utf-8"))


class RebuttalEvidenceBundleTests(unittest.TestCase):
    def test_expected_assets_exist_with_explicit_provenance_tiers(self):
        for name, expected_status in EXPECTED_JSON.items():
            path = CURATED / name
            self.assertTrue(path.is_file(), path)
            self.assertEqual(load_json(name)["provenance_status"], expected_status)

        self.assertTrue((CURATED / "phase16_99run_manifest.csv").is_file())
        for path in CURATED.glob("*.json"):
            self.assertNotEqual(load_json(path.name).get("provenance_status"), "trace-only")

    def test_raw_backed_mla_asset_preserves_source_identity_and_seedwise_values(self):
        data = load_json("table18_mla_3seed_aggregate.json")
        self.assertEqual(
            data["source"]["sha256"],
            "1e44d30bb880e4b7427ae55bd7034782989152bd2afca9217495f9b8ece30953",
        )
        self.assertEqual(data["seeds"], [42, 43, 88])
        self.assertEqual(set(data["extended"]), {
            "GEO",
            "GEO+YaRN(s=2)",
            "GEO+YaRN(s=4)",
            "EVQ",
            "EVQ+YaRN(s=2)",
            "EVQ+YaRN(s=4)",
        })

    def test_mla_source_is_reconstructed_byte_for_byte_from_portable_snapshot(self):
        reconstructed = build_rebuttal_evidence_bundle.reconstruct_mla_source_bytes(
            CURATED / "table18_mla_3seed_aggregate.json"
        )
        self.assertEqual(
            hashlib.sha256(reconstructed).hexdigest(),
            "1e44d30bb880e4b7427ae55bd7034782989152bd2afca9217495f9b8ece30953",
        )
        self.assertEqual(
            (CURATED / "eval_3seeds_full_results.json").read_bytes(),
            reconstructed,
        )

    def test_primary1_builder_preserves_full_raw_payload_and_recomputes_table_means(self):
        source = ROOT / "data" / "results_5090b" / "evq_yarn_10pct_allseeds.json"
        built = build_rebuttal_evidence_bundle.build_primary1_snapshot(source)

        self.assertEqual(built["source"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(len(built["raw_payload"]["results"]), 6)
        self.assertEqual(built["protocol"]["seeds"], [7, 42, 123])
        rows = {row["id"]: row for row in built["recomputed_rows"]}
        self.assertAlmostEqual(rows["geo_raw"]["ppl_8192_mean"], 161.8626666667)
        self.assertAlmostEqual(rows["geo_yarn_s8"]["pk_8192_mean"], 0.6133333333)
        self.assertAlmostEqual(rows["evq_raw"]["pk_8192_mean"], 0.5333333333)
        self.assertAlmostEqual(rows["evq_yarn_s8"]["ppl_8192_mean"], 70.8506666667)

    def test_primary2_builder_recovers_only_fixed_evq_tau5_three_seed_arm(self):
        seed42 = (
            ROOT
            / "data"
            / "evq_128tok_results"
            / "extended_sweep"
            / "results_final.json"
        )
        extra_seeds = (
            ROOT
            / "data"
            / "evq_128tok_results"
            / "phase7"
            / "multiseed"
            / "results_final.json"
        )
        built = build_rebuttal_evidence_bundle.build_primary2_tau5_snapshot(
            seed42, extra_seeds
        )

        self.assertEqual([run["seed"] for run in built["runs"]], [42, 137, 256])
        self.assertTrue(all(run["tau"] == 5.0 for run in built["runs"]))
        self.assertAlmostEqual(built["summary"]["ppl_128_mean"], 182.6046666667)
        self.assertAlmostEqual(built["summary"]["ppl_8192_mean"], 335.7103333333)
        self.assertIn("does not recover", built["claim_boundary"])
        self.assertIn("Geo", built["claim_boundary"])
        self.assertIn("DAPE", built["claim_boundary"])

    def test_phase11_tracked_snapshot_reconstructs_both_source_payloads(self):
        snapshot = CURATED / "phase11_l256_3seed_recovered.json"
        data = load_json(snapshot.name)

        self.assertEqual(len(data["raw_runs"]), 9)
        self.assertEqual(len(data["yarn_runs"]), 9)
        self.assertEqual(set(data["raw_runs"]), set(data["yarn_runs"]))
        for field, source, expected_key in (
            ("raw_runs", "raw", "phase11_raw"),
            ("yarn_runs", "yarn", "phase11_yarn"),
        ):
            reconstructed = json.dumps(data[field], indent=2).encode("utf-8")
            reconstructed_sha = hashlib.sha256(reconstructed).hexdigest()
            self.assertEqual(reconstructed_sha, data["sources"][source]["sha256"])
            self.assertEqual(
                reconstructed_sha,
                build_rebuttal_evidence_bundle.EXPECTED_SHA256[expected_key],
            )

        self.assertEqual(
            build_rebuttal_evidence_bundle.validated_phase11_snapshot_bytes(snapshot),
            snapshot.read_bytes(),
        )

    def test_phase11_tracked_snapshot_rejects_top_level_tampering(self):
        snapshot = CURATED / "phase11_l256_3seed_recovered.json"
        mutations = {
            "claim_boundary": lambda payload: payload.__setitem__(
                "claim_boundary", "tampered claim boundary"
            ),
            "private_machine_path": lambda payload: payload.__setitem__(
                "private_machine_path", "/private/training-host/checkpoint.pt"
            ),
            "arbitrary_top_level_field": lambda payload: payload.__setitem__(
                "unexpected_top_level_field", {"status": "tampered"}
            ),
        }

        with tempfile.TemporaryDirectory() as tmp:
            for name, mutate in mutations.items():
                with self.subTest(mutation=name):
                    payload = json.loads(snapshot.read_text(encoding="utf-8"))
                    mutate(payload)
                    tampered = Path(tmp) / f"phase11_{name}.json"
                    tampered.write_text(
                        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
                    )

                    with self.assertRaises(ValueError):
                        build_rebuttal_evidence_bundle.validated_phase11_snapshot_bytes(
                            tampered
                        )

    def test_phase11b_builder_preserves_all_runs_and_protocol_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            scaling_source = Path(tmp) / "scaling.json"
            dape_source = Path(tmp) / "dape.json"
            scaling = {
                f"125m_{method}_seed{seed}": {
                    "run_id": f"125m_{method}_seed{seed}",
                    "seed": seed,
                    "ppl": {"256": 70.0, "8192": 200.0},
                    "use_dape": False,
                }
                for method in ("geo", "evq2.0", "evq4.0")
                for seed in (42, 137, 256)
            }
            dape = {
                f"125m_{method}_dape_seed{seed}": {
                    "run_id": f"125m_{method}_dape_seed{seed}",
                    "seed": seed,
                    "ppl": {"256": 68.0, "8192": 56.0},
                    "use_dape": True,
                }
                for method in ("geo", "evq4.0")
                for seed in (42, 137, 256)
            }
            scaling_source.write_text(json.dumps(scaling), encoding="utf-8")
            dape_source.write_text(json.dumps(dape), encoding="utf-8")
            expected = {
                "phase11b_scaling": hashlib.sha256(
                    scaling_source.read_bytes()
                ).hexdigest(),
                "phase11b_dape": hashlib.sha256(dape_source.read_bytes()).hexdigest(),
            }
            with mock.patch.dict(
                build_rebuttal_evidence_bundle.EXPECTED_SHA256,
                expected,
            ):
                built = build_rebuttal_evidence_bundle.build_phase11b_snapshot(
                    scaling_source, dape_source
                )

            self.assertEqual(len(built["scaling_runs"]), 9)
            self.assertEqual(len(built["dape_runs"]), 6)
            self.assertEqual(built["protocol"]["seeds"], [42, 137, 256])
            self.assertIn("L_train=256", built["claim_boundary"])
            self.assertIn("not", built["claim_boundary"])

    def test_phase11b_tracked_snapshot_reconstructs_both_source_payloads(self):
        snapshot = CURATED / "phase11b_125m_l256_3seed.json"
        data = load_json(snapshot.name)

        self.assertEqual(len(data["scaling_runs"]), 9)
        self.assertEqual(len(data["dape_runs"]), 6)
        all_runs = [*data["scaling_runs"].values(), *data["dape_runs"].values()]
        self.assertEqual(sorted({run["seed"] for run in all_runs}), [42, 137, 256])
        self.assertTrue(
            all(not run.get("use_dape") for run in data["scaling_runs"].values())
        )
        self.assertTrue(
            all(run.get("use_dape") for run in data["dape_runs"].values())
        )
        for field, source, expected_key in (
            ("scaling_runs", "scaling", "phase11b_scaling"),
            ("dape_runs", "dape", "phase11b_dape"),
        ):
            reconstructed = json.dumps(data[field], indent=2).encode("utf-8")
            reconstructed_sha = hashlib.sha256(reconstructed).hexdigest()
            self.assertEqual(reconstructed_sha, data["sources"][source]["sha256"])
            self.assertEqual(
                reconstructed_sha,
                build_rebuttal_evidence_bundle.EXPECTED_SHA256[expected_key],
            )

        self.assertEqual(
            build_rebuttal_evidence_bundle.validated_phase11b_snapshot_bytes(snapshot),
            snapshot.read_bytes(),
        )

    def test_phase11b_tracked_snapshot_rejects_top_level_tampering(self):
        snapshot = CURATED / "phase11b_125m_l256_3seed.json"
        mutations = {
            "claim_boundary": lambda payload: payload.__setitem__(
                "claim_boundary", "tampered claim boundary"
            ),
            "private_machine_path": lambda payload: payload.__setitem__(
                "private_machine_path", "/private/training-host/checkpoint.pt"
            ),
            "arbitrary_top_level_field": lambda payload: payload.__setitem__(
                "unexpected_top_level_field", {"status": "tampered"}
            ),
        }

        with tempfile.TemporaryDirectory() as tmp:
            for name, mutate in mutations.items():
                with self.subTest(mutation=name):
                    payload = json.loads(snapshot.read_text(encoding="utf-8"))
                    mutate(payload)
                    tampered = Path(tmp) / f"phase11b_{name}.json"
                    tampered.write_text(
                        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
                    )

                    with self.assertRaises(ValueError):
                        build_rebuttal_evidence_bundle.validated_phase11b_snapshot_bytes(
                            tampered
                        )

    def test_quality_asset_is_raw_backed_and_sanitized(self):
        data = load_json("quality_454m_full_eval.json")
        self.assertEqual(data["provenance_status"], "raw-json-backed")
        self.assertEqual(
            data["source"]["sha256"],
            "5fc3254cb7b44a918328056ccd505d01e5539dc596d4c06273c9914ec93e3caa",
        )
        self.assertEqual(data["protocol"]["eval_samples"], 2086)
        serialized = json.dumps(data).lower()
        self.assertNotIn("server", serialized)
        self.assertNotIn("checkpoint", serialized)

    def test_quality_builder_verifies_hash_and_omits_machine_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "quality.json"
            payload = {
                "setup": {
                    "model_tier": "454M",
                    "architecture": "24L/16H/1024d, head_dim=64",
                    "pretrain_context": 2048,
                    "continue_train_context": 4096,
                    "finetune_context": 4096,
                    "finetune_steps": 2000,
                    "finetune_seed": 42,
                    "eval_samples": 2086,
                    "eval_scoring": "length_normalized_option_nll (4-option accuracy)",
                    "random_baseline_accuracy": 25.0,
                    "server": "private-machine",
                },
                "models": {"geo": {"checkpoint": "/private/geo.pt"}},
                "results_raw": {
                    length: {
                        "geo_accuracy": 24.0,
                        "geo_correct": 500,
                        "geo_gold_nll": 3.0,
                        "evq_accuracy": 25.0,
                        "evq_correct": 520,
                        "evq_gold_nll": 2.0,
                    }
                    for length in ("4k", "8k", "16k")
                },
                "results_yarn": {
                    "8k_yarn_scale2": {
                        "geo_accuracy": 24.0,
                        "geo_correct": 500,
                        "geo_gold_nll": 3.0,
                        "evq_accuracy": 25.0,
                        "evq_correct": 520,
                        "evq_gold_nll": 2.0,
                    }
                },
            }
            source.write_text(json.dumps(payload), encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with mock.patch.dict(
                build_rebuttal_evidence_bundle.EXPECTED_SHA256,
                {"quality": digest},
            ):
                built = build_rebuttal_evidence_bundle.build_quality_snapshot(source)
            self.assertEqual(built["source"]["sha256"], digest)
            self.assertEqual(len(built["rows"]), 4)
            self.assertNotIn("private-machine", json.dumps(built))
            self.assertNotIn("/private/geo.pt", json.dumps(built))

    def test_base_builder_preserves_four_source_hashes_and_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources = {}
            expected = {}
            for base in (10000, 500000):
                for method, offset in (("geo", 0.0), ("evq", -10.0)):
                    key = f"base_{base}_{method}"
                    path = Path(tmp) / f"{key}.json"
                    path.write_text(
                        json.dumps(
                            {
                                "ppl": {
                                    "512": 100.0 + offset,
                                    "1024": 120.0 + offset,
                                    "2048": 150.0 + offset,
                                    "4096": 200.0 + offset,
                                }
                            }
                        ),
                        encoding="utf-8",
                    )
                    sources[key] = path
                    expected[key] = hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch.dict(
                build_rebuttal_evidence_bundle.EXPECTED_SHA256,
                expected,
            ):
                built = build_rebuttal_evidence_bundle.build_base_snapshot(sources)
            self.assertEqual(built["provenance_status"], "raw-json-backed")
            self.assertEqual(set(built["sources"]), set(expected))
            self.assertEqual(len(built["rows"]), 2)
            self.assertIn("single-seed", built["claim_boundary"])

    def test_component_selection_can_skip_missing_mla_source(self):
        args = build_rebuttal_evidence_bundle.parse_args(
            ["--only", "phase11", "--only", "quality", "--only", "base"]
        )
        self.assertEqual(args.only, ["phase11", "quality", "base"])

    def test_component_selection_exposes_recovered_primary_assets(self):
        args = build_rebuttal_evidence_bundle.parse_args(
            ["--only", "primary1", "--only", "primary2_tau5", "--only", "mla_raw"]
        )
        self.assertEqual(args.only, ["primary1", "primary2_tau5", "mla_raw"])

    def test_phase11_defaults_use_tracked_curated_snapshots(self):
        args = build_rebuttal_evidence_bundle.parse_args([])
        self.assertEqual(
            args.phase11_snapshot_source,
            CURATED / "phase11_l256_3seed_recovered.json",
        )
        self.assertEqual(
            args.phase11b_snapshot_source,
            CURATED / "phase11b_125m_l256_3seed.json",
        )
        self.assertIsNone(args.phase11_raw_source)
        self.assertIsNone(args.phase11_yarn_source)
        self.assertIsNone(args.phase11b_scaling_source)
        self.assertIsNone(args.phase11b_dape_source)

    def test_phase11_raw_import_sources_must_be_provided_in_pairs(self):
        for option in (
            "--phase11-raw-source",
            "--phase11-yarn-source",
            "--phase11b-scaling-source",
            "--phase11b-dape-source",
        ):
            with self.subTest(option=option), mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    build_rebuttal_evidence_bundle.parse_args(
                        [option, str(CURATED / "unused.json")]
                    )

    def test_phase11_default_build_copies_validated_snapshots_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with mock.patch(
                "sys.argv",
                [
                    "build_rebuttal_evidence_bundle.py",
                    "--only",
                    "phase11",
                    "--only",
                    "phase11b",
                    "--output-dir",
                    str(output_dir),
                ],
            ):
                build_rebuttal_evidence_bundle.main()

            for name in (
                "phase11_l256_3seed_recovered.json",
                "phase11b_125m_l256_3seed.json",
            ):
                self.assertEqual(
                    (output_dir / name).read_bytes(),
                    (CURATED / name).read_bytes(),
                )

    def test_phase16_manifest_has_all_99_sanitized_run_rows(self):
        path = CURATED / "phase16_99run_manifest.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
        metadata = load_json("phase16_99run_manifest.meta.json")
        self.assertEqual(len(rows), 99)
        self.assertEqual(metadata["row_count"], 99)
        self.assertEqual(
            metadata["source_bundle"]["expected_inputs"],
            [
                "pilot_plan.json",
                "confirm_plan.json",
                "runs/*/result.json",
            ],
        )
        self.assertEqual({row["stage"] for row in rows}, {"pilot", "confirm"})
        self.assertTrue(all(row["inv_freq_hash"] for row in rows))
        self.assertEqual(
            validate_rebuttal_evidence_bundle.validate_phase16_manifest(
                path, CURATED / "phase16_99run_manifest.meta.json"
            ),
            [],
        )

        with tempfile.TemporaryDirectory() as tmp:
            broken_csv = Path(tmp) / "manifest.csv"
            broken_meta = Path(tmp) / "manifest.meta.json"
            rows[1]["run_id"] = rows[0]["run_id"]
            with broken_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            broken_metadata = dict(metadata)
            broken_metadata["sha256"] = validate_rebuttal_evidence_bundle.digest(
                broken_csv
            )
            broken_meta.write_text(json.dumps(broken_metadata), encoding="utf-8")
            errors = validate_rebuttal_evidence_bundle.validate_phase16_manifest(
                broken_csv, broken_meta
            )
            self.assertTrue(any("unique run_id" in error for error in errors), errors)

    def test_public_bundle_is_anonymous_and_path_safe(self):
        forbidden = re.compile(
            r"misaya|yanghej|hejaz|/Users/|/root/autodl-tmp|sshpass|seetacloud|"
            r"BEGIN (?:OPENSSH|RSA) PRIVATE KEY|hf_[A-Za-z0-9]{20,}|"
            r"ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}",
            re.IGNORECASE,
        )
        for path in PUBLIC_BUNDLE_FILES:
            self.assertIsNone(forbidden.search(path.read_text(encoding="utf-8")), path)

    def test_validator_rejects_identity_markers_not_only_absolute_paths(self):
        for marker in ("Misaya", "yanghej", "hejaz"):
            self.assertIsNotNone(
                validate_rebuttal_evidence_bundle.FORBIDDEN.search(marker), marker
            )

    def test_validator_ignores_legacy_rebuttal_prose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            curated = root / "data" / "curated"
            curated.mkdir(parents=True)
            stale_handoff = root / "stale_handoff.md"
            stale_summary = root / "stale_summary.md"
            stale_handoff.write_text("incomplete legacy handoff", encoding="utf-8")
            stale_summary.write_text("legacy summary", encoding="utf-8")

            with mock.patch.multiple(
                validate_rebuttal_evidence_bundle,
                ROOT=root,
                CURATED=curated,
                EXPECTED_JSON={},
                EXPECTED_RAW_SHA256={},
            ):
                with mock.patch.object(
                    validate_rebuttal_evidence_bundle,
                    "REBUTTAL_DOC",
                    stale_handoff,
                    create=True,
                ), mock.patch.object(
                    validate_rebuttal_evidence_bundle,
                    "CORE_ASSET_SUMMARY",
                    stale_summary,
                    create=True,
                ), mock.patch.object(
                    validate_rebuttal_evidence_bundle,
                    "validate_phase16_manifest",
                    return_value=[],
                ):
                    errors = validate_rebuttal_evidence_bundle.validate_bundle(
                        require_tracked=False
                    )

            self.assertEqual(errors, [])

    def test_reviewer_supplement_includes_raw_base_and_excludes_internal_contract_test(self):
        self.assertNotIn(
            "text_base_10k_500k_pilot.json", package_supplement.EXCLUDE_NAMES
        )
        self.assertIn(
            "test_rebuttal_evidence_bundle.py", package_supplement.EXCLUDE_NAMES
        )
        self.assertIn(
            "test_paper_experiment_workspace.py", package_supplement.EXCLUDE_NAMES
        )
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            renamed = stage / "data" / "curated" / "renamed_quarantine.json"
            renamed.parent.mkdir(parents=True)
            renamed.write_text(
                json.dumps({"provenance_status": "trace-only"}), encoding="utf-8"
            )
            self.assertEqual(
                package_supplement.scan_for_trace_only(stage),
                ["data/curated/renamed_quarantine.json"],
            )

    def test_reviewer_supplement_includes_rebuttal_evidence_entrypoints(self):
        required = {
            "scripts/core_text_phases/run_gqa_evq_experiment.py",
            "scripts/core_text_phases/eval_dsr.py",
            "scripts/core_text_phases/export_phase16_manifest.py",
            "scripts/core_text_phases/phase18_base_generalization_sweep.py",
        }
        self.assertTrue(required.issubset(package_supplement.ALLOWLIST))

    def test_phase16_exporter_reconstructs_the_portable_99_row_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            output = Path(tmp) / "manifest.csv"
            source.mkdir()

            plans = {"pilot": [], "confirm": []}
            for stage, count in (("pilot", 45), ("confirm", 54)):
                for index in range(count):
                    run_id = f"{stage}_run_{index:02d}"
                    spec = {
                        "stage": stage,
                        "run_id": run_id,
                        "tier": "50m",
                        "seq_len": 256,
                        "num_heads": 4,
                        "head_dim": 128,
                        "tau": float(index),
                        "theory_tau": 8.0,
                        "seed": 42,
                        "train_tokens": 8_388_608,
                        "eval_lengths": [256, 512],
                        "passkey_lengths": [512],
                        "passkey_trials": 8,
                    }
                    plans[stage].append(spec)
                    run_dir = source / "runs" / run_id
                    run_dir.mkdir(parents=True)
                    (run_dir / "result.json").write_text(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "stage": stage,
                                "tier": spec["tier"],
                                "seed": spec["seed"],
                                "seq_len": spec["seq_len"],
                                "num_heads": spec["num_heads"],
                                "head_dim": spec["head_dim"],
                                "tau": spec["tau"],
                                "theory_tau": spec["theory_tau"],
                                "inv_freq_hash": f"hash-{stage}-{index}",
                                "ppl": {"512": 100.0 + index},
                                "passkey": {
                                    "summary": {"L=512": {"retrieval_rate": 0.5}}
                                },
                            }
                        ),
                        encoding="utf-8",
                    )

            (source / "pilot_plan.json").write_text(
                json.dumps(plans["pilot"]), encoding="utf-8"
            )
            (source / "confirm_plan.json").write_text(
                json.dumps(plans["confirm"]), encoding="utf-8"
            )

            rows = export_phase16_manifest.load_rows(source)
            export_phase16_manifest.write_csv(rows, output)

            self.assertEqual(len(rows), 99)
            self.assertEqual({row["stage"] for row in rows}, {"pilot", "confirm"})
            with output.open(newline="", encoding="utf-8") as handle:
                exported = list(csv.DictReader(handle))
            self.assertEqual(len(exported), 99)
            self.assertEqual(
                exported[0]["passkey_summary_json"],
                '{"L=512":{"retrieval_rate":0.5}}',
            )

            mismatched_path = source / "runs" / "pilot_run_00" / "result.json"
            mismatched = json.loads(mismatched_path.read_text(encoding="utf-8"))
            mismatched["head_dim"] = 64
            mismatched_path.write_text(json.dumps(mismatched), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "head_dim mismatch"):
                export_phase16_manifest.load_rows(source)


if __name__ == "__main__":
    unittest.main()
