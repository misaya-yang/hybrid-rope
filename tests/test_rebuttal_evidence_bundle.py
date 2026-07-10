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
REBUTTAL = ROOT / "rebuttal_7"

EXPECTED_JSON = {
    "learnable_tau_128tok_evidence.json": "report-backed",
    "mla_channel_count_125m_pilot.json": "report-backed",
    "phase11_l256_3seed_recovered.json": "raw-json-backed",
    "phase16_99run_manifest.meta.json": "sanitized-run-manifest",
    "quality_454m_full_eval.json": "raw-json-backed",
    "table18_mla_3seed_aggregate.json": "raw-json-backed",
    "text_base_10k_500k_pilot.json": "raw-json-backed",
}

PUBLIC_BUNDLE_FILES = [
    *(CURATED / name for name in EXPECTED_JSON),
    CURATED / "phase16_99run_manifest.csv",
    REBUTTAL / "IGNORED_ASSET_RECONCILIATION.md",
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
        self.assertTrue((REBUTTAL / "IGNORED_ASSET_RECONCILIATION.md").is_file())
        self.assertFalse(
            (REBUTTAL / "trace_only" / "text_base_10k_500k_pilot.json").exists()
        )
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

    def test_phase11_asset_contains_both_nine_run_sources(self):
        data = load_json("phase11_l256_3seed_recovered.json")
        self.assertEqual(
            data["sources"]["raw"]["sha256"],
            "6bdf97335365ea3a92c15ff84fc52f292ddad96b0f6e142f8b98199295dffa30",
        )
        self.assertEqual(
            data["sources"]["yarn"]["sha256"],
            "1f9550c46fa5b51b24b4d2e805c4dbbba8d639c19f664e812072bf8659b85321",
        )
        self.assertEqual(len(data["raw_runs"]), 9)
        self.assertEqual(len(data["yarn_runs"]), 9)

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

    def test_local_phase11_directory_takes_precedence_over_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / "results" / "core_text" / "phase11"
            archive = root / "07 - rebuttal" / "all_paper_experiment_code"
            local.mkdir(parents=True)
            archive.mkdir(parents=True)
            self.assertEqual(
                build_rebuttal_evidence_bundle.default_phase11_dir(root), local
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

    def test_detailed_handoff_covers_inventory_all_questions_and_company_setup(self):
        text = (REBUTTAL / "IGNORED_ASSET_RECONCILIATION.md").read_text(
            encoding="utf-8"
        )
        for family in (
            "07 - rebuttal/",
            "RESULT_PROVENANCE_MANIFEST.md",
            ".codex_tmp/",
            "results/",
            ".venv/",
        ):
            self.assertIn(family, text)
        for question in range(1, 19):
            heading = f"### F5-Q{question} "
            self.assertEqual(text.count(heading), 1, heading)
        self.assertIn("git fetch origin", text)
        self.assertIn("git switch", text)
        self.assertIn("validate_rebuttal_evidence_bundle.py", text)

    def test_reviewer_supplement_includes_raw_base_and_excludes_internal_contract_test(self):
        self.assertNotIn(
            "text_base_10k_500k_pilot.json", package_supplement.EXCLUDE_NAMES
        )
        self.assertIn(
            "test_rebuttal_evidence_bundle.py", package_supplement.EXCLUDE_NAMES
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
