#!/usr/bin/env python3
"""Contract tests for the portable July rebuttal evidence bundle."""

from __future__ import annotations

import csv
import json
import re
import tempfile
import unittest
from pathlib import Path

from scripts import package_supplement, validate_rebuttal_evidence_bundle
from scripts.core_text_phases import export_phase16_manifest


ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data" / "curated"
REBUTTAL = ROOT / "rebuttal_7"
TRACE_ONLY = REBUTTAL / "trace_only" / "text_base_10k_500k_pilot.json"

EXPECTED_JSON = {
    "learnable_tau_128tok_evidence.json": "report-backed",
    "mla_channel_count_125m_pilot.json": "report-backed",
    "phase11_l256_3seed_recovered.json": "raw-json-backed",
    "phase16_99run_manifest.meta.json": "sanitized-run-manifest",
    "quality_454m_full_eval.json": "report-backed",
    "table18_mla_3seed_aggregate.json": "raw-json-backed",
}

PUBLIC_BUNDLE_FILES = [
    *(CURATED / name for name in EXPECTED_JSON),
    CURATED / "phase16_99run_manifest.csv",
    TRACE_ONLY,
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
        self.assertTrue(TRACE_ONLY.is_file())
        self.assertEqual(
            json.loads(TRACE_ONLY.read_text(encoding="utf-8"))["provenance_status"],
            "trace-only",
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

    def test_quality_asset_cannot_masquerade_as_raw_json_backed(self):
        data = load_json("quality_454m_full_eval.json")
        self.assertEqual(data["provenance_status"], "report-backed")
        self.assertFalse(data["raw_artifact"]["available"])
        self.assertEqual(data["raw_artifact"]["expected_eval_samples"], 2086)
        self.assertNotIn("surviving aggregate", json.dumps(data).lower())

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

    def test_reviewer_supplement_excludes_trace_only_and_internal_contract_test(self):
        self.assertIn(
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
