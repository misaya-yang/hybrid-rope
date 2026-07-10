#!/usr/bin/env python3
"""Contract tests for the paper experiment code workspace."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from scripts import package_supplement


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "paper_experiments"


class PaperExperimentWorkspaceTests(unittest.TestCase):
    def test_workspace_covers_every_paper_experiment_family(self):
        manifest_path = WORKSPACE / "MANIFEST.json"
        self.assertTrue(manifest_path.is_file(), manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        required_families = {
            "shared_rope_and_training",
            "primary_i_evq_yarn",
            "primary_ii_pe_dominant",
            "primary_iii_mla",
            "theory_and_mechanism",
            "supporting_text",
            "supporting_video_dit",
            "supporting_lora_8b",
            "data_preparation",
            "paper_figures",
        }
        self.assertEqual(set(manifest["families"]), required_families)

        sources = {entry["source"] for entry in manifest["files"]}
        for source in (
            "scripts/lib/rope/schedules.py",
            "scripts/core_text_phases/run_evq_sweep.py",
            "scripts/core_text_phases/phase14c_multiscale_evq_yarn.py",
            "scripts/core_text_phases/phase11b_125m_dape.py",
            "scripts/core_text_phases/run_gqa_evq_experiment.py",
            "scripts/core_text_phases/phase16_formula_optimality_sweep.py",
            "scripts/core_text_phases/phase21b_quality_eval_clean.py",
            "scripts/video_temporal/run_dit_temporal.py",
            "experiments/lora_evq_v2/train_evq_lora.py",
            "scripts/figures/fig3_pe_dominant_scaling.py",
        ):
            self.assertIn(source, sources)

        self.assertNotIn("scripts/core_text_phases/phase21b_quality_eval.py", sources)
        self.assertNotIn("scripts/core_text_phases/eval_passkey.py", sources)

    def test_links_resolve_inside_repo_and_match_manifest_hashes(self):
        manifest = json.loads(
            (WORKSPACE / "MANIFEST.json").read_text(encoding="utf-8")
        )
        root = ROOT.resolve()
        for entry in manifest["files"]:
            link = WORKSPACE / entry["workspace_path"]
            self.assertTrue(link.is_symlink(), link)
            resolved = link.resolve(strict=True)
            self.assertTrue(resolved.is_relative_to(root), resolved)
            self.assertEqual(resolved, (ROOT / entry["source"]).resolve())
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            self.assertEqual(digest, entry["sha256"], entry["source"])

    def test_workspace_metadata_passes_anonymity_scan(self):
        for path in (WORKSPACE / "README.md", WORKSPACE / "MANIFEST.json"):
            self.assertTrue(path.is_file(), path)
            self.assertIsNone(package_supplement.LEAK_PATTERNS.search(path.read_bytes()), path)


if __name__ == "__main__":
    unittest.main()
