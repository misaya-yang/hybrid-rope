import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryNavigationTests(unittest.TestCase):
    def test_canonical_navigation_files_exist(self):
        for relative in (
            "AGENTS.md",
            "README.md",
            "ai-handoff.md",
            "REPO_MAP.md",
            "rebuttal/README.md",
            "rebuttal/rebuttal_0723/README.md",
            "rebuttal/rebuttal_0723/00_REVIEWER_27BE_OFFICIAL_REVIEW.md",
            "rebuttal/pre_rebuttal/README.md",
            "rebuttal/pre_rebuttal/rebuttal_playbook.md",
            "docs/overview/RESULT_PROVENANCE_MANIFEST.md",
            "paper_experiments/MANIFEST.json",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_handoff_is_public_safe_and_points_to_authorities(self):
        text = (ROOT / "ai-handoff.md").read_text(encoding="utf-8")
        for required in (
            "AGENTS.md",
            "REPO_MAP.md",
            "rebuttal/README.md",
            "rebuttal/rebuttal_0723/README.md",
            "rebuttal/rebuttal_0723/00_REVIEWER_27BE_OFFICIAL_REVIEW.md",
            "docs/overview/RESULT_PROVENANCE_MANIFEST.md",
            "paper/main.pdf",
            "Known issues / current breakage",
        ):
            self.assertIn(required, text)

        forbidden_markers = (
            "/" + "Users" + "/",
            "/root/" + "autodl",
            "seeta" + "cloud",
            "ssh" + "pass",
        )
        lower = text.lower()
        for marker in forbidden_markers:
            self.assertNotIn(marker.lower(), lower)

    def test_only_one_rebuttal_control_room_exists(self):
        self.assertTrue((ROOT / "rebuttal").is_dir())
        self.assertTrue((ROOT / "rebuttal" / "pre_rebuttal").is_dir())
        self.assertTrue((ROOT / "rebuttal" / "rebuttal_0723").is_dir())
        self.assertFalse((ROOT / "rebuttal_7").exists())
        self.assertFalse((ROOT / "07 - rebuttal").exists())
        for retired_top_level in (
            "rebuttal_playbook.md",
            "FULL_PAPER_INTEGRITY_AUDIT_20260713.md",
            "REBUTTAL_MASTER_QUESTION_LEDGER_20260711.md",
            "evq_seed42_retrieval_repair",
            "frequency_adaptation_8b",
            "simulated_reviews",
        ):
            self.assertFalse(
                (ROOT / "rebuttal" / retired_top_level).exists(),
                retired_top_level,
            )

    def test_paper_has_one_root_pdf_and_no_retired_root_files(self):
        self.assertEqual(
            sorted(path.name for path in (ROOT / "paper").glob("*.pdf")),
            ["main.pdf"],
        )
        for relative in (
            "paper/EVQ-Cosh_NeurIPS2026.pdf",
            "paper/neurips_2025.sty",
            "paper/REBUTTAL_PLAYBOOK.md",
            "paper/REVIEW_PROMPT.md",
            "paper/CITATION_AUDIT_REPORT.md",
            "paper/build_aidemo",
            "paper/build_tectonic",
            "paper/figs/unused",
        ):
            self.assertFalse((ROOT / relative).exists(), relative)

    def test_root_has_no_stale_provenance_duplicate(self):
        self.assertFalse((ROOT / "RESULT_PROVENANCE_MANIFEST.md").exists())
        self.assertTrue(
            (ROOT / "docs/overview/RESULT_PROVENANCE_MANIFEST.md").is_file()
        )

    def test_primary_curated_assets_are_present(self):
        curated = ROOT / "data/curated"
        for name in (
            "primary1_evq_yarn_10pct_raw.json",
            "fig3_extreme_128.json",
            "eval_3seeds_full_results.json",
            "table18_mla_3seed_aggregate.json",
            "phase16_99run_manifest.csv",
        ):
            self.assertTrue((curated / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
