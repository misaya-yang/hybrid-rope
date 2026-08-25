import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryNavigationTests(unittest.TestCase):
    def test_canonical_navigation_files_exist(self):
        for relative in (
            "AGENTS.md",
            "README.md",
            "INDEX.md",
            "paper-2027/HANDOFF.md",
            "rebuttal/README.md",
            "rebuttal/rebuttal_0723/README.md",
            "rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md",
            "rebuttal/rebuttal_0723/01_REBUTTAL_PLAYBOOK.md",
            "rebuttal/rebuttal_0723/02_RESPONSE_QUESTIONS_AND_OUTCOMES.md",
            "rebuttal/rebuttal_0723/theory_results/EVQ_COSH_REBUTTAL_PRINCIPLES.md",
            "rebuttal/rebuttal_0723/theory_results/REVIEWER_USABLE_EVIDENCE_LEDGER_20260726.md",
            "rebuttal/pre_rebuttal/README.md",
            "rebuttal/pre_rebuttal/seed42_lora_eval_20260713/REPORT.md",
            "docs/overview/RESULT_PROVENANCE_MANIFEST.md",
            "paper_experiments/MANIFEST.json",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_routing_docs_are_public_safe_and_point_to_authorities(self):
        routing_docs = (
            ROOT / "README.md",
            ROOT / "INDEX.md",
            ROOT / "AGENTS.md",
            ROOT / "rebuttal" / "README.md",
            ROOT / "rebuttal" / "rebuttal_0723" / "README.md",
        )
        text = "\n".join(path.read_text(encoding="utf-8") for path in routing_docs)
        for required in (
            "AGENTS.md",
            "INDEX.md",
            "paper-2027/HANDOFF.md",
            "rebuttal/rebuttal_0723/README.md",
            "rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md",
            "rebuttal/rebuttal_0723/01_REBUTTAL_PLAYBOOK.md",
            "docs/overview/RESULT_PROVENANCE_MANIFEST.md",
            "paper/main.pdf",
            "main_0726",
        ):
            self.assertIn(required, text)

        forbidden_markers = (
            "/" + "Users" + "/",
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
        for retired_pre_rebuttal in (
            "rebuttal_playbook.md",
            "REVIEWER_TRIAGE_PLAYBOOK.md",
            "REBUTTAL_MASTER_QUESTION_LEDGER_20260711.md",
            "simulated_reviews",
        ):
            self.assertFalse(
                (ROOT / "rebuttal" / "pre_rebuttal" / retired_pre_rebuttal).exists(),
                retired_pre_rebuttal,
            )

    def test_restored_paper_archive_is_present_without_build_directories(self):
        self.assertEqual(
            sorted(path.name for path in (ROOT / "paper").glob("*.pdf")),
            ["main.pdf"],
        )
        for relative in (
            "paper/main.pdf",
            "paper/neurips_2025.sty",
            "paper/REBUTTAL_PLAYBOOK.md",
            "paper/REVIEW_PROMPT.md",
            "paper/CITATION_AUDIT_REPORT.md",
            "paper/figs/unused",
        ):
            self.assertTrue((ROOT / relative).exists(), relative)
        for relative in ("paper/build_aidemo", "paper/build_tectonic"):
            self.assertFalse((ROOT / relative).exists(), relative)

    def test_exactly_three_navigation_authorities(self):
        """Rules / index / state. A fourth root authority is a defect."""
        for required in ("AGENTS.md", "INDEX.md", "paper-2027/HANDOFF.md"):
            self.assertTrue((ROOT / required).is_file(), required)
        for retired in (
            "REPO_MAP.md",
            "Agent.md",
            "HANDOFF.md",
            "ROADMAP.md",
            "docs/INDEX.md",
            "docs/REPO_MAP.md",
            "docs/HANDOFF.md",
            "docs/archive/INDEX.md",
            "paper-2027/INDEX.md",
            "paper-2027/REPO_MAP.md",
        ):
            self.assertFalse((ROOT / retired).exists(), retired)

    def test_agents_defers_navigation_and_agenda_to_index(self):
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        for required in ("INDEX.md", "paper-2027/HANDOFF.md"):
            self.assertIn(required, agents)

    def test_index_carries_the_closed_route_ledger(self):
        """The falsified-route table is the repository's anti-repetition gate."""
        index = (ROOT / "INDEX.md").read_text(encoding="utf-8")
        for required in (
            "KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md",
            "LEROPE_PROFILE_ORACLE_AUDIT_20260820.md",
            "RETROFIT_AXIS_FALSIFICATION_20260822.md",
            "DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md",
            "ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md",
            "EXACT_RANGE_151M_3SEED_RESULT_20260820.md",
            "FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md",
        ):
            self.assertIn(required, index)

    def test_static_rank_diagnostic_has_an_owner_script(self):
        """Computed internal numbers need an owner and an honest search scope."""
        owner = ROOT / "scripts" / "analysis" / "third_axis_ceiling.py"
        self.assertTrue(owner.is_file(), "third-axis static-rank owner is missing")
        index = (ROOT / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn("scripts/analysis/third_axis_ceiling.py", index)
        source = owner.read_text(encoding="utf-8")
        for required in ("LM-quality", "restarts", "algebraic", "not a global ceiling"):
            self.assertTrue(required in source, f"static-rank owner must state: {required}")

    def test_static_search_is_not_promoted_to_optimality_or_support_invariance(self):
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        index = (ROOT / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn("best-found value", agents)
        self.assertIn("不建立 support invariance", index)
        self.assertNotIn("上限只由 $(K,L)$ 决定", index)

    def test_m4_screen_owner_uses_locked_identity_and_verdict(self):
        extended = (
            ROOT
            / "paper-2027/research/attention-aware-retrofit/results/PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md"
        ).read_text(encoding="utf-8")
        initial = (
            ROOT
            / "paper-2027/research/attention-aware-retrofit/results/PHASE_ISOTROPY_50M_M4_RESULT_20260824.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Canonical verdict: **SCREEN_UNRESOLVED**", extended)
        self.assertIn("Geo (raw key `FMRoPE`)", extended)
        self.assertIn("Canonical verdict: **SCREEN_UNRESOLVED**", initial)

    def test_reviewer_objection_ledger_exists_and_is_internal(self):
        router = ROOT / "paper-2027" / "research" / "README.md"
        text = router.read_text(encoding="utf-8")
        self.assertIn("## Reviewer objections", text)
        for tag in ("| R1 |", "| R2 |", "| R3 |", "| R4 |", "| R5 |", "| R6 |", "| R7 |"):
            self.assertIn(tag, text)
        # The ledger is an internal adversarial artifact, never manuscript text.
        for section in (ROOT / "paper-2027" / "sections").glob("*.tex"):
            self.assertNotIn("Reviewer objections", section.read_text(encoding="utf-8"))

    def test_state_layer_does_not_restate_the_agenda(self):
        """Rules > index > state: the handoff routes the agenda, never owns it."""
        handoff = (ROOT / "paper-2027" / "HANDOFF.md").read_text(encoding="utf-8")
        self.assertIn("The research agenda is not state", handoff)
        self.assertNotIn("The only active research implementation step", handoff)

    def test_root_routing_links_resolve(self):
        pattern = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
        for doc in (
            ROOT / "README.md",
            ROOT / "INDEX.md",
            ROOT / "docs" / "README.md",
            ROOT / "docs" / "tau_algor" / "README.md",
            ROOT / "docs" / "archive" / "README.md",
        ):
            for target in pattern.findall(doc.read_text(encoding="utf-8")):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path = target.split("#")[0]
                if not path:
                    continue
                self.assertTrue(
                    (doc.parent / path).exists(),
                    f"{doc.relative_to(ROOT)} -> {target}",
                )

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
