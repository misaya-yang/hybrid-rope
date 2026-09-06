import hashlib
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
            "paper-2027/README.md",
            "paper-2027/NARRATIVE_GUIDE.md",
            "paper-2027/REVISION_BRIEF.md",
            "paper-2027/SUBMISSION_CHECKLIST.md",
            "paper-2027/research/README.md",
            "paper-2027/research/history/TIMELINE.md",
            "paper-2027/research/foundations/README.md",
            "paper-2027/research/evidence/README.md",
            "paper-2027/main.tex",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_routing_docs_are_public_safe_and_point_to_authorities(self):
        routing_docs = (
            ROOT / "README.md",
            ROOT / "INDEX.md",
            ROOT / "AGENTS.md",
            ROOT / "paper-2027" / "HANDOFF.md",
            ROOT / "paper-2027" / "README.md",
            ROOT / "paper-2027" / "NARRATIVE_GUIDE.md",
            ROOT / "paper-2027" / "REVISION_BRIEF.md",
            ROOT / "paper-2027" / "SUBMISSION_CHECKLIST.md",
            ROOT / "paper-2027" / "research" / "README.md",
            ROOT / "paper-2027" / "research" / "external-reviews" / "README.md",
            ROOT / "paper_experiments" / "README.md",
            ROOT / "docs" / "overview" / "README.md",
        )
        text = "\n".join(path.read_text(encoding="utf-8") for path in routing_docs)
        for required in (
            "AGENTS.md",
            "INDEX.md",
            "paper-2027/HANDOFF.md",
            "NARRATIVE_GUIDE.md",
            "REVISION_BRIEF.md",
            "EXACT_RANGE_151M_3SEED_RESULT_20260820.md",
            "FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md",
            "paper/main.pdf",
        ):
            self.assertIn(required, text)

        forbidden_markers = (
            "/" + "Users" + "/",
            "/" + "root" + "/",
            "/" + "home" + "/",
            "seeta" + "cloud",
            "ssh" + "pass",
            "BEGIN " + "OPENSSH PRIVATE KEY",
            "BEGIN " + "RSA PRIVATE KEY",
        )
        lower = text.lower()
        for marker in forbidden_markers:
            self.assertNotIn(marker.lower(), lower)
        self.assertIsNone(re.search(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)", text))
        self.assertIsNone(re.search(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16})\b", text))

    def test_only_one_rebuttal_control_room_exists(self):
        self.assertTrue((ROOT / "rebuttal").is_dir())
        self.assertTrue((ROOT / "rebuttal" / "pre_rebuttal").is_dir())
        self.assertTrue((ROOT / "rebuttal" / "rebuttal_0723").is_dir())
        for relative in (
            "rebuttal/rebuttal_0723/README.md",
            "rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md",
            "rebuttal/rebuttal_0723/01_REBUTTAL_PLAYBOOK.md",
            "rebuttal/rebuttal_0723/02_RESPONSE_QUESTIONS_AND_OUTCOMES.md",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)
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
        self.assertEqual(
            hashlib.sha256((ROOT / "paper" / "main.pdf").read_bytes()).hexdigest(),
            "fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772",
        )

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
        for required in (
            "Every new or amended owner",
            "supported and unsupported claims",
            "A correction is complete only when",
        ):
            self.assertIn(required, agents)

    def test_secondary_entrypoints_preserve_cold_start_order(self):
        for relative, needles in (
            (
                "paper-2027/README.md",
                ("../AGENTS.md", "../README.md", "HANDOFF.md", "../INDEX.md"),
            ),
            (
                "docs/overview/README.md",
                (
                    "../../AGENTS.md",
                    "../../README.md",
                    "../../paper-2027/HANDOFF.md",
                    "../../INDEX.md",
                ),
            ),
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            offsets = [text.index(needle) for needle in needles]
            self.assertEqual(offsets, sorted(offsets), relative)

    def test_machine_profiles_keep_aidemo_on_the_work_machine(self):
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        handoff = (ROOT / "paper-2027" / "HANDOFF.md").read_text(encoding="utf-8")
        self.assertIn("work machine", agents.lower())
        self.assertIn("low-configuration personal pc", agents.lower())
        self.assertIn("`aidemo`", agents)
        self.assertIn(
            "install or recreate the work-machine environment",
            " ".join(agents.split()),
        )
        self.assertNotIn("`aidemo`", readme)
        self.assertIn("documentation/planning host", handoff)
        self.assertIn("Do not\n  install or recreate", handoff)

    def test_default_cold_start_is_bounded_and_progressive(self):
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        handoff = (ROOT / "paper-2027" / "HANDOFF.md").read_text(encoding="utf-8")
        index = (ROOT / "INDEX.md").read_text(encoding="utf-8")
        self.assertLessEqual(
            sum(text.count("\n") + 1 for text in (agents, readme, handoff)),
            320,
        )
        self.assertIn("Do not read the timeline", readme)
        self.assertIn("Do not read every linked file", index)
        self.assertIn("Do not\n   batch-read", agents)

    def test_index_carries_the_closed_route_ledger(self):
        """The falsified-route table is the repository's anti-repetition gate."""
        index = (ROOT / "INDEX.md").read_text(encoding="utf-8")
        for required in (
            "KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md",
            "LEROPE_PROFILE_ORACLE_AUDIT_20260820.md",
            "RETROFIT_AXIS_FALSIFICATION_20260822.md",
            "DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md",
            "ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md",
            "ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md",
            "EXACT_RANGE_151M_3SEED_RESULT_20260820.md",
            "FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md",
        ):
            self.assertIn(required, index)
        self.assertIn("only this implementation is closed", index)

    def test_mature_result_router_prioritizes_paper_owners(self):
        router = (
            ROOT
            / "paper-2027/research/attention-aware-retrofit/results/README.md"
        ).read_text(encoding="utf-8")
        headings = (
            "## Paper-facing owners",
            "## Active single-table transport evidence",
            "## Correction-only 2026-09-02 materials",
        )
        offsets = [router.index(heading) for heading in headings]
        self.assertEqual(offsets, sorted(offsets))
        self.assertLess(
            router.index("SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md"),
            router.index("ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md"),
        )

    def test_static_rank_diagnostic_has_an_owner_script(self):
        """Computed internal numbers need an owner and an honest search scope."""
        owner = ROOT / "scripts" / "analysis" / "third_axis_ceiling.py"
        self.assertTrue(owner.is_file(), "third-axis static-rank owner is missing")
        index = (ROOT / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn("scripts/analysis/third_axis_ceiling.py", index)
        source = owner.read_text(encoding="utf-8")
        for required in ("LM-quality", "restarts", "algebraic", "not a global ceiling"):
            self.assertTrue(required in source, f"static-rank owner must state: {required}")

    def test_current_route_keeps_research_active_and_evidence_separate(self):
        index = (ROOT / "INDEX.md").read_text(encoding="utf-8")
        normalized_index = " ".join(index.split())
        current_routes = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                ROOT / "README.md",
                ROOT / "INDEX.md",
                ROOT / "paper-2027" / "HANDOFF.md",
                ROOT
                / "paper-2027/research/attention-aware-retrofit/theory/README.md",
            )
        )
        retired_preflight = (
            ROOT
            / "paper-2027/research/attention-aware-retrofit/preflights"
            / "zero-training-deployment"
            / "ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830.md"
        )
        audit = ROOT / "scripts/analysis/finite_k_cosh_regret_audit.py"
        self.assertTrue(retired_preflight.is_file())
        self.assertTrue(audit.is_file())
        for required in (
            "## 0. Current paper status",
            "fully frozen model-relative structured",
            "SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831",
            "## 5. What to do",
            "one static table/gain",
            "about `0.12` maximum damage",
            "untouched 8x/16x/32x capability",
            "no GPU run is active",
            "2026-09-17",
            "2026-09-18",
            "2026-09-25",
            "paper-2027/research/history/TIMELINE.md",
        ):
            self.assertIn(required, normalized_index)
        for retired in (
            "No new submission experiment is planned",
            "No GPU method-development experiment is currently active",
            "GPU method development is stopped",
        ):
            self.assertNotIn(retired, current_routes)
        source = audit.read_text(encoding="utf-8")
        for boundary in ("not r2", "LM loss", "table selector"):
            self.assertIn(boundary, source)
        text = retired_preflight.read_text(encoding="utf-8")
        for required in (
            "RETIRED 2026-08-31",
            "not a current protocol",
            "not a current protocol, runner",
        ):
            self.assertIn(required, text)
        self.assertFalse(
            (ROOT / "scripts/core_text_phases/optimize_static_z.py").exists()
        )

    def test_active_method_governance_uses_real_endpoints_and_hard_stops(self):
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        for required in (
            "only two routes",
            "end-to-end generated-task",
            "never capability selectors",
            "zero primary long-generation score",
            "Native damage",
            "not the method class",
            "Submission dates do not prohibit research",
        ):
            self.assertIn(required, agents)

    def test_invalid_band_restoration_is_not_routed_to_execution(self):
        preflight = (
            ROOT
            / "paper-2027/research/attention-aware-retrofit/preflights"
            / "zero-training-deployment"
            / "ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830.md"
        ).read_text(encoding="utf-8")
        archive = (
            ROOT
            / "paper-2027/research/attention-aware-retrofit/analysis"
            / "PROTECTED_RAMP_RIGOROUS_COMPOSITE_20260828.md"
        ).read_text(encoding="utf-8")
        for text in (preflight, archive):
            self.assertIn("invalid as written", text)
            self.assertIn("frequency", text)
            self.assertIn("ordering", text)
        self.assertIn("Do not execute", archive)

    def test_static_search_is_not_promoted_to_optimality_or_support_invariance(self):
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        index = (ROOT / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn("best-found value", agents)
        self.assertIn(
            "not a global or behavioural ceiling",
            " ".join(index.split()),
        )

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

    def test_external_model_reviews_are_archived_not_active_routing(self):
        router = (ROOT / "paper-2027" / "research" / "README.md").read_text(
            encoding="utf-8"
        )
        archive = (
            ROOT / "paper-2027" / "research" / "external-reviews" / "README.md"
        ).read_text(encoding="utf-8")
        self.assertIn("external-reviews/", router)
        self.assertIn("untrusted historical analysis", router)
        self.assertNotIn("## Reviewer objections", router)
        self.assertIn("Frozen audit archive", archive)
        self.assertIn("Current use", archive)

    def test_state_layer_does_not_restate_the_agenda(self):
        """Rules > index > state: the handoff routes the agenda, never owns it."""
        handoff = (ROOT / "paper-2027" / "HANDOFF.md").read_text(encoding="utf-8")
        self.assertIn("This file owns no scientific verdict", handoff)
        self.assertIn("## Latest changes", handoff)
        self.assertIn("[`../INDEX.md`](../INDEX.md)", handoff)
        for forbidden in (
            "omega_k",
            "z × adaptation",
            "W0/F1",
            "leave-one-band-out",
            "The only active research implementation step",
            "38-row constructed exact-length Hotpot",
            "T4 actual",
            "T5 torus",
            "T7 ratio",
        ):
            self.assertNotIn(forbidden, handoff)

    def test_index_distinguishes_document_update_from_evidence_cutoff(self):
        index = (ROOT / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn("**Updated:**", index)
        self.assertIn("**Evidence cut-off:**", index)

    def test_manuscript_cold_start_preserves_author_doctrine(self):
        narrative = (ROOT / "paper-2027" / "NARRATIVE_GUIDE.md").read_text(
            encoding="utf-8"
        )
        for required in (
            "## Non-negotiable author doctrine",
            "## Claims we make—and questions the paper does not need to answer",
            "## Future-session stop rules",
            "every arbitrary non-geometric $z$",
            "Treat controls as scientific instruments, not opponents",
            "If it cannot answer all five",
        ):
            self.assertIn(required, narrative)

        index = (ROOT / "INDEX.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        brief = (ROOT / "paper-2027" / "REVISION_BRIEF.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("## 5. What to do", index)
        for required in (
            "under-studied coordinate",
            "## Evidence hierarchy",
            "fully frozen zero-training",
            "matched low-rank adaptation",
            "from-training/co-adaptation",
            "2026-09-17",
            "2026-09-18",
            "2026-09-25",
        ):
            self.assertIn(required, readme)
        self.assertNotIn(
            "下一项有决策价值的研究协议只有 **matched-content phase 2x2**",
            index,
        )
        self.assertIn("does not prohibit new training", brief)
        self.assertIn("## 5. Active single-table research programme", brief)
        self.assertIn("one global request-static table/gain", brief)

    def test_root_routing_links_resolve(self):
        pattern = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
        for doc in (
            ROOT / "README.md",
            ROOT / "AGENTS.md",
            ROOT / "INDEX.md",
            ROOT / "paper-2027" / "HANDOFF.md",
            ROOT / "paper-2027" / "README.md",
            ROOT / "paper-2027" / "NARRATIVE_GUIDE.md",
            ROOT / "paper-2027" / "REVISION_BRIEF.md",
            ROOT / "paper-2027" / "SUBMISSION_CHECKLIST.md",
            ROOT / "paper-2027" / "research" / "README.md",
            ROOT / "paper-2027" / "research" / "foundations" / "README.md",
            ROOT / "paper-2027" / "research" / "evidence" / "README.md",
            ROOT / "paper-2027" / "research" / "archive" / "README.md",
            ROOT / "paper-2027" / "research" / "history" / "TIMELINE.md",
            ROOT / "paper-2027" / "research" / "external-reviews" / "README.md",
            ROOT / "paper_experiments" / "README.md",
            ROOT / "docs" / "README.md",
            ROOT / "docs" / "overview" / "README.md",
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

    def test_historical_reports_are_grouped_by_month(self):
        exp = ROOT / "docs" / "exp"
        for month in ("2026-02", "2026-03", "2026-04", "2026-07"):
            self.assertTrue((exp / month).is_dir(), month)
        self.assertEqual(list(exp.glob("2026-??-*.md")), [])
        for report in exp.glob("2026-??/*.md"):
            self.assertTrue(report.name.startswith(report.parent.name + "-"), report)

    def test_research_root_separates_foundations_evidence_and_history(self):
        research = ROOT / "paper-2027" / "research"
        for relative in (
            "foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md",
            "evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md",
            "history/TIMELINE.md",
            "archive/2026-08/CODEX_CLAUDE_PAPER_REVIEW_LOG.md",
        ):
            self.assertTrue((research / relative).is_file(), relative)
        for retired_flat in (
            "FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md",
            "EXACT_RANGE_151M_3SEED_RESULT_20260820.md",
            "CODEX_CLAUDE_PAPER_REVIEW_LOG.md",
        ):
            self.assertFalse((research / retired_flat).exists(), retired_flat)

    def test_root_has_no_stale_provenance_duplicate(self):
        self.assertFalse((ROOT / "RESULT_PROVENANCE_MANIFEST.md").exists())
        manifest = ROOT / "docs/overview/RESULT_PROVENANCE_MANIFEST.md"
        overview = ROOT / "docs/overview/README.md"
        self.assertTrue(manifest.is_file())
        overview_hash = hashlib.sha256(overview.read_bytes()).hexdigest()
        self.assertIn(overview_hash, manifest.read_text(encoding="utf-8"))

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
