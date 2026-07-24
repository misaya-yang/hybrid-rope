import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OVERVIEW = ROOT / "docs" / "overview"


def _read(path):
    return (ROOT / path).read_text(encoding="utf-8")


class Opus48AuditDocsTests(unittest.TestCase):
    def test_every_o48_issue_has_checklist_and_resolution_rows(self):
        expected = {f"O48-{i:02d}" for i in range(1, 26)}
        checklist = _read("docs/overview/OPUS48_REVIEW_AUDIT_CHECKLIST.md")
        ledger = _read("docs/overview/OPUS48_ISSUE_RESOLUTION_LEDGER.md")

        checklist_ids = set(re.findall(r"\|\s*(O48-\d{2})\s*\|", checklist))
        ledger_ids = set(re.findall(r"\|\s*(O48-\d{2})\s*\|", ledger))

        self.assertEqual(checklist_ids, expected)
        self.assertEqual(ledger_ids, expected)

    def test_resolution_labels_are_from_known_set(self):
        ledger = _read("docs/overview/OPUS48_ISSUE_RESOLUTION_LEDGER.md")
        allowed = {
            "Resolved for wording",
            "Resolved in code",
            "Evidence-gated",
            "Experiment-gated",
            "Concede/scope",
            "Resolved in code / Evidence-gated",
        }
        labels = re.findall(r"\|\s*O48-\d{2}\s*\|\s*([^|]+?)\s*\|", ledger)
        self.assertEqual(len(labels), 25)
        self.assertTrue(set(labels).issubset(allowed), sorted(set(labels) - allowed))

    def test_core_audit_markdown_links_exist(self):
        docs = [
            OVERVIEW / "README.md",
            OVERVIEW / "OPUS48_REBUTTAL_MASTER_BRIEF.md",
            OVERVIEW / "OPUS48_AUDIT_CONTROL_CENTER.md",
            OVERVIEW / "OPUS48_ARTIFACT_RECOVERY_RUNBOOK.md",
            OVERVIEW / "OPUS48_COMPLETION_AUDIT.md",
            OVERVIEW / "OPUS48_ISSUE_RESOLUTION_LEDGER.md",
            OVERVIEW / "OPUS48_REBUTTAL_RESPONSE_MATRIX.md",
            OVERVIEW / "PAPER_CLAIMS_MAP.md",
            OVERVIEW / "RESULT_PROVENANCE_MANIFEST.md",
        ]
        for doc in docs:
            text = doc.read_text(encoding="utf-8")
            missing = []
            for target in re.findall(r"`([^`]+\.md)`", text):
                if not (ROOT / target).exists():
                    missing.append(target)
            self.assertEqual(missing, [], f"missing markdown links in {doc}")

    def test_high_risk_old_phrases_do_not_reenter_paper_or_rebuttal(self):
        files = (
            list((ROOT / "paper").glob("**/*.tex"))
            + [
                ROOT / "rebuttal" / "pre_rebuttal" / "REVIEWER_TRIAGE_PLAYBOOK.md",
                OVERVIEW / "PAPER_CLAIMS_MAP.md",
            ]
        )
        text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in files)
        forbidden = [
            "d_eff=d_head=128",
            "best at every tested length",
            "PK is autoregressive",
            "The 1B run proves durability",
            "EVQ replaces YaRN",
            "No evidence of convergence between EVQ and Geo at any training duration",
            "No experiment has ever shown Geo+YaRN outperforming EVQ+YaRN",
            "✅ Low",
            "strictly stronger result",
            "132 unit tests",
            "derived from first principles",
            "EVQ exceeds the empirical bar",
            "will include this in the camera-ready version",
            "exactly what the optimal allocation should be",
            "production adoption is straightforward",
        ]
        for phrase in forbidden:
            self.assertNotIn(phrase, text)

    def test_result_provenance_manifest_hashes_are_current(self):
        manifest = _read("docs/overview/RESULT_PROVENANCE_MANIFEST.md")
        rows = re.findall(r"\| `([^`]+)` \| [^|]+ \| `([0-9a-f]{64})` \|", manifest)
        self.assertGreater(len(rows), 0)

        stale = []
        missing = []
        for path_s, expected in rows:
            path = ROOT / path_s
            if not path.exists():
                missing.append(path_s)
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                stale.append((path_s, expected, actual))

        self.assertEqual(missing, [])
        self.assertEqual(stale, [])


if __name__ == "__main__":
    unittest.main()
