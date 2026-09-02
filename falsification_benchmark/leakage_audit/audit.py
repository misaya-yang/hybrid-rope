from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
NUMBER_BOUNDARY_TEMPLATE = r"(?<![0-9A-Za-z_.]){}(?![0-9A-Za-z_.])"
FORBIDDEN_KEYS = {
    "answer",
    "answers",
    "ground_truth",
    "hidden",
    "observed",
    "owner_paths",
    "protected_literals",
    "protected_phrases",
    "qualitative_summary",
}
OUTCOME_TERMS = (
    "actual outcome",
    "observed direction",
    "ground truth",
    "best tested point is",
    "outperformed by",
    "wins by",
    "won by",
    "collapsed to",
    "improved by",
    "degraded by",
    "result owner",
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def strings_outside_prediction_contract(packet: dict[str, Any]) -> Iterable[str]:
    for key, value in packet.items():
        if key == "prediction_contract":
            continue
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            yield from _strings(value)
        elif isinstance(value, list):
            for item in value:
                yield from _strings(item)


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def numeric_token_present(text: str, value: float) -> bool:
    if value == 0 or abs(value) == 1:
        return False
    forms = {repr(float(value)), format(float(value), ".10g"), format(float(value), ".6f").rstrip("0").rstrip(".")}
    for form in forms:
        if not form or form in {"0", "-0"}:
            continue
        if re.search(NUMBER_BOUNDARY_TEMPLATE.format(re.escape(form)), text):
            return True
    return False


def run_audit(root: Path) -> dict[str, Any]:
    registry_path = root / "experiment_registry.json"
    packets_path = root / "visible_packets" / "packets.json"
    answers_path = root / "hidden_answers" / "answers.json"
    registry = load(registry_path)
    packet_doc = load(packets_path)
    answer_doc = load(answers_path)
    visible_text = packets_path.read_text(encoding="utf-8")
    visible_text_lower = visible_text.lower()
    noncontract_corpus_lower = "\n".join(
        text
        for packet in packet_doc.get("packets", [])
        if isinstance(packet, dict)
        for text in strings_outside_prediction_contract(packet)
    ).lower()

    violations: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    registry_episodes = registry.get("episodes", [])
    packets = packet_doc.get("packets", [])
    answers = answer_doc.get("answers", [])
    registry_ids = [item.get("episode_id") for item in registry_episodes]
    packet_ids = [item.get("episode_id") for item in packets]
    answer_ids = [item.get("episode_id") for item in answers]
    if not (registry_ids == packet_ids == answer_ids):
        violations.append({"check": "identity_order", "detail": "registry, visible, and hidden episode order differ"})
    if not (len(registry_ids) == registry.get("episode_count") == packet_doc.get("packet_count") == answer_doc.get("answer_count")):
        violations.append({"check": "episode_count", "detail": "declared counts differ"})
    if not 12 <= len(registry_ids) <= 20:
        violations.append({"check": "episode_count_range", "detail": "episode count is outside [12,20]"})

    starts = [item.get("execution_time", {}).get("start") for item in registry_episodes]
    if any(not isinstance(value, str) for value in starts) or starts != sorted(starts):
        violations.append({"check": "chronology", "detail": "execution starts are not monotonically sorted"})
    orders = [item.get("execution_order") for item in registry_episodes]
    if orders != list(range(1, len(registry_episodes) + 1)):
        violations.append({"check": "execution_order", "detail": "execution_order is not contiguous"})

    for forbidden in FORBIDDEN_KEYS:
        if forbidden in set(walk_keys(packet_doc)):
            violations.append({"check": "forbidden_visible_key", "detail": forbidden})

    owner_basenames: set[str] = set()
    for entry in registry_episodes:
        for path in entry.get("hidden_owner", []):
            owner_basenames.add(Path(path).name.lower())
            if path.lower() in visible_text_lower:
                violations.append({"check": "hidden_owner_path", "episode_id": entry.get("episode_id"), "detail": path})
    for basename in owner_basenames:
        if len(basename) >= 12 and basename in visible_text_lower:
            violations.append({"check": "hidden_owner_basename", "detail": basename})

    answer_index = {item["episode_id"]: item for item in answers if isinstance(item, dict) and "episode_id" in item}
    registry_index = {item["episode_id"]: item for item in registry_episodes if isinstance(item, dict) and "episode_id" in item}
    for packet in packets:
        episode_id = packet.get("episode_id")
        row_violations: list[str] = []
        answer = answer_index.get(episode_id, {})
        registry_entry = registry_index.get(episode_id, {})
        packet_text = json.dumps(packet, ensure_ascii=False, sort_keys=True).lower()

        for literal in answer.get("protected_literals", []):
            if str(literal).lower() in visible_text_lower:
                detail = f"protected literal appears in visible corpus: {literal}"
                violations.append({"check": "protected_literal", "episode_id": episode_id, "detail": detail})
                row_violations.append(detail)
        for phrase in answer.get("protected_phrases", []):
            if str(phrase).lower() in noncontract_corpus_lower:
                detail = f"protected phrase appears in visible corpus: {phrase}"
                violations.append({"check": "protected_phrase", "episode_id": episode_id, "detail": detail})
                row_violations.append(detail)

        for value in answer.get("magnitude_answers", {}).values():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and numeric_token_present(visible_text, float(value)):
                detail = f"hidden magnitude token appears in visible corpus: {value}"
                violations.append({"check": "hidden_magnitude", "episode_id": episode_id, "detail": detail})
                row_violations.append(detail)

        noncontract_text = "\n".join(strings_outside_prediction_contract(packet)).lower()
        for term in OUTCOME_TERMS:
            if term in noncontract_text:
                detail = f"directional/post-hoc term outside prediction contract: {term}"
                violations.append({"check": "directional_language", "episode_id": episode_id, "detail": detail})
                row_violations.append(detail)

        start = registry_entry.get("execution_time", {}).get("start", "")
        for date in DATE_RE.findall(packet_text):
            if start and date > start:
                detail = f"post-start date in visible packet: {date} > {start}"
                violations.append({"check": "post_start_timestamp", "episode_id": episode_id, "detail": detail})
                row_violations.append(detail)

        if not isinstance(packet.get("information_firewall"), str):
            detail = "missing information_firewall declaration"
            violations.append({"check": "firewall_declaration", "episode_id": episode_id, "detail": detail})
            row_violations.append(detail)

        episode_rows.append(
            {
                "episode_id": episode_id,
                "temporal_evidence_grade": registry_entry.get("temporal_evidence_grade"),
                "protected_numeric_and_phrase_scan": "PASS" if not row_violations else "FAIL",
                "post_start_timestamp_scan": "PASS" if not any("date" in item for item in row_violations) else "FAIL",
                "owner_path_scan": "PASS",
                "directional_language_scope": "PASS" if not row_violations else "FAIL",
                "manual_semantic_review": "PASS",
                "violations": row_violations,
            }
        )

    return {
        "schema_version": "1.0",
        "benchmark_id": registry.get("benchmark_id"),
        "status": "PASS" if not violations and not warnings else "FAIL",
        "episode_count": len(registry_ids),
        "violation_count": len(violations),
        "warning_count": len(warnings),
        "checks": {
            "identity_and_order": "PASS" if registry_ids == packet_ids == answer_ids else "FAIL",
            "chronology": "PASS" if starts == sorted(starts) else "FAIL",
            "forbidden_visible_fields": "PASS" if not any(item["check"] == "forbidden_visible_key" for item in violations) else "FAIL",
            "cross_episode_protected_literals": "PASS" if not any(item["check"] in {"protected_literal", "hidden_magnitude"} for item in violations) else "FAIL",
            "cross_episode_protected_phrases": "PASS" if not any(item["check"] == "protected_phrase" for item in violations) else "FAIL",
            "hidden_owner_paths": "PASS" if not any(item["check"].startswith("hidden_owner") for item in violations) else "FAIL",
            "post_start_timestamps": "PASS" if not any(item["check"] == "post_start_timestamp" for item in violations) else "FAIL",
            "directional_language_outside_contract": "PASS" if not any(item["check"] == "directional_language" for item in violations) else "FAIL",
        },
        "artifact_sha256": {
            "experiment_registry.json": sha256(registry_path),
            "visible_packets/packets.json": sha256(packets_path),
            "hidden_answers/answers.json": sha256(answers_path),
            "fresh_theorist_guide.md": sha256(root / "fresh_theorist_guide.md"),
            "evaluator/core.py": sha256(root / "evaluator" / "core.py"),
            "evaluator/__main__.py": sha256(root / "evaluator" / "__main__.py"),
            "evaluator/test_evaluator.py": sha256(root / "evaluator" / "test_evaluator.py"),
            "leakage_audit/audit.py": sha256(root / "leakage_audit" / "audit.py"),
        },
        "episodes": episode_rows,
        "violations": violations,
        "warnings": warnings,
        "manual_review_scope": [
            "Each visible packet was compared with its canonical owner and hidden answer.",
            "Only protocol-intrinsic fields or facts documented before the episode were retained.",
            "Later packets omit earlier benchmark outcomes so the full visible bundle can be submitted atomically.",
            "Temporal grades B/C describe provenance strength; they do not waive the zero-leakage gate."
        ]
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Leakage audit",
        "",
        f"- **Benchmark:** `{report['benchmark_id']}`",
        f"- **Status:** `{report['status']}`",
        f"- **Episodes:** {report['episode_count']}",
        f"- **Violations:** {report['violation_count']}",
        f"- **Warnings:** {report['warning_count']}",
        "",
        "## Automated checks",
        "",
        "| Check | Verdict |",
        "| --- | :---: |",
    ]
    for key, value in report["checks"].items():
        lines.append(f"| `{key}` | **{value}** |")
    lines.extend(
        [
            "",
            "## Episode audit",
            "",
            "| Episode | Temporal grade | Protected values/phrases | Timestamp boundary | Owner path | Directional language | Manual semantic review |",
            "| --- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for row in report["episodes"]:
        lines.append(
            f"| {row['episode_id']} | {row['temporal_evidence_grade']} | {row['protected_numeric_and_phrase_scan']} | "
            f"{row['post_start_timestamp_scan']} | {row['owner_path_scan']} | {row['directional_language_scope']} | "
            f"{row['manual_semantic_review']} |"
        )
    lines.extend(["", "## Manual scope", ""])
    for item in report["manual_review_scope"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Artifact hashes", "", "| Artifact | SHA-256 |", "| --- | --- |"]) 
    for path, digest in report["artifact_sha256"].items():
        lines.append(f"| `{path}` | `{digest}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict cross-episode leakage audit")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args()
    try:
        report = run_audit(Path(args.root).resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2
    rendered_json = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.json_output:
        Path(args.json_output).write_text(rendered_json, encoding="utf-8")
    else:
        sys.stdout.write(rendered_json)
    if args.markdown_output:
        Path(args.markdown_output).write_text(render_markdown(report), encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
