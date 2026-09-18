#!/usr/bin/env python3
"""Audit confirmed key-value misbinding transitions in completed T/C panels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from transformers import AutoTokenizer


TASKS = ("niah_multikey_1", "niah_multikey_2", "niah_multikey_3")
RECORD = re.compile(r"One of the special magic numbers for ([^\n]+?) is: (\d+)\.")
QUESTION = re.compile(r"What is the special magic number for (.+?) mentioned in the provided text\?")
NUMBER = re.compile(r"(?<!\d)\d{7}(?!\d)")
UUID_RECORD = re.compile(r"One of the special magic uuids for ([0-9a-f-]{36}) is: ([0-9a-f-]{36})\.")
UUID_QUESTION = re.compile(r"What is the special magic uuid for ([0-9a-f-]{36}) mentioned in the provided text\?")
UUID = re.compile(r"(?<![0-9a-f-])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-f-])")


def read_jsonl(path: Path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_outputs(paths: list[Path]) -> dict[str, dict]:
    rows = {}
    for path in paths:
        for row in read_jsonl(path):
            if row.get("task") not in TASKS:
                continue
            row_id = row["row_id"]
            if row_id in rows:
                raise ValueError(f"duplicate output row: {row_id}")
            rows[row_id] = row
    return rows


def find_subsequences(haystack: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return []
    first = needle[0]
    stop = len(haystack) - len(needle) + 1
    positions = []
    for index in range(stop):
        if haystack[index] == first and haystack[index : index + len(needle)] == needle:
            positions.append(index)
    return positions


def parse_prompt(row: dict, tokenizer) -> dict:
    prompt_ids = list(row["prompt_ids"])
    text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    pairs = RECORD.findall(text) + UUID_RECORD.findall(text)
    mapping = {}
    duplicate_keys = set()
    for key, value in pairs:
        if key in mapping and mapping[key] != value:
            duplicate_keys.add(key)
        mapping[key] = value
    questions = QUESTION.findall(text) + UUID_QUESTION.findall(text)
    target = questions[-1] if questions else None
    references = [str(value) for value in row.get("references") or []]
    gold = references[0] if len(references) == 1 else None
    mapped_gold = mapping.get(target) if target is not None else None
    values = list(mapping.values())
    source_ambiguous = (
        target is None
        or target in duplicate_keys
        or gold is None
        or mapped_gold != gold
        or len(values) != len(set(values))
    )
    evidence_index = None
    if not source_ambiguous:
        # The RULER prompt tokenizer can fuse the sentence-final period with
        # the following newline, so a separately tokenized full sentence is
        # not guaranteed to be a literal subsequence.  The seven-digit gold
        # value is unique by the source-qualified contract and gives a stable
        # evidence-token anchor.
        gold_ids = tokenizer.encode(gold, add_special_tokens=False)
        positions = find_subsequences(prompt_ids, gold_ids)
        if len(positions) == 1:
            evidence_index = positions[0]
    return {
        "gold": gold,
        "candidate_values": sorted(set(values)),
        "candidate_count": len(values),
        "source_ambiguous": source_ambiguous,
        "evidence_index": evidence_index,
        "query_index": len(prompt_ids),
        "dependency_distance": None if evidence_index is None else len(prompt_ids) - evidence_index,
    }


def classify(text: str, prompt: dict) -> str:
    if not str(text).strip():
        return "empty"
    candidates = set(prompt["candidate_values"])
    normalized = str(text).lower()
    mentioned = {
        value for value in NUMBER.findall(normalized) + UUID.findall(normalized)
        if value in candidates
    }
    if len(mentioned) > 1:
        return "ambiguous"
    if len(mentioned) == 1:
        value = next(iter(mentioned))
        return "correct" if value == prompt["gold"] else "wrong_binding"
    return "other"


def summarize(rows: list[dict]) -> dict:
    transitions = Counter((row["c_class"], row["t_class"]) for row in rows)
    distances = [row["dependency_distance"] for row in rows if row["dependency_distance"] is not None]
    classes = ("correct", "wrong_binding", "other", "ambiguous", "empty")
    matrix = {
        left: {right: transitions[(left, right)] for right in classes}
        for left in classes
    }
    repairs = transitions[("wrong_binding", "correct")]
    damages = transitions[("correct", "wrong_binding")]
    return {
        "rows": len(rows),
        "tailspline_classes": dict(Counter(row["t_class"] for row in rows)),
        "control_classes": dict(Counter(row["c_class"] for row in rows)),
        "transition_matrix_control_to_tailspline": matrix,
        "confirmed_wrong_binding_repairs": repairs,
        "confirmed_wrong_binding_damages": damages,
        "net_confirmed_wrong_binding_correction": repairs - damages,
        "net_correction_rate_all_rows": (repairs - damages) / len(rows) if rows else None,
        "tailspline_unique_correct": sum(row["t_class"] == "correct" and row["c_class"] != "correct" for row in rows),
        "control_unique_correct": sum(row["c_class"] == "correct" and row["t_class"] != "correct" for row in rows),
        "both_correct": sum(row["c_class"] == row["t_class"] == "correct" for row in rows),
        "both_incorrect": sum(row["c_class"] != "correct" and row["t_class"] != "correct" for row in rows),
        "mean_dependency_distance": sum(distances) / len(distances) if distances else None,
        "distance_anchor_coverage": len(distances) / len(rows) if rows else None,
        "mean_candidate_count": sum(row["candidate_count"] for row in rows) / len(rows) if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel", action="append", type=Path, required=True)
    parser.add_argument("--tailspline", action="append", type=Path, required=True)
    parser.add_argument("--control", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tailspline = load_outputs(args.tailspline)
    control = load_outputs(args.control)
    if set(tailspline) != set(control):
        raise ValueError("T/C output row IDs differ")

    paired = {}
    panel_seen = set()
    ambiguous_sources = []
    missing_outputs = []
    for panel_path in args.panel:
        for source in read_jsonl(panel_path):
            if source.get("task") not in TASKS:
                continue
            row_id = source["row_id"]
            if row_id in panel_seen:
                raise ValueError(f"duplicate panel row: {row_id}")
            panel_seen.add(row_id)
            if row_id not in tailspline or row_id not in control:
                missing_outputs.append(row_id)
                continue
            if tailspline[row_id]["prompt_sha256"] != source["prompt_sha256"] or control[row_id]["prompt_sha256"] != source["prompt_sha256"]:
                raise ValueError(f"prompt identity mismatch: {row_id}")
            parsed = parse_prompt(source, tokenizer)
            if parsed["source_ambiguous"]:
                ambiguous_sources.append(row_id)
                continue
            length = int(source["length_cap"])
            paired[row_id] = {
                "row_id": row_id,
                "task": source["task"],
                "length": length,
                "t_class": classify(tailspline[row_id].get("output_text", ""), parsed),
                "c_class": classify(control[row_id].get("output_text", ""), parsed),
                "t_eos": bool(tailspline[row_id].get("ended_eos")),
                "c_eos": bool(control[row_id].get("ended_eos")),
                "t_cap": bool(tailspline[row_id].get("hit_cap")),
                "c_cap": bool(control[row_id].get("hit_cap")),
                "dependency_distance": None if parsed["dependency_distance"] is None else int(parsed["dependency_distance"]),
                "candidate_count": int(parsed["candidate_count"]),
            }

    expected = set(tailspline)
    coverage = len(paired) / len(expected) if expected else 0.0
    distance_coverage = sum(row["dependency_distance"] is not None for row in paired.values()) / len(paired) if paired else 0.0
    groups = defaultdict(list)
    for row in paired.values():
        groups[(row["length"], row["task"])].append(row)
    by_length_task = {
        str(length): {task: summarize(groups[(length, task)]) for task in TASKS}
        for length in sorted({key[0] for key in groups})
    }
    by_length = {
        str(length): summarize([row for row in paired.values() if row["length"] == length])
        for length in sorted({row["length"] for row in paired.values()})
    }

    s32 = by_length_task.get("32768", {})
    gate = {
        "mapping_coverage_at_least_99pct": coverage >= 0.99,
        "aggregate_32k_net_positive": by_length.get("32768", {}).get("net_confirmed_wrong_binding_correction", 0) > 0,
        "multikey2_32k_nonnegative": s32.get("niah_multikey_2", {}).get("net_confirmed_wrong_binding_correction", -1) >= 0,
        "multikey3_32k_nonnegative": s32.get("niah_multikey_3", {}).get("net_confirmed_wrong_binding_correction", -1) >= 0,
        "multikey2_or_3_32k_positive": max(
            s32.get("niah_multikey_2", {}).get("net_confirmed_wrong_binding_correction", 0),
            s32.get("niah_multikey_3", {}).get("net_confirmed_wrong_binding_correction", 0),
        ) > 0,
    }
    qualified = all(gate.values())
    report = {
        "status": "TC_EXISTING_MULTIKEY_BEHAVIOR_AUDIT_COMPLETE_V1",
        "rows_expected": len(expected),
        "rows_qualified": len(paired),
        "mapping_coverage": coverage,
        "distance_anchor_coverage": distance_coverage,
        "ambiguous_source_rows": ambiguous_sources,
        "missing_output_rows": missing_outputs,
        "by_length": by_length,
        "by_length_task": by_length_task,
        "gpu_distance_competition_gate": gate,
        "gpu_stage_qualified": qualified,
        "classification_contract": {
            "correct": "exactly one prompt candidate value is mentioned and it is the gold value",
            "wrong_binding": "exactly one prompt candidate value is mentioned and it belongs to another key",
            "ambiguous": "multiple distinct prompt candidate values are mentioned",
            "other": "no prompt candidate value is mentioned in a non-empty output",
            "empty": "empty decoded output",
            "denominator": "all source-qualified paired multikey rows",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({
        "status": report["status"],
        "rows": len(paired),
        "coverage": coverage,
        "by_length_net": {length: value["net_confirmed_wrong_binding_correction"] for length, value in by_length.items()},
        "gpu_stage_qualified": qualified,
        "gate": gate,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
