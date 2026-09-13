#!/usr/bin/env python3
"""Compare new and archived Llama runner outputs on identical 64K rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.experiments.olmo_fast_screen.ruler_bench import score as official_score


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def compare(prepared_rows: list[dict], new_rows: list[dict], old_rows: list[dict]) -> dict:
    prepared = {row["row_id"]: row for row in prepared_rows}
    new = {row["row_id"]: row for row in new_rows}
    old = {row["row_id"]: row for row in old_rows}
    if len(new) != len(new_rows) or len(old) != len(old_rows) or len(prepared) != len(prepared_rows):
        raise ValueError("duplicate row ids")
    if not new or not set(new) <= set(old) or not set(new) <= set(prepared):
        raise ValueError("new canary rows are absent from archived or prepared rows")
    details = []
    for row_id in sorted(new):
        source, current, archived = prepared[row_id], new[row_id], old[row_id]
        for field in ("task", "length_cap", "references", "prompt_sha256"):
            if current[field] != source[field] or archived[field] != source[field]:
                raise ValueError(f"row identity differs at {field}: {row_id}")
        current_text = current.get("output_text", current.get("output", current.get("raw_text")))
        archived_text = archived.get("output", archived.get("raw_text"))
        current_score = official_score(source, current_text)
        archived_score = official_score(source, archived_text)
        details.append({
            "row_id": row_id,
            "task": source["task"],
            "exact_token_ids": current.get("generated_ids", current.get("output_ids")) == archived["output_ids"],
            "exact_text": current_text == archived_text,
            "same_official_score": current_score == archived_score,
            "new_official_score": current_score,
            "archived_official_score": archived_score,
        })
    exact_tokens = sum(row["exact_token_ids"] for row in details)
    exact_text = sum(row["exact_text"] for row in details)
    same_score = sum(row["same_official_score"] for row in details)
    if exact_tokens == len(details):
        status = "EXACT_RUNNER_PARITY"
    elif same_score == len(details):
        status = "SCORE_PARITY_ONLY"
    else:
        status = "RUNNER_PARITY_FAILED"
    return {
        "status": status,
        "rows": len(details),
        "exact_token_rows": exact_tokens,
        "exact_text_rows": exact_text,
        "same_official_score_rows": same_score,
        "details": details,
        "decision": "reuse archived full baselines only for EXACT_RUNNER_PARITY or an explicitly accepted score-only comparison",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True,
                        help="new-run directory containing generations.jsonl, or a runner JSONL file")
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    new_path = args.new_run if args.new_run.is_file() else args.new_run / "generations.jsonl"
    result = compare(
        read_rows(args.prepared),
        read_rows(new_path),
        read_rows(args.archived),
    )
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "rows", "exact_token_rows", "exact_text_rows", "same_official_score_rows"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
