#!/usr/bin/env python3
"""Report paired Kanana 128K InfiniteBench English-QA for TailSpline and YaRN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report_qa_three_arm import atomic_json, load_arm, outcomes, sha256, summarize


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--tailspline", type=Path, required=True)
    parser.add_argument("--official-yarn", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    panel = read_jsonl(args.panel)
    if len(panel) != 118 or len({row["source_cluster_id"] for row in panel}) != 23:
        raise ValueError("Kanana QA128K panel identity drift")
    paths = {"tailspline": args.tailspline, "official_yarn": args.official_yarn}
    rows = {arm: load_arm(path, panel) for arm, path in paths.items()}
    if set(rows["tailspline"]) != set(rows["official_yarn"]):
        raise ValueError("two arms are not paired on identical QA prompts")
    arms = {arm: summarize(values, panel) for arm, values in rows.items()}
    delta = arms["tailspline"]["qa_f1"] - arms["official_yarn"]["qa_f1"]
    report = {
        "status": "KANANA_128K_INFINITEBENCH_EN_QA_TWO_ARM_COMPLETE_V1",
        "model": "kakaocorp/kanana-1.5-8b-instruct-2505",
        "target_length": 131_072,
        "benchmark": "InfiniteBench longbook_qa_eng",
        "metric": "official English-QA token F1",
        "rows_per_arm": len(panel),
        "source_context_clusters": 23,
        "input_token_range": [
            min(int(row["input_tokens"]) for row in panel),
            max(int(row["input_tokens"]) for row in panel),
        ],
        "arms": arms,
        "delta_tailspline_minus_official_yarn": delta,
        "paired_row_outcomes": outcomes(rows["tailspline"], rows["official_yarn"]),
        "identity": {
            "panel_sha256": sha256(args.panel),
            "raw_sha256": {arm: sha256(path) for arm, path in paths.items()},
            "same_ordered_prompts": True,
            "complete_context_preserved": True,
            "mrpro_status": "USER_SKIPPED_BEFORE_START",
        },
    }
    atomic_json(args.out, report)
    print(json.dumps({
        "tailspline": arms["tailspline"]["qa_f1"],
        "official_yarn": arms["official_yarn"]["qa_f1"],
        "delta": delta,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
