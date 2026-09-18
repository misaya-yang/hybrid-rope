#!/usr/bin/env python3
"""Report paired Kanana 64K InfiniteBench English-QA for three static tables."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from experiments.iclr2027_strong_evidence_20260915.run_natural_long import (
    score_natural_output,
)


ARMS = ("tailspline", "official_yarn", "mrpro")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_arm(path: Path, panel: list[dict]) -> dict[str, dict]:
    values = read_jsonl(path)
    if len(values) != len(panel):
        raise ValueError(f"generation count differs from panel: {path}")
    result = {}
    for expected, actual in zip(panel, values):
        for key in ("row_id", "task", "prompt_sha256", "references"):
            if actual.get(key) != expected.get(key):
                raise ValueError(f"input identity drift at {expected['row_id']}/{key}")
        scored = score_natural_output(expected, actual.get("output_text", ""))
        result[str(expected["row_id"])] = {**actual, **scored}
    return result


def summarize(values: dict[str, dict], panel: list[dict]) -> dict:
    ids = [str(row["row_id"]) for row in panel]
    scores = [float(values[row_id]["official_score"]) for row_id in ids]
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in panel:
        clusters[str(row["source_cluster_id"])].append(
            float(values[str(row["row_id"])]["official_score"])
        )
    return {
        "rows": len(ids),
        "source_context_clusters": len(clusters),
        "qa_f1": sum(scores) / len(scores),
        "cluster_equal_qa_f1": sum(sum(v) / len(v) for v in clusters.values()) / len(clusters),
        "empty_rate": sum(bool(values[row_id].get("empty")) for row_id in ids) / len(ids),
        "eos_rate": sum(bool(values[row_id].get("ended_eos")) for row_id in ids) / len(ids),
        "cap_rate": sum(bool(values[row_id].get("hit_cap")) for row_id in ids) / len(ids),
    }


def outcomes(left: dict[str, dict], right: dict[str, dict]) -> dict[str, int]:
    pairs = [
        (float(left[key]["official_score"]), float(right[key]["official_score"]))
        for key in sorted(left)
    ]
    wins = sum(a > b for a, b in pairs)
    losses = sum(a < b for a, b in pairs)
    return {"left_wins": wins, "ties": len(pairs) - wins - losses, "left_losses": losses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--target-length", type=int, required=True)
    for arm in ARMS:
        parser.add_argument(f"--{arm.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    panel = read_jsonl(args.panel)
    if not panel or {row.get("task") for row in panel} != {"longbook_qa_eng"}:
        raise ValueError("panel is not a non-empty InfiniteBench English-QA panel")
    paths = {arm: getattr(args, arm) for arm in ARMS}
    rows = {arm: load_arm(path, panel) for arm, path in paths.items()}
    if len({frozenset(values) for values in rows.values()}) != 1:
        raise ValueError("three arms are not paired on identical QA prompts")
    arms = {arm: summarize(values, panel) for arm, values in rows.items()}
    scores = {arm: value["qa_f1"] for arm, value in arms.items()}
    report = {
        "status": "KANANA_64K_INFINITEBENCH_EN_QA_TRIARM_COMPLETE_V1",
        "model": "kakaocorp/kanana-1.5-8b-instruct-2505",
        "target_length": args.target_length,
        "benchmark": "InfiniteBench longbook_qa_eng",
        "metric": "official English-QA token F1",
        "rows_per_arm": len(panel),
        "source_context_clusters": len({row["source_cluster_id"] for row in panel}),
        "input_token_range": [
            min(int(row["input_tokens"]) for row in panel),
            max(int(row["input_tokens"]) for row in panel),
        ],
        "arms": arms,
        "scores": scores,
        "deltas": {
            "tailspline_minus_official_yarn": scores["tailspline"] - scores["official_yarn"],
            "tailspline_minus_mrpro": scores["tailspline"] - scores["mrpro"],
            "mrpro_minus_official_yarn": scores["mrpro"] - scores["official_yarn"],
        },
        "paired_row_outcomes": {
            "tailspline_vs_official_yarn": outcomes(rows["tailspline"], rows["official_yarn"]),
            "tailspline_vs_mrpro": outcomes(rows["tailspline"], rows["mrpro"]),
            "mrpro_vs_official_yarn": outcomes(rows["mrpro"], rows["official_yarn"]),
        },
        "identity": {
            "panel_sha256": sha256(args.panel),
            "raw_sha256": {arm: sha256(path) for arm, path in paths.items()},
            "same_ordered_prompts": True,
            "complete_context_preserved": True,
            "selection": (
                "all complete eligible official rows above the 32K native window "
                f"whose prompt and answer budget fit within {args.target_length} tokens"
            ),
        },
    }
    atomic_json(args.out, report)
    print(json.dumps({"scores": scores, "deltas": report["deltas"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
