#!/usr/bin/env python3
"""Paired Native/NCP report for the independent Full-13 and Natural-QA panels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer import TASKS as RULER_TASKS
from scripts.eval.longbench_metrics import qa_f1_score


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paired(panel_path: Path, native_path: Path, ncp_path: Path) -> tuple[dict, dict, dict]:
    panel = {str(row["row_id"]): row for row in rows(panel_path)}
    outputs = {}
    for name, path in (("native", native_path), ("ncp", ncp_path)):
        values = rows(path)
        mapping = {str(row["row_id"]): row for row in values}
        if len(mapping) != len(values) or set(mapping) != set(panel):
            raise ValueError(f"unpaired {name} rows")
        for row_id, row in mapping.items():
            for field in ("task", "prompt_sha256", "references", "input_tokens"):
                if row.get(field) != panel[row_id].get(field):
                    raise ValueError(f"input identity drift: {name}/{row_id}/{field}")
        outputs[name] = mapping
    return panel, outputs["native"], outputs["ncp"]


def bootstrap_task_equal(panel, native, ncp, *, metric, draws=20_000, seed=20260916):
    rng = np.random.default_rng(seed)
    task_draws, by_task = [], {}
    for task in sorted({row["task"] for row in panel.values()}):
        ids = [row_id for row_id, row in panel.items() if row["task"] == task]
        delta = np.asarray([metric(ncp[row_id]) - metric(native[row_id]) for row_id in ids])
        by_task[task] = float(delta.mean())
        indices = rng.integers(len(delta), size=(draws, len(delta)))
        task_draws.append(delta[indices].mean(axis=1))
    values = np.mean(task_draws, axis=0)
    return {
        "estimate": float(np.mean(list(by_task.values()))), "by_task": by_task,
        "paired_task_equal_bootstrap_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
    }


def bootstrap_cluster_task_equal(panel, native, ncp, *, draws=20_000, seed=20260917):
    rng = np.random.default_rng(seed)
    task_draws, by_task = [], {}
    for task in sorted({row["task"] for row in panel.values()}):
        groups = defaultdict(list)
        for row_id, row in panel.items():
            if row["task"] == task:
                groups[row["document_cluster_id"]].append(row_id)
        group_delta = np.asarray([
            np.mean([qa_f1_score(ncp[row_id]["output_text"], panel[row_id]["references"])
                     - qa_f1_score(native[row_id]["output_text"], panel[row_id]["references"])
                     for row_id in ids])
            for ids in groups.values()
        ])
        by_task[task] = float(group_delta.mean())
        indices = rng.integers(len(group_delta), size=(draws, len(group_delta)))
        task_draws.append(group_delta[indices].mean(axis=1))
    values = np.mean(task_draws, axis=0)
    return {
        "estimate": float(np.mean(list(by_task.values()))), "by_task": by_task,
        "paired_source_cluster_task_equal_bootstrap_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-panel", type=Path, required=True)
    parser.add_argument("--ruler-native", type=Path, required=True)
    parser.add_argument("--ruler-ncp", type=Path, required=True)
    parser.add_argument("--qa-panel", type=Path, required=True)
    parser.add_argument("--qa-native", type=Path, required=True)
    parser.add_argument("--qa-ncp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ruler_panel, ruler_native, ruler_ncp = paired(
        args.ruler_panel, args.ruler_native, args.ruler_ncp,
    )
    if Counter(row["task"] for row in ruler_panel.values()) != Counter({task: 100 for task in RULER_TASKS}):
        raise ValueError("confirmation RULER panel is not Full-13 x 100")
    qa_panel, qa_native, qa_ncp = paired(args.qa_panel, args.qa_native, args.qa_ncp)
    qa_counts = Counter(row["task"] for row in qa_panel.values())
    if set(qa_counts) != {"2wikimqa", "hotpotqa", "qasper"} or any(
        count <= 0 or count > 80 for count in qa_counts.values()
    ):
        raise ValueError("confirmation Natural-QA panel is not the three-task cap-80 census")
    report = {
        "status": "OLMO_NATIVE_NCP_INDEPENDENT_CONFIRMATION_COMPLETE_V1",
        "ruler": {
            "rows_per_arm": len(ruler_panel),
            "metric": "official RULER score, task-equal across 13 tasks",
            "ncp_minus_native": bootstrap_task_equal(
                ruler_panel, ruler_native, ruler_ncp,
                metric=lambda row: float(row["ruler_official_score"]),
            ),
        },
        "natural_qa": {
            "rows_per_arm": len(qa_panel),
            "rows_by_task": dict(qa_counts),
            "metric": "LongBench normalized token F1, task-equal across 3 tasks",
            "ncp_minus_native": bootstrap_cluster_task_equal(qa_panel, qa_native, qa_ncp),
        },
        "scope": (
            "Frozen NCP versus Native on output-blind new RULER worlds/questions and complete "
            "untruncated Native-window Natural-QA source rows."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": report["status"], "ruler_rows": 1300, "qa_rows": len(qa_panel)}))


if __name__ == "__main__":
    main()
