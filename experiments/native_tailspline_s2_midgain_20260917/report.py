#!/usr/bin/env python3
"""Create the paired preregistered NTS2 decision report."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer import TASKS
from scripts.eval.longbench_metrics import qa_f1_score


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def map_outputs(panel: dict[str, dict], path: Path) -> dict[str, dict]:
    values = rows(path)
    mapped = {str(row["row_id"]): row for row in values}
    if len(mapped) != len(values) or set(mapped) != set(panel):
        raise ValueError(f"unpaired output rows: {path}")
    for row_id, row in mapped.items():
        for field in ("task", "prompt_sha256", "references", "input_tokens"):
            if row.get(field) != panel[row_id].get(field):
                raise ValueError(f"input identity drift: {path}/{row_id}/{field}")
    return mapped


def task_equal(panel, outputs, metric):
    grouped = defaultdict(list)
    for row_id, prompt in panel.items():
        grouped[prompt["task"]].append(metric(outputs[row_id], prompt))
    by_task = {task: float(np.mean(values)) for task, values in sorted(grouped.items())}
    return {"score": float(np.mean(list(by_task.values()))), "by_task": by_task}


def cluster_bootstrap_delta(panel, left, right, *, draws=20_000, seed=20260917):
    rng = np.random.default_rng(seed)
    task_draws = []
    for task in sorted({row["task"] for row in panel.values()}):
        groups = defaultdict(list)
        for row_id, prompt in panel.items():
            if prompt["task"] == task:
                groups[prompt["document_cluster_id"]].append(row_id)
        deltas = np.asarray([
            np.mean([
                qa_f1_score(right[row_id]["output_text"], panel[row_id]["references"])
                - qa_f1_score(left[row_id]["output_text"], panel[row_id]["references"])
                for row_id in ids
            ])
            for ids in groups.values()
        ])
        indices = rng.integers(len(deltas), size=(draws, len(deltas)))
        task_draws.append(deltas[indices].mean(axis=1))
    values = np.mean(task_draws, axis=0)
    return np.quantile(values, [0.025, 0.975]).tolist()


def lm_scores(manifest: dict, paths: dict[str, Path]) -> dict:
    expected = {sample["pair_id"]: sample for sample in manifest["samples"]}
    values = defaultdict(dict)
    hashes = {}
    for arm, path in paths.items():
        hashes[arm] = sha256(path)
        for row in rows(path):
            pair_id = row["pair_id"]
            key = (arm, row["context"])
            if pair_id not in expected or key in values[pair_id]:
                raise ValueError("invalid or duplicate LM row")
            if row["target_sha256"] != expected[pair_id]["target_sha256"]:
                raise ValueError("LM target identity drift")
            values[pair_id][key] = float(row["nll_sum"]) / int(row["target_count"])
    conditions = {(arm, context) for arm in paths for context in ("full", "recent")}
    if set(values) != set(expected) or any(set(item) != conditions for item in values.values()):
        raise ValueError("LM panel is incomplete")
    by_document = defaultdict(list)
    for pair_id, sample in expected.items():
        by_document[sample["document_id"]].append(values[pair_id])
    means = {}
    for arm in paths:
        for context in ("full", "recent"):
            means[f"{arm}_{context}"] = float(np.mean([
                np.mean([row[(arm, context)] for row in document_rows])
                for document_rows in by_document.values()
            ]))
    return {"pairs": len(expected), "documents": len(by_document), "means": means, "raw_sha256": hashes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ruler_panel_path = args.baseline_root / "assets/ruler_confirm_13x10/panels/4096/inputs.jsonl"
    ruler_panel = {str(row["row_id"]): row for row in rows(ruler_panel_path)}
    if Counter(row["task"] for row in ruler_panel.values()) != Counter({task: 10 for task in TASKS}):
        raise ValueError("RULER panel is not Full-13 x 10")
    ruler = {
        "native": map_outputs(ruler_panel, args.baseline_root / "runs/confirm/ruler/native/generations.jsonl"),
        "ncp": map_outputs(ruler_panel, args.baseline_root / "runs/confirm/ruler/ncp/generations.jsonl"),
        "nts2": map_outputs(ruler_panel, args.root / "runs/ruler/generations.jsonl"),
    }
    ruler_scores = {
        arm: task_equal(ruler_panel, output, lambda row, _: float(row["ruler_official_score"]))
        for arm, output in ruler.items()
    }

    qa_panel_path = args.baseline_root / "assets/naturalqa_3x80/inputs.jsonl"
    qa_panel = {str(row["row_id"]): row for row in rows(qa_panel_path)}
    qa = {
        "native": map_outputs(qa_panel, args.baseline_root / "runs/confirm/qa/native/generations.jsonl"),
        "ncp": map_outputs(qa_panel, args.baseline_root / "runs/confirm/qa/ncp/generations.jsonl"),
        "nts2": map_outputs(qa_panel, args.root / "runs/qa/generations.jsonl"),
    }
    qa_scores = {
        arm: task_equal(
            qa_panel, output,
            lambda row, prompt: qa_f1_score(row["output_text"], prompt["references"]),
        )
        for arm, output in qa.items()
    }

    manifest = json.loads((args.baseline_root / "assets/lm128/manifest.json").read_text())
    lm = lm_scores(manifest, {
        "native": args.baseline_root / "runs/confirm/lm_native/scores_native.jsonl",
        "ncp": args.baseline_root / "runs/confirm/lm_ncp/scores_ncp.jsonl",
        "nts2": args.root / "runs/lm/scores_candidate.jsonl",
    })
    ruler_native_delta = ruler_scores["nts2"]["score"] - ruler_scores["native"]["score"]
    ruler_ncp_delta = ruler_scores["nts2"]["score"] - ruler_scores["ncp"]["score"]
    qa_native_delta = qa_scores["nts2"]["score"] - qa_scores["native"]["score"]
    lm_ncp_delta = lm["means"]["nts2_full"] - lm["means"]["ncp_full"]
    gates = {
        "ruler_vs_native_ge_5pp": ruler_native_delta >= 0.05,
        "ruler_vs_ncp_ge_2pp": ruler_ncp_delta >= 0.02,
        "natural_qa_not_below_native": qa_native_delta >= 0.0,
        "lm_nll_within_ncp_plus_0p001": lm_ncp_delta <= 0.001,
    }
    report = {
        "status": "NTS2_DECISION_COMPLETE_V1",
        "method": "native_tailspline_s2_midgain_v1",
        "ruler": {"scores": ruler_scores, "nts2_minus_native": ruler_native_delta,
                  "nts2_minus_ncp": ruler_ncp_delta},
        "natural_qa": {
            "scores": qa_scores,
            "nts2_minus_native": qa_native_delta,
            "nts2_minus_native_cluster_bootstrap_ci95": cluster_bootstrap_delta(
                qa_panel, qa["native"], qa["nts2"],
            ),
        },
        "lm": {**lm, "nts2_minus_ncp_full": lm_ncp_delta},
        "preregistered_gates": gates,
        "all_gates_pass": all(gates.values()),
        "input_sha256": {"ruler": sha256(ruler_panel_path), "qa": sha256(qa_panel_path),
                         "lm_manifest": sha256(args.baseline_root / "assets/lm128/manifest.json")},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": report["status"], "all_gates_pass": report["all_gates_pass"]}))


if __name__ == "__main__":
    main()
