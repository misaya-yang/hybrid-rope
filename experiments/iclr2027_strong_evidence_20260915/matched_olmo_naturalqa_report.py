#!/usr/bin/env python3
"""Compare canonical TailSpline and MrPro on the frozen OLMo Natural-QA pool."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.eval.longbench_metrics import qa_f1_score
from experiments.fixed_rope_three_interfaces_20260913.matched_naturalqa_report import (
    TASKS,
    bootstrap,
    health,
    pooled_cluster_bootstrap,
    rows,
    sha256,
    subset_result,
    task_macro,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native-length", type=int, default=4096)
    args = parser.parse_args()

    panel_rows = rows(args.panel)
    panel = {str(row["row_id"]): row for row in panel_rows}
    if len(panel_rows) != 631 or len(panel) != 631:
        raise ValueError("OLMo Natural-QA panel is not the frozen 631-row pool")
    if set(row["task"] for row in panel_rows) != set(TASKS):
        raise ValueError("OLMo Natural-QA panel does not cover all five tasks")
    if any(int(row["input_tokens"]) <= args.native_length for row in panel_rows):
        raise ValueError("OLMo Natural-QA primary pool contains a native-window row")

    outputs = {}
    for name, path in (("tailspline", args.candidate), ("mrpro", args.baseline)):
        values = rows(path)
        mapping = {str(row["row_id"]): row for row in values}
        if len(values) != 631 or set(mapping) != set(panel):
            raise ValueError(f"unpaired OLMo Natural-QA rows: {name}")
        for row_id, row in mapping.items():
            source = panel[row_id]
            for key in ("task", "prompt_sha256", "input_tokens", "references"):
                if row.get(key) != source.get(key):
                    raise ValueError(f"input identity drift: {name}/{row_id}/{key}")
            score = qa_f1_score(row["output_text"], row["references"])
            if abs(score - float(row["whole_response_f1"])) > 1e-12:
                raise ValueError(f"score drift: {name}/{row_id}")
        outputs[name] = mapping

    ids = list(panel)
    candidate_macro, candidate_tasks = task_macro(outputs["tailspline"], ids)
    baseline_macro, baseline_tasks = task_macro(outputs["mrpro"], ids)
    bands = {
        "4k_to_8k": [row_id for row_id in ids if panel[row_id]["input_tokens"] <= 8192],
        "8k_to_16k": [row_id for row_id in ids if panel[row_id]["input_tokens"] > 8192],
    }
    cluster_tasks = defaultdict(set)
    for row_id in ids:
        cluster_tasks[panel[row_id]["document_cluster_id"]].add(panel[row_id]["task"])
    shared = {cluster: sorted(tasks) for cluster, tasks in cluster_tasks.items() if len(tasks) > 1}
    result = {
        "status": "COMPLETE",
        "contract": "TAILSPLINE_OLMO_S4_NATURAL_QA_FROZEN631_PAIRED_V1",
        "rows_per_arm": 631,
        "tasks": list(TASKS),
        "primary_pool": "exact historical OLMo-tokenizer >4096 frozen source pool",
        "metric": "five-task equal macro whole-response LongBench-normalized token F1",
        "arms": {
            "tailspline": {"macro_f1": candidate_macro, "by_task": candidate_tasks},
            "mrpro": {"macro_f1": baseline_macro, "by_task": baseline_tasks},
        },
        "candidate_minus_baseline": {
            "estimate": candidate_macro - baseline_macro,
            "by_task": {
                task: candidate_tasks[task] - baseline_tasks[task] for task in TASKS
            },
            **bootstrap(panel, outputs["tailspline"], outputs["mrpro"], ids, seed=20260930),
        },
        "question_equal_sensitivity": pooled_cluster_bootstrap(
            panel, outputs["tailspline"], outputs["mrpro"], ids, seed=20261030
        ),
        "length_bands": {
            name: subset_result(
                panel,
                outputs,
                "tailspline",
                "mrpro",
                band_ids,
                seed=20261001 + index,
            )
            for index, (name, band_ids) in enumerate(bands.items())
        },
        "input_length_audit": {
            "native_length": args.native_length,
            "input_token_min": min(row["input_tokens"] for row in panel_rows),
            "input_token_max": max(row["input_tokens"] for row in panel_rows),
            "rows_by_task": dict(Counter(row["task"] for row in panel_rows)),
            "rows_by_band": {name: len(band_ids) for name, band_ids in bands.items()},
        },
        "output_health": {name: health(values, ids) for name, values in outputs.items()},
        "source_cluster_audit": {
            "unique_source_documents": len(cluster_tasks),
            "clusters_shared_across_tasks": len(shared),
            "shared_cluster_tasks": shared,
        },
        "raw_sha256": {"tailspline": sha256(args.candidate), "mrpro": sha256(args.baseline)},
        "panel_sha256": sha256(args.panel),
        "scope": (
            "Paired five-task natural-QA transfer evidence on a previously defined finite OLMo source pool; "
            "not full LongBench, not a new independent sample, and not reusable with a different table identity"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".incomplete")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps(result["candidate_minus_baseline"], sort_keys=True))


if __name__ == "__main__":
    main()
