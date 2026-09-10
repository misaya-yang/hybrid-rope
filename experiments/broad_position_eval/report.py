"""Reuse the paired report and add explicit balanced-vs-uniform comparisons."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from experiments.position_overnight import report as existing
from .prepare import read_rows, write_json


def sampling_comparison(balanced, uniform):
    a = json.loads((balanced / "broad_execution.json").read_text())
    b = json.loads((uniform / "broad_execution.json").read_text())
    if a["data_sha256"] != b["data_sha256"] or a["scoring_version"] != b["scoring_version"]:
        raise ValueError("sampling comparison needs identical data and scoring")
    ma = json.loads((balanced / "manifest.json").read_text())
    mb = json.loads((uniform / "manifest.json").read_text())
    for field in ("model", "dtype", "backend"):
        if ma[field] != mb[field]:
            raise ValueError(f"sampling comparison confounded by {field}")
    ca, cb = ma["config"], mb["config"]
    for key in ("samples_per_head", "horizon", "seed", "keep_fraction", "sink_tokens", "recent_tokens", "value_objective"):
        if ca[key] != cb[key]:
            raise ValueError(f"sampling comparison confounded by {key}")
    left = {(r["row_id"], r["arm"]): r for r in read_rows(balanced / "per_example.jsonl")}
    right = {(r["row_id"], r["arm"]): r for r in read_rows(uniform / "per_example.jsonl")}
    cells = defaultdict(list)
    for key in left.keys() & right.keys():
        x, y = left[key], right[key]
        if key[1] not in ("P", "C", "U"):
            continue
        for metric in ("exact_plus_eos", "qa_f1", "official_sequence_ratio"):
            if metric in x and metric in y:
                cells[(x["task"], key[1], metric)].append((key[0], x["material_cluster_id"], float(x[metric]), float(y[metric])))
    result = []
    for (task, arm, metric), pairs in sorted(cells.items()):
        low, high, units, note = existing.paired_bootstrap(pairs, 2000, 2026090927)
        result.append(dict(task=task, arm=arm, metric=metric, rows=len(pairs), independent_units=units,
                           balanced_percent=100*mean(x[2] for x in pairs),
                           uniform_percent=100*mean(x[3] for x in pairs),
                           paired_difference_pp=100*mean(x[2]-x[3] for x in pairs),
                           bootstrap_95=[low, high], note=note))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--balanced", type=Path)
    p.add_argument("--uniform", type=Path)
    args = p.parse_args()
    existing.METRICS = ("official_recall", "exact_plus_eos", "qa_f1", "official_sequence_ratio")
    existing.PC2_CANDIDATES = (*existing.PC2_CANDIDATES, "exact_mass")
    existing.PC2_REFERENCES = (*existing.PC2_REFERENCES, "dense")
    existing.summarize(args.runs, args.output, replicates=2000, seed=2026090927)
    if bool(args.balanced) != bool(args.uniform):
        p.error("pass both --balanced and --uniform for the matched sampling comparison")
    if args.balanced:
        write_json(args.output / "sampling_comparison.json", sampling_comparison(args.balanced, args.uniform))
    write_json(args.output / "EVIDENCE_SCOPE.json", {
        "unit": "original source document/conversation or matched background family",
        "exact_mass": "dense-key-access diagnostic reference, not PC2 approximation success",
        "metrics": "QA F1, MRCR sequence ratio, literal answer plus EOS stay separate",
        "partial": "only completed common rows compared; intervals with few source units are descriptive",
        "generalization": "new domains test transfer; repeated-background retrieval alone does not establish broad benefit"})


if __name__ == "__main__":
    main()
