#!/usr/bin/env python3
"""Report the frozen Llama-3-8B NTS2 Native-window transfer."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer import TASKS


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_map(panel: dict[str, dict], path: Path) -> dict[str, dict]:
    values = rows(path)
    mapped = {str(row["row_id"]): row for row in values}
    if len(mapped) != len(values) or set(mapped) != set(panel):
        raise ValueError(f"unpaired generation rows: {path}")
    for row_id, row in mapped.items():
        for field in ("task", "prompt_sha256", "references", "input_tokens"):
            if row.get(field) != panel[row_id].get(field):
                raise ValueError(f"input identity drift: {row_id}/{field}")
    return mapped


def ruler_score(panel, output):
    grouped = defaultdict(list)
    for row_id, prompt in panel.items():
        grouped[prompt["task"]].append(float(output[row_id]["ruler_official_score"]))
    by_task = {task: sum(values) / len(values) for task, values in sorted(grouped.items())}
    return {"score": sum(by_task.values()) / len(by_task), "by_task": by_task}


def lm_score(path: Path) -> dict:
    values = rows(path)
    if len(values) != 46 or Counter(int(row["length"]) for row in values) != Counter({8192: 46}):
        raise ValueError("Llama Native PPL panel is not 46 documents at 8K")
    whole = sum(float(row["whole_loss_sum"]) for row in values) / sum(int(row["whole_target_count"]) for row in values)
    tail = sum(float(row["tail128_loss_sum"]) for row in values) / sum(int(row["tail128_target_count"]) for row in values)
    return {"documents": 46, "whole_nll": whole, "tail128_nll": tail, "raw_sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--ppl-baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    panel_path = args.baseline_root / "assets/pilot/inputs.jsonl"
    panel = {str(row["row_id"]): row for row in rows(panel_path)}
    if Counter(row["task"] for row in panel.values()) != Counter({task: 10 for task in TASKS}):
        raise ValueError("Llama panel is not Full-13 x 10")
    native_path = args.baseline_root / "runs/N0/generations.jsonl"
    nts2_path = args.root / "runs/ruler/generations.jsonl"
    native = ruler_score(panel, output_map(panel, native_path))
    nts2 = ruler_score(panel, output_map(panel, nts2_path))
    native_lm = lm_score(args.ppl_baseline / "lm_rows.jsonl")
    nts2_lm = lm_score(args.root / "runs/ppl/lm_rows.jsonl")
    delta_nll = nts2_lm["whole_nll"] - native_lm["whole_nll"]
    report = {
        "status": "LLAMA_NTS2_NATIVE_TRANSFER_COMPLETE_V1",
        "method": "native_tailspline_s2_midgain_v1",
        "model": "Meta-Llama-3-8B-Instruct",
        "ruler": {"native": native, "nts2": nts2, "nts2_minus_native": nts2["score"] - native["score"]},
        "lm_8k": {
            "native": native_lm, "nts2": nts2_lm,
            "nts2_minus_native_nll": delta_nll,
            "nts2_minus_native_ppl_percent": 100.0 * math.expm1(delta_nll),
        },
        "input_sha256": {"ruler": sha256(panel_path)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": report["status"], "ruler_delta": report["ruler"]["nts2_minus_native"]}))


if __name__ == "__main__":
    main()
