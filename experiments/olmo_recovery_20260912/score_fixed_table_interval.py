#!/usr/bin/env python3
"""Score a fixed-table interval panel without turning its metrics into one endpoint."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.olmo_recovery_20260912.recovery_v2_runtime import table_for_config


CANDIDATES = ("BetaSym_gamma3_g8", "RangeBridge50_g8", "BM_g8_RangeGain")
BASELINES = ("MrPro_g8", "BM_g8")
ARMS = CANDIDATES + BASELINES


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def mean(values) -> float:
    values = list(values)
    if not values:
        raise ValueError("empty mean")
    return sum(values) / len(values)


def log_auc(curve: dict[int, float]) -> float:
    caps = sorted(curve)
    area = sum(
        0.5 * (curve[left] + curve[right]) * (math.log(right) - math.log(left))
        for left, right in zip(caps, caps[1:])
    )
    return area / (math.log(caps[-1]) - math.log(caps[0]))


def summarize(rows: list[dict], expected: dict[str, dict], caps: list[int], tasks: list[str]) -> dict:
    if {row["eval_id"].split(":", 1)[1] for row in rows} != set(expected):
        raise ValueError("arm row identities differ from the prepared panel")
    cells = defaultdict(list)
    for row in rows:
        source = expected[row["eval_id"].split(":", 1)[1]]
        for field in ("task", "length_cap", "references", "prompt_sha256"):
            if row[field] != source[field]:
                raise ValueError(f"stored generation differs at {field}")
        cells[(row["length_cap"], row["task"])].append(row)
    by_length = {}
    for cap in caps:
        by_task = {}
        for task in tasks:
            items = cells[(cap, task)]
            if len(items) != 4:
                raise ValueError(f"expected four rows for {cap}/{task}")
            by_task[task] = {
                "official": mean(item["ruler_official_score"] for item in items),
                "exact_plus_eos": mean(item["exact_plus_eos"] for item in items),
                "eos_rate": mean(item["ended_eos"] for item in items),
                "cap_exhaustion_rate": mean(item["hit_cap"] for item in items),
            }
        by_length[str(cap)] = {
            "task_macro_official": mean(value["official"] for value in by_task.values()),
            "tasks": by_task,
        }
    curve = {cap: by_length[str(cap)]["task_macro_official"] for cap in caps}
    powers = {cap: curve[cap] for cap in caps if cap in (8192, 16384, 32768, 65536)}
    return {
        "by_length": by_length,
        "log_length_auc": log_auc(curve) if len(caps) > 1 else None,
        "power_of_two_log_length_auc": log_auc(powers) if len(powers) > 1 else None,
        "interval_min": min(curve.values()),
        "interior_min": min((curve[cap] for cap in caps[1:-1]), default=None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / "prepared_llama_g8_r0" / "manifest.json").read_text())
    fixed = manifest["fixed_table_contract"]
    if not (
        fixed["one_table_per_arm_for_entire_session"]
        and fixed["same_table_at_every_runtime_length"]
        and fixed["same_table_in_every_layer"]
        and not fixed["runtime_table_switching"]
    ):
        raise ValueError("prepared panel is not the fixed-table problem")
    source_rows = read_rows(args.root / "prepared_llama_g8_r0" / "screen.jsonl")
    supplement = args.root / "supplement_48k" / "prepared"
    supplement_manifest = json.loads((supplement / "manifest.json").read_text())
    if supplement_manifest["fixed_table_contract"] != fixed:
        raise ValueError("48K supplement changed the fixed-table contract")
    source_rows.extend(read_rows(supplement / "screen.jsonl"))
    expected = {row["row_id"]: row for row in source_rows}
    caps = sorted(set(manifest["runtime_lengths"] + supplement_manifest["runtime_lengths"]))
    tasks = list(manifest["tasks"])
    summaries = {}
    for arm in ARMS:
        if arm == "BM_g8_RangeGain":
            rows = read_rows(args.root / "BM_g8_RangeGain_all" / "generations.jsonl")
        else:
            rows = read_rows(args.root / arm / "generations.jsonl")
            rows += read_rows(args.root / "supplement_48k" / arm / "generations.jsonl")
        summaries[arm] = summarize(rows, expected, caps, tasks)

    native_rows = read_rows(args.root / "Native_8k" / "generations.jsonl")
    native_expected = {key: value for key, value in expected.items() if value["length_cap"] == caps[0]}
    native = summarize(native_rows, native_expected, [caps[0]], tasks)
    bm = summaries["BM_g8"]
    mrpro = summaries["MrPro_g8"]
    envelope = {
        cap: max(
            bm["by_length"][str(cap)]["task_macro_official"],
            mrpro["by_length"][str(cap)]["task_macro_official"],
        )
        for cap in caps
    }
    contrasts = {}
    native_score = native["by_length"][str(caps[0])]["task_macro_official"]
    for name in CANDIDATES:
        candidate = summaries[name]
        curve = {cap: candidate["by_length"][str(cap)]["task_macro_official"] for cap in caps}
        contrasts[name] = {
            "minus_BM_by_length": {
                str(cap): curve[cap] - bm["by_length"][str(cap)]["task_macro_official"]
                for cap in caps
            },
            "minus_MrPro_by_length": {
                str(cap): curve[cap] - mrpro["by_length"][str(cap)]["task_macro_official"]
                for cap in caps
            },
            "minus_BM_auc": candidate["log_length_auc"] - bm["log_length_auc"],
            "minus_MrPro_auc": candidate["log_length_auc"] - mrpro["log_length_auc"],
            "native_regret_at_1x": native_score - curve[caps[0]],
            "endpoint_regret_vs_strong_static_floor": envelope[caps[-1]] - curve[caps[-1]],
            "worst_regret_vs_pointwise_strong_static_baseline": max(envelope[cap] - curve[cap] for cap in caps),
        }

    raw_config = json.loads((Path(manifest["model"]) / "config.json").read_text())
    config = SimpleNamespace(**raw_config)
    tables = {}
    for arm in ARMS:
        table = table_for_config(config, arm)
        values = np.asarray(table["values_float32"], dtype="<f4")
        tables[arm] = {
            "sha256_float32": hashlib.sha256(values.tobytes()).hexdigest(),
            "gain": table["gain"],
            "construction": table["construction"],
        }
    result = {
        "status": "COMPLETE",
        "scope": manifest["scope"],
        "problem": "one fixed Native-relative table from 8K through the 64K deployment horizon",
        "fixed_table_contract": fixed,
        "threshold_status": "Native, endpoint, and interior tolerances remain descriptive until their numeric experiment-contract values are declared",
        "native_8k": native,
        "arms": summaries,
        "contrasts": contrasts,
        "tables": tables,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "contrasts": contrasts}, sort_keys=True))


if __name__ == "__main__":
    main()
