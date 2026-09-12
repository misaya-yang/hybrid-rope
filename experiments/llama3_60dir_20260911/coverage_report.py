"""Summarize the MR scale-policy triangle and the fixed-S=6 coverage curve."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError(f"empty result: {path}")
    return rows


def task_macro(rows: list[dict], cap: int) -> tuple[float, dict[str, float]]:
    by_task = defaultdict(list)
    for row in rows:
        if int(row["length_cap"]) == cap:
            by_task[row["task"]].append(float(row["partial_score"]))
    if not by_task:
        raise ValueError(f"no rows at cap {cap}")
    task_scores = {task: float(np.mean(values)) for task, values in by_task.items()}
    return float(np.mean(list(task_scores.values()))), task_scores


def paired(left: list[dict], right: list[dict], cap: int) -> dict:
    left_index = {row["row_id"]: row for row in left if int(row["length_cap"]) == cap}
    right_index = {row["row_id"]: row for row in right if int(row["length_cap"]) == cap}
    if left_index.keys() != right_index.keys() or not left_index:
        raise ValueError(f"unaligned rows at cap {cap}")
    differences = np.array([
        float(left_index[key]["partial_score"])
        - float(right_index[key]["partial_score"])
        for key in left_index
    ])
    return {
        "delta": float(differences.mean()),
        "wins": int((differences > 0).sum()),
        "losses": int((differences < 0).sum()),
        "ties": int((differences == 0).sum()),
    }


def log_length_auc(scores: dict[str, float], caps: tuple[int, ...]) -> float:
    """Trapezoidal AUC over log length, normalized to the covered interval."""
    log_caps = np.log(np.asarray(caps, dtype=np.float64))
    values = np.asarray([scores[str(cap)] for cap in caps], dtype=np.float64)
    widths = np.diff(log_caps)
    area = np.sum(0.5 * (values[:-1] + values[1:]) * widths)
    return float(area / (log_caps[-1] - log_caps[0]))


def matrix_report(planb: Path, reroute: Path) -> dict:
    sources = {
        1: {
            8192: planb / "results/S/Native/Native.jsonl",
        },
        2: {
            8192: reroute / "results/mr_scale_matrix/s2_8k/MR.jsonl",
            16384: reroute / "results/s2_16k/MR.jsonl",
        },
        4: {
            cap: planb / "results/S/MR/MR.jsonl"
            for cap in (8192, 16384, 32768)
        },
        8: {
            8192: reroute / "results/mr_scale_matrix/s8_8k32k/MR.jsonl",
            16384: reroute / "results/mr_scale_matrix/s8_8k32k/MR.jsonl",
            32768: reroute / "results/mr_scale_matrix/s8_8k32k/MR.jsonl",
            65536: reroute / "results/s8_64k_compact/MR.jsonl",
        },
        16: {
            8192: reroute / "results/s16_8k32k/MR.jsonl",
            16384: reroute / "results/s16_16k/MR.jsonl",
            32768: reroute / "results/s16_8k32k/MR.jsonl",
            65536: reroute / "results/s16_64k_compact/MR.jsonl",
        },
    }
    cells = {}
    row_sets = defaultdict(dict)
    for scale, by_cap in sources.items():
        for cap, path in by_cap.items():
            rows = load(path)
            macro, tasks = task_macro(rows, cap)
            cells[f"{cap}|{scale}"] = {
                "input_cap": cap, "input_ratio": cap / 8192,
                "table_scale": scale, "macro": macro,
                "task_scores": tasks, "source": str(path),
            }
            row_sets[cap][scale] = {
                row["row_id"] for row in rows if int(row["length_cap"]) == cap
            }
    for cap, sets in row_sets.items():
        values = list(sets.values())
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"matrix row mismatch at cap {cap}: {sets}")
    by_cap = {}
    for cap in sorted(row_sets):
        candidates = [cell for cell in cells.values() if cell["input_cap"] == cap]
        best = max(candidates, key=lambda cell: cell["macro"])
        fixed16 = next(cell for cell in candidates if cell["table_scale"] == 16)
        by_cap[str(cap)] = {
            "best_tested_scale": best["table_scale"],
            "best_tested_macro": best["macro"],
            "fixed16_macro": fixed16["macro"],
            "fixed16_regret": best["macro"] - fixed16["macro"],
        }
    return {"cells": cells, "by_cap": by_cap}


def anytime_report(reroute: Path) -> dict:
    result_root = reroute / "results/anytime_s6_d"
    native = load(result_root / "native8k/Native.jsonl")
    arms = {
        name: load(result_root / f"fixed_s6/{name}.jsonl")
        for name in ("MR", "OfficialYaRN", "BM")
    }
    caps = (8192, 16384, 24576, 32768, 40960, 49152)
    curves = {}
    for name, rows in arms.items():
        scores = {}
        tasks = {}
        for cap in caps:
            scores[str(cap)], tasks[str(cap)] = task_macro(rows, cap)
        equal_milestone_mean = float(np.mean(list(scores.values())))
        curves[name] = {
            "scores": scores,
            "task_scores": tasks,
            "equal_milestone_mean": equal_milestone_mean,
            # Retained so existing readers do not silently break.  This legacy
            # field is an arithmetic mean, not an integrated AUC.
            "equal_milestone_auc": equal_milestone_mean,
            "log_length_trapezoidal_auc": log_length_auc(scores, caps),
        }
    native_macro, native_tasks = task_macro(native, 8192)
    for name, curve in curves.items():
        curve["native_tax"] = curve["scores"]["8192"] - native_macro
        curve["endpoint_48k"] = curve["scores"]["49152"]
    contrasts = {}
    for baseline in ("MR", "OfficialYaRN"):
        contrasts[f"BM_vs_{baseline}"] = {
            str(cap): paired(arms["BM"], arms[baseline], cap) for cap in caps
        }
        contrasts[f"BM_vs_{baseline}"].update({
            "equal_milestone_mean_delta": (
                curves["BM"]["equal_milestone_mean"]
                - curves[baseline]["equal_milestone_mean"]),
            "log_length_trapezoidal_auc_delta": (
                curves["BM"]["log_length_trapezoidal_auc"]
                - curves[baseline]["log_length_trapezoidal_auc"]),
            # Backward-compatible alias for the old arithmetic-mean field.
            "auc_delta": (
                curves["BM"]["equal_milestone_mean"]
                - curves[baseline]["equal_milestone_mean"]),
        })
    return {
        "native_8k": {"macro": native_macro, "task_scores": native_tasks},
        "curves": curves, "contrasts": contrasts,
        "metric_note": (
            "equal_milestone_auc is a legacy arithmetic-mean field; use "
            "log_length_trapezoidal_auc for the normalized integral over "
            "log context length"),
        "claim_scope": (
            "independent-per-length development coverage curve; "
            "not a nested or closed-loop agent trajectory"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planb", type=Path,
        default=Path("/root/autodl-tmp/llama3_planb_20260911"))
    parser.add_argument(
        "--reroute", type=Path,
        default=Path("/root/autodl-tmp/llama3_mrrope_s16_20260911"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    report = {
        "mr_scale_policy_matrix": matrix_report(args.planb, args.reroute),
        "anytime_s6": anytime_report(args.reroute),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": "COMPLETE", "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
