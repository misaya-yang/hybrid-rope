#!/usr/bin/env python3
"""Audit frozen RoPE profiles by discrete Native-code OOD and target SEP."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import numpy as np


def position_codes(frequencies: np.ndarray, distances: np.ndarray) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=np.float64)
    phase = np.outer(np.asarray(distances, dtype=np.float64), frequencies)
    codes = np.empty((phase.shape[0], 2 * phase.shape[1]), dtype=np.float64)
    codes[:, 0::2] = np.cos(phase)
    codes[:, 1::2] = np.sin(phase)
    codes /= math.sqrt(frequencies.size)
    return codes


def separation_curve(frequencies: np.ndarray, target: int) -> np.ndarray:
    h = np.arange(1, int(target) + 1, dtype=np.float64)[:, None]
    phase = 0.5 * h * np.asarray(frequencies, dtype=np.float64)[None, :]
    squared = 4.0 * np.square(np.sin(phase)).mean(axis=1)
    return np.sqrt(np.maximum(squared, 0.0))


def nearest_native_distances(
    native_codes: np.ndarray,
    frequencies: np.ndarray,
    target: int,
    *,
    block_size: int,
) -> np.ndarray:
    result = np.empty(int(target) + 1, dtype=np.float64)
    native_transpose = native_codes.T
    for start in range(0, int(target) + 1, int(block_size)):
        stop = min(start + int(block_size), int(target) + 1)
        deployed = position_codes(frequencies, np.arange(start, stop))
        maximum_inner_product = np.max(deployed @ native_transpose, axis=1)
        result[start:stop] = np.sqrt(np.maximum(2.0 - 2.0 * maximum_inner_product, 0.0))
    return result


def summarize_profile(job: tuple) -> dict:
    name, score, m, native, native_codes, length, target, block_size = job
    deployed = native * np.power(4.0, -np.asarray(m, dtype=np.float64))
    nearest = nearest_native_distances(
        native_codes,
        deployed,
        target,
        block_size=block_size,
    )
    sep = separation_curve(deployed, target)
    split = int(length) + 1
    return {
        "name": name,
        "task_score": float(score),
        "ood_max": float(nearest.max()),
        "ood_p95": float(np.quantile(nearest, 0.95)),
        "ood_mean": float(nearest.mean()),
        "native_drift_max": float(nearest[:split].max()),
        "extrapolation_ood_max": float(nearest[split:].max()),
        "sep_min": float(sep.min()),
        "most_dangerous_collision_distance": int(np.argmin(sep) + 1),
        "theorem_sep_le_2ood": bool(float(sep.min()) <= 2.0 * float(nearest.max()) + 1e-12),
    }


def dominance(rows: list[dict], ood_key: str) -> tuple[list[str], list[dict]]:
    frontier = []
    contradictions = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            no_worse = other[ood_key] <= candidate[ood_key] and other["sep_min"] >= candidate["sep_min"]
            strict = other[ood_key] < candidate[ood_key] or other["sep_min"] > candidate["sep_min"]
            if no_worse and strict:
                dominated = True
                if other["task_score"] < candidate["task_score"]:
                    contradictions.append({
                        "geometric_dominator": other["name"],
                        "task_better_profile": candidate["name"],
                        "task_score_delta": other["task_score"] - candidate["task_score"],
                        "ood_delta": other[ood_key] - candidate[ood_key],
                        "sep_delta": other["sep_min"] - candidate["sep_min"],
                    })
        if not dominated:
            frontier.append(candidate["name"])
    contradictions.sort(key=lambda row: row["task_score_delta"])
    return frontier, contradictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists() or args.workers <= 0 or args.block_size <= 0:
        raise ValueError("use a new output path and positive execution sizes")
    payload = json.loads(args.input.read_text())
    length = int(payload["native_window"])
    target = int(payload["test_distance"])
    count = int(payload["K"])
    base = float(payload["native_base"])
    native = np.power(base, -np.arange(count, dtype=np.float64) / count)
    native_codes = position_codes(native, np.arange(length + 1))
    jobs = [
        (
            row["name"], row["score"], row["m"], native, native_codes,
            length, target, args.block_size,
        )
        for row in payload["rows"]
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        rows = list(pool.map(summarize_profile, jobs))
    from scipy.stats import spearmanr

    scores = np.asarray([row["task_score"] for row in rows])
    max_ood = np.asarray([row["ood_max"] for row in rows])
    p95_ood = np.asarray([row["ood_p95"] for row in rows])
    sep = np.asarray([row["sep_min"] for row in rows])
    max_frontier, max_contradictions = dominance(rows, "ood_max")
    p95_frontier, p95_contradictions = dominance(rows, "ood_p95")
    result = {
        "status": "COMPLETE",
        "definition": {
            "native_window": length,
            "target": target,
            "pairs": count,
            "ood_reference": "discrete Native codes at integer distances 0..L",
            "nearest_neighbor": "exact block matrix inner products in float64",
            "sep_domain": "integer separations 1..T",
        },
        "rows": rows,
        "rank_correlations": {
            "task_vs_negative_ood_max": float(spearmanr(scores, -max_ood).statistic),
            "task_vs_negative_ood_p95": float(spearmanr(scores, -p95_ood).statistic),
            "task_vs_sep_min": float(spearmanr(scores, sep).statistic),
        },
        "pareto_ood_max_sep": {
            "frontier": max_frontier,
            "geometric_dominance_task_reversals": max_contradictions,
        },
        "pareto_ood_p95_sep": {
            "frontier": p95_frontier,
            "geometric_dominance_task_reversals": p95_contradictions,
        },
        "decision": (
            "OOD_AND_SEP_INSUFFICIENT"
            if max_contradictions or p95_contradictions
            else "NO_DOMINANCE_REVERSAL_ON_THIS_PANEL"
        ),
        "scope": "zero-GPU audit of 12 frozen OLMo profiles; geometry is not a task score",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "decision": result["decision"],
        "rank_correlations": result["rank_correlations"],
        "frontier_ood_max_sep": max_frontier,
        "dominance_reversals": len(max_contradictions),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
