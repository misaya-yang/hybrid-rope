#!/usr/bin/env python3
"""Paired statistics for Anchored vs Geometric from LongBench raw compare outputs.

Input files must be raw outputs produced by scripts/eval_longbench.py, containing:
- models.hybrid_lora.tasks.<task>.per_sample_traces

The script pairs samples by (task, index), computes paired diff = anchored - geometric,
and reports bootstrap CI, sign-flip p-value, permutation p-value, and effect size.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

LB21_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
    "multi_news_zh",
    "vcsum",
    "dureader",
    "lsht",
    "passage_retrieval_zh",
]


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def score_from_trace(tr: Dict) -> float:
    raw = tr.get("score_raw")
    if isinstance(raw, (int, float)):
        return float(raw)
    pct = tr.get("score_pct")
    if isinstance(pct, (int, float)):
        return float(pct) / 100.0
    s = tr.get("score")
    if isinstance(s, (int, float)):
        # Some old artifacts store pct under `score`.
        return float(s) / 100.0 if float(s) > 1.0 else float(s)
    return 0.0


def traces_to_indexed_scores(task_blob: Dict) -> Dict[int, float]:
    out: Dict[int, float] = {}
    traces = task_blob.get("per_sample_traces", [])
    if not isinstance(traces, list):
        return out
    for i, tr in enumerate(traces):
        if not isinstance(tr, dict):
            continue
        idx_raw = tr.get("index", i)
        try:
            idx = int(idx_raw)
        except Exception:
            idx = i
        out[idx] = score_from_trace(tr)
    return out


def collect_diffs(anchored_json: Path, geometric_json: Path, tasks: List[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    a_obj = load_json(anchored_json)
    g_obj = load_json(geometric_json)
    a_tasks = a_obj.get("models", {}).get("hybrid_lora", {}).get("tasks", {})
    g_tasks = g_obj.get("models", {}).get("hybrid_lora", {}).get("tasks", {})

    per_task: Dict[str, np.ndarray] = {}
    pooled: List[float] = []
    for task in tasks:
        a_blob = a_tasks.get(task)
        g_blob = g_tasks.get(task)
        if not isinstance(a_blob, dict) or not isinstance(g_blob, dict):
            continue
        a_scores = traces_to_indexed_scores(a_blob)
        g_scores = traces_to_indexed_scores(g_blob)
        common = sorted(set(a_scores.keys()) & set(g_scores.keys()))
        if not common:
            continue
        diffs = np.asarray([float(a_scores[i] - g_scores[i]) for i in common], dtype=np.float64)
        per_task[task] = diffs
        pooled.extend(diffs.tolist())

    return per_task, np.asarray(pooled, dtype=np.float64)


def bootstrap_ci_mean(diffs: np.ndarray, n_bootstrap: int, seed: int) -> Tuple[float, float, float]:
    if diffs.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = int(diffs.shape[0])
    boot = np.zeros(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(diffs[idx]))
    return float(np.mean(diffs)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def sign_flip_pvalue(diffs: np.ndarray, seed: int = 42, n_mc: int = 20000) -> float:
    n = int(diffs.shape[0])
    if n <= 0:
        return float("nan")
    observed = abs(float(np.mean(diffs)))
    if n <= 20:
        total = 1 << n
        count = 0
        for mask in range(total):
            signs = np.ones(n, dtype=np.float64)
            for i in range(n):
                if (mask >> i) & 1:
                    signs[i] = -1.0
            m = abs(float(np.mean(diffs * signs)))
            if m >= observed:
                count += 1
        return float(count / total)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(int(n_mc)):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n, replace=True)
        m = abs(float(np.mean(diffs * signs)))
        if m >= observed:
            count += 1
    return float(count / float(n_mc))


def paired_permutation_pvalue(diffs: np.ndarray, seed: int = 52, n_mc: int = 20000) -> float:
    # Pairwise label swap on (A, B) is equivalent to sign flips on (A-B),
    # but we keep this separate output for reporting completeness.
    return sign_flip_pvalue(diffs=diffs, seed=seed, n_mc=n_mc)


def effect_size_dz(diffs: np.ndarray) -> float:
    if diffs.size <= 1:
        return 0.0
    std = float(np.std(diffs, ddof=1))
    if std <= 1e-12:
        return float("inf") if float(np.mean(diffs)) > 0 else float("-inf")
    return float(np.mean(diffs) / std)


def claim_grade(mean_diff_pct: float, ci_low_pct: float, p_perm: float) -> str:
    if mean_diff_pct > 0 and ci_low_pct > 0 and p_perm < 0.05:
        return "significant"
    if mean_diff_pct > 0:
        return "directional"
    return "inconclusive"


def summarize_diffs(diffs: np.ndarray, n_bootstrap: int, seed: int) -> Dict[str, float]:
    mean_raw, ci_low_raw, ci_high_raw = bootstrap_ci_mean(diffs=diffs, n_bootstrap=n_bootstrap, seed=seed)
    p_sign = sign_flip_pvalue(diffs=diffs, seed=seed)
    p_perm = paired_permutation_pvalue(diffs=diffs, seed=seed + 1)
    dz = effect_size_dz(diffs)
    mean_pct = mean_raw * 100.0
    ci_low_pct = ci_low_raw * 100.0
    ci_high_pct = ci_high_raw * 100.0
    return {
        "n_pairs": int(diffs.shape[0]),
        "mean_diff_raw": float(mean_raw),
        "ci95_low_raw": float(ci_low_raw),
        "ci95_high_raw": float(ci_high_raw),
        "mean_diff_pct": float(mean_pct),
        "ci95_low_pct": float(ci_low_pct),
        "ci95_high_pct": float(ci_high_pct),
        "p_sign_flip": float(p_sign),
        "p_permutation": float(p_perm),
        "effect_size_dz": float(dz),
        "claim_grade": claim_grade(mean_diff_pct=mean_pct, ci_low_pct=ci_low_pct, p_perm=float(p_perm)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired stats for llama8k theory validation (Anchored vs Geometric)")
    ap.add_argument("--anchored_jsons", type=str, required=True, help="Comma-separated list of anchored raw compare json files")
    ap.add_argument("--geometric_jsons", type=str, required=True, help="Comma-separated list of geometric raw compare json files")
    ap.add_argument("--tasks", type=str, default=",".join(LB21_TASKS))
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument("--output_md", type=str, required=True)
    args = ap.parse_args()

    anchored_paths = [Path(p) for p in parse_csv(args.anchored_jsons)]
    geometric_paths = [Path(p) for p in parse_csv(args.geometric_jsons)]
    if len(anchored_paths) != len(geometric_paths):
        raise ValueError("anchored_jsons and geometric_jsons must have the same count")
    tasks = parse_csv(args.tasks)

    pooled_all: List[float] = []
    per_task_all: Dict[str, List[float]] = {t: [] for t in tasks}
    run_pairs: List[Dict[str, str]] = []

    for a_path, g_path in zip(anchored_paths, geometric_paths):
        if not a_path.exists() or not g_path.exists():
            continue
        run_pairs.append({"anchored": a_path.as_posix(), "geometric": g_path.as_posix()})
        per_task, pooled = collect_diffs(a_path, g_path, tasks)
        pooled_all.extend(pooled.tolist())
        for task, diffs in per_task.items():
            per_task_all.setdefault(task, []).extend(diffs.tolist())

    pooled_arr = np.asarray(pooled_all, dtype=np.float64)
    pooled_summary = summarize_diffs(diffs=pooled_arr, n_bootstrap=int(args.n_bootstrap), seed=int(args.seed))

    per_task_summary: Dict[str, Dict[str, float]] = {}
    for task in tasks:
        vals = np.asarray(per_task_all.get(task, []), dtype=np.float64)
        if vals.size == 0:
            continue
        per_task_summary[task] = summarize_diffs(diffs=vals, n_bootstrap=int(args.n_bootstrap), seed=int(args.seed) + 17)

    out = {
        "meta": {
            "tasks": tasks,
            "run_pairs": run_pairs,
            "n_bootstrap": int(args.n_bootstrap),
            "seed": int(args.seed),
            "score_unit": "raw_and_pct",
            "comparison": "anchored_minus_geometric",
        },
        "pooled": pooled_summary,
        "per_task": per_task_summary,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    save_json(output_json, out)

    lines = [
        "# Paired Statistics: Anchored vs Geometric",
        "",
        f"- run_pairs: `{len(run_pairs)}`",
        f"- claim_grade: `{pooled_summary.get('claim_grade', 'inconclusive')}`",
        f"- pooled_mean_diff_pct: `{pooled_summary.get('mean_diff_pct', 0.0):.4f}`",
        f"- pooled_ci95_pct: `[{pooled_summary.get('ci95_low_pct', 0.0):.4f}, {pooled_summary.get('ci95_high_pct', 0.0):.4f}]`",
        f"- p_sign_flip: `{pooled_summary.get('p_sign_flip', float('nan')):.6f}`",
        f"- p_permutation: `{pooled_summary.get('p_permutation', float('nan')):.6f}`",
        f"- effect_size_dz: `{pooled_summary.get('effect_size_dz', 0.0):.6f}`",
        "",
        "| task | n_pairs | mean_diff_pct | ci95_low_pct | ci95_high_pct | p_permutation | effect_size_dz | claim_grade |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for task, row in per_task_summary.items():
        lines.append(
            "| {task} | {n_pairs} | {mean_diff_pct:.4f} | {ci95_low_pct:.4f} | {ci95_high_pct:.4f} | {p_permutation:.6f} | {effect_size_dz:.6f} | {claim_grade} |".format(
                task=task,
                n_pairs=int(row.get("n_pairs", 0)),
                mean_diff_pct=float(row.get("mean_diff_pct", 0.0)),
                ci95_low_pct=float(row.get("ci95_low_pct", 0.0)),
                ci95_high_pct=float(row.get("ci95_high_pct", 0.0)),
                p_permutation=float(row.get("p_permutation", float("nan"))),
                effect_size_dz=float(row.get("effect_size_dz", 0.0)),
                claim_grade=str(row.get("claim_grade", "inconclusive")),
            )
        )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
