#!/usr/bin/env python3
"""Paired statistics for EVQ-vs-Geometric from LongBench raw compare outputs.

Input files must be raw outputs produced by scripts/eval_longbench.py, containing:
- models.hybrid_lora.tasks.<task>.per_sample_traces

The script pairs samples by (task, index), computes paired diff = EVQ - geometric,
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
    """Extract a [0, 1] raw score from a per-sample trace dict.

    Priority order (first present wins):
      1. score_raw  — already in [0, 1], used directly.
      2. score_pct  — percentage [0, 100], divided by 100.
      3. score      — legacy key: if > 1.0 assumed pct and divided by 100,
                      otherwise used as raw.  Note: a true pct value of
                      exactly 1.0 (i.e. 1 %) would be mis-interpreted as
                      raw; this edge case is negligible for standard metrics.

    Current eval_longbench.py always emits both score_raw and score_pct, so
    the score fallback should never be reached in production runs.
    """
    raw = tr.get("score_raw")
    if isinstance(raw, (int, float)):
        return float(raw)
    pct = tr.get("score_pct")
    if isinstance(pct, (int, float)):
        return float(pct) / 100.0
    s = tr.get("score")
    if isinstance(s, (int, float)):
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


def collect_diffs(method_a_json: Path, geometric_json: Path, tasks: List[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    a_obj = load_json(method_a_json)
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
    mean = float(np.mean(diffs))
    std = float(np.std(diffs, ddof=1))
    if std <= 1e-12:
        if abs(mean) <= 1e-12:
            return 0.0
        return float("inf") if mean > 0 else float("-inf")
    return float(mean / std)


def claim_grade(mean_diff_pct: float, ci_low_pct: float, p_perm: float, run_pair_count: int, min_run_pairs: int) -> str:
    if int(run_pair_count) < int(max(1, min_run_pairs)):
        return "insufficient_seed_replication"
    if mean_diff_pct > 0 and ci_low_pct > 0 and p_perm < 0.05:
        return "significant"
    if mean_diff_pct > 0:
        return "directional"
    return "inconclusive"


def fdr_bh_adjust(pvals: List[float]) -> List[float]:
    n = len(pvals)
    if n == 0:
        return []
    indexed = []
    for i, p in enumerate(pvals):
        pv = float(p)
        if not math.isfinite(pv):
            pv = 1.0
        pv = min(1.0, max(0.0, pv))
        indexed.append((i, pv))
    indexed.sort(key=lambda x: x[1])
    adjusted_sorted = [1.0] * n
    prev = 1.0
    for rank in range(n, 0, -1):
        idx, pv = indexed[rank - 1]
        adj = min(prev, (pv * n) / float(rank))
        adjusted_sorted[rank - 1] = adj
        prev = adj
    adjusted = [1.0] * n
    for rank, (idx, _) in enumerate(indexed):
        adjusted[idx] = float(min(1.0, max(0.0, adjusted_sorted[rank])))
    return adjusted


def summarize_diffs(
    diffs: np.ndarray,
    n_bootstrap: int,
    seed: int,
    run_pair_count: int,
    min_run_pairs: int,
) -> Dict[str, float]:
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
        "claim_grade": claim_grade(
            mean_diff_pct=mean_pct,
            ci_low_pct=ci_low_pct,
            p_perm=float(p_perm),
            run_pair_count=int(run_pair_count),
            min_run_pairs=int(min_run_pairs),
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired stats for llama8k theory validation (EVQ vs Geometric)")
    ap.add_argument(
        "--evq_jsons",
        type=str,
        default="",
        help="Comma-separated list of EVQ raw compare json files.",
    )
    ap.add_argument(
        "--anchored_jsons",
        type=str,
        default="",
        help="Deprecated alias for --evq_jsons (kept for backward compatibility).",
    )
    ap.add_argument(
        "--method_a_jsons",
        type=str,
        default="",
        help="Generic alias for method-A jsons. Prefer --evq_jsons for this pipeline.",
    )
    ap.add_argument("--method_a_name", type=str, default="evq_cosh", help="Method-A name used in metadata/report labels.")
    ap.add_argument("--geometric_jsons", type=str, required=True, help="Comma-separated list of geometric raw compare json files")
    ap.add_argument("--tasks", type=str, default=",".join(LB21_TASKS))
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--claim_min_run_pairs", type=int, default=2)
    ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument("--output_md", type=str, required=True)
    args = ap.parse_args()

    method_a_name = "_".join(str(args.method_a_name).strip().lower().split()) or "evq_cosh"
    method_a_csv = (
        str(args.evq_jsons).strip()
        or str(args.method_a_jsons).strip()
        or str(args.anchored_jsons).strip()
    )
    if not method_a_csv:
        raise ValueError("Missing method-a jsons: provide --evq_jsons (or --method_a_jsons / legacy --anchored_jsons).")
    method_paths = [Path(p) for p in parse_csv(method_a_csv)]
    geometric_paths = [Path(p) for p in parse_csv(args.geometric_jsons)]
    if len(method_paths) != len(geometric_paths):
        raise ValueError("evq/method_a jsons and geometric_jsons must have the same count")
    tasks = parse_csv(args.tasks)

    pooled_all: List[float] = []
    per_task_all: Dict[str, List[float]] = {t: [] for t in tasks}
    run_pairs: List[Dict[str, str]] = []

    for a_path, g_path in zip(method_paths, geometric_paths):
        if not a_path.exists() or not g_path.exists():
            continue
        run_pairs.append({"method_a": a_path.as_posix(), "geometric": g_path.as_posix()})
        per_task, pooled = collect_diffs(a_path, g_path, tasks)
        pooled_all.extend(pooled.tolist())
        for task, diffs in per_task.items():
            per_task_all.setdefault(task, []).extend(diffs.tolist())

    if len(run_pairs) == 0:
        raise RuntimeError("No valid run pairs found. Both method-a and geometric jsons must exist on disk.")
    pooled_arr = np.asarray(pooled_all, dtype=np.float64)
    if pooled_arr.size == 0:
        raise RuntimeError("No paired per-sample traces found across provided run pairs/tasks.")
    pooled_summary = summarize_diffs(
        diffs=pooled_arr,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
        run_pair_count=len(run_pairs),
        min_run_pairs=int(args.claim_min_run_pairs),
    )

    per_task_summary: Dict[str, Dict[str, float]] = {}
    for task in tasks:
        vals = np.asarray(per_task_all.get(task, []), dtype=np.float64)
        if vals.size == 0:
            continue
        per_task_summary[task] = summarize_diffs(
            diffs=vals,
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed) + 17,
            run_pair_count=len(run_pairs),
            min_run_pairs=int(args.claim_min_run_pairs),
        )

    fdr_tasks = list(per_task_summary.keys())
    fdr_raw = [float(per_task_summary[t].get("p_permutation", 1.0)) for t in fdr_tasks]
    fdr_adj = fdr_bh_adjust(fdr_raw)
    for t, p_adj in zip(fdr_tasks, fdr_adj):
        row = per_task_summary[t]
        row["p_permutation_fdr_bh"] = float(p_adj)
        ci_low_pct = float(row.get("ci95_low_pct", 0.0))
        mean_diff_pct = float(row.get("mean_diff_pct", 0.0))
        if int(len(run_pairs)) < int(max(1, args.claim_min_run_pairs)):
            row["claim_grade_fdr_bh"] = "insufficient_seed_replication"
        elif mean_diff_pct > 0 and ci_low_pct > 0 and float(p_adj) < 0.05:
            row["claim_grade_fdr_bh"] = "significant_fdr_bh"
        elif mean_diff_pct > 0:
            row["claim_grade_fdr_bh"] = "directional"
        else:
            row["claim_grade_fdr_bh"] = "inconclusive"

    # Task-weighted macro pooled: equal weight per task (avoids domination by
    # high-sample tasks like triviaqa).  Construct one mean-diff per task,
    # then bootstrap / permutation-test across task means.
    task_means: List[float] = []
    for task in per_task_summary:
        task_means.append(float(per_task_summary[task].get("mean_diff_raw", 0.0)))
    if task_means:
        task_means_arr = np.asarray(task_means, dtype=np.float64)
        macro_pooled_summary = summarize_diffs(
            diffs=task_means_arr,
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed) + 31,
            run_pair_count=len(run_pairs),
            min_run_pairs=int(args.claim_min_run_pairs),
        )
        macro_pooled_summary["n_tasks"] = len(task_means)
        macro_pooled_summary["note"] = (
            "equal weight per task; each task contributes one mean diff value"
        )
    else:
        macro_pooled_summary = {}

    out = {
        "meta": {
            "tasks": tasks,
            "run_pairs": run_pairs,
            "n_bootstrap": int(args.n_bootstrap),
            "seed": int(args.seed),
            "score_unit": "raw_and_pct",
            "method_a_name": method_a_name,
            "comparison": f"{method_a_name}_minus_geometric",
            "run_pair_count": int(len(run_pairs)),
            "claim_min_run_pairs": int(args.claim_min_run_pairs),
            "seed_replication_ok": bool(len(run_pairs) >= int(max(1, args.claim_min_run_pairs))),
            "inference_scope": (
                "paired sample-level difference over provided model-pair outputs; "
                "does not estimate cross-seed training-run variance unless multiple "
                "full-eval training seeds are included in run_pairs"
            ),
        },
        "pooled": pooled_summary,
        "pooled_macro": macro_pooled_summary,
        "per_task": per_task_summary,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    save_json(output_json, out)

    lines = [
        f"# Paired Statistics: {method_a_name} vs Geometric",
        "",
        f"- run_pairs: `{len(run_pairs)}`",
        f"- claim_min_run_pairs: `{int(args.claim_min_run_pairs)}`",
        f"- seed_replication_ok: `{bool(len(run_pairs) >= int(max(1, args.claim_min_run_pairs)))}`",
        f"- inference_scope: `paired sample-level difference over provided {method_a_name}/geometric model pairs`",
        "- caveat: `This is not cross-seed training significance unless multiple full-eval training seeds are present.`",
        f"- claim_grade: `{pooled_summary.get('claim_grade', 'inconclusive')}`",
        f"- pooled_mean_diff_pct: `{pooled_summary.get('mean_diff_pct', 0.0):.4f}`",
        f"- pooled_ci95_pct: `[{pooled_summary.get('ci95_low_pct', 0.0):.4f}, {pooled_summary.get('ci95_high_pct', 0.0):.4f}]`",
        f"- p_sign_flip: `{pooled_summary.get('p_sign_flip', float('nan')):.6f}`",
        f"- p_permutation: `{pooled_summary.get('p_permutation', float('nan')):.6f}`",
        f"- effect_size_dz: `{pooled_summary.get('effect_size_dz', 0.0):.6f}`",
        "",
        "## Macro-pooled (equal weight per task)",
        "",
        f"- macro_claim_grade: `{macro_pooled_summary.get('claim_grade', 'N/A')}`",
        f"- macro_mean_diff_pct: `{float(macro_pooled_summary.get('mean_diff_pct', 0.0)):.4f}`",
        f"- macro_ci95_pct: `[{float(macro_pooled_summary.get('ci95_low_pct', 0.0)):.4f}, {float(macro_pooled_summary.get('ci95_high_pct', 0.0)):.4f}]`",
        f"- macro_p_permutation: `{float(macro_pooled_summary.get('p_permutation', float('nan'))):.6f}`",
        f"- macro_effect_size_dz: `{float(macro_pooled_summary.get('effect_size_dz', 0.0)):.6f}`",
        f"- macro_n_tasks: `{int(macro_pooled_summary.get('n_tasks', 0))}`",
        "",
        "## Per-task details",
        "",
        "| task | n_pairs | mean_diff_pct | ci95_low_pct | ci95_high_pct | p_permutation | p_permutation_fdr_bh | effect_size_dz | claim_grade | claim_grade_fdr_bh |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for task, row in per_task_summary.items():
        lines.append(
            "| {task} | {n_pairs} | {mean_diff_pct:.4f} | {ci95_low_pct:.4f} | {ci95_high_pct:.4f} | {p_permutation:.6f} | {p_permutation_fdr_bh:.6f} | {effect_size_dz:.6f} | {claim_grade} | {claim_grade_fdr_bh} |".format(
                task=task,
                n_pairs=int(row.get("n_pairs", 0)),
                mean_diff_pct=float(row.get("mean_diff_pct", 0.0)),
                ci95_low_pct=float(row.get("ci95_low_pct", 0.0)),
                ci95_high_pct=float(row.get("ci95_high_pct", 0.0)),
                p_permutation=float(row.get("p_permutation", float("nan"))),
                p_permutation_fdr_bh=float(row.get("p_permutation_fdr_bh", float("nan"))),
                effect_size_dz=float(row.get("effect_size_dz", 0.0)),
                claim_grade=str(row.get("claim_grade", "inconclusive")),
                claim_grade_fdr_bh=str(row.get("claim_grade_fdr_bh", "inconclusive")),
            )
        )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
