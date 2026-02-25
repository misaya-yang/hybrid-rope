#!/usr/bin/env python3
"""
Paired significance testing for LongBench results.

Supports three evidence levels:
1) per-task paired tests
2) per-sample pooled paired tests
3) cross-seed paired tests (when multiple roots are provided)

Expected result JSON structure includes task-level `per_sample_scores`
from scripts/eval_longbench.py.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_TASKS = [
    "multi_news",
    "narrativeqa",
    "qasper",
    "2wikimqa",
    "hotpotqa",
    "gov_report",
]

METHODS = ["baseline", "pi", "yarn", "sigmoid", "anchored_sigmoid", "ntk_aware", "longrope"]
COMPARISONS = [
    ("anchored_sigmoid", "baseline", "Anc-Sig vs Baseline"),
    ("anchored_sigmoid", "pi", "Anc-Sig vs PI"),
    ("anchored_sigmoid", "yarn", "Anc-Sig vs YaRN"),
    ("anchored_sigmoid", "sigmoid", "Anc-Sig vs Sigmoid (ablation)"),
    ("anchored_sigmoid", "ntk_aware", "Anc-Sig vs NTK-aware"),
    ("anchored_sigmoid", "longrope", "Anc-Sig vs LongRoPE"),
    ("sigmoid", "baseline", "Sigmoid vs Baseline"),
    ("sigmoid", "pi", "Sigmoid vs PI"),
]


@dataclass
class PairStats:
    n_pairs: int
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_bootstrap: float
    p_signflip: float


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def resolve_data_dir(data_dir: str) -> Path:
    """Resolve user-provided data_dir, including common batch_report wrappers."""
    base = Path(data_dir)
    if base.exists():
        return base

    cwd = Path.cwd()
    for batch_root in sorted(cwd.glob("batch_report_*")):
        candidate = batch_root / data_dir
        if candidate.exists():
            return candidate

    return base


def _extract_scores(obj) -> Optional[np.ndarray]:
    if obj is None:
        return None

    if isinstance(obj, list):
        if not obj:
            return None
        if all(isinstance(x, (int, float)) for x in obj):
            return np.asarray(obj, dtype=float)
        if all(isinstance(x, dict) for x in obj):
            for key in ("score", "f1", "rouge", "accuracy", "em"):
                vals = [x.get(key) for x in obj if isinstance(x.get(key), (int, float))]
                if vals:
                    return np.asarray(vals, dtype=float)
        return None

    if isinstance(obj, dict):
        # Prefer canonical raw scores for robust cross-run comparability.
        for key in ("per_sample_scores_raw", "per_sample_scores", "scores", "results"):
            vals = obj.get(key)
            if isinstance(vals, list) and vals and all(isinstance(x, (int, float)) for x in vals):
                return np.asarray(vals, dtype=float)

        examples = obj.get("examples")
        if isinstance(examples, list):
            vals = [
                x.get("score_raw")
                for x in examples
                if isinstance(x, dict) and isinstance(x.get("score_raw"), (int, float))
            ]
            if vals:
                return np.asarray(vals, dtype=float)
            vals = [
                x.get("score")
                for x in examples
                if isinstance(x, dict) and isinstance(x.get("score"), (int, float))
            ]
            if vals:
                return np.asarray(vals, dtype=float)

    return None


def load_per_sample_scores(data_dir: str, method: str, task: str) -> Optional[np.ndarray]:
    candidates = [
        os_path(data_dir, method, "longbench", f"{task}.json"),
        os_path(data_dir, method, "longbench", f"{task}_results.json"),
        os_path(data_dir, method, f"longbench_{task}.json"),
        os_path(data_dir, "downstream_eval_autorun", "longbench", f"{method}.json"),
        os_path(data_dir, "downstream_eval_parallel_seed42_m2", "longbench", f"{method}.json"),
    ]
    candidates.extend(
        sorted(glob.glob(os_path(data_dir, "downstream_eval_*", "longbench", f"{method}.json")))
    )
    candidates = list(dict.fromkeys(candidates))

    for path in candidates:
        p = Path(path)
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        # direct task-keyed or root-level extraction
        scores = _extract_scores(data if not isinstance(data, dict) else data.get(task, data))
        if scores is not None and len(scores) > 0:
            return scores

        # consolidated format
        if isinstance(data, dict) and isinstance(data.get("models"), dict):
            models = data["models"]
            preferred = [k for k in ("hybrid_lora", method) if k in models]
            preferred += [k for k in models.keys() if k not in preferred and k != "base_unfinetuned"]
            if "base_unfinetuned" in models:
                preferred.append("base_unfinetuned")

            for model_key in preferred:
                task_blob = models.get(model_key, {}).get("tasks", {}).get(task)
                scores = _extract_scores(task_blob)
                if scores is not None and len(scores) > 0:
                    return scores

    single_file = Path(os_path(data_dir, method, "longbench", "results.json"))
    if single_file.exists():
        try:
            data = json.loads(single_file.read_text(encoding="utf-8"))
            scores = _extract_scores(data.get(task) if isinstance(data, dict) else None)
            if scores is not None and len(scores) > 0:
                return scores
        except Exception:
            pass

    return None


def os_path(*parts: str) -> str:
    return str(Path(*parts))


def bootstrap_diff_from_diffs(diffs: np.ndarray, n_bootstrap: int, seed: int) -> Tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = int(diffs.shape[0])
    observed = float(np.mean(diffs))

    boot = np.zeros(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(diffs[idx]))

    centered = boot - float(np.mean(boot))
    p_boot = float(np.mean(np.abs(centered) >= abs(observed)))
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))
    return observed, p_boot, ci_lo, ci_hi


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


def fdr_bh(pvals: List[float]) -> List[float]:
    m = len(pvals)
    if m == 0:
        return []
    pairs = [(i, float(p)) for i, p in enumerate(pvals)]
    pairs.sort(key=lambda x: x[1])
    adj_sorted = [0.0] * m
    for rank, (_, p) in enumerate(pairs, start=1):
        adj_sorted[rank - 1] = p * m / float(rank)
    # monotonic from tail
    for i in range(m - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    out = [0.0] * m
    for rank, (idx, _) in enumerate(pairs):
        out[idx] = float(min(max(adj_sorted[rank], 0.0), 1.0))
    return out


def fdr_by(pvals: List[float]) -> List[float]:
    m = len(pvals)
    if m == 0:
        return []
    c_m = float(np.sum(1.0 / np.arange(1, m + 1, dtype=np.float64)))
    pairs = [(i, float(p)) for i, p in enumerate(pvals)]
    pairs.sort(key=lambda x: x[1])
    adj_sorted = [0.0] * m
    for rank, (_, p) in enumerate(pairs, start=1):
        adj_sorted[rank - 1] = p * m * c_m / float(rank)
    for i in range(m - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    out = [0.0] * m
    for rank, (idx, _) in enumerate(pairs):
        out[idx] = float(min(max(adj_sorted[rank], 0.0), 1.0))
    return out


def claim_grade(diff: float, p_fdr_bh: Optional[float], p_raw: Optional[float]) -> str:
    if not np.isfinite(diff):
        return "insufficient_data"
    if diff <= 0:
        return "no_improvement"
    if isinstance(p_fdr_bh, (int, float)) and np.isfinite(float(p_fdr_bh)) and float(p_fdr_bh) < 0.05:
        return "significant_improvement"
    if isinstance(p_raw, (int, float)) and np.isfinite(float(p_raw)):
        return "directional_consistent_with_theory"
    return "directional_consistent_with_theory"


def paired_stats(a: np.ndarray, b: np.ndarray, n_bootstrap: int, seed: int) -> PairStats:
    n = min(len(a), len(b))
    if n <= 0:
        return PairStats(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    a_use = np.asarray(a[:n], dtype=np.float64)
    b_use = np.asarray(b[:n], dtype=np.float64)
    diffs = a_use - b_use
    diff, p_boot, ci_lo, ci_hi = bootstrap_diff_from_diffs(diffs=diffs, n_bootstrap=n_bootstrap, seed=seed)
    p_flip = sign_flip_pvalue(diffs=diffs, seed=seed)

    return PairStats(
        n_pairs=n,
        mean_a=float(np.mean(a_use)),
        mean_b=float(np.mean(b_use)),
        diff=float(diff),
        ci_low=float(ci_lo),
        ci_high=float(ci_hi),
        p_bootstrap=float(p_boot),
        p_signflip=float(p_flip),
    )


def hierarchical_bootstrap_diff(
    by_seed_task_diffs: Dict[str, Dict[str, np.ndarray]],
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float, float]:
    """Two-level bootstrap: seed -> task -> sample."""
    rng = np.random.default_rng(seed)
    seed_keys = [k for k, v in by_seed_task_diffs.items() if v]
    if not seed_keys:
        return float("nan"), float("nan"), float("nan")

    observed_vals = []
    for sk in seed_keys:
        task_vals = [float(np.mean(arr)) for arr in by_seed_task_diffs[sk].values() if len(arr) > 0]
        if task_vals:
            observed_vals.append(float(np.mean(task_vals)))
    observed = float(np.mean(observed_vals)) if observed_vals else float("nan")

    boot = np.zeros(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        chosen_seeds = rng.choice(seed_keys, size=len(seed_keys), replace=True)
        seed_means = []
        for sk in chosen_seeds:
            tmap = by_seed_task_diffs.get(str(sk), {})
            task_keys = [t for t, arr in tmap.items() if len(arr) > 0]
            if not task_keys:
                continue
            chosen_tasks = rng.choice(task_keys, size=len(task_keys), replace=True)
            task_means = []
            for tk in chosen_tasks:
                arr = np.asarray(tmap[tk], dtype=np.float64)
                idx = rng.integers(0, len(arr), size=len(arr))
                task_means.append(float(np.mean(arr[idx])))
            if task_means:
                seed_means.append(float(np.mean(task_means)))
        boot[i] = float(np.mean(seed_means)) if seed_means else 0.0

    return observed, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed-aware significance testing for LongBench")
    parser.add_argument("--data_dir", required=True, help="Primary result root")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument(
        "--task_list",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task list to include",
    )
    parser.add_argument(
        "--seed_grouped",
        type=str,
        default="",
        help="Comma-separated extra result roots for additional seeds",
    )
    parser.add_argument(
        "--hierarchical_bootstrap",
        action="store_true",
        help="Enable two-level hierarchical bootstrap (seed->task->sample) for pooled analysis",
    )
    parser.add_argument(
        "--fdr_method",
        type=str,
        default="both",
        choices=["bh", "by", "both"],
        help="FDR correction method(s) applied over each (comparison, level, family) group.",
    )
    parser.add_argument("--output_prefix", type=str, default="significance_seeded")
    args = parser.parse_args()

    tasks = parse_csv(args.task_list)
    if not tasks:
        raise ValueError("task_list is empty")

    roots = [resolve_data_dir(args.data_dir)]
    roots.extend(resolve_data_dir(x) for x in parse_csv(args.seed_grouped))

    dedup_roots: List[Path] = []
    seen = set()
    for r in roots:
        rr = r.resolve()
        if rr in seen:
            continue
        seen.add(rr)
        dedup_roots.append(rr)
    roots = dedup_roots

    missing = [str(r) for r in roots if not r.exists()]
    if missing:
        raise FileNotFoundError(f"Result root(s) not found: {missing}")

    print(f"[info] tasks={tasks}")
    print(f"[info] roots={roots}")

    all_scores: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for ridx, root in enumerate(roots):
        seed_key = f"seed_{ridx}"
        all_scores[seed_key] = {m: {} for m in METHODS}
        for method in METHODS:
            for task in tasks:
                arr = load_per_sample_scores(str(root), method, task)
                if arr is not None and len(arr) > 0:
                    all_scores[seed_key][method][task] = np.asarray(arr, dtype=np.float64)

    # sanity summary
    print("\n" + "=" * 72)
    print("SANITY CHECK: method averages by seed")
    print("=" * 72)
    for seed_key, per_method in all_scores.items():
        print(f"[{seed_key}]")
        for method in METHODS:
            vals = [float(np.mean(per_method[method][t])) for t in tasks if t in per_method[method]]
            if vals:
                print(f"  {method:18s}: {float(np.mean(vals)):.4f}")

    rows_csv: List[Dict[str, object]] = []
    correction_rows: List[Dict[str, object]] = []
    output_payload: Dict[str, object] = {
        "meta": {
            "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "roots": [str(r) for r in roots],
            "tasks": tasks,
            "n_bootstrap": int(args.n_bootstrap),
            "hierarchical_bootstrap": bool(args.hierarchical_bootstrap),
            "fdr_method": args.fdr_method,
        },
        "comparisons": [],
    }

    for method_a, method_b, label in COMPARISONS:
        comp_obj: Dict[str, object] = {
            "comparison": label,
            "method_a": method_a,
            "method_b": method_b,
            "per_task": {},
            "per_sample": {},
            "cross_seed": {},
            "fdr_summary": {},
        }

        # per-task pooled over seeds (within each seed first, then concatenate pairs)
        for task in tasks:
            a_parts = []
            b_parts = []
            for seed_key in all_scores:
                a = all_scores[seed_key][method_a].get(task)
                b = all_scores[seed_key][method_b].get(task)
                if a is None or b is None:
                    continue
                n = min(len(a), len(b))
                if n <= 0:
                    continue
                a_parts.append(a[:n])
                b_parts.append(b[:n])
            if not a_parts:
                continue

            a_all = np.concatenate(a_parts)
            b_all = np.concatenate(b_parts)
            st = paired_stats(a_all, b_all, n_bootstrap=args.n_bootstrap, seed=42)
            comp_obj["per_task"][task] = {
                "n_pairs": st.n_pairs,
                "mean_a": st.mean_a,
                "mean_b": st.mean_b,
                "diff": st.diff,
                "ci_low": st.ci_low,
                "ci_high": st.ci_high,
                "p_bootstrap": st.p_bootstrap,
                "p_signflip": st.p_signflip,
                "p_raw": st.p_bootstrap,
                "p_fdr_bh": None,
                "p_fdr_by": None,
                "claim_grade": None,
                "family": f"{label}|per_task",
            }
            rows_csv.append(
                {
                    "comparison": label,
                    "level": "per_task",
                    "task": task,
                    "family": f"{label}|per_task",
                    "n_pairs": st.n_pairs,
                    "diff": st.diff,
                    "ci_low": st.ci_low,
                    "ci_high": st.ci_high,
                    "p_bootstrap": st.p_bootstrap,
                    "p_signflip": st.p_signflip,
                    "p_raw": st.p_bootstrap,
                    "p_fdr_bh": None,
                    "p_fdr_by": None,
                    "claim_grade": None,
                }
            )
            correction_rows.append(
                {
                    "comparison": label,
                    "level": "per_task",
                    "task": task,
                    "family": f"{label}|per_task",
                    "p_raw": float(st.p_bootstrap),
                    "diff": float(st.diff),
                }
            )

        # per-sample pooled analysis
        pooled_a = []
        pooled_b = []
        by_seed_task_diffs: Dict[str, Dict[str, np.ndarray]] = {}
        for seed_key in all_scores:
            by_seed_task_diffs[seed_key] = {}
            for task in tasks:
                a = all_scores[seed_key][method_a].get(task)
                b = all_scores[seed_key][method_b].get(task)
                if a is None or b is None:
                    continue
                n = min(len(a), len(b))
                if n <= 0:
                    continue
                a_use = np.asarray(a[:n], dtype=np.float64)
                b_use = np.asarray(b[:n], dtype=np.float64)
                pooled_a.append(a_use)
                pooled_b.append(b_use)
                by_seed_task_diffs[seed_key][task] = a_use - b_use

        if pooled_a:
            a_cat = np.concatenate(pooled_a)
            b_cat = np.concatenate(pooled_b)
            st = paired_stats(a_cat, b_cat, n_bootstrap=args.n_bootstrap, seed=42)
            per_sample_obj = {
                "n_pairs": st.n_pairs,
                "mean_a": st.mean_a,
                "mean_b": st.mean_b,
                "diff": st.diff,
                "ci_low": st.ci_low,
                "ci_high": st.ci_high,
                "p_bootstrap": st.p_bootstrap,
                "p_signflip": st.p_signflip,
                "p_raw": st.p_bootstrap,
                "p_fdr_bh": None,
                "p_fdr_by": None,
                "claim_grade": None,
                "family": f"{label}|per_sample",
            }

            if args.hierarchical_bootstrap:
                h_obs, h_lo, h_hi = hierarchical_bootstrap_diff(
                    by_seed_task_diffs=by_seed_task_diffs,
                    n_bootstrap=args.n_bootstrap,
                    seed=52,
                )
                per_sample_obj["hierarchical_diff"] = h_obs
                per_sample_obj["hierarchical_ci_low"] = h_lo
                per_sample_obj["hierarchical_ci_high"] = h_hi

            comp_obj["per_sample"] = per_sample_obj
            rows_csv.append(
                {
                    "comparison": label,
                    "level": "per_sample",
                    "task": "all",
                    "family": f"{label}|per_sample",
                    "n_pairs": st.n_pairs,
                    "diff": st.diff,
                    "ci_low": st.ci_low,
                    "ci_high": st.ci_high,
                    "p_bootstrap": st.p_bootstrap,
                    "p_signflip": st.p_signflip,
                    "p_raw": st.p_bootstrap,
                    "p_fdr_bh": None,
                    "p_fdr_by": None,
                    "claim_grade": None,
                }
            )
            correction_rows.append(
                {
                    "comparison": label,
                    "level": "per_sample",
                    "task": "all",
                    "family": f"{label}|per_sample",
                    "p_raw": float(st.p_bootstrap),
                    "diff": float(st.diff),
                }
            )

        # cross-seed analysis (seed-level aggregated deltas)
        seed_deltas = []
        for seed_key in all_scores:
            task_means = []
            for task in tasks:
                a = all_scores[seed_key][method_a].get(task)
                b = all_scores[seed_key][method_b].get(task)
                if a is None or b is None:
                    continue
                n = min(len(a), len(b))
                if n <= 0:
                    continue
                task_means.append(float(np.mean(a[:n] - b[:n])))
            if task_means:
                seed_deltas.append(float(np.mean(task_means)))

        if seed_deltas:
            arr = np.asarray(seed_deltas, dtype=np.float64)
            obs, p_boot, ci_lo, ci_hi = bootstrap_diff_from_diffs(arr, n_bootstrap=args.n_bootstrap, seed=62)
            p_flip = sign_flip_pvalue(arr, seed=62)
            comp_obj["cross_seed"] = {
                "n_seeds": int(len(arr)),
                "seed_deltas": [float(x) for x in arr.tolist()],
                "diff": float(obs),
                "ci_low": float(ci_lo),
                "ci_high": float(ci_hi),
                "p_bootstrap": float(p_boot),
                "p_signflip": float(p_flip),
                "p_raw": float(p_boot),
                "p_fdr_bh": None,
                "p_fdr_by": None,
                "claim_grade": None,
                "family": f"{label}|cross_seed",
            }
            rows_csv.append(
                {
                    "comparison": label,
                    "level": "cross_seed",
                    "task": "seed_mean",
                    "family": f"{label}|cross_seed",
                    "n_pairs": int(len(arr)),
                    "diff": float(obs),
                    "ci_low": float(ci_lo),
                    "ci_high": float(ci_hi),
                    "p_bootstrap": float(p_boot),
                    "p_signflip": float(p_flip),
                    "p_raw": float(p_boot),
                    "p_fdr_bh": None,
                    "p_fdr_by": None,
                    "claim_grade": None,
                }
            )
            correction_rows.append(
                {
                    "comparison": label,
                    "level": "cross_seed",
                    "task": "seed_mean",
                    "family": f"{label}|cross_seed",
                    "p_raw": float(p_boot),
                    "diff": float(obs),
                }
            )

        output_payload["comparisons"].append(comp_obj)

    # Apply FDR correction by family and attach claim grades.
    families: Dict[str, List[Dict[str, object]]] = {}
    for r in correction_rows:
        fam = str(r.get("family", ""))
        families.setdefault(fam, []).append(r)

    correction_map: Dict[Tuple[str, str, str], Dict[str, Optional[float]]] = {}
    for fam, items in families.items():
        pvals = [float(x.get("p_raw", 1.0)) for x in items]
        p_bh = fdr_bh(pvals) if args.fdr_method in {"bh", "both"} else [None] * len(pvals)
        p_by = fdr_by(pvals) if args.fdr_method in {"by", "both"} else [None] * len(pvals)
        for i, item in enumerate(items):
            key = (
                str(item.get("comparison", "")),
                str(item.get("level", "")),
                str(item.get("task", "")),
            )
            raw = float(item.get("p_raw", 1.0))
            bh = p_bh[i] if isinstance(p_bh[i], float) else None
            by = p_by[i] if isinstance(p_by[i], float) else None
            diff = float(item.get("diff", float("nan")))
            grade = claim_grade(diff=diff, p_fdr_bh=bh, p_raw=raw)
            correction_map[key] = {
                "p_raw": raw,
                "p_fdr_bh": bh,
                "p_fdr_by": by,
                "claim_grade": grade,
            }

    # Patch CSV rows.
    for row in rows_csv:
        key = (str(row.get("comparison", "")), str(row.get("level", "")), str(row.get("task", "")))
        corr = correction_map.get(key)
        if not corr:
            continue
        row["p_raw"] = corr["p_raw"]
        row["p_fdr_bh"] = corr["p_fdr_bh"]
        row["p_fdr_by"] = corr["p_fdr_by"]
        row["claim_grade"] = corr["claim_grade"]

    # Patch JSON payload and per-comparison family summaries.
    for comp in output_payload["comparisons"]:
        label = str(comp.get("comparison", ""))
        fsum = {}

        per_task = comp.get("per_task") or {}
        if isinstance(per_task, dict):
            for task, obj in per_task.items():
                key = (label, "per_task", str(task))
                corr = correction_map.get(key)
                if corr and isinstance(obj, dict):
                    obj["p_raw"] = corr["p_raw"]
                    obj["p_fdr_bh"] = corr["p_fdr_bh"]
                    obj["p_fdr_by"] = corr["p_fdr_by"]
                    obj["claim_grade"] = corr["claim_grade"]

        per_sample = comp.get("per_sample") or {}
        if isinstance(per_sample, dict) and per_sample:
            key = (label, "per_sample", "all")
            corr = correction_map.get(key)
            if corr:
                per_sample["p_raw"] = corr["p_raw"]
                per_sample["p_fdr_bh"] = corr["p_fdr_bh"]
                per_sample["p_fdr_by"] = corr["p_fdr_by"]
                per_sample["claim_grade"] = corr["claim_grade"]

        cross_seed = comp.get("cross_seed") or {}
        if isinstance(cross_seed, dict) and cross_seed:
            key = (label, "cross_seed", "seed_mean")
            corr = correction_map.get(key)
            if corr:
                cross_seed["p_raw"] = corr["p_raw"]
                cross_seed["p_fdr_bh"] = corr["p_fdr_bh"]
                cross_seed["p_fdr_by"] = corr["p_fdr_by"]
                cross_seed["claim_grade"] = corr["claim_grade"]

        fam_key = f"{label}|per_task"
        fam_items = families.get(fam_key, [])
        if fam_items:
            ps = [float(x.get("p_raw", 1.0)) for x in fam_items]
            fsum[fam_key] = {
                "n_hypotheses": len(ps),
                "min_p_raw": float(min(ps)),
                "method": args.fdr_method,
            }

        comp["fdr_summary"] = fsum

    out_root = roots[0]
    out_json = out_root / f"{args.output_prefix}.json"
    out_csv = out_root / f"{args.output_prefix}.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "comparison",
                "level",
                "task",
                "family",
                "n_pairs",
                "diff",
                "ci_low",
                "ci_high",
                "p_bootstrap",
                "p_signflip",
                "p_raw",
                "p_fdr_bh",
                "p_fdr_by",
                "claim_grade",
            ],
        )
        writer.writeheader()
        for row in rows_csv:
            writer.writerow(row)

    claim_md = out_root / "claim_policy_report.md"
    md_lines = [
        "# Claim Policy Report",
        "",
        "Policy:",
        "- Use `significant improvement` only when `p_fdr_bh < 0.05` and `diff > 0`.",
        "- Otherwise use `directional improvement consistent with theory (p_raw=..., FDR-adjusted p=...)`.",
        "",
        "| Comparison | Level | diff | p_raw | p_fdr_bh | p_fdr_by | claim_grade |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows_csv:
        if str(row.get("level")) != "per_sample":
            continue
        md_lines.append(
            "| {comparison} | {level} | {diff:.6f} | {p_raw:.6f} | {p_fdr_bh} | {p_fdr_by} | {claim_grade} |".format(
                comparison=str(row.get("comparison", "")),
                level=str(row.get("level", "")),
                diff=float(row.get("diff", 0.0)),
                p_raw=float(row.get("p_raw", 1.0)),
                p_fdr_bh=("NA" if row.get("p_fdr_bh") is None else f"{float(row.get('p_fdr_bh')):.6f}"),  # type: ignore[arg-type]
                p_fdr_by=("NA" if row.get("p_fdr_by") is None else f"{float(row.get('p_fdr_by')):.6f}"),  # type: ignore[arg-type]
                claim_grade=str(row.get("claim_grade", "")),
            )
        )
    claim_md.write_text("\\n".join(md_lines) + "\\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print("SIGNIFICANCE SUMMARY (per_sample level)")
    print("=" * 72)
    for comp in output_payload["comparisons"]:
        ps = comp.get("per_sample") or {}
        if not ps:
            continue
        p = ps.get("p_raw", ps.get("p_bootstrap"))
        p_bh = ps.get("p_fdr_bh")
        grade = ps.get("claim_grade", "n/a")
        p_show = float(p) if isinstance(p, (int, float)) else float("nan")
        if isinstance(p_bh, (int, float)):
            status = "SIGNIFICANT" if float(p_bh) < 0.05 else "NOT significant"
            print(f"{comp['comparison']:30s} p_raw={p_show:.4f} p_fdr_bh={float(p_bh):.4f} {status} grade={grade}")
        else:
            status = "NOT significant" if isinstance(p, (int, float)) and float(p) >= 0.05 else "SIGNIFICANT"
            print(f"{comp['comparison']:30s} p_raw={p_show:.4f} {status} grade={grade}")

    print(f"\n[ok] wrote json: {out_json}")
    print(f"[ok] wrote csv : {out_csv}")


if __name__ == "__main__":
    main()
