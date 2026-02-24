#!/usr/bin/env python3
"""Build task-level LongBench decomposition assets for paper writing."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METHOD_ORDER = ["baseline", "pi", "yarn", "sigmoid", "anchored_sigmoid"]
DELTA_COMPARATORS = ["baseline", "pi", "yarn", "sigmoid"]
TASK_ORDER = ["2wikimqa", "gov_report", "hotpotqa", "multi_news", "narrativeqa", "qasper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_dir", type=Path, default=None)
    parser.add_argument("--best_metrics_csv", type=Path, default=None)
    parser.add_argument("--task_scores_csv", type=Path, default=None)
    parser.add_argument("--out_pdf", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.best_metrics_csv and args.task_scores_csv:
        return args.best_metrics_csv, args.task_scores_csv
    if not args.report_dir:
        raise ValueError("Provide either --report_dir or both --best_metrics_csv and --task_scores_csv")
    return args.report_dir / "method_metrics_best_available.csv", args.report_dir / "longbench_task_scores.csv"


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def build_decomposition(best_csv: Path, scores_csv: Path) -> tuple[list[dict], dict[str, dict[str, float]]]:
    best_rows = _read_rows(best_csv)
    score_rows = _read_rows(scores_csv)

    best_profile = {r["method"]: r["source_profile"] for r in best_rows}
    selected = [
        r
        for r in score_rows
        if r.get("method") in best_profile and r.get("profile") == best_profile[r.get("method")]
    ]

    tasks = sorted({r["task"] for r in selected})
    ordered_tasks = [t for t in TASK_ORDER if t in tasks] + [t for t in tasks if t not in TASK_ORDER]
    if not ordered_tasks:
        raise ValueError(
            "No matching task rows found. Check --report_dir/CSV paths and source_profile alignment."
        )
    pivot: dict[str, dict[str, float]] = {
        task: {m: float("nan") for m in DEFAULT_METHOD_ORDER} for task in ordered_tasks
    }

    for r in selected:
        task, method = r["task"], r["method"]
        if task in pivot and method in pivot[task]:
            pivot[task][method] = _to_float(r["score"])

    table_rows: list[dict] = []
    for task in ordered_tasks:
        row = {"task_name": task}
        for m in DEFAULT_METHOD_ORDER:
            row[f"score_{m}"] = pivot[task][m]
        for m in DELTA_COMPARATORS:
            row[f"delta_anchored_minus_{m}"] = pivot[task]["anchored_sigmoid"] - pivot[task][m]
        table_rows.append(row)

    mean_row = {"task_name": "mean_over_tasks"}
    numeric_keys = [k for k in table_rows[0] if k != "task_name"]
    for key in numeric_keys:
        values = np.array([r[key] for r in table_rows], dtype=float)
        mean_row[key] = float(np.nanmean(values))
    table_rows.append(mean_row)

    return table_rows, pivot


def plot_task_deltas(pivot: dict[str, dict[str, float]], out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.45, 1.0]})

    tasks = list(pivot.keys())
    x = np.arange(len(tasks))
    width = 0.18

    colors = {
        "baseline": "#4C78A8",
        "pi": "#72B7B2",
        "yarn": "#F58518",
        "sigmoid": "#E45756",
    }

    for i, method in enumerate(DELTA_COMPARATORS):
        deltas = np.array([pivot[t]["anchored_sigmoid"] - pivot[t][method] for t in tasks], dtype=float)
        axes[0].bar(x + (i - 1.5) * width, deltas, width=width, label=f"anchored - {method}", color=colors[method])

    axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tasks, rotation=25, ha="right")
    axes[0].set_ylabel("Task-level score delta")
    axes[0].set_title("LongBench task heterogeneity (anchored-sigmoid vs comparators)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(axis="y", alpha=0.25)

    mean_scores = {}
    for method in DEFAULT_METHOD_ORDER:
        vals = np.array([pivot[t][method] for t in tasks], dtype=float)
        mean_scores[method] = float(np.nanmean(vals))
    mean_methods = sorted(mean_scores.keys(), key=lambda m: mean_scores[m], reverse=True)
    mean_values = np.array([mean_scores[m] for m in mean_methods], dtype=float)
    colors_h = ["#2F9E44" if m == "anchored_sigmoid" else "#9AA0A6" for m in mean_methods]
    axes[1].barh(mean_methods, mean_values, color=colors_h)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean LongBench score (6 tasks)")
    axes[1].set_title("Task-mean ranking")
    axes[1].grid(axis="x", alpha=0.25)

    for i, v in enumerate(mean_values):
        axes[1].text(v + 0.0004, i, f"{v:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    best_csv, scores_csv = resolve_paths(args)
    table_rows, pivot = build_decomposition(best_csv, scores_csv)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(table_rows[0].keys())
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in table_rows:
            w.writerow(row)
    plot_task_deltas(pivot, args.out_pdf)

    print(f"[ok] wrote CSV: {args.out_csv}")
    print(f"[ok] wrote PDF: {args.out_pdf}")


if __name__ == "__main__":
    main()
