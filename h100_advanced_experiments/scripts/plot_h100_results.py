#!/usr/bin/env python3
"""
Aggregate RoPE experiment JSON results and produce:
1) CSV summary
2) Markdown table
3) PPL-vs-length line chart
4) PPL@16K bar chart (mean +/- std)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _as_ppl(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        ppl = value.get("ppl")
        if isinstance(ppl, (int, float)):
            return float(ppl)
    return None


def extract_rows(data: dict, source: str) -> List[dict]:
    rows: List[dict] = []
    seed = data.get("seed")
    if seed is None and isinstance(data.get("meta"), dict):
        seed = data["meta"].get("seed")

    experiments = data.get("experiments")
    if not isinstance(experiments, dict):
        return rows

    for config_name, length_map in experiments.items():
        if not isinstance(length_map, dict):
            continue
        for length_key, metric in length_map.items():
            try:
                length = int(length_key)
            except Exception:
                continue
            ppl = _as_ppl(metric)
            if ppl is None or not math.isfinite(ppl):
                continue
            rows.append(
                {
                    "source": source,
                    "config": str(config_name),
                    "seed": seed if isinstance(seed, int) else None,
                    "length": length,
                    "ppl": float(ppl),
                }
            )
    return rows


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from exc
    return plt


def collect_rows(input_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(input_dir.rglob("*.json")):
        # Skip schema/config JSON files.
        if "schema" in path.name.lower():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            rows.extend(extract_rows(data, str(path)))
    return rows


def summarize(rows: Iterable[dict]) -> List[dict]:
    groups: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    seeds: Dict[Tuple[str, int], set] = defaultdict(set)

    for r in rows:
        key = (r["config"], int(r["length"]))
        groups[key].append(float(r["ppl"]))
        if r["seed"] is not None:
            seeds[key].add(int(r["seed"]))

    out: List[dict] = []
    for (config, length), vals in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        mean_val = statistics.fmean(vals)
        std_val = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        out.append(
            {
                "config": config,
                "length": length,
                "n": int(len(vals)),
                "seed_count": int(len(seeds[(config, length)])),
                "mean_ppl": float(mean_val),
                "std_ppl": float(std_val),
                "min_ppl": float(min(vals)),
                "max_ppl": float(max(vals)),
            }
        )
    return out


def write_csv(summary_rows: List[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["config", "length", "n", "seed_count", "mean_ppl", "std_ppl", "min_ppl", "max_ppl"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)


def write_markdown(summary_rows: List[dict], out_md: Path) -> None:
    pivot: Dict[str, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    lengths = sorted({int(r["length"]) for r in summary_rows})
    configs = sorted({r["config"] for r in summary_rows})

    for r in summary_rows:
        pivot[r["config"]][int(r["length"])] = (float(r["mean_ppl"]), float(r["std_ppl"]))

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# PPL Summary (mean ± std)\n\n")
        f.write("| Config | " + " | ".join(str(L) for L in lengths) + " |\n")
        f.write("|---" + "|---" * len(lengths) + "|\n")
        for cfg in configs:
            cells = []
            for L in lengths:
                item = pivot[cfg].get(L)
                if item is None:
                    cells.append("N/A")
                else:
                    cells.append(f"{item[0]:.3f} ± {item[1]:.3f}")
            f.write(f"| {cfg} | " + " | ".join(cells) + " |\n")


def plot_lines(summary_rows: List[dict], fig_path: Path, title: str) -> None:
    plt = _import_matplotlib()
    by_cfg: Dict[str, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    for r in summary_rows:
        by_cfg[r["config"]][int(r["length"])] = (float(r["mean_ppl"]), float(r["std_ppl"]))

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    for cfg, entries in sorted(by_cfg.items()):
        lengths = sorted(entries.keys())
        means = [entries[L][0] for L in lengths]
        stds = [entries[L][1] for L in lengths]
        plt.plot(lengths, means, marker="o", label=cfg)
        lower = [max(m - s, 0.0) for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        plt.fill_between(lengths, lower, upper, alpha=0.15)

    plt.xlabel("Context Length")
    plt.ylabel("PPL (lower is better)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()


def plot_16k_bar(summary_rows: List[dict], fig_path: Path, title: str) -> None:
    plt = _import_matplotlib()
    target_length = 16384
    rows = [r for r in summary_rows if int(r["length"]) == target_length]
    if not rows:
        return
    rows.sort(key=lambda x: x["mean_ppl"])

    labels = [r["config"] for r in rows]
    means = [r["mean_ppl"] for r in rows]
    stds = [r["std_ppl"] for r in rows]

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    x = list(range(len(labels)))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("PPL@16384")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw result JSON files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for csv/md/figures outputs")
    parser.add_argument("--title", type=str, default="RoPE Frequency Experiment")
    args = parser.parse_args()

    rows = collect_rows(args.input_dir)
    if not rows:
        raise SystemExit(f"No valid result rows found in: {args.input_dir}")

    summary_rows = summarize(rows)
    out_csv = args.output_dir / "summary_by_length.csv"
    out_md = args.output_dir / "summary_table.md"
    fig_dir = args.output_dir / "figures"
    line_fig = fig_dir / "ppl_vs_length.png"
    bar_fig = fig_dir / "ppl_16k_bar.png"

    write_csv(summary_rows, out_csv)
    write_markdown(summary_rows, out_md)
    plot_lines(summary_rows, line_fig, args.title)
    plot_16k_bar(summary_rows, bar_fig, f"{args.title} (PPL@16K)")

    print(f"[done] rows: {len(rows)}")
    print(f"[done] csv:  {out_csv}")
    print(f"[done] md:   {out_md}")
    print(f"[done] fig:  {line_fig}")
    if bar_fig.exists():
        print(f"[done] fig:  {bar_fig}")


if __name__ == "__main__":
    main()
