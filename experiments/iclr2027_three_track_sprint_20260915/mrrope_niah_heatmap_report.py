#!/usr/bin/env python3
"""Report and plot the paired TailSpline/MrPro NIAH heatmap."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np

from experiments.iclr2027_three_track_sprint_20260915.prepare_mrrope_niah_heatmap import (
    DEPTHS,
    LENGTHS,
    REPEATS,
)


ARMS = ("tailspline", "mrpro")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def rouge1_recall(output: str, reference: str) -> float:
    predicted = Counter(re.findall(r"\w+", output.lower()))
    wanted = Counter(re.findall(r"\w+", reference.lower()))
    total = sum(wanted.values())
    return sum(min(predicted[token], count) for token, count in wanted.items()) / total if total else 0.0


def load_arm(path: Path, panel: dict[str, dict], arm: str) -> dict[str, dict]:
    status = json.loads((path / "status.json").read_text())
    rows = read_jsonl(path / "generations.jsonl")
    if status != {"status": "COMPLETE", "rows": 108, "lm_rows": 0} or len(rows) != 108:
        raise ValueError(f"incomplete NIAH arm {arm}: {status}/{len(rows)}")
    mapping = {str(row["row_id"]): row for row in rows}
    if set(mapping) != set(panel):
        raise ValueError(f"NIAH prompt coverage drift for {arm}")
    for row_id, row in mapping.items():
        source = panel[row_id]
        for key in ("task", "length_cap", "prompt_sha256", "references"):
            if row.get(key) != source.get(key):
                raise ValueError(f"NIAH identity drift: {arm}/{row_id}/{key}")
        reference = source["references"][0]
        row["rouge1_recall"] = rouge1_recall(row["output_text"], reference)
        if abs(row["rouge1_recall"] - float(row["ruler_official_score"])) > 1e-12:
            raise ValueError(f"single-number ROUGE-1 and substring recall disagree: {arm}/{row_id}")
    return mapping


def cell_matrix(mapping: dict[str, dict], panel: dict[str, dict]) -> dict:
    cells = {}
    for length in LENGTHS:
        for depth in DEPTHS:
            selected = [
                mapping[row_id]["rouge1_recall"]
                for row_id, source in panel.items()
                if source["length_cap"] == length and source["depth_percent"] == depth
            ]
            if len(selected) != REPEATS:
                raise ValueError(f"NIAH result cell drift: {length}/{depth}")
            cells[f"{length}/{depth}"] = float(np.mean(selected))
    by_length = {
        str(length): float(np.mean([
            cells[f"{length}/{depth}"] for depth in DEPTHS
        ])) for length in LENGTHS
    }
    by_depth = {
        str(depth): float(np.mean([
            cells[f"{length}/{depth}"] for length in LENGTHS
        ])) for depth in DEPTHS
    }
    eligible = [length for length in LENGTHS if by_length[str(length)] >= 0.90]
    return {
        "cells": cells,
        "by_length_macro": by_length,
        "by_depth_macro": by_depth,
        "all_cell_macro": float(np.mean(list(cells.values()))),
        "max_length_with_depth_macro_at_least_0p90": max(eligible) if eligible else None,
    }


def paired_bootstrap(runs, panel, *, draws=20_000, seed=20260928):
    rng = np.random.default_rng(seed)
    deltas = []
    for length in LENGTHS:
        for depth in DEPTHS:
            values = []
            for repeat in range(REPEATS):
                row_id = f"mrrope_niah_l{length}_d{depth:02d}_r{repeat}"
                values.append(
                    runs["tailspline"][row_id]["rouge1_recall"]
                    - runs["mrpro"][row_id]["rouge1_recall"]
                )
            deltas.append(values)
    delta = np.asarray(deltas)
    sampled = np.empty(draws)
    for draw in range(draws):
        indices = rng.integers(REPEATS, size=delta.shape)
        sampled[draw] = np.take_along_axis(delta, indices, axis=1).mean()
    return {
        "draws": draws,
        "seed": seed,
        "resampling": "paired repeat resampling within each frozen length-depth cell; cells equally weighted",
        "mean_delta": float(delta.mean()),
        "ci95": [float(value) for value in np.quantile(sampled, [0.025, 0.975])],
        "probability_delta_gt_zero": float(np.mean(sampled > 0.0)),
    }


def plot(report: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = {}
    for arm in ARMS:
        arrays[arm] = np.asarray([
            [report["arms"][arm]["cells"][f"{length}/{depth}"] for length in LENGTHS]
            for depth in DEPTHS
        ])
    arrays["delta"] = arrays["tailspline"] - arrays["mrpro"]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), constrained_layout=True)
    titles = {"tailspline": "TailSpline", "mrpro": "MrRoPE-Pro", "delta": "TailSpline - MrRoPE-Pro"}
    for axis, name in zip(axes, ("tailspline", "mrpro", "delta")):
        data = arrays[name]
        if name == "delta":
            limit = max(1 / 3, float(np.max(np.abs(data))))
            image = axis.imshow(data, vmin=-limit, vmax=limit, cmap="RdBu", aspect="auto")
        else:
            image = axis.imshow(data, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        axis.set_title(titles[name])
        axis.set_xticks(range(len(LENGTHS)), [f"{value // 1024}K" for value in LENGTHS])
        axis.set_yticks(range(len(DEPTHS)), [f"{value}%" for value in DEPTHS])
        axis.set_xlabel("Context cap")
        if axis is axes[0]:
            axis.set_ylabel("Needle depth")
        for y in range(len(DEPTHS)):
            for x in range(len(LENGTHS)):
                axis.text(x, y, f"{data[y, x]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="ARM=RUN_DIR")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot-prefix", type=Path, required=True)
    args = parser.parse_args()
    paths = {}
    for item in args.run:
        arm, value = item.split("=", 1)
        if arm in paths:
            raise ValueError(f"duplicate arm {arm}")
        paths[arm] = Path(value)
    if set(paths) != set(ARMS):
        raise ValueError("NIAH report requires exactly tailspline and mrpro")
    panel_rows = read_jsonl(args.panel)
    panel = {str(row["row_id"]): row for row in panel_rows}
    if len(panel) != 108:
        raise ValueError("NIAH panel must contain 108 unique rows")
    runs = {arm: load_arm(path, panel, arm) for arm, path in paths.items()}
    arms = {arm: cell_matrix(mapping, panel) for arm, mapping in runs.items()}
    report = {
        "status": "TAILSPLINE_MRPRO_LLAMA_S4_NIAH_HEATMAP_COMPLETE_V1",
        "model": "Meta-Llama-3-8B-Instruct",
        "candidate": "tailspline",
        "baseline": "mrpro",
        "lengths": list(LENGTHS),
        "depths_percent": list(DEPTHS),
        "repeats_per_cell": REPEATS,
        "rows_per_arm": len(panel),
        "metric": "ROUGE-1 recall; single numeric reference; exactly equals official substring recall on every row",
        "arms": arms,
        "delta_cells": {
            key: arms["tailspline"]["cells"][key] - arms["mrpro"]["cells"][key]
            for key in arms["tailspline"]["cells"]
        },
        "paired_inference": paired_bootstrap(runs, panel),
        "scope": (
            "MrRoPE-style diagnostic over 1x-4x; supports location of retrieval failures, "
            "but does not add an independent benchmark beyond the existing RULER retrieval family"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.plot_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot(report, args.plot_prefix)
    print(json.dumps({"status": report["status"], "delta": report["paired_inference"]}))


if __name__ == "__main__":
    main()
