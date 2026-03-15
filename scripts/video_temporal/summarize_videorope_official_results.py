#!/usr/bin/env python3
"""Summarize downloaded official VideoRoPE V-NIAH-D baselines."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MODE_MAP = {
    "vanilla_rope": "vanilla_rope",
    "time_rope": "tad_rope",
    "m_rope": "m_rope",
    "t_scale2_change_freq": "videorope",
}


def parse_average(path: Path) -> float:
    match = re.search(r"([0-9]*\.?[0-9]+)", path.read_text())
    if not match:
        raise ValueError(f"could not parse accuracy from {path}")
    return float(match.group(1))


def infer_mode(dirname: str) -> str:
    for key, value in MODE_MAP.items():
        if key in dirname:
            return value
    return dirname


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize official VideoRoPE benchmark outputs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/video_temporal/external/videorope_official/dataset/vision_niah_d/niah_output"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/supporting_video/videorope_official_baselines.json"),
    )
    args = parser.parse_args()

    summary = {}
    for avg_file in sorted(args.root.glob("*/avg_accuracy.txt")):
        mode = infer_mode(avg_file.parent.name)
        summary[mode] = {
            "mode": mode,
            "average_accuracy": parse_average(avg_file),
            "source_dir": str(avg_file.parent),
        }

    payload = {
        "experiment": "videorope_official_v_niah_d_baselines",
        "root": str(args.root),
        "baselines": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
