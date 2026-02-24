#!/usr/bin/env python3
"""Summarize plan-v2 run registry into paper-friendly tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize artifacts/registry.jsonl into tables.")
    ap.add_argument("--registry", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    return ap.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def maybe_load(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(registry_path)
    flat_rows: List[Dict] = []
    for r in rows:
        row = dict(r)
        summary_path = Path(str(r.get("summary_json", "")))
        summary = maybe_load(summary_path) if summary_path else {}
        metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
        row["ppl"] = metrics.get("ppl")
        row["longbench_avg"] = metrics.get("longbench_avg")
        row["needle_tf_accuracy"] = metrics.get("needle_tf_accuracy")
        row["needle_margin"] = metrics.get("needle_margin")
        flat_rows.append(row)

    if not flat_rows:
        (out_dir / "README.txt").write_text("No registry rows found.\n", encoding="utf-8")
        return

    df = pd.DataFrame(flat_rows)
    df.to_csv(out_dir / "registry_flat.csv", index=False)

    # Table 1: E1 main comparison.
    t1 = (
        df[df["exp"].astype(str).str.upper() == "E1"]
        .sort_values(["ctx", "method"])
        [["run_id", "ctx", "method", "longbench_avg", "ppl", "needle_tf_accuracy", "needle_margin", "status"]]
    )
    t1.to_csv(out_dir / "table1_main.csv", index=False)

    # Table 2: E2 disentanglement.
    t2 = (
        df[df["exp"].astype(str).str.upper() == "E2"]
        .sort_values(["ctx", "method"])
        [["run_id", "ctx", "method", "rope_base", "rope_shape", "longbench_avg", "ppl", "needle_tf_accuracy", "status"]]
    )
    t2.to_csv(out_dir / "table2_e2.csv", index=False)

    # Overview for quick status.
    overview = (
        df.groupby(["exp", "status"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["exp", "status"])
    )
    overview.to_csv(out_dir / "overview_counts.csv", index=False)

    print(f"saved: {out_dir / 'registry_flat.csv'}")
    print(f"saved: {out_dir / 'table1_main.csv'}")
    print(f"saved: {out_dir / 'table2_e2.csv'}")
    print(f"saved: {out_dir / 'overview_counts.csv'}")


if __name__ == "__main__":
    main()

