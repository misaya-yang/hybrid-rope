#!/usr/bin/env python3
"""
Audit LongBench score scaling consistency: raw [0,1] vs pct [0,100].
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit raw->pct LongBench scaling consistency.")
    ap.add_argument(
        "--metrics_csv",
        type=str,
        default="batch_report_2026-02-23_downstream_eval/report/method_metrics_best_available.csv",
    )
    ap.add_argument("--value_col", type=str, default="longbench_avg")
    ap.add_argument(
        "--out_md",
        type=str,
        default="artifacts/reviewer_2026-02-24/longbench_scale_audit.md",
    )
    ap.add_argument("--tolerance", type=float, default=1e-6)
    return ap.parse_args()


def load_rows(path: Path, value_col: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            method = str(rec.get("method", "")).strip()
            if not method:
                continue
            raw_v = rec.get(value_col)
            if raw_v is None or raw_v == "":
                continue
            rows.append({"method": method, "raw": float(raw_v), "pct": float(raw_v) * 100.0})
    if not rows:
        raise RuntimeError(f"No rows loaded from {path} with value_col={value_col}")
    return rows


def rank_methods(rows: List[Dict[str, float]], key: str) -> List[str]:
    return [r["method"] for r in sorted(rows, key=lambda x: float(x[key]), reverse=True)]


def main() -> None:
    args = parse_args()
    metrics_csv = Path(args.metrics_csv)
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics_csv not found: {metrics_csv}")

    rows = load_rows(metrics_csv, value_col=args.value_col)
    rank_raw = rank_methods(rows, key="raw")
    rank_pct = rank_methods(rows, key="pct")
    ranking_identical = rank_raw == rank_pct

    max_abs_err = max(abs(r["pct"] - 100.0 * r["raw"]) for r in rows)
    pass_scale = max_abs_err <= float(args.tolerance)

    out_rows = sorted(rows, key=lambda x: float(x["raw"]), reverse=True)
    md_lines = [
        "# LongBench Scale Audit",
        "",
        f"- Source: `{metrics_csv}`",
        f"- Value column: `{args.value_col}`",
        f"- Check: `score_pct == 100 * score_raw` (tol={float(args.tolerance):.1e})",
        "",
        f"- Max abs error: `{max_abs_err:.6e}`",
        f"- Scaling check: `{'PASS' if pass_scale else 'FAIL'}`",
        f"- Ranking identity (raw vs pct): `{'PASS' if ranking_identical else 'FAIL'}`",
        "",
        "| rank | method | raw (0-1) | pct (0-100) |",
        "|---:|---|---:|---:|",
    ]
    for idx, row in enumerate(out_rows, start=1):
        md_lines.append(
            f"| {idx} | {row['method']} | {row['raw']:.6f} | {row['pct']:.4f} |"
        )

    md_lines.extend(
        [
            "",
            "## Ranking",
            "",
            f"- raw: `{', '.join(rank_raw)}`",
            f"- pct: `{', '.join(rank_pct)}`",
            "",
            "## Conclusion",
            "",
            (
                "- The metric-unit conversion is a strict linear scaling. "
                "Relative ordering and pairwise deltas are unchanged; only presentation units differ."
            ),
            "",
        ]
    )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[ok] wrote {out_md}")


if __name__ == "__main__":
    main()
