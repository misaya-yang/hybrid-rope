#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def pct_delta(v: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (v - base) / base * 100.0


def main() -> None:
    report_dir = Path(__file__).resolve().parent
    src = report_dir / "method_metrics_best_available.csv"
    out_csv = report_dir / "unified_master_table.csv"
    out_md = report_dir / "unified_master_table.md"

    rows: List[Dict[str, str]] = []
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise SystemExit(f"empty source: {src}")

    # Primary ranking key follows common reviewer reading order:
    # LongBench avg (descending), then passkey margin, then NIAH mean.
    for r in rows:
        r["_longbench"] = float(r["longbench_avg"])
        r["_passkey_margin"] = float(r["passkey_margin_16k"])
        r["_niah"] = float(r["niah_mean"])
        r["_passkey_tf"] = float(r["passkey_tf_16k"])
        r["_coverage"] = int(r["coverage"])

    rows.sort(
        key=lambda x: (
            x["_longbench"],
            x["_passkey_margin"],
            x["_niah"],
            x["_passkey_tf"],
            x["_coverage"],
        ),
        reverse=True,
    )

    base_row = next((r for r in rows if r["method"] == "baseline"), None)
    if base_row is None:
        raise SystemExit("baseline row missing in method_metrics_best_available.csv")

    base_lb = base_row["_longbench"]
    base_margin = base_row["_passkey_margin"]

    out_rows: List[Dict[str, object]] = []
    for i, r in enumerate(rows, start=1):
        out_rows.append(
            {
                "rank": i,
                "method": r["method"],
                "longbench_avg": round(r["_longbench"], 4),
                "longbench_delta_vs_baseline_pct": round(pct_delta(r["_longbench"], base_lb), 2),
                "niah_mean": round(r["_niah"], 4),
                "passkey_tf_16k": round(r["_passkey_tf"], 4),
                "passkey_margin_16k": round(r["_passkey_margin"], 4),
                "passkey_margin_delta_vs_baseline_pct": round(pct_delta(r["_passkey_margin"], base_margin), 2),
                "source_profile": r["source_profile"],
                "coverage": r["_coverage"],
            }
        )

    # Write CSV.
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    # Write Markdown.
    md_lines = [
        "# Unified Master Table",
        "",
        "Ranking rule: `longbench_avg` desc, then `passkey_margin_16k`, then `niah_mean`.",
        "",
        "| rank | method | longbench_avg | ΔLongBench vs baseline | niah_mean | passkey_tf@16k | passkey_margin@16k | Δmargin vs baseline | source_profile | coverage |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in out_rows:
        md_lines.append(
            "| {rank} | {method} | {longbench_avg:.4f} | {lb:+.2f}% | {niah_mean:.4f} | {passkey_tf_16k:.4f} | {passkey_margin_16k:.4f} | {pm:+.2f}% | {source_profile} | {coverage} |".format(
                rank=r["rank"],
                method=r["method"],
                longbench_avg=r["longbench_avg"],
                lb=r["longbench_delta_vs_baseline_pct"],
                niah_mean=r["niah_mean"],
                passkey_tf_16k=r["passkey_tf_16k"],
                passkey_margin_16k=r["passkey_margin_16k"],
                pm=r["passkey_margin_delta_vs_baseline_pct"],
                source_profile=r["source_profile"],
                coverage=r["coverage"],
            )
        )
    md_lines.append("")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
