#!/usr/bin/env python3
"""Compare one transferred Llama g8 table with reused fixed-table baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .score_fixed_table_interval import log_auc, read_rows, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prepared = args.root / "prepared_llama_g8_r0"
    supplement = args.root / "supplement_48k" / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text())
    supplement_manifest = json.loads((supplement / "manifest.json").read_text())
    if manifest["fixed_table_contract"] != supplement_manifest["fixed_table_contract"]:
        raise ValueError("base and 48K panels use different fixed-table contracts")
    fixed = manifest["fixed_table_contract"]
    if not (
        fixed["one_table_per_arm_for_entire_session"]
        and fixed["same_table_at_every_runtime_length"]
        and fixed["same_table_in_every_layer"]
        and not fixed["runtime_table_switching"]
    ):
        raise ValueError("panel is not the one-fixed-table problem")
    expected_rows = read_rows(prepared / "screen.jsonl") + read_rows(supplement / "screen.jsonl")
    expected = {row["row_id"]: row for row in expected_rows}
    if len(expected) != len(expected_rows) or len(expected) != 60:
        raise ValueError("expected the frozen 60-row Llama g8 panel")
    caps = sorted(set(manifest["runtime_lengths"] + supplement_manifest["runtime_lengths"]))
    tasks = list(manifest["tasks"])

    rows = {
        "candidate": read_rows(args.candidate_run / "generations.jsonl"),
        "BM_g8": read_rows(args.root / "BM_g8" / "generations.jsonl")
        + read_rows(args.root / "supplement_48k" / "BM_g8" / "generations.jsonl"),
        "MrPro_g8": read_rows(args.root / "MrPro_g8" / "generations.jsonl")
        + read_rows(args.root / "supplement_48k" / "MrPro_g8" / "generations.jsonl"),
    }
    summaries = {label: summarize(values, expected, caps, tasks) for label, values in rows.items()}
    native_expected = {key: value for key, value in expected.items() if value["length_cap"] == caps[0]}
    native = summarize(read_rows(args.root / "Native_8k" / "generations.jsonl"), native_expected, [caps[0]], tasks)
    candidate_curve = {
        cap: summaries["candidate"]["by_length"][str(cap)]["task_macro_official"] for cap in caps
    }
    envelope = {
        cap: max(
            summaries[arm]["by_length"][str(cap)]["task_macro_official"]
            for arm in ("BM_g8", "MrPro_g8")
        )
        for cap in caps
    }
    contrasts = {}
    for arm in ("BM_g8", "MrPro_g8"):
        curve = {cap: summaries[arm]["by_length"][str(cap)]["task_macro_official"] for cap in caps}
        contrasts[f"candidate_minus_{arm}"] = {
            "by_length": {str(cap): candidate_curve[cap] - curve[cap] for cap in caps},
            "five_point_log_auc": log_auc(candidate_curve) - log_auc(curve),
        }
    contrasts["candidate_regret"] = {
        "native_at_8k": native["by_length"][str(caps[0])]["task_macro_official"] - candidate_curve[caps[0]],
        "endpoint_vs_strong_static_floor": envelope[caps[-1]] - candidate_curve[caps[-1]],
        "worst_vs_pointwise_static_envelope": max(envelope[cap] - candidate_curve[cap] for cap in caps),
    }
    result = {
        "status": "COMPLETE",
        "panel": "Llama-3-8B one-fixed-g8-table 8/16/32/48/64K development panel",
        "fixed_table_contract": fixed,
        "summaries": summaries,
        "native_8k": native,
        "contrasts": contrasts,
        "scope": "cross-model zero-training development evidence; three tasks x four rows per length",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "candidate_auc": summaries["candidate"]["log_length_auc"],
        "candidate_min": summaries["candidate"]["interval_min"],
        "contrasts": contrasts,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
