"""Read and reaggregate existing experiments before proposing another method.

No model execution, new table construction, or score-driven candidate selection.
Reported confidence intervals remain attached to their original report; this
script independently checks paired rows and exact observed task means.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime
import json
import math
from pathlib import Path
import statistics


def read(path: Path):
    return json.loads(path.read_text())


def raw(paths: list[Path], length: int | None = None) -> dict[str, dict]:
    rows = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if length is not None and row["length_cap"] != length:
                continue
            # Historical source blocks restart row_id at zero. Match actual
            # prompt identity across blocks, not the reused display label.
            key = (row["task"], row["length_cap"], row["prompt_sha256"])
            if key in rows:
                raise ValueError(f"duplicate row: {path}/{key}")
            rows[key] = row
    if not rows:
        raise ValueError("no existing rows")
    return rows


def paired(candidate: dict[str, dict], baseline: dict[str, dict], *, score_field: str) -> dict:
    if set(candidate) != set(baseline):
        raise ValueError("input sets differ; no silent intersection or winner filtering")
    cells = defaultdict(list)
    identities = ("task", "length_cap", "prompt_sha256", "input_tokens", "references")
    for key in sorted(candidate):
        c, b = candidate[key], baseline[key]
        for field in identities:
            if field not in c or field not in b or c[field] != b[field]:
                raise ValueError(f"paired input identity mismatch: {key}/{field}")
        if c.get("max_new_tokens") != b.get("max_new_tokens"):
            raise ValueError("generation budget mismatch")
        scores = [float(row[score_field]) for row in (c, b)]
        if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise ValueError("invalid official task score")
        cells[c["length_cap"], c["task"]].append((c, b, *scores))
    result = {}
    for length in sorted({key[0] for key in cells}):
        tasks = {}
        for (cap, task), items in sorted(cells.items()):
            if cap != length:
                continue
            deltas = [c - b for _, _, c, b in items]
            tasks[task] = {
                "rows": len(items), "candidate": statistics.mean(x[2] for x in items),
                "baseline": statistics.mean(x[3] for x in items), "delta": statistics.mean(deltas),
                "wins": sum(d > 0 for d in deltas), "losses": sum(d < 0 for d in deltas),
                "ties": sum(d == 0 for d in deltas),
                "candidate_immediate_eos": sum(len(x[0].get("generated_ids", [])) == 1 and x[0].get("ended_eos", False) for x in items),
                "baseline_immediate_eos": sum(len(x[1].get("generated_ids", [])) == 1 and x[1].get("ended_eos", False) for x in items),
                "both_finished_eos_rows": sum(x[0].get("ended_eos", False) and x[1].get("ended_eos", False) for x in items),
            }
        result[str(length)] = {"tasks": tasks, "rows": sum(x["rows"] for x in tasks.values()),
                               **{k: statistics.mean(x[k] for x in tasks.values()) for k in ("candidate", "baseline", "delta")}}
    return {"rows": len(candidate), "score_field": score_field, "paired_fields": list(identities),
            "by_length": result, "new_confidence_intervals": False,
            "scope": "Exact task-equal reaggregation of recorded scores; not rescoring raw text or a new model experiment."}


def review_report(path: Path) -> dict:
    report = read(path)
    source_files = {name: [Path(p) for p in paths] for name, paths in report["source_files"].items()}
    length = report.get("length")
    rows = {name: raw(paths, length) for name, paths in source_files.items()}
    candidate = report["candidate"]
    comparisons = {}
    for baseline in report["baselines"]:
        comparison = paired(rows[candidate], rows[baseline], score_field="ruler_official_score")
        # Verify actual arm means rather than the bootstrap distribution mean.
        for arm, label in ((candidate, "candidate"), (baseline, "baseline")):
            declared = report["summaries"][arm]["by_length"]
            for cap, value in comparison["by_length"].items():
                if abs(value[label] - declared[cap]["task_macro_official"]) > 1e-12:
                    raise ValueError(f"report/raw score disagreement: {path}/{arm}/{cap}")
        comparison["original_report_contrast"] = report["contrasts"][baseline]
        comparisons[baseline] = comparison
    contracts = {}
    for arm, paths in source_files.items():
        contracts[arm] = []
        for source in paths:
            contract_path = source.parent / "contract.json"
            c = read(contract_path) if contract_path.exists() else {}
            contracts[arm].append({"path": str(contract_path),
                                   **{k: c.get(k) for k in ("arm", "unadapted", "static_table", "runtime_versions", "batch_size", "prefill_chunk_size")}})
    return {"path": str(path), "status": report["status"], "candidate": candidate,
            "sources": {k: [str(p) for p in v] for k, v in source_files.items()},
            "contracts": contracts, "comparisons": comparisons}


def run_review(root: Path) -> dict:
    reports = {
        "x4_clean": "today_rope_plan_20260914/strong_evidence/llama_s4_clean_matched_dose_c/reports/tailspline_vs_control_and_mrpro.json",
        "qwen15_native_cumulative": "fixed_rope_three_interfaces_20260913/reports/qwen15_s2_mix075_vs_native_core6_32k18.json",
        "qwen15_native_new_block": "fixed_rope_three_interfaces_20260913/reports/qwen15_s2_mix075_vs_native_core6_32k_block12_rows6_17.json",
        "qwen15_all_methods": "fixed_rope_three_interfaces_20260913/reports/qwen15_s2_mix075_vs_mrpro_yarn_bm_c42_core6_32k48k64k6.json",
        "llama_all_methods": "fixed_rope_three_interfaces_20260913/reports/llama_s4_mix075_loggain_mid_vs_bm_mrpro_c42_core6_balanced12.json",
    }
    reviewed = {name: review_report(root / path) for name, path in reports.items()}
    llama_range = reviewed["llama_all_methods"]
    llama_native_path = root / "fixed_rope_three_interfaces_20260913/runs/llama_native_core6_8k_balanced12/generations.jsonl"
    llama_native = paired(raw([Path(p) for p in llama_range["sources"][llama_range["candidate"]]], 8192),
                          raw([llama_native_path], 8192), score_field="ruler_official_score")
    llama_native_report = read(root / "fixed_rope_three_interfaces_20260913/reports/llama_s4_mix075_loggain_mid_vs_native_8k_balanced12.json")
    if abs(llama_native["by_length"]["8192"]["delta"] - llama_native_report["delta_task_macro_official"]) > 1e-12:
        raise ValueError("legacy Llama native report disagrees with raw")
    reviewed["llama_native_gain_mid"] = {"comparison": llama_native, "original_report": llama_native_report,
                                          "baseline_source": str(llama_native_path)}
    legacy_pairs = {}
    for name, candidate_path, baseline_path in (
        ("qwen3_gap_capped", "bm_transfer_20260908/gap_capped_run_01/GapCapped.jsonl", "bm_transfer_20260908/run_qwen3_01/MrPro.jsonl"),
        ("qwen3_native_windows", "bm_transfer_20260908/native_windows_full_01/NativeWindowMrPro.jsonl", "bm_transfer_20260908/run_qwen3_01/MrPro.jsonl"),
        ("olmo_native_windows", "olmo_fast_screen_20260908/native_windows_full_01/NativeWindowMrPro.jsonl", "olmo_fast_screen_20260908/run_holdout_01/MrPro.jsonl"),
    ):
        legacy_pairs[name] = {"candidate_path": str(root / candidate_path), "baseline_path": str(root / baseline_path),
                              **paired(raw([root / candidate_path]), raw([root / baseline_path]), score_field="correct")}
    mechanisms = {}
    for name, relative in (
        ("qwen_cross_cache", "bm_transfer_20260908/cross_cache_run_02"),
        ("qwen_cross_cache_first", "bm_transfer_20260908/cross_cache_run_01"),
        ("olmo_stage", "olmo_fast_screen_20260908/stage_run_02"),
        ("olmo_kv_factor", "olmo_fast_screen_20260908/factor_run_01"),
    ):
        directory = root / relative
        records = read(directory / "results.json")
        mechanisms[name] = {"path": str(directory), "status": read(directory / "status.json"),
                            "qualification": read(directory / "qualification.json"),
                            "records": [{k: v for k, v in row.items() if k not in ("token_trace", "generated_ids")} for row in records],
                            "scope": "Previously selected retrospective examples; retain same-table replay qualifications and per-case interpretation."}
    nongeo_path = root / "nongeometric_screen_20260909/development_summary.json"
    nongeo = [{k: row[k] for k in ("method", "rows", "complete_original_panel", "macro_delta", "nll", "wins", "losses") if k in row}
             for row in read(nongeo_path)]
    nongeo_raw = {}
    mr_source = root / "bm_transfer_20260908/run_qwen3_01/MrPro.jsonl"
    mr_rows = raw([mr_source])
    native_source = root / "nongeometric_screen_20260909/native_reference/ruler.jsonl"
    native_rows = raw([native_source])
    for directory in sorted((root / "nongeometric_screen_20260909/results").iterdir()):
        source = directory / "ruler.jsonl"
        if not source.is_file() or not source.stat().st_size:
            continue
        rows = raw([source])
        if not set(rows).issubset(mr_rows):
            raise ValueError(f"unrecognized development IDs: {source}")
        # Explicit partial-panel comparison, with coverage retained in every row.
        baseline_subset = {key: mr_rows[key] for key in rows}
        item = {"source": str(source), "reference_source": str(mr_source),
                "original_panel_rows": len(mr_rows), "complete_original_core6_panel": set(rows) == set(mr_rows),
                "comparison_to_mrpro": paired(rows, baseline_subset, score_field="correct"),
                "contract": read(directory / "contract.json")}
        short = {key: row for key, row in rows.items() if row["length_cap"] == 32768}
        if set(short) == set(native_rows):
            item["comparison_to_native_12"] = paired(short, native_rows, score_field="correct")
        nongeo_raw[directory.name] = item
    binding = {}
    for name in ("MrPro", "E1_s28_less"):
        p = root / f"nongeometric_screen_20260909/counterfactual_bindings/{name}.json"
        binding[name] = {k: v for k, v in read(p).items() if k not in ("token_trace", "generated_ids")}
    return {"status": "EXISTING_EVIDENCE_REVIEW_CPU_ONLY", "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "root": str(root), "model_execution": False, "new_methods_constructed": False,
            "verified_reports": reviewed, "legacy_raw_pairs": legacy_pairs, "mechanism_records": mechanisms,
            "nongeometric_development": {"source": str(nongeo_path), "evidence_level": "report-only; not each method's raw rows", "methods": nongeo},
            "nongeometric_current_raw": nongeo_raw,
            "native_reference": {"source": str(native_source), "contract": read(native_source.parent / "contract.json")},
            "existing_binding_counterfactual": binding,
            "scope": "Nominated method lineage and real matched comparisons, not an exhaustive audit of every historical run. Files and cumulative/holdout reports are not independent experiment counts."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_review(args.root)
    if args.out.exists():
        raise ValueError("use a new audit receipt path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "reports": len(result["verified_reports"]), "legacy_pairs": len(result["legacy_raw_pairs"])}))


if __name__ == "__main__":
    main()
