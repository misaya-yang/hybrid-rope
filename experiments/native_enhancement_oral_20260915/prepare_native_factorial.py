"""Prepare the two missing gain controls around an existing native positive result.

Reuses the frozen Qwen1.5B mix075 table and both original data blocks. This does
not create another frequency shape, select answers, or start a GPU process.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

from .evidence_review import raw


def gain_control(table: dict, gain: float) -> dict:
    if not 0 < gain < 10:
        raise ValueError("invalid gain")
    result = copy.deepcopy(table)
    result["gain"] = gain
    # A inherited hash names a full prior method in some legacy receipts;
    # retain its provenance separately rather than claiming the same method.
    result["construction"] = {"rule": "same frozen frequencies; gain-only factorial cell",
                               "source_gain": table["gain"], "new_fitting_in_this_preparation": False,
                               "source_design": "inherits the development-informed parent; not a claim of calibration-free discovery"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/root/autodl-tmp/fixed_rope_three_interfaces_20260913"))
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/qwen25_1p5b_32k"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("use a new preparation directory")
    root = args.source_root
    config = json.loads((args.model / "config.json").read_text())
    if (config.get("model_type"), config.get("hidden_size"), config.get("num_hidden_layers"), config.get("max_position_embeddings")) != ("qwen2", 1536, 28, 32768):
        raise ValueError("this historical positive anchor is Qwen2.5-1.5B native32K")
    source_report = json.loads((root / "reports/qwen15_s2_mix075_vs_native_core6_32k18.json").read_text())
    baseline = raw([Path(p) for p in source_report["source_files"]["native"]], 32768)
    original = raw([Path(p) for p in source_report["source_files"]["candidate"]], 32768)
    source_table = json.loads((root / "runs/qwen15_s2_candidate_core6_32k48k64k6/contract.json").read_text())["static_table"]
    other_table = json.loads((root / "runs/qwen15_s2_candidate_core6_block12_rows6_17/contract.json").read_text())["static_table"]
    if source_table != other_table:
        raise ValueError("candidate table differs between historical blocks")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from experiments.fixed_rope_three_interfaces_20260913.tables import model_geometry, runtime_native_inv_freq
    native = {"values_float32": runtime_native_inv_freq(model_geometry(config)).tolist(), "gain": 1.0}
    tables = {"mix075_gain1": gain_control(source_table, 1.0),
              "native_gain_mid": gain_control(native, float(source_table["gain"]))}
    rows = []
    for block, folder in (("development6", "qwen_low108"), ("additional12", "qwen_s2_core6_block12_rows6_17")):
        path = root / "panels" / folder / "screen.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["length_cap"] != 32768:
                continue
            key = (row["task"], row["length_cap"], row["prompt_sha256"])
            if key not in baseline or key not in original:
                raise ValueError("prepared prompt lacks both reusable original arms")
            for field in ("references", "input_tokens", "task", "length_cap"):
                if row[field] != baseline[key][field] or row[field] != original[key][field]:
                    raise ValueError("source panel and baseline identity differ")
            if len(row["prompt_ids"]) != row["input_tokens"] or row["input_tokens"] + row["max_new_tokens"] > 32768:
                raise ValueError("actual prompt exceeds native budget")
            rows.append({**row, "historical_row_id": row["row_id"], "historical_block": block,
                         "row_id": f"native_factorial:{block}:{len(rows):04d}"})
    if len(rows) != 108 or len({r["prompt_sha256"] for r in rows}) != 108 or len(baseline) != 108:
        raise ValueError("expected 108 unique paired native prompts")
    counts = Counter(row["task"] for row in rows)
    if len(counts) != 6 or set(counts.values()) != {18}:
        raise ValueError("original Core6 x 18 coverage differs")
    args.out.mkdir(parents=True)
    (args.out / "tables").mkdir()
    with (args.out / "inputs.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    # --skip-lm needs an existing JSON manifest, but no LM inputs are read.
    data = args.out / "data.json"
    data.write_text("{}\n")
    commands = {}
    for name, table in tables.items():
        table_path = args.out / "tables" / f"{name}.json"
        table_path.write_text(json.dumps(table, indent=2) + "\n")
        commands[name] = [sys.executable, "-m", "experiments.olmo_recovery_20260912.recovery_v2_eval",
                          "--data", str(data), "--model", str(args.model), "--arm", "Native",
                          "--extra-panel", str(args.out / "inputs.jsonl"), "--only-extra-panels", "--skip-lm",
                          "--length-cap", "32768", "--batch-size", "1", "--prefill-chunk-size", "8192",
                          "--static-table-json", str(table_path), "--table-label", name,
                          "--out", str(args.out / "runs" / name)]
    manifest = {"status": "CPU_PREPARED_NO_GPU_AUTHORIZATION", "model": str(args.model),
                "native_length": 32768, "rows_per_new_arm": 108, "new_generations_if_authorized": 216,
                "historical_blocks": dict(Counter(r["historical_block"] for r in rows)),
                "new_frequency_shape": False, "frozen_weights": True, "source_gain": source_table["gain"],
                "reusable_arms": source_report["source_files"], "commands_without_execute": commands,
                "estimands": {"frequency_at_gain1": "Y(mix,1)-Y(native,1)",
                               "gain_at_native": "Y(native,g)-Y(native,1)",
                               "interaction": "Y(mix,g)-Y(mix,1)-Y(native,g)+Y(native,1)"},
                "evidence_scope": "Retrospective attribution of a previously positive Core6 result. Report blocks separately. New independent confirmation remains necessary for generalization; changed slow endpoint means frequency-only, not fixed-support pure-z."}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("status", "rows_per_new_arm", "new_generations_if_authorized", "historical_blocks", "source_gain")}))


if __name__ == "__main__":
    main()
