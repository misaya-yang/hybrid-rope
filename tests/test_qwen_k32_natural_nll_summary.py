"""Synthetic CPU-only tests for the fixed packed-natural NLL summary."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from scripts.analysis import summarize_qwen_k32_natural_nll as summary


def hashed(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def panel(index32=3.05, native64=3.5, index64=3.3, yarn64=3.4):
    profiles = [{"name": arm, "tensor_sha256": values[0], "file_sha256": values[1],
                 "attention_scaling": values[2]} for arm, values in summary.PROFILE.items()]
    manifest = {
        "status": summary.FROZEN_STATUS, "checkpoint_weight_sha256": summary.WEIGHT_SHA256,
        "config_sha256": summary.CONFIG_SHA256, "data_manifest_sha256": hashed("manifest"),
        "data_rows_sha256": hashed("rows"), "source": {"name": "fresh.parquet",
            "sha256": hashed("source"), "start_row": 20000, "consumed_row_range": [20000, 29999]},
        "tokenizer_files": [{"name": "tokenizer.json", "sha256": hashed("tokenizer"), "bytes": 123}],
        "packing_contract": summary.PACKING_CONTRACT, "lengths": list(summary.LENGTHS),
        "natural_streams": summary.STREAMS, "target_tokens": summary.TARGET_TOKENS,
        "arm_order": list(summary.ARMS), "profiles": profiles,
        "script_sha256": summary.sha256(Path("scripts/eval/eval_qwen_k32_natural_nll.py")),
        "model_source_sha256": hashed("model"), "attention_source_sha256": hashed("attention"),
        "use_cache": False, "compile": False, "model_updates": 0, "profile_selection": False,
        "all_profiles_loaded_before_inference": True,
    }
    means = {
        ("Native", 32768): 3.0, ("normalized_raw_index", 32768): index32,
        ("official_equation_yarn", 32768): 3.1, ("Native", 65536): native64,
        ("normalized_raw_index", 65536): index64, ("official_equation_yarn", 65536): yarn64,
    }
    rows = []
    for index in range(summary.STREAMS):
        sid = f"qwen-k32-natural-{index:03d}"
        for length in summary.LENGTHS:
            for arm in summary.ARMS:
                rows.append({"arm": arm, "sample_id": sid, "length": length,
                    "family": "natural", "variant": "packed_natural", "split": "holdout",
                    "source_row_start": 20000 + index * 10, "source_row_end": 20009 + index * 10,
                    "source_document_count": 10, "source_set_sha256": hashed(f"source-set-{index}"),
                    "prompt_ids_sha256": hashed(f"prompt-{index}-{length}"),
                    "target_ids_sha256": hashed(f"target-{index}"),
                    "target_start": length - summary.TARGET_TOKENS,
                    "target_tokens": summary.TARGET_TOKENS,
                    "table_sha256_float32": summary.PROFILE[arm][0],
                    "attention_scaling": summary.PROFILE[arm][2],
                    "nll": means[arm, length] + index / 10000})
    result = {"status": summary.TERMINAL_STATUS, "rows": len(rows)}
    refresh(result, rows)
    return result, manifest, rows


def refresh(result, rows):
    native = {(row["sample_id"], row["length"]): row["nll"]
              for row in rows if row["arm"] == "Native"}
    curves = {}
    for arm in summary.ARMS:
        curves[arm] = {}
        for length in summary.LENGTHS:
            cell = [row for row in rows if row["arm"] == arm and row["length"] == length]
            mean = sum(row["nll"] for row in cell) / len(cell)
            curves[arm][str(length)] = {"streams": len(cell), "mean_tail_nll": mean,
                "mean_paired_delta_vs_native": sum(
                    row["nll"] - native[row["sample_id"], length] for row in cell) / len(cell)}
    result["curves"] = curves


def test_summary_reports_curves_retention_intervals_and_resolver_pass():
    args = panel()
    report = summary.summarize(*args)
    assert report == summary.summarize(*args)
    assert report["classification"] == {"resolver": "PASS", "index_vs_yarn": "INDEX_FAVORED"}
    assert report["curves"]["32768"]["Native"]["mean_nll"] == pytest.approx(3.00155)
    assert report["curves"]["32768"]["Native"]["ppl"] == pytest.approx(math.exp(3.00155))
    assert report["index_native_ppl_retention_32k"]["value"] == pytest.approx(math.exp(-.05))
    assert report["index_minus_native_64k"]["mean_delta_nll"] == pytest.approx(-.2)
    assert report["index_minus_native_64k"]["joint_paired_stream_ci95"] == pytest.approx([-.2, -.2])
    assert report["index_minus_yarn_64k"]["joint_paired_stream_ci95"] == pytest.approx([-.1, -.1])
    assert report["bootstrap"]["replicates"] == 10000
    assert report["bootstrap"]["unit"].startswith("same 32 stream")


@pytest.mark.parametrize("kwargs,resolver,ranking", [
    ({"index32": 3.2}, "NOT_PASS", "INDEX_FAVORED"),
    ({"index64": 3.6}, "NOT_PASS", "YARN_FAVORED"),
    ({"index64": 3.4}, "PASS", "UNRESOLVED"),
])
def test_fact_classifications_follow_only_registered_rules(kwargs, resolver, ranking):
    report = summary.summarize(*panel(**kwargs))
    assert report["classification"] == {"resolver": resolver, "index_vs_yarn": ranking}


@pytest.mark.parametrize("mutation", [
    "partial", "missing", "duplicate", "arm", "length", "family", "target_start", "target",
    "prompt", "source_set", "source_range", "table", "gain", "nan", "negative", "rows",
    "curve", "extra_curve", "profile", "weight", "config", "data_hash", "source", "tokenizer", "code",
    "packing", "streams",
])
def test_fail_closed_on_raw_result_manifest_profile_or_hash_drift(mutation):
    result, manifest, rows = panel()
    if mutation == "partial": result["status"] = "RUNNING"
    elif mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows.append(copy.deepcopy(rows[0]))
    elif mutation == "arm": rows[0]["arm"] = "other"
    elif mutation == "length": rows[0]["length"] = 8192
    elif mutation == "family": rows[0]["variant"] = "document"
    elif mutation == "target_start": rows[0]["target_start"] -= 1
    elif mutation == "target": rows[0]["target_ids_sha256"] = hashed("different")
    elif mutation == "prompt": rows[0]["prompt_ids_sha256"] = hashed("different")
    elif mutation == "source_set": rows[0]["source_set_sha256"] = hashed("different")
    elif mutation == "source_range": rows[0]["source_row_start"] += 1
    elif mutation == "table": rows[0]["table_sha256_float32"] = "0" * 64
    elif mutation == "gain": rows[0]["attention_scaling"] = 2.0
    elif mutation == "nan": rows[0]["nll"] = float("nan")
    elif mutation == "negative": rows[0]["nll"] = -1
    elif mutation == "rows": result["rows"] -= 1
    elif mutation == "curve": result["curves"]["Native"]["32768"]["mean_tail_nll"] = 0
    elif mutation == "extra_curve": result["curves"]["Native"]["8192"] = copy.deepcopy(
        result["curves"]["Native"]["32768"])
    elif mutation == "profile": manifest["profiles"][1]["tensor_sha256"] = "0" * 64
    elif mutation == "weight": manifest["checkpoint_weight_sha256"] = "0" * 64
    elif mutation == "config": manifest["config_sha256"] = "0" * 64
    elif mutation == "data_hash": manifest["data_manifest_sha256"] = "not-a-hash"
    elif mutation == "source": manifest["source"]["sha256"] = "not-a-hash"
    elif mutation == "tokenizer": manifest["tokenizer_files"][0]["sha256"] = "not-a-hash"
    elif mutation == "code": manifest["script_sha256"] = "0" * 64
    elif mutation == "packing": manifest["packing_contract"] = "changed"
    elif mutation == "streams": manifest["natural_streams"] = 31
    with pytest.raises((ValueError, KeyError)):
        summary.summarize(result, manifest, rows)


def test_compact_output_whitelists_hashes_without_raw_prompts_or_paths():
    result, manifest, rows = panel()
    rows[0]["raw_prompt"] = "sensitive raw prompt"
    manifest["private_path"] = "/private/checkpoint"
    manifest["profiles"][0]["private_path"] = "/private/table"
    encoded = json.dumps(summary.summarize(result, manifest, rows))
    assert "sensitive raw prompt" not in encoded
    assert "/private/" not in encoded
    assert summary.WEIGHT_SHA256 in encoded
    assert hashed("source") in encoded


def test_ppl_overflow_is_null_and_json_remains_standard():
    result, manifest, rows = panel()
    for row in rows:
        row["nll"] += 1000
    refresh(result, rows)
    report = summary.summarize(result, manifest, rows)
    assert report["curves"]["32768"]["Native"]["ppl"] is None
    json.dumps(report, allow_nan=False)


def test_loader_requires_exact_raw_result_and_manifest_hashes(tmp_path):
    result, manifest, rows = panel()
    examples = tmp_path / "examples.jsonl"
    examples.write_text("".join(json.dumps(row) + "\n" for row in rows))
    run = tmp_path / "run_manifest.json"
    run.write_text(json.dumps(manifest))
    result.update(examples_sha256=summary.sha256(examples), run_manifest_sha256=summary.sha256(run))
    (tmp_path / "results.json").write_text(json.dumps(result))
    report = summary.load_and_summarize(tmp_path)
    assert report["raw_hashes"]["examples_sha256"] == result["examples_sha256"]
    run.write_text(run.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        summary.load_and_summarize(tmp_path)


def test_cli_invalid_panel_never_promotes_partial_metrics(tmp_path):
    output = tmp_path / "report.json"
    assert summary.main(["--root", str(tmp_path / "missing"), "--output", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["status"] == "INVALID_OR_INCOMPLETE_NLL_PANEL"
    assert "curves" not in report
