"""CPU-only synthetic panels; no model outputs or GPU are used."""
import copy
import hashlib
import json
import math

import pytest

from scripts.analysis import summarize_reference_coupling_nll as summary


def panel(lengths=(4096, 8192)):
    hashed = lambda text: hashlib.sha256(text.encode()).hexdigest()
    profiles = [{"name": arm, "tensor_sha256": hashed(arm), "file_sha256": None if arm == "Native" else hashed(arm + "file"),
                 "attention_scaling": 1.0 if arm == "Native" else 1.05} for arm in summary.ARMS]
    manifest = {"status": "REFERENCE_COUPLING_NLL_FROZEN", "lengths": list(lengths),
        "natural_documents": 32, "target_tokens": 256, "arm_order": list(summary.ARMS),
        "profiles": {"profiles": profiles, "L_config": 8192, "L_ref": 4096,
                     "target_length": max(lengths), "scale": max(lengths) / 4096,
                     "coupling_manifest_sha256": "c" * 64, "baseline_manifest_sha256": "b" * 64}}
    for key in ("checkpoint_weight_sha256", "config_sha256", "data_manifest_sha256", "data_rows_sha256",
                "script_sha256", "model_source_sha256", "attention_source_sha256"):
        manifest[key] = hashed(key)
    rows = []
    for i in range(32):
        for length in lengths:
            for ai, arm in enumerate(summary.ARMS):
                rows.append({"arm": arm, "sample_id": f"reference-coupling-natural-{i:03d}", "length": length,
                    "family": "natural", "variant": "natural", "split": "holdout", "source_row": 20000 + i,
                    "source_text_sha256": hashed(f"source{i}"), "target_ids_sha256": hashed(f"target{i}"),
                    "prompt_ids_sha256": hashed(f"prompt{i}-{length}"), "target_tokens": 256, "target_start": length - 256,
                    "table_sha256_float32": profiles[ai]["tensor_sha256"], "attention_scaling": profiles[ai]["attention_scaling"],
                    "nll": 3 + i / 100 + ai * .01})
    return {"status": "REFERENCE_COUPLING_NLL_COMPLETE", "rows": len(rows)}, manifest, rows


def test_summary_reproduces_ppl_retention_and_paired_intervals():
    args = panel()
    result = summary.summarize(*args)
    assert result == summary.summarize(*args)
    physical = result["curves"]["4096"]["dimensionless_x"]
    assert physical["mean_nll"] == pytest.approx(3.165)
    assert physical["ppl"] == pytest.approx(math.exp(3.165))
    assert physical["ppl_retention_vs_native"] == pytest.approx(math.exp(-.01))
    contrast = result["paired_contrasts"]["4096"]["physical_minus_index"]
    assert contrast["paired_document_ci95"] == pytest.approx([-.01, -.01])
    assert result["bootstrap"]["seed"] == 202609024
    assert result["profile_selection_performed"] is False


def test_reference_gate_is_separate_from_long_gains():
    result, manifest, rows = panel()
    for row in rows:
        if row["arm"] == "dimensionless_x":
            row["nll"] += .5 if row["length"] == 4096 else -1
    report = summary.summarize(result, manifest, rows)
    assert report["native_point_gate"]["arms"]["dimensionless_x"]["status"] == "FAIL"
    assert report["curves"]["8192"]["dimensionless_x"]["ppl_retention_vs_native"] > 1


@pytest.mark.parametrize("mutation", ["partial", "missing", "duplicate", "extra_arm", "source", "target", "table", "gain", "nan", "metadata"])
def test_contract_failures_rejected(mutation):
    result, manifest, rows = panel()
    if mutation == "partial":
        result["status"] = "RUNNING"
    elif mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "extra_arm":
        rows[0]["arm"] = "static_ntk"
    elif mutation in ("source", "target"):
        rows[0][f"{mutation}_text_sha256" if mutation == "source" else "target_ids_sha256"] = "f" * 64
    elif mutation == "table":
        rows[0]["table_sha256_float32"] = "f" * 64
    elif mutation == "gain":
        rows[0]["attention_scaling"] = 2
    elif mutation == "nan":
        rows[0]["nll"] = float("nan")
    else:
        rows[0]["source_row"] += 1
    with pytest.raises(ValueError):
        summary.summarize(result, manifest, rows)


def test_compact_output_whitelists_identity_and_omits_private_fields():
    result, manifest, rows = panel()
    manifest["private_path"] = "/private/secret-checkpoint"
    manifest["profiles"]["profiles"][0]["private_path"] = "/private/secret-table"
    rows[0]["prompt"] = "sensitive raw prompt"
    encoded = json.dumps(summary.summarize(result, manifest, rows))
    assert "/private/" not in encoded
    assert "sensitive raw prompt" not in encoded


def test_three_length_s4_uses_same_paired_document_set():
    report = summary.summarize(*panel((4096, 8192, 16384)))
    assert report["lengths"] == [4096, 8192, 16384]
    assert report["scale"] == 4
    assert report["documents"] == 32


def test_ppl_overflow_is_null_not_nonstandard_json():
    result, manifest, rows = panel()
    for row in rows:
        row["nll"] += 1000
    report = summary.summarize(result, manifest, rows)
    assert report["curves"]["4096"]["Native"]["ppl"] is None
    json.dumps(report, allow_nan=False)


def test_loader_requires_exact_raw_and_manifest_hashes(tmp_path):
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


def test_terminal_aggregate_must_match_raw_nll():
    result, manifest, rows = panel()
    result["curves"] = {"Native": {"4096": {"documents": 32, "mean_tail_nll": 0}}}
    with pytest.raises(ValueError, match="aggregate differs"):
        summary.summarize(result, manifest, rows)
