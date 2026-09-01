from __future__ import annotations

import copy
import hashlib
import json
import math
import sys

import numpy as np
import pytest

from scripts.analysis.summarize_native_reference_calibration import (
    ALPHA, COUNTS, GRID, NLL_MARGIN, SCHEMA, clopper_pearson,
    confirmation_lengths, contiguous_frontier, digest, paired_bootstrap,
    main, summarize, verify_frozen_inputs,
)


def manifest():
    return {"sample_ids": {split: {family: [f"{split}-{family}-{i}" for i in range(count)]
                                  for family, count in families.items()}
                            for split, families in COUNTS.items()}}


def rows_for(data, phase="calibration", lengths=GRID, deltas=None, cap_counts=None,
             compact_count=None):
    deltas = deltas or {length: 0.0 for length in lengths}
    ncap = COUNTS[phase]["capability"]
    cap_counts = cap_counts or {length: ncap for length in lengths}
    compact_count = ncap if compact_count is None else compact_count
    rows = []
    for i, sid in enumerate(data["sample_ids"][phase]["natural"]):
        for length in lengths:
            rows.append({"family": "natural", "variant": "natural", "split": phase,
                         "sample_id": sid, "length": length,
                         "nll": 2 + i / 1000 + deltas[length]})
    for i, sid in enumerate(data["sample_ids"][phase]["capability"]):
        for variant, length in [("distributed", length) for length in lengths] + [("compact", 0)]:
            count = compact_count if variant == "compact" else cap_counts[length]
            rows.append({"family": "capability", "variant": variant, "split": phase,
                         "sample_id": sid, "length": length, "depth_stratum": i % 4,
                         "exact_match": i < count, "terminated": True})
    return rows


def selection(data, length):
    return {"schema": SCHEMA, "phase": "calibration", "status": "PROVISIONAL",
            "selected_length": length, "confirmation_lengths": confirmation_lengths(length),
            "manifest_digest": digest(data)}


def test_frontier_requires_contiguous_prefix():
    assert contiguous_frontier([True, True, False, False]) == (2048, False)
    assert contiguous_frontier([True, False, True, False]) == (1024, True)
    assert contiguous_frontier([False] * 4) == (None, False)
    assert confirmation_lengths(1024) == [1024, 2048]
    assert confirmation_lengths(4096) == [1024, 4096, 8192]
    assert confirmation_lengths(8192) == [1024, 8192]


def test_point_estimate_calibration_freezes_identical_frontier():
    data = manifest()
    rows = rows_for(data, deltas={1024: 0, 2048: .01, 4096: .2, 8192: .3},
                    cap_counts={1024: 64, 2048: 60, 4096: 48, 8192: 32})
    result = summarize(rows, data, phase="calibration")
    assert result["status"] == "PROVISIONAL"
    assert result["frontiers"] == {"natural": 2048, "capability": 2048}
    assert result["selected_length"] == 2048
    assert result["confirmation_lengths"] == [1024, 2048, 4096]
    assert result["rule"]["nll_margin"] == pytest.approx(-math.log(.875))
    json.dumps(result, allow_nan=False)


def test_different_frontiers_abstain_instead_of_taking_minimum():
    data = manifest()
    rows = rows_for(data, deltas={1024: 0, 2048: .01, 4096: .2, 8192: .3})
    result = summarize(rows, data, phase="calibration")
    assert result["status"] == "ABSTAIN"
    assert "INCOMPATIBLE_FAMILY_FRONTIERS" in result["reasons"]
    assert result["selected_length"] is None


def test_fail_pass_reentry_abstains_even_if_prefixes_match():
    data = manifest()
    rows = rows_for(data, deltas={1024: 0, 2048: .2, 4096: 0, 8192: .3},
                    cap_counts={1024: 64, 2048: 40, 4096: 64, 8192: 40})
    result = summarize(rows, data, phase="calibration")
    assert result["frontiers"] == {"natural": 1024, "capability": 1024}
    assert result["status"] == "ABSTAIN"
    assert "NONMONOTONE_FAIL_TO_PASS_REENTRY" in result["reasons"]


@pytest.mark.parametrize("instrument", ["baseline", "compact"])
def test_weak_instrument_abstains(instrument):
    data = manifest()
    rows = rows_for(data, cap_counts={length: 40 if instrument == "baseline" else 64 for length in GRID},
                    compact_count=40 if instrument == "compact" else 64)
    result = summarize(rows, data, phase="calibration")
    assert result["status"] == "ABSTAIN"
    assert f"{instrument.upper()}_COMPETENCE_FAILED" in result["reasons"]


def test_exact_match_without_eos_is_failure():
    data = manifest()
    rows = rows_for(data)
    for row in rows:
        if row["family"] == "capability":
            row["terminated"] = False
    result = summarize(rows, data, phase="calibration")
    assert result["capability"]["instruments"]["baseline"]["success_rate"] == 0
    assert result["validation"]["exact_without_termination"] == 64 * 5
    assert result["status"] == "ABSTAIN"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unowned", "mixed_split", "depth", "nonfinite", "bool_length"])
def test_invalid_rows_abstain(mutation):
    data = manifest()
    rows = rows_for(data)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "unowned":
        rows[0]["sample_id"] = "not-in-manifest"
    elif mutation == "mixed_split":
        rows[0]["split"] = "confirmation"
    elif mutation == "depth":
        rows[-1]["depth_stratum"] = "changed"
    elif mutation == "nonfinite":
        rows[0]["nll"] = float("nan")
    else:
        rows[0]["length"] = True
    result = summarize(rows, data, phase="calibration")
    assert result["status"] == "ABSTAIN"
    assert result["validation"]["valid"] is False


def test_manifest_counts_and_split_independence_are_required():
    data = manifest()
    rows = rows_for(data)
    data["sample_ids"]["confirmation"]["natural"][0] = data["sample_ids"]["calibration"]["natural"][0]
    assert "overlap" in summarize(rows, data, phase="calibration")["reasons"][0]
    data = manifest()
    data["sample_ids"]["calibration"]["natural"].pop()
    assert summarize(rows, data, phase="calibration")["validation"]["valid"] is False


def test_prepared_data_v1_manifest_id_contract():
    data = manifest()
    for split, families in COUNTS.items():
        for family, count in families.items():
            data["sample_ids"][split][family] = [f"{family}-{split}-{i:03d}" for i in range(count)]
    rows = rows_for(data)
    prepared = {"status": "NATIVE_REFERENCE_CALIBRATION_DATA_READY_V1", "schema_version": 1,
                "grid": list(GRID), "files": {split: {
                    "natural_documents": counts["natural"], "capability_blueprints": counts["capability"],
                    "rows": counts["natural"] * 4 + counts["capability"] * 5}
                    for split, counts in COUNTS.items()}}
    assert summarize(rows, prepared, phase="calibration")["status"] == "PROVISIONAL"
    prepared["files"]["calibration"]["natural_documents"] = 31
    assert not summarize(rows, prepared, phase="calibration")["validation"]["valid"]


def test_cp_bounds_are_not_degenerate_at_perfect_success():
    lower, upper = clopper_pearson(128, 128, ALPHA)
    assert 0 < lower < 1
    assert upper == 1
    assert clopper_pearson(0, 128, ALPHA)[0] == 0
    assert clopper_pearson(0, 128, ALPHA)[1] > 0


def test_confirmation_accepts_one_point_and_confirms_next_failure():
    data = manifest()
    frozen = selection(data, 2048)
    rows = rows_for(data, "confirmation", frozen["confirmation_lengths"],
                    deltas={1024: 0, 2048: .01, 4096: .3},
                    cap_counts={1024: 128, 2048: 128, 4096: 40})
    result = summarize(rows, data, phase="confirmation", selection=frozen)
    assert result["status"] == "ACCEPTED_OPERATING_POINT"
    assert result["selected_length"] == 2048
    assert result["boundary_confirmed"] is True
    assert result["next_length_secondary"]["promotion_allowed"] is False
    assert result["capability"]["lengths"]["2048"]["lower_conservative_cp_retention"] < 1


def test_confirmation_next_pass_never_promotes():
    data = manifest()
    frozen = selection(data, 2048)
    rows = rows_for(data, "confirmation", frozen["confirmation_lengths"])
    result = summarize(rows, data, phase="confirmation", selection=frozen)
    assert result["status"] == "ACCEPTED_OPERATING_POINT"
    assert result["selected_length"] == 2048
    assert result["next_length_secondary"]["status"] == "PASS"
    assert not result["boundary_confirmed"]


def test_confirmation_failure_does_not_descend_grid():
    data = manifest()
    frozen = selection(data, 4096)
    rows = rows_for(data, "confirmation", frozen["confirmation_lengths"],
                    deltas={1024: 0, 4096: .3, 8192: .3})
    result = summarize(rows, data, phase="confirmation", selection=frozen)
    assert result["status"] == "ABSTAIN"
    assert result["provisional_length"] == 4096
    assert result["selected_length"] is None
    assert "CONFIRMATION_NATURAL_FAIL" in result["reasons"]


def test_point_pass_but_uncertain_cp_is_not_accepted():
    data = manifest()
    frozen = selection(data, 2048)
    rows = rows_for(data, "confirmation", frozen["confirmation_lengths"],
                    cap_counts={1024: 128, 2048: 115, 4096: 115})
    result = summarize(rows, data, phase="confirmation", selection=frozen)
    assert result["capability"]["lengths"]["2048"]["point_pass"]
    assert result["confirmation_gates"]["capability_retention"] == "UNRESOLVED"
    assert result["status"] == "ABSTAIN"


def test_identity_reference_still_requires_population_competence():
    data = manifest()
    frozen = selection(data, 1024)
    rows = rows_for(data, "confirmation", frozen["confirmation_lengths"],
                    cap_counts={1024: 98, 2048: 98})
    result = summarize(rows, data, phase="confirmation", selection=frozen)
    assert result["natural"]["1024"]["relative_identity"]
    assert result["capability"]["lengths"]["1024"]["relative_identity"]
    assert result["confirmation_gates"]["capability_retention"] == "PASS"
    assert result["confirmation_gates"]["baseline"] == "UNRESOLVED"
    assert result["status"] == "ABSTAIN"


@pytest.mark.parametrize("mutation", ["absent", "different_manifest", "lengths", "abstained", "extra_length"])
def test_confirmation_requires_frozen_selection_and_only_selected_lengths(mutation):
    data = manifest()
    frozen = selection(data, 2048)
    rows = rows_for(data, "confirmation", frozen["confirmation_lengths"])
    if mutation == "absent":
        frozen = None
    elif mutation == "different_manifest":
        frozen["manifest_digest"] = "0" * 64
    elif mutation == "lengths":
        frozen["confirmation_lengths"] = list(GRID)
    elif mutation == "abstained":
        frozen["status"] = "ABSTAIN"
    else:
        rows = rows_for(data, "confirmation")
    result = summarize(rows, data, phase="confirmation", selection=frozen)
    assert result["validation"]["valid"] is False


def test_bootstrap_is_paired_stratified_reproducible():
    values = np.array([[0, 1], [0, 1], [1, 2], [1, 2]], dtype=float)
    first = paired_bootstrap(values, ["a", "a", "b", "b"])
    second = paired_bootstrap(values, ["a", "a", "b", "b"])
    assert np.array_equal(first, second)
    assert np.all(first[:, 1] - first[:, 0] == 1)
    assert np.all(first[:, 0] == .5)


def prepared_files(tmp_path):
    data = manifest()
    for split, families in COUNTS.items():
        for family, count in families.items():
            data["sample_ids"][split][family] = [f"{family}-{split}-{i:03d}" for i in range(count)]
    prepared = {"status": "NATIVE_REFERENCE_CALIBRATION_DATA_READY_V1", "schema_version": 1,
                "grid": list(GRID), "files": {}}
    for split, counts in COUNTS.items():
        rows = rows_for(data, split)
        inputs = [{name: value for name, value in row.items()
                   if name not in {"nll", "exact_match", "terminated"}} for row in rows]
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in inputs))
        prepared["files"][split] = {"path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(inputs),
            "natural_documents": counts["natural"], "capability_blueprints": counts["capability"]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(prepared))
    return prepared, manifest_path, rows_for(data)


def test_frozen_input_hash_and_depth_metadata_are_verified(tmp_path):
    prepared, manifest_path, rows = prepared_files(tmp_path)
    receipt = verify_frozen_inputs(rows, prepared, manifest_path, "calibration")
    assert receipt["status"] == "HASH_AND_METADATA_VERIFIED"
    assert receipt["other_split_model_outcomes_read"] is False
    rows[-1]["depth_stratum"] = "changed"
    with pytest.raises(ValueError, match="metadata differs"):
        verify_frozen_inputs(rows, prepared, manifest_path, "calibration")
    path = tmp_path / "confirmation.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_frozen_inputs(rows, prepared, manifest_path, "calibration")


def test_cli_emits_runner_compatible_manifest_hash(tmp_path, monkeypatch):
    prepared, manifest_path, rows = prepared_files(tmp_path)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for row in rows:
        row["data_manifest_sha256"] = manifest_hash
    examples = tmp_path / "examples.jsonl"
    examples.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "decision.json"
    monkeypatch.setattr(sys, "argv", ["summarize", "--examples", str(examples),
        "--data-manifest", str(manifest_path), "--output", str(output), "--phase", "calibration"])
    assert main() == 0
    result = json.loads(output.read_text())
    assert result["status"] == "PROVISIONAL"
    assert result["data_manifest_sha256"] == manifest_hash
    assert result["manifest_digest"] == digest(prepared)
    assert result["confirmation_lengths"] == [1024, 8192]
    assert result["validation"]["frozen_inputs"]["status"] == "HASH_AND_METADATA_VERIFIED"
