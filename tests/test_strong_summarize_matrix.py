import json

import pytest

from experiments.iclr2027_strong_evidence_20260915.summarize_matrix import (
    build_summary,
    load_source,
    normalize_source,
    render_markdown,
)


def source(
    *,
    experiment_id="olmo_clean_s4",
    model_id="olmo2_1b",
    benchmark="ruler",
    contract="clean",
    scale=4,
    native=4096,
    multiples=(2, 4),
    scores=((0.8, 0.7), (0.6, 0.45)),
    rows=650,
):
    cells = []
    for multiple, (candidate, baseline) in zip(multiples, scores):
        cells.append({
            "length_tokens": native * multiple,
            "length_multiple": multiple,
            "expected_rows_per_arm": rows,
            "paired_rows": rows,
            "arms": {
                "tailspline": {"status": "complete", "rows": rows, "score": candidate},
                "mrpro": {"status": "complete", "rows": rows, "score": baseline},
            },
            "contrasts": [{
                "candidate": "tailspline",
                "baseline": "mrpro",
                "delta": candidate - baseline,
                "ci95": [candidate - baseline - 0.02, candidate - baseline + 0.02],
                "uncertainty_unit": "paired_prompts",
            }],
        })
    return {
        "schema": "STRONG_MATRIX_SOURCE_V1",
        "status": "complete",
        "experiment_id": experiment_id,
        "identity": {
            "model_id": model_id,
            "benchmark_family": benchmark,
            "data_contract": contract,
            "scale": scale,
            "native_length_tokens": native,
            "evaluation_contract": "full13-source-order-v1" if benchmark == "ruler" else "heldout-docs-v1",
            "metric": {
                "name": "task_macro_official" if benchmark == "ruler" else "mean_nll",
                "direction": "higher" if benchmark == "ruler" else "lower",
                "unit": "fraction" if benchmark == "ruler" else "nats",
            },
            "expected_length_multiples": list(multiples),
            "required_arms": ["tailspline", "mrpro"],
            "required_contrasts": [{"candidate": "tailspline", "baseline": "mrpro"}],
        },
        "cells": cells,
    }


def test_builds_model_by_normalized_length_matrix_and_descriptive_mean():
    olmo = normalize_source(source())
    llama = normalize_source(source(
        experiment_id="llama_clean_s4",
        model_id="llama3_8b",
        native=8192,
        scores=((0.85, 0.80), (0.68, 0.56)),
    ))
    report = build_summary([olmo, llama])

    assert len(report["matrices"]) == 1
    matrix = report["matrices"][0]
    assert matrix["columns"] == ["2L", "4L"]
    assert [row["model_id"] for row in matrix["rows"]] == ["llama3_8b", "olmo2_1b"]
    llama_row = matrix["rows"][0]
    assert llama_row["cells"]["4L"]["length_tokens"] == 32768
    assert matrix["cross_model_descriptive"]["4L"]["contrasts"]["tailspline_minus_mrpro"] == {
        "n_models": 2,
        "models": ["llama3_8b", "olmo2_1b"],
        "mean_delta": pytest.approx(0.135),
        "min_delta": pytest.approx(0.12),
        "max_delta": pytest.approx(0.15),
    }
    assert matrix["cross_model_uncertainty"] is None
    assert "not pooled" in matrix["cross_model_note"]


def test_partitions_clean_classic_scale_and_benchmark_contracts():
    payloads = [
        source(experiment_id="clean_s4"),
        source(experiment_id="classic_s4", contract="classic"),
        source(experiment_id="clean_s2", scale=2),
        source(experiment_id="clean_s16", scale=16),
        source(experiment_id="ppl_s4", benchmark="ppl"),
        source(experiment_id="natural_s4", benchmark="natural"),
    ]
    report = build_summary(normalize_source(payload) for payload in payloads)
    identities = {
        (matrix["identity"]["benchmark_family"], matrix["identity"]["data_contract"], matrix["identity"]["scale"])
        for matrix in report["matrices"]
    }
    assert identities == {
        ("ruler", "clean", "S4"),
        ("ruler", "classic", "S4"),
        ("ruler", "clean", "S2"),
        ("ruler", "clean", "S16"),
        ("ppl", "clean", "S4"),
        ("natural", "clean", "S4"),
    }


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda payload: payload.update(status="running"), "not a complete experiment"),
        (lambda payload: payload["cells"][0]["arms"]["tailspline"].update(status="running"), "is incomplete"),
        (lambda payload: payload["cells"][0]["arms"]["tailspline"].update(rows=649), "row count is incomplete"),
        (lambda payload: payload["cells"].pop(), "declared lengths"),
        (lambda payload: payload["cells"][0].update(paired_rows=649), "not complete on paired rows"),
    ],
)
def test_rejects_incomplete_experiments(mutate, match):
    payload = source()
    mutate(payload)
    with pytest.raises(ValueError, match=match):
        normalize_source(payload)


def test_rejects_arm_level_contract_mixing_and_duplicate_matrix_cells():
    mixed = source()
    mixed["cells"][0]["arms"]["mrpro"]["data_contract"] = "classic"
    with pytest.raises(ValueError, match="mixes evaluation contracts"):
        normalize_source(mixed)

    first = normalize_source(source(experiment_id="part_a", multiples=(2,), scores=((0.8, 0.7),)))
    duplicate = normalize_source(source(experiment_id="part_b", multiples=(2,), scores=((0.8, 0.7),)))
    with pytest.raises(ValueError, match="duplicate matrix cell"):
        build_summary([first, duplicate])


def test_rejects_cross_model_uncertainty_claim():
    payload = source()
    payload["cells"][0]["contrasts"][0]["uncertainty_unit"] = "pooled_models"
    with pytest.raises(ValueError, match="model-level uncertainty"):
        normalize_source(payload)


def test_json_loading_and_markdown_are_portable(tmp_path):
    path = tmp_path / "complete.json"
    path.write_text(json.dumps(source()), encoding="utf-8")
    report = build_summary([load_source(path)])
    markdown = render_markdown(report)

    assert "olmo2_1b" in markdown
    assert "Cross-model deltas" in markdown
    assert str(tmp_path) not in markdown
    encoded = json.dumps(report)
    assert str(tmp_path) not in encoded
    assert "misaya." not in (markdown + encoded).lower()


def test_rejects_personal_or_absolute_contract_labels():
    personal = source()
    personal["identity"]["evaluation_contract"] = "/opt/private/report.json"
    with pytest.raises(ValueError, match="personal path fragment"):
        normalize_source(personal)

    absolute = source()
    absolute["identity"]["evaluation_contract"] = "/tmp/report.json"
    with pytest.raises(ValueError, match="absolute path"):
        normalize_source(absolute)
