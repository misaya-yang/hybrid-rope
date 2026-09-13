import pytest

from experiments.olmo_recovery_20260912.compare_llama_runner_parity import compare


def rows():
    source = [{
        "row_id": "r", "task": "niah_single_1", "length_cap": 65536,
        "references": ["123"], "prompt_sha256": "p",
    }]
    current = [{
        **source[0], "output_text": "123", "generated_ids": [1, 2],
    }]
    archived = [{
        **source[0], "output": "123", "output_ids": [1, 2],
    }]
    return source, current, archived


def test_exact_runner_parity():
    result = compare(*rows())
    assert result["status"] == "EXACT_RUNNER_PARITY"
    assert result["same_official_score_rows"] == 1


def test_score_only_parity_is_not_called_exact():
    source, current, archived = rows()
    archived[0]["output"] = "answer: 123"
    archived[0]["output_ids"] = [3]
    result = compare(source, current, archived)
    assert result["status"] == "SCORE_PARITY_ONLY"


def test_identity_drift_is_rejected():
    source, current, archived = rows()
    archived[0]["prompt_sha256"] = "other"
    with pytest.raises(ValueError, match="identity differs"):
        compare(source, current, archived)


def test_accepts_archived_runner_schema_on_both_sides():
    source, current, archived = rows()
    current[0]["output"] = current[0].pop("output_text")
    current[0]["output_ids"] = current[0].pop("generated_ids")
    assert compare(source, current, archived)["status"] == "EXACT_RUNNER_PARITY"
