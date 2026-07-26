from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    evaluate_instruct_ruler_screen as evaluate,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    prepare_instruct_ruler_long_gap as prepare,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _screen_fixture(
    tmp_path: Path,
    *,
    status: str,
    manifest_lengths: list[int],
) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "tokenizer.json").write_text(
        '{"version":"test"}\n',
        encoding="utf-8",
    )
    root = tmp_path / "screen"
    data_path = root / "L8192" / "niah_single_1" / "test.jsonl"
    rows = [
        {"input": f"row-{index}", "outputs": [str(index)]}
        for index in range(2)
    ]
    data_path.parent.mkdir(parents=True)
    data_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    _write_json(
        root / "manifest.json",
        {
            "status": status,
            "task": "niah_single_1",
            "checkpoint": str(checkpoint.resolve()),
            "tokenizer_sha256": _sha256(
                checkpoint / "tokenizer.json"
            ),
            "ruler_commit": "test-commit",
            "lengths": manifest_lengths,
            "samples_per_length": 2,
            "files": {
                "8192": {
                    "relative_path": str(data_path.relative_to(root)),
                    "sha256": _sha256(data_path),
                    "rows": 2,
                }
            },
        },
    )
    return checkpoint, root


@pytest.mark.parametrize(
    "status",
    evaluate.SUPPORTED_DATA_STATUSES,
)
def test_validate_data_accepts_registered_statuses(
    tmp_path: Path,
    status: str,
) -> None:
    checkpoint, root = _screen_fixture(
        tmp_path,
        status=status,
        manifest_lengths=[8192],
    )

    receipt, rows = evaluate.validate_data(
        root,
        checkpoint,
        "niah_single_1",
        (8192,),
        2,
    )

    assert receipt["preparation_status"] == status
    assert len(rows) == 2
    assert rows[0]["_nominal_length"] == 8192
    assert rows[1]["_local_index"] == 1


def test_validate_data_rejects_unregistered_length(
    tmp_path: Path,
) -> None:
    checkpoint, root = _screen_fixture(
        tmp_path,
        status=prepare.STATUS,
        manifest_lengths=[8192],
    )

    with pytest.raises(
        RuntimeError,
        match="absent from data manifest",
    ):
        evaluate.validate_data(
            root,
            checkpoint,
            "niah_single_1",
            (4096,),
            2,
        )


def test_parse_row_identities() -> None:
    row = {
        "input": (
            "One of the special magic numbers for amber-lake is: "
            "1234567.\n"
            "What is the special magic number for amber-lake mentioned "
            "in the provided text?"
        ),
        "outputs": ["1234567"],
    }

    identities = prepare.parse_row_identities(row)

    assert identities == {
        "source_keys": {"amber-lake"},
        "source_values": {"1234567"},
        "queries": {"amber-lake"},
        "answers": {"1234567"},
    }


def test_training_gap_support(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "answer_start": 4000,
                    "source_token_position_answer": 67,
                },
                {
                    "answer_start": 3900,
                    "source_token_position_answer": 3838,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert prepare.training_gap_support(path) == {
        "rows": 2,
        "minimum_tokens": 62,
        "maximum_tokens": 3933,
    }
