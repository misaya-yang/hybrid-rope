from pathlib import Path

import pytest

from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    evaluate_instruct_ruler_transfer as evaluate,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    prepare_instruct_ruler_transfer as prepare,
)


EXPECTED_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)


def test_complete_official_ruler_task_registry() -> None:
    assert prepare.DEFAULT_TASKS == EXPECTED_TASKS
    assert set(prepare.TASK_CONFIGS) == set(EXPECTED_TASKS)
    assert prepare.TASK_CONFIGS["niah_multikey_1"]["args"] == {
        "type_haystack": "essay",
        "type_needle_k": "words",
        "type_needle_v": "numbers",
        "num_needle_k": 4,
        "num_needle_v": 1,
        "num_needle_q": 1,
    }
    assert prepare.TASK_CONFIGS["niah_multivalue"]["args"][
        "num_needle_v"
    ] == 4
    assert prepare.TASK_CONFIGS["niah_multiquery"]["args"][
        "num_needle_q"
    ] == 4
    assert prepare.TASK_CONFIGS["qa_1"]["args"] == {"dataset": "squad"}
    assert prepare.TASK_CONFIGS["qa_2"]["args"] == {
        "dataset": "hotpotqa"
    }


def test_official_task_specific_metrics() -> None:
    prediction = "Answer: Alpha and gamma"
    references = ["alpha", "beta", "gamma"]
    assert evaluate.official_string_match_all(
        prediction, references
    ) == pytest.approx(2 / 3)
    assert evaluate.official_string_match_part(
        prediction, references
    ) == 1.0
    assert evaluate.official_task_score(
        prediction, references, "string_match_all"
    ) == pytest.approx(2 / 3)
    assert evaluate.official_task_score(
        prediction, references, "string_match_part"
    ) == 1.0
    with pytest.raises(RuntimeError, match="unsupported"):
        evaluate.official_task_score(
            prediction, references, "rouge"
        )


def test_qa_generator_uses_official_dataset_argument(
    tmp_path: Path,
) -> None:
    config = prepare.TASK_CONFIGS["qa_1"]
    command = prepare._generator_command(
        python="python",
        generator=tmp_path / "qa.py",
        save_dir=tmp_path / "data",
        task="qa_1",
        checkpoint=tmp_path / "checkpoint",
        nominal_length=4096,
        chat_overhead=11,
        samples=100,
        seed=20260728,
        template="{context} {query}",
        config=config,
    )
    dataset_index = command.index("--dataset")
    assert command[dataset_index + 1] == "squad"
    assert command[command.index("--num_samples") + 1] == "100"


def test_qa_assets_fail_closed_when_missing(
    tmp_path: Path,
) -> None:
    synthetic = tmp_path / "synthetic"
    (synthetic / "json").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="squad.json"):
        prepare.verify_external_assets(synthetic, ("qa_1",))
