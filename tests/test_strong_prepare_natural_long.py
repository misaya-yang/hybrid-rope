import json
from pathlib import Path

import pytest

from experiments.iclr2027_strong_evidence_20260915.prepare_natural_long import (
    DataRootError,
    INFINITEBENCH,
    LONGBENCH_V2,
    PrepareConfig,
    encode_chat_prompt,
    locate_sources,
    longbench_v2_prompt,
    main,
    prepare_dataset,
)


class WordTokenizer:
    chat_template = "fake-chat-v1"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        text = "<user> " + messages[0]["content"] + " <assistant>"
        return list(range(1, len(text.split()) + 1))


class MappingTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return {"input_ids": [1, 2, 3]}


def test_encode_chat_prompt_accepts_mapping_style_batch_encoding():
    assert encode_chat_prompt(MappingTokenizer(), "prompt") == [1, 2, 3]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def lb2_row(row_id: str, context_words: int, *, domain: str = "Single-Document QA") -> dict:
    return {
        "_id": row_id,
        "domain": domain,
        "sub_domain": "test",
        "difficulty": "easy",
        "length": "short",
        "question": "Which answer is supported?",
        "choice_A": "alpha",
        "choice_B": "beta",
        "choice_C": "gamma",
        "choice_D": "delta",
        "answer": "A",
        "context": "word " * context_words,
    }


def test_longbench_inventory_uses_full_prompt_stable_ids_and_all_reasons(tmp_path):
    source = tmp_path / "data" / "longbench_v2.jsonl"
    write_jsonl(source, [lb2_row("b", 170), lb2_row("a", 160), lb2_row("short", 5), lb2_row("huge", 1200)])
    out = tmp_path / "frozen"
    config = PrepareConfig(
        benchmark=LONGBENCH_V2,
        model_id="meta-llama/Llama-3-8B-Instruct",
        data_root=source.parent,
        out=out,
        scale=8,
        lengths=(320, 640, 1024),
        native_length=128,
        rows_per_task=0,
    )
    manifest = prepare_dataset(config, WordTokenizer(), {LONGBENCH_V2: source})

    candidates = read_jsonl(out / "candidates.jsonl")
    inputs = read_jsonl(out / "inputs.jsonl")
    assert [row["source_id"] for row in candidates] == ["a", "b", "huge", "short"]
    assert [row["source_id"] for row in inputs] == ["a", "b"]
    assert all(row["score_contract"] == "longbench_v2_mc_direct_answer_v1" for row in inputs)
    assert all(row["references"] == ["A"] for row in inputs)
    assert all(row["input_tokens"] == len(row["prompt_ids"]) for row in inputs)
    assert manifest["summary"]["by_reason"]["eligible"] == 2
    assert manifest["summary"]["by_reason"]["within_native"] == 1
    assert manifest["summary"]["by_reason"]["exceeds_max_complete_budget"] == 1
    assert manifest["summary"]["source_clusters"]["selected"] == 2
    assert manifest["source_files"][LONGBENCH_V2]["file"] == "longbench_v2.jsonl"
    assert str(tmp_path) not in json.dumps(manifest)
    assert "<text>" in longbench_v2_prompt(lb2_row("x", 2))


def test_longbench_prompt_matches_official_field_stripping():
    row = lb2_row("strip", 2)
    row.update({
        "context": "\n context body \t",
        "question": "  Which answer?\n",
        "choice_A": " alpha ",
        "choice_B": "\nbeta\n",
        "choice_C": " gamma\t",
        "choice_D": "\tdelta ",
    })
    prompt = longbench_v2_prompt(row)
    assert "<text>\ncontext body\n</text>" in prompt
    assert "question: Which answer?\nChoices:" in prompt
    assert "(A) alpha\n(B) beta\n(C) gamma\n(D) delta\n" in prompt


def infinite_row(index: int, words: int) -> dict:
    return {"id": f"row-{index}", "context": "dialogue " * words, "input": "Who?", "answer": f"name-{index}"}


def test_infinitebench_keeps_source_order_and_full_candidate_inventory(tmp_path):
    root = tmp_path / "InfiniteBench" / "data"
    sources = {}
    for task in ("longdialogue_qa_eng", "longbook_qa_eng"):
        path = root / f"{task}.jsonl"
        write_jsonl(path, [infinite_row(0, 80), infinite_row(1, 90), infinite_row(2, 2)])
        sources[task] = path
    out = tmp_path / "out"
    config = PrepareConfig(
        benchmark=INFINITEBENCH,
        model_id="meta-llama/Llama-3-8B-Instruct",
        data_root=tmp_path,
        out=out,
        scale=4,
        lengths=(256,),
        native_length=64,
        rows_per_task=1,
    )
    manifest = prepare_dataset(config, WordTokenizer(), sources)
    candidates = read_jsonl(out / "candidates.jsonl")
    inputs = read_jsonl(out / "inputs.jsonl")

    assert len(candidates) == 6
    assert [(row["task"], row["source_index"]) for row in inputs] == [
        ("longdialogue_qa_eng", 0), ("longbook_qa_eng", 0),
    ]
    limited = [row for row in candidates if row["selection_reason"] == "source_order_limit"]
    assert {(row["task"], row["source_index"]) for row in limited} == {
        ("longdialogue_qa_eng", 1), ("longbook_qa_eng", 1),
    }
    assert manifest["summary"]["selected_rows"] == 2
    assert manifest["summary"]["eligible_rows"] == 4
    assert set(manifest["summary"]["by_task"]) == {"longdialogue_qa_eng", "longbook_qa_eng"}
    assert all("prompt_ids" not in row for row in candidates)


def test_infinitebench_can_freeze_one_task_and_enforce_minimum_length(tmp_path):
    root = tmp_path / "InfiniteBench" / "data"
    sources = {}
    for task in ("longdialogue_qa_eng", "longbook_qa_eng"):
        path = root / f"{task}.jsonl"
        write_jsonl(path, [infinite_row(0, 40), infinite_row(1, 100), infinite_row(2, 110)])
        sources[task] = path
    out = tmp_path / "out"
    config = PrepareConfig(
        benchmark=INFINITEBENCH, model_id="llama", data_root=tmp_path,
        out=out, scale=8, lengths=(256,), native_length=32, rows_per_task=1,
        tasks=("longdialogue_qa_eng",), minimum_input_tokens=130,
    )
    manifest = prepare_dataset(config, WordTokenizer(), sources)
    candidates = read_jsonl(out / "candidates.jsonl")
    inputs = read_jsonl(out / "inputs.jsonl")

    assert {row["task"] for row in candidates} == {"longdialogue_qa_eng"}
    assert len(inputs) == 1 and inputs[0]["source_index"] == 1
    assert manifest["tasks"] == ["longdialogue_qa_eng"]
    assert manifest["minimum_input_tokens"] == 130
    assert manifest["summary"]["by_reason"]["below_minimum_input_tokens"] == 1


def test_missing_or_ambiguous_official_data_is_explicit(tmp_path):
    with pytest.raises(DataRootError, match="missing local official data files"):
        locate_sources(INFINITEBENCH, tmp_path)

    first = tmp_path / "a" / "longbench_v2.jsonl"
    second = tmp_path / "b" / "longbench-v2.jsonl"
    write_jsonl(first, [])
    write_jsonl(second, [])
    with pytest.raises(DataRootError, match="ambiguous"):
        locate_sources(LONGBENCH_V2, tmp_path)


def test_single_infinitebench_task_does_not_require_the_other_source(tmp_path):
    path = tmp_path / "data" / "longdialogue_qa_eng.jsonl"
    write_jsonl(path, [infinite_row(0, 100)])
    assert locate_sources(
        INFINITEBENCH, tmp_path, tasks=("longdialogue_qa_eng",),
    ) == {"longdialogue_qa_eng": path.resolve()}


def test_cli_missing_data_writes_portable_status_without_loading_a_model(tmp_path):
    out = tmp_path / "missing"
    with pytest.raises(SystemExit, match="2"):
        main([
            "--benchmark", "infinitebench",
            "--model", str(tmp_path / "model-does-not-exist"),
            "--model-id", "llama3_8b",
            "--data-root", str(tmp_path / "data-does-not-exist"),
            "--out", str(out),
            "--scale", "16",
            "--lengths", "131072",
        ])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "MISSING_DATA"
    assert manifest["downloads_attempted"] is False
    assert str(tmp_path) not in json.dumps(manifest)


def test_invalid_official_row_is_retained_but_not_frozen(tmp_path):
    source = tmp_path / "longbench_v2.jsonl"
    invalid = lb2_row("bad", 170)
    invalid.pop("choice_D")
    write_jsonl(source, [invalid])
    out = tmp_path / "out"
    config = PrepareConfig(
        benchmark=LONGBENCH_V2,
        model_id="llama",
        data_root=tmp_path,
        out=out,
        scale=4,
        lengths=(256, 512),
        native_length=128,
        rows_per_task=0,
    )
    manifest = prepare_dataset(config, WordTokenizer(), {LONGBENCH_V2: source})
    candidate = read_jsonl(out / "candidates.jsonl")[0]
    assert candidate["reason"] == "missing_or_invalid_fields:choice_D"
    assert read_jsonl(out / "inputs.jsonl") == []
    assert manifest["summary"]["selected_rows"] == 0
