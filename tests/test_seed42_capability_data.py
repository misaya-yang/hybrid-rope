from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest

from experiments.lora_evq_v2.prepare_seed42_capability_data import (
    MCQA_REVISIONS,
    build_longbench_examples,
    build_nolima_hard_examples,
    build_passkey_examples,
    import_ruler_examples,
    load_jsonl,
    load_mcqa_examples,
    parse_args,
    prepare_suite,
    _chat_prompt_ids,
    _tokenizer_identity,
    truncate_document_only,
    validate_records,
    write_suite_atomic,
)


class FakeTokenizer:
    name_or_path = "fake-tokenizer"
    vocab_size = 256
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids, **_: object) -> str:
        return "".join(chr(int(token_id)) for token_id in token_ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        rendered = "".join(
            f"<{message['role']}>\n{message['content']}\n" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>\n"
        return self.encode(rendered) if tokenize else rendered


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


def test_truncate_document_only_preserves_suffix():
    ids = truncate_document_only(
        prefix_ids=[1, 2],
        document_ids=list(range(100)),
        suffix_ids=[7, 8, 9],
        max_prompt_tokens=12,
    )
    assert ids[:2] == [1, 2]
    assert ids[-3:] == [7, 8, 9]
    assert len(ids) == 12


def test_missing_requested_source_fails_closed(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "missing.jsonl")


def test_chat_prompt_accepts_batch_encoding_like_mapping(fake_tokenizer):
    class MappingTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            value = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
            return {"input_ids": value, "attention_mask": [1] * len(value)} if tokenize else value

    tokenizer = MappingTokenizer()
    assert _chat_prompt_ids(tokenizer, "hello") == FakeTokenizer.apply_chat_template(
        tokenizer,
        [{"role": "user", "content": "hello"}],
        tokenize=True,
        add_generation_prompt=True,
    )


def test_passkey_grid_has_all_lengths_depths_and_unique_ids(fake_tokenizer):
    rows = build_passkey_examples(
        fake_tokenizer,
        lengths=(8192, 16384, 32768),
        depths=(10, 25, 50, 75, 90),
        trials=2,
        seed=42,
    )
    assert len(rows) == 30
    assert len({row["example_id"] for row in rows}) == 30


def test_truncate_document_only_rejects_oversized_fixed_parts():
    with pytest.raises(ValueError, match="prefix plus suffix"):
        truncate_document_only([1, 2, 3], [4, 5], [6, 7], 4)


def test_passkey_records_are_exact_length_deterministic_and_valid(fake_tokenizer):
    kwargs = {
        "lengths": (512,),
        "depths": (10, 90),
        "trials": 2,
        "seed": 42,
    }
    first = build_passkey_examples(fake_tokenizer, **kwargs)
    second = build_passkey_examples(fake_tokenizer, **kwargs)
    assert first == second
    assert {len(row["prompt_ids"]) for row in first} == {512}
    assert validate_records(first) == first


def test_ruler_import_preserves_prompt_and_references(tmp_path: Path, fake_tokenizer):
    path = tmp_path / "s_niah_8192.jsonl"
    prompt = "  Keep this prompt byte-for-byte.\nAnswer: "
    references = ["  first reference ", "second"]
    path.write_text(
        json.dumps(
            {
                "id": "official-1",
                "task": "niah_single_1",
                "input": prompt,
                "outputs": references,
                "length": 8192,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = import_ruler_examples(fake_tokenizer, [path])

    assert len(rows) == 1
    expected = fake_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    assert rows[0]["prompt_ids"] == expected
    assert rows[0]["answers"] == references
    assert rows[0]["target_length"] == 8192
    assert rows[0]["metric"] == "ruler_string_match"
    assert rows[0]["source"]["match_type"] == "all"
    assert rows[0]["generation_tokens"] == 128


def test_ruler_infers_task_from_official_directory_layout(tmp_path: Path, fake_tokenizer):
    path = tmp_path / "8192" / "data" / "niah_single_1" / "validation.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "index": 0,
                "input": "prompt",
                "answer_prefix": "\nAnswer: ",
                "outputs": ["answer"],
                "length": 8000,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = import_ruler_examples(fake_tokenizer, [path])

    assert rows[0]["task"] == "niah_single_1"
    assert rows[0]["target_length"] == 8192
    assert rows[0]["prompt_ids"] == fake_tokenizer.apply_chat_template(
        [{"role": "user", "content": "prompt\nAnswer: "}],
        tokenize=True,
        add_generation_prompt=True,
    )


def test_ruler_rejects_prompt_that_exceeds_declared_length(tmp_path: Path, fake_tokenizer):
    path = tmp_path / "niah_single_1.jsonl"
    path.write_text(
        json.dumps({"input": "12345", "outputs": ["answer"], "length": 4}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exceeds declared target length"):
        import_ruler_examples(fake_tokenizer, [path])


def test_tokenizer_identity_does_not_publish_absolute_local_paths(fake_tokenizer):
    fake_tokenizer.name_or_path = "/srv/models/Meta-Llama-3-8B-Instruct"

    identity = _tokenizer_identity(
        fake_tokenizer,
        "/srv/models/Meta-Llama-3-8B-Instruct",
    )

    assert identity["requested"] == "Meta-Llama-3-8B-Instruct"
    assert identity["name_or_path"] == "Meta-Llama-3-8B-Instruct"
    assert identity["identifier"] == "Meta-Llama-3-8B-Instruct"


def test_tokenizer_identity_hashes_local_tokenizer_files(tmp_path: Path, fake_tokenizer):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer_json.write_text('{"version":"fixture"}\n', encoding="utf-8")
    fake_tokenizer.name_or_path = str(tokenizer_dir)

    identity = _tokenizer_identity(fake_tokenizer, tokenizer_dir)

    assert identity["identifier"] == "tokenizer"
    assert identity["files"]["tokenizer.json"] == hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()


def test_nolima_hard_uses_official_needles_and_book_tokens(tmp_path: Path, fake_tokenizer):
    needles = tmp_path / "needle_set_hard.json"
    needles.write_text(
        json.dumps(
            [
                {
                    "id": "0001",
                    "system_prompt": "",
                    "task_template": "P{haystack}Q{question}S",
                    "needle": "The visitor was {CHAR}.",
                    "questions": {"twohop": "Who visited {1}?"},
                    "character_set": ["Ada", "Lin"],
                    "tests": {"T01": {"input_args": ["Paris"]}},
                }
            ]
        ),
        encoding="utf-8",
    )
    books = tmp_path / "books"
    books.mkdir()
    (books / "book.txt").write_text("book-token\n" * 100, encoding="utf-8")

    rows = build_nolima_hard_examples(
        fake_tokenizer,
        needles,
        books,
        lengths=(256,),
        depths=(50,),
        seed=42,
    )

    assert len(rows) == 1
    assert len(rows[0]["prompt_ids"]) == 256
    assert rows[0]["answers"][0] in {"Ada", "Lin"}
    assert rows[0]["source"]["needle_set"] == "needle_set_hard.json"
    assert rows[0]["suite"] == "nolima_hard_exact_context"
    assert rows[0]["metric"] == "contains"
    assert rows[0]["generation_tokens"] == 192


def test_longbench_keeps_complete_prompts_and_truncates_only_diagnostics(tmp_path: Path, fake_tokenizer):
    path = tmp_path / "narrativeqa.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "_id": "short",
                    "context": "short story",
                    "input": "What happened?",
                    "answers": ["Something"],
                },
                {
                    "_id": "long",
                    "context": "x" * 1000,
                    "input": "What happened?",
                    "answers": ["Something else"],
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_longbench_examples(
        fake_tokenizer,
        [path],
        max_prompt_tokens=512,
        min_complete_prompt_tokens=1,
        diagnostic_lengths=(500,),
    )

    complete = [row for row in rows if row["source"]["selection"] == "complete"]
    diagnostic = [row for row in rows if row["source"]["selection"] == "fixed_diagnostic"]
    assert [row["source"]["source_id"] for row in complete] == ["short"]
    assert len(diagnostic) == 1
    assert len(diagnostic[0]["prompt_ids"]) <= 500


def test_longbench_directory_selects_only_narrativeqa_and_qasper(tmp_path: Path, fake_tokenizer):
    source_dir = tmp_path / "longbench"
    source_dir.mkdir()
    for task in ("narrativeqa", "qasper", "hotpotqa"):
        (source_dir / f"{task}.jsonl").write_text(
            json.dumps(
                {
                    "_id": task,
                    "context": "context",
                    "input": "question",
                    "answers": ["answer"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    rows = build_longbench_examples(
        fake_tokenizer,
        source_dir,
        max_prompt_tokens=2048,
        min_complete_prompt_tokens=1,
    )

    assert {row["task"] for row in rows} == {"narrativeqa", "qasper"}


def test_longbench_complete_prompt_is_tokenized_as_one_string(tmp_path: Path):
    class BoundarySensitiveTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [len(text)]

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            rendered = "|".join(message["content"] for message in messages)
            if add_generation_prompt:
                rendered += "|assistant"
            return self.encode(rendered) if tokenize else rendered

    path = tmp_path / "narrativeqa.jsonl"
    path.write_text(
        json.dumps(
            {
                "_id": "one",
                "context": "story",
                "input": "question",
                "answers": ["answer"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_longbench_examples(
        BoundarySensitiveTokenizer(),
        path,
        max_prompt_tokens=10,
        min_complete_prompt_tokens=1,
    )

    assert len(rows[0]["prompt_ids"]) == 1


def test_longbench_merges_duplicate_prompts_as_reference_variants(tmp_path: Path, fake_tokenizer):
    source = tmp_path / "narrativeqa.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"_id": "a", "context": "same story", "input": "same question", "answers": ["first"]}),
                json.dumps({"_id": "b", "context": "same story", "input": "same question", "answers": ["second"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = build_longbench_examples(
        fake_tokenizer,
        source,
        max_prompt_tokens=32768,
        min_complete_prompt_tokens=1,
    )
    assert len(rows) == 1
    assert rows[0]["answers"] == ["first", "second"]
    assert rows[0]["source"]["merged_source_ids"] == ["a", "b"]


def test_requested_longbench_with_no_selected_tasks_fails_without_manifest(tmp_path: Path, fake_tokenizer):
    hotpotqa = tmp_path / "hotpotqa.jsonl"
    hotpotqa.write_text(
        json.dumps(
            {
                "_id": "other",
                "context": "context",
                "input": "question",
                "answers": ["answer"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "suite"

    with pytest.raises(ValueError, match="LongBench sources produced no selected"):
        prepare_suite(
            Namespace(
                tokenizer=fake_tokenizer,
                output_dir=output_dir,
                ruler_jsonl=(),
                longbench_jsonl=[hotpotqa],
                nolima_needle_set=None,
                nolima_books_dir=None,
                skip_mcqa=True,
                overwrite=False,
                seed=42,
            )
        )

    assert not (output_dir / "manifest.json").exists()


@pytest.mark.parametrize("option", ["--ruler-jsonl", "--longbench-jsonl"])
def test_cli_requested_local_source_requires_at_least_one_path(tmp_path: Path, option: str):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--tokenizer",
                "tokenizer",
                "--output-dir",
                str(tmp_path / "suite"),
                option,
            ]
        )


def test_mcqa_loads_every_source_at_the_pinned_revision(fake_tokenizer):
    calls = []

    def fake_loader(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        rows = {
            "cais/mmlu": [{"question": "M?", "choices": ["a", "b"], "answer": 1}],
            "allenai/ai2_arc": [
                {
                    "id": "arc",
                    "question": "A?",
                    "choices": {"label": ["A", "B"], "text": ["a", "b"]},
                    "answerKey": "A",
                }
            ],
            "Rowan/hellaswag": [{"ind": 3, "ctx": "H?", "endings": ["a", "b"], "label": "0"}],
            "allenai/openbookqa": [
                {
                    "id": "obqa",
                    "question_stem": "O?",
                    "choices": {"label": ["A", "B"], "text": ["a", "b"]},
                    "answerKey": "B",
                }
            ],
            "allenai/winogrande": [
                {
                    "qID": "w",
                    "sentence": "The _ won.",
                    "option1": "a",
                    "option2": "b",
                    "answer": "2",
                }
            ],
        }
        return rows[name]

    rows = load_mcqa_examples(fake_tokenizer, dataset_loader=fake_loader)

    assert len(rows) == 5
    assert {name: kwargs["revision"] for name, _, kwargs in calls} == MCQA_REVISIONS
    assert MCQA_REVISIONS["allenai/openbookqa"] == ("388097ea7776314e93a529163e0fea805b8a6454")
    openbook_call = next(call for call in calls if call[0] == "allenai/openbookqa")
    assert openbook_call[1] == ("main",)
    assert openbook_call[2]["split"] == "validation"
    assert all(row["metric"] == "mcqa" for row in rows)
    assert all(row["answers"] == [row["choices"][row["answer_index"]]] for row in rows)
    invalid = dict(rows[0], answers=["not the indexed choice"])
    with pytest.raises(ValueError, match="indexed choice"):
        validate_records([invalid])


def test_mcqa_uses_explicit_local_arrow_rows_without_hub_loader(fake_tokenizer):
    rows = load_mcqa_examples(
        fake_tokenizer,
        sources=("cais/mmlu",),
        dataset_loader=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Hub loader called")),
        local_datasets={
            "cais/mmlu": [
                {"question": "Cached?", "choices": ["no", "yes"], "answer": 1}
            ]
        },
    )
    assert len(rows) == 1
    assert rows[0]["answers"] == ["yes"]
    assert rows[0]["source"]["revision"] == MCQA_REVISIONS["cais/mmlu"]


def test_mcqa_arrow_cache_resolves_exact_pinned_revision(tmp_path: Path):
    expected = (
        tmp_path
        / "datasets/cais___mmlu/all/0.0.0"
        / MCQA_REVISIONS["cais/mmlu"]
        / "mmlu-test.arrow"
    )
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"arrow")
    from experiments.lora_evq_v2.prepare_seed42_capability_data import mcqa_arrow_path

    assert mcqa_arrow_path(tmp_path, "cais/mmlu") == expected


def test_mcqa_max_examples_must_be_positive(fake_tokenizer):
    with pytest.raises(ValueError, match="must be positive"):
        load_mcqa_examples(
            fake_tokenizer,
            sources=("cais/mmlu",),
            dataset_loader=lambda *args, **kwargs: [],
            max_examples_per_source=0,
        )


def test_empty_requested_mcqa_source_fails_closed(fake_tokenizer):
    with pytest.raises(ValueError, match="cais/mmlu.*produced no eligible records"):
        load_mcqa_examples(
            fake_tokenizer,
            sources=("cais/mmlu",),
            dataset_loader=lambda *args, **kwargs: [],
        )


def test_atomic_writer_rejects_duplicate_cell_hash_and_writes_manifest_last(tmp_path: Path, fake_tokenizer):
    rows = build_passkey_examples(fake_tokenizer, lengths=(512,), depths=(50,), trials=1, seed=42)
    duplicate = dict(rows[0], example_id="different-id")
    with pytest.raises(ValueError, match="duplicate prompt hash"):
        validate_records(rows + [duplicate])

    output_dir = tmp_path / "suite"
    manifest = write_suite_atomic(
        output_dir=output_dir,
        records_by_file={"passkey.jsonl": rows},
        tokenizer_identity={"name_or_path": "fake-tokenizer"},
        source_revisions={"passkey": {"seed": 42}},
    )

    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "passkey.jsonl").is_file()
    assert not list(output_dir.glob("*.incomplete"))
    assert manifest["files"]["passkey.jsonl"]["row_count"] == 1


def test_atomic_overwrite_removes_stale_managed_jsonl(tmp_path: Path, fake_tokenizer):
    rows = build_passkey_examples(fake_tokenizer, lengths=(512,), depths=(50,), trials=2, seed=42)
    output_dir = tmp_path / "suite"
    write_suite_atomic(
        output_dir=output_dir,
        records_by_file={"passkey.jsonl": [rows[0]], "mcqa.jsonl": [rows[1]]},
        tokenizer_identity={"name_or_path": "fake-tokenizer"},
        source_revisions={"first": "v1"},
    )

    manifest = write_suite_atomic(
        output_dir=output_dir,
        records_by_file={"passkey.jsonl": [rows[0]]},
        tokenizer_identity={"name_or_path": "fake-tokenizer"},
        source_revisions={"second": "v2"},
        overwrite=True,
    )

    assert set(manifest["files"]) == {"passkey.jsonl"}
    assert not (output_dir / "mcqa.jsonl").exists()
