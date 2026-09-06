from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.data_prep.target_free_context_builder import (
    BUCKET_NAMES,
    PG19_MULTIPLIERS,
    PG19_TAIL_NLL_TOKENS,
    build_pg19_anchors,
    build_longbench,
    choose_bucket,
    evidence_receipt,
    official_reserve,
    row_sha256,
)


class FixtureTokenizer:
    """Small deterministic tokenizer with a real chat-template boundary."""

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return {"input_ids": self._encode(text)}

    def _encode(self, text: str) -> list[int]:
        return [
            10 + sum((index + 1) * ord(char) for index, char in enumerate(word)) % 1000
            for word in text.replace("\n", " \n ").split()
        ]

    def apply_chat_template(self, messages, *, add_generation_prompt: bool, tokenize: bool):
        assert add_generation_prompt is True
        rendered = "<user> " + messages[0]["content"] + " <assistant>"
        return self._encode(rendered) if tokenize else rendered


def test_relative_native_buckets_are_closed_and_unpadded() -> None:
    assert choose_bucket(16, 16).name == "retention"
    assert choose_bucket(17, 16).name == "near"
    assert choose_bucket(32, 16).name == "near"
    assert choose_bucket(33, 16).name == "far"
    assert choose_bucket(64, 16).name == "far"
    rejected = choose_bucket(65, 16)
    assert rejected.name is None
    assert rejected.reason is not None


def test_reserves_and_hashes_are_deterministic() -> None:
    row = {"_id": "x", "context": "natural", "input": "question"}
    assert row_sha256(row) == row_sha256(dict(row))
    assert official_reserve("qasper", {"gov_report": 512}) == (
        64,
        "target-free QA reserve contract",
    )
    assert official_reserve("gov_report", {"gov_report": 512}) == (
        512,
        "LongBench dataset2maxlen.json",
    )
    assert BUCKET_NAMES == ("retention", "near", "far")


def test_evidence_distance_is_recorded_only_when_token_offset_exists() -> None:
    assert evidence_receipt(
        {"evidence_token_start": 7}, query_start_token=19
    )["evidence_to_query_token_distance"] == 12
    missing = evidence_receipt({"supporting_facts": [["doc", 2]]})
    assert missing["evidence_annotation_present"] is True
    assert missing["evidence_to_query_token_distance"] is None


def test_longbench_builder_uses_chat_template_reserves_and_row_hash_order(
    tmp_path: Path,
) -> None:
    longbench_root = tmp_path / "longbench"
    (longbench_root / "official_config").mkdir(parents=True)
    (longbench_root / "extracted" / "longbench").mkdir(parents=True)
    (longbench_root / "extracted" / "longbench_e").mkdir(parents=True)
    (longbench_root / "official_config" / "dataset2prompt.json").write_text(
        json.dumps(
            {
                "qasper": "Q: {input}\\nC: {context}",
                "gov_report": "Summarize {input}\\nReport: {context}",
            }
        ),
        encoding="utf-8",
    )
    (longbench_root / "official_config" / "dataset2maxlen.json").write_text(
        json.dumps({"gov_report": 4}),
        encoding="utf-8",
    )
    rows = [
        {
            "_id": "q-1",
            "input": "What is retained?",
            "context": "A natural archival record is retained.",
            "answers": ["record"],
        },
        {
            "_id": "g-1",
            "input": "Give the finding.",
            "context": "The report states a measured finding.",
            "answers": ["finding"],
        },
    ]
    (longbench_root / "extracted" / "longbench" / "qasper.jsonl").write_text(
        json.dumps(rows[:1]) + "\n", encoding="utf-8"
    )
    (longbench_root / "extracted" / "longbench" / "gov_report.jsonl").write_text(
        json.dumps(rows[1:]) + "\n", encoding="utf-8"
    )
    output = tmp_path / "prepared"
    manifest = build_longbench(
        longbench_root=longbench_root,
        tokenizer=FixtureTokenizer(),
        tokenizer_sha256="fixture-tokenizer",
        native_context_length=128,
        output=output,
    )
    assert manifest["padding"] is False
    assert manifest["synthetic_needles"] is False
    assert manifest["main_result_truncation"] is False
    qasper = manifest["cells"]["longbench:qasper"]
    gov = manifest["cells"]["longbench:gov_report"]
    assert qasper["generation_reserve"] == 64
    assert gov["generation_reserve"] == 4
    assert qasper["kept_rows"] == 1
    assert gov["kept_rows"] == 1
    qasper_row = json.loads(
        (output / "longbench/qasper/rows.jsonl").read_text().strip()
    )
    assert qasper_row["tokenizer_sha256"] == "fixture-tokenizer"
    assert qasper_row["input_tokens"] + 64 == qasper_row["total_tokens_with_reserve"]
    assert qasper_row["row_sha256"] == row_sha256(rows[0])
    assert qasper_row["references"] == ["record"]


def test_pg19_nested_anchor_views_share_the_same_tail(tmp_path: Path) -> None:
    download_root = tmp_path / "download"
    pg19_root = download_root / "pg19"
    (pg19_root / "books" / "test").mkdir(parents=True)
    (download_root / "download_manifest.json").write_text(
        json.dumps({"status": "DOWNLOAD_COMPLETE_TOKEN_MANIFEST_PENDING"}),
        encoding="utf-8",
    )
    # One token per word; the fixture is intentionally much longer than 4x L
    # and keeps a full 512-token tail in the 1x view.
    text = " ".join(f"word-{index}" for index in range(2_100))
    (pg19_root / "books" / "test" / "book-a.txt").write_text(
        text,
        encoding="utf-8",
    )
    tokenizer = FixtureTokenizer()
    output = tmp_path / "prepared"
    manifest = build_pg19_anchors(
        pg19_root=pg19_root,
        tokenizer=tokenizer,
        tokenizer_sha256="fixture-tokenizer",
        native_context_length=512,
        output=output,
        anchor_count=1,
    )
    assert manifest["multipliers"] == list(PG19_MULTIPLIERS)
    assert manifest["nll_target_tokens"] == PG19_TAIL_NLL_TOKENS
    rows = [
        json.loads(line)
        for line in (output / "pg19_nested_anchors.jsonl").read_text().splitlines()
        if line
    ]
    assert [row["context_tokens"] for row in rows] == [512, 1024, 2048]
    assert len({row["nll_target_sha256"] for row in rows}) == 1
    assert all(row["nll_target_tokens"] == PG19_TAIL_NLL_TOKENS for row in rows)
