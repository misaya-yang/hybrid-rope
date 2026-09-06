"""Synthetic CPU-only input contracts; no real parquet/tokenizer/model execution."""

import argparse
import hashlib
import json

import pytest

from scripts.data import prepare_native_qk_calibration as builder


class StubTokenizer:
    bos_token_id = 1

    def __len__(self):
        return 200000

    def encode(self, text, *, add_special_tokens, truncation, max_length):
        assert not add_special_tokens and truncation and max_length == 32768
        return [ord(char) + 10 for char in text[:max_length]]

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("Native QK data must not use chat templates")


def long_text(index):
    return f"document-{index}: " + "x" * 32768


def test_bos_prefers_valid_tokenizer_then_explicit_config_only():
    tokenizer = StubTokenizer()
    assert builder.choose_bos(tokenizer, {"bos_token_id": 4}) == {
        "token_id": 1, "source": "tokenizer.bos_token_id"}
    for invalid in (None, -1, True, 200000):
        tokenizer.bos_token_id = invalid
        assert builder.choose_bos(tokenizer, {"bos_token_id": 4}) == {
            "token_id": 4, "source": "checkpoint_config.bos_token_id"}
    with pytest.raises(ValueError, match="no valid BOS"):
        builder.choose_bos(tokenizer, {})


def test_fixed_stream_selection_exclusions_dedup_and_eight_eight_split():
    excluded_text = long_text("excluded")
    excluded_hash = hashlib.sha256(excluded_text.encode()).hexdigest()
    texts = [(39999, long_text("too_early")), (40000, excluded_text), (40001, "short"),
             (40002, long_text(0)), (40003, long_text(0))]
    texts.extend((40004 + i, long_text(i + 1)) for i in range(15))
    rows, stats = builder.build_rows(texts, StubTokenizer(), {"token_id": 1}, {excluded_hash})
    assert [len(rows[split]) for split in builder.SPLITS] == [8, 8]
    assert rows["calibration"][0]["source_row"] == 40002
    assert stats["excluded_source_rows"] == stats["excluded_unique_source_documents_encountered"] == 1
    assert stats["short_source_rows"] == stats["duplicate_selected_source_rows"] == 1
    hashes = {row["source_text_sha256"] for values in rows.values() for row in values}
    assert len(hashes) == 16 and excluded_hash not in hashes
    for split, values in rows.items():
        for row in values:
            assert row["split"] == split
            assert len(row["input_ids"]) == row["length"] == 32768
            assert row["input_ids"][0] == 1
            assert row["input_ids"][-1] == ord("x") + 10  # Not an appended EOS.
            assert row["target_start"] == 32768 - 256
            assert row["target_tokens"] == 256
            assert row["input_ids_sha256"] == builder.common.ids_hash(row["input_ids"])
            assert row["target_ids_sha256"] == builder.common.ids_hash(row["input_ids"][-256:])
    with pytest.raises(ValueError, match="found 0"):
        builder.build_rows([], StubTokenizer(), {"token_id": 1}, set())


def test_exclusions_only_use_natural_identity_and_report_actual_counts(tmp_path):
    path = tmp_path / "prior-inputs.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [
        {"family": "natural", "source_text_sha256": "a" * 64},
        {"family": "natural", "source_text_sha256": "a" * 64},
        {"family": "capability", "source_text_sha256": "b" * 64},
    ]))
    excluded, receipts = builder.load_input_exclusions([path])
    assert excluded == {"a" * 64}
    assert receipts[0]["natural_input_rows"] == 2
    assert receipts[0]["unique_source_documents"] == 1
    assert receipts[0]["ignored_non_natural_rows"] == 1
    assert receipts[0]["name"] == path.name


@pytest.mark.parametrize("row", [
    {"decision": "CONFIRMED_CROSSING"},
    {"family": "natural", "source_text_sha256": "a" * 64, "nll": 1.2},
    {"family": "natural", "source_text_sha256": "invalid"},
])
def test_exclusions_reject_model_outputs_decisions_or_invalid_hashes(tmp_path, row):
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        builder.load_input_exclusions([path])


def test_prepare_synthetic_inputs_hashes_privacy_and_fresh_output(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "model_type": "qwen2", "max_position_embeddings": 32768, "bos_token_id": 4}))
    (checkpoint / "tokenizer.json").write_text("{}")
    source = tmp_path / "source.parquet"
    source.write_bytes(b"synthetic source; parquet reader replaced by stub")
    exclude = tmp_path / "prior.jsonl"
    exclude.write_text(json.dumps({"family": "natural", "source_text_sha256": "a" * 64}))
    tokenizer = StubTokenizer()
    tokenizer.bos_token_id = None
    monkeypatch.setattr(builder.common, "load_tokenizer", lambda path: tokenizer)
    def texts(path, start_row):
        assert start_row == 40000
        return ((40000 + i, long_text(i)) for i in range(16))
    monkeypatch.setattr(builder.common, "iter_source_texts", texts)
    args = argparse.Namespace(source=source, checkpoint=checkpoint, exclude_inputs=[exclude], output=tmp_path / "output")
    manifest = builder.prepare(args)
    assert manifest["native_length"] == 32768
    assert manifest["model_evaluation_status"] == "NOT_RUN"
    assert manifest["rope_tables_constructed"] is False
    assert manifest["bos"] == {"token_id": 4, "source": "checkpoint_config.bos_token_id"}
    assert manifest["unique_excluded_source_documents"] == 1
    assert manifest["source_scan"]["excluded_unique_source_documents_encountered"] == 0
    assert manifest["source"]["sha256"] == builder.common.sha256_file(source)
    assert str(tmp_path) not in json.dumps(manifest)
    for split in builder.SPLITS:
        receipt = manifest["files"][split]
        path = args.output / receipt["path"]
        assert receipt["rows"] == 8
        assert receipt["sha256"] == builder.common.sha256_file(path)
        assert len(path.read_text().splitlines()) == 8
    with pytest.raises(FileExistsError):
        builder.prepare(args)
