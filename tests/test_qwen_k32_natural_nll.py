"""Synthetic CPU-only contracts; no real parquet, model, or GPU use."""

import argparse
import hashlib
import json

import numpy as np
import pytest

from scripts.data import prepare_qwen_k32_natural_nll as builder
from scripts.eval import eval_qwen_k32_natural_nll as evaluator


class FakeTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [10 + ord(char) % 47 for char in text]


@pytest.fixture
def small_contract(monkeypatch):
    monkeypatch.setattr(builder, "GRID", (8, 16))
    monkeypatch.setattr(builder, "DOCUMENTS", 2)
    monkeypatch.setattr(builder, "TARGET_TOKENS", 4)
    monkeypatch.setattr(evaluator, "GRID", (8, 16))
    monkeypatch.setattr(evaluator, "DOCUMENTS", 2)
    monkeypatch.setattr(evaluator, "TARGET_TOKENS", 4)


def test_packing_is_source_ordered_nonoverlapping_and_suffix_paired(small_contract):
    rows = [(100 + index, chr(65 + index) * 3) for index in range(20)]
    streams = builder.pack_streams(rows, FakeTokenizer())
    assert len(streams) == 2
    assert all(len(stream["input_ids"]) == 16 for stream in streams)
    used_rows = [item["source_row"] for stream in streams for item in stream["sources"]]
    assert used_rows == sorted(used_rows)
    assert len(used_rows) == len(set(used_rows))
    paired = builder.paired_rows(streams)
    assert len(paired) == 4
    for index in range(2):
        short, long = [row for row in paired if row["sample_id"].endswith(f"{index:03d}")]
        assert short["length"] == 8 and long["length"] == 16
        assert short["input_ids"] == long["input_ids"][-8:]
        assert short["target_ids_sha256"] == long["target_ids_sha256"]
    assert streams[0]["input_ids"].count(FakeTokenizer.eos_token_id) >= 1


def test_last_document_is_truncated_without_reuse(small_contract):
    streams = builder.pack_streams([(1, "x" * 40), (2, "y" * 40)], FakeTokenizer())
    assert [stream["sources"][0]["source_row"] for stream in streams] == [1, 2]
    assert all(stream["sources"][0]["truncated_to_finish_stream"] for stream in streams)
    assert all(len(stream["input_ids"]) == 16 for stream in streams)


def checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "qwen2", "head_dim": 64,
        "max_position_embeddings": builder.GRID[0], "rope_scaling": None,
        "use_sliding_window": False}))
    (root / "tokenizer.json").write_text("{}")
    return root


def test_preparer_and_loader_roundtrip_with_hash_and_packing_receipts(
        tmp_path, monkeypatch, small_contract):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"synthetic parquet; reader is mocked")
    ckpt = checkpoint(tmp_path)
    monkeypatch.setattr(builder, "iter_source_texts", lambda *_: iter(
        [(500 + index, chr(65 + index) * 20) for index in range(8)]))
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: FakeTokenizer())
    output = tmp_path / "data"
    manifest = builder.prepare(argparse.Namespace(
        source=source, expected_source_sha256=builder.sha256_file(source),
        start_row=500, checkpoint=ckpt, output=output))
    assert manifest["model_outcomes_read"] is False
    assert manifest["source"]["consumed_row_range"] == [500, 501]
    assert len(manifest["stream_receipts"]) == 2
    loaded, rows = evaluator.load_data(output, ckpt)
    assert loaded["file"]["sha256"] == builder.sha256_file(output / "rows.jsonl")
    assert len(rows) == 4
    path = output / "rows.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluator.load_data(output, ckpt)


def test_three_profiles_are_strictly_hash_bound(tmp_path, monkeypatch):
    index = np.geomspace(1, 1e-4, 32).astype(np.float32)
    yarn = np.geomspace(1, 5e-5, 32).astype(np.float32)
    index_path, yarn_path = tmp_path / "index.npy", tmp_path / "yarn.npy"
    np.save(index_path, index, allow_pickle=False); np.save(yarn_path, yarn, allow_pickle=False)
    monkeypatch.setattr(evaluator, "EXPECTED_INDEX_SHA256", evaluator.tensor_hash(index))
    monkeypatch.setattr(evaluator, "EXPECTED_INDEX_FILE_SHA256", evaluator.sha256_file(index_path))
    monkeypatch.setattr(evaluator, "EXPECTED_YARN_SHA256", evaluator.tensor_hash(yarn))
    monkeypatch.setattr(evaluator, "EXPECTED_YARN_FILE_SHA256", evaluator.sha256_file(yarn_path))
    profiles = evaluator.load_profiles(index_path, yarn_path)
    assert [profile["name"] for profile in profiles] == list(evaluator.ARM_ORDER)
    assert profiles[0]["attention_scaling"] == 1
    assert profiles[1]["attention_scaling"] == pytest.approx(1 + .074 * np.log(2))
    assert profiles[2]["attention_scaling"] == pytest.approx(1 + .1 * np.log(2))
    index[4] = index[3]
    np.save(index_path, index, allow_pickle=False)
    monkeypatch.setattr(evaluator, "EXPECTED_INDEX_FILE_SHA256", evaluator.sha256_file(index_path))
    monkeypatch.setattr(evaluator, "EXPECTED_INDEX_SHA256", evaluator.tensor_hash(index))
    with pytest.raises(ValueError, match="invalid frozen"):
        evaluator.load_profiles(index_path, yarn_path)


def test_alignment_uses_exact_final_four_targets(small_contract):
    logits = np.arange(2 * 5 * 3).reshape(2, 5, 3)
    ids = np.tile(np.arange(16), (2, 1))
    aligned, targets = evaluator.aligned_suffix(logits, ids)
    assert aligned.shape == (2, 4, 3)
    assert np.array_equal(targets[0], [12, 13, 14, 15])
    with pytest.raises(ValueError, match="257 logits"):
        evaluator.aligned_suffix(logits[:, :-1], ids)


def test_data_row_target_corruption_fails_closed(tmp_path, monkeypatch, small_contract):
    source = tmp_path / "source.parquet"; source.write_bytes(b"source")
    ckpt = checkpoint(tmp_path)
    monkeypatch.setattr(builder, "iter_source_texts", lambda *_: iter(
        [(700 + index, chr(65 + index) * 20) for index in range(8)]))
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: FakeTokenizer())
    output = tmp_path / "data"
    builder.prepare(argparse.Namespace(source=source,
        expected_source_sha256=builder.sha256_file(source), start_row=700,
        checkpoint=ckpt, output=output))
    path = output / "rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["target_ids_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file"]["sha256"] = builder.sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid or duplicated"):
        evaluator.load_data(output, ckpt)
