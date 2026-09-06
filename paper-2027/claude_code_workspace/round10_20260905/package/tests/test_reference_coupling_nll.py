"""CPU fake-data tests only: no models, downloads, SSH, or GPU evaluation."""

import argparse
import hashlib
import json
import math

import numpy as np
import pytest

from scripts.data import prepare_reference_coupling_nll as builder
from scripts.eval import eval_reference_coupling_nll as evaluator


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, **kwargs):
        return [ord(char) + 10 for char in text[:kwargs.get("max_length", len(text))]]


def checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "gemma", "head_dim": 256,
        "max_position_embeddings": 8192, "rope_theta": 10000, "rope_scaling": None}))
    (root / "tokenizer.json").write_text("{}")
    return root


def p0_inputs(tmp_path, ckpt):
    root = tmp_path / "p0"
    root.mkdir()
    manifest = {"status": "NATIVE_REFERENCE_CALIBRATION_DATA_READY_V1",
                "config_sha256": builder.sha256_file(ckpt / "config.json"),
                "tokenizer_files": builder.tokenizer_file_receipts(ckpt), "files": {}}
    for split, count in (("calibration", 32), ("confirmation", 64)):
        rows = [{"family": "natural", "split": split, "sample_id": f"natural-{split}-{i:03d}",
                 "length": length, "source_text_sha256": hashlib.sha256(f"{split}-{i}".encode()).hexdigest()}
                for i in range(count) for length in (1024, 2048, 4096, 8192)]
        path = root / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["files"][split] = {"path": path.name, "sha256": builder.sha256_file(path),
                                    "rows": len(rows), "natural_documents": count}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def fresh_data(tmp_path, monkeypatch):
    ckpt = checkpoint(tmp_path)
    p0 = p0_inputs(tmp_path, ckpt)
    source = tmp_path / "source.parquet"
    source.write_bytes(b"fake parquet; reader mocked")
    texts = [(20000 + i, f"doc-{i:03d}-" + "x" * 16400) for i in range(32)]
    monkeypatch.setattr(builder, "load_tokenizer", lambda _: FakeTokenizer())
    monkeypatch.setattr(builder, "iter_source_texts", lambda *_: iter(texts))
    args = argparse.Namespace(source=source, checkpoint=ckpt, p0_data_root=p0, output=tmp_path / "holdout")
    manifest = builder.prepare(args)
    return ckpt, args.output, manifest


def profile_files(tmp_path, ckpt, target=8192):
    native = (10000 ** (-np.arange(128, dtype=float) / 128)).astype(np.float32)
    native_hash = evaluator.tensor_hash(native)
    config_hash = builder.sha256_file(ckpt / "config.json")
    reference = {"status": "NATIVE_REFERENCE_CONFIRMED", "reference_length": 4096,
                 "checkpoint_weight_sha256": "a" * 64, "config_sha256": config_hash,
                 "native_sha256_float32": native_hash, "confirmation_decision_sha256": "b" * 64,
                 "data_manifest_sha256": "c" * 64, "receipt_sha256": "d" * 64}
    identity = {"config_sha256": config_hash, "native_sha256_float32": native_hash,
                "native_length": 8192, "reference_length": 4096, "pairs": 128}
    scale = target / 4096
    coupling = {"status": "FROZEN_COUPLING_TRANSPORT_EXPORTED",
        "checkpoint": {**identity, "reference_calibration": reference},
        "reference_scope": {"L_config": 8192, "L_ref": 4096, "target_length": target,
                            "s": scale, "table_parameters_refit": False},
        "law": {"family": "clipped_affine", "x_high": evaluator.DEFAULT_X_HIGH,
                "x_low": evaluator.DEFAULT_X_LOW, "scale": scale},
        "gain": {"coefficient": .074, "attention_scaling": 1 + .074 * math.log(scale)}, "tables": {}}
    baseline = {"status": "STATIC_ROPE_BASELINES_EXPORTED", "checkpoint": identity,
        "reference_receipt": reference, "reference_length": 4096, "target_length": target,
        "factor": scale, "search_performed": False, "long_benchmark_scores_used": False,
        "tables": {"static_ntk": {"path": "deliberately_missing_and_must_not_be_loaded.npy"}}}
    paths = []
    for directory, manifest, names in (("coupling", coupling, ["dimensionless_x", "normalized_raw_index"]),
                                       ("baseline", baseline, ["official_equation_yarn"])):
        root = tmp_path / directory
        root.mkdir()
        for name in names:
            values = (native / scale).astype(np.float32)
            path = root / f"{name}.npy"
            np.save(path, values, allow_pickle=False)
            entry = {"path": path.name, "file_sha256": builder.sha256_file(path),
                     "tensor_sha256": evaluator.tensor_hash(values)}
            if name == "official_equation_yarn":
                entry.update(attention_scaling=1 + .1 * math.log(scale), beta_fast=32,
                             beta_slow=1, original_max_position_embeddings=4096)
            manifest["tables"][name] = entry
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        paths.append(path)
    return paths, native


def test_p0_excludes_all_96_inputs_and_checks_hashes(tmp_path):
    ckpt = checkpoint(tmp_path)
    p0 = p0_inputs(tmp_path, ckpt)
    excluded, receipt, _ = builder.load_p0_exclusions(p0)
    assert len(excluded) == 96
    assert receipt["model_outcomes_read"] is False
    path = p0 / "confirmation.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        builder.load_p0_exclusions(p0)


def test_selection_is_first_unique_long_unexcluded_in_source_order():
    excluded_text = "excluded" + "x" * 16384
    first = "first" + "x" * 16384
    rows = [(19999, "early" + "x" * 16384), (20000, excluded_text), (20001, "short"),
            (20002, first), (20003, first)]
    rows += [(20004 + i, f"doc-{i:03d}" + "x" * 16384) for i in range(31)]
    excluded = {hashlib.sha256(excluded_text.encode()).hexdigest()}
    selected = builder.select_documents(rows, FakeTokenizer(), excluded)
    assert len(selected) == 32
    assert selected[0]["source_row"] == 20002
    assert len({doc["source_text_sha256"] for doc in selected}) == 32
    assert not {doc["source_text_sha256"] for doc in selected} & excluded
    with pytest.raises(ValueError, match="found 0"):
        builder.select_documents([], FakeTokenizer(), set())


def test_nested_targets_are_exact_and_no_eos_is_appended():
    document = {"document_ids": list(range(100, 16484)), "source_row": 20000,
                "source_text_sha256": "a" * 64}
    rows = list(builder.natural_rows(document, 0, 1))
    assert [row["length"] for row in rows] == [4096, 8192, 16384]
    for row in rows:
        assert len(row["input_ids"]) == row["length"]
        assert row["input_ids"][0] == 1
        assert row["input_ids"][-256:] == document["document_ids"][:16383][-256:]
        assert row["input_ids"][-1] != 2
        assert row["target_start"] == row["length"] - 256
    assert len({row["target_ids_sha256"] for row in rows}) == 1


def test_prepared_holdout_roundtrips_and_rejects_data_corruption(tmp_path, monkeypatch):
    ckpt, root, manifest = fresh_data(tmp_path, monkeypatch)
    assert manifest["model_evaluation_status"] == "NOT_RUN"
    assert manifest["p0_inputs"]["excluded_documents"] == 96
    _, rows = evaluator.load_data(root, ckpt, [4096, 8192])
    assert len(rows) == 64
    assert {row["length"] for row in rows} == {4096, 8192}
    path = root / "holdout.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="file hash mismatch"):
        evaluator.load_data(root, ckpt, [4096, 8192])


def test_profiles_load_exactly_four_arms_not_ntk_and_keep_reference_distinct(tmp_path):
    ckpt = checkpoint(tmp_path)
    paths, native = profile_files(tmp_path, ckpt)
    profiles, receipt = evaluator.load_profiles(*paths, ckpt / "config.json", "a" * 64, [4096, 8192])
    assert tuple(profile["name"] for profile in profiles) == evaluator.ARM_ORDER
    assert profiles[0]["values"] is None
    assert profiles[0]["attention_scaling"] == 1
    assert receipt["L_config"] == 8192 and receipt["L_ref"] == 4096
    assert receipt["target_length"] == 8192 and receipt["scale"] == 2
    assert receipt["native_sha256_float32"] == evaluator.tensor_hash(native)


@pytest.mark.parametrize("mutation", ["target", "reference", "gain", "weights", "construction", "file_hash", "order"])
def test_profile_identity_fails_closed(tmp_path, mutation):
    ckpt = checkpoint(tmp_path)
    paths, _ = profile_files(tmp_path, ckpt)
    coupling = json.loads(paths[0].read_text())
    lengths, weight = [4096, 8192], "a" * 64
    if mutation == "target":
        lengths = [16384]
    elif mutation == "reference":
        coupling["reference_scope"]["L_ref"] = 8192
    elif mutation == "gain":
        coupling["gain"]["attention_scaling"] = 1.0
    elif mutation == "weights":
        weight = "f" * 64
    elif mutation == "construction":
        coupling["law"]["x_high"] += .01
    else:
        entry = coupling["tables"]["dimensionless_x"]
        table_path = paths[0].parent / entry["path"]
        values = np.load(table_path, allow_pickle=False)
        values[1] = values[0]
        np.save(table_path, values, allow_pickle=False)
        if mutation == "order":
            entry["file_sha256"] = builder.sha256_file(table_path)
            entry["tensor_sha256"] = evaluator.tensor_hash(values)
    paths[0].write_text(json.dumps(coupling))
    with pytest.raises(ValueError):
        evaluator.load_profiles(*paths, ckpt / "config.json", weight, lengths)


def test_s4_profile_may_cover_16k_without_changing_reference(tmp_path):
    ckpt = checkpoint(tmp_path)
    paths, _ = profile_files(tmp_path, ckpt, target=16384)
    _, receipt = evaluator.load_profiles(*paths, ckpt / "config.json", "a" * 64, [4096, 8192, 16384])
    assert receipt["L_ref"] == 4096 and receipt["scale"] == 4


def test_last_257_logits_alignment_uses_final_256_gold_tokens():
    logits = np.arange(2 * 257 * 3).reshape(2, 257, 3)
    ids = np.tile(np.arange(4096), (2, 1))
    aligned, targets = evaluator.aligned_suffix(logits, ids)
    assert aligned.shape == (2, 256, 3)
    assert np.array_equal(aligned, logits[:, :256])
    assert np.array_equal(targets[0], np.arange(3840, 4096))
    with pytest.raises(ValueError, match="last 257"):
        evaluator.aligned_suffix(logits[:, :-1], ids)
