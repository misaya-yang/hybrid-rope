from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090 import (
    prepare_identifiable_data as data,
)


class FakeTokenizer:
    bos_token = "<BOS>"
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 3

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        words = text.replace("\n", " \n ").split()
        return [
            10 + sum((index + 1) * ord(char) for index, char in enumerate(word)) % 700
            for word in words
        ]


def source_tokens(rows: int) -> np.ndarray:
    return (
        np.arange(rows * data.SOURCE_LENGTH, dtype=np.uint32).reshape(
            rows, data.SOURCE_LENGTH
        )
        + 1_000
    )


def source_receipt_rows(train: int, validation: int) -> list[dict[str, object]]:
    return [
        {
            "row": index,
            "split": "train" if index < train else "validation",
            "document_id": f"document-{index:04d}",
        }
        for index in range(train + validation)
    ]


def test_pair_is_identifiable_and_compact_long_positions_match() -> None:
    tokens = source_tokens(4)
    tokenizer = FakeTokenizer()
    semantic = data.build_semantic_group(
        source_tokens=tokens,
        source_group_rows=[0, 1, 2, 3],
        tokenizer=tokenizer,
        contract=data.SPLITS["train"],
        seed=91,
        row_index=0,
    )
    rendered = {
        length: data.render_semantic_group(
            semantic=semantic,
            source_tokens=tokens,
            tokenizer=tokenizer,
            contract=data.SPLITS["train"],
            length=length,
        )
        for length in data.PAIR_LENGTHS
    }
    compact = rendered[4_096]
    assert compact["query_anchor_spans"].shape == (16, 2)
    assert compact["target_token_ids"].shape == (2, 129)
    assert np.array_equal(
        compact["answer_origin_indices"][0], np.arange(16)
    )
    assert np.all(
        compact["answer_origin_indices"][1] != np.arange(16)
    )

    for length, view in rendered.items():
        assert view["input_ids"].shape == (2, length)
        assert np.array_equal(
            view["source_answer_spans"], compact["source_answer_spans"]
        )
        shift = length - 4_096
        assert np.array_equal(
            view["target_positions"], compact["target_positions"] + shift
        )
        assert np.array_equal(
            view["query_anchor_spans"],
            compact["query_anchor_spans"] + shift,
        )
        extension_start, extension_stop = view["extension_span"]
        assert int(extension_stop - extension_start) == shift
        query_start = int(view["query_span"][0])
        for variant in range(2):
            for query_slot, owner in enumerate(view["queried_indices"]):
                source_start, source_stop = view["source_answer_spans"][owner]
                query_start, query_stop = view["query_anchor_spans"][query_slot]
                output_start, output_stop = view["output_answer_spans"][query_slot]
                anchor = view["input_ids"][
                    variant, source_start - data.ANCHOR_TOKENS : source_start
                ]
                assert data.occurrences(
                    view["input_ids"][variant, :query_start], anchor
                ) == 1
                assert np.array_equal(
                    anchor,
                    view["input_ids"][variant, query_start:query_stop],
                )
                assert np.array_equal(
                    view["input_ids"][variant, source_start:source_stop],
                    view["input_ids"][variant, output_start:output_stop],
                )


def test_dataset_splits_templates_auxiliary_views_and_hashes(tmp_path: Path) -> None:
    tokens = source_tokens(14)
    rows = source_receipt_rows(train=9, validation=5)
    output = tmp_path / "prepared"
    root = data.build_dataset(
        source_tokens=tokens,
        source_rows=rows,
        tokenizer=FakeTokenizer(),
        tokenizer_info={"composite_sha256": "tokenizer"},
        source_info={"tensor_array_sha256_uint32": data.sha256_array(tokens)},
        output=output,
        train_rows=1,
        train_eos_rows=1,
        component_gate_rows=1,
        final_validation_rows=1,
        warmup_rows=1,
        retention_rows=1,
        seed=2026,
    )
    assert root["status"] == data.ROOT_STATUS
    assert root["all_source_roles_disjoint"] is True
    assert root["train_eos_rows"] == 1
    assert root["train_eos_reuses_train_semantics"] is True
    assert root["train_eos_adds_source_rows"] is False
    assert "train_eos" not in root["source_role_rows"]
    assert root["final_validation_for_method_selection"] is False
    role_sets = [set(values) for values in root["source_role_rows"].values()]
    assert all(
        not first & second
        for index, first in enumerate(role_sets)
        for second in role_sets[index + 1 :]
    )
    assert {
        name: contract["query_count"]
        for name, contract in root["split_contracts"].items()
    } == {
        "train": 16,
        "train_eos": 1,
        "component_gate": 1,
        "final_validation": 1,
    }
    assert len(
        {
            contract["template_id"]
            for contract in root["split_contracts"].values()
        }
    ) == 4
    assert len(root["pair_views"]) == 12
    assert root["pair_views"]["train_eos4k"][
        "reuses_semantic_source_split"
    ] == "train"
    assert root["split_contracts"]["train_eos"]["semantic_hashes"] == root[
        "split_contracts"
    ]["train"]["semantic_hashes"][:1]

    for split in data.VIEW_CONTRACTS:
        compact_positions = np.load(
            output / f"{split}4k" / "target_positions.npy"
        )
        compact_sources = np.load(
            output / f"{split}4k" / "source_answer_spans.npy"
        )
        for length in (8_192, 16_384):
            view = output / f"{split}{length // 1024}k"
            assert np.array_equal(
                np.load(view / "target_positions.npy"),
                compact_positions + length - 4_096,
            )
            assert np.array_equal(
                np.load(view / "source_answer_spans.npy"),
                compact_sources,
            )
        manifest = json.loads(
            (output / f"{split}4k" / "manifest.json").read_text()
        )
        assert manifest["method_selection_allowed"] is data.VIEW_CONTRACTS[
            split
        ].method_selection_allowed
        assert manifest["strict_autoregressive_capability"] is (
            split != "train"
        )
        if split != "train":
            prompt_stops = np.load(
                output / f"{split}4k" / "generation_prompt_stops.npy"
            )
            targets = np.load(
                output / f"{split}4k" / "target_positions.npy"
            )
            assert np.array_equal(
                targets[0], np.arange(prompt_stops[0], prompt_stops[0] + 9)
            )
        for name, receipt in manifest["files"].items():
            path = output / f"{split}4k" / name
            assert data.sha256_file(path) == receipt["sha256"]
            assert path.stat().st_size == receipt["bytes"]

    warmup_rows = np.load(output / "warmup4k_clm" / "source_rows.npy")
    warmup_ids = np.load(output / "warmup4k_clm" / "input_ids.npy")
    warmup_labels = np.load(output / "warmup4k_clm" / "labels.npy")
    assert np.array_equal(warmup_ids, tokens[warmup_rows])
    assert np.all(warmup_labels[:, 0] == -100)
    assert np.array_equal(warmup_labels[:, 1:], warmup_ids[:, 1:])
    retention_rows = np.load(output / "retention4k_raw" / "source_rows.npy")
    retention_ids = np.load(output / "retention4k_raw" / "input_ids.npy")
    assert np.array_equal(retention_ids, tokens[retention_rows])

    assert [
        data.SPLITS["component_gate"].queried_indices(index)[0]
        for index in range(16)
    ] == list(range(16))
    assert set(
        data.SPLITS["final_validation"].queried_indices(index)[0]
        for index in range(16)
    ) == set(range(16))
    assert set(
        data.TRAIN_EOS.queried_indices(index)[0] for index in range(16)
    ) == set(range(16))


def test_source_receipt_and_no_benchmark_import(tmp_path: Path) -> None:
    tokens = torch.from_numpy(source_tokens(6).astype(np.int64))
    tensor_path = tmp_path / "source.pt"
    torch.save(tokens, tensor_path)
    receipt_path = tmp_path / "source_receipt.json"
    receipt_path.write_text(
        json.dumps({"rows": source_receipt_rows(train=4, validation=2)}),
        encoding="utf-8",
    )
    loaded, rows, receipt = data.load_source_tensor(
        tensor_path, receipt_path, validation_source_rows=2
    )
    assert loaded.dtype == np.uint32
    assert [row["split"] for row in rows] == [
        "train",
        "train",
        "train",
        "train",
        "validation",
        "validation",
    ]
    assert receipt["tensor_array_sha256_uint32"] == data.sha256_array(loaded)

    source_path = Path(data.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    marker = "r" + "uler"
    assert all(marker not in name.lower() for name in imports)
    assert data._source_code_has_benchmark_import(source_path) is False
