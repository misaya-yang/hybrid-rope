from __future__ import annotations

import argparse
import hashlib
import math
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from rebuttal.pre_rebuttal.evq_seed42_retrieval_repair.prepare_data import (
    _key_text,
    artifact_plan,
    build_counterfactual_group,
    build_exact_retrieval_example,
    build_passkey_prompt,
    build_messages,
    partition_filler,
    partition_nonce_pools,
    render_complete_chat,
    stack_bundle,
    task_schedule,
    validate_bundle,
    validate_prepared_dir,
    validate_tokenizer_identity,
    write_bundle_atomic,
    write_manifest,
)
from rebuttal.pre_rebuttal.evq_seed42_retrieval_repair.protocol import (
    decide_gate,
    evaluation_budget,
    extract_first_passkey,
    get_stage,
    registered_factor_for_length,
    score_text_answer,
    segment_contract,
)
from rebuttal.pre_rebuttal.evq_seed42_retrieval_repair.evaluate import (
    build_gate_report,
    evaluator_code_sha256,
    merge_temporal_guardrail,
    parse_registered_factor,
    run_gate,
    score_generated_tokens,
    summarize_passkey_records,
    summarize_repair_records,
    summarize_temporal_domains,
    validate_gate_file,
    validate_result_file,
)
from rebuttal.pre_rebuttal.evq_seed42_retrieval_repair import train as repair_train
from rebuttal.pre_rebuttal.evq_seed42_retrieval_repair.train import (
    apply_runtime_frequency,
    runtime_frequency_contract,
    validate_gate_transition,
    validate_parent_adapter,
)
from experiments.lora_evq_v2.legacy_lora_protocol import (
    PAPER_LONGALPACA_PROVENANCE_STATUS,
    PAPER_LONGALPACA_RAW_SHA256,
    PAPER_LONGALPACA_REVISION,
    PAPER_LONGALPACA_SOURCE,
    sha256_file,
)
from experiments.lora_evq_v2.train_evq_lora import build_training_inv_freq


def _passing_summary(stage: str) -> dict[str, object]:
    return {
        "pair_consistency": 0.80 if stage == "r8" else 0.50,
        "source_removal_positive_fraction": 0.75,
        "passkey_containment": 0.50,
        "temporal_delta_nll": 0.20,
        "finite": True,
        "task_types": ["kv", "update"],
    }


def test_stage_contract_has_fixed_physical_token_budget() -> None:
    r8 = get_stage("r8")
    r16 = get_stage("r16")

    assert (r8.seq_len, r8.accumulation, r8.factor) == (8192, 4, 1.0)
    assert (r16.seq_len, r16.accumulation, r16.factor) == (16384, 2, 2.0)
    assert r8.tokens_per_segment == r16.tokens_per_segment == 1_048_576
    assert segment_contract("r8", 2)["optimizer_state"] == "fresh"
    assert segment_contract("r8", 1)["row_range"] == [0, 128]
    assert segment_contract("r8", 2)["row_range"] == [128, 256]


def test_stage_contract_rejects_unknown_stage_and_segment() -> None:
    with pytest.raises(ValueError, match="expected r8 or r16"):
        get_stage("r32")
    with pytest.raises(ValueError, match="segment must be 1 or 2"):
        segment_contract("r8", 3)


def test_registered_factor_is_tied_to_context_length() -> None:
    assert registered_factor_for_length(8192) == 1.0
    assert registered_factor_for_length(16384) == 2.0
    assert registered_factor_for_length(32768) == 4.0
    with pytest.raises(ValueError, match="registered context length"):
        registered_factor_for_length(4096)


def test_text_metrics_separate_strict_extracted_containment_and_eos() -> None:
    score = score_text_answer("The key is 12345678. Extra.", "12345678", False)

    assert score == {
        "strict_exact": False,
        "first_value_exact": True,
        "gold_containment": True,
        "extracted_value": "12345678",
        "eos_terminated": False,
    }


def test_passkey_extraction_does_not_match_inside_longer_number() -> None:
    assert extract_first_passkey("x 12345678 y 87654321") == "12345678"
    assert extract_first_passkey("9123456780") is None
    score = score_text_answer("9123456780", "12345678", True)
    assert score["gold_containment"] is False
    assert score["first_value_exact"] is False


def test_r8_gate_passes_only_when_all_registered_checks_pass() -> None:
    summary = _passing_summary("r8")
    gate = decide_gate("r8", summary, {"pair_consistency": 0.0}, segment=1)

    assert gate["status"] == "pass"
    assert gate["failed_checks"] == []

    summary["passkey_containment"] = 0.49
    failed = decide_gate("r8", summary, {"pair_consistency": 0.0}, segment=1)
    assert failed["status"] == "rescue_allowed"
    assert "passkey_containment" in failed["failed_checks"]

    no_gain = decide_gate("r8", summary, {"pair_consistency": 0.75}, segment=1)
    assert no_gain["status"] == "stop"


def test_rescue_requires_ten_point_gain_and_is_only_available_after_segment_one() -> None:
    summary = _passing_summary("r16")
    summary.update(
        pair_consistency=0.20,
        source_removal_positive_fraction=0.50,
        passkey_containment=0.20,
        temporal_delta_nll=0.10,
    )
    parent = {"pair_consistency": 0.10}

    assert decide_gate("r16", summary, parent, segment=1)["status"] == "rescue_allowed"
    assert decide_gate("r16", summary, parent, segment=2)["status"] == "stop"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temporal_delta_nll", 0.2000001),
        ("finite", False),
        ("task_types", ["kv"]),
        ("pair_consistency", math.nan),
    ],
)
def test_gate_fails_closed_on_guardrail_or_finite_errors(field: str, value: object) -> None:
    summary = _passing_summary("r8")
    summary[field] = value

    gate = decide_gate("r8", summary, {"pair_consistency": 0.0}, segment=1)

    assert gate["status"] == "stop"
    assert field in gate["failed_checks"] or "finite" in gate["failed_checks"]


class CharacterChatTokenizer:
    """Small tokenizer whose direct and rendered chat paths are measurable."""

    eos_token_id = ord("§")
    vocab_size = 4096
    all_special_ids = [0]

    @staticmethod
    def _render(messages: Sequence[dict[str, str]], add_generation_prompt: bool) -> str:
        rendered = ""
        for message in messages:
            if message["role"] == "assistant":
                rendered += f"<assistant>\n{message['content']}§"
            else:
                rendered += f"<{message['role']}>\n{message['content']}\n"
        if add_generation_prompt:
            rendered += "<assistant>\n"
        return rendered

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        rendered = self._render(messages, add_generation_prompt)
        return self(rendered, add_special_tokens=False)["input_ids"] if tokenize else rendered

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> dict[str, object]:
        assert add_special_tokens is False
        result: dict[str, object] = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def decode(self, token_ids, **_: object) -> str:
        if isinstance(token_ids, int):
            if 900 <= token_ids < 930:
                index = token_ids - 900
                return f"word{chr(97 + index // 26)}{chr(97 + index % 26)}"
            if 1000 <= token_ids < 1030:
                index = token_ids - 1000
                return f" word{chr(97 + index // 26)}{chr(97 + index % 26)}"
            return chr(token_ids)
        return "".join(self.decode(int(token_id)) for token_id in token_ids)


class TrimmingAssistantTokenizer(CharacterChatTokenizer):
    """Mirror templates such as Llama 3 that trim message content."""

    @staticmethod
    def _render(messages: Sequence[dict[str, str]], add_generation_prompt: bool) -> str:
        rendered = ""
        for message in messages:
            content = message["content"].strip()
            if message["role"] == "assistant":
                rendered += f"<assistant>\n{content}§"
            else:
                rendered += f"<{message['role']}>\n{content}\n"
        if add_generation_prompt:
            rendered += "<assistant>\n"
        return rendered


def test_nonce_pools_are_disjoint_across_split_and_role() -> None:
    pools = partition_nonce_pools(CharacterChatTokenizer(), minimum=4)

    flattened = {
        (split, role): set(values)
        for split, roles in pools.items()
        for role, values in roles.items()
    }
    assert set(flattened) == {
        (split, role)
        for split in ("train", "validation", "test")
        for role in ("key", "value")
    }
    assert all(len(values) == 4 for values in flattened.values())
    items = list(flattened.items())
    for index, (_, left) in enumerate(items):
        for _, right in items[index + 1 :]:
            assert left.isdisjoint(right)


def test_nonce_pools_require_explicit_word_boundaries() -> None:
    tokenizer = CharacterChatTokenizer()
    pools = partition_nonce_pools(tokenizer, minimum=4)

    assert all(
        tokenizer.decode(token_id).startswith(" ")
        for roles in pools.values()
        for token_ids in roles.values()
        for token_id in token_ids
    )


def test_key_text_strips_pool_boundaries_before_validation() -> None:
    tokenizer = CharacterChatTokenizer()

    assert _key_text(tokenizer, [1000, 1001, 1002]) == "wordaa-wordab-wordac"


def test_task_schedule_is_exactly_seventy_five_twenty_five() -> None:
    schedule = task_schedule(16, seed=42)

    assert schedule.count("kv") == 12
    assert schedule.count("update") == 4
    assert task_schedule(16, seed=42) == schedule
    with pytest.raises(ValueError, match="divisible by four"):
        task_schedule(15, seed=42)


def test_validation_filler_is_split_into_non_overlapping_row_regions() -> None:
    validation = torch.arange(8 * 16, dtype=torch.int32).reshape(8, 16)
    regions = partition_filler(validation)

    assert torch.equal(regions["validation"], validation[:2].reshape(-1))
    assert torch.equal(regions["test"], validation[2:4].reshape(-1))
    assert torch.equal(regions["passkey_validation"], validation[4:6].reshape(-1))
    assert torch.equal(regions["passkey_final"], validation[6:].reshape(-1))
    assert all(
        region.untyped_storage().data_ptr() == validation.untyped_storage().data_ptr()
        for region in regions.values()
    )


def test_complete_chat_render_has_direct_template_parity_and_answer_span() -> None:
    tokenizer = CharacterChatTokenizer()
    messages, markers = build_messages(
        "validation",
        before_text="alpha",
        after_text="beta",
        key_text="keyone",
        value_text=" valuetwelve",
    )

    rendered = render_complete_chat(tokenizer, messages, markers)
    direct = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    assert rendered.input_ids.tolist() == direct
    assert rendered.answer_start < rendered.answer_end
    assert tokenizer.decode(rendered.input_ids[rendered.answer_start : rendered.answer_value_end]) == " valuetwelve"
    assert rendered.input_ids[rendered.answer_end - 1].item() == tokenizer.eos_token_id
    assert rendered.source_value_start < rendered.answer_start


def test_complete_chat_render_uses_generation_boundary_when_template_trims_answer() -> None:
    tokenizer = TrimmingAssistantTokenizer()
    messages, markers = build_messages(
        "validation",
        before_text="alpha",
        after_text="beta",
        key_text="keyone",
        value_text=" valuetwelve",
    )

    rendered = render_complete_chat(tokenizer, messages, markers)

    assert tokenizer.decode(
        rendered.input_ids[rendered.answer_start : rendered.answer_value_end]
    ) == "valuetwelve"
    assert rendered.answer_start == len(
        tokenizer.apply_chat_template(
            messages[:1], tokenize=True, add_generation_prompt=True
        )
    )


def test_split_wording_is_distinct_and_test_is_not_eval_alias() -> None:
    contents = []
    for split in ("train", "validation", "test"):
        messages, _ = build_messages(
            split,
            before_text="left",
            after_text="right",
            key_text="key",
            value_text=" value",
        )
        contents.append(messages[0]["content"])
    assert len(set(contents)) == 3


def test_exact_retrieval_example_measures_final_full_chat_tokens() -> None:
    tokenizer = CharacterChatTokenizer()
    example = build_exact_retrieval_example(
        tokenizer,
        split="validation",
        before_filler_ids=[ord("a")] * 512,
        after_filler_ids=[ord("b")] * 512,
        key_text="keyone",
        value_text=" valueanswer",
        seq_len=256,
        target_distance=80,
        task_type="kv",
    )

    assert example.rendered.input_ids.numel() == 256
    assert example.rendered.distance == 80
    assert example.task_type == "kv"
    assert example.split == "validation"
    direct = tokenizer.apply_chat_template(
        example.messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    assert example.rendered.input_ids.tolist() == direct


def test_counterfactual_group_preserves_positions_and_changes_only_contract_spans() -> None:
    tokenizer = CharacterChatTokenizer()
    example = build_exact_retrieval_example(
        tokenizer,
        split="test",
        before_filler_ids=[ord("c")] * 512,
        after_filler_ids=[ord("d")] * 512,
        key_text="keytwo",
        value_text=" valueanswer",
        seq_len=256,
        target_distance=80,
        task_type="kv",
    )
    group = build_counterfactual_group(
        tokenizer,
        example,
        swapped_value_text=" otheranswer",
        removal_fill_ids=[ord("z") + index % 3 for index in range(
            example.rendered.source_end - example.rendered.source_start
        )],
        group_id="g0",
    )

    assert [row.variant for row in group] == ["original", "swapped", "source_removed"]
    assert all(row.group_id == "g0" for row in group)
    assert all(row.rendered.input_ids.numel() == 256 for row in group)
    original, swapped, removed = group
    assert original.rendered.distance == swapped.rendered.distance == removed.rendered.distance
    source_slice = slice(original.rendered.source_start, original.rendered.source_end)
    answer_slice = slice(original.rendered.answer_start, original.rendered.answer_value_end)
    keep = torch.ones(256, dtype=torch.bool)
    keep[source_slice] = False
    keep[answer_slice] = False
    assert torch.equal(original.rendered.input_ids[keep], swapped.rendered.input_ids[keep])
    outside_source = torch.ones(256, dtype=torch.bool)
    outside_source[source_slice] = False
    assert torch.equal(
        original.rendered.input_ids[outside_source],
        removed.rendered.input_ids[outside_source],
    )
    replacement = removed.rendered.input_ids[source_slice]
    assert torch.unique(replacement).numel() > 1
    assert not torch.equal(replacement, original.rendered.input_ids[source_slice])
    assert "keytwo" not in tokenizer.decode(replacement)
    assert "valueanswer" not in tokenizer.decode(replacement)


def test_bundle_validation_requires_segment_shard_and_answer_tail() -> None:
    tokenizer = CharacterChatTokenizer()
    example = build_exact_retrieval_example(
        tokenizer,
        split="train",
        before_filler_ids=[ord("e")] * 512,
        after_filler_ids=[ord("f")] * 512,
        key_text="keythree",
        value_text=" valueanswer",
        seq_len=256,
        target_distance=80,
        task_type="kv",
    )
    bundle = stack_bundle(
        [example],
        stage="r8",
        split="train",
        seed=42,
        segment=1,
        expected_seq_len=256,
    )

    assert validate_bundle(
        bundle,
        stage="r8",
        split="train",
        segment=1,
        expected_seq_len=256,
        expected_rows=1,
    ) is bundle
    bundle["metadata"][0]["segment"] = 2
    with pytest.raises(ValueError, match="segment metadata"):
        validate_bundle(
            bundle,
            stage="r8",
            split="train",
            segment=1,
            expected_seq_len=256,
            expected_rows=1,
        )


def test_passkey_target_length_is_prompt_only_and_chat_templated() -> None:
    tokenizer = CharacterChatTokenizer()
    row = build_passkey_prompt(
        tokenizer,
        filler_ids=[ord("g")] * 1024,
        target_length=256,
        depth_percent=50,
        passkey="12345678",
    )

    assert row["prompt_ids"].numel() == 256
    assert row["length_semantics"] == "prompt_tokens_before_generation"
    assert row["answer"] == "12345678"
    direct = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=True,
        add_generation_prompt=True,
    )
    assert row["prompt_ids"].tolist() == direct


def test_prepared_manifest_binds_bundle_hash_and_rejects_tampering(tmp_path) -> None:
    tokenizer = CharacterChatTokenizer()
    example = build_exact_retrieval_example(
        tokenizer,
        split="train",
        before_filler_ids=[ord("h")] * 512,
        after_filler_ids=[ord("i")] * 512,
        key_text="keyfour",
        value_text=" valueanswer",
        seq_len=256,
        target_distance=80,
        task_type="kv",
    )
    bundle = stack_bundle(
        [example],
        stage="r8",
        split="train",
        seed=42,
        segment=1,
        expected_seq_len=256,
    )
    root = tmp_path / "prepared"
    root.mkdir()
    record = write_bundle_atomic(
        bundle,
        root / "train_r8_segment1.pt",
        stage="r8",
        split="train",
        segment=1,
        expected_seq_len=256,
        expected_rows=1,
    )
    write_manifest(
        root,
        files={"train_r8_segment1.pt": record},
        tokenizer_identity={"identifier": "fake"},
        filler_manifest_sha256="a" * 64,
        nonce_pool_sha256="b" * 64,
        seed=42,
    )

    assert validate_prepared_dir(root)["status"] == "prepared_no_results"
    with (root / "train_r8_segment1.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="SHA-256"):
        validate_prepared_dir(root)


def test_artifact_plan_separates_training_segments_and_eval_splits() -> None:
    plan = artifact_plan()

    assert plan["train_r8_segment1.pt"]["rows"] == 128
    assert plan["train_r8_segment2.pt"]["rows"] == 128
    assert plan["train_r16_segment1.pt"]["rows"] == 64
    assert plan["train_r16_segment2.pt"]["rows"] == 64
    assert plan["validation_r8.pt"]["rows"] == 48
    assert plan["test_r8.pt"]["rows"] == 96
    assert plan["passkey_validation_8192.pt"]["evaluation_split"] == "validation"
    assert plan["passkey_validation_16384.pt"]["evaluation_split"] == "validation"
    assert plan["passkey_final_32768.pt"]["rows"] == 25
    assert plan["passkey_final_32768.pt"]["evaluation_split"] == "final_test"
    assert plan["passkey_final_32768.pt"]["length_semantics"] == "prompt_tokens_before_generation"


def test_tokenized_artifacts_are_bound_to_the_exact_tokenizer_files() -> None:
    recorded = {"identifier": "model", "files": {"tokenizer.json": "a" * 64}}

    assert validate_tokenizer_identity(recorded, recorded) == recorded
    with pytest.raises(ValueError, match="tokenizer identity"):
        validate_tokenizer_identity(
            recorded,
            {"identifier": "model", "files": {"tokenizer.json": "b" * 64}},
        )


def test_runtime_factor_is_always_derived_from_canonical_evq() -> None:
    canonical, _ = build_training_inv_freq("evq_cosh", 128, 500000.0, 1.414)

    r8 = runtime_frequency_contract(canonical, stage="r8")
    r16 = runtime_frequency_contract(canonical, stage="r16")

    assert torch.equal(r8["substrate_inv_freq"], canonical.to(torch.float64))
    assert torch.equal(r8["runtime_inv_freq"], canonical.to(torch.float64))
    assert r8["factor"] == 1.0
    assert r8["mscale"] == 1.0
    assert r16["factor"] == 2.0
    assert r16["mscale"] > 1.0
    assert r16["label"] == "YaRN-derived generalization on the EVQ substrate"
    assert r16["operator"]["mode"] == "yarn_derived_virtual_dim"


class _Rotary(torch.nn.Module):
    def __init__(self, inv_freq: torch.Tensor):
        super().__init__()
        self.register_buffer("inv_freq", inv_freq.clone())
        self.register_buffer("original_inv_freq", inv_freq.clone())
        self.attention_scaling = 1.0
        self.max_seq_len_cached = 8192
        self._cos_cached = torch.ones(1)


class _RotaryModel(torch.nn.Module):
    def __init__(self, inv_freq: torch.Tensor):
        super().__init__()
        self.first = _Rotary(inv_freq)
        self.second = _Rotary(inv_freq)


def test_runtime_frequency_application_sets_tensor_mscale_and_clears_cache() -> None:
    canonical, _ = build_training_inv_freq("evq_cosh", 128, 500000.0, 1.414)
    model = _RotaryModel(canonical)
    contract = runtime_frequency_contract(canonical, stage="r16")

    result = apply_runtime_frequency(model, contract)

    assert result["patched_modules"] == 2
    for rotary in (model.first, model.second):
        assert torch.allclose(rotary.inv_freq.double(), contract["runtime_inv_freq"])
        assert torch.allclose(rotary.original_inv_freq.double(), contract["runtime_inv_freq"])
        assert rotary.attention_scaling == pytest.approx(contract["mscale"])
        assert rotary.max_seq_len_cached == 0
        assert rotary._cos_cached is None


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_parent_validation_calls_full_artifact_check_and_longalpaca_receipt(
    tmp_path,
    monkeypatch,
) -> None:
    adapter_dir = tmp_path / "parent"
    adapter_dir.mkdir()
    data_manifest = tmp_path / "longalpaca_manifest.json"
    model_manifest = tmp_path / "model_manifest.json"
    _write_json(
        data_manifest,
        {
            "source": {
                "source_id": PAPER_LONGALPACA_SOURCE,
                "revision": PAPER_LONGALPACA_REVISION,
                "split": "train",
                "filename": "LongAlpaca-12k_raw.json",
                "raw_sha256": PAPER_LONGALPACA_RAW_SHA256,
                "provenance_status": PAPER_LONGALPACA_PROVENANCE_STATUS,
            }
        },
    )
    _write_json(model_manifest, {"model": "fake", "files": []})
    canonical, _ = build_training_inv_freq("evq_cosh", 128, 500000.0, 1.414)
    torch.save(
        {
            "inv_freq": canonical,
            "method": "evq_cosh",
            "head_dim": 128,
            "base": 500000.0,
            "tau": 1.414,
            "midpoint": True,
        },
        adapter_dir / "custom_inv_freq.pt",
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter-fixture")
    for filename in (
        "adapter_config.json",
        "experiment_meta.json",
        "trainer_state.json",
        "run_protocol.json",
    ):
        _write_json(adapter_dir / filename, {"fixture": filename})
    called = {}

    def fake_validate_artifact(path, **kwargs):
        called.update(path=path, kwargs=kwargs)
        return {
            "adapter_sha256": "c" * 64,
            "data_manifest_sha256": sha256_file(data_manifest),
            "model_manifest_sha256": sha256_file(model_manifest),
            "protocol": {"method": "evq_cosh", "seed": 42},
        }

    monkeypatch.setattr(repair_train, "validate_artifact", fake_validate_artifact)

    identity = validate_parent_adapter(
        adapter_dir,
        longalpaca_manifest=data_manifest,
        model_manifest=model_manifest,
    )

    assert called["path"] == adapter_dir
    assert called["kwargs"] == {
        "expected_method": "evq_cosh",
        "expected_seed": 42,
        "expected_data_manifest_sha256": sha256_file(data_manifest),
    }
    assert identity["adapter_sha256"] == "c" * 64
    assert identity["frequency"]["method"] == "evq_cosh"


def test_parent_validation_rejects_longalign_receipt_before_training(tmp_path, monkeypatch) -> None:
    adapter_dir = tmp_path / "parent"
    adapter_dir.mkdir()
    data_manifest = tmp_path / "manifest.json"
    model_manifest = tmp_path / "model.json"
    _write_json(data_manifest, {"source": {"source_id": "zai-org/LongAlign-10k"}})
    _write_json(model_manifest, {"files": []})
    monkeypatch.setattr(repair_train, "validate_artifact", lambda *args, **kwargs: {})

    with pytest.raises((ValueError, KeyError), match="source|receipt"):
        validate_parent_adapter(
            adapter_dir,
            longalpaca_manifest=data_manifest,
            model_manifest=model_manifest,
        )


def test_gate_transition_binds_every_registered_identity(tmp_path) -> None:
    bindings = {
        "stage": "r8",
        "segment": 1,
        "factor": 1.0,
        "parent_adapter_sha256": "a" * 64,
        "parent_adapter_receipt_sha256": "8" * 64,
        "checkpoint_adapter_sha256": "b" * 64,
        "checkpoint_adapter_receipt_sha256": "9" * 64,
        "model_manifest_sha256": "c" * 64,
        "longalpaca_manifest_sha256": "d" * 64,
        "repair_manifest_sha256": "e" * 64,
        "validation_bundle_sha256": "f" * 64,
        "operator_tensor_sha256": "1" * 64,
        "evaluator_code_sha256": "2" * 64,
        "retrieval_result_sha256": "3" * 64,
        "passkey_result_sha256": "4" * 64,
        "parent_temporal_sha256": "5" * 64,
        "checkpoint_temporal_sha256": "6" * 64,
    }
    gate = tmp_path / "gate.json"
    _write_json(
        gate,
        {
            "format_version": 1,
            "purpose": "evq_seed42_retrieval_repair_gate",
            "status": "pass",
            "decision": {"status": "pass"},
            "bindings": bindings,
        },
    )

    assert validate_gate_transition(
        gate,
        allowed_statuses={"pass"},
        expected_bindings=bindings,
    )["status"] == "pass"
    stale = dict(bindings, repair_manifest_sha256="9" * 64)
    with pytest.raises(RuntimeError, match="repair_manifest_sha256"):
        validate_gate_transition(
            gate,
            allowed_statuses={"pass"},
            expected_bindings=stale,
        )

    tampered = json.loads(gate.read_text(encoding="utf-8"))
    tampered["status"] = "rescue_allowed"
    _write_json(gate, tampered)
    with pytest.raises(RuntimeError, match="decision status"):
        validate_gate_transition(
            gate,
            allowed_statuses={"rescue_allowed"},
            expected_bindings=bindings,
        )


def test_evaluation_budget_registers_every_paid_forward_before_gpu_use() -> None:
    r8 = evaluation_budget("r8")
    r16 = evaluation_budget("r16")

    assert r8 == {
        "stage": "r8",
        "seq_len": 8192,
        "controlled_rows": 48,
        "controlled_generation_rows": 32,
        "passkey_rows": 25,
        "temporal_domains": 3,
        "temporal_packs_per_domain": 1,
        "input_tokens_per_checkpoint_eval": 133 * 8192,
        "max_generated_tokens": 1312,
    }
    assert r16["input_tokens_per_checkpoint_eval"] == 133 * 16384


def test_repair_factor_parser_accepts_one_registered_factor_only() -> None:
    assert parse_registered_factor("1") == 1.0
    assert parse_registered_factor("2") == 2.0
    assert parse_registered_factor("4") == 4.0
    with pytest.raises(ValueError, match="exactly one"):
        parse_registered_factor("2,4")
    with pytest.raises(ValueError, match="registered"):
        parse_registered_factor("8")


class _MetricTokenizer:
    eos_token_id = 2

    def decode(self, token_ids, **_):
        return " ".join(str(int(token_id)) for token_id in token_ids)


def test_generated_token_metrics_separate_prefix_strict_containment_and_eos() -> None:
    score = score_generated_tokens(
        [11, 12, 13, 11, 12, 13],
        [11, 12, 13],
        _MetricTokenizer(),
    )
    assert score["strict_exact"] is False
    assert score["first_value_exact"] is True
    assert score["gold_containment"] is True
    assert score["eos_terminated"] is False
    assert score["generated_token_count"] == 6

    exact = score_generated_tokens([11, 12, 13, 2], [11, 12, 13], _MetricTokenizer())
    assert exact["strict_exact"] is True
    assert exact["eos_terminated"] is True
    assert exact["generated_token_count"] == 3


def _triplet(group: str, task: str, distance: int, *, both: bool, removed: float):
    generated = [11, 12, 2] if both else [99, 2]
    return [
        {
            "group_id": group,
            "variant": "original",
            "task_type": task,
            "distance": distance,
            "mean_nll": 1.0,
            "first_value_exact": both,
            "strict_exact": both,
            "gold_containment": both,
            "eos_terminated": True,
            "generated_token_count": len(generated) - 1,
            "generated_ids": generated,
            "expected_ids": [11, 12],
            "finite": True,
        },
        {
            "group_id": group,
            "variant": "swapped",
            "task_type": task,
            "distance": distance,
            "mean_nll": 1.1,
            "first_value_exact": both,
            "strict_exact": both,
            "gold_containment": both,
            "eos_terminated": True,
            "generated_token_count": len(generated) - 1,
            "generated_ids": generated,
            "expected_ids": [11, 12],
            "finite": True,
        },
        {
            "group_id": group,
            "variant": "source_removed",
            "task_type": task,
            "distance": distance,
            "mean_nll": removed,
            "first_value_exact": None,
            "finite": True,
        },
    ]


def test_controlled_summary_requires_triplets_and_scores_source_dependence() -> None:
    records = _triplet("a", "kv", 3000, both=True, removed=1.5)
    records += _triplet("b", "update", 5000, both=False, removed=0.8)

    summary = summarize_repair_records(records, stage="r8")

    assert summary["groups"] == 2
    assert summary["pair_consistency"] == 0.5
    assert summary["source_removal_positive_fraction"] == 0.5
    assert set(summary["by_task"]) == {"kv", "update"}
    assert summary["finite"] is True
    with pytest.raises(ValueError, match="all three variants"):
        summarize_repair_records(records[:-1], stage="r8")


def _temporal(adapter_sha: str, nll: float, *, selection: str = "a" * 64):
    return {
        "schema": "evq_cosh.seed42_retrieval_repair_temporal.v1",
        "stage": "r8",
        "factor": 1.0,
        "adapter_sha256": adapter_sha,
        "temporal_selection_sha256": selection,
        "summary": {"mean_nll": nll, "finite": True},
    }


def test_temporal_guardrail_requires_same_factor_and_frozen_examples() -> None:
    merged = merge_temporal_guardrail(
        "r8", _temporal("a" * 64, 2.0), _temporal("b" * 64, 2.19)
    )
    assert merged["temporal_delta_nll"] == pytest.approx(0.19)

    wrong = _temporal("b" * 64, 2.1, selection="c" * 64)
    with pytest.raises(ValueError, match="selection"):
        merge_temporal_guardrail("r8", _temporal("a" * 64, 2.0), wrong)


def test_gate_report_binds_all_evidence_and_uses_registered_decision() -> None:
    bindings = {
        "parent_adapter_sha256": "a" * 64,
        "parent_adapter_receipt_sha256": "8" * 64,
        "checkpoint_adapter_sha256": "b" * 64,
        "checkpoint_adapter_receipt_sha256": "9" * 64,
        "model_manifest_sha256": "c" * 64,
        "longalpaca_manifest_sha256": "d" * 64,
        "repair_manifest_sha256": "e" * 64,
        "validation_bundle_sha256": "f" * 64,
        "passkey_bundle_sha256": "0" * 64,
        "operator_tensor_sha256": "1" * 64,
        "evaluator_code_sha256": "2" * 64,
        "parent_retrieval_result_sha256": "3" * 64,
        "retrieval_result_sha256": "4" * 64,
        "passkey_result_sha256": "5" * 64,
        "parent_temporal_sha256": "6" * 64,
        "checkpoint_temporal_sha256": "7" * 64,
    }
    report = build_gate_report(
        stage="r8",
        segment=1,
        repair_summary={
            "pair_consistency": 0.80,
            "source_removal_positive_fraction": 0.75,
            "finite": True,
            "task_types": ["kv", "update"],
        },
        passkey_summary={"passkey_containment": 0.52, "finite": True},
        parent_repair_summary={"pair_consistency": 0.0},
        parent_temporal=_temporal("a" * 64, 2.0),
        checkpoint_temporal=_temporal("b" * 64, 2.2),
        bindings=bindings,
    )

    assert report["status"] == "pass"
    assert report["bindings"]["stage"] == "r8"
    assert report["bindings"]["segment"] == 1
    assert report["bindings"]["factor"] == 1.0
    assert set(bindings).issubset(report["bindings"])


def _passkey_rows() -> list[dict[str, object]]:
    return [
        {
            "strict_exact": index < 13,
            "first_value_exact": index < 13,
            "gold_containment": index < 13,
            "eos_terminated": index < 13,
            "answer": "12345678",
            "prediction": "12345678" if index < 13 else "00000000",
            "extracted_value": "12345678" if index < 13 else "00000000",
            "generated_ids": [2] if index < 13 else [],
            "generated_token_count": 0,
            "nll_sum": float(index + 1),
            "answer_tokens": 2,
            "finite": True,
        }
        for index in range(25)
    ]


def _temporal_domains() -> dict[str, object]:
    return {
        name: {
            "packs": [
                {
                    "pack_index": 0,
                    "nll_sum": nll * 10,
                    "scored_tokens": 10,
                    "mean_nll": nll,
                    "finite": True,
                }
            ],
            "nll_sum": nll * 10,
            "scored_tokens": 10,
            "mean_nll": nll,
            "finite": True,
        }
        for name, nll in (("a", 1.0), ("b", 2.0), ("c", 3.0))
    }


def _gate_result_records(tmp_path: Path) -> dict[str, Path]:
    code_sha = evaluator_code_sha256()
    common = {
        "stage": "r8",
        "factor": 1.0,
        "model_manifest_sha256": "c" * 64,
        "longalpaca_manifest_sha256": "d" * 64,
        "repair_manifest_sha256": "e" * 64,
        "runtime_tensor_sha256": "1" * 64,
        "evaluator_code_sha256": code_sha,
    }
    parent_rows = []
    checkpoint_rows = []
    for index in range(16):
        task = "kv" if index < 12 else "update"
        distance = 3000 if index % 2 == 0 else 5000
        parent_rows += _triplet(
            f"p-{index}", task, distance, both=False, removed=1.5
        )
        checkpoint_rows += _triplet(
            f"c-{index}", task, distance, both=True, removed=1.5
        )
    passkey_rows = _passkey_rows()
    domains = _temporal_domains()
    selection = {
        "stage": "r8",
        "factor": 1.0,
        "domains": {name: {"pack_indices": [0]} for name in domains},
    }
    selection_sha = hashlib.sha256(
        json.dumps(
            selection,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    records = {
        "parent_controlled": {
            "schema": "evq_cosh.seed42_retrieval_repair_controlled.v1",
            **common,
            "adapter_sha256": "a" * 64,
            "adapter_receipt_sha256": "8" * 64,
            "bundle_sha256": "f" * 64,
            "bundle_file": "validation_r8.pt",
            "split": "validation",
            "eos_token_id": 2,
            "results": parent_rows,
            "summary": summarize_repair_records(parent_rows, stage="r8"),
        },
        "checkpoint_controlled": {
            "schema": "evq_cosh.seed42_retrieval_repair_controlled.v1",
            **common,
            "adapter_sha256": "b" * 64,
            "adapter_receipt_sha256": "9" * 64,
            "bundle_sha256": "f" * 64,
            "bundle_file": "validation_r8.pt",
            "split": "validation",
            "eos_token_id": 2,
            "results": checkpoint_rows,
            "summary": summarize_repair_records(checkpoint_rows, stage="r8"),
        },
        "passkey_result": {
            "schema": "evq_cosh.seed42_retrieval_repair_passkey.v1",
            **common,
            "adapter_sha256": "b" * 64,
            "adapter_receipt_sha256": "9" * 64,
            "bundle_sha256": "0" * 64,
            "bundle_file": "passkey_validation_8192.pt",
            "evaluation_split": "validation",
            "target_length": 8192,
            "eos_token_id": 2,
            "results": passkey_rows,
            "summary": summarize_passkey_records(passkey_rows),
        },
    }
    for role, adapter, offset in (
        ("parent_temporal", "a" * 64, 0.0),
        ("checkpoint_temporal", "b" * 64, 0.1),
    ):
        adjusted = json.loads(json.dumps(domains))
        for domain in adjusted.values():
            domain["packs"][0]["nll_sum"] += offset * 10
            domain["packs"][0]["mean_nll"] += offset
            domain["nll_sum"] += offset * 10
            domain["mean_nll"] += offset
        records[role] = {
            "schema": "evq_cosh.seed42_retrieval_repair_temporal.v1",
            **common,
            "adapter_sha256": adapter,
            "adapter_receipt_sha256": "8" * 64 if role == "parent_temporal" else "9" * 64,
            "temporal_selection": selection,
            "temporal_selection_sha256": selection_sha,
            "domains": adjusted,
            "summary": summarize_temporal_domains(adjusted),
        }

    paths = {}
    for role, record in records.items():
        path = tmp_path / f"{role}.json"
        _write_json(path, record)
        paths[role] = path
    return paths


def test_gate_recomputes_raw_evidence_and_rejects_tampering(tmp_path) -> None:
    paths = _gate_result_records(tmp_path)
    gate_path = tmp_path / "gate.json"
    args = argparse.Namespace(stage="r8", segment=1, output=gate_path, **paths)

    report = run_gate(args)

    assert report["status"] == report["decision"]["status"] == "pass"
    assert validate_gate_file(gate_path)["status"] == "pass"

    tampered_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    tampered_gate["status"] = "stop"
    _write_json(gate_path, tampered_gate)
    with pytest.raises((RuntimeError, ValueError), match="status|recompute"):
        validate_gate_file(gate_path)

    gate_path.unlink()
    run_gate(args)
    controlled_path = paths["checkpoint_controlled"]
    tampered_result = json.loads(controlled_path.read_text(encoding="utf-8"))
    tampered_result["summary"]["pair_consistency"] = 0.0
    _write_json(controlled_path, tampered_result)
    with pytest.raises((RuntimeError, ValueError), match="summary|SHA-256"):
        validate_gate_file(gate_path)


def test_result_validation_recomputes_summary_from_raw_rows(tmp_path) -> None:
    paths = _gate_result_records(tmp_path)
    result = paths["passkey_result"]

    assert validate_result_file(result, kind="passkey")["summary"]["rows"] == 25
    record = json.loads(result.read_text(encoding="utf-8"))
    record["summary"]["passkey_containment"] = 1.0
    _write_json(result, record)
    with pytest.raises(ValueError, match="summary"):
        validate_result_file(result, kind="passkey")


def test_launcher_has_explicit_non_advancing_commands_and_gpu_lock() -> None:
    launcher = Path("rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/run_seed42.sh").read_text()

    for command in (
        "prepare)",
        "preflight)",
        "baseline)",
        "train-r8)",
        "gate-r8)",
        "train-r16)",
        "gate-r16)",
        "final)",
    ):
        assert command in launcher
    assert "flock" in launcher
    assert "nvidia-smi" in launcher
    assert "--dry-run" in launcher


def test_launcher_never_downloads_or_autostarts_the_next_stage() -> None:
    launcher = Path("rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/run_seed42.sh").read_text()

    assert "git clone" not in launcher
    assert "wget " not in launcher
    assert "curl " not in launcher
    train_r8 = launcher.split("train-r8)", 1)[1].split(";;", 1)[0]
    assert "train-r16)" not in train_r8
    gate_r8 = launcher.split("gate-r8)", 1)[1].split(";;", 1)[0]
    assert "train-r16)" not in gate_r8


def test_launcher_requires_external_paths_without_private_defaults() -> None:
    launcher = Path("rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/run_seed42.sh").read_text()

    for variable in (
        "EVQ_REPAIR_MODEL",
        "EVQ_REPAIR_MODEL_MANIFEST",
        "EVQ_REPAIR_LONGALPACA_MANIFEST",
        "EVQ_REPAIR_PARENT_ADAPTER",
        "EVQ_REPAIR_FILLER_DIR",
        "EVQ_REPAIR_TEMPORAL_ROOT",
        "EVQ_REPAIR_CAPABILITY_DIR",
        "EVQ_REPAIR_WORK_DIR",
        "PYTHON_BIN",
    ):
        assert variable in launcher
    private_server_root = "/" + "root/autodl-tmp"
    assert private_server_root not in launcher
    assert "connect.westb" not in launcher


def test_capability_filter_keeps_optional_longbench_rows() -> None:
    from rebuttal.pre_rebuttal.evq_seed42_retrieval_repair.evaluate import CAPABILITY_SUITES

    assert "longbench" in CAPABILITY_SUITES


def test_launcher_revalidates_temporal_reuse_against_source_selection() -> None:
    launcher = Path("rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/run_seed42.sh").read_text()

    assert '--temporal-root "$EVQ_REPAIR_TEMPORAL_ROOT"' in launcher
    assert '--max-packs-per-domain "$max_packs"' in launcher
    final_report = launcher.split(" final-report ", 1)[1]
    assert '--temporal-root "$EVQ_REPAIR_TEMPORAL_ROOT"' in final_report


def test_training_and_evaluation_share_one_global_gpu_lock() -> None:
    training = Path("rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/run_seed42.sh").read_text()
    evaluation = Path("scripts/2026-07/09_lora_evq_official_yarn_eval.sh").read_text()

    assignment = 'GPU_LOCK="${EVQ_GPU_LOCK:-/tmp/evq_lora_eval_gpu.lock}"'
    assert assignment in training
    assert assignment in evaluation


def test_gpu_phases_require_a_content_bound_complete_preflight_receipt() -> None:
    launcher = Path("rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/run_seed42.sh").read_text()

    assert 'PREFLIGHT_RECEIPT="$EVQ_REPAIR_WORK_DIR/preflight_complete.json"' in launcher
    gpu_gate = launcher.split("require_gpu_lock()", 1)[1].split("verify_result()", 1)[0]
    assert 'require_file "$PREFLIGHT_RECEIPT"' in gpu_gate
    assert " preflight-complete " in gpu_gate
    preflight = launcher.split("  preflight)", 1)[1].split("    ;;", 1)[0]
    assert 'rm -f "$PREFLIGHT_RECEIPT"' in preflight
    assert preflight.rstrip().endswith('--output "$PREFLIGHT_RECEIPT"')
