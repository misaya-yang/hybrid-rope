from __future__ import annotations

import math
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from rebuttal.evq_seed42_retrieval_repair.prepare_data import (
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
    write_bundle_atomic,
    write_manifest,
)
from rebuttal.evq_seed42_retrieval_repair.protocol import (
    decide_gate,
    extract_first_passkey,
    get_stage,
    registered_factor_for_length,
    score_text_answer,
    segment_contract,
)
from rebuttal.evq_seed42_retrieval_repair import train as repair_train
from rebuttal.evq_seed42_retrieval_repair.train import (
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
            if 1000 <= token_ids < 1030:
                index = token_ids - 1000
                return f" word{chr(97 + index // 26)}{chr(97 + index % 26)}"
            return chr(token_ids)
        return "".join(self.decode(int(token_id)) for token_id in token_ids)


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

    assert torch.equal(regions["validation"], validation[:4].reshape(-1))
    assert torch.equal(regions["test"], validation[4:].reshape(-1))
    assert regions["validation"].untyped_storage().data_ptr() == validation.untyped_storage().data_ptr()
    assert regions["test"].untyped_storage().data_ptr() == validation.untyped_storage().data_ptr()


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
        removal_fill_id=ord("z"),
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
    assert torch.all(removed.rendered.input_ids[source_slice] == ord("z"))


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
    assert plan["passkey_32768.pt"]["rows"] == 25
    assert plan["passkey_32768.pt"]["length_semantics"] == "prompt_tokens_before_generation"


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
        "checkpoint_adapter_sha256": "b" * 64,
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
