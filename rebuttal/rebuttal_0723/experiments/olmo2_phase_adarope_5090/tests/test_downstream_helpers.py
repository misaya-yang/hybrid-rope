from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090 import evaluate_downstream as downstream
from rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090 import evaluate_frozen_2wiki as frozen_qa


def test_strict_eos_boundary_requires_one_terminal_eos() -> None:
    terminal = downstream.strict_eos_boundary([4, 5, 2], 2)
    assert terminal["strict_terminal_eos_boundary"] is True
    assert downstream.strict_eos_boundary([4, 2, 5], 2)["strict_terminal_eos_boundary"] is False
    assert downstream.strict_eos_boundary([4, 2, 2], 2)["eos_count"] == 2


def test_new_adarope_arm_contract_has_no_static_evq_identity() -> None:
    assert downstream.ARMS == (
        "lora_only_null",
        "native_scale",
        "context_stretch_exp_negative",
        "phase_chord",
        "moment_matched_same_sign_control",
    )
    assert "anchored_evq_cosh_tau_2" not in downstream.ARMS
    assert "lora_only_control" not in downstream.ARMS
    assert "matched_exponential" not in downstream.ARMS
    assert downstream.ARM_CONTRACT["lora_only_null"]["target_key"] == "phase_chord_olmo_r0_lambda_0p1"
    assert downstream.ARM_CONTRACT["native_scale"]["target_key"] == "phase_chord_olmo_r0_lambda_0p1"
    assert downstream.ARM_CONTRACT["context_stretch_exp_negative"]["target_key"] == "matched_exponential_control"
    assert downstream.ARM_CONTRACT["phase_chord"]["target_key"] == "phase_chord_olmo_r0_lambda_0p1"
    assert downstream.ARM_CONTRACT["moment_matched_same_sign_control"]["target_key"] == "moment_matched_same_sign_control"
    assert downstream.ARM_CONTRACT["moment_matched_same_sign_control"]["target_name"] == "moment_matched_control"
    assert downstream.ARM_CONTRACT["context_stretch_exp_negative"]["receipt_arm"] == "context_stretch_exp_negative"
    assert downstream.ARM_CONTRACT["phase_chord"]["receipt_arm"] == "phase_chord"
    assert downstream.ARM_CONTRACT["moment_matched_same_sign_control"]["receipt_arm"] == "moment_matched_same_sign_control"


def test_control_and_scale_bind_existing_phase_manifest_key(tmp_path: Path) -> None:
    table = [float(value) for value in torch.logspace(0, -3, 64).tolist()]
    manifest = tmp_path / "targets.json"
    manifest.write_text(json.dumps({"candidates": {
        "phase_chord_olmo_r0_lambda_0p1": {"inv_freq": table},
        "matched_exponential_control": {"inv_freq": table},
        "moment_matched_same_sign_control": {"inv_freq": table},
    }}), encoding="utf-8")
    for arm in ("lora_only_null", "native_scale", "phase_chord"):
        result = downstream.validate_target_manifest(manifest, arm)
        assert result["target_key"] == "phase_chord_olmo_r0_lambda_0p1"
    result = downstream.validate_target_manifest(manifest, "context_stretch_exp_negative")
    assert result["target_key"] == "matched_exponential_control"
    result = downstream.validate_target_manifest(manifest, "moment_matched_same_sign_control")
    assert result["target_key"] == "moment_matched_same_sign_control"


def test_generation_cache_ignores_window_label_but_not_generation_identity() -> None:
    base = {
        "input_ids": [1, 2, 3],
        "max_new_tokens": 4,
        "references": ["answer"],
        "metric": "string_match_all",
        "nominal_length": 4096,
    }
    alias = {**base, "nominal_length": 16384}
    same_budget_alias = {**base, "input_ids": [1, 2, 3]}
    changed = {**alias, "references": ["different"]}
    assert downstream.generation_cache_key(base) == downstream.generation_cache_key(same_budget_alias)
    assert downstream.generation_cache_key(base) != downstream.generation_cache_key(alias)
    assert downstream.generation_cache_key(base) != downstream.generation_cache_key(changed)


def test_natural_gate_matches_new_trainer_complete_and_final_split(tmp_path: Path) -> None:
    data_root = tmp_path / "final"
    data_root.mkdir()
    manifest = data_root / "manifest.json"
    manifest.write_text(json.dumps({"status": "DATA", "split": "final_validation", "method_selection_allowed": False, "length": 4096}), encoding="utf-8")
    target = {"path": str(tmp_path / "targets.json"), "target_key": "phase_chord_olmo_r0_lambda_0p1", "manifest_sha256": "target-sha"}
    (tmp_path / "targets.json").write_text("{}", encoding="utf-8")
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"target_key": target["target_key"], "target_manifest": target["path"], "target_manifest_sha256": target["manifest_sha256"]}), encoding="utf-8")
    gate = tmp_path / "natural.json"
    gate.write_text(json.dumps({
        "status": "COMPLETE",
        "data_manifest_sha256": downstream.sha256_file(manifest),
        "examples": 1,
        "exact_answer_terminal_eos": 1.0,
        "source_follow_positive_fraction": 1.0,
        "terminal_eos_rate": 1.0,
        "parent_receipt_sha256": downstream.sha256_file(parent),
    }), encoding="utf-8")
    result = downstream.validate_natural_final_gate(gate, natural_data_root=data_root, parent_receipt=parent, target=target, bundle_receipt=parent)
    assert result["data_manifest_sha256"] == downstream.sha256_file(manifest)
    with pytest.raises(RuntimeError, match="evaluated bundle receipt"):
        downstream.validate_natural_final_gate(gate, natural_data_root=data_root, parent_receipt=parent, target=target, bundle_receipt=tmp_path / "other-bundle" / "receipt.json")


def test_natural_gate_requires_all_three_lengths_without_threshold_selection(tmp_path: Path) -> None:
    target = {"path": str(tmp_path / "targets.json"), "target_key": "phase_chord_olmo_r0_lambda_0p1", "manifest_sha256": "target-sha"}
    (tmp_path / "targets.json").write_text("{}", encoding="utf-8")
    parent = tmp_path / "bundle-receipt.json"
    parent.write_text(json.dumps({"target_key": target["target_key"], "target_manifest": target["path"], "target_manifest_sha256": target["manifest_sha256"]}), encoding="utf-8")
    gates: list[Path] = []
    roots: list[Path] = []
    for length in (4096, 8192, 16384):
        root = tmp_path / str(length)
        root.mkdir()
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"split": "final_validation", "method_selection_allowed": False, "length": length}), encoding="utf-8")
        gate = root / "natural.json"
        gate.write_text(json.dumps({"status": "COMPLETE", "data_manifest_sha256": downstream.sha256_file(manifest), "examples": 1, "parent_receipt_sha256": downstream.sha256_file(parent), "exact_answer_terminal_eos": 0.0}), encoding="utf-8")
        roots.append(root)
        gates.append(gate)
    result = downstream.validate_natural_final_gates(gates, roots, parent_receipt=parent, target=target, bundle_receipt=parent)
    assert result["lengths"] == [4096, 8192, 16384]
    assert result["threshold_selection"] is False


def test_atomic_jsonl_resume_rejects_identity_drift(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    expected = [{"row_id": "a", "row_sha256": "a"}, {"row_id": "b", "row_sha256": "b"}]
    downstream.atomic_append_jsonl(path, {**expected[0], "ordinal": 0, "score": 1.0})
    assert downstream.load_resumable_rows(path, expected)["a"]["score"] == 1.0
    with pytest.raises(RuntimeError, match="duplicate/missing resumable row id"):
        downstream.atomic_append_jsonl(path, {"row_id": "a", "ordinal": 0, "row_sha256": "changed"})
        downstream.load_resumable_rows(path, expected)


def test_official_2wiki_loader_requires_exact_200_rows(tmp_path: Path) -> None:
    path = tmp_path / "official.zip"
    rows = [{"input": f"q{i}", "answers": [f"a{i}"], "context": "ctx"} for i in range(200)]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data/2wikimqa.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
    package = downstream.load_official_2wiki_rows(path)
    assert package["dataset"] == "THUDM/LongBench:2wikimqa"
    assert package["member"] == "data/2wikimqa.jsonl"
    assert len(package["rows"]) == 200


@pytest.mark.parametrize("module", [downstream, frozen_qa])
def test_chat_prompt_budget_includes_template_overhead(module: object) -> None:
    class Encoding:
        def __init__(self, size: int) -> None:
            self.input_ids = list(range(size))

    class Tokenizer:
        def __call__(self, prompt: str, *, add_special_tokens: bool) -> Encoding:
            assert add_special_tokens is False
            return Encoding(len(prompt))

        def apply_chat_template(self, messages: list[dict[str, str]], **_: object) -> torch.Tensor:
            return torch.arange(len(messages[0]["content"]) + 10).unsqueeze(0)

        def decode(self, ids: list[int], **_: object) -> str:
            return "x" * len(ids)

    ids, truncated = module._fit_chat_prompt(  # type: ignore[attr-defined]
        Tokenizer(), "x" * 100, length=110, max_new_tokens=5,
    )
    assert truncated is True
    assert len(ids) + 5 == 110


def test_2wiki_scores_match_longbench_normalization() -> None:
    assert downstream.token_f1("The Eiffel Tower", ["Eiffel Tower"]) == 1.0
    assert downstream.normalized_exact("an Eiffel Tower", ["Eiffel Tower"]) == 1.0


class _Output:
    def __init__(self, logits: torch.Tensor, past: object) -> None:
        self.logits = logits
        self.past_key_values = past


class _KeepOneModel:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.budgets: list[int] = []
        self.step = 0

    def __call__(self, *, input_ids: torch.Tensor, logits_to_keep: int, phase_context_budget: int, **_: object) -> _Output:
        self.calls.append(int(logits_to_keep))
        self.budgets.append(int(phase_context_budget))
        vocab = 5
        logits = torch.full((1, 1, vocab), -10.0)
        logits[0, 0, 1 if self.step == 0 else 2] = 10.0
        self.step += 1
        return _Output(logits, object())


def test_prefill_decode_passes_logits_to_keep_one_on_prefill_and_decode() -> None:
    model = _KeepOneModel()
    output = downstream.prefill_decode_greedy(model, torch.tensor([[4, 3]]), max_new_tokens=2, eos_token_id=None)
    assert output.tolist() == [[1, 2]]
    assert model.calls == [1, 1, 1]
    assert model.budgets == [4, 4, 4]


def test_prefill_decode_fails_closed_when_logits_to_keep_is_unsupported() -> None:
    class NoKeep:
        def __call__(self, **kwargs: object) -> object:
            if "logits_to_keep" in kwargs:
                raise TypeError("unexpected keyword")
            raise AssertionError("must not retry without logits_to_keep")

    with pytest.raises(RuntimeError, match="logits_to_keep=1"):
        downstream.prefill_decode_greedy(NoKeep(), torch.tensor([[1]]), max_new_tokens=1, eos_token_id=None)


def test_aggregate_cells_is_unweighted_over_family_length_cells() -> None:
    result = downstream.aggregate_cells([
        {"family": "a", "nominal_length": 4096, "score": 1.0},
        {"family": "a", "nominal_length": 4096, "score": 0.0},
        {"family": "b", "nominal_length": 8192, "score": 0.5},
    ])
    assert result["cells"] == {"a": {"4096": 0.5}, "b": {"8192": 0.5}}
    assert result["macro"] == 0.5
