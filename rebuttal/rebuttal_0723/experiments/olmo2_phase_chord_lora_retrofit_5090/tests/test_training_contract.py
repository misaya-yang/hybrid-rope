from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import json

from rebuttal.rebuttal_0723.experiments.olmo2_phase_chord_lora_retrofit_5090 import (
    receipts,
)
from rebuttal.rebuttal_0723.experiments.olmo2_phase_chord_lora_retrofit_5090 import (
    train_static_table_lora as trainer,
)
from rebuttal.rebuttal_0723.experiments.olmo2_phase_chord_lora_retrofit_5090 import (
    evaluate_static_peft_ruler as ruler_eval,
)


def test_smoothstep_morph_is_exact_target_at_step_60() -> None:
    native = torch.tensor([1.0, 0.5, 0.25])
    target = torch.tensor([1.0, 0.4, 0.25])
    first, first_receipt = receipts.smoothstep_log_morph(
        native, target, step=1, morph_steps=60
    )
    middle, middle_receipt = receipts.smoothstep_log_morph(
        native, target, step=30, morph_steps=60
    )
    final, final_receipt = receipts.smoothstep_log_morph(
        native, target, step=60, morph_steps=60
    )
    assert torch.equal(final, target)
    assert final_receipt["is_exact_target"] is True
    assert first_receipt["smoothstep_amount"] < middle_receipt["smoothstep_amount"] < 1.0
    assert torch.equal(first[[0, -1]], native[[0, -1]])
    assert torch.equal(middle[[0, -1]], native[[0, -1]])


def _protocol_args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "arm": "phase_chord_olmo_r0_lambda_0p1",
        "steps": 300,
        "morph_steps": 60,
        "rank": 64,
        "alpha": 128.0,
        "micro_pairs": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-5,
        "warmup_steps": 20,
        "compile_mode": "none",
        "primary_tokens": 8,
        "primary_weight": 1.0,
        "continuation_weight": 0.25,
        "effect_weight": 1.0,
        "margin_weight": 1.0,
        "correct_ce_weight": 0.1,
        "source_margin": 1.0,
        "min_teacher_effect": 0.0,
        "short_replay_every": 1,
        "seed": 20260822,
    }
    values.update(overrides)
    return Namespace(**values)


def test_protocol_locks_matched_science_and_forbids_pure_ce() -> None:
    protocol = trainer.protocol_from_args(_protocol_args())
    assert protocol["steps"] == 300
    assert protocol["morph_steps"] == 60
    assert protocol["lora"]["implementation"] == "standard_peft"
    assert protocol["lora"]["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    assert protocol["pair_batch"] == "physical8k_correct_plus_swapped"
    with pytest.raises(ValueError, match="300 steps"):
        trainer.protocol_from_args(_protocol_args(steps=299))
    with pytest.raises(ValueError, match="rank64 alpha128"):
        trainer.protocol_from_args(_protocol_args(rank=32))
    with pytest.raises(ValueError, match="pure answer CE"):
        trainer.protocol_from_args(
            _protocol_args(effect_weight=0.0, margin_weight=0.0)
        )


class _Rotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.tensor([1.0, 0.5, 0.25]))
        self.original_inv_freq = self.inv_freq


class _Base(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.rotary_emb = _Rotary()


class _PeftLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = _Base()

    def get_base_model(self) -> _Base:
        return self.base


def test_standard_hf_rotary_injection_supports_base_and_peft() -> None:
    value = torch.tensor([1.0, 0.4, 0.25])
    base = _Base()
    trainer._set_inv_freq(base, value)
    assert torch.equal(base.model.rotary_emb.inv_freq, value)
    peft = _PeftLike()
    trainer._set_inv_freq(peft, value)
    assert torch.equal(peft.get_base_model().model.rotary_emb.inv_freq, value)


class _FakeSavedModel:
    def save_pretrained(self, path: Path, safe_serialization: bool) -> None:
        assert safe_serialization is True
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (path / "adapter_model.safetensors").write_bytes(b"fake-adapter")


class _FakeLoaded:
    def __init__(self) -> None:
        self.base_model = type("LoraModel", (), {})()
        self.base_model.model = type("CausalLM", (), {})()
        self.base_model.model.model = type("Backbone", (), {})()
        self.base_model.model.model.rotary_emb = _Rotary()


def test_fake_peft_save_load_roundtrip_preserves_frequency_and_state(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    table = torch.tensor([1.0, 0.4, 0.25])
    save = receipts.save_standard_peft_bundle(
        model=_FakeSavedModel(),
        output=bundle,
        inv_freq=table,
        metadata={"arm": "phase_chord_olmo_r0_lambda_0p1"},
    )
    assert "adapter/adapter_config.json" in save["files"]
    loaded, load = receipts.load_standard_peft_bundle(
        base_model=object(),
        bundle=bundle,
        peft_loader=lambda _base, _path: _FakeLoaded(),
    )
    assert torch.equal(
        loaded.base_model.model.model.rotary_emb.inv_freq, table
    )
    assert len(load["active_inv_freq_float32_sha256"]) == 64
    expected = {"q.lora_A": torch.tensor([1.0]), "q.lora_B": torch.tensor([2.0])}
    observed = {name: value.clone() for name, value in expected.items()}
    receipt = receipts.assert_peft_state_roundtrip(expected, observed)
    assert receipt["bitwise_equal"] is True
    observed["q.lora_B"] += 1
    with pytest.raises(RuntimeError, match="roundtrip drift"):
        receipts.assert_peft_state_roundtrip(expected, observed)


def test_ready_drift_and_dual_authorization_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "ready.json"
    protocol = {"arm": "native", "steps": 300}
    inputs = {"data_sha256": "abc"}
    code = {"trainer.py": "def"}
    receipts.write_ready_receipt(
        path=path,
        protocol=protocol,
        inputs=inputs,
        code_sha256=code,
        run_output=tmp_path / "run",
        cuda_available=False,
        cuda_initialized=False,
    )
    receipts.validate_ready_receipt(
        path=path,
        protocol=protocol,
        inputs=inputs,
        code_sha256=code,
        run_output=tmp_path / "run",
    )
    with pytest.raises(RuntimeError, match="READY receipt drift"):
        receipts.validate_ready_receipt(
            path=path,
            protocol=protocol,
            inputs={"data_sha256": "changed"},
            code_sha256=code,
            run_output=tmp_path / "run",
        )
    with pytest.raises(RuntimeError, match="requires --authorize"):
        receipts.require_gpu_authorization(cli_authorize=True, environment={})
    with pytest.raises(RuntimeError, match="requires --authorize"):
        receipts.require_gpu_authorization(
            cli_authorize=False, environment={receipts.AUTH_ENV: "1"}
        )
    receipts.require_gpu_authorization(
        cli_authorize=True, environment={receipts.AUTH_ENV: "1"}
    )


def test_target_contract_proves_causal_shift() -> None:
    correct = np.asarray([[5, 6, 7, 8]], dtype=np.uint32)
    swapped = np.asarray([[1, 2, 7, 8]], dtype=np.uint32)
    mask = np.asarray([[False, False, True, True]])
    positions, tokens = trainer._target_contract(correct, swapped, mask, mask)
    assert positions.tolist() == [[2, 3]]
    assert tokens.tolist() == [[7, 8]]
    bad = mask.copy()
    bad[0, 0] = True
    with pytest.raises(RuntimeError, match="position zero"):
        trainer._target_contract(correct, swapped, bad, bad)


def test_verified_ready_checkpoint_format_is_strictly_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    weight = checkpoint / "model.safetensors"
    weight.write_bytes(b"small fake weight")
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "olmo2",
                "hidden_size": 2048,
                "num_hidden_layers": 16,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "vocab_size": 100352,
            }
        ),
        encoding="utf-8",
    )
    ready = tmp_path / "ready.json"
    ready.write_text(
        json.dumps(
            {
                "status": "VERIFIED_READY",
                "model_dir": str(checkpoint),
                "canonical_identity": {
                    "historical_model_sha256": receipts.EXPECTED_CHECKPOINT_SHA256
                },
                "repository_files": {
                    "files": [
                        {
                            "path": "model.safetensors",
                            "sha256": receipts.EXPECTED_CHECKPOINT_SHA256,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    real_sha = receipts.sha256_file
    monkeypatch.setattr(
        receipts,
        "sha256_file",
        lambda path: (
            receipts.EXPECTED_CHECKPOINT_SHA256
            if Path(path).name == "model.safetensors"
            else real_sha(path)
        ),
    )
    result = receipts.checkpoint_receipt(checkpoint, ready)
    assert result["weight_sha256"] == receipts.EXPECTED_CHECKPOINT_SHA256
    payload = json.loads(ready.read_text(encoding="utf-8"))
    payload["repository_files"]["files"][0]["sha256"] = "0" * 64
    ready.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="repository weight digest drift"):
        receipts.checkpoint_receipt(checkpoint, ready)


def test_ruler_eos_boundary_and_macro_aggregation_are_explicit() -> None:
    boundary = ruler_eval.strict_eos_boundary([9, 10, 2], eos_token_id=2)
    assert boundary["strict_terminal_eos_boundary"] is True
    repeated = ruler_eval.strict_eos_boundary([9, 2, 2], eos_token_id=2)
    assert repeated["strict_terminal_eos_boundary"] is False
    assert repeated["tokens_after_first_eos"] == 1
    missing = ruler_eval.strict_eos_boundary([9, 10], eos_token_id=2)
    assert missing["eos_observed"] is False

    rows = [
        {
            "task": "niah_single_1",
            "nominal_length": 4096,
            "official_task_score": 1.0,
            "reference_recall": 1.0,
            "generated_tokens": 3,
            "elapsed_seconds": 0.5,
            "eos_boundary": boundary,
        },
        {
            "task": "niah_single_1",
            "nominal_length": 4096,
            "official_task_score": 0.0,
            "reference_recall": 0.0,
            "generated_tokens": 2,
            "elapsed_seconds": 0.7,
            "eos_boundary": missing,
        },
    ]
    aggregate = ruler_eval.aggregate_rows(
        rows, tasks=("niah_single_1",), lengths=(4096,)
    )
    assert aggregate["macro_official_score"] == pytest.approx(0.5)
    assert aggregate["macro_official_score_by_length"]["4096"] == pytest.approx(0.5)
    assert aggregate["cells"]["niah_single_1"]["4096"][
        "strict_terminal_eos_boundary_rate"
    ] == pytest.approx(0.5)
