from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.lora_evq_v2.eval_official_yarn_capability import (
    adapter_artifact_receipt,
    apply_official_yarn_runtime,
    capability_arm_contract,
    load_capability_suite,
    parse_yarn_factors,
    score_capability_prediction,
    select_rows,
    summarize_results,
    validate_adapter_identity_metadata,
    generate_answer,
    score_generation_metrics,
)
from experiments.lora_evq_v2.prepare_seed42_capability_data import _make_record
from scripts.lib.rope.official_yarn import native_endpoint_inv_freq
from scripts.lib.rope.schedules import evq_cosh_inv_freq


class _FakeRotary(torch.nn.Module):
    def __init__(self, inv_freq: torch.Tensor):
        super().__init__()
        self.register_buffer("inv_freq", inv_freq.clone())
        self.attention_scaling = 1.0
        self.max_seq_len_cached = 8192


class _FakeModel(torch.nn.Module):
    def __init__(self, inv_freq: torch.Tensor):
        super().__init__()
        self.rotary_emb = _FakeRotary(inv_freq)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_suite(root: Path) -> list[dict]:
    rows = [
        _make_record(
            example_id="passkey-L16384-d50-t00-s42",
            suite="passkey",
            task="passkey",
            target_length=16384,
            prompt_ids=[1, 2, 3],
            answers=["12345"],
            metric="exact_match",
            depth_percent=50.0,
            source={"generator": "unit-test"},
        ),
        _make_record(
            example_id="arc-L8192-000",
            suite="mcqa",
            task="allenai-ai2-arc",
            target_length=8192,
            prompt_ids=[4, 5, 6],
            answers=["B"],
            choices=["A", "B", "C"],
            answer_index=1,
            metric="mcqa",
            source={"dataset": "unit-test"},
        ),
    ]
    data_path = root / "suite.jsonl"
    data_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema": "evq_cosh.seed42_capability_manifest.v2",
        "example_schema": "evq_cosh.seed42_capability_example.v2",
        "row_count": len(rows),
        "files": {
            data_path.name: {
                "sha256": _sha256(data_path),
                "size_bytes": data_path.stat().st_size,
                "row_count": len(rows),
            }
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return rows


def test_parse_yarn_factors_requires_registered_x2_x4_pair():
    assert parse_yarn_factors("2,4") == (2.0, 4.0)
    with pytest.raises(ValueError, match="2 and 4"):
        parse_yarn_factors("4")


def test_arm_contract_distinguishes_geo_and_evq_substrates():
    geo = capability_arm_contract("native_geo", (2.0, 4.0))
    evq = capability_arm_contract("evq_cosh", (2.0, 4.0))
    assert geo["adapter"] == "geo_longalpaca_s42"
    assert geo["operator"] == "official_yarn"
    assert evq["adapter"] == "evq_longalpaca_tau1414_s42"
    assert evq["operator"] == "yarn_derived_virtual_dim"
    assert evq["label"] == "YaRN-derived generalization on the EVQ substrate"
    assert geo["factors"] == evq["factors"] == [2.0, 4.0]


def test_adapter_identity_metadata_is_seed42_and_hash_bound():
    digest = "a" * 64
    metadata = {
        "objective": "legacy_longalign_full_token_causal_lm_v2",
        "status": "complete",
        "global_step": 300,
        "adapter_sha256": digest,
        "data_manifest_sha256": "b" * 64,
        "protocol": {
            "method": "native_geo",
            "seed": 42,
            "data_manifest_sha256": "b" * 64,
        },
    }
    config = {
        "r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    validated = validate_adapter_identity_metadata(
        metadata,
        config,
        substrate="native_geo",
        training_manifest_sha256="b" * 64,
        adapter_sha256=digest,
    )
    assert validated["protocol"]["seed"] == 42
    with pytest.raises(ValueError, match="method"):
        validate_adapter_identity_metadata(
            metadata,
            config,
            substrate="evq_cosh",
            training_manifest_sha256="b" * 64,
            adapter_sha256=digest,
        )


def test_adapter_receipt_binds_all_runtime_artifacts(tmp_path: Path):
    names = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "experiment_meta.json",
        "custom_inv_freq.pt",
    )
    for name in names:
        (tmp_path / name).write_bytes(name.encode("utf-8"))

    receipt = adapter_artifact_receipt(tmp_path)

    assert set(receipt["files"]) == set(names)
    original = receipt["receipt_sha256"]
    (tmp_path / "custom_inv_freq.pt").write_bytes(b"changed")
    assert adapter_artifact_receipt(tmp_path)["receipt_sha256"] != original


def test_apply_runtime_sets_full_yarn_frequency_and_mscale():
    native = native_endpoint_inv_freq(128, 500000.0)
    model = _FakeModel(native)
    result = apply_official_yarn_runtime(
        model,
        native,
        head_dim=128,
        base=500000.0,
        factor=4.0,
        original_max_position_embeddings=8192,
    )
    assert result["patched_count"] == 1
    assert result["operator"]["mode"] == "official_yarn_native"
    assert torch.allclose(model.rotary_emb.inv_freq, result["inv_freq"])
    assert model.rotary_emb.attention_scaling == pytest.approx(result["mscale"])
    assert model.rotary_emb.max_seq_len_cached == 0


def test_apply_runtime_keeps_evq_as_input_to_official_transform():
    evq = evq_cosh_inv_freq(
        head_dim=128,
        tau=1.414,
        base=500000.0,
        midpoint=True,
        dtype=torch.float64,
    )
    model = _FakeModel(evq)
    result = apply_official_yarn_runtime(
        model,
        evq,
        head_dim=128,
        base=500000.0,
        factor=2.0,
        original_max_position_embeddings=8192,
    )
    assert result["operator"]["mode"] == "yarn_derived_virtual_dim"
    assert not torch.allclose(model.rotary_emb.inv_freq.to(torch.float64), evq)
    assert result["input_inv_freq_sha256"] != result["output_inv_freq_sha256"]


def test_load_suite_checks_manifest_hashes_and_counts(tmp_path: Path):
    expected = _write_suite(tmp_path)
    manifest, rows = load_capability_suite(tmp_path)
    assert manifest["row_count"] == 2
    assert rows == expected
    (tmp_path / "suite.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_capability_suite(tmp_path)


def test_pilot_selects_one_example_per_task_length_depth_cell():
    rows = [
        {"task": "passkey", "target_length": 16384, "depth_percent": 50.0, "example_id": "a"},
        {"task": "passkey", "target_length": 16384, "depth_percent": 50.0, "example_id": "b"},
        {"task": "passkey", "target_length": 16384, "depth_percent": 90.0, "example_id": "c"},
    ]
    assert [row["example_id"] for row in select_rows(rows, mode="pilot")] == ["a", "c"]
    assert select_rows(rows, mode="full") == rows


def test_summary_keeps_factor_task_length_and_depth_separate():
    rows = [
        {
            "factor": 2.0,
            "task": "passkey",
            "target_length": 16384,
            "depth_percent": 50.0,
            "nll_sum": 2.0,
            "answer_tokens": 2,
            "metric_score": 1.0,
        },
        {
            "factor": 2.0,
            "task": "passkey",
            "target_length": 16384,
            "depth_percent": 50.0,
            "nll_sum": 4.0,
            "answer_tokens": 2,
            "metric_score": 0.0,
        },
    ]
    summary = summarize_results(rows)
    cell = summary["x2"]["passkey"]["16K"]["depth_50"]
    assert cell == {"examples": 2, "nll": 1.5, "metric_mean": 0.5, "answer_tokens": 4}


def test_generation_passes_explicit_all_one_attention_mask():
    class FakeModel:
        def generate(self, **kwargs):
            assert torch.equal(kwargs["attention_mask"], torch.ones_like(kwargs["input_ids"]))
            return torch.cat((kwargs["input_ids"], torch.tensor([[9]], dtype=torch.long)), dim=1)

    class FakeTokenizer:
        eos_token_id = 0

        def decode(self, ids, skip_special_tokens):
            assert skip_special_tokens is True
            return "answer"

    assert generate_answer(
        FakeModel(),
        FakeTokenizer(),
        prompt_ids=[1, 2],
        metric="exact_match",
        device=torch.device("cpu"),
    ) == "answer"


def test_generation_metrics_do_not_collapse_retrieval_into_whole_output_exactness():
    metrics = score_generation_metrics(
        "The passkey is 12345678. trailing text",
        ["12345678"],
        eos_terminated=False,
        generated_token_count=12,
    )

    assert metrics == {
        "strict_exact": False,
        "first_value_exact": True,
        "gold_containment": True,
        "eos_terminated": False,
        "generated_token_count": 12,
    }


def test_official_ruler_and_nolima_scorers_preserve_their_match_contracts():
    assert score_capability_prediction(
        "ruler_string_match",
        "Found ALPHA and then beta with trailing prose.",
        ["alpha", "beta"],
        source={"match_type": "all"},
    ) == 1.0
    assert score_capability_prediction(
        "ruler_string_match",
        "Only beta appears.",
        ["alpha", "beta"],
        source={"match_type": "all"},
    ) == 0.5
    assert score_capability_prediction(
        "ruler_string_match",
        "Only beta appears.",
        ["alpha", "beta"],
        source={"match_type": "part"},
    ) == 1.0
    assert score_capability_prediction(
        "contains", "Answer: Ada.", ["Ada"], source={}
    ) == 1.0
    assert score_capability_prediction(
        "contains", "Answer: ada.", ["Ada"], source={}
    ) == 0.0


def test_unified_server_entrypoint_runs_both_registered_substrates():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/2026-07/09_lora_evq_official_yarn_eval.sh").read_text()
    assert 'run_arm "$GEO_ADAPTER" native_geo "$GEO_OUTPUT"' in script
    assert "geo_longalpaca_s42" in script
    assert 'run_arm "$EVQ_ADAPTER" evq_cosh "$EVQ_OUTPUT"' in script
    assert "evq_longalpaca_tau1414_s42" in script
    assert script.count("/tmp/") >= 2
    assert "--yarn_factors 2,4" in script
    assert "--model_manifest" in script
    private_server_root = "/" + "root/autodl-tmp"
    assert private_server_root not in script
