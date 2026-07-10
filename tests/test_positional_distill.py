from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

from experiments.lora_evq_v2.eval_positional_distill import (
    representation_recovery,
    result_filename,
    resolve_candidate_frequency_artifact,
)
from experiments.lora_evq_v2.prepare_positional_distill_data import (
    iter_plain_text_jsonl,
    pack_token_sequences,
)
from experiments.lora_evq_v2.summarize_positional_distill import summarize_results
from experiments.lora_evq_v2.train_positional_distill import (
    build_distill_metadata,
    fingerprint_model_source,
    normalized_bucket_hidden_mse,
    position_bucket_ranges,
    validate_distill_manifest,
)


def test_plain_text_jsonl_accepts_only_nonempty_text(tmp_path: Path) -> None:
    path = tmp_path / "plain.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"text": " alpha "}),
                json.dumps({"text": ""}),
                json.dumps({"text": "beta"}),
            ]
        ),
        encoding="utf-8",
    )

    assert list(iter_plain_text_jsonl(path)) == ["alpha", "beta"]


def test_plain_text_jsonl_rejects_instruction_rows(tmp_path: Path) -> None:
    path = tmp_path / "chat.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="plain text"):
        list(iter_plain_text_jsonl(path))


def test_pack_token_sequences_carries_remainder() -> None:
    packed, remainder = pack_token_sequences(
        [[1, 2, 3], [4, 5, 6]],
        seq_len=4,
    )

    assert packed.dtype == torch.int32
    assert packed.tolist() == [[1, 2, 3, 4]]
    assert remainder == [5, 6]


def test_position_bucket_ranges_match_protocol() -> None:
    assert position_bucket_ranges(8192) == (
        (0, 2048),
        (2048, 4096),
        (4096, 8192),
    )


def test_normalized_bucket_hidden_mse_is_zero_for_exact_match() -> None:
    teacher = torch.randn(2, 8, 4)
    student = teacher.clone().requires_grad_(True)
    attention_mask = torch.ones(2, 8, dtype=torch.long)

    loss, bucket_losses = normalized_bucket_hidden_mse(
        student,
        teacher,
        attention_mask,
        buckets=((0, 2), (2, 4), (4, 8)),
    )

    assert loss.item() == pytest.approx(0.0)
    assert bucket_losses == pytest.approx([0.0, 0.0, 0.0])


def test_normalized_bucket_hidden_mse_weights_buckets_equally() -> None:
    teacher = torch.ones(1, 8, 1)
    student = teacher.clone()
    student[:, 0:2] += 1.0
    student[:, 2:4] += 2.0
    student[:, 4:8] += 3.0
    attention_mask = torch.ones(1, 8, dtype=torch.long)

    loss, bucket_losses = normalized_bucket_hidden_mse(
        student,
        teacher,
        attention_mask,
        buckets=((0, 2), (2, 4), (4, 8)),
    )

    assert bucket_losses == pytest.approx([1.0, 4.0, 9.0])
    assert loss.item() == pytest.approx((1.0 + 4.0 + 9.0) / 3.0)


def test_normalized_bucket_hidden_mse_ignores_masked_tokens() -> None:
    teacher = torch.ones(1, 4, 1)
    student = teacher.clone()
    student[:, 2:] = 100.0
    attention_mask = torch.tensor([[1, 1, 0, 0]])

    loss, bucket_losses = normalized_bucket_hidden_mse(
        student,
        teacher,
        attention_mask,
        buckets=((0, 2), (2, 4)),
    )

    assert bucket_losses == pytest.approx([0.0])
    assert loss.item() == pytest.approx(0.0)


def test_validate_distill_manifest_rejects_wrong_protocol(tmp_path: Path) -> None:
    train = torch.zeros((2400, 8192), dtype=torch.int32)
    train_path = tmp_path / "train.pt"
    torch.save(train, train_path)
    manifest = {
        "seq_len": 4096,
        "train_sequences": 2400,
        "validation_sequences": 128,
        "files": {"train": {"name": "train.pt"}},
    }

    with pytest.raises(ValueError, match="seq_len=8192"):
        validate_distill_manifest(manifest, tmp_path)


def test_distill_metadata_records_clean_protocol() -> None:
    metadata = build_distill_metadata(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        student_method="evq_cosh",
        tau=1.414,
        seed=42,
        max_steps=300,
        seq_len=8192,
        per_device_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        lora_targets=("q_proj", "k_proj"),
        data_manifest_sha256="a" * 64,
        train_time_hours=2.5,
        final_loss=0.01,
        peak_cuda_allocated_gb=71.5,
        peak_cuda_reserved_gb=74.0,
        base_model_fingerprint={"identifier": "Meta-Llama-3-8B-Instruct"},
        adapter_sha256="c" * 64,
    )

    assert metadata["objective"] == "positional_hidden_distillation"
    assert metadata["teacher_method"] == "native_geo"
    assert metadata["student_method"] == "evq_cosh"
    assert metadata["lora_targets"] == ["q_proj", "k_proj"]
    assert metadata["effective_batch_size"] == 8
    assert metadata["data_manifest_sha256"] == "a" * 64
    assert metadata["peak_cuda_allocated_gb"] == pytest.approx(71.5)
    assert metadata["peak_cuda_reserved_gb"] == pytest.approx(74.0)
    assert metadata["base_model_fingerprint"]["identifier"] == "Meta-Llama-3-8B-Instruct"
    assert metadata["adapter_sha256"] == "c" * 64


def test_model_fingerprint_is_path_safe(tmp_path: Path) -> None:
    model_dir = tmp_path / "Meta-Llama-3-8B-Instruct"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"rope_theta": 500000}', encoding="utf-8")
    (model_dir / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"weights")

    fingerprint = fingerprint_model_source(str(model_dir))

    assert fingerprint["identifier"] == "Meta-Llama-3-8B-Instruct"
    assert len(fingerprint["config_sha256"]) == 64
    assert len(fingerprint["index_sha256"]) == 64
    assert fingerprint["shards"] == [
        {"name": "model-00001-of-00002.safetensors", "size_bytes": 7}
    ]
    assert str(tmp_path) not in json.dumps(fingerprint)


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("base_geo", "positional_distill_base_geo.json"),
        ("base_evq", "positional_distill_base_evq.json"),
        ("geo_distill_s42", "positional_distill_geo_distill_s42.json"),
        ("evq_distill_s42", "positional_distill_evq_distill_s42.json"),
    ],
)
def test_result_filename_is_stable(variant: str, expected: str) -> None:
    assert result_filename(variant) == expected


def test_result_filename_rejects_unsafe_variant() -> None:
    with pytest.raises(ValueError, match="variant"):
        result_filename("../evq")


def test_representation_recovery_is_bounded() -> None:
    assert representation_recovery(1.0, 0.1) == pytest.approx(0.9)
    assert representation_recovery(1.0, 2.0) == pytest.approx(-1.0)
    assert representation_recovery(1.0, -0.1) == pytest.approx(1.0)
    assert representation_recovery(0.0, 0.0) is None


def test_adapter_evaluation_requires_matching_frequency_artifact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    torch.save(
        {
            "inv_freq": torch.ones(64),
            "method": "native_geo",
            "head_dim": 128,
            "base": 500_000.0,
            "midpoint": False,
        },
        checkpoint / "custom_inv_freq.pt",
    )

    with pytest.raises(RuntimeError, match="method mismatch"):
        resolve_candidate_frequency_artifact(
            checkpoint,
            candidate_method="evq_cosh",
        )


def test_seed42_launcher_encodes_the_approved_protocol() -> None:
    launcher = (
        ROOT / "scripts" / "2026-07" / "01_lora_positional_distill_seed42.sh"
    ).read_text(encoding="utf-8")

    assert 'case "${1:-}"' in launcher
    for phase in ("prepare)", "train)", "eval)", "all)"):
        assert phase in launcher
    assert "train_one geo_distill_s42 native_geo" in launcher
    assert "train_one evq_distill_s42 evq_cosh" in launcher
    assert '--student_method "${method}"' in launcher
    assert "--lora_targets q_proj,k_proj" in launcher
    assert "--seed 42" in launcher
    assert "--max_steps 300" in launcher
    assert "nvidia-smi" in launcher
    assert "/" + "root/autodl-tmp" not in launcher


def test_summary_applies_the_approved_decision_gates() -> None:
    def arm(ppl_8k: float, ppl_16k: float, ppl_32k: float, hidden: float) -> dict:
        return {
            "model": "Meta-Llama-3-8B-Instruct",
            "base_model_fingerprint": {
                "identifier": "Meta-Llama-3-8B-Instruct",
                "config_sha256": "d" * 64,
            },
            "data_manifest_sha256": "b" * 64,
            "ppl": {
                "8K": {"ppl": ppl_8k},
                "16K": {"ppl": ppl_16k},
                "32K": {"ppl": ppl_32k},
            },
            "hidden_error": {"normalized_mse": hidden},
        }

    summary = summarize_results(
        {
            "base_geo": arm(10.0, 100.0, 1000.0, 0.0),
            "base_evq": arm(13.0, 80.0, 700.0, 1.0),
            "geo_distill_s42": arm(10.05, 90.0, 800.0, 0.0),
            "evq_distill_s42": arm(10.4, 40.0, 200.0, 0.1),
        }
    )

    assert summary["metrics"]["geo_8k_drift_pct"] == pytest.approx(0.5)
    assert summary["metrics"]["evq_8k_drift_pct"] == pytest.approx(4.0)
    assert summary["metrics"]["evq_16k_improvement_factor"] == pytest.approx(2.5)
    assert summary["metrics"]["evq_32k_improvement_factor"] == pytest.approx(5.0)
    assert summary["metrics"]["representation_recovery"] == pytest.approx(0.9)
    assert all(summary["gates"].values())
    assert summary["pilot_pass"] is True


def test_summary_rejects_mixed_data_manifests() -> None:
    row = {
        "model": "Meta-Llama-3-8B-Instruct",
        "base_model_fingerprint": {
            "identifier": "Meta-Llama-3-8B-Instruct",
            "config_sha256": "d" * 64,
        },
        "ppl": {name: {"ppl": 1.0} for name in ("8K", "16K", "32K")},
        "hidden_error": {"normalized_mse": 0.1},
    }
    results = {
        variant: {**row, "data_manifest_sha256": str(index) * 64}
        for index, variant in enumerate(
            ("base_geo", "base_evq", "geo_distill_s42", "evq_distill_s42"),
            start=1,
        )
    }

    with pytest.raises(ValueError, match="data manifest"):
        summarize_results(results)


def test_summary_rejects_mixed_model_fingerprints() -> None:
    def arm(config_hash: str) -> dict:
        return {
            "model": "Meta-Llama-3-8B-Instruct",
            "base_model_fingerprint": {
                "identifier": "Meta-Llama-3-8B-Instruct",
                "config_sha256": config_hash,
            },
            "data_manifest_sha256": "b" * 64,
            "ppl": {name: {"ppl": 1.0} for name in ("8K", "16K", "32K")},
            "hidden_error": {"normalized_mse": 0.1},
        }

    results = {variant: arm("d" * 64) for variant in (
        "base_geo",
        "base_evq",
        "geo_distill_s42",
        "evq_distill_s42",
    )}
    results["evq_distill_s42"] = arm("e" * 64)

    with pytest.raises(ValueError, match="model fingerprint"):
        summarize_results(results)
