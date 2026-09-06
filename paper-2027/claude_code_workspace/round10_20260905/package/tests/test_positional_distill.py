from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

from experiments.lora_evq_v2.eval_positional_distill import (
    chunked_causal_nll,
    non_overlapping_chunk_ranges,
    representation_recovery,
    result_filename,
    resolve_candidate_frequency_artifact,
    validate_adapter_metadata,
    validate_eval_claim_protocol,
)
from experiments.lora_evq_v2.prepare_positional_distill_data import (
    collect_disjoint_train_validation_sequences,
    iter_plain_text_jsonl,
    pack_token_sequences,
)
from experiments.lora_evq_v2.summarize_positional_distill import summarize_results
from experiments.lora_evq_v2.train_positional_distill import (
    build_distill_metadata,
    build_compiled_student_backbone,
    checkpoint_starting_global_step,
    configure_packed_free_causal_sdpa,
    ensure_immutable_run_protocol,
    fingerprint_model_source,
    normalized_bucket_hidden_mse,
    position_bucket_ranges,
    validate_claim_protocol,
    validate_distill_manifest,
)
from experiments.lora_evq_v2.validate_checkpoint_artifact import (
    append_invocation_ledger,
    finalize_claim_ready,
    validate_claim_ready,
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


def test_plain_text_jsonl_rejects_mixed_chat_and_text_rows(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": "flattened assistant answer",
                "messages": [{"role": "assistant", "content": "answer"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="instruction/chat fields"):
        list(iter_plain_text_jsonl(path))


def test_pack_token_sequences_carries_remainder() -> None:
    packed, remainder = pack_token_sequences(
        [[1, 2, 3], [4, 5, 6]],
        seq_len=4,
    )

    assert packed.dtype == torch.int32
    assert packed.tolist() == [[1, 2, 3, 4]]
    assert remainder == [5, 6]


def test_frozen_splits_discard_boundary_document_remainder() -> None:
    class Tokenizer:
        eos_token_id = 99

        def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
            assert add_special_tokens is False
            return {"input_ids": [int(text)] * 4}

    train, validation, counts = collect_disjoint_train_validation_sequences(
        tokenizer=Tokenizer(),
        texts=iter(["1", "2", "3"]),
        seq_len=4,
        train_sequences=1,
        validation_sequences=1,
    )

    assert validation.tolist() == [[1, 1, 1, 1]]
    assert train.tolist() == [[2, 2, 2, 2]]
    assert counts == {"validation": 1, "train": 1}


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
    assert bucket_losses.tolist() == pytest.approx([0.0, 0.0, 0.0])


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

    assert bucket_losses.tolist() == pytest.approx([1.0, 4.0, 9.0])
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

    assert bucket_losses.tolist() == pytest.approx([0.0])
    assert loss.item() == pytest.approx(0.0)


def test_validate_distill_manifest_rejects_wrong_protocol(tmp_path: Path) -> None:
    train = torch.zeros((2400, 8192), dtype=torch.int32)
    train_path = tmp_path / "train.pt"
    torch.save(train, train_path)
    manifest = {
        "format_version": 1,
        "purpose": "llama8b_positional_hidden_distillation",
        "seed": 42,
        "seq_len": 4096,
        "train_sequences": 2400,
        "validation_sequences": 128,
        "files": {"train": {"name": "train.pt"}},
    }

    with pytest.raises(ValueError, match="seq_len=8192"):
        validate_distill_manifest(manifest, tmp_path)


def test_validate_distill_manifest_rejects_non_seed42_data(tmp_path: Path) -> None:
    manifest = {
        "format_version": 1,
        "purpose": "llama8b_positional_hidden_distillation",
        "seed": 7,
        "seq_len": 8192,
        "train_sequences": 2400,
        "validation_sequences": 128,
        "files": {},
    }

    with pytest.raises(ValueError, match="seed=42"):
        validate_distill_manifest(manifest, tmp_path)


def test_claim_protocol_allows_faster_microbatch_with_same_effective_batch() -> None:
    validate_claim_protocol(
        student_method="evq_cosh",
        tau=1.414,
        seed=42,
        max_steps=300,
        per_device_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_steps=30,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        lora_targets=("q_proj", "k_proj"),
        bf16=True,
    )


def test_claim_protocol_rejects_scientific_hyperparameter_drift() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        validate_claim_protocol(
            student_method="evq_cosh",
            tau=1.414,
            seed=42,
            max_steps=300,
            per_device_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            warmup_steps=30,
            weight_decay=0.01,
            max_grad_norm=1.0,
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.0,
            lora_targets=("q_proj", "k_proj"),
            bf16=True,
        )


def test_student_compile_wraps_backbone_without_replacing_model(monkeypatch) -> None:
    class CausalModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Linear(2, 2)

    model = CausalModel()
    seen = {}

    def fake_compile(module, **kwargs):
        seen.update(kwargs)
        return ("compiled", module)

    monkeypatch.setattr(torch, "compile", fake_compile)
    compiled = build_compiled_student_backbone(
        model,
        enabled=True,
        backend="inductor",
        mode="default",
    )

    assert compiled == ("compiled", model.model)
    assert seen == {"backend": "inductor", "mode": "default", "dynamic": False}
    assert isinstance(model.model, torch.nn.Linear)


def test_packed_free_causal_sdpa_avoids_materialized_transformers_mask() -> None:
    from transformers import LlamaConfig, LlamaModel
    from transformers.masking_utils import create_causal_mask

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=8,
    )
    model = LlamaModel(config)

    backend = configure_packed_free_causal_sdpa(model)
    embeds = torch.zeros((2, 8, 16))
    positions = torch.arange(8)
    causal_mask = create_causal_mask(
        config=model.config,
        input_embeds=embeds,
        attention_mask=None,
        cache_position=positions,
        past_key_values=None,
        position_ids=positions.unsqueeze(0),
    )

    assert backend == "evq_packed_free_causal_sdpa"
    assert model.config._attn_implementation == backend
    assert causal_mask is None


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
        warmup_steps=30,
        weight_decay=0.01,
        max_grad_norm=1.0,
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
    assert metadata["weight_decay"] == pytest.approx(0.01)
    assert metadata["max_grad_norm"] == pytest.approx(1.0)
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
        {
            "name": "model-00001-of-00002.safetensors",
            "size_bytes": 7,
            "sha256": "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c",
        }
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


def test_adapter_evaluation_rejects_noncanonical_frequency_values(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    torch.save(
        {
            "inv_freq": torch.ones(64),
            "method": "evq_cosh",
            "tau": 1.414,
            "head_dim": 128,
            "base": 500_000.0,
            "midpoint": True,
        },
        checkpoint / "custom_inv_freq.pt",
    )

    with pytest.raises(RuntimeError, match="canonical schedule"):
        resolve_candidate_frequency_artifact(
            checkpoint,
            candidate_method="evq_cosh",
        )


def test_adapter_metadata_is_bound_to_clean_protocol() -> None:
    metadata = {
        "objective": "positional_hidden_distillation",
        "student_method": "evq_cosh",
        "rope_method": "evq_cosh",
        "tau": 1.414,
        "seed": 42,
        "max_steps": 300,
        "effective_batch_size": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.0,
        "lora_targets": ["q_proj", "k_proj"],
        "data_manifest_sha256": "a" * 64,
        "adapter_sha256": "b" * 64,
        "student_frequency": {"method": "evq_cosh"},
    }

    validate_adapter_metadata(
        metadata,
        candidate_method="evq_cosh",
        data_manifest_sha256="a" * 64,
        adapter_sha256="b" * 64,
    )
    metadata["objective"] = "causal_lm"
    with pytest.raises(RuntimeError, match="objective"):
        validate_adapter_metadata(
            metadata,
            candidate_method="evq_cosh",
            data_manifest_sha256="a" * 64,
            adapter_sha256="b" * 64,
        )


def test_chunked_causal_nll_matches_full_logits() -> None:
    torch.manual_seed(0)
    hidden = torch.randn(2, 7, 5)
    input_ids = torch.randint(0, 11, (2, 7))
    lm_head = torch.nn.Linear(5, 11, bias=False)
    full_logits = lm_head(hidden[:, :-1]).float()
    expected = torch.nn.functional.cross_entropy(
        full_logits.reshape(-1, full_logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    )

    actual = chunked_causal_nll(hidden, input_ids, lm_head, chunk_tokens=3)

    assert actual.item() == pytest.approx(expected.item(), rel=1e-6)


def test_ppl_chunk_ranges_require_the_preregistered_count() -> None:
    assert non_overlapping_chunk_ranges(100, length=20, chunks=5) == (
        (0, 20),
        (20, 40),
        (40, 60),
        (60, 80),
        (80, 100),
    )
    with pytest.raises(ValueError, match="requires 100 tokens"):
        non_overlapping_chunk_ranges(99, length=20, chunks=5)


def test_eval_claim_protocol_requires_full_preregistered_evidence() -> None:
    validate_eval_claim_protocol(
        ppl_lengths=(8192, 16384, 32768),
        ppl_chunks=5,
        hidden_sequences=128,
        hidden_batch_size=4,
        bf16=True,
    )

    with pytest.raises(ValueError, match="ppl_chunks=5"):
        validate_eval_claim_protocol(
            ppl_lengths=(8192, 16384, 32768),
            ppl_chunks=1,
            hidden_sequences=128,
            hidden_batch_size=4,
            bf16=True,
        )
    with pytest.raises(ValueError, match="hidden_sequences=128"):
        validate_eval_claim_protocol(
            ppl_lengths=(8192, 16384, 32768),
            ppl_chunks=5,
            hidden_sequences=8,
            hidden_batch_size=4,
            bf16=True,
        )
    with pytest.raises(ValueError, match="hidden_batch_size=4"):
        validate_eval_claim_protocol(
            ppl_lengths=(8192, 16384, 32768),
            ppl_chunks=5,
            hidden_sequences=128,
            hidden_batch_size=3,
            bf16=True,
        )


def test_checkpoint_starting_global_step_is_explicit(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 100}),
        encoding="utf-8",
    )

    assert checkpoint_starting_global_step(checkpoint) == 100
    assert checkpoint_starting_global_step(None) == 0


def test_run_protocol_is_immutable_across_resume(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    protocol = {"objective": "positional_hidden_distillation", "compile": True}

    path = ensure_immutable_run_protocol(output_dir, protocol)
    assert json.loads(path.read_text(encoding="utf-8")) == protocol
    assert ensure_immutable_run_protocol(output_dir, protocol) == path

    with pytest.raises(RuntimeError, match="protocol mismatch"):
        ensure_immutable_run_protocol(
            output_dir,
            {"objective": "positional_hidden_distillation", "compile": False},
        )


def test_resume_checkpoint_without_run_protocol_fails_closed(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    (output_dir / "checkpoint-100").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="run_protocol.json"):
        ensure_immutable_run_protocol(output_dir, {"objective": "x"})


def test_claim_ready_binds_all_resume_invocations(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    logs = tmp_path / "logs"
    checkpoint.mkdir()
    logs.mkdir()
    (checkpoint / "run_protocol.json").write_text(
        json.dumps({"scientific": {"max_steps": 2}}) + "\n",
        encoding="utf-8",
    )

    def evidence(suffix: str) -> tuple[Path, Path, Path]:
        paths = tuple(logs / f"{name}_{suffix}.txt" for name in ("gpu", "hw", "train"))
        for path in paths:
            path.write_text(f"evidence-{suffix}\n", encoding="utf-8")
        return paths

    first_checkpoint = checkpoint / "checkpoint-1"
    first_checkpoint.mkdir()
    (first_checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 1}), encoding="utf-8"
    )
    first = evidence("first")
    append_invocation_ledger(
        checkpoint,
        logs,
        telemetry=first[0],
        hardware_record=first[1],
        train_log=first[2],
        process_status=1,
    )

    (checkpoint / "experiment_meta.json").write_text(
        json.dumps({"invocation": {"ending_global_step": 2}}),
        encoding="utf-8",
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 2}),
        encoding="utf-8",
    )
    second = evidence("second")
    append_invocation_ledger(
        checkpoint,
        logs,
        telemetry=second[0],
        hardware_record=second[1],
        train_log=second[2],
        process_status=0,
    )
    for name in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "custom_inv_freq.pt",
    ):
        (checkpoint / name).write_bytes(name.encode("utf-8"))
    finalize_claim_ready(
        checkpoint,
        logs,
        telemetry=second[0],
        hardware_record=second[1],
        train_log=second[2],
    )

    marker = validate_claim_ready(checkpoint, logs)
    assert marker["format_version"] == 1
    first[2].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invocation evidence mismatch"):
        validate_claim_ready(checkpoint, logs)


def test_representation_recovery_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        representation_recovery(1.0, math.nan)


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
    assert "train_one evq_distill_s42 evq_cosh 300 30" in launcher
    assert "train_one geo_distill_s42 native_geo 1 0" in launcher
    assert '--max_steps "${max_steps}"' in launcher
    assert "--compile_mode" in launcher
    assert "nvidia-smi" in launcher
    assert launcher.count('nvidia-smi -i "${CUDA_SELECTOR}"') == 2
    assert "set CUDA_VISIBLE_DEVICES to exactly one GPU index or UUID" in launcher
    assert "--append-invocation-ledger" in launcher
    assert "--require-claim-ready" in launcher
    assert "benchmark)" in launcher
    assert "/" + "root/autodl-tmp" not in launcher


def test_summary_applies_the_approved_decision_gates() -> None:
    def arm(
        variant: str,
        method: str,
        ppl_8k: float,
        ppl_16k: float,
        ppl_32k: float,
        hidden: float,
    ) -> dict:
        requires_adapter = not variant.startswith("base_")
        return {
            "variant": variant,
            "candidate_method": method,
            "seed": 42,
            "adapter": variant if requires_adapter else None,
            "adapter_protocol_validated": requires_adapter,
            "claim_ready_sha256": "r" * 64 if requires_adapter else None,
            "adapter_run_protocol": (
                {
                    "performance": {"per_device_batch_size": 2},
                    "runtime": {"torch": "2.8.0"},
                }
                if requires_adapter
                else None
            ),
            "adapter_training_protocol": (
                {
                    "objective": "positional_hidden_distillation",
                    "seed": 42,
                    "max_steps": 1 if method == "native_geo" else 300,
                    "warmup_steps": 0 if method == "native_geo" else 30,
                    "effective_batch_size": 8,
                    "learning_rate": 2e-5,
                    "weight_decay": 0.01,
                    "max_grad_norm": 1.0,
                    "lora_r": 64,
                    "lora_alpha": 128,
                    "lora_dropout": 0.0,
                    "lora_targets": ["q_proj", "k_proj"],
                    "data_manifest_sha256": "b" * 64,
                }
                if requires_adapter
                else None
            ),
            "frequency_provenance": {
                "method": method,
                "tensor_sha256": ("f" if method == "native_geo" else "e") * 64,
            },
            "eval_corpus": {"identifier": "wikitext.txt", "sha256": "c" * 64},
            "tokenizer_fingerprint": {"identifier": "Meta-Llama-3-8B-Instruct"},
            "eval_config": {
                "bf16": True,
                "ppl_lengths": [8192, 16384, 32768],
                "ppl_chunks": 5,
                "hidden_sequences": 128,
                "hidden_batch_size": 4,
            },
            "code_sha256": {"eval_positional_distill.py": "a" * 64},
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
            "base_geo": arm("base_geo", "native_geo", 10.0, 100.0, 1000.0, 0.0),
            "base_evq": arm("base_evq", "evq_cosh", 13.0, 80.0, 700.0, 1.0),
            "geo_distill_s42": arm(
                "geo_distill_s42", "native_geo", 10.05, 100.5, 1005.0, 0.0
            ),
            "evq_distill_s42": arm(
                "evq_distill_s42", "evq_cosh", 10.4, 40.0, 200.0, 0.1
            ),
        }
    )

    assert summary["metrics"]["geo_8k_drift_pct"] == pytest.approx(0.5)
    assert summary["metrics"]["evq_8k_drift_pct"] == pytest.approx(4.0)
    assert summary["metrics"]["evq_16k_improvement_factor"] == pytest.approx(2.5)
    assert summary["metrics"]["evq_32k_improvement_factor"] == pytest.approx(5.0)
    assert summary["metrics"]["matched_16k_improvement_factor"] == pytest.approx(
        100.5 / 40.0
    )
    assert summary["metrics"]["matched_32k_improvement_factor"] == pytest.approx(
        1005.0 / 200.0
    )
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
    results = {}
    for index, variant in enumerate(
        ("base_geo", "base_evq", "geo_distill_s42", "evq_distill_s42"),
        start=1,
    ):
        method = "native_geo" if variant in {"base_geo", "geo_distill_s42"} else "evq_cosh"
        requires_adapter = not variant.startswith("base_")
        results[variant] = {
            **row,
            "variant": variant,
            "candidate_method": method,
            "seed": 42,
            "adapter": variant if requires_adapter else None,
            "adapter_protocol_validated": requires_adapter,
            "claim_ready_sha256": "r" * 64 if requires_adapter else None,
            "adapter_run_protocol": (
                {
                    "performance": {"per_device_batch_size": 2},
                    "runtime": {"torch": "2.8.0"},
                }
                if requires_adapter
                else None
            ),
            "adapter_training_protocol": (
                {
                    "objective": "positional_hidden_distillation",
                    "seed": 42,
                    "max_steps": 1 if method == "native_geo" else 300,
                    "warmup_steps": 0 if method == "native_geo" else 30,
                    "effective_batch_size": 8,
                    "learning_rate": 2e-5,
                    "weight_decay": 0.01,
                    "max_grad_norm": 1.0,
                    "lora_r": 64,
                    "lora_alpha": 128,
                    "lora_dropout": 0.0,
                    "lora_targets": ["q_proj", "k_proj"],
                    "data_manifest_sha256": str(index) * 64,
                }
                if requires_adapter
                else None
            ),
            "frequency_provenance": {
                "method": method,
                "tensor_sha256": ("f" if method == "native_geo" else "e") * 64,
            },
            "eval_corpus": {"identifier": "wikitext.txt", "sha256": "c" * 64},
            "tokenizer_fingerprint": {"identifier": "Meta-Llama-3-8B-Instruct"},
            "eval_config": {
                "bf16": True,
                "ppl_lengths": [8192, 16384, 32768],
                "ppl_chunks": 5,
                "hidden_sequences": 128,
                "hidden_batch_size": 4,
            },
            "code_sha256": {"eval_positional_distill.py": "a" * 64},
            "data_manifest_sha256": str(index) * 64,
        }

    with pytest.raises(ValueError, match="data manifest"):
        summarize_results(results)


def test_summary_rejects_mislabeled_arm() -> None:
    row = {
        "variant": "wrong",
        "candidate_method": "native_geo",
        "seed": 42,
        "adapter": None,
        "adapter_protocol_validated": False,
        "model": "Meta-Llama-3-8B-Instruct",
        "base_model_fingerprint": {"config_sha256": "d" * 64},
        "data_manifest_sha256": "b" * 64,
        "ppl": {name: {"ppl": 1.0} for name in ("8K", "16K", "32K")},
        "hidden_error": {"normalized_mse": 0.1},
    }
    results = {variant: dict(row) for variant in (
        "base_geo",
        "base_evq",
        "geo_distill_s42",
        "evq_distill_s42",
    )}

    with pytest.raises(ValueError, match="variant mismatch"):
        summarize_results(results)


def test_summary_rejects_mixed_model_fingerprints() -> None:
    def arm(variant: str, config_hash: str) -> dict:
        method = "native_geo" if variant in {"base_geo", "geo_distill_s42"} else "evq_cosh"
        requires_adapter = not variant.startswith("base_")
        return {
            "variant": variant,
            "candidate_method": method,
            "seed": 42,
            "adapter": variant if requires_adapter else None,
            "adapter_protocol_validated": requires_adapter,
            "claim_ready_sha256": "r" * 64 if requires_adapter else None,
            "adapter_run_protocol": (
                {
                    "performance": {"per_device_batch_size": 2},
                    "runtime": {"torch": "2.8.0"},
                }
                if requires_adapter
                else None
            ),
            "adapter_training_protocol": (
                {
                    "objective": "positional_hidden_distillation",
                    "seed": 42,
                    "max_steps": 1 if method == "native_geo" else 300,
                    "warmup_steps": 0 if method == "native_geo" else 30,
                    "effective_batch_size": 8,
                    "learning_rate": 2e-5,
                    "weight_decay": 0.01,
                    "max_grad_norm": 1.0,
                    "lora_r": 64,
                    "lora_alpha": 128,
                    "lora_dropout": 0.0,
                    "lora_targets": ["q_proj", "k_proj"],
                    "data_manifest_sha256": "b" * 64,
                }
                if requires_adapter
                else None
            ),
            "frequency_provenance": {
                "method": method,
                "tensor_sha256": ("f" if method == "native_geo" else "e") * 64,
            },
            "eval_corpus": {"identifier": "wikitext.txt", "sha256": "c" * 64},
            "tokenizer_fingerprint": {"identifier": "Meta-Llama-3-8B-Instruct"},
            "eval_config": {
                "bf16": True,
                "ppl_lengths": [8192, 16384, 32768],
                "ppl_chunks": 5,
                "hidden_sequences": 128,
                "hidden_batch_size": 4,
            },
            "code_sha256": {"eval_positional_distill.py": "a" * 64},
            "model": "Meta-Llama-3-8B-Instruct",
            "base_model_fingerprint": {
                "identifier": "Meta-Llama-3-8B-Instruct",
                "config_sha256": config_hash,
            },
            "data_manifest_sha256": "b" * 64,
            "ppl": {name: {"ppl": 1.0} for name in ("8K", "16K", "32K")},
            "hidden_error": {"normalized_mse": 0.1},
        }

    results = {variant: arm(variant, "d" * 64) for variant in (
        "base_geo",
        "base_evq",
        "geo_distill_s42",
        "evq_distill_s42",
    )}
    results["evq_distill_s42"] = arm("evq_distill_s42", "e" * 64)

    with pytest.raises(ValueError, match="model fingerprint"):
        summarize_results(results)
