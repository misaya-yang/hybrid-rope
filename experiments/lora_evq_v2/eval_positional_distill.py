#!/usr/bin/env python3
"""Evaluate the four arms of the clean positional-distillation pilot."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_positional_distill_data import sha256_file
from train_evq_lora import (
    build_training_inv_freq,
    inject_inv_freq,
    load_frequency_artifact,
    public_artifact_identifier,
    public_model_identifier,
    resolve_model_rope_geometry,
    verify_model_inv_freq,
)
from train_positional_distill import (
    FrozenSequenceDataset,
    FrequencySwitcher,
    causal_backbone,
    configure_packed_free_causal_sdpa,
    fingerprint_model_source,
    normalized_bucket_hidden_mse,
    position_bucket_ranges,
    validate_distill_manifest,
    validate_single_gpu_runtime,
)
from validate_checkpoint_artifact import validate_claim_ready


ARM_CONTRACT = {
    "base_geo": ("native_geo", False),
    "base_evq": ("evq_cosh", False),
    "geo_distill_s42": ("native_geo", True),
    "evq_distill_s42": ("evq_cosh", True),
}


def result_filename(variant: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", variant):
        raise ValueError("variant must contain only filename-safe characters")
    return f"positional_distill_{variant}.json"


def validate_variant_request(
    variant: str,
    candidate_method: str,
    adapter_dir: Optional[Path],
) -> None:
    if variant not in ARM_CONTRACT:
        raise ValueError(f"unknown positional-distillation variant: {variant}")
    expected_method, requires_adapter = ARM_CONTRACT[variant]
    if candidate_method != expected_method:
        raise ValueError(
            f"variant {variant} requires candidate_method={expected_method}"
        )
    if (adapter_dir is not None) != requires_adapter:
        requirement = "an adapter" if requires_adapter else "no adapter"
        raise ValueError(f"variant {variant} requires {requirement}")


def tokenizer_fingerprint(model_name: str) -> dict:
    output = {"identifier": public_model_identifier(model_name)}
    model_path = Path(model_name).expanduser()
    if model_path.is_dir():
        files = {}
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            path = model_path / name
            if path.is_file():
                files[name] = sha256_file(path)
        output["files"] = files
    return output


def representation_recovery(
    injection_error: float,
    adapted_error: float,
) -> Optional[float]:
    if not math.isfinite(injection_error) or not math.isfinite(adapted_error):
        raise ValueError("representation errors must be finite")
    if injection_error <= 0:
        return None
    return 1.0 - max(0.0, adapted_error) / injection_error


def frequency_tensor_sha256(inv_freq: torch.Tensor) -> str:
    canonical = inv_freq.detach().cpu().to(torch.float64).contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(tuple(canonical.shape)).encode("ascii"))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def resolve_candidate_frequency_artifact(
    adapter_dir: Path,
    candidate_method: str,
):
    expected_method = candidate_method
    path = Path(adapter_dir) / "custom_inv_freq.pt"
    inv_freq, data, provenance = load_frequency_artifact(
        path,
        expected_method=expected_method,
    )
    if int(data.get("head_dim", -1)) != 128 or not math.isclose(
        float(data.get("base", float("nan"))), 500_000.0
    ):
        raise RuntimeError("frequency artifact has the wrong LLaMA-3 geometry")
    if candidate_method == "evq_cosh":
        if data.get("tau") != 1.414 or data.get("midpoint") is not True:
            raise RuntimeError("EVQ frequency artifact must use tau=1.414 midpoint allocation")
    elif data.get("tau") is not None or data.get("midpoint") is not False:
        raise RuntimeError("native Geo frequency artifact metadata is invalid")
    canonical, _ = build_training_inv_freq(
        rope_method=candidate_method,
        head_dim=int(data.get("head_dim", 2 * inv_freq.numel())),
        base=float(data["base"]),
        tau=float(data.get("tau") or 0.0),
    )
    if not torch.allclose(
        inv_freq.to(torch.float64),
        canonical.to(torch.float64),
        rtol=1e-7,
        atol=1e-12,
    ):
        raise RuntimeError("frequency artifact does not match the canonical schedule")
    provenance["tensor_sha256"] = frequency_tensor_sha256(inv_freq)
    return inv_freq, data, provenance


def validate_adapter_metadata(
    metadata: dict,
    *,
    candidate_method: str,
    data_manifest_sha256: str,
    adapter_sha256: str,
    adapter_config: Optional[dict] = None,
    base_model_fingerprint: Optional[dict] = None,
    run_protocol_sha256: Optional[str] = None,
) -> None:
    required = {
        "objective": "positional_hidden_distillation",
        "student_method": candidate_method,
        "rope_method": candidate_method,
        "seed": 42,
        "max_steps": 1 if candidate_method == "native_geo" else 300,
        "effective_batch_size": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.0,
        "lora_targets": ["q_proj", "k_proj"],
        "data_manifest_sha256": data_manifest_sha256,
        "adapter_sha256": adapter_sha256,
    }
    if candidate_method == "evq_cosh":
        required["tau"] = 1.414
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"adapter metadata {key} mismatch: expected {expected!r}, "
                f"found {metadata.get(key)!r}"
            )
    expected_warmup = 0 if candidate_method == "native_geo" else 30
    if metadata.get("student_frequency", {}).get("method") != candidate_method:
        raise RuntimeError("adapter metadata student_frequency method mismatch")
    if adapter_config is not None:
        expected_config = {
            "r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
        }
        for key, expected in expected_config.items():
            if adapter_config.get(key) != expected:
                raise RuntimeError(f"adapter config {key} mismatch")
        if sorted(adapter_config.get("target_modules", [])) != ["k_proj", "q_proj"]:
            raise RuntimeError("adapter config target_modules mismatch")
    if metadata.get("warmup_steps", expected_warmup) != expected_warmup:
        raise RuntimeError("adapter metadata warmup_steps mismatch")
    if (
        base_model_fingerprint is not None
        and metadata.get("base_model_fingerprint") != base_model_fingerprint
    ):
        raise RuntimeError("adapter metadata base-model fingerprint mismatch")
    if (
        run_protocol_sha256 is not None
        and metadata.get("run_protocol_sha256") != run_protocol_sha256
    ):
        raise RuntimeError("adapter metadata run-protocol fingerprint mismatch")


def non_overlapping_chunk_ranges(
    available_tokens: int,
    *,
    length: int,
    chunks: int,
) -> tuple[tuple[int, int], ...]:
    required = length * chunks
    if available_tokens < required:
        raise ValueError(
            f"evaluation at length={length} with {chunks} chunks requires "
            f"{required} tokens; found {available_tokens}"
        )
    return tuple((index * length, (index + 1) * length) for index in range(chunks))


def validate_eval_claim_protocol(
    *,
    ppl_lengths: tuple[int, ...],
    ppl_chunks: int,
    hidden_sequences: int,
    hidden_batch_size: int,
    bf16: bool,
) -> None:
    if ppl_lengths != (8192, 16384, 32768):
        raise ValueError(
            "approved evaluator requires ppl_lengths=(8192, 16384, 32768)"
        )
    if ppl_chunks != 5:
        raise ValueError("approved evaluator requires ppl_chunks=5")
    if hidden_sequences != 128:
        raise ValueError("approved evaluator requires hidden_sequences=128")
    if hidden_batch_size != 4:
        raise ValueError("approved evaluator requires hidden_batch_size=4")
    if not bf16:
        raise ValueError("approved evaluator requires bf16=True")


def chunked_causal_nll(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    lm_head,
    *,
    chunk_tokens: int,
) -> torch.Tensor:
    if hidden_states.shape[:2] != input_ids.shape:
        raise ValueError("hidden states and input_ids must share batch/sequence axes")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    total_nll = torch.zeros((), device=hidden_states.device, dtype=torch.float64)
    token_count = 0
    for start in range(0, input_ids.shape[1] - 1, chunk_tokens):
        end = min(start + chunk_tokens, input_ids.shape[1] - 1)
        logits = lm_head(hidden_states[:, start:end]).float()
        labels = input_ids[:, start + 1 : end + 1]
        total_nll = total_nll + torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="sum",
        ).to(torch.float64)
        token_count += labels.numel()
    if token_count == 0:
        raise ValueError("causal NLL requires at least two tokens")
    return (total_nll / token_count).to(torch.float32)


def load_candidate(
    model_name: str,
    candidate_method: str,
    tau: float,
    adapter_dir: Optional[Path],
    bf16: bool,
    data_manifest_sha256: str,
    base_model_fingerprint: dict,
    claim_log_dir: Optional[Path],
):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    configure_packed_free_causal_sdpa(model)
    geometry = resolve_model_rope_geometry(model.config)
    if (
        getattr(model.config, "model_type", None) != "llama"
        or getattr(model.config, "rope_scaling", None) not in (None, {})
        or geometry.head_dim != 128
        or not math.isclose(geometry.rope_base, 500_000.0)
        or int(getattr(model.config, "max_position_embeddings", -1)) != 8192
    ):
        raise ValueError("evaluator requires the approved default-RoPE LLaMA-3-8B geometry")
    adapter_metadata = None
    adapter_run_protocol = None
    claim_ready_sha256 = None
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        adapter_path = adapter_dir / "adapter_model.safetensors"
        adapter_sha256 = sha256_file(adapter_path)
        metadata_path = adapter_dir / "experiment_meta.json"
        if not metadata_path.is_file():
            raise FileNotFoundError("adapter experiment_meta.json is missing")
        adapter_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        adapter_config_path = adapter_dir / "adapter_config.json"
        if not adapter_config_path.is_file():
            raise FileNotFoundError("adapter adapter_config.json is missing")
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        run_protocol_path = adapter_dir / "run_protocol.json"
        if not run_protocol_path.is_file():
            raise FileNotFoundError("adapter run_protocol.json is missing")
        if claim_log_dir is None:
            raise ValueError("adapter evaluation requires claim_log_dir")
        validate_claim_ready(adapter_dir, claim_log_dir)
        claim_ready_sha256 = sha256_file(adapter_dir / "claim_ready.json")
        adapter_run_protocol = json.loads(
            run_protocol_path.read_text(encoding="utf-8")
        )
        validate_adapter_metadata(
            adapter_metadata,
            candidate_method=candidate_method,
            data_manifest_sha256=data_manifest_sha256,
            adapter_sha256=adapter_sha256,
            adapter_config=adapter_config,
            base_model_fingerprint=base_model_fingerprint,
            run_protocol_sha256=sha256_file(run_protocol_path),
        )
        candidate_inv_freq, _, provenance = resolve_candidate_frequency_artifact(
            adapter_dir,
            candidate_method,
        )
    else:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        geometry = resolve_model_rope_geometry(config)
        candidate_inv_freq, schedule_meta = build_training_inv_freq(
            rope_method=candidate_method,
            head_dim=geometry.head_dim,
            base=geometry.rope_base,
            tau=tau,
        )
        provenance = {
            "artifact": None,
            "method": schedule_meta["method"],
            "tau": schedule_meta["tau"],
            "midpoint": schedule_meta["midpoint"],
            "tensor_sha256": frequency_tensor_sha256(candidate_inv_freq),
        }
    model.to("cuda")
    inject_inv_freq(model, candidate_inv_freq)
    verification = verify_model_inv_freq(model, candidate_inv_freq)
    provenance["verified_modules"] = verification["verified_count"]
    provenance["max_error"] = verification["max_error"]
    model.eval()
    return (
        model,
        tokenizer,
        candidate_inv_freq,
        provenance,
        adapter_metadata,
        adapter_run_protocol,
        claim_ready_sha256,
    )


def evaluate_ppl(
    model,
    tokenizer,
    text_path: Path,
    lengths: tuple[int, ...],
    chunks: int,
    lm_head_chunk_tokens: int,
) -> tuple[dict, int]:
    text = Path(text_path).read_text(encoding="utf-8")
    full_ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    results = {}
    device = next(model.parameters()).device
    backbone = causal_backbone(model)
    lm_head = model.get_output_embeddings()
    for length in lengths:
        losses = []
        ranges = non_overlapping_chunk_ranges(
            len(full_ids),
            length=length,
            chunks=chunks,
        )
        for start, end in ranges:
            input_ids = full_ids[start:end].unsqueeze(0).to(device)
            with torch.inference_mode():
                hidden = backbone(
                    input_ids=input_ids,
                    attention_mask=None,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state
                nll = chunked_causal_nll(
                    hidden,
                    input_ids,
                    lm_head,
                    chunk_tokens=lm_head_chunk_tokens,
                )
            value = float(nll.cpu())
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite PPL NLL at length={length}")
            losses.append(value)
        mean_nll = float(np.mean(losses))
        results[f"{length // 1024}K"] = {
            "nll": round(mean_nll, 6),
            "ppl": round(math.exp(mean_nll), 6),
            "chunks": len(losses),
            "scored_tokens": len(losses) * (length - 1),
        }
    return results, len(full_ids)


def evaluate_hidden_error(
    model,
    validation_tensor: torch.Tensor,
    candidate_inv_freq: torch.Tensor,
    teacher_inv_freq: torch.Tensor,
    sequences: int,
    batch_size: int,
) -> dict:
    dataset = FrozenSequenceDataset(validation_tensor)
    if sequences <= 0 or sequences > len(dataset):
        raise ValueError("hidden sequences must be in [1, validation size]")
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(sequences)),
        batch_size=batch_size,
        shuffle=False,
    )
    backbone = causal_backbone(model)
    frequency_switcher = FrequencySwitcher(model, teacher_inv_freq, candidate_inv_freq)
    device = next(model.parameters()).device
    losses = []
    bucket_history = []
    batch_weights = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        frequency_switcher.install("teacher")
        disable_adapter = getattr(model, "disable_adapter", None)
        adapter_context = disable_adapter() if disable_adapter else contextlib.nullcontext()
        with adapter_context, torch.inference_mode():
            teacher_hidden = backbone(
                input_ids=input_ids,
                attention_mask=None,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        frequency_switcher.install("student")
        with torch.inference_mode():
            candidate_hidden = backbone(
                input_ids=input_ids,
                attention_mask=None,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        loss, bucket_losses = normalized_bucket_hidden_mse(
            candidate_hidden,
            teacher_hidden,
            attention_mask,
            position_bucket_ranges(input_ids.shape[1]),
            assume_all_tokens_valid=True,
        )
        loss_value = float(loss.cpu())
        bucket_values = bucket_losses.cpu().tolist()
        if not math.isfinite(loss_value) or not all(
            math.isfinite(value) for value in bucket_values
        ):
            raise RuntimeError("non-finite hidden representation error")
        losses.append(loss_value)
        bucket_history.append(bucket_values)
        batch_weights.append(input_ids.shape[0])
    frequency_switcher.install("student")
    if not losses:
        raise ValueError("hidden-error evaluation processed no validation batches")
    return {
        "normalized_mse": round(float(np.average(losses, weights=batch_weights)), 8),
        "bucket_normalized_mse": [
            round(float(value), 8)
            for value in np.average(
                np.asarray(bucket_history), axis=0, weights=batch_weights
            )
        ],
        "batches": len(losses),
        "sequences": sequences,
        "batch_size": batch_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--candidate_method",
        choices=("native_geo", "evq_cosh"),
        required=True,
    )
    parser.add_argument("--tau", type=float, default=1.414)
    parser.add_argument("--adapter_dir", type=Path, default=None)
    parser.add_argument("--claim_log_dir", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--wikitext_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ppl_lengths", default="8192,16384,32768")
    parser.add_argument("--ppl_chunks", type=int, default=5)
    parser.add_argument("--lm_head_chunk_tokens", type=int, default=2048)
    parser.add_argument("--hidden_sequences", type=int, default=128)
    parser.add_argument("--hidden_batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("the approved evaluator requires seed=42")
    if args.candidate_method == "evq_cosh" and args.tau != 1.414:
        raise ValueError("the approved evaluator requires EVQ tau=1.414")
    ppl_lengths = tuple(int(item) for item in args.ppl_lengths.split(","))
    validate_eval_claim_protocol(
        ppl_lengths=ppl_lengths,
        ppl_chunks=args.ppl_chunks,
        hidden_sequences=args.hidden_sequences,
        hidden_batch_size=args.hidden_batch_size,
        bf16=args.bf16,
    )
    validate_variant_request(args.variant, args.candidate_method, args.adapter_dir)
    runtime = validate_single_gpu_runtime()
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_distill_manifest(manifest, args.data_dir)
    manifest_hash = sha256_file(manifest_path)
    base_model_fingerprint = fingerprint_model_source(args.model_name)
    validation_path = args.data_dir / manifest["files"]["validation"]["name"]
    validation_tensor = torch.load(
        validation_path,
        map_location="cpu",
        weights_only=True,
    )

    (
        model,
        tokenizer,
        candidate_inv_freq,
        provenance,
        adapter_metadata,
        adapter_run_protocol,
        claim_ready_sha256,
    ) = load_candidate(
        model_name=args.model_name,
        candidate_method=args.candidate_method,
        tau=args.tau,
        adapter_dir=args.adapter_dir,
        bf16=args.bf16,
        data_manifest_sha256=manifest_hash,
        base_model_fingerprint=base_model_fingerprint,
        claim_log_dir=args.claim_log_dir,
    )
    tokenizer_identity = tokenizer_fingerprint(args.model_name)
    if manifest["tokenizer"] != tokenizer_identity:
        raise RuntimeError("evaluation tokenizer fingerprint does not match frozen data")
    config = model.config
    geometry = resolve_model_rope_geometry(config)
    teacher_inv_freq, _ = build_training_inv_freq(
        rope_method="native_geo",
        head_dim=geometry.head_dim,
        base=geometry.rope_base,
        tau=0.0,
    )

    started = time.time()
    ppl, eval_corpus_token_count = evaluate_ppl(
        model=model,
        tokenizer=tokenizer,
        text_path=args.wikitext_path,
        lengths=ppl_lengths,
        chunks=args.ppl_chunks,
        lm_head_chunk_tokens=args.lm_head_chunk_tokens,
    )
    verify_model_inv_freq(model, candidate_inv_freq)
    hidden_error = evaluate_hidden_error(
        model=model,
        validation_tensor=validation_tensor,
        candidate_inv_freq=candidate_inv_freq,
        teacher_inv_freq=teacher_inv_freq,
        sequences=args.hidden_sequences,
        batch_size=args.hidden_batch_size,
    )
    output = {
        "format_version": 1,
        "variant": args.variant,
        "seed": args.seed,
        "model": public_model_identifier(args.model_name),
        "base_model_fingerprint": base_model_fingerprint,
        "adapter": (
            public_artifact_identifier(args.adapter_dir)
            if args.adapter_dir is not None
            else None
        ),
        "adapter_sha256": (
            sha256_file(args.adapter_dir / "adapter_model.safetensors")
            if args.adapter_dir is not None
            else None
        ),
        "adapter_protocol_validated": adapter_metadata is not None,
        "claim_ready_sha256": claim_ready_sha256,
        "adapter_run_protocol": (
            {
                "performance": adapter_run_protocol.get("performance"),
                "runtime": adapter_run_protocol.get("runtime"),
            }
            if adapter_run_protocol is not None
            else None
        ),
        "adapter_training_protocol": (
            {
                key: adapter_metadata.get(key)
                for key in (
                    "objective",
                    "seed",
                    "max_steps",
                    "warmup_steps",
                    "effective_batch_size",
                    "learning_rate",
                    "weight_decay",
                    "max_grad_norm",
                    "lora_r",
                    "lora_alpha",
                    "lora_dropout",
                    "lora_targets",
                    "data_manifest_sha256",
                )
            }
            if adapter_metadata is not None
            else None
        ),
        "candidate_method": args.candidate_method,
        "tau": args.tau if args.candidate_method == "evq_cosh" else None,
        "frequency_provenance": provenance,
        "data_manifest_sha256": manifest_hash,
        "eval_corpus": {
            "identifier": args.wikitext_path.name,
            "sha256": sha256_file(args.wikitext_path),
            "token_count": eval_corpus_token_count,
        },
        "tokenizer_fingerprint": tokenizer_identity,
        "eval_config": {
            "bf16": args.bf16,
            "ppl_lengths": list(ppl_lengths),
            "ppl_chunks": args.ppl_chunks,
            "lm_head_chunk_tokens": args.lm_head_chunk_tokens,
            "hidden_sequences": args.hidden_sequences,
            "hidden_batch_size": args.hidden_batch_size,
        },
        "runtime": runtime,
        "code_sha256": {
            "eval_positional_distill.py": sha256_file(Path(__file__)),
            "train_positional_distill.py": sha256_file(
                SCRIPT_DIR / "train_positional_distill.py"
            ),
            "train_evq_lora.py": sha256_file(SCRIPT_DIR / "train_evq_lora.py"),
            "prepare_positional_distill_data.py": sha256_file(
                SCRIPT_DIR / "prepare_positional_distill_data.py"
            ),
            "validate_checkpoint_artifact.py": sha256_file(
                SCRIPT_DIR / "validate_checkpoint_artifact.py"
            ),
        },
        "ppl": ppl,
        "hidden_error": hidden_error,
        "eval_time_min": round((time.time() - started) / 60.0, 3),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / result_filename(args.variant)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved {output_path.name}")


if __name__ == "__main__":
    main()
