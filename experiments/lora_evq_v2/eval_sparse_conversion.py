#!/usr/bin/env python3
"""Checkpoint-only EVQ attention-score and sparse-conversion experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from experiments.lora_evq_v2.eval_official_yarn_capability import (
    adapter_artifact_receipt,
    load_capability_suite,
    score_capability_prediction,
    score_generation_metrics,
    sha256_file,
    validate_adapter_identity,
)
from experiments.lora_evq_v2.prepare_legacy_model_manifest import (
    validate_model_manifest,
)
from experiments.lora_evq_v2.prepare_positional_distill_data import (
    tokenizer_source_fingerprint,
)
from experiments.lora_evq_v2.prepare_seed42_capability_data import (
    build_passkey_examples,
    load_jsonl,
)
from experiments.lora_evq_v2.train_positional_distill import causal_backbone
from experiments.lora_evq_v2.train_evq_lora import (
    build_training_inv_freq,
    compute_evq_cosh_inv_freq,
    find_rotary_modules,
    inject_inv_freq,
    load_frequency_artifact,
    resolve_model_rope_geometry,
    verify_model_inv_freq,
)


LEGACY_EXAMPLE_SCHEMA = "evq_cosh.seed42_capability_example.v1"
LEGACY_MANIFEST_SCHEMA = "evq_cosh.seed42_capability_manifest.v1"
PASSKEY_SHA256 = "21f365daed1b77e06b0a870ccb20f4965cba454c802f48dd48825c8f7ff2990d"
PASSKEY_SIZE_BYTES = 26_083_876
PASSKEY_ROWS = 300
PASSKEY_RECEIPT_SCHEMA = "evq_cosh.sparse_conversion_passkey.v1"
PHASE0_SCHEMA = "evq_cosh.lora_sparse_conversion_phase0.v1"
PHASE0_SUMMARY_SCHEMA = "evq_cosh.lora_sparse_conversion_phase0_summary.v1"
PHASE1_SCHEMA = "evq_cosh.lora_sparse_conversion_phase1.v1"
PHASE1_SUMMARY_SCHEMA = "evq_cosh.lora_sparse_conversion_phase1_summary.v1"
RAW_CAPABILITY_SCHEMA = "evq_cosh.lora_raw_capability_eval.v1"
CANARY_SCHEMA = "evq_cosh.lora_source_dependence_canary.v1"
CANARY_SUMMARY_SCHEMA = "evq_cosh.lora_source_dependence_canary_summary.v1"
ATTENTION_IMPL = "evq_exact_block"
SPARSE_CONFIG = {
    "block_size": 128,
    "top_blocks": 16,
    "local_window": 1024,
    "sink_tokens": 4,
}
MODEL_CONTRACT = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 8192,
    "rope_theta": 500000.0,
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prompt_sha256(prompt_ids: Sequence[int]) -> str:
    payload = json.dumps(list(prompt_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _script_sha256() -> str:
    return sha256_file(Path(__file__))


def _parse_ints(value: str | Iterable[int]) -> tuple[int, ...]:
    pieces = value.split(",") if isinstance(value, str) else value
    parsed = tuple(int(piece) for piece in pieces)
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def _find_subsequence(values: Sequence[int], needle: Sequence[int]) -> list[int]:
    if not needle:
        raise ValueError("needle tokenization is empty")
    return [
        index
        for index in range(len(values) - len(needle) + 1)
        if list(values[index : index + len(needle)]) == list(needle)
    ]


def _answer_ids(tokenizer: Any, answer: str) -> list[int]:
    ids = tokenizer(answer, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if not ids:
        raise ValueError("answer tokenization produced no tokens")
    return [int(token_id) for token_id in ids]


def _needle_span(tokenizer: Any, row: Mapping[str, Any]) -> tuple[int, int]:
    if row.get("task") != "passkey_retrieval" or len(row.get("answers", [])) != 1:
        raise ValueError("Phase 0 requires one deterministic passkey answer")
    needle_ids = _answer_ids(
        tokenizer,
        f"\nThe retrieval passkey is {row['answers'][0]}. Remember this exact passkey.\n",
    )
    matches = _find_subsequence(row["prompt_ids"], needle_ids)
    if len(matches) != 1:
        raise ValueError(
            f"{row.get('example_id')} has {len(matches)} exact needle spans; expected one"
        )
    return matches[0], matches[0] + len(needle_ids)


def _answer_span(tokenizer: Any, row: Mapping[str, Any]) -> tuple[int, int]:
    answer_ids = _answer_ids(tokenizer, str(row["answers"][0]))
    matches = _find_subsequence(row["prompt_ids"], answer_ids)
    if len(matches) != 1:
        raise ValueError(
            f"{row.get('example_id')} has {len(matches)} exact answer spans; expected one"
        )
    return matches[0], matches[0] + len(answer_ids)


def _legacy_passkey_rows(tokenizer: Any) -> list[dict[str, Any]]:
    rows = build_passkey_examples(tokenizer)
    legacy = []
    for row in rows:
        normalized = dict(row)
        normalized["schema"] = LEGACY_EXAMPLE_SCHEMA
        normalized.pop("generation_tokens", None)
        normalized.pop("scorer", None)
        legacy.append(normalized)
    return legacy


def prepare_passkey(args: argparse.Namespace) -> dict[str, Any]:
    reference = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    if reference.get("schema") != LEGACY_MANIFEST_SCHEMA:
        raise ValueError("reference capability manifest is not the frozen v1 manifest")
    expected = reference.get("files", {}).get("passkey.jsonl")
    if not isinstance(expected, Mapping):
        raise ValueError("reference manifest has no passkey.jsonl record")
    registered = {
        "sha256": PASSKEY_SHA256,
        "size_bytes": PASSKEY_SIZE_BYTES,
        "row_count": PASSKEY_ROWS,
    }
    for key, value in registered.items():
        if expected.get(key) != value:
            raise ValueError(f"reference passkey {key} differs from the preregistered value")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=Path(args.tokenizer).is_dir(),
    )
    rows = _legacy_passkey_rows(tokenizer)
    if len(rows) != PASSKEY_ROWS:
        raise ValueError(f"rebuilt {len(rows)} passkey rows; expected {PASSKEY_ROWS}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "passkey.jsonl"
    if output.exists() or (args.output_dir / "manifest.json").exists():
        raise FileExistsError(args.output_dir)
    temporary = output.with_suffix(".jsonl.incomplete")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    actual = {
        "sha256": sha256_file(temporary),
        "size_bytes": temporary.stat().st_size,
        "row_count": len(rows),
    }
    if actual != registered:
        raise ValueError(f"rebuilt passkey artifact mismatch: {actual}")
    os.replace(temporary, output)

    receipt = {
        "schema": PASSKEY_RECEIPT_SCHEMA,
        "source_manifest_sha256": sha256_file(args.reference_manifest),
        "source_manifest_schema": reference["schema"],
        "example_schema": LEGACY_EXAMPLE_SCHEMA,
        "file": {"name": output.name, **actual},
        "tokenizer": reference.get("tokenizer"),
        "selection": {
            "lengths": [8192, 16384, 32768],
            "depths": [10, 25, 50, 75, 90],
            "trials_per_cell": 20,
            "seed": 42,
        },
    }
    _atomic_json(args.output_dir / "manifest.json", receipt)
    return receipt


def load_passkey_rows(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PASSKEY_RECEIPT_SCHEMA:
        raise ValueError("passkey receipt schema mismatch")
    record = manifest.get("file")
    if not isinstance(record, Mapping) or record.get("name") != "passkey.jsonl":
        raise ValueError("passkey receipt has an unsafe file record")
    path = root / "passkey.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": sum(1 for line in path.open(encoding="utf-8") if line.strip()),
    }
    expected = {key: record.get(key) for key in actual}
    if actual != expected or actual != {
        "sha256": PASSKEY_SHA256,
        "size_bytes": PASSKEY_SIZE_BYTES,
        "row_count": PASSKEY_ROWS,
    }:
        raise ValueError("passkey artifact does not match the frozen v1 file")
    rows = load_jsonl(path)
    for row in rows:
        if row.get("schema") != LEGACY_EXAMPLE_SCHEMA:
            raise ValueError("passkey example schema mismatch")
        if row.get("task") != "passkey_retrieval":
            raise ValueError("passkey file contains another task")
        if _prompt_sha256(row["prompt_ids"]) != row.get("prompt_sha256"):
            raise ValueError(f"passkey prompt hash mismatch: {row.get('example_id')}")
        if len(row["prompt_ids"]) != int(row["target_length"]):
            raise ValueError(f"passkey prompt length mismatch: {row.get('example_id')}")
    return manifest, rows


def _validate_config(model_name: str) -> dict[str, Any]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=Path(model_name).is_dir(),
    )
    rope = getattr(config, "rope_parameters", None)
    if rope is None:
        rope = getattr(config, "rope_scaling", None)
    rope = dict(rope) if isinstance(rope, Mapping) else {}
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = rope.get("rope_theta")
    observed = {
        key: (rope_theta if key == "rope_theta" else getattr(config, key, None))
        for key in MODEL_CONTRACT
    }
    if observed != MODEL_CONTRACT:
        raise ValueError(f"model architecture contract mismatch: {observed}")
    if rope and (rope.get("rope_type", "default") != "default" or set(rope) - {"rope_type", "rope_theta"}):
        raise ValueError(f"raw extrapolation requires default unscaled RoPE, found {rope}")
    return observed


def _validate_arm(
    *,
    adapter_dir: Path,
    substrate: str,
    training_manifest: Path,
    model_manifest: Path,
) -> dict[str, Any]:
    training_hash = sha256_file(training_manifest)
    model_manifest_hash = sha256_file(model_manifest)
    metadata = validate_adapter_identity(
        adapter_dir,
        substrate=substrate,
        training_manifest_sha256=training_hash,
    )
    if metadata.get("model_manifest_sha256") != model_manifest_hash:
        raise ValueError("adapter model-manifest hash mismatch")
    inv_freq, frequency_record, provenance = load_frequency_artifact(
        adapter_dir / "custom_inv_freq.pt",
        expected_method=substrate,
    )
    head_dim = int(frequency_record.get("head_dim", 2 * inv_freq.numel()))
    base = float(frequency_record.get("base", MODEL_CONTRACT["rope_theta"]))
    canonical, _ = build_training_inv_freq(
        rope_method=substrate,
        head_dim=head_dim,
        base=base,
        tau=1.414,
    )
    if not torch.allclose(
        inv_freq.to(torch.float64), canonical.to(torch.float64), rtol=0.0, atol=1e-12
    ):
        raise ValueError("frequency artifact differs from the canonical substrate")
    return {
        "substrate": substrate,
        "adapter_sha256": metadata["adapter_sha256"],
        "frequency": provenance,
        "model_manifest_sha256": model_manifest_hash,
        "training_manifest_sha256": training_hash,
        "protocol_sha256": _json_sha256(metadata.get("protocol")),
        "code_sha256": metadata.get("code_sha256"),
        "metadata": metadata,
    }


def _validate_stage2_arm(
    *,
    adapter_dir: Path,
    substrate: str,
    training_manifest: Path,
    model_manifest: Path,
) -> dict[str, Any]:
    metadata_path = adapter_dir / "stage2_meta.json"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    for path in (metadata_path, adapter_path, config_path, adapter_dir / "custom_inv_freq.pt"):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "evq_cosh.lora_stage2_retrieval.v1":
        raise ValueError("stage2 metadata schema mismatch")
    expected = {
        "status": "complete",
        "substrate": substrate,
        "max_steps": 50,
        "seed": 42,
        "objective": "answer_only_chat_causal_lm",
        "training_manifest_sha256": sha256_file(training_manifest),
        "model_manifest_sha256": sha256_file(model_manifest),
        "adapter_sha256": sha256_file(adapter_path),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"stage2 metadata mismatch at {key}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("r", -1)) != 64 or int(config.get("lora_alpha", -1)) != 128:
        raise ValueError("stage2 adapter rank/alpha mismatch")
    if set(config.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj"}:
        raise ValueError("stage2 adapter target modules mismatch")
    inv_freq, _, provenance = load_frequency_artifact(
        adapter_dir / "custom_inv_freq.pt", expected_method=substrate
    )
    canonical, _ = build_training_inv_freq(
        rope_method=substrate,
        head_dim=2 * inv_freq.numel(),
        base=MODEL_CONTRACT["rope_theta"],
        tau=1.414,
    )
    if not torch.allclose(inv_freq.to(torch.float64), canonical, rtol=0.0, atol=1e-12):
        raise ValueError("stage2 frequency artifact differs from the canonical substrate")
    return {
        "substrate": substrate,
        "adapter_sha256": metadata["adapter_sha256"],
        "frequency": provenance,
        "model_manifest_sha256": metadata["model_manifest_sha256"],
        "training_manifest_sha256": metadata["training_manifest_sha256"],
        "stage2_metadata": metadata,
    }


def _tokenizer_identity_matches(
    recorded: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    identifier = recorded.get("identifier") or recorded.get("requested") or recorded.get("name_or_path")
    if identifier is not None:
        identifier = Path(str(identifier)).name
    return identifier == expected.get("identifier") and recorded.get("files") == expected.get("files")


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    config = _validate_config(args.model_name)
    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    validate_model_manifest(Path(args.model_name), model_manifest, verify_hashes=False)
    passkey_manifest, rows = load_passkey_rows(args.passkey_root)

    expected_tokenizer = tokenizer_source_fingerprint(args.model_name)
    recorded_tokenizer = passkey_manifest.get("tokenizer", {})
    if not _tokenizer_identity_matches(recorded_tokenizer, expected_tokenizer):
        raise ValueError("rebuilt passkey tokenizer differs from the model tokenizer")

    geo = _validate_arm(
        adapter_dir=args.geo_adapter,
        substrate="native_geo",
        training_manifest=args.training_data_manifest,
        model_manifest=args.model_manifest,
    )
    evq = _validate_arm(
        adapter_dir=args.evq_adapter,
        substrate="evq_cosh",
        training_manifest=args.training_data_manifest,
        model_manifest=args.model_manifest,
    )
    for key in ("model_manifest_sha256", "training_manifest_sha256", "code_sha256"):
        if geo[key] != evq[key]:
            raise ValueError(f"matched-arm identity differs at {key}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=Path(args.model_name).is_dir(),
    )
    spans = []
    answer_spans = []
    for row in rows:
        start, end = _needle_span(tokenizer, row)
        spans.append(end - start)
        answer_start, answer_end = _answer_span(tokenizer, row)
        if not start <= answer_start < answer_end <= end:
            raise ValueError(f"answer span is outside the needle: {row['example_id']}")
        answer_spans.append(answer_end - answer_start)

    suite_manifest, suite_rows = load_capability_suite(args.suite_root)
    receipt = {
        "schema": "evq_cosh.lora_sparse_conversion_dry_run.v1",
        "status": "pass",
        "model_contract": config,
        "passkey": {
            "manifest_sha256": sha256_file(args.passkey_root / "manifest.json"),
            "file_sha256": PASSKEY_SHA256,
            "rows": len(rows),
            "needle_span_tokens": {"min": min(spans), "max": max(spans)},
            "answer_span_tokens": {"min": min(answer_spans), "max": max(answer_spans)},
        },
        "capability_suite": {
            "manifest_sha256": sha256_file(args.suite_root / "manifest.json"),
            "rows": len(suite_rows),
            "task_counts": suite_manifest.get("task_counts"),
        },
        "arms": {
            "native_geo": {key: value for key, value in geo.items() if key != "metadata"},
            "evq_cosh": {key: value for key, value in evq.items() if key != "metadata"},
        },
        "sparse_config": SPARSE_CONFIG,
        "phase0_selection": {
            "trials": [0, 1],
            "lengths": [8192, 16384, 32768],
            "cases_per_arm": 30,
        },
        "script_sha256": _script_sha256(),
    }
    _atomic_json(args.output, receipt)
    return receipt


def _attention_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if all(
            hasattr(module, attr)
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim", "layer_idx")
        )
    ]
    expected = int(getattr(model.config, "num_hidden_layers", -1))
    if len(modules) != expected:
        raise RuntimeError(f"found {len(modules)} attention modules; expected {expected}")
    return modules


def _repeat_kv(hidden_states: torch.Tensor, repetitions: int) -> torch.Tensor:
    return hidden_states.repeat_interleave(int(repetitions), dim=1)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    while cos.ndim < value.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return value * cos + _rotate_half(value) * sin


def _static_keep(length: int, config: Mapping[str, int], device: torch.device) -> torch.Tensor:
    keep = torch.zeros(length, dtype=torch.bool, device=device)
    keep[: min(length, int(config["sink_tokens"]))] = True
    keep[max(0, length - int(config["local_window"])) :] = True
    return keep


def _block_layout(
    scores: torch.Tensor,
    config: Mapping[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if scores.ndim != 4 or scores.shape[-2] != 1:
        raise ValueError("block selector requires [batch, heads, 1, keys] scores")
    length = scores.shape[-1]
    block_size = int(config["block_size"])
    padded_length = math.ceil(length / block_size) * block_size
    static = _static_keep(length, config, scores.device)
    remote = ~static
    padded_remote = F.pad(remote, (0, padded_length - length), value=False)
    valid_blocks = padded_remote.view(-1, block_size).any(dim=-1)
    masked = scores.masked_fill(~remote.view(1, 1, 1, -1), float("-inf"))
    masked = F.pad(masked, (0, padded_length - length), value=float("-inf"))
    block_scores = masked.view(*scores.shape[:-1], -1, block_size).amax(dim=-1)
    return static, valid_blocks, block_scores, padded_length


def _sparse_keep_mask(
    scores: torch.Tensor,
    *,
    mode: str,
    config: Mapping[str, int],
) -> torch.Tensor:
    if mode in {"dense", "full"}:
        return torch.ones_like(scores, dtype=torch.bool)
    if mode not in {"score", "fixed"}:
        raise ValueError(f"unsupported attention mode: {mode}")
    static, valid_blocks, block_scores, padded_length = _block_layout(scores, config)
    block_size = int(config["block_size"])
    candidate_ids = torch.nonzero(valid_blocks, as_tuple=False).flatten()
    if candidate_ids.numel() == 0:
        return static.view(1, 1, 1, -1).expand_as(scores)
    budget = min(int(config["top_blocks"]), int(candidate_ids.numel()))
    selected = torch.zeros_like(block_scores, dtype=torch.bool)
    if mode == "score":
        available = block_scores.masked_fill(
            ~valid_blocks.view(1, 1, 1, -1), float("-inf")
        )
        indices = available.topk(budget, dim=-1).indices
        selected.scatter_(-1, indices, True)
    else:
        count = int(candidate_ids.numel())
        offsets = [min(count - 1, int((index + 0.5) * count / budget)) for index in range(budget)]
        fixed_ids = candidate_ids[torch.tensor(offsets, device=scores.device)]
        selected[..., fixed_ids] = True
    selected_tokens = selected.repeat_interleave(block_size, dim=-1)[..., : scores.shape[-1]]
    static_tokens = static.view(1, 1, 1, -1).expand_as(scores)
    if selected_tokens.shape[-1] != scores.shape[-1] or padded_length < scores.shape[-1]:
        raise AssertionError("block selection changed the token index space")
    return selected_tokens | static_tokens


def exact_block_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference decode attention; it masks original-position KV without gathering."""
    if query.shape[-2] != 1:
        raise RuntimeError("exact-score attention is decode-only; dense prefill must use SDPA")
    config = getattr(module, "_evq_sparse_config", None)
    if not isinstance(config, Mapping):
        raise RuntimeError("attention module has no sparse-conversion configuration")
    repetitions = query.shape[1] // key.shape[1]
    if repetitions <= 0 or query.shape[1] != key.shape[1] * repetitions:
        raise RuntimeError("query/KV head geometry is not an integer GQA mapping")
    key_states = _repeat_kv(key, repetitions)
    value_states = _repeat_kv(value, repetitions)
    weights = torch.matmul(query, key_states.transpose(2, 3)) * float(scaling)
    if attention_mask is not None:
        weights = weights + attention_mask
    keep = _sparse_keep_mask(weights, mode=str(config["mode"]), config=config)
    weights = weights.masked_fill(~keep, torch.finfo(weights.dtype).min)
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = F.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
    return output, weights


def _register_attention() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(ATTENTION_IMPL, exact_block_attention_forward)


def _set_attention_mode(
    model: torch.nn.Module,
    implementation: str,
    *,
    mode: str = "dense",
    config: Mapping[str, int] = SPARSE_CONFIG,
) -> None:
    for _, module in _attention_modules(model):
        module.config._attn_implementation = implementation
        module._evq_sparse_config = {**config, "mode": mode}
    model.config._attn_implementation = implementation


def _load_arm_model(args: argparse.Namespace) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU phase requires CUDA")
    validator = _validate_stage2_arm if getattr(args, "stage2", False) else _validate_arm
    identity = validator(
        adapter_dir=args.adapter_dir,
        substrate=args.substrate,
        training_manifest=args.training_data_manifest,
        model_manifest=args.model_manifest,
    )
    _validate_config(args.model_name)
    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    validate_model_manifest(Path(args.model_name), model_manifest, verify_hashes=False)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local = Path(args.model_name).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=local,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=local,
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.to(torch.device("cuda"))
    model.eval()
    model.config.use_cache = True

    inv_freq, _, _ = load_frequency_artifact(
        args.adapter_dir / "custom_inv_freq.pt",
        expected_method=args.substrate,
    )
    geometry = resolve_model_rope_geometry(model.config)
    canonical, _ = build_training_inv_freq(
        rope_method=args.substrate,
        head_dim=geometry.head_dim,
        base=geometry.rope_base,
        tau=1.414,
    )
    if not torch.allclose(inv_freq.to(torch.float64), canonical, rtol=0.0, atol=1e-12):
        raise ValueError("runtime frequency artifact differs from canonical")
    inject_inv_freq(model, inv_freq)
    for name, module in find_rotary_modules(model):
        if not hasattr(module, "attention_scaling"):
            raise RuntimeError(f"rotary module {name} has no attention_scaling")
        module.attention_scaling = 1.0
        original = getattr(module, "original_inv_freq", None)
        if torch.is_tensor(original):
            if original.shape != module.inv_freq.shape:
                raise RuntimeError(f"original_inv_freq shape mismatch at {name}")
            original.copy_(module.inv_freq)
    verify_model_inv_freq(model, inv_freq)
    if getattr(args, "stage2", False):
        identity["adapter_artifact_receipt"] = {
            "format_version": 1,
            "files": {
                name: {
                    "sha256": sha256_file(args.adapter_dir / name),
                    "size_bytes": (args.adapter_dir / name).stat().st_size,
                }
                for name in (
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "custom_inv_freq.pt",
                    "stage2_meta.json",
                )
            },
        }
    else:
        identity["adapter_artifact_receipt"] = adapter_artifact_receipt(args.adapter_dir)
    return model, tokenizer, identity


def _probe_metrics(
    scores: torch.Tensor,
    *,
    needle_start: int,
    needle_end: int,
    config: Mapping[str, int],
) -> dict[str, Any]:
    if scores.ndim != 2:
        raise ValueError("probe scores must have [heads, keys] shape")
    expanded = scores.unsqueeze(0).unsqueeze(2)
    static, valid_blocks, block_scores, _ = _block_layout(expanded, config)
    block_scores = block_scores[0, :, 0]
    block_size = int(config["block_size"])
    gold_blocks = sorted(set(range(needle_start // block_size, (needle_end - 1) // block_size + 1)))
    gold_remote = [block for block in gold_blocks if bool(valid_blocks[block])]
    dense_weights = F.softmax(scores.float(), dim=-1)
    dense_mass = dense_weights[:, needle_start:needle_end].sum(dim=-1)

    if gold_remote:
        gold_score = block_scores[:, gold_remote].amax(dim=-1)
        ranks = 1 + (block_scores > gold_score.unsqueeze(-1)).logical_and(
            valid_blocks.unsqueeze(0)
        ).sum(dim=-1)
        distractor_mask = valid_blocks.clone()
        distractor_mask[gold_remote] = False
        distractor = block_scores.masked_fill(~distractor_mask.unsqueeze(0), float("-inf")).amax(dim=-1)
        margin = gold_score - distractor
    else:
        ranks = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        margin = torch.full((scores.shape[0],), float("nan"), device=scores.device)

    keep16 = _sparse_keep_mask(expanded, mode="score", config=config)[0, :, 0]
    sparse_scores = scores.masked_fill(~keep16, torch.finfo(scores.dtype).min)
    sparse_weights = F.softmax(sparse_scores.float(), dim=-1)
    sparse_mass = sparse_weights[:, needle_start:needle_end].sum(dim=-1)
    mass_gain = sparse_mass / dense_mass.clamp_min(1e-30)
    full_lse = torch.logsumexp(scores.float(), dim=-1)
    kept_lse = torch.logsumexp(sparse_scores.float(), dim=-1)
    remote = ~static
    if bool(remote[needle_start:needle_end].any()):
        gold_token_score = scores[:, needle_start:needle_end].amax(dim=-1)
        token_rank = 1 + (scores > gold_token_score.unsqueeze(-1)).logical_and(
            remote.unsqueeze(0)
        ).sum(dim=-1)
        remote_tokens = int(remote.sum())
        token_percentile = 1.0 - (token_rank.float() - 1.0) / max(1, remote_tokens)
    else:
        token_rank = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        token_percentile = torch.full((scores.shape[0],), float("nan"), device=scores.device)

    hits: dict[str, list[bool]] = {}
    for budget in (8, 16, 32):
        selected_config = {**config, "top_blocks": budget}
        keep = _sparse_keep_mask(expanded, mode="score", config=selected_config)[0, :, 0]
        hits[str(budget)] = [
            bool(value)
            for value in keep[:, needle_start:needle_end].any(dim=-1).detach().cpu().tolist()
        ]
    return {
        "block_rank": [int(value) if gold_remote else None for value in ranks.detach().cpu().tolist()],
        "answer_token_rank": [
            int(value) if gold_remote else None for value in token_rank.detach().cpu().tolist()
        ],
        "answer_token_percentile": [
            float(value) if gold_remote else None for value in token_percentile.detach().cpu().tolist()
        ],
        "hit_at": hits,
        "margin": [float(value) if gold_remote else None for value in margin.detach().cpu().tolist()],
        "dense_answer_mass": [float(value) for value in dense_mass.detach().cpu().tolist()],
        "sparse_answer_mass_at_16": [float(value) for value in sparse_mass.detach().cpu().tolist()],
        "mass_gain_at_16": [float(value) for value in mass_gain.detach().cpu().tolist()],
        "removed_tail_log_normalizer": [
            float(value) for value in (full_lse - kept_lse).detach().cpu().tolist()
        ],
        "gold_is_static": not gold_remote,
        "gold_blocks": gold_blocks,
    }


@torch.inference_mode()
def _probe_one(
    model: torch.nn.Module,
    input_ids: Sequence[int],
    *,
    needle_start: int,
    needle_end: int,
    config: Mapping[str, int],
) -> list[dict[str, Any]]:
    device = torch.device("cuda")
    tensor = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(tensor)
    captured: dict[int, dict[str, Any]] = {}
    hooks = []

    def make_hook(layer_index: int):
        def hook(module: torch.nn.Module, positional: tuple[Any, ...], keyword: dict[str, Any]) -> None:
            hidden = positional[0] if positional else keyword.get("hidden_states")
            embeddings = keyword.get("position_embeddings")
            if not torch.is_tensor(hidden) or not isinstance(embeddings, tuple):
                raise RuntimeError("attention hook did not receive hidden states and position embeddings")
            cos, sin = embeddings
            head_dim = int(module.head_dim)
            query = module.q_proj(hidden[:, -1:, :]).view(1, 1, -1, head_dim).transpose(1, 2)
            key = module.k_proj(hidden).view(1, hidden.shape[1], -1, head_dim).transpose(1, 2)
            query = _apply_rotary(query, cos[:, -1:, :], sin[:, -1:, :])
            key = _apply_rotary(key, cos, sin)
            if query.shape[1] % key.shape[1] != 0:
                raise RuntimeError("Phase 0 found a non-integer GQA head mapping")
            key = _repeat_kv(key, query.shape[1] // key.shape[1])
            scores = torch.matmul(query, key.transpose(2, 3))[0, :, 0] * float(module.scaling)
            captured[layer_index] = {
                "layer": layer_index,
                **_probe_metrics(
                    scores,
                    needle_start=needle_start,
                    needle_end=needle_end,
                    config=config,
                ),
            }

        return hook

    for _, module in _attention_modules(model):
        hooks.append(module.register_forward_pre_hook(make_hook(int(module.layer_idx)), with_kwargs=True))
    try:
        causal_backbone(model)(
            input_ids=tensor,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for hook in hooks:
            hook.remove()
    if sorted(captured) != list(range(int(model.config.num_hidden_layers))):
        raise RuntimeError("Phase 0 did not capture every attention layer")
    del tensor, attention_mask
    return [captured[index] for index in sorted(captured)]


def _select_passkey_rows(
    rows: Sequence[dict[str, Any]],
    *,
    lengths: Sequence[int],
    trials: Sequence[int] | None,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if int(row["target_length"]) in set(lengths)
        and (trials is None or int(row.get("source", {}).get("trial", -1)) in set(trials))
    ]
    expected = len(lengths) * 5 * (20 if trials is None else len(trials))
    if len(selected) != expected:
        raise ValueError(f"selected {len(selected)} passkey rows; expected {expected}")
    return selected


def run_phase0(args: argparse.Namespace) -> dict[str, Any]:
    _, all_rows = load_passkey_rows(args.passkey_root)
    lengths = _parse_ints(args.lengths)
    trials = _parse_ints(args.trials)
    rows = _select_passkey_rows(all_rows, lengths=lengths, trials=trials)
    model, tokenizer, identity = _load_arm_model(args)
    _set_attention_mode(model, "sdpa")
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    results = []
    for index, row in enumerate(rows, start=1):
        needle_start, needle_end = _needle_span(tokenizer, row)
        answer_start, answer_end = _answer_span(tokenizer, row)
        if not needle_start <= answer_start < answer_end <= needle_end:
            raise ValueError(f"answer span is outside the needle: {row['example_id']}")
        layers = _probe_one(
            model,
            row["prompt_ids"],
            needle_start=answer_start,
            needle_end=answer_end,
            config=SPARSE_CONFIG,
        )
        results.append(
            {
                "example_id": row["example_id"],
                "target_length": int(row["target_length"]),
                "depth_percent": float(row["depth_percent"]),
                "trial": int(row["source"]["trial"]),
                "prompt_sha256": row["prompt_sha256"],
                "needle_span": [needle_start, needle_end],
                "answer_span": [answer_start, answer_end],
                "layers": layers,
            }
        )
        print(
            json.dumps(
                {
                    "phase": 0,
                    "substrate": args.substrate,
                    "progress": f"{index}/{len(rows)}",
                    "length": row["target_length"],
                    "depth": row["depth_percent"],
                    "trial": row["source"]["trial"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    output = {
        "schema": PHASE0_SCHEMA,
        "substrate": args.substrate,
        "single_seed_supporting": True,
        "raw_extrapolation": True,
        "adapter": identity,
        "passkey_sha256": PASSKEY_SHA256,
        "selection": {"lengths": list(lengths), "trials": list(trials)},
        "sparse_config": SPARSE_CONFIG,
        "query_contract": "last_prompt_token_predicting_first_answer_token",
        "script_sha256": _script_sha256(),
        "results": results,
        "runtime": {
            "seconds": time.time() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        },
    }
    _atomic_json(args.output, output)
    return output


def _phase0_cases(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if document.get("schema") != PHASE0_SCHEMA:
        raise ValueError("Phase 0 raw schema mismatch")
    cases = {row["example_id"]: row for row in document["results"]}
    if len(cases) != len(document["results"]):
        raise ValueError("Phase 0 output has duplicate example IDs")
    return cases


def _head_values(case: Mapping[str, Any], key: str) -> dict[tuple[int, int], Any]:
    output = {}
    for layer in case["layers"]:
        values = layer[key]
        for head, value in enumerate(values):
            output[(int(layer["layer"]), head)] = value
    return output


def _phase0_gate_checks(lengths: Mapping[str, Any], gate_lengths: tuple[int, ...]) -> dict[str, bool]:
    if gate_lengths == (16384,):
        cell = lengths["16384"]
        return {
            "16k_evq_wins_at_least_8_of_10": cell["evq_wins"] >= 8,
            "16k_median_hit_delta_positive": cell["median_hit_delta_evq_minus_geo"] > 0,
            "16k_median_evq_hit_at_16_at_least_0p50": cell["median_evq_hit_at_16"] >= 0.50,
            "16k_median_geo_hit_at_16_at_most_0p25": cell["median_geo_hit_at_16"] <= 0.25,
        }
    if gate_lengths != (16384, 32768):
        raise ValueError("gate lengths must be 16384 or 16384,32768")
    return {
        "32k_evq_wins_at_least_8_of_10": lengths["32768"]["evq_wins"] >= 8,
        "32k_median_hit_delta_positive": lengths["32768"]["median_hit_delta_evq_minus_geo"] > 0,
        "32k_evq_mass_gain_at_least_1p5": lengths["32768"]["median_evq_mass_gain_at_16"] >= 1.5,
        "16k_no_hit_reversal": lengths["16384"]["median_hit_delta_evq_minus_geo"] >= 0,
    }


def summarize_phase0(args: argparse.Namespace) -> dict[str, Any]:
    geo_document = json.loads(args.geo.read_text(encoding="utf-8"))
    evq_document = json.loads(args.evq.read_text(encoding="utf-8"))
    if geo_document.get("substrate") != "native_geo" or evq_document.get("substrate") != "evq_cosh":
        raise ValueError("Phase 0 summary received the wrong arms")
    if geo_document.get("selection") != evq_document.get("selection"):
        raise ValueError("Phase 0 arms used different selections")
    if geo_document.get("sparse_config") != evq_document.get("sparse_config"):
        raise ValueError("Phase 0 arms used different sparse configurations")
    geo = _phase0_cases(geo_document)
    evq = _phase0_cases(evq_document)
    if set(geo) != set(evq):
        raise ValueError("Phase 0 arms have different examples")

    reciprocal: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
    for cases in (geo, evq):
        for case in cases.values():
            if int(case["target_length"]) != 8192:
                continue
            for head, rank in _head_values(case, "block_rank").items():
                if rank is not None:
                    reciprocal[head].append(1.0 / float(rank))
    if len(reciprocal) != 32 * 32:
        raise ValueError("8K calibration did not cover every layer/head")
    ranked_heads = sorted(
        reciprocal,
        key=lambda head: (-statistics.fmean(reciprocal[head]), head[0], head[1]),
    )
    retrieval_heads = ranked_heads[:32]

    gate_lengths = _parse_ints(args.gate_lengths)
    lengths: dict[str, Any] = {}
    for length in gate_lengths:
        paired = []
        for example_id in sorted(geo):
            if int(geo[example_id]["target_length"]) != length:
                continue
            arm_values = {}
            for label, cases in (("geo", geo), ("evq", evq)):
                ranks = _head_values(cases[example_id], "block_rank")
                gains = _head_values(cases[example_id], "mass_gain_at_16")
                selected_ranks = [ranks[head] for head in retrieval_heads if ranks[head] is not None]
                selected_gains = [gains[head] for head in retrieval_heads if ranks[head] is not None]
                if len(selected_ranks) != len(retrieval_heads):
                    raise ValueError(f"{example_id} has a static gold block at a test length")
                arm_values[label] = {
                    "hit_at_16": sum(rank <= 16 for rank in selected_ranks) / len(selected_ranks),
                    "mass_gain_at_16": statistics.median(selected_gains),
                }
            paired.append(
                {
                    "example_id": example_id,
                    **arm_values,
                    "hit_delta_evq_minus_geo": arm_values["evq"]["hit_at_16"]
                    - arm_values["geo"]["hit_at_16"],
                }
            )
        if len(paired) != 10:
            raise ValueError(f"Phase 0 gate requires 10 paired cases at {length}, found {len(paired)}")
        deltas = [row["hit_delta_evq_minus_geo"] for row in paired]
        lengths[str(length)] = {
            "paired_cases": len(paired),
            "evq_wins": sum(value > 0 for value in deltas),
            "median_evq_hit_at_16": statistics.median(
                row["evq"]["hit_at_16"] for row in paired
            ),
            "median_geo_hit_at_16": statistics.median(
                row["geo"]["hit_at_16"] for row in paired
            ),
            "median_hit_delta_evq_minus_geo": statistics.median(deltas),
            "median_evq_mass_gain_at_16": statistics.median(
                row["evq"]["mass_gain_at_16"] for row in paired
            ),
            "median_geo_mass_gain_at_16": statistics.median(
                row["geo"]["mass_gain_at_16"] for row in paired
            ),
            "cases": paired,
        }
    checks = _phase0_gate_checks(lengths, gate_lengths)
    output = {
        "schema": PHASE0_SUMMARY_SCHEMA,
        "status": "pass" if all(checks.values()) else "stop",
        "scope": "exploratory_16k" if gate_lengths == (16384,) else "preregistered_16k_32k",
        "checks": checks,
        "retrieval_head_contract": {
            "selection_length": 8192,
            "selection_metric": "pooled_geo_evq_mean_reciprocal_gold_block_rank",
            "count": len(retrieval_heads),
            "heads": [{"layer": layer, "head": head} for layer, head in retrieval_heads],
        },
        "lengths": lengths,
        "inputs": {"geo_sha256": sha256_file(args.geo), "evq_sha256": sha256_file(args.evq)},
        "single_seed_supporting": True,
    }
    _atomic_json(args.output, output)
    return output


def _configure_decode(model: torch.nn.Module, mode: str) -> None:
    _register_attention()
    _set_attention_mode(model, ATTENTION_IMPL, mode=mode)


@torch.inference_mode()
def _prefill(model: torch.nn.Module, prompt_ids: Sequence[int]) -> tuple[Any, int]:
    if len(prompt_ids) < 2:
        raise ValueError("decode experiment requires at least two prompt tokens")
    _set_attention_mode(model, "sdpa")
    device = torch.device("cuda")
    prefix = torch.tensor([list(prompt_ids[:-1])], dtype=torch.long, device=device)
    output = causal_backbone(model)(
        input_ids=prefix,
        attention_mask=torch.ones_like(prefix),
        use_cache=True,
        return_dict=True,
    )
    return output.past_key_values, int(prompt_ids[-1])


@torch.inference_mode()
def _decode_logits(
    model: torch.nn.Module,
    past: Any,
    token_id: int,
    *,
    seen_tokens: int,
) -> tuple[torch.Tensor, Any]:
    device = torch.device("cuda")
    token = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, seen_tokens + 1), dtype=torch.long, device=device)
    output = model(
        input_ids=token,
        attention_mask=attention_mask,
        past_key_values=past,
        use_cache=True,
        return_dict=True,
    )
    return output.logits[:, -1].float(), output.past_key_values


@torch.inference_mode()
def _answer_nll(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    answer_ids: Sequence[int],
    *,
    mode: str,
) -> dict[str, float | int]:
    past, current = _prefill(model, prompt_ids)
    _configure_decode(model, mode)
    seen = len(prompt_ids) - 1
    total = 0.0
    try:
        for label in answer_ids:
            logits, past = _decode_logits(model, past, current, seen_tokens=seen)
            target = torch.tensor([int(label)], dtype=torch.long, device=logits.device)
            total += float(F.cross_entropy(logits, target, reduction="sum").double().cpu())
            current = int(label)
            seen += 1
    finally:
        del past
    return {
        "nll_sum": total,
        "answer_tokens": len(answer_ids),
        "mean_logprob": -total / len(answer_ids),
    }


def _counterfactual_prompts(
    prompt_ids: Sequence[int],
    *,
    needle_span: tuple[int, int],
    answer_span: tuple[int, int],
    original_answer_ids: Sequence[int],
    swapped_answer_ids: Sequence[int],
) -> dict[str, list[int]]:
    """Build equal-length source-swapped and source-removed token prompts."""
    prompt = [int(value) for value in prompt_ids]
    needle_start, needle_end = needle_span
    answer_start, answer_end = answer_span
    original = [int(value) for value in original_answer_ids]
    swapped_answer = [int(value) for value in swapped_answer_ids]
    if not (0 <= needle_start <= answer_start < answer_end <= needle_end <= len(prompt)):
        raise ValueError("counterfactual spans are invalid")
    if prompt[answer_start:answer_end] != original:
        raise ValueError("registered answer span does not match the original answer")
    if len(original) != len(swapped_answer) or original == swapped_answer:
        raise ValueError("counterfactual answers must be distinct and token-length matched")

    swapped = list(prompt)
    swapped[answer_start:answer_end] = swapped_answer
    if _find_subsequence(swapped, original):
        raise ValueError("source-swapped prompt still contains the original answer")

    width = needle_end - needle_start
    filler = None
    for start in range(0, len(prompt) - width + 1):
        end = start + width
        if start < needle_end and needle_start < end:
            continue
        candidate = prompt[start:end]
        if _find_subsequence(candidate, original) or _find_subsequence(
            candidate, swapped_answer
        ):
            continue
        filler = candidate
        break
    if filler is None:
        raise ValueError("no answer-free equal-length filler span exists")
    removed = list(prompt)
    removed[needle_start:needle_end] = filler
    if _find_subsequence(removed, original) or _find_subsequence(removed, swapped_answer):
        raise ValueError("source-removed prompt still contains a registered answer")
    if not (len(prompt) == len(swapped) == len(removed)):
        raise AssertionError("counterfactual prompt lengths differ")
    return {"original": prompt, "swapped": swapped, "source_removed": removed}


def _mean_nll(score: Mapping[str, float | int]) -> float:
    return float(score["nll_sum"]) / int(score["answer_tokens"])


def _counterfactual_pairs(
    tokenizer: Any, rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["target_length"]), float(row["depth_percent"]))].append(row)
    pairs = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda row: int(row["source"]["trial"]))
        if len(group) != 2:
            raise ValueError(f"counterfactual canary requires two trials per cell: {key}")
        for row, swap_row in ((group[0], group[1]), (group[1], group[0])):
            original_ids = _answer_ids(tokenizer, str(row["answers"][0]))
            swapped_ids = _answer_ids(tokenizer, str(swap_row["answers"][0]))
            prompts = _counterfactual_prompts(
                row["prompt_ids"],
                needle_span=_needle_span(tokenizer, row),
                answer_span=_answer_span(tokenizer, row),
                original_answer_ids=original_ids,
                swapped_answer_ids=swapped_ids,
            )
            pairs.append(
                {
                    "example_id": row["example_id"],
                    "target_length": int(row["target_length"]),
                    "depth_percent": float(row["depth_percent"]),
                    "trial": int(row["source"]["trial"]),
                    "original_answer_ids": original_ids,
                    "swapped_answer_ids": swapped_ids,
                    "prompts": prompts,
                }
            )
    return pairs


def run_counterfactual_canary(args: argparse.Namespace) -> dict[str, Any]:
    _, all_rows = load_passkey_rows(args.passkey_root)
    rows = _select_passkey_rows(
        all_rows,
        lengths=_parse_ints(args.lengths),
        trials=_parse_ints(args.trials),
    )
    model, tokenizer, identity = _load_arm_model(args)
    if args.runtime_frequency != "trained":
        geometry = resolve_model_rope_geometry(model.config)
        if args.runtime_frequency == "native_geo":
            runtime_inv_freq, _ = build_training_inv_freq(
                rope_method="native_geo",
                head_dim=geometry.head_dim,
                base=geometry.rope_base,
                tau=1.414,
            )
        elif args.runtime_frequency == "midpoint_geo":
            runtime_inv_freq = compute_evq_cosh_inv_freq(
                head_dim=geometry.head_dim,
                base=geometry.rope_base,
                tau=0.0,
                midpoint=True,
            )
        else:
            runtime_inv_freq, _ = build_training_inv_freq(
                rope_method="evq_cosh",
                head_dim=geometry.head_dim,
                base=geometry.rope_base,
                tau=1.414,
            )
        inject_inv_freq(model, runtime_inv_freq)
        for name, module in find_rotary_modules(model):
            if not hasattr(module, "attention_scaling"):
                raise RuntimeError(f"rotary module {name} has no attention_scaling")
            module.attention_scaling = 1.0
            original = getattr(module, "original_inv_freq", None)
            if torch.is_tensor(original):
                if original.shape != module.inv_freq.shape:
                    raise RuntimeError(f"original_inv_freq shape mismatch at {name}")
                original.copy_(module.inv_freq)
        verify_model_inv_freq(model, runtime_inv_freq)
    pairs = _counterfactual_pairs(tokenizer, rows)
    if args.disable_adapter and not args.arm_name.startswith("base_"):
        raise ValueError("only base arms may disable the adapter")
    context = model.disable_adapter() if args.disable_adapter else nullcontext()
    results = []
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with context:
        for index, pair in enumerate(pairs, start=1):
            original_ids = pair["original_answer_ids"]
            swapped_ids = pair["swapped_answer_ids"]
            prompts = pair["prompts"]
            scores = {
                "original_own": _answer_nll(
                    model, prompts["original"], original_ids, mode="dense"
                ),
                "original_swapped": _answer_nll(
                    model, prompts["original"], swapped_ids, mode="dense"
                ),
                "swapped_own": _answer_nll(
                    model, prompts["swapped"], swapped_ids, mode="dense"
                ),
                "swapped_original": _answer_nll(
                    model, prompts["swapped"], original_ids, mode="dense"
                ),
                "removed_original": _answer_nll(
                    model, prompts["source_removed"], original_ids, mode="dense"
                ),
            }
            original_prefers_own = _mean_nll(scores["original_own"]) < _mean_nll(
                scores["original_swapped"]
            )
            swapped_prefers_own = _mean_nll(scores["swapped_own"]) < _mean_nll(
                scores["swapped_original"]
            )
            removal_delta = _mean_nll(scores["removed_original"]) - _mean_nll(
                scores["original_own"]
            )
            results.append(
                {
                    "example_id": pair["example_id"],
                    "target_length": pair["target_length"],
                    "depth_percent": pair["depth_percent"],
                    "trial": pair["trial"],
                    "prompt_sha256": {
                        name: _prompt_sha256(prompt) for name, prompt in prompts.items()
                    },
                    "scores": scores,
                    "original_prefers_own": original_prefers_own,
                    "swapped_prefers_own": swapped_prefers_own,
                    "pair_consistent": original_prefers_own and swapped_prefers_own,
                    "source_removal_delta_nll": removal_delta,
                    "source_removal_positive": removal_delta > 0.0,
                }
            )
            print(
                json.dumps(
                    {
                        "arm": args.arm_name,
                        "progress": f"{index}/{len(pairs)}",
                        "depth": pair["depth_percent"],
                        "pair_consistent": results[-1]["pair_consistent"],
                        "source_removal_delta_nll": removal_delta,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    aggregate = {
        "cases": len(results),
        "pair_consistency": statistics.fmean(
            float(row["pair_consistent"]) for row in results
        ),
        "source_removal_positive_fraction": statistics.fmean(
            float(row["source_removal_positive"]) for row in results
        ),
        "source_removal_delta_nll_mean": statistics.fmean(
            float(row["source_removal_delta_nll"]) for row in results
        ),
        "source_removal_delta_nll_median": statistics.median(
            float(row["source_removal_delta_nll"]) for row in results
        ),
        "original_answer_nll": statistics.fmean(
            _mean_nll(row["scores"]["original_own"]) for row in results
        ),
    }
    output = {
        "schema": CANARY_SCHEMA,
        "status": "complete",
        "arm": args.arm_name,
        "raw_extrapolation": True,
        "runtime_frequency": args.runtime_frequency,
        "adapter_enabled": not args.disable_adapter,
        "identity": identity,
        "selection": {
            "lengths": list(_parse_ints(args.lengths)),
            "trials": list(_parse_ints(args.trials)),
        },
        "aggregate": aggregate,
        "results": results,
        "single_seed_supporting": True,
        "script_sha256": _script_sha256(),
        "runtime": {
            "seconds": time.time() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        },
    }
    _atomic_json(args.output, output)
    return output


def summarize_counterfactual_canary(args: argparse.Namespace) -> dict[str, Any]:
    documents = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in (
            ("base_native", args.base_native),
            ("geo_lora_native", args.geo_lora_native),
            ("evq_lora_evq", args.evq_lora_evq),
        )
    }
    for name, document in documents.items():
        if document.get("schema") != CANARY_SCHEMA or document.get("arm") != name:
            raise ValueError(f"wrong canary arm: {name}")
    ids = [{row["example_id"] for row in document["results"]} for document in documents.values()]
    if ids[0] != ids[1] or ids[0] != ids[2]:
        raise ValueError("canary arms scored different examples")
    aggregate = {name: document["aggregate"] for name, document in documents.items()}
    checks = {
        "base_pair_consistency_at_least_0p60": aggregate["base_native"][
            "pair_consistency"
        ]
        >= 0.60,
        "geo_pair_consistency_at_least_0p60": aggregate["geo_lora_native"][
            "pair_consistency"
        ]
        >= 0.60,
        "base_source_removal_positive_at_least_0p60": aggregate["base_native"][
            "source_removal_positive_fraction"
        ]
        >= 0.60,
        "geo_source_removal_positive_at_least_0p60": aggregate["geo_lora_native"][
            "source_removal_positive_fraction"
        ]
        >= 0.60,
    }
    output = {
        "schema": CANARY_SUMMARY_SCHEMA,
        "status": "pass" if all(checks.values()) else "stop",
        "checks": checks,
        "aggregate": aggregate,
        "evq_minus_geo": {
            "pair_consistency": aggregate["evq_lora_evq"]["pair_consistency"]
            - aggregate["geo_lora_native"]["pair_consistency"],
            "source_removal_positive_fraction": aggregate["evq_lora_evq"][
                "source_removal_positive_fraction"
            ]
            - aggregate["geo_lora_native"]["source_removal_positive_fraction"],
            "source_removal_delta_nll_mean": aggregate["evq_lora_evq"][
                "source_removal_delta_nll_mean"
            ]
            - aggregate["geo_lora_native"]["source_removal_delta_nll_mean"],
            "original_answer_nll": aggregate["evq_lora_evq"]["original_answer_nll"]
            - aggregate["geo_lora_native"]["original_answer_nll"],
        },
        "inputs": {
            name: sha256_file(path)
            for name, path in (
                ("base_native", args.base_native),
                ("geo_lora_native", args.geo_lora_native),
                ("evq_lora_evq", args.evq_lora_evq),
            )
        },
        "single_seed_supporting": True,
    }
    _atomic_json(args.output, output)
    return output


@torch.inference_mode()
def _generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    *,
    mode: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    past, current = _prefill(model, prompt_ids)
    _configure_decode(model, mode)
    seen = len(prompt_ids) - 1
    generated = []
    eos = tokenizer.eos_token_id
    try:
        for _ in range(max_new_tokens):
            logits, past = _decode_logits(model, past, current, seen_tokens=seen)
            current = int(logits.argmax(dim=-1).item())
            generated.append(current)
            seen += 1
            if eos is not None and current == int(eos):
                break
    finally:
        del past
    eos_terminated = eos is not None and generated and generated[-1] == int(eos)
    content = generated[:-1] if eos_terminated else generated
    return {
        "prediction": tokenizer.decode(content, skip_special_tokens=True).strip(),
        "generated_ids": generated,
        "generated_token_count": len(content),
        "eos_terminated": eos_terminated,
    }


@torch.inference_mode()
def _first_step_logits(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    *,
    mode: str,
) -> torch.Tensor:
    past, current = _prefill(model, prompt_ids)
    _configure_decode(model, mode)
    try:
        logits, past = _decode_logits(
            model,
            past,
            current,
            seen_tokens=len(prompt_ids) - 1,
        )
        return logits.detach().cpu()
    finally:
        del past


def _phase1_rows(args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    lengths = _parse_ints(args.lengths)
    if not set(lengths) <= {16384, 32768}:
        raise ValueError("Phase 1 lengths must be 16384 and/or 32768")
    if args.dataset == "passkey":
        _, rows = load_passkey_rows(args.data_root)
        trials = (0, 1) if args.selection == "pilot" else None
        selected = _select_passkey_rows(rows, lengths=lengths, trials=trials)
        normalized = []
        for row in selected:
            normalized.append(
                {
                    **row,
                    "metric": "exact_match",
                    "generation_tokens": 32,
                    "scorer": "normalized_exact_match",
                }
            )
        return PASSKEY_SHA256, normalized

    manifest, rows = load_capability_suite(args.data_root)
    selected = []
    for row in rows:
        if int(row["target_length"]) not in set(lengths):
            continue
        if row["suite"] == "ruler" and str(row["task"]).startswith("niah_"):
            selected.append(row)
        elif row["suite"] in {"nolima_hard_exact_context", "longbench"}:
            selected.append(row)
    if not selected:
        raise ValueError("retrieval suite selection is empty")
    return sha256_file(args.data_root / "manifest.json"), selected


def _score_phase1_row(
    model: torch.nn.Module,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    answer_scores = [
        _answer_nll(
            model,
            row["prompt_ids"],
            _answer_ids(tokenizer, answer),
            mode=mode,
        )
        for answer in row["answers"]
    ]
    selected_index = max(
        range(len(answer_scores)), key=lambda index: answer_scores[index]["mean_logprob"]
    )
    selected = answer_scores[selected_index]
    generation = _generate(
        model,
        tokenizer,
        row["prompt_ids"],
        mode=mode,
        max_new_tokens=int(row["generation_tokens"]),
    )
    metric_score = score_capability_prediction(
        row["metric"],
        generation["prediction"],
        row["answers"],
        source=row.get("source"),
    )
    generation.update(
        score_generation_metrics(
            generation["prediction"],
            row["answers"],
            eos_terminated=bool(generation["eos_terminated"]),
            generated_token_count=int(generation["generated_token_count"]),
        )
    )
    return {
        "example_id": row["example_id"],
        "suite": row["suite"],
        "task": row["task"],
        "target_length": int(row["target_length"]),
        "depth_percent": row.get("depth_percent"),
        "prompt_sha256": row["prompt_sha256"],
        "metric": row["metric"],
        "mode": mode,
        "nll_sum": selected["nll_sum"],
        "answer_tokens": selected["answer_tokens"],
        "mean_logprob": selected["mean_logprob"],
        "metric_score": metric_score,
        "selected_reference_index": selected_index,
        "reference_mean_logprobs": [score["mean_logprob"] for score in answer_scores],
        "references": list(row["answers"]),
        "generation": generation,
    }


def run_phase1(args: argparse.Namespace) -> dict[str, Any]:
    gate = json.loads(args.phase0_gate.read_text(encoding="utf-8"))
    if gate.get("schema") != PHASE0_SUMMARY_SCHEMA or gate.get("status") != "pass":
        raise ValueError("Phase 1 requires a passing Phase 0 gate")
    data_hash, rows = _phase1_rows(args)
    target_lengths = sorted({int(row["target_length"]) for row in rows})
    if any(str(length) not in gate.get("lengths", {}) for length in target_lengths):
        raise ValueError("Phase 1 requested a length not covered by the passing gate")
    modes = tuple(piece.strip() for piece in args.modes.split(",") if piece.strip())
    if not modes or any(mode not in {"dense", "score", "fixed"} for mode in modes):
        raise ValueError("Phase 1 modes must be dense,score,fixed")
    model, tokenizer, identity = _load_arm_model(args)
    _register_attention()
    sanity_row = min(rows, key=lambda row: int(row["target_length"]))
    dense_logits = _first_step_logits(model, sanity_row["prompt_ids"], mode="dense")
    full_logits = _first_step_logits(model, sanity_row["prompt_ids"], mode="full")
    max_abs = float((dense_logits - full_logits).abs().max())
    sanity = {
        "example_id": sanity_row["example_id"],
        "max_abs_logit_difference": max_abs,
        "top1_equal": int(dense_logits.argmax()) == int(full_logits.argmax()),
        "tolerance": 1e-6,
    }
    if max_abs > sanity["tolerance"] or not sanity["top1_equal"]:
        raise RuntimeError(f"full-budget sparse attention differs from dense: {sanity}")
    del dense_logits, full_logits

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    results = []
    total = len(rows) * len(modes)
    progress = 0
    for row in rows:
        for mode in modes:
            progress += 1
            result = _score_phase1_row(model, tokenizer, row, mode=mode)
            results.append(result)
            print(
                json.dumps(
                    {
                        "phase": 1,
                        "substrate": args.substrate,
                        "progress": f"{progress}/{total}",
                        "mode": mode,
                        "task": row["task"],
                        "length": row["target_length"],
                        "metric_score": result["metric_score"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    output = {
        "schema": PHASE1_SCHEMA,
        "substrate": args.substrate,
        "single_seed_supporting": True,
        "raw_extrapolation": True,
        "dataset": args.dataset,
        "selection": args.selection,
        "target_lengths": target_lengths,
        "data_sha256": data_hash,
        "phase0_gate_sha256": sha256_file(args.phase0_gate),
        "adapter": identity,
        "sparse_config": SPARSE_CONFIG,
        "operator_contract": {
            "scope": "answer_side_decode_after_dense_prompt_prefill",
            "score": "exact_per_query_head_max_qk_per_128_token_block",
            "position_handling": "mask_only_original_rotary_kv_indices_no_reordering",
            "efficiency_claim": False,
            "fixed_control": "same_remote_block_budget_uniform_content_independent_blocks",
        },
        "full_budget_sanity": sanity,
        "script_sha256": _script_sha256(),
        "results": results,
        "runtime": {
            "seconds": time.time() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        },
    }
    _atomic_json(args.output, output)
    return output


def run_raw_capability(args: argparse.Namespace) -> dict[str, Any]:
    """Score the frozen long-context suite without range scaling or a Phase-0 gate."""
    manifest, rows = load_capability_suite(args.data_root)
    expected_tokenizer = tokenizer_source_fingerprint(args.model_name)
    if manifest.get("tokenizer", {}).get("identifier") != expected_tokenizer.get(
        "identifier"
    ) or manifest.get("tokenizer", {}).get("files") != expected_tokenizer.get("files"):
        raise ValueError("capability suite tokenizer differs from the model tokenizer")
    lengths = set(_parse_ints(args.lengths))
    selected = [row for row in rows if int(row["target_length"]) in lengths]
    if not selected:
        raise ValueError("raw capability selection is empty")

    model, tokenizer, identity = _load_arm_model(args)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    results = []
    for index, row in enumerate(selected, start=1):
        result = _score_phase1_row(model, tokenizer, row, mode="dense")
        results.append(result)
        print(
            json.dumps(
                {
                    "arm": args.arm_name,
                    "progress": f"{index}/{len(selected)}",
                    "task": row["task"],
                    "length": row["target_length"],
                    "metric_score": result["metric_score"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    output = {
        "schema": RAW_CAPABILITY_SCHEMA,
        "status": "complete",
        "arm": args.arm_name,
        "substrate": args.substrate,
        "single_seed_supporting": True,
        "raw_extrapolation": True,
        "selection": {"lengths": sorted(lengths), "rows": len(selected)},
        "data_sha256": sha256_file(args.data_root / "manifest.json"),
        "adapter": identity,
        "results": results,
        "aggregate": _phase1_aggregate(results),
        "script_sha256": _script_sha256(),
        "runtime": {
            "seconds": time.time() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        },
    }
    _atomic_json(args.output, output)
    return output


def _phase1_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["mode"]), int(row["target_length"]))].append(row)
    output = {}
    for (mode, length), cell in sorted(grouped.items()):
        tokens = sum(int(row["answer_tokens"]) for row in cell)
        output.setdefault(mode, {})[str(length)] = {
            "examples": len(cell),
            "metric_mean": statistics.fmean(float(row["metric_score"]) for row in cell),
            "nll": sum(float(row["nll_sum"]) for row in cell) / tokens,
            "answer_tokens": tokens,
        }
    return output


def summarize_phase1(args: argparse.Namespace) -> dict[str, Any]:
    geo = json.loads(args.geo.read_text(encoding="utf-8"))
    evq = json.loads(args.evq.read_text(encoding="utf-8"))
    if geo.get("schema") != PHASE1_SCHEMA or evq.get("schema") != PHASE1_SCHEMA:
        raise ValueError("Phase 1 raw schema mismatch")
    if geo.get("substrate") != "native_geo" or evq.get("substrate") != "evq_cosh":
        raise ValueError("Phase 1 summary received the wrong arms")
    for key in (
        "dataset",
        "selection",
        "target_lengths",
        "data_sha256",
        "phase0_gate_sha256",
        "sparse_config",
    ):
        if geo.get(key) != evq.get(key):
            raise ValueError(f"Phase 1 arms differ at {key}")
    geo_keys = {(row["example_id"], row["mode"]) for row in geo["results"]}
    evq_keys = {(row["example_id"], row["mode"]) for row in evq["results"]}
    if geo_keys != evq_keys:
        raise ValueError("Phase 1 arms have different example/mode cells")
    aggregate = {
        "native_geo": _phase1_aggregate(geo["results"]),
        "evq_cosh": _phase1_aggregate(evq["results"]),
    }
    did = {}
    for mode in ("score", "fixed"):
        if mode not in aggregate["native_geo"] or mode not in aggregate["evq_cosh"]:
            continue
        did[mode] = {}
        for length in sorted(aggregate["native_geo"][mode]):
            geo_dense = aggregate["native_geo"]["dense"][length]
            geo_mode = aggregate["native_geo"][mode][length]
            evq_dense = aggregate["evq_cosh"]["dense"][length]
            evq_mode = aggregate["evq_cosh"][mode][length]
            did[mode][length] = {
                "metric": (evq_mode["metric_mean"] - evq_dense["metric_mean"])
                - (geo_mode["metric_mean"] - geo_dense["metric_mean"]),
                "nll_improvement": (evq_dense["nll"] - evq_mode["nll"])
                - (geo_dense["nll"] - geo_mode["nll"]),
            }
    score_cells = list(did.get("score", {}).values())
    pilot_checks = {
        "score_metric_did_at_least_0p20_each_length": bool(score_cells)
        and all(cell["metric"] >= 0.20 for cell in score_cells),
        "score_nll_did_positive_each_length": bool(score_cells)
        and all(cell["nll_improvement"] > 0 for cell in score_cells),
    }
    output = {
        "schema": PHASE1_SUMMARY_SCHEMA,
        "status": "positive" if all(pilot_checks.values()) else "stop",
        "pilot_checks": pilot_checks,
        "aggregate": aggregate,
        "difference_in_differences": did,
        "inputs": {"geo_sha256": sha256_file(args.geo), "evq_sha256": sha256_file(args.evq)},
        "single_seed_supporting": True,
    }
    _atomic_json(args.output, output)
    return output


def _add_arm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--training-data-manifest", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--substrate", choices=("native_geo", "evq_cosh"), required=True)
    parser.add_argument("--output", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-passkey")
    prepare.add_argument("--tokenizer", required=True)
    prepare.add_argument("--reference-manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    dry = commands.add_parser("dry-run")
    dry.add_argument("--model-name", required=True)
    dry.add_argument("--model-manifest", type=Path, required=True)
    dry.add_argument("--training-data-manifest", type=Path, required=True)
    dry.add_argument("--geo-adapter", type=Path, required=True)
    dry.add_argument("--evq-adapter", type=Path, required=True)
    dry.add_argument("--passkey-root", type=Path, required=True)
    dry.add_argument("--suite-root", type=Path, required=True)
    dry.add_argument("--output", type=Path, required=True)

    phase0 = commands.add_parser("phase0")
    _add_arm_arguments(phase0)
    phase0.add_argument("--passkey-root", type=Path, required=True)
    phase0.add_argument("--lengths", default="8192,16384,32768")
    phase0.add_argument("--trials", default="0,1")

    phase0_summary = commands.add_parser("summarize-phase0")
    phase0_summary.add_argument("--geo", type=Path, required=True)
    phase0_summary.add_argument("--evq", type=Path, required=True)
    phase0_summary.add_argument(
        "--gate-lengths", choices=("16384", "16384,32768"), default="16384,32768"
    )
    phase0_summary.add_argument("--output", type=Path, required=True)

    phase1 = commands.add_parser("phase1")
    _add_arm_arguments(phase1)
    phase1.add_argument("--phase0-gate", type=Path, required=True)
    phase1.add_argument("--dataset", choices=("passkey", "retrieval-suite"), default="passkey")
    phase1.add_argument("--data-root", type=Path, required=True)
    phase1.add_argument("--selection", choices=("pilot", "full"), default="pilot")
    phase1.add_argument("--lengths", default="16384,32768")
    phase1.add_argument("--modes", default="dense,score,fixed")

    phase1_summary = commands.add_parser("summarize-phase1")
    phase1_summary.add_argument("--geo", type=Path, required=True)
    phase1_summary.add_argument("--evq", type=Path, required=True)
    phase1_summary.add_argument("--output", type=Path, required=True)

    raw_capability = commands.add_parser("raw-capability")
    _add_arm_arguments(raw_capability)
    raw_capability.add_argument(
        "--arm-name", choices=("geo_lora_native", "evq_lora_evq"), required=True
    )
    raw_capability.add_argument("--data-root", type=Path, required=True)
    raw_capability.add_argument("--lengths", default="16384")
    raw_capability.add_argument("--stage2", action="store_true")

    canary = commands.add_parser("counterfactual-canary")
    _add_arm_arguments(canary)
    canary.add_argument(
        "--arm-name",
        choices=(
            "base_native",
            "base_midpoint",
            "base_evq",
            "geo_lora_native",
            "geo_lora_evq_cross",
            "evq_lora_native_cross",
            "evq_lora_evq",
        ),
        required=True,
    )
    canary.add_argument("--passkey-root", type=Path, required=True)
    canary.add_argument("--lengths", default="8192")
    canary.add_argument("--trials", default="0,1")
    canary.add_argument("--disable-adapter", action="store_true")
    canary.add_argument(
        "--runtime-frequency",
        choices=("trained", "native_geo", "midpoint_geo", "evq_cosh"),
        default="trained",
    )

    canary_summary = commands.add_parser("summarize-counterfactual-canary")
    canary_summary.add_argument("--base-native", type=Path, required=True)
    canary_summary.add_argument("--geo-lora-native", type=Path, required=True)
    canary_summary.add_argument("--evq-lora-evq", type=Path, required=True)
    canary_summary.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare-passkey":
        result = prepare_passkey(args)
    elif args.command == "dry-run":
        result = dry_run(args)
    elif args.command == "phase0":
        result = run_phase0(args)
    elif args.command == "summarize-phase0":
        result = summarize_phase0(args)
    elif args.command == "phase1":
        result = run_phase1(args)
    elif args.command == "summarize-phase1":
        result = summarize_phase1(args)
    elif args.command == "raw-capability":
        result = run_raw_capability(args)
    elif args.command == "counterfactual-canary":
        result = run_counterfactual_canary(args)
    else:
        result = summarize_counterfactual_canary(args)
    print(json.dumps({key: result.get(key) for key in ("schema", "status", "runtime")}, indent=2))


if __name__ == "__main__":
    main()
