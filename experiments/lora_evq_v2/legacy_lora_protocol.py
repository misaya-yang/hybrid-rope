#!/usr/bin/env python3
"""Pure contracts for the protocol-matched legacy LongAlign LoRA rerun.

This module deliberately has no Transformers dependency so launchers and
artifact validators can fail before allocating a GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


LEGACY_OBJECTIVE = "legacy_longalign_full_token_causal_lm_v2"
LEGACY_METHODS = ("native_geo", "evq_cosh")
LEGACY_SEEDS = (42, 43, 44)
OFFICIAL_LONGALIGN_SOURCE = "zai-org/LongAlign-10k"
OFFICIAL_LONGALIGN_REVISION = "12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc"
OFFICIAL_LONGALIGN_RAW_SHA256 = "d7a1c39738e645ae1d0f8609a3cd62e50a3fa982e23cb752f64630bdbb7cee08"

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_source_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    required = {"source_id", "revision", "split", "filename", "raw_sha256"}
    missing = sorted(required - set(receipt))
    if missing:
        raise ValueError(f"source receipt missing fields: {', '.join(missing)}")
    if receipt["source_id"] != OFFICIAL_LONGALIGN_SOURCE:
        raise ValueError(
            "legacy claim data must identify the official "
            f"{OFFICIAL_LONGALIGN_SOURCE} release"
        )
    if receipt["revision"] != OFFICIAL_LONGALIGN_REVISION:
        raise ValueError("source revision does not match the pinned official LongAlign release")
    if receipt["split"] != "train":
        raise ValueError("legacy LongAlign preparation requires split=train")
    if receipt["filename"] != "long.jsonl":
        raise ValueError("official LongAlign-10k receipt must name long.jsonl")
    if receipt["raw_sha256"] != OFFICIAL_LONGALIGN_RAW_SHA256:
        raise ValueError("raw_sha256 does not match the pinned official long.jsonl bytes")
    return dict(receipt)


def legacy_run_name(method: str, seed: int) -> str:
    if method not in LEGACY_METHODS:
        raise ValueError(f"unsupported legacy method: {method}")
    if seed not in LEGACY_SEEDS:
        raise ValueError(f"unsupported legacy seed: {seed}")
    prefix = "geo_longalign" if method == "native_geo" else "evq_longalign_tau1414"
    return f"{prefix}_s{seed}"


def legacy_eval_filename(variant: str) -> str:
    allowed = {"base_geo", "base_evq_tau1414"}
    allowed.update(
        legacy_run_name(method, seed)
        for method in LEGACY_METHODS
        for seed in LEGACY_SEEDS
    )
    if variant not in allowed:
        raise ValueError(f"unsupported legacy evaluation variant: {variant}")
    return f"eval_{variant}.json"


def canonical_training_protocol(
    *,
    method: str,
    seed: int,
    data_manifest_sha256: str,
    model_manifest_sha256: str,
    code_sha256: str,
) -> Dict[str, Any]:
    if method not in LEGACY_METHODS:
        raise ValueError(f"unsupported legacy method: {method}")
    if seed not in LEGACY_SEEDS:
        raise ValueError(f"unsupported legacy seed: {seed}")
    protocol = {
        "format_version": 1,
        "objective": LEGACY_OBJECTIVE,
        "method": method,
        "tau": 1.414 if method == "evq_cosh" else None,
        "seed": seed,
        "split_seed": 42,
        "data_manifest_sha256": data_manifest_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "code_sha256": code_sha256,
        "model_geometry": {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 128256,
            "rope_base": 500000.0,
            "max_position_embeddings": 8192,
        },
        "max_seq_len": 8192,
        "max_samples": 8000,
        "validation_ratio": 0.02,
        "minimum_tokens": 64,
        "labels": "all_non_padding_input_tokens",
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "precision": "bfloat16",
        "quantization": None,
        "max_steps": 300,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 8,
        "learning_rate": 1e-4,
        "warmup_steps": 60,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "optimizer": "adamw_torch",
        "lr_scheduler": "cosine",
        "gradient_checkpointing": True,
        "save_steps": 100,
        "save_total_limit": 2,
        "torch_compile": True,
        "torch_compile_mode": "default",
    }
    return validate_legacy_protocol(protocol)


def validate_legacy_protocol(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    method = protocol.get("method")
    seed = protocol.get("seed")
    if method not in LEGACY_METHODS:
        raise ValueError(f"unsupported legacy method: {method}")
    if seed not in LEGACY_SEEDS:
        raise ValueError(f"unsupported legacy seed: {seed}")
    for field in ("data_manifest_sha256", "model_manifest_sha256", "code_sha256"):
        if not _HEX_64.fullmatch(str(protocol.get(field, ""))):
            raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    expected = {
            "format_version": 1,
            "objective": LEGACY_OBJECTIVE,
            "method": method,
            "tau": 1.414 if method == "evq_cosh" else None,
            "seed": seed,
            "split_seed": 42,
            "data_manifest_sha256": protocol["data_manifest_sha256"],
            "model_manifest_sha256": protocol["model_manifest_sha256"],
            "code_sha256": protocol["code_sha256"],
            "model_geometry": {
                "model_type": "llama",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 128256,
                "rope_base": 500000.0,
                "max_position_embeddings": 8192,
            },
            "max_seq_len": 8192,
            "max_samples": 8000,
            "validation_ratio": 0.02,
            "minimum_tokens": 64,
            "labels": "all_non_padding_input_tokens",
            "lora_r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "precision": "bfloat16",
            "quantization": None,
            "max_steps": 300,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 8,
            "learning_rate": 1e-4,
            "warmup_steps": 60,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "optimizer": "adamw_torch",
            "lr_scheduler": "cosine",
            "gradient_checkpointing": True,
            "save_steps": 100,
            "save_total_limit": 2,
            "torch_compile": True,
            "torch_compile_mode": "default",
    }
    missing = sorted(set(expected) - set(protocol))
    if missing:
        raise ValueError(f"legacy protocol missing fields: {', '.join(missing)}")
    extra = sorted(set(protocol) - set(expected))
    if extra:
        raise ValueError(f"legacy protocol has unregistered fields: {', '.join(extra)}")
    for key, value in expected.items():
        actual = protocol[key]
        if isinstance(value, float):
            if not isinstance(actual, (int, float)) or not math.isclose(
                float(actual), value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"legacy protocol mismatch for {key}: {actual!r} != {value!r}")
        elif actual != value:
            raise ValueError(f"legacy protocol mismatch for {key}: {actual!r} != {value!r}")
    return dict(protocol)


def validate_complete_matrix(records: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows = [dict(record) for record in records]
    expected_pairs = {
        (method, seed) for method in LEGACY_METHODS for seed in LEGACY_SEEDS
    }
    actual_pairs = {(row.get("method"), row.get("seed")) for row in rows}
    if actual_pairs != expected_pairs or len(rows) != len(expected_pairs):
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs, key=str)
        raise ValueError(f"legacy matrix mismatch; missing={missing}, extra={extra}")
    if any(row.get("status") != "complete" for row in rows):
        raise ValueError("all six legacy arms must have status=complete")
    for identity_field in ("data_manifest_sha256", "model_manifest_sha256", "code_sha256"):
        identities = {row.get(identity_field) for row in rows}
        if len(identities) != 1 or not _HEX_64.fullmatch(str(next(iter(identities), ""))):
            raise ValueError(f"all six arms must share one valid {identity_field}")
    return sorted(rows, key=lambda row: (LEGACY_SEEDS.index(row["seed"]), LEGACY_METHODS.index(row["method"])))


def _metric_stats(values: list[float]) -> Dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty metric list")
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def paired_metric_summary(
    geo_by_seed: Mapping[int, float],
    evq_by_seed: Mapping[int, float],
) -> Dict[str, Any]:
    seeds = sorted(set(geo_by_seed) & set(evq_by_seed))
    if seeds != list(LEGACY_SEEDS) or set(geo_by_seed) != set(evq_by_seed):
        raise ValueError("paired summary requires Geo and EVQ values for seeds 42,43,44")
    geo = [float(geo_by_seed[seed]) for seed in seeds]
    evq = [float(evq_by_seed[seed]) for seed in seeds]
    delta = [evq_value - geo_value for geo_value, evq_value in zip(geo, evq)]
    return {
        "seeds": seeds,
        "geo": _metric_stats(geo),
        "evq": _metric_stats(evq),
        "paired_delta_evq_minus_geo": _metric_stats(delta),
    }
