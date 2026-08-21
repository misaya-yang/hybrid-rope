"""Frozen contracts for the OLMo-2 finite function-morph audit.

This protocol is an inference-only diagnostic.  It does not train parameters
and it does not turn a finite morph grid into an optimizer or a continuous
optimality claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


METHOD_ID = "olmo2_function_morph_r0_v1"
SCHEMA_VERSION = 1
MODEL_LABEL = "OLMo-2-0425-1B-Instruct"
MODEL_REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"
ROPE_BASE = 500_000.0
HEAD_DIM = 128
ROTARY_PAIRS = 64
LAYERS = 16
ATTENTION_HEADS = 16
SHORT_LENGTH = 4_096
FAR_LENGTHS = (8_192, 16_384)
LENGTHS = (SHORT_LENGTH, *FAR_LENGTHS)
ROWS_PER_LENGTH = 4
TAIL_TOKENS = 64
PHASE_BINS = 64
PHASE_LAMBDA = 0.1
EVQ_TAU = HEAD_DIM / math.sqrt(SHORT_LENGTH)
MORPH_GRID = (0.0, 0.05, 0.25, 0.5, 0.75, 1.0)
CANDIDATES = (
    "phase_chord_olmo_r0_lambda_0p1",
    "matched_exponential_control",
    "anchored_evq_cosh_tau_2",
)


class ContractError(ValueError):
    """Raised when an experiment identity or finite-table contract drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def validate_model_config(config: Mapping[str, Any]) -> None:
    expected = {
        "model_type": "olmo2",
        "hidden_size": 2_048,
        "num_attention_heads": ATTENTION_HEADS,
        "num_key_value_heads": ATTENTION_HEADS,
        "num_hidden_layers": LAYERS,
        "max_position_embeddings": SHORT_LENGTH,
        "rope_theta": ROPE_BASE,
        "rope_scaling": None,
    }
    for key, wanted in expected.items():
        if config.get(key) != wanted:
            raise ContractError(
                f"model config drift for {key}: expected {wanted!r}, "
                f"observed {config.get(key)!r}"
            )
    observed_head_dim = int(config["hidden_size"]) // int(
        config["num_attention_heads"]
    )
    if observed_head_dim != HEAD_DIM:
        raise ContractError(
            f"head_dim drift: expected {HEAD_DIM}, observed {observed_head_dim}"
        )


def validate_table(values: Sequence[float], *, label: str) -> tuple[float, ...]:
    table = tuple(float(value) for value in values)
    if len(table) != ROTARY_PAIRS:
        raise ContractError(
            f"{label} must have {ROTARY_PAIRS} frequencies, got {len(table)}"
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in table):
        raise ContractError(f"{label} must contain finite positive frequencies")
    if any(left <= right for left, right in zip(table, table[1:])):
        raise ContractError(f"{label} must be strictly decreasing")
    return table


def protocol_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "model": {
            "label": MODEL_LABEL,
            "revision": MODEL_REVISION,
            "layers": LAYERS,
            "attention_heads": ATTENTION_HEADS,
            "head_dim": HEAD_DIM,
            "rotary_pairs": ROTARY_PAIRS,
            "rope_base": ROPE_BASE,
            "native_context": SHORT_LENGTH,
        },
        "candidate_order": list(CANDIDATES),
        "candidate_roles": {
            CANDIDATES[0]: (
                "attention-distance-derived candidate; operator-activity prior, "
                "not an LM-risk derivative"
            ),
            CANDIDATES[1]: (
                "non-attention-aware endpoint-preserving control matched only "
                "on RMS log-frequency displacement"
            ),
            CANDIDATES[2]: (
                "endpoint-anchored closed-form EVQ-Cosh comparator at the "
                "zero-search tau rule"
            ),
        },
        "target_parameters": {
            "phase_bins": PHASE_BINS,
            "phase_lambda": PHASE_LAMBDA,
            "evq_tau": EVQ_TAU,
        },
        "evaluation": {
            "lengths": list(LENGTHS),
            "rows_per_length": ROWS_PER_LENGTH,
            "tail_tokens": TAIL_TOKENS,
            "morph_grid": list(MORPH_GRID),
            "morph": "linear interpolation in log inverse-frequency space",
            "short_cost": "Native-teacher tail-token forward KL and tail NLL delta at 4096",
            "far_benefit": "tail NLL delta at 8192 and 16384",
            "decoding": "teacher-forced only; no autoregressive exact-match claim",
        },
        "execution": {
            "training": False,
            "optimizer": False,
            "gradients": False,
            "downloads": False,
            "network": False,
            "attention_backend": "PyTorch SDPA forced to FLASH_ATTENTION only",
            "math_attention_fallback": False,
        },
        "claim_boundary": (
            "A finite candidate-specific morph audit can rank measured "
            "function-preservation cost and far-NLL benefit on this protocol. "
            "It does not identify a Fisher optimum, a continuous basin, or a "
            "general retrofit solution."
        ),
    }
    payload["protocol_sha256"] = canonical_json_hash(payload)
    return payload
