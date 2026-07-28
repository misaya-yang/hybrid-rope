#!/usr/bin/env python3
"""Contracts for a Native-importance-protected EVQ retrofit experiment.

This is a mature-model research candidate, not the submitted EVQ-Cosh method.
Protected rotary pairs retain their exact Native frequencies.  Every other
pair uses the exact EVQ-Cosh frequency, and Q/K LoRA updates are masked away
from the protected output coordinates.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    MODEL_CONTRACT,
    endpoint_evq_inv_freq,
    endpoint_geo_inv_freq,
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    install_adaptation,
)

from .evq_attention_restoration import (
    CapturedAttention,
    normalized_context_mse,
)


METHOD_ID = "native_importance_protected_evq_qk_restoration_v1"
FREQUENCY_NAME = "native_importance_protected_evq"
ADAPTATION_NAME = "masked_qk_attention_restoration"
DIAGNOSTIC_STATUS = "OLMO2_NATIVE_BAND_IMPORTANCE_DIAGNOSTIC_V1"
DIAGNOSTIC_PREPARED_STATUS = (
    "OLMO2_NATIVE_BAND_DIAGNOSTIC_PREPARED_NO_GPU_V1"
)
PREPARED_STATUS = "OLMO2_NATIVE_PROTECTED_EVQ_PREPARED_NO_GPU_V1"
READY_STATUS = "OLMO2_NATIVE_PROTECTED_EVQ_GPU_READY_V1"
RESULT_STATUS = "OLMO2_NATIVE_PROTECTED_EVQ_TRAINING_COMPLETE_V1"
BAND_IMPORTANCE_BACKEND = "native_band_importance_sdpa"

SEQUENCE_LENGTH = 4_096
PAIR_COUNT = 64
HEAD_DIM = 128
ATTENTION_HEADS = 16
LAYERS = 16
CALIBRATION_ROWS = 16
QUERY_POSITIONS = 16
IMPORTANCE_MASS_TARGET = 0.80
MAX_PROTECTED_PAIRS = 16
MIN_SPLIT_SCORE_COSINE = 0.98
MIN_SPLIT_JACCARD = 0.60
MIN_PER_LAYER_CAPTURED_MASS = 0.50


def identical_native_evq_pair_indices() -> tuple[int, ...]:
    native = endpoint_geo_inv_freq()
    evq = endpoint_evq_inv_freq()
    return tuple(
        int(index)
        for index in torch.nonzero(native == evq, as_tuple=False).flatten()
    )


def canonical_pair_indices(
    values: Any,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    pairs = tuple(sorted(int(value) for value in values))
    if (
        (not pairs and not allow_empty)
        or len(set(pairs)) != len(pairs)
        or any(value < 0 or value >= PAIR_COUNT for value in pairs)
    ):
        raise ValueError("protected rotary-pair indices are invalid")
    return pairs


def native_protected_evq_inv_freq(
    protected_pairs: Any,
) -> torch.Tensor:
    """Build one exact hybrid from independent Native and EVQ snapshots."""
    protected = canonical_pair_indices(
        protected_pairs, allow_empty=True
    )
    native = endpoint_geo_inv_freq().clone()
    evq = endpoint_evq_inv_freq().clone()
    if native.shape != (PAIR_COUNT,) or evq.shape != (PAIR_COUNT,):
        raise RuntimeError("OLMo rotary frequency cardinality drift")
    hybrid = evq.clone()
    hybrid[list(protected)] = native[list(protected)]
    protected_mask = torch.zeros(PAIR_COUNT, dtype=torch.bool)
    protected_mask[list(protected)] = True
    if not torch.equal(hybrid[protected_mask], native[protected_mask]):
        raise RuntimeError("protected Native frequency identity drift")
    if not torch.equal(hybrid[~protected_mask], evq[~protected_mask]):
        raise RuntimeError("unprotected EVQ frequency identity drift")
    return hybrid


def apply_native_protected_evq(
    model: Any,
    protected_pairs: Any,
) -> dict[str, Any]:
    protected = canonical_pair_indices(
        protected_pairs, allow_empty=True
    )
    native = endpoint_geo_inv_freq()
    active = model.model.rotary_emb.inv_freq
    observed = active.detach().cpu().to(torch.float32)
    if not torch.equal(observed, native):
        raise RuntimeError("model is not at the Native frequency anchor")
    hybrid = native_protected_evq_inv_freq(protected)
    with torch.no_grad():
        active.copy_(hybrid.to(device=active.device, dtype=active.dtype))
    model.model.rotary_emb.original_inv_freq = active.detach().clone()
    realized = active.detach().cpu().to(torch.float32)
    if not torch.equal(realized, hybrid):
        raise RuntimeError("Native-protected EVQ frequency write drift")
    unprotected = tuple(
        index for index in range(PAIR_COUNT) if index not in set(protected)
    )
    return {
        "active_frequency": FREQUENCY_NAME,
        "active_sha256_float32": tensor_sha256(realized),
        "native_frequency_sha256_float32": tensor_sha256(native),
        "evq_frequency_sha256_float32": tensor_sha256(
            endpoint_evq_inv_freq()
        ),
        "protected_native_pair_indices": list(protected),
        "unprotected_evq_pair_indices": list(unprotected),
        "protected_pair_count": len(protected),
        "unprotected_pair_count": len(unprotected),
    }


def qk_unprotected_output_mask(
    config: Any,
    protected_pairs: Any,
) -> torch.Tensor:
    """Return a Q/K output mask in OLMo's split-half RoPE layout."""
    protected = canonical_pair_indices(
        protected_pairs, allow_empty=True
    )
    expected = MODEL_CONTRACT
    architecture = {
        "hidden_size": int(config.hidden_size),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(
            getattr(
                config,
                "head_dim",
                int(config.hidden_size) // int(config.num_attention_heads),
            )
        ),
    }
    for name, value in architecture.items():
        if value != int(expected[name]):
            raise RuntimeError(f"OLMo architecture drift for {name}")
    if architecture["num_attention_heads"] != architecture[
        "num_key_value_heads"
    ]:
        raise RuntimeError(
            "one shared Q/K output mask requires equal Q and KV head counts"
        )
    protected_set = set(protected)
    unprotected = [
        index for index in range(PAIR_COUNT) if index not in protected_set
    ]
    mask = torch.zeros(architecture["hidden_size"], dtype=torch.float32)
    for head in range(architecture["num_attention_heads"]):
        offset = head * architecture["head_dim"]
        coordinates = [
            offset + pair
            for pair in unprotected
        ] + [
            offset + PAIR_COUNT + pair
            for pair in unprotected
        ]
        mask[coordinates] = 1.0
    expected_active = (
        architecture["num_attention_heads"] * 2 * len(unprotected)
    )
    if int(mask.sum().item()) != expected_active:
        raise RuntimeError("Q/K unprotected-output mask cardinality drift")
    return mask


def install_masked_qk_lora(
    model: nn.Module,
    *,
    protected_pairs: Any,
    rank: int,
    alpha: float,
) -> tuple[int, torch.Tensor]:
    mask = qk_unprotected_output_mask(model.config, protected_pairs)
    readout = install_adaptation(
        model,
        "qk_answer",
        rank=int(rank),
        alpha=float(alpha),
        qk_output_mask=mask,
    )
    if readout is not None:
        raise RuntimeError("Native-protected EVQ does not admit a readout")
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if (
        len(trainable) != 64
        or any(
            not (
                (".q_proj." in name or ".k_proj." in name)
                and name.endswith((".a", ".b"))
            )
            for name, _ in trainable
        )
    ):
        raise RuntimeError("masked Q/K LoRA trainable scope drift")
    return int(sum(parameter.numel() for _, parameter in trainable)), mask


def _map_kv_heads(
    value: torch.Tensor,
    *,
    query_heads: int,
) -> torch.Tensor:
    key_heads = int(value.shape[1])
    if query_heads % key_heads != 0:
        raise ValueError("query heads must be divisible by key heads")
    if key_heads == query_heads:
        return value
    mapping = torch.arange(query_heads, device=value.device) // (
        query_heads // key_heads
    )
    return value[:, mapping, :, :]


def exact_pair_ablation_forward_kl(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    query_positions: torch.Tensor,
    scaling: float | None = None,
    pair_chunk_size: int = 8,
) -> torch.Tensor:
    """Compute exact KL(A || A_without_pair) for sampled causal rows.

    Q/K use OLMo's split-half rotary layout: pair ``p`` occupies coordinates
    ``p`` and ``p + D/2``.  The identity

      KL(P || softmax(S-s_p)) = E_P[s_p] + log E_P[exp(-s_p)]

    avoids 64 full attention forwards while remaining exact for each
    post-RoPE pair ablation.
    """
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must have shape [B,H,T,D]")
    batch, query_heads, length, head_dim = query.shape
    if int(head_dim) % 2 != 0:
        raise ValueError("rotary head dimension must be even")
    pair_count = int(head_dim) // 2
    if pair_count != PAIR_COUNT:
        raise ValueError(f"expected {PAIR_COUNT} rotary pairs")
    key = _map_kv_heads(key, query_heads=query_heads)
    if (
        key.shape[0] != batch
        or key.shape[2] != length
        or key.shape[3] != head_dim
    ):
        raise ValueError("mapped key shape drift")
    positions = query_positions.to(device=query.device, dtype=torch.long)
    if (
        positions.ndim != 1
        or positions.numel() == 0
        or int(positions.min()) < 0
        or int(positions.max()) >= length
        or int(torch.unique(positions).numel()) != int(positions.numel())
    ):
        raise ValueError("query positions are invalid")
    scale = (
        float(scaling)
        if scaling is not None
        else 1.0 / math.sqrt(head_dim)
    )
    q_rows = query[:, :, positions, :].float()
    k_all = key.float()
    logits = torch.matmul(
        q_rows, k_all.transpose(-1, -2)
    ) * scale
    keys = torch.arange(length, device=query.device)
    valid = keys[None, :] <= positions[:, None]
    minimum = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(
        ~valid[None, None, :, :], minimum
    )
    log_partition = torch.logsumexp(masked_logits, dim=-1)
    probabilities = torch.softmax(masked_logits, dim=-1)
    output = torch.empty(
        (
            batch,
            query_heads,
            int(positions.numel()),
            pair_count,
        ),
        device=query.device,
        dtype=torch.float32,
    )
    chunk = int(pair_chunk_size)
    if chunk <= 0:
        raise ValueError("pair chunk size must be positive")
    for start in range(0, pair_count, chunk):
        stop = min(start + chunk, pair_count)
        indices = torch.arange(start, stop, device=query.device)
        q_pair = torch.stack(
            (
                q_rows[..., indices],
                q_rows[..., indices + pair_count],
            ),
            dim=-1,
        )
        k_pair = torch.stack(
            (
                k_all[..., indices],
                k_all[..., indices + pair_count],
            ),
            dim=-1,
        )
        contribution = torch.einsum(
            "bhqcp,bhtcp->bhqtc", q_pair, k_pair
        ) * scale
        removed = logits.unsqueeze(-1) - contribution
        removed = removed.masked_fill(
            ~valid[None, None, :, :, None], minimum
        )
        removed_log_partition = torch.logsumexp(removed, dim=3)
        expectation = (
            probabilities.unsqueeze(-1) * contribution
        ).sum(dim=3)
        local = (
            expectation
            + removed_log_partition
            - log_partition.unsqueeze(-1)
        )
        output[..., start:stop] = local.clamp_min(0.0)
    return output


def _select_by_mass(
    score: np.ndarray,
    *,
    mass_target: float,
    maximum_pairs: int,
) -> tuple[list[int], float, bool]:
    values = np.asarray(score, dtype=np.float64)
    if (
        values.shape != (PAIR_COUNT,)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or float(values.sum()) <= 0
    ):
        raise ValueError("band-importance score is invalid")
    order = np.argsort(-values, kind="stable")
    cumulative = np.cumsum(values[order]) / float(values.sum())
    required = int(np.searchsorted(cumulative, mass_target) + 1)
    selected = sorted(int(value) for value in order[:required])
    admitted = required <= int(maximum_pairs)
    reported = selected if admitted else sorted(
        int(value) for value in order[: int(maximum_pairs)]
    )
    mass = float(values[reported].sum() / values.sum())
    return reported, mass, admitted


def _score_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_value = np.asarray(left, dtype=np.float64)
    right_value = np.asarray(right, dtype=np.float64)
    denominator = math.sqrt(
        float(np.square(left_value).sum())
        * float(np.square(right_value).sum())
    )
    if denominator == 0:
        return 0.0
    return float(np.dot(left_value, right_value) / denominator)


def select_protected_pairs(
    per_row_layer_head_pair: np.ndarray,
    *,
    mass_target: float = IMPORTANCE_MASS_TARGET,
    maximum_pairs: int = MAX_PROTECTED_PAIRS,
    minimum_split_score_cosine: float = MIN_SPLIT_SCORE_COSINE,
    minimum_split_jaccard: float = MIN_SPLIT_JACCARD,
    minimum_per_layer_mass: float = MIN_PER_LAYER_CAPTURED_MASS,
) -> dict[str, Any]:
    """Apply the predeclared concentration and split-stability gate."""
    values = np.asarray(per_row_layer_head_pair, dtype=np.float64)
    if (
        values.ndim != 4
        or values.shape[0] < 4
        or values.shape[1:] != (LAYERS, ATTENTION_HEADS, PAIR_COUNT)
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError(
            "importance must have shape [R,16,16,64] and be finite/nonnegative"
        )
    identical = identical_native_evq_pair_indices()
    aggregate = values.mean(axis=(0, 1, 2))
    aggregate[list(identical)] = 0.0
    protected, aggregate_mass, concentrated = _select_by_mass(
        aggregate,
        mass_target=float(mass_target),
        maximum_pairs=int(maximum_pairs),
    )
    halves = (values[0::2], values[1::2])
    half_scores = [
        half.mean(axis=(0, 1, 2)) for half in halves
    ]
    for score in half_scores:
        score[list(identical)] = 0.0
    half_selections = [
        _select_by_mass(
            score,
            mass_target=float(mass_target),
            maximum_pairs=int(maximum_pairs),
        )
        for score in half_scores
    ]
    left_set = set(half_selections[0][0])
    right_set = set(half_selections[1][0])
    union = left_set | right_set
    jaccard = (
        float(len(left_set & right_set) / len(union))
        if union
        else 0.0
    )
    score_cosine = _score_cosine(half_scores[0], half_scores[1])
    per_layer = values.mean(axis=(0, 2))
    per_layer[:, list(identical)] = 0.0
    per_layer_mass = []
    for layer_score in per_layer:
        denominator = float(layer_score.sum())
        per_layer_mass.append(
            0.0
            if denominator <= 0
            else float(layer_score[protected].sum() / denominator)
        )
    gates = {
        "aggregate_concentration": bool(concentrated),
        "both_halves_concentrated": bool(
            half_selections[0][2] and half_selections[1][2]
        ),
        "split_score_cosine": bool(
            score_cosine >= float(minimum_split_score_cosine)
        ),
        "split_jaccard": bool(
            jaccard >= float(minimum_split_jaccard)
        ),
        "per_layer_coverage": bool(
            min(per_layer_mass) >= float(minimum_per_layer_mass)
        ),
    }
    return {
        "passed": bool(all(gates.values())),
        "protected_pair_indices": protected,
        "protected_pair_count": len(protected),
        "already_identical_native_evq_pair_indices": list(identical),
        "aggregate_importance_mass": aggregate_mass,
        "aggregate_score": aggregate.tolist(),
        "split_half_score": [score.tolist() for score in half_scores],
        "split_half_protected_pair_indices": [
            value[0] for value in half_selections
        ],
        "split_half_importance_mass": [
            float(value[1]) for value in half_selections
        ],
        "split_score_cosine": score_cosine,
        "split_jaccard": jaccard,
        "per_layer_protected_importance_mass": per_layer_mass,
        "minimum_per_layer_protected_importance_mass": min(per_layer_mass),
        "thresholds": {
            "importance_mass_target": float(mass_target),
            "maximum_protected_pairs": int(maximum_pairs),
            "minimum_split_score_cosine": float(
                minimum_split_score_cosine
            ),
            "minimum_split_jaccard": float(minimum_split_jaccard),
            "minimum_per_layer_mass": float(minimum_per_layer_mass),
        },
        "gates": gates,
        "importance_mass_note": (
            "Normalized sums of separate leave-one-pair-out attention KL "
            "values are a selection score, not an additive causal decomposition."
        ),
    }


class BandImportanceCollector:
    """Collect per-row, per-layer, per-head exact pair-ablation KL."""

    def __init__(
        self,
        *,
        query_positions: torch.Tensor,
        pair_chunk_size: int = 8,
    ) -> None:
        self.query_positions = query_positions.detach().cpu().long()
        self.pair_chunk_size = int(pair_chunk_size)
        self.current: dict[int, torch.Tensor] = {}

    def clear(self) -> None:
        self.current.clear()

    def observe(
        self,
        *,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        scaling: float | None,
    ) -> None:
        layer = getattr(module, "layer_idx", None)
        if layer is None or not 0 <= int(layer) < LAYERS:
            raise RuntimeError("attention layer index is unavailable")
        if int(layer) in self.current:
            raise RuntimeError("band importance observed one layer twice")
        values = exact_pair_ablation_forward_kl(
            query,
            key,
            query_positions=self.query_positions.to(query.device),
            scaling=scaling,
            pair_chunk_size=self.pair_chunk_size,
        )
        self.current[int(layer)] = (
            values.mean(dim=(0, 2)).detach().cpu().contiguous()
        )

    def require_row(self) -> torch.Tensor:
        if set(self.current) != set(range(LAYERS)):
            raise RuntimeError(
                f"incomplete band-importance layers: {sorted(self.current)}"
            )
        return torch.stack(
            [self.current[layer] for layer in range(LAYERS)], dim=0
        )


def band_importance_mask(
    *,
    attention_mask: torch.Tensor | None = None,
    **_: Any,
) -> None:
    if attention_mask is not None:
        raise RuntimeError("band diagnostic admits only full unpadded rows")
    return None


def band_importance_sdpa(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    dropout: float = 0.0,
    scaling: float | None = None,
    band_importance_collector: BandImportanceCollector | None = None,
    **_: Any,
) -> tuple[torch.Tensor, None]:
    if attention_mask is not None:
        raise RuntimeError("band diagnostic received a materialized mask")
    if query.shape[-2] != key.shape[-2]:
        raise RuntimeError("band diagnostic does not admit KV-cache decoding")
    if float(dropout) != 0.0:
        raise RuntimeError("band diagnostic requires zero dropout")
    if band_importance_collector is None:
        raise RuntimeError("band diagnostic collector is required")
    band_importance_collector.observe(
        module=module,
        query=query,
        key=key,
        scaling=scaling,
    )
    context = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        scale=scaling,
        is_causal=query.shape[-2] > 1,
    )
    return context.transpose(1, 2).contiguous(), None


def configure_band_importance_attention(model: nn.Module) -> None:
    AttentionInterface.register(
        BAND_IMPORTANCE_BACKEND, band_importance_sdpa
    )
    AttentionMaskInterface.register(
        BAND_IMPORTANCE_BACKEND, band_importance_mask
    )
    if (
        ALL_ATTENTION_FUNCTIONS[BAND_IMPORTANCE_BACKEND]
        is not band_importance_sdpa
        or ALL_MASK_ATTENTION_FUNCTIONS[BAND_IMPORTANCE_BACKEND]
        is not band_importance_mask
    ):
        raise RuntimeError("band-importance backend registration drift")
    model.config._attn_implementation = BAND_IMPORTANCE_BACKEND


class MultiLayerRelationCapture:
    """Capture all 16 Native-teacher and student attention functions."""

    def __init__(self) -> None:
        self.values: dict[str, dict[int, CapturedAttention]] = {
            "teacher": {},
            "student": {},
        }

    def clear(self) -> None:
        for values in self.values.values():
            values.clear()

    def store(
        self,
        *,
        mode: str,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        context: torch.Tensor,
    ) -> None:
        if mode not in self.values:
            raise RuntimeError(f"unknown relation mode {mode!r}")
        layer = getattr(module, "layer_idx", None)
        if layer is None or not 0 <= int(layer) < LAYERS:
            raise RuntimeError("relation layer index is unavailable")
        layer = int(layer)
        if layer in self.values[mode]:
            raise RuntimeError("relation layer captured twice")
        if mode == "teacher":
            captured = CapturedAttention(
                query=query.detach().contiguous(),
                key=key.detach().contiguous(),
                value=value.detach().contiguous(),
                context=context.detach().contiguous(),
            )
        else:
            captured = CapturedAttention(
                query=query.contiguous(),
                key=key.contiguous(),
                value=value.contiguous(),
                context=context.contiguous(),
            )
        self.values[mode][layer] = captured

    def require_layers(
        self,
    ) -> list[tuple[CapturedAttention, CapturedAttention]]:
        expected = set(range(LAYERS))
        if any(set(values) != expected for values in self.values.values()):
            raise RuntimeError("incomplete all-layer attention capture")
        return [
            (self.values["teacher"][layer], self.values["student"][layer])
            for layer in range(LAYERS)
        ]


def all_layer_attention_restoration_loss(
    *,
    captures: MultiLayerRelationCapture,
    kernel: Callable[..., torch.Tensor],
    attention_weight: float,
    context_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match actual causal QK attention and A@V context at every layer."""
    layer_qk: list[torch.Tensor] = []
    layer_context: list[torch.Tensor] = []
    for teacher, student in captures.require_layers():
        batch, heads, length, head_dim = student.query.shape
        if teacher.query.shape != student.query.shape:
            raise RuntimeError("teacher/student query shape drift")
        scale = 1.0 / math.sqrt(int(head_dim))
        qk = kernel(
            student.query,
            student.key,
            teacher.query,
            teacher.key,
            attn_mask=None,
            causal=True,
            sm_scale_s=scale,
            sm_scale_t=scale,
        ) / float(batch * heads * length)
        context = normalized_context_mse(
            student.context, teacher.context
        )
        layer_qk.append(qk)
        layer_context.append(context)
    qk_mean = torch.stack(layer_qk).mean()
    context_mean = torch.stack(layer_context).mean()
    total = (
        float(attention_weight) * qk_mean
        + float(context_weight) * context_mean
    )
    return total, {
        "all_layer_qk_attention_kl_mean": qk_mean,
        "all_layer_context_normalized_mse_mean": context_mean,
        "layer_qk_attention_kl": torch.stack(layer_qk),
        "layer_context_normalized_mse": torch.stack(layer_context),
    }
