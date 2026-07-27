"""OLMo-2-only static frequency/LoRA contract for exact long generation.

The candidate preserves Native pairs 0..51 and replaces only the twelve
lowest-frequency pairs 52..63 with their EVQ-Cosh values. Q/K LoRA updates
are masked to the corresponding rotary coordinates in every attention head;
V/O and all other base weights remain frozen.
"""

from __future__ import annotations

from typing import Any

import torch

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    MODEL_CONTRACT,
    endpoint_evq_inv_freq,
    endpoint_geo_inv_freq,
    tensor_sha256,
)


NATIVE_PAIR_COUNT = 52
EVQ_PAIR_INDICES = tuple(range(NATIVE_PAIR_COUNT, 64))
EVQ_COORDINATES_PER_HEAD = (
    *EVQ_PAIR_INDICES,
    *(index + 64 for index in EVQ_PAIR_INDICES),
)
HYBRID_FREQUENCY_NAME = "hybrid_evq_low12"
HYBRID_FREQUENCY_SHA256 = (
    "54b8d165a612bcd3e08e0bab0adf68264a39240a51af9115092a35ec9d5f0695"
)
QK_EVQ_TAIL_OUTPUT_MASK_SHA256 = (
    "eba79ecb7c38fc22b8e8560a6e6413061b4fabb740390982bcbb63130b63ae48"
)


def hybrid_evq_low12_inv_freq() -> torch.Tensor:
    native = endpoint_geo_inv_freq().clone()
    evq = endpoint_evq_inv_freq()
    hybrid = native.clone()
    hybrid[list(EVQ_PAIR_INDICES)] = evq[list(EVQ_PAIR_INDICES)]
    if tensor_sha256(hybrid) != HYBRID_FREQUENCY_SHA256:
        raise RuntimeError("OLMo hybrid frequency identity drift")
    return hybrid


def apply_hybrid_evq_low12(model: Any) -> dict[str, Any]:
    active = model.model.rotary_emb.inv_freq
    native = endpoint_geo_inv_freq()
    if not torch.equal(
        active.detach().cpu().to(torch.float32),
        native,
    ):
        raise RuntimeError("OLMo model is not at the Native frequency anchor")
    hybrid = hybrid_evq_low12_inv_freq()
    with torch.no_grad():
        active.copy_(hybrid.to(device=active.device, dtype=active.dtype))
    model.model.rotary_emb.original_inv_freq = active
    realized = active.detach().cpu().to(torch.float32)
    if not torch.equal(realized, hybrid):
        raise RuntimeError("OLMo hybrid frequency write drift")
    return {
        "active_frequency": HYBRID_FREQUENCY_NAME,
        "active_sha256_float32": tensor_sha256(realized),
        "native_pair_indices": list(range(NATIVE_PAIR_COUNT)),
        "evq_pair_indices": list(EVQ_PAIR_INDICES),
        "native_pair_count": NATIVE_PAIR_COUNT,
        "evq_pair_count": len(EVQ_PAIR_INDICES),
    }


def qk_evq_tail_output_mask(config: Any) -> torch.Tensor:
    hidden_size = int(config.hidden_size)
    attention_heads = int(config.num_attention_heads)
    key_value_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", hidden_size // attention_heads))
    expected = MODEL_CONTRACT
    if (
        hidden_size != int(expected["hidden_size"])
        or attention_heads != int(expected["num_attention_heads"])
        or key_value_heads != int(expected["num_key_value_heads"])
        or head_dim != int(expected["head_dim"])
    ):
        raise RuntimeError("OLMo-2 1.485B Q/K mask architecture drift")
    mask = torch.zeros(hidden_size, dtype=torch.float32)
    for head in range(attention_heads):
        offset = head * head_dim
        mask[
            [
                offset + coordinate
                for coordinate in EVQ_COORDINATES_PER_HEAD
            ]
        ] = 1.0
    expected_active = (
        attention_heads * 2 * len(EVQ_PAIR_INDICES)
    )
    if int(mask.sum().item()) != expected_active:
        raise RuntimeError("OLMo Q/K EVQ-tail mask cardinality drift")
    if tensor_sha256(mask) != QK_EVQ_TAIL_OUTPUT_MASK_SHA256:
        raise RuntimeError("OLMo Q/K EVQ-tail mask identity drift")
    return mask


def phase_geometry_receipt() -> dict[str, Any]:
    native = endpoint_geo_inv_freq().to(torch.float64)
    hybrid = hybrid_evq_low12_inv_freq().to(torch.float64)
    delta = (hybrid - native).abs()
    lengths = {}
    for length in (4_096, 8_192, 16_384):
        phase = delta * (length - 1)
        lengths[str(length)] = {
            "maximum_absolute_phase_shift": float(phase.max()),
            "pairs_above_0p5_radians": int((phase > 0.5).sum()),
            "pairs_above_1_radian": int((phase > 1.0).sum()),
            "mean_geometry_alignment_cosine": float(
                torch.cos(phase).mean()
            ),
        }
    return {
        "frequency": HYBRID_FREQUENCY_NAME,
        "frequency_sha256_float32": HYBRID_FREQUENCY_SHA256,
        "native_pairs": [0, NATIVE_PAIR_COUNT - 1],
        "evq_pairs": [NATIVE_PAIR_COUNT, 63],
        "lengths": lengths,
        "claim_boundary": (
            "geometry-only feasibility; not trained capability or retention"
        ),
    }
