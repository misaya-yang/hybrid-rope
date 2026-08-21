"""Offline R3-prime heterogeneous RoPE preparation and preflight helpers.

This package is intentionally CPU-only.  It prepares per-layer frequency
tables, installs them without changing an attention kernel, and emits
reviewable dry-run receipts.  It does not launch training or GPU inference.
"""

from .protocol import (
    HeterogeneousRopePlan,
    LayerFrequencySpec,
    build_layer_inv_freqs,
    build_model_config,
    extend_layerwise_rope,
    hash_tensor_raw,
    install_layerwise_rope,
    load_r0_json,
    model_parameter_contract,
    plan_from_values,
    per_head_feasibility_gate,
    realized_frequency_receipt,
)

__all__ = [
    "HeterogeneousRopePlan",
    "LayerFrequencySpec",
    "build_layer_inv_freqs",
    "build_model_config",
    "extend_layerwise_rope",
    "hash_tensor_raw",
    "install_layerwise_rope",
    "load_r0_json",
    "model_parameter_contract",
    "plan_from_values",
    "per_head_feasibility_gate",
    "realized_frequency_receipt",
]
