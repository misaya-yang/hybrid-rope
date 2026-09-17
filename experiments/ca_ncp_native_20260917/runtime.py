"""Install frozen CA-NCP rank-2 planes after Q/K norm and before RoPE."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import apply_group_planes_torch
from .io_utils import file_sha256


def load_alignment(path: Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        data = {name: payload[name] for name in payload.files}
    required = {
        "a", "b", "v_real", "v_imag", "identity", "active_indices",
        "carrier_local", "method_receipt_sha256", "statistics_receipt_sha256",
    }
    if not required.issubset(data):
        raise ValueError(f"alignment lacks fields: {sorted(required - set(data))}")
    a, b, vr, vi = data["a"], data["b"], data["v_real"], data["v_imag"]
    if a.ndim != 2 or b.shape != a.shape or vr.shape != vi.shape or vr.shape[:2] != a.shape:
        raise ValueError("alignment plane shapes differ")
    if vr.shape[2] != data["active_indices"].size:
        raise ValueError("alignment active dimension differs")
    if not all(np.isfinite(value).all() for value in (a, b, vr, vi)):
        raise ValueError("alignment contains nonfinite plane coefficients")
    return data


def _reshape(value, *, heads: int, head_dim: int):
    shape = tuple(value.shape)
    if value.ndim == 3 and value.shape[-1] == heads * head_dim:
        return value.view(*value.shape[:-1], heads, head_dim), ("flat", shape)
    if value.ndim == 4 and value.shape[-2:] == (heads, head_dim):
        return value, ("bthd", shape)
    if value.ndim == 4 and value.shape[1] == heads and value.shape[-1] == head_dim:
        return value.transpose(1, 2), ("bhtd", shape)
    raise RuntimeError(f"unsupported post-norm Q/K layout {shape}")


def _restore(value, layout):
    kind, shape = layout
    if kind == "flat":
        return value.reshape(shape)
    if kind == "bthd":
        return value
    if kind == "bhtd":
        return value.transpose(1, 2)
    raise AssertionError(kind)


def transform_post_norm(
    value,
    *,
    heads: int,
    head_dim: int,
    active,
    carrier_local: int,
    a,
    b,
    v_real,
    v_imag,
    head_to_group,
):
    shaped, layout = _reshape(value, heads=heads, head_dim=head_dim)
    transformed = apply_group_planes_torch(
        shaped, active, carrier_local, a, b, v_real, v_imag, head_to_group,
    )
    if transformed is shaped:
        return value
    return _restore(transformed, layout)


def install_alignment(model, alignment_path: Path) -> tuple[list, dict]:
    """Register non-trainable plane buffers and Q/K post-norm hooks."""
    import torch
    from torch import nn

    data = load_alignment(alignment_path)
    layers = list(model.model.layers)
    if data["a"].shape[0] != len(layers):
        raise ValueError("alignment layer count differs from model")
    config = model.config
    query_heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", query_heads))
    head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // query_heads)
    if query_heads % kv_heads or data["a"].shape[1] != kv_heads:
        raise ValueError("alignment KV group count differs from model")
    active = np.asarray(data["active_indices"], dtype=np.int64)
    if np.any(active < 0) or np.any(active >= head_dim // 2):
        raise ValueError("alignment active indices exceed rotary pairs")
    carrier_local = int(data["carrier_local"])
    handles = []
    nonidentity = 0

    class FrozenPlanes(nn.Module):
        def __init__(self, layer: int):
            super().__init__()
            self.register_buffer("active", torch.as_tensor(active, dtype=torch.long), persistent=True)
            self.register_buffer("a", torch.as_tensor(data["a"][layer], dtype=torch.float32), persistent=True)
            self.register_buffer("b", torch.as_tensor(data["b"][layer], dtype=torch.float32), persistent=True)
            self.register_buffer("v_real", torch.as_tensor(data["v_real"][layer], dtype=torch.float32), persistent=True)
            self.register_buffer("v_imag", torch.as_tensor(data["v_imag"][layer], dtype=torch.float32), persistent=True)
            self.carrier_local = carrier_local

        def apply(self, value, *, heads: int, head_to_group):
            return transform_post_norm(
                value,
                heads=heads,
                head_dim=head_dim,
                active=self.active,
                carrier_local=self.carrier_local,
                a=self.a,
                b=self.b,
                v_real=self.v_real,
                v_imag=self.v_imag,
                head_to_group=head_to_group,
            )

    for layer_index, block in enumerate(layers):
        attention = block.self_attn
        if hasattr(attention, "ca_ncp_alignment"):
            raise RuntimeError("CA-NCP alignment is already installed")
        if not hasattr(attention, "q_norm") or not hasattr(attention, "k_norm"):
            raise RuntimeError("model lacks post-norm/pre-RoPE Q/K hook points")
        planes = FrozenPlanes(layer_index).to(device=next(attention.parameters()).device)
        attention.add_module("ca_ncp_alignment", planes)
        q_map = np.repeat(np.arange(kv_heads, dtype=np.int64), query_heads // kv_heads)
        k_map = np.arange(kv_heads, dtype=np.int64)

        def q_hook(_module, _inputs, output, *, p=planes, mapping=q_map):
            return p.apply(output, heads=query_heads, head_to_group=mapping)

        def k_hook(_module, _inputs, output, *, p=planes, mapping=k_map):
            return p.apply(output, heads=kv_heads, head_to_group=mapping)

        handles.append(attention.q_norm.register_forward_hook(q_hook))
        handles.append(attention.k_norm.register_forward_hook(k_hook))
        nonidentity += int(np.sum((np.abs(data["b"][layer_index]) > 0) | (np.abs(data["a"][layer_index] - 1) > 0)))
    if any(parameter.requires_grad for block in layers for parameter in block.self_attn.ca_ncp_alignment.parameters()):
        raise AssertionError("alignment unexpectedly has trainable parameters")
    receipt = {
        "status": "CA_NCP_ALIGNMENT_INSTALLED",
        "alignment_path": str(Path(alignment_path).resolve()),
        "alignment_sha256": file_sha256(alignment_path),
        "layers": len(layers),
        "kv_groups": kv_heads,
        "query_heads": query_heads,
        "head_dim": head_dim,
        "active_indices_zero_based": active.tolist(),
        "carrier_local_zero_based": carrier_local,
        "nonidentity_planes": nonidentity,
        "runtime_plane_dtype": "float32",
        "output_dtype": "original Q/K dtype",
        "hook_location": "q_norm/k_norm forward output before view and RoPE",
        "method_receipt_sha256": str(data["method_receipt_sha256"]),
        "statistics_receipt_sha256": str(data["statistics_receipt_sha256"]),
    }
    return handles, receipt
