"""CUDA and Flash-only qualification for the active 32 GiB Ada server."""

from __future__ import annotations

from typing import Any

import torch


def validate(old: Any) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    properties = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    architectures = list(torch.cuda.get_arch_list())
    compatible = [
        value
        for value in architectures
        if value.startswith("sm_")
        and value[3:].isdigit()
        and int(value[3:]) // 10 == major
        and int(value[3:]) % 10 <= minor
    ]
    if not compatible:
        raise RuntimeError(
            f"no compatible cubin for capability {major}.{minor}: {architectures}"
        )
    kernels = old.configure_cuda_kernels()
    q = torch.randn(1, 12, 128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    output.float().square().mean().backward()
    torch.cuda.synchronize()
    if not all(torch.isfinite(value).all() for value in (output, q.grad, k.grad, v.grad)):
        raise RuntimeError("Flash forward/backward produced a non-finite value")
    return {
        "name": properties.name,
        "capability": [major, minor],
        "compiled_architectures": architectures,
        "compatible_cubins": compatible,
        "total_memory_bytes": int(properties.total_memory),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "kernels": kernels,
        "flash_forward_backward": "PASS",
    }
