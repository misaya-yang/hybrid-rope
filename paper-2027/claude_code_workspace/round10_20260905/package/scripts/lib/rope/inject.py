#!/usr/bin/env python3
"""In-place RoPE injection and hashing helpers."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import torch


def find_rotary_modules_with_inv_freq(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    out: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if hasattr(module, "inv_freq"):
            inv = getattr(module, "inv_freq")
            if torch.is_tensor(inv):
                out.append((name, module))
    return out


def clear_rotary_cache(module: torch.nn.Module) -> None:
    for attr in (
        "max_seq_len_cached",
        "_cos_cached",
        "_sin_cached",
        "cos_cached",
        "sin_cached",
        "_cos_cache",
        "_sin_cache",
    ):
        if not hasattr(module, attr):
            continue
        cur = getattr(module, attr)
        if isinstance(cur, (int, float)):
            setattr(module, attr, 0)
        else:
            setattr(module, attr, None)


def apply_inv_freq_inplace(
    model: torch.nn.Module,
    inv_freq: torch.Tensor,
) -> Dict[str, object]:
    modules = find_rotary_modules_with_inv_freq(model)
    if not modules:
        raise RuntimeError("No rotary modules with inv_freq found in model.")

    changed = []
    expected = inv_freq.detach().cpu().view(-1)
    for name, module in modules:
        old = module.inv_freq
        if old.ndim != 1:
            raise RuntimeError(f"{name}.inv_freq is not 1D: shape={tuple(old.shape)}")
        if old.numel() != expected.numel():
            raise RuntimeError(
                f"Shape mismatch at {name}: old={tuple(old.shape)} new={tuple(expected.shape)}"
            )
        before = old.detach().clone()
        with torch.no_grad():
            old.copy_(expected.to(device=old.device, dtype=old.dtype))
        clear_rotary_cache(module)
        if not torch.equal(before, old):
            changed.append(name)

    return {
        "patched_count": len(modules),
        "changed_modules": changed,
        "all_modules": [name for name, _ in modules],
    }


def hash_tensor_sha256(x: torch.Tensor) -> str:
    arr = x.detach().to("cpu", dtype=torch.float64).contiguous().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()
