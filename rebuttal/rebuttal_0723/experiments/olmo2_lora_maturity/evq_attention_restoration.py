#!/usr/bin/env python3
"""Shared contracts for Native-teacher/full-EVQ attention restoration.

The training objective is intentionally stronger than Q/Q, K/K, V/V
self-relation matching alone.  It includes the actual post-RoPE causal QK
attention distribution and the resulting per-head A@V context.
"""

from __future__ import annotations

import importlib
import hashlib
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    LoRALinear,
)


ATTENTION_BACKEND = "evq_attention_restoration_sdpa"
METHOD_ID = "native_teacher_full_evq_attention_function_restoration_v1"
PREPARED_STATUS = "OLMO2_EVQ_ATTENTION_RESTORATION_PREPARED_NO_GPU"
READY_STATUS = "OLMO2_EVQ_ATTENTION_RESTORATION_GPU_READY"
RESULT_STATUS = "OLMO2_EVQ_ATTENTION_RESTORATION_COMPLETE"
SEQUENCE_LENGTH = 4_096
FINAL_LAYER_INDEX = 15
LINEARARD_COMMIT = "23866f68a8b65da796c75439a06d4cb996bcb7bb"
LINEARARD_KERNEL_SHA256 = {
    "integration.py": (
        "9bbc08e87cacdec086a782f4d0a1034b868105c8ccdec65040d0a9732695f0eb"
    ),
    "kernel_A.py": (
        "20d6b327973355d5a948f86f8ab29f3d45e1a016665f612a6cc78bf26f849bb6"
    ),
    "kernel_B.py": (
        "8fcdb57a82e935bf9899a6252885f8d2da9458c26fe9dd9666f06fb3bc4873ec"
    ),
    "kernel_C.py": (
        "fe11c409663b0629ae2504af116ebc2d47138ba3da9bb42dc6da7bd14d4226f8"
    ),
}


@dataclass
class CapturedAttention:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    context: torch.Tensor


class RelationCapture:
    """Capture one post-RoPE attention layer for teacher and student."""

    def __init__(self, layer_index: int = FINAL_LAYER_INDEX) -> None:
        self.layer_index = int(layer_index)
        self.values: dict[str, CapturedAttention] = {}

    def clear(self) -> None:
        self.values.clear()

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
        layer_index = getattr(module, "layer_idx", None)
        if layer_index != self.layer_index:
            return
        if mode not in {"teacher", "student"}:
            raise RuntimeError(f"unknown capture mode {mode!r}")
        if mode in self.values:
            raise RuntimeError(
                "attention capture repeated; gradient checkpointing or an "
                "unexpected second forward is active"
            )
        if mode == "teacher":
            item = CapturedAttention(
                query=query.detach().contiguous(),
                key=key.detach().contiguous(),
                value=value.detach().contiguous(),
                context=context.detach().contiguous(),
            )
        else:
            item = CapturedAttention(
                query=query.contiguous(),
                key=key.contiguous(),
                value=value.contiguous(),
                context=context.contiguous(),
            )
        self.values[mode] = item

    def require_pair(self) -> tuple[CapturedAttention, CapturedAttention]:
        if set(self.values) != {"teacher", "student"}:
            raise RuntimeError(
                f"incomplete attention capture: {sorted(self.values)}"
            )
        return self.values["teacher"], self.values["student"]


def relation_capture_mask(
    *,
    attention_mask: torch.Tensor | None = None,
    **_: Any,
) -> None:
    """Use SDPA's causal flag for full, unpadded 4K restoration rows."""
    if attention_mask is not None:
        raise RuntimeError(
            "attention restoration admits only full unpadded rows"
        )
    return None


def relation_capture_sdpa(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    dropout: float = 0.0,
    scaling: float | None = None,
    relation_capture: RelationCapture | None = None,
    relation_mode: str | None = None,
    **_: Any,
) -> tuple[torch.Tensor, None]:
    """Flash-SDPA-compatible attention that records post-RoPE Q/K/V/A@V."""
    if attention_mask is not None:
        raise RuntimeError(
            "attention restoration received a materialized mask"
        )
    if query.shape[-2] != key.shape[-2]:
        raise RuntimeError(
            "attention restoration does not admit KV-cache decoding"
        )
    if float(dropout) != 0.0:
        raise RuntimeError("attention restoration requires zero dropout")
    context = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        scale=scaling,
        is_causal=query.shape[-2] > 1,
    )
    if relation_capture is not None or relation_mode is not None:
        if relation_capture is None or relation_mode is None:
            raise RuntimeError("capture object and mode must be supplied together")
        relation_capture.store(
            mode=relation_mode,
            module=module,
            query=query,
            key=key,
            value=value,
            context=context,
        )
    return context.transpose(1, 2).contiguous(), None


def configure_relation_capture_attention(model: nn.Module) -> None:
    AttentionInterface.register(ATTENTION_BACKEND, relation_capture_sdpa)
    AttentionMaskInterface.register(
        ATTENTION_BACKEND, relation_capture_mask
    )
    if ALL_ATTENTION_FUNCTIONS[ATTENTION_BACKEND] is not relation_capture_sdpa:
        raise RuntimeError("attention backend registration identity drift")
    if (
        ALL_MASK_ATTENTION_FUNCTIONS[ATTENTION_BACKEND]
        is not relation_capture_mask
    ):
        raise RuntimeError("attention-mask backend registration identity drift")
    model.config._attn_implementation = ATTENTION_BACKEND


def install_qkv_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
) -> int:
    """Freeze the backbone and install zero-output LoRA on every Q/K/V."""
    if int(rank) <= 0:
        raise ValueError("LoRA rank must be positive")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in model.model.layers:
        attention = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj"):
            base = getattr(attention, name)
            if isinstance(base, LoRALinear):
                raise RuntimeError(f"{name} already contains LoRA")
            setattr(
                attention,
                name,
                LoRALinear(base, rank=int(rank), alpha=float(alpha)),
            )
    count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if count <= 0:
        raise RuntimeError("QKV LoRA installation created no trainable tensors")
    return int(count)


def _map_kv_heads(
    value: torch.Tensor,
    *,
    query_heads: int,
) -> torch.Tensor:
    key_heads = int(value.shape[1])
    if query_heads % key_heads != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    if key_heads == query_heads:
        return value
    mapping = torch.arange(query_heads, device=value.device) // (
        query_heads // key_heads
    )
    return value[:, mapping, :, :]


def dense_relation_forward_kl(
    student_left: torch.Tensor,
    student_right: torch.Tensor,
    teacher_left: torch.Tensor,
    teacher_right: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    causal: bool = True,
    student_scale: float | None = None,
    teacher_scale: float | None = None,
) -> torch.Tensor:
    """Reference forward KL for CPU tests and small GPU parity checks.

    The return value is a sum over batch, query-head and valid query rows,
    matching the normalization contract used by the linear-memory kernel.
    """
    tensors = (
        student_left,
        student_right,
        teacher_left,
        teacher_right,
    )
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("relation tensors must have shape [B,H,T,D]")
    batch, query_heads, length, student_dim = student_left.shape
    if teacher_left.shape[:3] != (batch, query_heads, length):
        raise ValueError("teacher/student left tensor shape drift")
    if student_right.shape[0] != batch or student_right.shape[2] != length:
        raise ValueError("student right tensor shape drift")
    if teacher_right.shape[0] != batch or teacher_right.shape[2] != length:
        raise ValueError("teacher right tensor shape drift")
    student_right_mapped = _map_kv_heads(
        student_right, query_heads=query_heads
    )
    teacher_right_mapped = _map_kv_heads(
        teacher_right, query_heads=query_heads
    )
    if student_right_mapped.shape[-1] != student_dim:
        raise ValueError("student relation dimension mismatch")
    teacher_dim = int(teacher_left.shape[-1])
    if teacher_right_mapped.shape[-1] != teacher_dim:
        raise ValueError("teacher relation dimension mismatch")
    student_scale = (
        float(student_scale)
        if student_scale is not None
        else 1.0 / math.sqrt(student_dim)
    )
    teacher_scale = (
        float(teacher_scale)
        if teacher_scale is not None
        else 1.0 / math.sqrt(teacher_dim)
    )
    student_logits = torch.matmul(
        student_left.float(),
        student_right_mapped.float().transpose(-1, -2),
    ) * student_scale
    teacher_logits = torch.matmul(
        teacher_left.float(),
        teacher_right_mapped.float().transpose(-1, -2),
    ) * teacher_scale
    if attention_mask is None:
        token_valid = torch.ones(
            (batch, length),
            device=student_left.device,
            dtype=torch.bool,
        )
    else:
        if attention_mask.shape != (batch, length):
            raise ValueError("attention mask shape drift")
        token_valid = attention_mask.to(torch.bool)
    valid = token_valid[:, None, :, None] & token_valid[:, None, None, :]
    if causal:
        causal_mask = torch.ones(
            (length, length),
            device=student_left.device,
            dtype=torch.bool,
        ).tril()
        valid = valid & causal_mask[None, None, :, :]
    valid_rows = valid.any(dim=-1)
    minimum = torch.finfo(student_logits.dtype).min
    student_logits = student_logits.masked_fill(~valid, minimum)
    teacher_logits = teacher_logits.masked_fill(~valid, minimum)
    student_logits = torch.where(
        valid_rows[..., None], student_logits, torch.zeros_like(student_logits)
    )
    teacher_logits = torch.where(
        valid_rows[..., None], teacher_logits, torch.zeros_like(teacher_logits)
    )
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    row_kl = (
        teacher_probs * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)
    return torch.where(
        valid_rows, row_kl, torch.zeros_like(row_kl)
    ).sum()


def normalized_context_mse(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError("teacher/student context shape drift")
    teacher_fp32 = teacher.float()
    denominator = teacher_fp32.square().mean().detach().clamp_min(1e-6)
    return (student.float() - teacher_fp32).square().mean() / denominator


def import_linearard_kernel(
    root: Path,
) -> Callable[..., torch.Tensor]:
    """Import the pinned upstream exact linear-memory KL operator."""
    root = root.resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != LINEARARD_COMMIT:
        raise RuntimeError(
            f"LinearARD commit drift: {commit} != {LINEARARD_COMMIT}"
        )
    kernel_root = root / "kernels" / "attention_KL"
    for name, expected_sha256 in LINEARARD_KERNEL_SHA256.items():
        candidate = kernel_root / name
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise RuntimeError(f"LinearARD {name} hash drift")
    for name in tuple(sys.modules):
        if name == "kernels" or name.startswith("kernels."):
            del sys.modules[name]
    root_text = str(root)
    sys.path.insert(0, root_text)
    module = importlib.import_module("kernels.attention_KL.integration")
    expected_module = (kernel_root / "integration.py").resolve()
    actual_module = Path(str(module.__file__)).resolve()
    if actual_module != expected_module:
        raise RuntimeError(
            f"LinearARD module path drift: {actual_module} != {expected_module}"
        )
    operator = getattr(module, "attn_kl_align", None)
    if not callable(operator):
        raise RuntimeError("LinearARD attn_kl_align is unavailable")
    return operator


def relation_loss(
    *,
    teacher: CapturedAttention,
    student: CapturedAttention,
    kernel: Callable[..., torch.Tensor],
    attention_weight: float,
    context_weight: float,
    self_relation_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute normalized QK, context and auxiliary self-relation losses."""
    batch, heads, length, head_dim = student.query.shape
    if teacher.query.shape != student.query.shape:
        raise RuntimeError("teacher/student query capture shape drift")
    scale = 1.0 / math.sqrt(int(head_dim))
    denominator = float(batch * heads * length)

    def aligned(
        student_left: torch.Tensor,
        student_right: torch.Tensor,
        teacher_left: torch.Tensor,
        teacher_right: torch.Tensor,
    ) -> torch.Tensor:
        return kernel(
            student_left,
            student_right,
            teacher_left,
            teacher_right,
            attn_mask=None,
            causal=True,
            sm_scale_s=scale,
            sm_scale_t=scale,
        ) / denominator

    qk = aligned(
        student.query,
        student.key,
        teacher.query,
        teacher.key,
    )
    qq = aligned(
        student.query,
        student.query,
        teacher.query,
        teacher.query,
    )
    kk = aligned(
        student.key,
        student.key,
        teacher.key,
        teacher.key,
    )
    vv = aligned(
        student.value,
        student.value,
        teacher.value,
        teacher.value,
    )
    context = normalized_context_mse(student.context, teacher.context)
    auxiliary = (qq + kk + vv) / 3.0
    total = (
        float(attention_weight) * qk
        + float(context_weight) * context
        + float(self_relation_weight) * auxiliary
    )
    return total, {
        "qk_attention_kl": qk,
        "context_normalized_mse": context,
        "qq_relation_kl": qq,
        "kk_relation_kl": kk,
        "vv_relation_kl": vv,
        "self_relation_mean": auxiliary,
    }
