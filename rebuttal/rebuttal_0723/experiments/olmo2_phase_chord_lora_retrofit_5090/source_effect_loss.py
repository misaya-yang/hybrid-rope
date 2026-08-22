"""Tensor-only source-effect loss with Native-teacher positive gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SourceEffectConfig:
    primary_tokens: int = 8
    primary_weight: float = 1.0
    continuation_weight: float = 0.25
    effect_weight: float = 1.0
    margin_weight: float = 1.0
    correct_ce_weight: float = 0.1
    source_margin: float = 1.0
    min_teacher_effect: float = 0.0

    def validate(self) -> None:
        if int(self.primary_tokens) <= 0:
            raise ValueError("primary_tokens must be positive")
        for name in ("primary_weight", "continuation_weight", "effect_weight", "margin_weight", "correct_ce_weight", "source_margin"):
            value = float(getattr(self, name))
            if not torch.isfinite(torch.tensor(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not torch.isfinite(torch.tensor(float(self.min_teacher_effect))):
            raise ValueError("min_teacher_effect must be finite")


def validate_target_shift(input_ids: torch.Tensor, target_positions: torch.Tensor, target_tokens: torch.Tensor) -> None:
    if input_ids.ndim != 2 or target_positions.ndim != 2 or target_tokens.ndim != 2:
        raise ValueError("inputs, positions, and targets must be rank two")
    if target_positions.shape != target_tokens.shape or input_ids.shape[0] != target_tokens.shape[0]:
        raise ValueError("target shift shapes drift")
    if (target_positions < 1).any() or (target_positions >= input_ids.shape[1]).any():
        raise ValueError("target positions must be valid p>=1 positions")
    if target_positions.shape[1] > 1 and not torch.equal(target_positions[:, 1:], target_positions[:, :-1] + 1):
        raise ValueError("target positions must be contiguous")
    if not torch.equal(input_ids.gather(1, target_positions), target_tokens):
        raise ValueError("target tokens do not match input ids")


def gather_target_position_logits(sequence_logits: torch.Tensor, input_ids: torch.Tensor, target_positions: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    validate_target_shift(input_ids, target_positions, target_tokens)
    if sequence_logits.shape[:2] != input_ids.shape:
        raise ValueError("sequence logits and input ids drift")
    previous = target_positions - 1
    return sequence_logits.gather(1, previous.unsqueeze(-1).expand(-1, -1, sequence_logits.shape[-1]))


def source_effect_loss(
    *, teacher_correct: torch.Tensor, teacher_swapped: torch.Tensor,
    student_correct: torch.Tensor, student_swapped: torch.Tensor,
    target_tokens: torch.Tensor,
    values_are_logits: bool = True,
    config: SourceEffectConfig = SourceEffectConfig(),
    shift_contract: Optional[Mapping[str, tuple[torch.Tensor, torch.Tensor]]] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    config.validate()
    if shift_contract is not None:
        if set(shift_contract) != {"teacher_correct", "teacher_swapped", "student_correct", "student_swapped"}:
            raise ValueError("shift contract must cover all four branches")
        for ids, positions in shift_contract.values():
            validate_target_shift(ids, positions, target_tokens)
    values = (teacher_correct, teacher_swapped, student_correct, student_swapped)
    expected_ndim = 3 if values_are_logits else 2
    if any(value.ndim != expected_ndim or not torch.isfinite(value).all() for value in values):
        raise ValueError("branch predictions have invalid shape or non-finite values")
    if any(value.shape[:2] != target_tokens.shape for value in values):
        raise ValueError("branch target shape drift")
    if values_are_logits:
        target = target_tokens.unsqueeze(-1)
        teacher_correct_lp = F.log_softmax(teacher_correct.float(), -1).gather(-1, target).squeeze(-1).detach()
        teacher_swapped_lp = F.log_softmax(teacher_swapped.float(), -1).gather(-1, target).squeeze(-1).detach()
        student_correct_lp = F.log_softmax(student_correct.float(), -1).gather(-1, target).squeeze(-1)
        student_swapped_lp = F.log_softmax(student_swapped.float(), -1).gather(-1, target).squeeze(-1)
    else:
        teacher_correct_lp = teacher_correct.detach().float()
        teacher_swapped_lp = teacher_swapped.detach().float()
        student_correct_lp = student_correct.float()
        student_swapped_lp = student_swapped.float()
    teacher_effect = teacher_correct_lp - teacher_swapped_lp
    selected = teacher_effect > float(config.min_teacher_effect)
    if not selected.any():
        raise ValueError("Native teacher identifies no positive source effect")
    student_effect = student_correct_lp - student_swapped_lp
    weights = torch.full_like(student_effect, float(config.continuation_weight))
    weights[:, : min(int(config.primary_tokens), weights.shape[1])] = float(config.primary_weight)
    weights = weights * selected.to(weights.dtype)
    denominator = weights.sum().clamp_min(1.0)
    effect = (
        F.smooth_l1_loss(student_effect, teacher_effect, reduction="none")
        * weights
    ).sum() / denominator
    margin = (
        F.softplus(float(config.source_margin) - student_effect) * weights
    ).sum() / denominator
    all_weights = torch.full_like(student_correct_lp, float(config.continuation_weight))
    all_weights[:, : min(int(config.primary_tokens), all_weights.shape[1])] = float(config.primary_weight)
    ce = -(student_correct_lp * all_weights).sum() / all_weights.sum().clamp_min(1.0)
    total = float(config.effect_weight) * effect + float(config.margin_weight) * margin + float(config.correct_ce_weight) * ce
    return total, {
        "loss": float(total.detach()),
        "teacher_selected_rate": float(selected.float().mean()),
        "source_follow_rate": float((student_effect > 0).float()[selected].mean()),
        "source_follow_rate_first8": float((student_effect[:, : min(8, student_effect.shape[1])] > 0).float()[selected[:, : min(8, selected.shape[1])]].mean()) if selected[:, : min(8, selected.shape[1])].any() else 0.0,
        "margin_violation_rate": float((student_effect < float(config.source_margin))[selected].float().mean()),
        "teacher_effect_mean": float(teacher_effect[selected].mean()),
        "teacher_selected_tokens": float(selected.sum()),
        "weighted_target_tokens": float(all_weights.sum()),
    }
