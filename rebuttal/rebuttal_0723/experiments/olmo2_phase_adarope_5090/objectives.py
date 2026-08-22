"""CPU-safe objective and component-gate contracts for phase AdaRoPE."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Stage0ObjectiveConfig:
    steps: int = 300
    rank: int = 64
    alpha: float = 128.0
    margin: float = 1.0
    margin_weight: float = 1.0
    answer_ce_weight: float = 1.0
    eos_weight: float = 1.0

    def validate(self) -> None:
        if self.steps != 300 or self.rank != 64 or self.alpha != 128.0:
            raise ValueError("Stage0 registered budget/scope drift")
        if self.margin < 0 or self.margin_weight < 0 or self.answer_ce_weight <= 0 or self.eos_weight < 0:
            raise ValueError("Stage0 objective weights are invalid")


def selected_hidden_lm_head(
    hidden_states: torch.Tensor,
    target_positions: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    input_ids: torch.Tensor | None = None,
    target_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the frozen lm_head at ``p-1`` for target token at position ``p``."""
    if hidden_states.ndim != 3 or target_positions.ndim != 2:
        raise ValueError("hidden_states must be [B,S,H] and positions [B,T]")
    if target_positions.shape[0] != hidden_states.shape[0]:
        raise ValueError("selected hidden batch mismatch")
    if (target_positions < 1).any() or (target_positions >= hidden_states.shape[1]).any():
        raise ValueError("selected hidden position is out of range")
    if input_ids is not None:
        if input_ids.ndim != 2 or input_ids.shape[0] != hidden_states.shape[0] or input_ids.shape[1] != hidden_states.shape[1]:
            raise ValueError("input_ids shape mismatch")
        if target_tokens is None or target_tokens.shape != target_positions.shape:
            raise ValueError("target_tokens are required with input_ids")
        actual = input_ids.gather(1, target_positions)
        if not torch.equal(actual, target_tokens):
            raise ValueError("input_ids at target positions do not match target_tokens")
    prediction_positions = target_positions - 1
    selected = hidden_states.gather(
        1, prediction_positions.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    )
    return lm_head(selected)


def stage0_answer_margin_loss(
    *,
    logits: torch.Tensor,
    answer_tokens: torch.Tensor,
    alternate_tokens: torch.Tensor,
    input_ids: torch.Tensor | None = None,
    target_positions: torch.Tensor | None = None,
    eos_mask: torch.Tensor | None = None,
    config: Stage0ObjectiveConfig = Stage0ObjectiveConfig(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Answer CE plus correct-vs-alternate margin and explicit EOS weight."""
    config.validate()
    if logits.ndim != 3 or answer_tokens.ndim != 2 or alternate_tokens.shape != answer_tokens.shape:
        raise ValueError("Stage0 logits/token shapes drift")
    if logits.shape[:2] != answer_tokens.shape:
        raise ValueError("Stage0 sequence shape drift")
    if (input_ids is None) != (target_positions is None):
        raise ValueError("input_ids and target_positions must be supplied together")
    if input_ids is not None:
        if input_ids.ndim != 2 or target_positions is None or target_positions.shape != answer_tokens.shape:
            raise ValueError("Stage0 input/position shape drift")
        if (target_positions < 1).any() or (target_positions >= input_ids.shape[1]).any():
            raise ValueError("Stage0 target position is out of range")
        if not torch.equal(input_ids.gather(1, target_positions), answer_tokens):
            raise ValueError("Stage0 target token does not match input_ids")
    if (answer_tokens < 0).any() or (answer_tokens >= logits.shape[-1]).any() or (alternate_tokens < 0).any() or (alternate_tokens >= logits.shape[-1]).any():
        raise ValueError("answer or alternate token is outside vocabulary")
    answer = logits.float().gather(-1, answer_tokens.unsqueeze(-1)).squeeze(-1)
    alternate = logits.float().gather(-1, alternate_tokens.unsqueeze(-1)).squeeze(-1)
    weights = torch.ones_like(answer)
    if eos_mask is not None:
        if eos_mask.shape != answer.shape:
            raise ValueError("EOS mask shape drift")
        weights = torch.where(eos_mask.bool(), torch.full_like(weights, config.eos_weight), weights)
    ce = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        answer_tokens.reshape(-1),
        reduction="none",
    ).reshape_as(answer)
    ce = (ce * weights).sum() / weights.sum().clamp_min(1e-12)
    if eos_mask is None:
        margin_active = torch.ones_like(answer, dtype=torch.bool)
    else:
        if eos_mask.shape != answer.shape:
            raise ValueError("EOS mask shape drift")
        margin_active = ~eos_mask.bool()
    margin_terms = F.softplus(float(config.margin) - (answer - alternate))
    margin = margin_terms[margin_active].mean() if bool(margin_active.any()) else margin_terms.new_zeros(())
    total = float(config.answer_ce_weight) * ce + float(config.margin_weight) * margin
    return total, {
        "loss": float(total.detach()),
        "answer_ce": float(ce.detach()),
        "correct_alternate_margin_loss": float(margin.detach()),
        "correct_over_alternate_rate": float((answer > alternate).float().mean()),
        "eos_tokens": float(eos_mask.sum()) if eos_mask is not None else 0.0,
    }


@dataclass(frozen=True)
class ComponentGateConfig:
    nll_ceiling: float = 0.10
    minimum_source_effect: float = 0.0
    minimum_positive_fraction: float = 0.90

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_positive_fraction <= 1.0:
            raise ValueError("minimum_positive_fraction must be in [0,1]")


TOURNAMENT_ORDER = ("native_scale", "context_stretch_exp_negative", "phase_chord")


def select_tournament_winner(evaluations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Select at most one Stage1 candidate against the paired null arm."""
    if set(evaluations) != set(TOURNAMENT_ORDER):
        raise ValueError("winner selection requires exactly the three registered candidates")
    candidates: list[dict[str, Any]] = []
    shared_docs: set[str] | None = None
    shared_parent: str | None = None
    shared_data_manifest: str | None = None
    shared_data_root: str | None = None
    for arm in TOURNAMENT_ORDER:
        payload = evaluations.get(arm)
        if not isinstance(payload, Mapping) or payload.get("status") != "COMPLETE" or int(payload.get("examples", 0)) != 32:
            raise ValueError(f"invalid completed Stage1 evaluation: {arm}")
        rows = payload.get("per_document_rows")
        if not isinstance(rows, list) or len(rows) != 32:
            raise ValueError(f"Stage1 evaluation must contain 32 rows: {arm}")
        ordered = sorted(rows, key=lambda row: str(row.get("document_id", "")))
        doc_ids = {str(row.get("document_id", "")) for row in ordered}
        if len(doc_ids) != 32 or any(row.get("split") != "component_gate" for row in ordered):
            raise ValueError(f"Stage1 rows are not 32 unique component-gate documents: {arm}")
        comparison_hash = payload.get("comparison_receipt_sha256")
        candidate_hash = payload.get("parent_receipt_sha256")
        data_manifest = payload.get("data_manifest_sha256")
        data_root = payload.get("data_root_manifest_sha256")
        if payload.get("comparison_arm") != "lora_only_null" or not isinstance(comparison_hash, str) or len(comparison_hash) != 64 or not isinstance(candidate_hash, str) or len(candidate_hash) != 64:
            raise ValueError(f"Stage1 evaluation is not bound to the LoRA null: {arm}")
        if shared_docs is None:
            shared_docs, shared_parent = doc_ids, comparison_hash
            shared_data_manifest, shared_data_root = data_manifest, data_root
        if doc_ids != shared_docs or comparison_hash != shared_parent or data_manifest != shared_data_manifest or data_root != shared_data_root:
            raise ValueError("Stage1 candidate evaluations do not share rows/data/null")
        deltas = []
        rank_deltas = []
        for row in ordered:
            candidate, parent = row.get("candidate"), row.get("parent")
            if not isinstance(candidate, Mapping) or not isinstance(parent, Mapping):
                continue
            deltas.append({"document_id": str(row.get("document_id")), "nll_delta": float(candidate["nll"]) - float(parent["nll"]), "source_effect_delta": float(candidate["source_effect"]) - float(parent["source_effect"])})
            if "first_token_gold_rank" in candidate and "first_token_gold_rank" in parent:
                rank_deltas.append(float(candidate["first_token_gold_rank"]) - float(parent["first_token_gold_rank"]))
        if len(deltas) != len(ordered) or len(rank_deltas) != len(ordered):
            raise ValueError(f"Stage1 candidate rows lack complete paired metrics: {arm}")
        half = len(deltas) // 2
        halves = [deltas[:half], deltas[-half:]]
        retention = payload.get("retention_4k") or {}
        retention_delta = float(retention.get("candidate_minus_native", float("inf")))
        positive_fraction = sum(d["source_effect_delta"] > 0 for d in deltas) / len(deltas)
        mean_nll = sum(d["nll_delta"] for d in deltas) / len(deltas)
        qualified = bool(retention_delta <= 0.10 and positive_fraction >= 0.90 and sum(d["source_effect_delta"] for d in deltas) > 0 and all(sum(d["source_effect_delta"] for d in half_rows) > 0 for half_rows in halves) and mean_nll <= 0.10 and rank_deltas and sum(rank_deltas) / len(rank_deltas) < 0)
        payload_sha = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        candidates.append({"arm": arm, "qualified": qualified, "candidate_receipt_sha256": candidate_hash, "null_receipt_sha256": comparison_hash, "evaluation_payload_sha256": payload_sha, "mean_nll_delta": mean_nll, "mean_source_effect_delta": sum(d["source_effect_delta"] for d in deltas) / len(deltas), "mean_first_token_rank_delta": sum(rank_deltas) / len(rank_deltas), "retention_delta": retention_delta, "document_deltas": deltas, "positive_fraction": positive_fraction})
    qualified = [row for row in candidates if row["qualified"]]
    winner = max(qualified, key=lambda row: (row["mean_source_effect_delta"], -(row["mean_first_token_rank_delta"] or float("inf")), -TOURNAMENT_ORDER.index(row["arm"]))) if qualified else None
    return {"status": "PASS", "selected_arm": winner["arm"] if winner else "lora_only_null", "shared_null_receipt_sha256": shared_parent, "data_manifest_sha256": shared_data_manifest, "data_root_manifest_sha256": shared_data_root, "candidates": candidates, "qualification_rule": "retention<=0.10; positive_fraction>=0.90; mean/halves source delta>0; mean NLL<=0.10; mean first-token gold-rank delta<0"}


def phase_attribution_gate(
    phase_vs_moment: Mapping[str, Any],
    moment_vs_null: Mapping[str, Any],
) -> dict[str, Any]:
    """Gate the conditional full moment control without changing the winner."""
    if phase_vs_moment.get("status") != "COMPLETE" or moment_vs_null.get("status") != "COMPLETE":
        raise ValueError("phase attribution requires two complete evaluations")
    if phase_vs_moment.get("comparison_arm") != "moment_matched_same_sign_control":
        raise ValueError("phase attribution comparison is not the moment control")
    if moment_vs_null.get("comparison_arm") != "lora_only_null" or moment_vs_null.get("parent_receipt_sha256") != phase_vs_moment.get("comparison_receipt_sha256"):
        raise ValueError("moment attribution evidence does not bind the same control/null")
    if phase_vs_moment.get("data_manifest_sha256") != moment_vs_null.get("data_manifest_sha256"):
        raise ValueError("phase/moment attribution data identity drift")
    rows = phase_vs_moment.get("per_document_rows")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("phase attribution requires 32 paired documents")
    ordered = sorted(rows, key=lambda row: str(row.get("document_id", "")))
    if len({str(row.get("document_id", "")) for row in ordered}) != 32 or any(row.get("split") != "component_gate" for row in ordered):
        raise ValueError("phase attribution documents are malformed")
    deltas: list[dict[str, float | str]] = []
    rank_deltas: list[float] = []
    for row in ordered:
        candidate, parent = row.get("candidate"), row.get("parent")
        if not isinstance(candidate, Mapping) or not isinstance(parent, Mapping):
            raise ValueError("phase attribution row lacks paired metrics")
        deltas.append({
            "document_id": str(row.get("document_id")),
            "source_effect_delta": float(candidate["source_effect"]) - float(parent["source_effect"]),
            "nll_delta": float(candidate["nll"]) - float(parent["nll"]),
        })
        rank_deltas.append(float(candidate["first_token_gold_rank"]) - float(parent["first_token_gold_rank"]))
    half = len(deltas) // 2
    phase_retention = float((phase_vs_moment.get("retention_4k") or {}).get("candidate_minus_native", float("inf")))
    moment_retention = float((moment_vs_null.get("retention_4k") or {}).get("candidate_minus_native", float("inf")))
    positive_fraction = sum(float(row["source_effect_delta"]) > 0 for row in deltas) / len(deltas)
    mean_source = sum(float(row["source_effect_delta"]) for row in deltas) / len(deltas)
    mean_nll = sum(float(row["nll_delta"]) for row in deltas) / len(deltas)
    mean_rank = sum(rank_deltas) / len(rank_deltas)
    half_means = [
        sum(float(row["source_effect_delta"]) for row in group) / len(group)
        for group in (deltas[:half], deltas[half:])
    ]
    passed = bool(
        phase_retention <= 0.10
        and moment_retention <= 0.10
        and positive_fraction >= 0.90
        and mean_source > 0
        and all(value > 0 for value in half_means)
        and mean_nll <= 0.10
        and mean_rank < 0
    )
    return {
        "status": "PASS" if passed else "STOP",
        "winner_unchanged": "phase_chord",
        "phase_receipt_sha256": phase_vs_moment.get("parent_receipt_sha256"),
        "moment_receipt_sha256": phase_vs_moment.get("comparison_receipt_sha256"),
        "phase_retention_delta": phase_retention,
        "moment_retention_delta": moment_retention,
        "positive_document_fraction": positive_fraction,
        "mean_source_effect_delta": mean_source,
        "half_mean_source_effect_delta": half_means,
        "mean_nll_delta": mean_nll,
        "mean_first_token_rank_delta": mean_rank,
        "document_deltas": deltas,
    }


def document_component_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_key: str,
    parent_key: str = "parent",
    split_key: str = "split",
    document_key: str = "document_id",
    nll_key: str = "nll",
    source_effect_key: str = "source_effect",
    final_manifest_sha256: str | None = None,
    final_documents: int | None = None,
    config: ComponentGateConfig = ComponentGateConfig(),
) -> dict[str, Any]:
    """Evaluate a Stage1 component arm using selection documents only.

    Rows marked ``final_validation`` are forbidden from arm selection. Metrics
    are first averaged within document, so multiple offsets cannot inflate the
    effective sample count.
    """
    if not rows:
        raise ValueError("component gate received no rows")
    splits = {str(row.get(split_key)) for row in rows}
    if not splits.issubset({"train", "selection", "gate", "component_gate", "final_validation"}):
        raise ValueError("component rows contain an unknown split")
    selection = [row for row in rows if str(row.get(split_key)) in {"selection", "gate", "component_gate"}]
    final = [row for row in rows if str(row.get(split_key)) == "final_validation"]
    train = [row for row in rows if str(row.get(split_key)) == "train"]
    if not selection:
        raise ValueError("component gate requires selection rows")
    train_documents = {str(row.get(document_key)) for row in train}
    selection_documents = {str(row.get(document_key)) for row in selection}
    final_row_documents = {str(row.get(document_key)) for row in final}
    if None in {row.get(document_key) for row in train + final}:
        raise ValueError("train/final row lacks document identity")
    if train_documents & selection_documents or train_documents & final_row_documents or selection_documents & final_row_documents:
        raise ValueError("train, gate, and final_validation documents must be disjoint")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in selection:
        document = row.get(document_key)
        if document is None:
            raise ValueError("selection row lacks document identity")
        if candidate_key not in row or parent_key not in row:
            raise ValueError("component row lacks candidate/parent metrics")
        groups.setdefault(str(document), []).append(row)
    document_deltas: list[dict[str, float | str]] = []
    for document, group in sorted(groups.items()):
        candidate_nll = sum(float(row[candidate_key][nll_key]) for row in group) / len(group)
        parent_nll = sum(float(row[parent_key][nll_key]) for row in group) / len(group)
        candidate_effect = sum(float(row[candidate_key][source_effect_key]) for row in group) / len(group)
        parent_effect = sum(float(row[parent_key][source_effect_key]) for row in group) / len(group)
        document_deltas.append({
            "document_id": document,
            "nll_delta": candidate_nll - parent_nll,
            "source_effect_delta": candidate_effect - parent_effect,
            "positive": float(candidate_effect > parent_effect),
        })
    positive_fraction = sum(float(row["positive"]) for row in document_deltas) / len(document_deltas)
    passed = bool(
        document_deltas
        and all(float(row["nll_delta"]) <= config.nll_ceiling for row in document_deltas)
        and sum(float(row["source_effect_delta"]) for row in document_deltas) / len(document_deltas)
        >= config.minimum_source_effect
        and positive_fraction >= config.minimum_positive_fraction
    )
    return {
        "status": "PASS" if passed else "STOP",
        "candidate_key": candidate_key,
        "selection_documents": len(document_deltas),
        "train_documents": len(train_documents),
        "final_validation_documents": int(final_documents if final_documents is not None else len(final_row_documents)),
        "final_manifest_sha256": final_manifest_sha256,
        "final_validation_used_for_selection": False,
        "document_deltas": document_deltas,
        "mean_nll_delta": sum(float(row["nll_delta"]) for row in document_deltas) / len(document_deltas),
        "mean_source_effect_delta": sum(float(row["source_effect_delta"]) for row in document_deltas) / len(document_deltas),
        "positive_document_fraction": positive_fraction,
        "minimum_positive_fraction": config.minimum_positive_fraction,
    }
