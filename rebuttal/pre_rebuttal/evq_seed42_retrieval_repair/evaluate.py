#!/usr/bin/env python3
"""Bounded evaluation and fail-closed gates for EVQ seed-42 retrieval repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.lora_evq_v2.eval_temporal_holdout_matched import (
    load_domain_artifacts,
)
from experiments.lora_evq_v2.eval_official_yarn_capability import (
    _score_one_record,
    load_capability_suite,
    score_capability_prediction,
    score_generation_metrics,
    select_rows,
    summarize_results,
)
from experiments.lora_evq_v2.legacy_lora_protocol import sha256_file
from experiments.lora_evq_v2.train_evq_lora import (
    load_frequency_artifact,
    resolve_model_rope_geometry,
)
from experiments.lora_evq_v2.train_positional_distill import causal_backbone
from experiments.lora_evq_v2.prepare_positional_distill_data import (
    tokenizer_source_fingerprint,
)
from experiments.lora_evq_v2.prepare_legacy_model_manifest import (
    validate_model_manifest,
)
from rebuttal.pre_rebuttal.frequency_adaptation_8b.train import load_model_identity
from scripts.lib.rope.official_yarn import official_yarn_on_inv_freq

from .prepare_data import (
    PASSKEY_BUNDLES,
    validate_bundle,
    validate_prepared_dir,
    validate_tokenizer_identity,
)
from .protocol import (
    decide_gate,
    evaluation_budget,
    get_stage,
    registered_factor_for_length,
    score_text_answer,
)
from .train import (
    EVQ_TAU,
    HEAD_DIM,
    ROPE_BASE,
    apply_runtime_frequency,
    runtime_frequency_contract,
    tensor_sha256,
    validate_gate_transition,
    validate_parent_adapter,
    validate_repair_parent,
)


CONTROLLED_SCHEMA = "evq_cosh.seed42_retrieval_repair_controlled.v1"
PASSKEY_SCHEMA = "evq_cosh.seed42_retrieval_repair_passkey.v1"
TEMPORAL_SCHEMA = "evq_cosh.seed42_retrieval_repair_temporal.v1"
CAPABILITY_SCHEMA = "evq_cosh.seed42_retrieval_repair_capability.v1"
GATE_PURPOSE = "evq_seed42_retrieval_repair_gate"
FINAL_REPORT_PURPOSE = "evq_seed42_retrieval_repair_final_report"
CAPABILITY_BUDGET_PURPOSE = "evq_seed42_retrieval_repair_capability_budget"
PREFLIGHT_COMPLETE_PURPOSE = "evq_seed42_retrieval_repair_preflight_complete"
RULER_TASKS = {
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
}
MCQA_TASKS = {"mmlu", "arc_challenge", "hellaswag", "openbookqa", "winogrande"}
LONGBENCH_TASKS = {"narrativeqa", "qasper"}
NOLIMA_DEPTHS = {10.0, 25.0, 50.0, 75.0, 90.0}
REQUIRED_CAPABILITY_SUITES = {
    8192: {"ruler", "mcqa"},
    16384: {"ruler", "nolima_hard_exact_context"},
    32768: {"ruler", "nolima_hard_exact_context"},
}
CAPABILITY_SUITES = set().union(*REQUIRED_CAPABILITY_SUITES.values()) | {
    "longbench"
}

_GATE_EVIDENCE = {
    "parent_controlled": (CONTROLLED_SCHEMA, "parent_retrieval_result_sha256"),
    "checkpoint_controlled": (CONTROLLED_SCHEMA, "retrieval_result_sha256"),
    "passkey_result": (PASSKEY_SCHEMA, "passkey_result_sha256"),
    "parent_temporal": (TEMPORAL_SCHEMA, "parent_temporal_sha256"),
    "checkpoint_temporal": (TEMPORAL_SCHEMA, "checkpoint_temporal_sha256"),
}

_HASH_FIELDS = {
    "parent_adapter_sha256",
    "parent_adapter_receipt_sha256",
    "checkpoint_adapter_sha256",
    "checkpoint_adapter_receipt_sha256",
    "model_manifest_sha256",
    "longalpaca_manifest_sha256",
    "repair_manifest_sha256",
    "validation_bundle_sha256",
    "passkey_bundle_sha256",
    "operator_tensor_sha256",
    "evaluator_code_sha256",
    "parent_retrieval_result_sha256",
    "retrieval_result_sha256",
    "passkey_result_sha256",
    "parent_temporal_sha256",
    "checkpoint_temporal_sha256",
}


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (
            (candidate / "AGENTS.md").is_file()
            and (candidate / "scripts/lib/rope/schedules.py").is_file()
        ):
            return candidate
    raise RuntimeError(f"could not locate repository root from {start}")


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluator_code_sha256() -> str:
    """Bind outputs to the evaluator, metric contract, and runtime operator."""
    project_root = _find_repo_root(Path(__file__).resolve())
    paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("protocol.py").resolve(),
        Path(__file__).with_name("train.py").resolve(),
        Path(__file__).with_name("prepare_data.py").resolve(),
        (project_root / "scripts/lib/rope/official_yarn.py").resolve(),
        (project_root / "experiments/lora_evq_v2/eval_official_yarn_capability.py").resolve(),
        (project_root / "experiments/lora_evq_v2/prepare_seed42_capability_data.py").resolve(),
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.incomplete")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_registered_factor(value: str | float | int) -> float:
    """Parse one repair factor without changing the legacy x2,x4 evaluator."""
    text = str(value).strip()
    if not text or "," in text:
        raise ValueError("repair evaluation requires exactly one YaRN factor")
    try:
        factor = float(text)
    except ValueError as exc:
        raise ValueError("repair evaluation factor must be numeric") from exc
    if factor not in {1.0, 2.0, 4.0}:
        raise ValueError("repair evaluation factor is not registered; expected 1, 2, or 4")
    return factor


def _int_tokens(value: Sequence[int] | torch.Tensor) -> list[int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().view(-1).tolist()
    return [int(token_id) for token_id in value]


def _contains_tokens(haystack: Sequence[int], needle: Sequence[int]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(list(haystack[start : start + width]) == list(needle) for start in range(len(haystack) - width + 1))


def score_generated_tokens(
    generated_ids: Sequence[int] | torch.Tensor,
    expected_ids: Sequence[int] | torch.Tensor,
    tokenizer: Any,
) -> dict[str, Any]:
    """Separate value retrieval, full-output formatting, and EOS stopping."""
    generated = _int_tokens(generated_ids)
    expected = _int_tokens(expected_ids)
    if not expected:
        raise ValueError("expected value tokens cannot be empty")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_index = generated.index(int(eos_id)) if eos_id is not None and int(eos_id) in generated else None
    content = generated if eos_index is None else generated[:eos_index]
    prefix_exact = content[: len(expected)] == expected
    return {
        "strict_exact": content == expected,
        "first_value_exact": prefix_exact,
        "gold_containment": _contains_tokens(content, expected),
        "eos_terminated": eos_index is not None,
        "generated_token_count": len(content),
        "prediction": tokenizer.decode(content, skip_special_tokens=True).strip(),
        "expected": tokenizer.decode(expected, skip_special_tokens=True).strip(),
    }


def _distance_bucket(stage: str, distance: int) -> str:
    spec = get_stage(stage)
    midpoint = (spec.min_distance + spec.max_distance) // 2
    if distance <= midpoint:
        return f"{spec.min_distance}-{midpoint}"
    return f"{midpoint + 1}-{spec.max_distance}"


def _summarize_groups(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(groups)
    if count == 0:
        raise ValueError("cannot summarize zero controlled groups")
    pair = sum(bool(group["pair_exact"]) for group in groups)
    positive = sum(float(group["removal_delta_nll"]) > 0.0 for group in groups)
    finite = all(bool(group["finite"]) for group in groups)
    return {
        "groups": count,
        "pair_consistency": pair / count,
        "source_removal_positive_fraction": positive / count,
        "finite": finite,
    }


def summarize_repair_records(
    records: Sequence[Mapping[str, Any]],
    *,
    stage: str,
) -> dict[str, Any]:
    """Require original/swapped/removed triplets and summarize source dependence."""
    get_stage(stage)
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        group_id = str(record.get("group_id", ""))
        variant = str(record.get("variant", ""))
        if not group_id:
            raise ValueError("controlled record is missing group_id")
        if variant not in {"original", "swapped", "source_removed"}:
            raise ValueError(f"controlled record has unregistered variant {variant!r}")
        if variant in grouped[group_id]:
            raise ValueError(f"controlled group {group_id} repeats variant {variant}")
        grouped[group_id][variant] = record
    if not grouped:
        raise ValueError("controlled evaluation has no groups")

    group_rows: list[dict[str, Any]] = []
    for group_id, variants in sorted(grouped.items()):
        if set(variants) != {"original", "swapped", "source_removed"}:
            raise ValueError(f"controlled group {group_id} must contain all three variants")
        original = variants["original"]
        swapped = variants["swapped"]
        removed = variants["source_removed"]
        task_types = {str(row.get("task_type")) for row in variants.values()}
        distances = {int(row.get("distance", -1)) for row in variants.values()}
        if len(task_types) != 1 or len(distances) != 1:
            raise ValueError(f"controlled group {group_id} metadata differs across variants")
        nll_values = [float(row.get("mean_nll", math.nan)) for row in variants.values()]
        finite = all(bool(row.get("finite")) for row in variants.values()) and all(
            math.isfinite(value) for value in nll_values
        )
        distance = distances.pop()
        group_rows.append(
            {
                "group_id": group_id,
                "task_type": task_types.pop(),
                "distance": distance,
                "distance_bucket": _distance_bucket(stage, distance),
                "pair_exact": bool(original.get("first_value_exact"))
                and bool(swapped.get("first_value_exact")),
                "removal_delta_nll": float(removed.get("mean_nll", math.nan))
                - float(original.get("mean_nll", math.nan)),
                "finite": finite,
            }
        )

    by_task: dict[str, Any] = {}
    by_distance: dict[str, Any] = {}
    for task in sorted({row["task_type"] for row in group_rows}):
        by_task[task] = _summarize_groups([row for row in group_rows if row["task_type"] == task])
    for bucket in sorted({row["distance_bucket"] for row in group_rows}):
        by_distance[bucket] = _summarize_groups(
            [row for row in group_rows if row["distance_bucket"] == bucket]
        )
    return {
        **_summarize_groups(group_rows),
        "stage": stage,
        "task_types": sorted(by_task),
        "by_task": by_task,
        "by_distance_bucket": by_distance,
        "group_records": group_rows,
    }


def summarize_passkey_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute every passkey aggregate from immutable per-row evidence."""
    if not records:
        raise ValueError("passkey evaluation has no rows")
    answer_tokens = sum(int(row.get("answer_tokens", 0)) for row in records)
    nll_sum = sum(float(row.get("nll_sum", math.nan)) for row in records)
    finite = (
        answer_tokens > 0
        and math.isfinite(nll_sum)
        and all(bool(row.get("finite")) for row in records)
    )
    if not finite:
        raise ValueError("passkey result contains non-finite or empty evidence")
    count = len(records)
    return {
        "rows": count,
        "passkey_strict_exact": sum(bool(row.get("strict_exact")) for row in records)
        / count,
        "passkey_first_value_exact": sum(
            bool(row.get("first_value_exact")) for row in records
        )
        / count,
        "passkey_containment": sum(bool(row.get("gold_containment")) for row in records)
        / count,
        "eos_fraction": sum(bool(row.get("eos_terminated")) for row in records) / count,
        "mean_nll": nll_sum / answer_tokens,
        "finite": True,
    }


def summarize_temporal_domains(domains: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute temporal domain and collection NLLs from pack-level evidence."""
    if len(domains) != 3:
        raise ValueError("temporal evidence must contain exactly three domains")
    means = []
    for name, raw_domain in sorted(domains.items()):
        if not isinstance(raw_domain, Mapping):
            raise ValueError(f"temporal domain {name} is not a mapping")
        packs = raw_domain.get("packs")
        if not isinstance(packs, list) or not packs:
            raise ValueError(f"temporal domain {name} has no pack evidence")
        indices = [int(pack.get("pack_index", -1)) for pack in packs]
        if len(indices) != len(set(indices)) or any(index < 0 for index in indices):
            raise ValueError(f"temporal domain {name} has invalid pack indices")
        nll_sum = sum(float(pack.get("nll_sum", math.nan)) for pack in packs)
        tokens = sum(int(pack.get("scored_tokens", 0)) for pack in packs)
        if (
            tokens <= 0
            or not math.isfinite(nll_sum)
            or not all(bool(pack.get("finite")) for pack in packs)
        ):
            raise ValueError(f"temporal domain {name} has non-finite pack evidence")
        mean_nll = nll_sum / tokens
        expected_domain = {
            "packs": packs,
            "nll_sum": nll_sum,
            "scored_tokens": tokens,
            "mean_nll": mean_nll,
            "finite": True,
        }
        if dict(raw_domain) != expected_domain:
            raise ValueError(f"temporal domain {name} aggregate differs from raw packs")
        means.append(mean_nll)
    mean_nll = sum(means) / len(means)
    return {"mean_nll": mean_nll, "finite": math.isfinite(mean_nll)}


def _require_json_equal(actual: Any, expected: Any, *, label: str) -> None:
    left = json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False)
    right = json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if left != right:
        raise ValueError(f"{label} differs from recomputed raw evidence")


def merge_temporal_guardrail(
    stage: str,
    parent: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare parent/checkpoint on the identical factor and frozen examples."""
    spec = get_stage(stage)
    for name, record in (("parent", parent), ("checkpoint", checkpoint)):
        if record.get("schema") != TEMPORAL_SCHEMA:
            raise ValueError(f"{name} temporal result schema mismatch")
        if record.get("stage") != stage:
            raise ValueError(f"{name} temporal result stage mismatch")
        if not math.isclose(float(record.get("factor", math.nan)), spec.factor, abs_tol=1e-12):
            raise ValueError(f"{name} temporal result factor mismatch")
    if parent.get("temporal_selection_sha256") != checkpoint.get("temporal_selection_sha256"):
        raise ValueError("parent/checkpoint temporal selection mismatch")
    parent_summary = parent.get("summary")
    checkpoint_summary = checkpoint.get("summary")
    if not isinstance(parent_summary, Mapping) or not isinstance(checkpoint_summary, Mapping):
        raise ValueError("temporal result is missing summary")
    parent_nll = float(parent_summary.get("mean_nll", math.nan))
    checkpoint_nll = float(checkpoint_summary.get("mean_nll", math.nan))
    finite = (
        bool(parent_summary.get("finite"))
        and bool(checkpoint_summary.get("finite"))
        and math.isfinite(parent_nll)
        and math.isfinite(checkpoint_nll)
    )
    if not finite:
        raise ValueError("temporal result contains non-finite values")
    return {
        "parent_mean_nll": parent_nll,
        "checkpoint_mean_nll": checkpoint_nll,
        "temporal_delta_nll": round(checkpoint_nll - parent_nll, 12),
        "temporal_selection_sha256": parent["temporal_selection_sha256"],
        "finite": True,
    }


def _validate_gate_bindings(bindings: Mapping[str, Any]) -> dict[str, str]:
    keys = set(bindings)
    if keys != _HASH_FIELDS:
        missing = sorted(_HASH_FIELDS - keys)
        extra = sorted(keys - _HASH_FIELDS)
        raise ValueError(f"gate evidence bindings differ; missing={missing}, extra={extra}")
    normalized = {key: str(bindings[key]) for key in sorted(_HASH_FIELDS)}
    for key, value in normalized.items():
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"gate binding {key} is not a lowercase SHA-256")
    return normalized


def build_gate_report(
    *,
    stage: str,
    segment: int,
    repair_summary: Mapping[str, Any],
    passkey_summary: Mapping[str, Any],
    parent_repair_summary: Mapping[str, Any],
    parent_temporal: Mapping[str, Any],
    checkpoint_temporal: Mapping[str, Any],
    bindings: Mapping[str, Any],
    evidence_files: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an uninterpreted gate whose evidence is entirely hash-bound."""
    evidence = _validate_gate_bindings(bindings)
    temporal = merge_temporal_guardrail(stage, parent_temporal, checkpoint_temporal)
    task_types = repair_summary.get("task_types")
    summary = {
        "pair_consistency": repair_summary.get("pair_consistency"),
        "source_removal_positive_fraction": repair_summary.get(
            "source_removal_positive_fraction"
        ),
        "passkey_containment": passkey_summary.get("passkey_containment"),
        "temporal_delta_nll": temporal["temporal_delta_nll"],
        "finite": bool(repair_summary.get("finite"))
        and bool(passkey_summary.get("finite"))
        and bool(temporal["finite"]),
        "task_types": task_types,
    }
    parent_summary = {"pair_consistency": parent_repair_summary.get("pair_consistency")}
    decision = decide_gate(stage, summary, parent_summary, segment=int(segment))
    report = {
        "format_version": 1,
        "purpose": GATE_PURPOSE,
        "status": decision["status"],
        "bindings": {
            "stage": stage,
            "segment": int(segment),
            "factor": get_stage(stage).factor,
            **evidence,
        },
        "summary": summary,
        "parent_summary": parent_summary,
        "temporal": temporal,
        "decision": decision,
    }
    if evidence_files is not None:
        report["evidence_files"] = {
            str(name): dict(record) for name, record in sorted(evidence_files.items())
        }
    return report


def _runtime_contract_for_length(canonical: torch.Tensor, target_length: int) -> dict[str, Any]:
    factor = registered_factor_for_length(target_length)
    if target_length in {8192, 16384}:
        return runtime_frequency_contract(canonical, stage="r8" if target_length == 8192 else "r16")
    substrate = canonical.detach().cpu().to(torch.float64).view(-1)
    runtime, mscale, operator = official_yarn_on_inv_freq(
        substrate,
        head_dim=HEAD_DIM,
        base=ROPE_BASE,
        scale=factor,
        original_max_position_embeddings=8192,
        beta_fast=32.0,
        beta_slow=1.0,
        extrapolation_factor=1.0,
        attn_factor=1.0,
    )
    if operator.get("mode") != "yarn_derived_virtual_dim":
        raise RuntimeError("32K EVQ runtime is not labeled as a virtual-dim YaRN derivation")
    return {
        "substrate_inv_freq": substrate,
        "runtime_inv_freq": runtime,
        "factor": factor,
        "mscale": float(mscale),
        "label": "YaRN-derived generalization on the EVQ substrate",
        "operator": operator,
        "substrate_tensor_sha256": tensor_sha256(substrate),
        "runtime_tensor_sha256": tensor_sha256(runtime),
    }


def _load_eval_identity(args: argparse.Namespace) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any]]:
    data_manifest = validate_prepared_dir(args.data_dir)
    expected_tokenizer = tokenizer_source_fingerprint(args.model_name)
    validate_tokenizer_identity(data_manifest.get("tokenizer", {}), expected_tokenizer)
    repair_manifest_sha = sha256_file(args.data_dir / "manifest.json")
    model_identity = load_model_identity(args.model_name, args.model_manifest)
    longalpaca_sha = sha256_file(args.longalpaca_manifest)
    if args.adapter_kind == "legacy":
        identity = validate_parent_adapter(
            args.adapter_dir,
            longalpaca_manifest=args.longalpaca_manifest,
            model_manifest=args.model_manifest,
        )
        canonical, _, _ = load_frequency_artifact(
            args.adapter_dir / "custom_inv_freq.pt", expected_method="evq_cosh"
        )
    else:
        identity = validate_repair_parent(args.adapter_dir)
        if identity.get("model_manifest_sha256") != model_identity["manifest_sha256"]:
            raise RuntimeError("repair adapter model manifest mismatch")
        if identity.get("longalpaca_manifest_sha256") != longalpaca_sha:
            raise RuntimeError("repair adapter LongAlpaca manifest mismatch")
        if identity.get("repair_manifest_sha256") != repair_manifest_sha:
            raise RuntimeError("repair adapter data manifest mismatch")
        canonical = identity["substrate_inv_freq"]
    provenance = {
        "adapter_kind": args.adapter_kind,
        "adapter_sha256": identity["adapter_sha256"],
        "adapter_receipt_sha256": identity["adapter_receipt_sha256"],
        "model_manifest_sha256": model_identity["manifest_sha256"],
        "longalpaca_manifest_sha256": longalpaca_sha,
        "repair_manifest_sha256": repair_manifest_sha,
        "data_status": data_manifest["status"],
    }
    return provenance, canonical.detach().cpu().to(torch.float64), model_identity


def _load_cuda_model(
    args: argparse.Namespace,
    canonical: torch.Tensor,
    *,
    target_length: int,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU evaluation requires CUDA")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    geometry = resolve_model_rope_geometry(model.config)
    if geometry.head_dim != HEAD_DIM or not math.isclose(
        geometry.rope_base, ROPE_BASE, rel_tol=0.0, abs_tol=1e-6
    ):
        raise RuntimeError("loaded model geometry differs from registered LLaMA-3-8B")
    model = PeftModel.from_pretrained(model, str(args.adapter_dir), is_trainable=False)
    model.eval()
    model.config.use_cache = True
    runtime = _runtime_contract_for_length(canonical, target_length)
    application = apply_runtime_frequency(model, runtime)
    runtime_record = {
        key: value for key, value in runtime.items() if not torch.is_tensor(value)
    }
    runtime_record["application"] = application
    return model, tokenizer, causal_backbone(model), model.get_output_embeddings(), runtime_record


@torch.inference_mode()
def _score_full_answer(
    backbone: Any,
    lm_head: Any,
    full_ids: torch.Tensor,
    *,
    answer_start: int,
    answer_end: int,
    device: torch.device,
) -> dict[str, Any]:
    input_ids = full_ids.to(device=device, dtype=torch.long).view(1, -1)
    hidden = backbone(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    predicting = hidden[:, int(answer_start) - 1 : int(answer_end) - 1]
    labels = input_ids[:, int(answer_start) : int(answer_end)]
    logits = lm_head(predicting).float()
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum"
    )
    token_count = int(labels.numel())
    nll_sum = float(nll.detach().double().cpu())
    return {
        "nll_sum": nll_sum,
        "answer_tokens": token_count,
        "mean_nll": nll_sum / token_count,
        "finite": math.isfinite(nll_sum),
    }


@torch.inference_mode()
def _generate_ids(
    model: Any,
    prompt_ids: torch.Tensor,
    *,
    tokenizer: Any,
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    input_ids = prompt_ids.to(device=device, dtype=torch.long).view(1, -1)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return _int_tokens(output[0, input_ids.shape[1] :])


def _bundle_record(data_dir: Path, filename: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = validate_prepared_dir(data_dir)
    record = manifest.get("files", {}).get(filename)
    if not isinstance(record, Mapping):
        raise FileNotFoundError(f"repair manifest has no {filename}")
    bundle = torch.load(data_dir / filename, map_location="cpu", weights_only=True)
    return dict(record), dict(bundle)


def run_controlled(args: argparse.Namespace) -> dict[str, Any]:
    spec = get_stage(args.stage)
    filename = f"{args.split}_{args.stage}.pt"
    record, bundle = _bundle_record(args.data_dir, filename)
    expected_rows = 48 if args.split == "validation" else 96
    validate_bundle(
        bundle,
        stage=args.stage,
        split=args.split,
        segment=None,
        expected_seq_len=spec.seq_len,
        expected_rows=expected_rows,
    )
    provenance, canonical, _ = _load_eval_identity(args)
    model, tokenizer, backbone, lm_head, runtime = _load_cuda_model(
        args, canonical, target_length=spec.seq_len
    )
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    input_ids = bundle["input_ids"]
    for index, metadata in enumerate(bundle["metadata"]):
        start = int(bundle["answer_start"][index])
        end = int(bundle["answer_end"][index])
        full = input_ids[index]
        scored = _score_full_answer(
            backbone, lm_head, full, answer_start=start, answer_end=end, device=device
        )
        row = {**dict(metadata), **scored, "row_index": index}
        if metadata["variant"] in {"original", "swapped"}:
            generated = _generate_ids(
                model,
                full[:start],
                tokenizer=tokenizer,
                max_new_tokens=16,
                device=device,
            )
            row.update(score_generated_tokens(generated, full[start : end - 1], tokenizer))
            row["generated_ids"] = generated
            row["expected_ids"] = _int_tokens(full[start : end - 1])
        else:
            row.update(
                strict_exact=None,
                first_value_exact=None,
                gold_containment=None,
                eos_terminated=None,
                generated_token_count=0,
                prediction=None,
            )
        rows.append(row)
        print(json.dumps({"controlled": f"{index + 1}/{len(bundle['metadata'])}", "variant": metadata["variant"]}), flush=True)
    summary = summarize_repair_records(rows, stage=args.stage)
    output = {
        "schema": CONTROLLED_SCHEMA,
        "stage": args.stage,
        "factor": spec.factor,
        "split": args.split,
        **provenance,
        "bundle_file": filename,
        "bundle_sha256": record["sha256"],
        "runtime": runtime,
        "runtime_tensor_sha256": runtime["runtime_tensor_sha256"],
        "evaluator_code_sha256": evaluator_code_sha256(),
        "eos_token_id": int(tokenizer.eos_token_id),
        "evaluation_budget": evaluation_budget(args.stage),
        "results": rows,
        "summary": summary,
    }
    _atomic_json_dump(output, args.output)
    return output


def run_passkey(args: argparse.Namespace) -> dict[str, Any]:
    target_length = int(args.target_length)
    factor = registered_factor_for_length(target_length)
    filename, evaluation_split, _ = PASSKEY_BUNDLES[target_length]
    record, bundle = _bundle_record(args.data_dir, filename)
    if bundle.get("evaluation_split") != evaluation_split:
        raise ValueError("passkey bundle does not match its registered evaluation split")
    provenance, canonical, _ = _load_eval_identity(args)
    model, tokenizer, backbone, lm_head, runtime = _load_cuda_model(
        args, canonical, target_length=target_length
    )
    device = torch.device("cuda")
    rows = []
    for index, metadata in enumerate(bundle["metadata"]):
        prompt = bundle["prompt_ids"][index]
        answer = str(metadata["answer"])
        expected_ids = tokenizer(
            answer, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        full = torch.cat(
            (prompt.to(torch.int32), torch.tensor(expected_ids, dtype=torch.int32))
        )
        scored = _score_full_answer(
            backbone,
            lm_head,
            full,
            answer_start=target_length,
            answer_end=full.numel(),
            device=device,
        )
        generated = _generate_ids(
            model,
            prompt,
            tokenizer=tokenizer,
            max_new_tokens=32,
            device=device,
        )
        prediction_ids = generated
        if tokenizer.eos_token_id in prediction_ids:
            prediction_ids = prediction_ids[: prediction_ids.index(tokenizer.eos_token_id)]
        prediction = tokenizer.decode(prediction_ids, skip_special_tokens=True).strip()
        metrics = score_text_answer(
            prediction,
            answer,
            tokenizer.eos_token_id in generated,
        )
        rows.append(
            {
                **dict(metadata),
                **scored,
                **metrics,
                "prediction": prediction,
                "generated_ids": generated,
                "generated_token_count": len(prediction_ids),
            }
        )
        print(json.dumps({"passkey": f"{index + 1}/{len(bundle['metadata'])}", "depth": metadata["depth_percent"]}), flush=True)
    summary = summarize_passkey_records(rows)
    output = {
        "schema": PASSKEY_SCHEMA,
        "stage": {8192: "r8", 16384: "r16", 32768: "x4"}[target_length],
        "factor": factor,
        "target_length": target_length,
        "evaluation_split": evaluation_split,
        **provenance,
        "bundle_file": filename,
        "bundle_sha256": record["sha256"],
        "runtime": runtime,
        "runtime_tensor_sha256": runtime["runtime_tensor_sha256"],
        "evaluator_code_sha256": evaluator_code_sha256(),
        "eos_token_id": int(tokenizer.eos_token_id),
        "results": rows,
        "summary": summary,
    }
    _atomic_json_dump(output, args.output)
    return output


def capability_pilot_budget(
    rows: Sequence[Mapping[str, Any]], *, target_length: int
) -> dict[str, Any]:
    """Validate coverage and register a bounded forward/generation budget."""
    target_length = int(target_length)
    selected = [row for row in rows if int(row["target_length"]) == target_length]
    if not selected:
        raise ValueError(f"capability suite has no rows at {target_length}")
    suites = {str(row["suite"]) for row in selected}
    missing_suites = REQUIRED_CAPABILITY_SUITES[target_length] - suites
    if missing_suites:
        raise ValueError(
            f"capability suite at {target_length} misses required suites: "
            f"{sorted(missing_suites)}"
        )
    ruler_tasks = {str(row["task"]) for row in selected if row["suite"] == "ruler"}
    if ruler_tasks != RULER_TASKS:
        raise ValueError(
            f"RULER task coverage mismatch at {target_length}; "
            f"missing={sorted(RULER_TASKS - ruler_tasks)}, "
            f"extra={sorted(ruler_tasks - RULER_TASKS)}"
        )
    if target_length == 8192:
        mcqa_tasks = {str(row["task"]) for row in selected if row["suite"] == "mcqa"}
        if mcqa_tasks != MCQA_TASKS:
            raise ValueError(
                f"MCQA task coverage mismatch; missing={sorted(MCQA_TASKS - mcqa_tasks)}, "
                f"extra={sorted(mcqa_tasks - MCQA_TASKS)}"
            )
    if target_length in {16384, 32768}:
        nolima_depths = {
            float(row["depth_percent"])
            for row in selected
            if row["suite"] == "nolima_hard_exact_context"
        }
        if nolima_depths != NOLIMA_DEPTHS:
            raise ValueError(
                f"NoLiMa depth coverage mismatch at {target_length}; "
                f"missing={sorted(NOLIMA_DEPTHS - nolima_depths)}"
            )
    prompt_tokens = 0
    nll_forwards = 0
    generation_rows = 0
    max_generated_tokens = 0
    counts: dict[str, int] = defaultdict(int)
    for row in selected:
        prompt_count = int(row.get("prompt_tokens", len(row.get("prompt_ids", []))))
        candidates = int(
            row.get(
                "answer_candidates",
                len(row.get("choices") or row.get("answers") or []),
            )
        )
        generation_tokens = int(row.get("generation_tokens", 0))
        if prompt_count <= 0 or candidates <= 0 or generation_tokens < 0:
            raise ValueError("capability row has an invalid forward budget")
        prompt_tokens += prompt_count * (candidates + int(generation_tokens > 0))
        nll_forwards += candidates
        generation_rows += int(generation_tokens > 0)
        max_generated_tokens += generation_tokens
        counts[f"{row['suite']}::{row['task']}"] += 1
    return {
        "target_length": target_length,
        "factor": registered_factor_for_length(target_length),
        "selection": "one_row_per_task_length_depth_cell",
        "rows": len(selected),
        "nll_forwards": nll_forwards,
        "generation_rows": generation_rows,
        "prompt_tokens_across_prefills": prompt_tokens,
        "max_generated_tokens": max_generated_tokens,
        "suites": sorted(suites),
        "task_counts": dict(sorted(counts.items())),
        "ruler_tasks": sorted(ruler_tasks),
    }


def _capability_budget_receipt(
    capability_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest, all_rows = load_capability_suite(capability_dir)
    budgets = {}
    selected_rows = []
    for target_length in (8192, 16384, 32768):
        length_rows = [
            row
            for row in all_rows
            if int(row["target_length"]) == target_length
            and str(row["suite"]) in CAPABILITY_SUITES
        ]
        selected = select_rows(length_rows, mode="pilot")
        budgets[str(target_length)] = capability_pilot_budget(
            selected, target_length=target_length
        )
        selected_rows.extend(selected)
    longbench_tasks = {
        str(row["task"]) for row in selected_rows if row["suite"] == "longbench"
    }
    if longbench_tasks != LONGBENCH_TASKS:
        raise ValueError(
            f"LongBench task coverage mismatch; "
            f"missing={sorted(LONGBENCH_TASKS - longbench_tasks)}, "
            f"extra={sorted(longbench_tasks - LONGBENCH_TASKS)}"
        )
    receipt = {
        "format_version": 1,
        "purpose": CAPABILITY_BUDGET_PURPOSE,
        "status": "bounded_pilot_valid",
        "capability_manifest_sha256": sha256_file(Path(capability_dir) / "manifest.json"),
        "manifest_rows": int(manifest["row_count"]),
        "budgets": budgets,
        "total": {
            key: sum(int(budget[key]) for budget in budgets.values())
            for key in (
                "rows",
                "nll_forwards",
                "generation_rows",
                "prompt_tokens_across_prefills",
                "max_generated_tokens",
            )
        },
    }
    return receipt, selected_rows


def run_capability(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate the frozen RULER/downstream suite at one registered length."""
    target_length = int(args.target_length)
    factor = registered_factor_for_length(target_length)
    manifest, all_rows = load_capability_suite(args.capability_dir)
    validate_tokenizer_identity(
        manifest.get("tokenizer", {}), tokenizer_source_fingerprint(args.model_name)
    )
    length_rows = [
        row
        for row in all_rows
        if int(row["target_length"]) == target_length
        and str(row["suite"]) in CAPABILITY_SUITES
    ]
    if args.mode != "pilot":
        raise ValueError("repair capability evaluation is bounded to mode=pilot")
    rows = select_rows(length_rows, mode="pilot") if length_rows else []
    budget = capability_pilot_budget(rows, target_length=target_length)
    provenance, canonical, _ = _load_eval_identity(args)
    model, tokenizer, backbone, lm_head, runtime = _load_cuda_model(
        args, canonical, target_length=target_length
    )
    device = torch.device("cuda")
    results = []
    for index, row in enumerate(rows):
        scored = _score_one_record(
            row=row,
            model=model,
            tokenizer=tokenizer,
            backbone=backbone,
            lm_head=lm_head,
            device=device,
            generation=True,
        )
        scored["factor"] = factor
        results.append(scored)
        print(
            json.dumps(
                {
                    "capability": f"{index + 1}/{len(rows)}",
                    "suite": row["suite"],
                    "task": row["task"],
                    "target_length": target_length,
                }
            ),
            flush=True,
        )
    output = {
        "schema": CAPABILITY_SCHEMA,
        "stage": {8192: "r8", 16384: "r16", 32768: "x4"}[target_length],
        "factor": factor,
        "target_length": target_length,
        "mode": args.mode,
        **provenance,
        "capability_manifest_sha256": sha256_file(
            args.capability_dir / "manifest.json"
        ),
        "capability_manifest_rows": manifest["row_count"],
        "runtime": runtime,
        "runtime_tensor_sha256": runtime["runtime_tensor_sha256"],
        "evaluator_code_sha256": evaluator_code_sha256(),
        "status": "complete_uninterpreted",
        "budget": budget,
        "results": results,
        "summary": summarize_results(results),
    }
    _atomic_json_dump(output, args.output)
    return output


@torch.inference_mode()
def _score_temporal_prefix(
    backbone: Any,
    lm_head: Any,
    input_ids: torch.Tensor,
    score_mask: torch.Tensor,
    *,
    chunk_tokens: int,
) -> dict[str, Any]:
    hidden = backbone(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    nll_sum = 0.0
    scored_tokens = 0
    for start in range(0, input_ids.shape[1] - 1, int(chunk_tokens)):
        end = min(start + int(chunk_tokens), input_ids.shape[1] - 1)
        logits = lm_head(hidden[:, start:end]).float()
        labels = input_ids[:, start + 1 : end + 1]
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
        ).reshape_as(labels)
        valid = score_mask[:, start + 1 : end + 1]
        nll_sum += float(losses[valid].detach().double().sum().cpu())
        scored_tokens += int(valid.sum().cpu())
    if scored_tokens <= 0 or not math.isfinite(nll_sum):
        raise RuntimeError("temporal prefix has no finite scoreable tokens")
    return {
        "nll_sum": nll_sum,
        "scored_tokens": scored_tokens,
        "mean_nll": nll_sum / scored_tokens,
        "finite": True,
    }


def _parse_temporal_domains(values: Sequence[str]) -> dict[str, Path]:
    domains: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--temporal-domain must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in domains:
            raise ValueError("temporal domain names must be unique and non-empty")
        domains[name] = Path(raw_path).expanduser()
    if len(domains) != 3:
        raise ValueError("registered temporal guardrail requires exactly three domains")
    return domains


def _temporal_domains_from_collection(root: Path) -> dict[str, Path]:
    collection_path = Path(root) / "collection_manifest.json"
    if not collection_path.is_file():
        raise FileNotFoundError(collection_path)
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if collection.get("schema") != "evq_cosh.temporal_holdout_2026.collection.v1":
        raise ValueError("temporal collection manifest schema mismatch")
    domains = {}
    for name, record in collection.get("domains", {}).items():
        manifest_path = Path(root) / str(record.get("manifest", ""))
        if sha256_file(manifest_path) != record.get("manifest_sha256"):
            raise ValueError(f"temporal collection hash mismatch for {name}")
        domains[str(name)] = manifest_path.parent
    if len(domains) != 3:
        raise ValueError("registered temporal collection requires exactly three domains")
    return domains


def _resolve_temporal_domains(args: argparse.Namespace) -> dict[str, Path]:
    temporal_root = getattr(args, "temporal_root", None)
    if temporal_root is not None:
        return _temporal_domains_from_collection(temporal_root)
    return _parse_temporal_domains(getattr(args, "temporal_domain", []))


def _load_temporal_selection(
    domain_paths: Mapping[str, Path], stage: str, *, max_packs_per_domain: int | None = 1
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], torch.Tensor, torch.Tensor]]]:
    spec = get_stage(stage)
    loaded = {}
    records = {}
    for name, root in sorted(domain_paths.items()):
        manifest, ids, mask = load_domain_artifacts(root)
        loaded[name] = (manifest, ids, mask)
        pack_count = int(ids.shape[0])
        if max_packs_per_domain is not None:
            if int(max_packs_per_domain) <= 0:
                raise ValueError("max_packs_per_domain must be positive or omitted")
            pack_count = min(pack_count, int(max_packs_per_domain))
        records[name] = {
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "pack_indices": list(range(pack_count)),
            "prefix_tokens": spec.seq_len,
            "input_file_sha256": manifest["files"]["input_ids"]["sha256"],
            "score_mask_file_sha256": manifest["files"]["score_mask"]["sha256"],
        }
    selection = {
        "stage": stage,
        "factor": spec.factor,
        "domains": records,
    }
    return selection, loaded


def _validate_temporal_tokenizers(
    domains: Mapping[str, tuple[dict[str, Any], torch.Tensor, torch.Tensor]],
    model_name: str,
) -> None:
    expected = tokenizer_source_fingerprint(model_name)
    for name, (manifest, _, _) in domains.items():
        try:
            validate_tokenizer_identity(manifest.get("tokenizer", {}), expected)
        except ValueError as exc:
            raise ValueError(f"temporal domain {name} tokenizer identity mismatch") from exc


def run_temporal(args: argparse.Namespace) -> dict[str, Any]:
    spec = get_stage(args.stage)
    paths = _resolve_temporal_domains(args)
    max_packs = getattr(args, "max_packs_per_domain", 1)
    if max_packs == 0:
        max_packs = None
    selection, domains = _load_temporal_selection(
        paths, args.stage, max_packs_per_domain=max_packs
    )
    _validate_temporal_tokenizers(domains, args.model_name)
    selection_sha = _json_sha256(selection)
    provenance, canonical, _ = _load_eval_identity(args)
    model, _, backbone, lm_head, runtime = _load_cuda_model(
        args, canonical, target_length=spec.seq_len
    )
    device = torch.device("cuda")
    domain_results = {}
    for name, (_, ids, mask) in sorted(domains.items()):
        pack_results = []
        for pack_index in selection["domains"][name]["pack_indices"]:
            input_ids = ids[pack_index : pack_index + 1, : spec.seq_len].to(
                device=device, dtype=torch.long
            )
            score_mask = mask[pack_index : pack_index + 1, : spec.seq_len].to(
                device=device
            )
            scored = _score_temporal_prefix(
                backbone,
                lm_head,
                input_ids,
                score_mask,
                chunk_tokens=args.lm_head_chunk_tokens,
            )
            pack_results.append({"pack_index": pack_index, **scored})
            print(
                json.dumps(
                    {
                        "temporal_domain": name,
                        "pack_index": pack_index,
                        "mean_nll": scored["mean_nll"],
                    }
                ),
                flush=True,
            )
        nll_sum = sum(float(row["nll_sum"]) for row in pack_results)
        scored_tokens = sum(int(row["scored_tokens"]) for row in pack_results)
        domain_results[name] = {
            "packs": pack_results,
            "nll_sum": nll_sum,
            "scored_tokens": scored_tokens,
            "mean_nll": nll_sum / scored_tokens,
            "finite": True,
        }
    temporal_summary = summarize_temporal_domains(domain_results)
    output = {
        "schema": TEMPORAL_SCHEMA,
        "stage": args.stage,
        "factor": spec.factor,
        **provenance,
        "temporal_selection": selection,
        "temporal_selection_sha256": selection_sha,
        "runtime": runtime,
        "runtime_tensor_sha256": runtime["runtime_tensor_sha256"],
        "evaluator_code_sha256": evaluator_code_sha256(),
        "domains": domain_results,
        "summary": temporal_summary,
    }
    _atomic_json_dump(output, args.output)
    return output


def _read_json(path: Path, *, schema: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise ValueError(f"result schema mismatch for {path}")
    return value


_SCHEMAS_BY_KIND = {
    "controlled": CONTROLLED_SCHEMA,
    "passkey": PASSKEY_SCHEMA,
    "temporal": TEMPORAL_SCHEMA,
    "capability": CAPABILITY_SCHEMA,
}


def _validate_controlled_raw_metrics(
    rows: Sequence[Mapping[str, Any]], *, eos_token_id: int
) -> None:
    for row in rows:
        if row.get("variant") == "source_removed":
            continue
        generated = _int_tokens(row.get("generated_ids", []))
        expected = _int_tokens(row.get("expected_ids", []))
        eos_index = generated.index(eos_token_id) if eos_token_id in generated else None
        content = generated if eos_index is None else generated[:eos_index]
        recomputed = {
            "strict_exact": content == expected,
            "first_value_exact": content[: len(expected)] == expected,
            "gold_containment": _contains_tokens(content, expected),
            "eos_terminated": eos_index is not None,
            "generated_token_count": len(content),
        }
        for field, expected_value in recomputed.items():
            if row.get(field) != expected_value:
                raise ValueError(f"controlled raw metric mismatch for {field}")


def _validate_passkey_raw_metrics(
    rows: Sequence[Mapping[str, Any]], *, eos_token_id: int
) -> None:
    for row in rows:
        generated = _int_tokens(row.get("generated_ids", []))
        eos = eos_token_id in generated
        recomputed = score_text_answer(
            str(row.get("prediction", "")), str(row.get("answer", "")), eos
        )
        for field, expected_value in recomputed.items():
            if row.get(field) != expected_value:
                raise ValueError(f"passkey raw metric mismatch for {field}")
        expected_count = (
            generated.index(eos_token_id) if eos else len(generated)
        )
        if int(row.get("generated_token_count", -1)) != expected_count:
            raise ValueError("passkey raw generated-token count mismatch")


def _validate_capability_raw_metrics(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        references = row.get("references")
        contract = row.get("scorer_contract")
        if not isinstance(references, list) or not references or not isinstance(contract, Mapping):
            raise ValueError("capability row is missing raw scorer evidence")
        metric = str(row.get("metric"))
        if metric == "mcqa":
            scores = row.get("choice_mean_logprobs")
            choices = row.get("choices")
            if not isinstance(scores, list) or not isinstance(choices, list) or len(scores) != len(choices):
                raise ValueError("MCQA row has invalid choice evidence")
            predicted = max(range(len(scores)), key=lambda index: float(scores[index]))
            gold = int(row.get("answer_index", -1))
            answer_tokens = int(row.get("answer_tokens", 0))
            if not 0 <= gold < len(scores) or answer_tokens <= 0:
                raise ValueError("MCQA gold/token evidence is invalid")
            if int(row.get("prediction_index", -1)) != predicted:
                raise ValueError("MCQA prediction index differs from raw scores")
            if float(row.get("metric_score", math.nan)) != float(predicted == gold):
                raise ValueError("MCQA metric differs from raw scores")
            wrong_best = max(float(scores[index]) for index in range(len(scores)) if index != gold)
            margin = float(scores[gold]) - wrong_best
            if not math.isclose(
                float(row.get("correct_minus_best_wrong_mean_logprob", math.nan)),
                margin,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("MCQA margin differs from raw scores")
            if not math.isclose(
                float(row.get("nll_sum", math.nan)),
                -float(scores[gold]) * answer_tokens,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("MCQA NLL differs from raw gold choice score")
            continue
        reference_scores = row.get("reference_mean_logprobs")
        answer_tokens = int(row.get("answer_tokens", 0))
        if (
            not isinstance(reference_scores, list)
            or len(reference_scores) != len(references)
            or answer_tokens <= 0
        ):
            raise ValueError("capability row has invalid reference NLL evidence")
        selected = max(
            range(len(reference_scores)), key=lambda index: float(reference_scores[index])
        )
        if int(row.get("selected_reference_index", -1)) != selected:
            raise ValueError("capability selected reference differs from raw scores")
        if not math.isclose(
            float(row.get("nll_sum", math.nan)),
            -float(reference_scores[selected]) * answer_tokens,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("capability NLL differs from selected reference score")
        source = {
            "match_type": contract.get("match_type"),
            "official_metric": contract.get("official_metric"),
        }
        recomputed_score = score_capability_prediction(
            metric,
            str(row.get("prediction", "")),
            [str(answer) for answer in references],
            source=source,
        )
        if not math.isclose(
            float(row.get("metric_score", math.nan)),
            recomputed_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("capability metric differs from raw prediction/references")
        generation = row.get("generation")
        if not isinstance(generation, Mapping):
            raise ValueError("capability row has no generation evidence")
        metrics = score_generation_metrics(
            str(row.get("prediction", "")),
            [str(answer) for answer in references],
            eos_terminated=bool(generation.get("eos_terminated")),
            generated_token_count=int(generation.get("generated_token_count", -1)),
        )
        for field, expected_value in metrics.items():
            if generation.get(field) != expected_value:
                raise ValueError(f"capability generation metric mismatch for {field}")


def validate_result_record(record: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    """Fail closed by recomputing every persisted aggregate from raw rows."""
    if kind not in _SCHEMAS_BY_KIND:
        raise ValueError(f"unknown result kind {kind!r}")
    value = dict(record)
    if value.get("schema") != _SCHEMAS_BY_KIND[kind]:
        raise ValueError(f"{kind} result schema mismatch")
    if value.get("evaluator_code_sha256") != evaluator_code_sha256():
        raise ValueError(f"{kind} result evaluator code hash mismatch")
    stage = str(value.get("stage", ""))
    if stage in {"r8", "r16"}:
        factor = get_stage(stage).factor
    elif stage == "x4":
        factor = 4.0
    else:
        raise ValueError(f"{kind} result stage mismatch")
    if not math.isclose(float(value.get("factor", math.nan)), factor, abs_tol=1e-12):
        raise ValueError(f"{kind} result factor mismatch")
    results = value.get("results")
    if kind != "temporal" and not isinstance(results, list):
        raise ValueError(f"{kind} result has no raw rows")

    if kind == "controlled":
        assert isinstance(results, list)
        eos_token_id = int(value.get("eos_token_id", -1))
        if eos_token_id < 0:
            raise ValueError("controlled result is missing eos_token_id")
        _validate_controlled_raw_metrics(results, eos_token_id=eos_token_id)
        split = value.get("split")
        if split not in {"validation", "test"}:
            raise ValueError("controlled result split mismatch")
        expected_rows = 48 if split == "validation" else 96
        if len(results) != expected_rows:
            raise ValueError("controlled result row budget mismatch")
        if value.get("bundle_file") != f"{split}_{stage}.pt":
            raise ValueError("controlled result bundle filename mismatch")
        expected_summary = summarize_repair_records(results, stage=stage)
    elif kind == "passkey":
        assert isinstance(results, list)
        eos_token_id = int(value.get("eos_token_id", -1))
        if eos_token_id < 0:
            raise ValueError("passkey result is missing eos_token_id")
        _validate_passkey_raw_metrics(results, eos_token_id=eos_token_id)
        target_length = int(value.get("target_length", 0))
        if registered_factor_for_length(target_length) != factor:
            raise ValueError("passkey result length/factor mismatch")
        filename, evaluation_split, _ = PASSKEY_BUNDLES[target_length]
        if value.get("bundle_file") != filename:
            raise ValueError("passkey result bundle filename mismatch")
        if value.get("evaluation_split") != evaluation_split:
            raise ValueError("passkey result evaluation split mismatch")
        if len(results) != 25:
            raise ValueError("passkey result row budget mismatch")
        expected_summary = summarize_passkey_records(results)
    elif kind == "temporal":
        selection = value.get("temporal_selection")
        if not isinstance(selection, Mapping):
            raise ValueError("temporal result has no frozen selection")
        if value.get("temporal_selection_sha256") != _json_sha256(selection):
            raise ValueError("temporal selection hash mismatch")
        domains = value.get("domains")
        if not isinstance(domains, Mapping):
            raise ValueError("temporal result has no domain evidence")
        selected_domains = selection.get("domains")
        if not isinstance(selected_domains, Mapping) or set(selected_domains) != set(domains):
            raise ValueError("temporal result domains differ from frozen selection")
        for name, raw_domain in domains.items():
            selected = selected_domains[name]
            if not isinstance(selected, Mapping):
                raise ValueError(f"temporal selection for {name} is invalid")
            selected_indices = [int(index) for index in selected.get("pack_indices", [])]
            result_indices = [
                int(pack.get("pack_index", -1)) for pack in raw_domain.get("packs", [])
            ]
            if result_indices != selected_indices:
                raise ValueError(
                    f"temporal result packs for {name} differ from frozen selection"
                )
        expected_summary = summarize_temporal_domains(domains)
    else:
        assert isinstance(results, list)
        _validate_capability_raw_metrics(results)
        target_length = int(value.get("target_length", 0))
        if registered_factor_for_length(target_length) != factor:
            raise ValueError("capability result length/factor mismatch")
        if value.get("status") != "complete_uninterpreted" or not results:
            raise ValueError("capability result must contain completed registered rows")
        if any(
            int(row.get("target_length", 0)) != target_length
            or not math.isclose(float(row.get("factor", math.nan)), factor, abs_tol=1e-12)
            for row in results
        ):
            raise ValueError("capability raw row identity mismatch")
        expected_budget = capability_pilot_budget(results, target_length=target_length)
        _require_json_equal(
            value.get("budget"), expected_budget, label="capability budget"
        )
        expected_summary = summarize_results(results)
    _require_json_equal(value.get("summary"), expected_summary, label=f"{kind} summary")
    return value


def validate_result_file(path: Path, *, kind: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return validate_result_record(
        json.loads(path.read_text(encoding="utf-8")),
        kind=kind,
    )


def _gate_input_bindings(
    *,
    stage: str,
    records: Mapping[str, Mapping[str, Any]],
    paths: Mapping[str, Path],
) -> dict[str, str]:
    spec = get_stage(stage)
    if any(record.get("stage") != stage for record in records.values()):
        raise ValueError("gate input stage mismatch")
    if any(
        not math.isclose(float(record.get("factor", math.nan)), spec.factor, abs_tol=1e-12)
        for record in records.values()
    ):
        raise ValueError("gate input factor mismatch")
    shared = (
        records["checkpoint_controlled"],
        records["passkey_result"],
        records["checkpoint_temporal"],
    )
    for field in (
        "adapter_sha256",
        "adapter_receipt_sha256",
        "model_manifest_sha256",
        "longalpaca_manifest_sha256",
        "repair_manifest_sha256",
        "runtime_tensor_sha256",
    ):
        if len({record.get(field) for record in shared}) != 1:
            raise ValueError(f"checkpoint gate inputs disagree on {field}")
    parent_controlled = records["parent_controlled"]
    parent_temporal = records["parent_temporal"]
    checkpoint = records["checkpoint_controlled"]
    passkey = records["passkey_result"]
    for field in ("adapter_sha256", "adapter_receipt_sha256"):
        if parent_controlled.get(field) != parent_temporal.get(field):
            raise ValueError(f"parent controlled/temporal {field} mismatch")
    for field in (
        "model_manifest_sha256",
        "longalpaca_manifest_sha256",
        "repair_manifest_sha256",
    ):
        if parent_controlled.get(field) != checkpoint.get(field):
            raise ValueError(f"parent/checkpoint {field} mismatch")
    return {
        "parent_adapter_sha256": str(parent_controlled["adapter_sha256"]),
        "parent_adapter_receipt_sha256": str(
            parent_controlled["adapter_receipt_sha256"]
        ),
        "checkpoint_adapter_sha256": str(checkpoint["adapter_sha256"]),
        "checkpoint_adapter_receipt_sha256": str(
            checkpoint["adapter_receipt_sha256"]
        ),
        "model_manifest_sha256": str(checkpoint["model_manifest_sha256"]),
        "longalpaca_manifest_sha256": str(checkpoint["longalpaca_manifest_sha256"]),
        "repair_manifest_sha256": str(checkpoint["repair_manifest_sha256"]),
        "validation_bundle_sha256": str(checkpoint["bundle_sha256"]),
        "passkey_bundle_sha256": str(passkey["bundle_sha256"]),
        "operator_tensor_sha256": str(checkpoint["runtime_tensor_sha256"]),
        "evaluator_code_sha256": evaluator_code_sha256(),
        **{
            binding: sha256_file(paths[role])
            for role, (_, binding) in _GATE_EVIDENCE.items()
        },
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        role: Path(getattr(args, role)) for role in _GATE_EVIDENCE
    }
    output_parent = Path(args.output).resolve().parent
    if any(path.resolve().parent != output_parent for path in paths.values()):
        raise ValueError("gate evidence and gate output must share one result directory")
    records = {
        "parent_controlled": validate_result_file(paths["parent_controlled"], kind="controlled"),
        "checkpoint_controlled": validate_result_file(paths["checkpoint_controlled"], kind="controlled"),
        "passkey_result": validate_result_file(paths["passkey_result"], kind="passkey"),
        "parent_temporal": validate_result_file(paths["parent_temporal"], kind="temporal"),
        "checkpoint_temporal": validate_result_file(paths["checkpoint_temporal"], kind="temporal"),
    }
    bindings = _gate_input_bindings(stage=args.stage, records=records, paths=paths)
    evidence_files = {
        role: {"path": path.name, "sha256": sha256_file(path)}
        for role, path in paths.items()
    }
    report = build_gate_report(
        stage=args.stage,
        segment=args.segment,
        repair_summary=records["checkpoint_controlled"]["summary"],
        passkey_summary=records["passkey_result"]["summary"],
        parent_repair_summary=records["parent_controlled"]["summary"],
        parent_temporal=records["parent_temporal"],
        checkpoint_temporal=records["checkpoint_temporal"],
        bindings=bindings,
        evidence_files=evidence_files,
    )
    _atomic_json_dump(report, args.output)
    return report


def validate_gate_file(path: Path) -> dict[str, Any]:
    """Hash-check gate evidence and reproduce the decision from raw rows."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("format_version") != 1 or gate.get("purpose") != GATE_PURPOSE:
        raise RuntimeError("repair gate identity mismatch")
    decision = gate.get("decision")
    if not isinstance(decision, Mapping) or gate.get("status") != decision.get("status"):
        raise RuntimeError("repair gate decision status differs from top-level status")
    bindings = gate.get("bindings")
    if not isinstance(bindings, Mapping):
        raise RuntimeError("repair gate has no bindings")
    stage = str(bindings.get("stage", ""))
    segment = int(bindings.get("segment", 0))
    spec = get_stage(stage)
    if segment not in (1, 2) or not math.isclose(
        float(bindings.get("factor", math.nan)), spec.factor, abs_tol=1e-12
    ):
        raise RuntimeError("repair gate stage/segment/factor mismatch")
    _validate_gate_bindings(
        {key: bindings.get(key) for key in _HASH_FIELDS}
    )
    if set(bindings) != {"stage", "segment", "factor", *_HASH_FIELDS}:
        raise RuntimeError("repair gate has missing or extra bindings")
    raw_evidence = gate.get("evidence_files")
    if not isinstance(raw_evidence, Mapping) or set(raw_evidence) != set(_GATE_EVIDENCE):
        raise RuntimeError("repair gate evidence file set mismatch")
    records: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for role, (schema, binding_name) in _GATE_EVIDENCE.items():
        evidence = raw_evidence.get(role)
        if not isinstance(evidence, Mapping):
            raise RuntimeError(f"repair gate evidence {role} is invalid")
        relative = Path(str(evidence.get("path", "")))
        if relative.is_absolute() or relative.name != str(relative):
            raise RuntimeError("repair gate evidence paths must be sibling filenames")
        evidence_path = path.parent / relative
        digest = sha256_file(evidence_path)
        if digest != evidence.get("sha256") or digest != bindings.get(binding_name):
            raise RuntimeError(f"repair gate evidence SHA-256 mismatch for {role}")
        kind = next(name for name, value in _SCHEMAS_BY_KIND.items() if value == schema)
        records[role] = validate_result_file(evidence_path, kind=kind)
        paths[role] = evidence_path
    expected_bindings = {
        "stage": stage,
        "segment": segment,
        "factor": spec.factor,
        **_gate_input_bindings(stage=stage, records=records, paths=paths),
    }
    _require_json_equal(bindings, expected_bindings, label="gate bindings")
    expected = build_gate_report(
        stage=stage,
        segment=segment,
        repair_summary=records["checkpoint_controlled"]["summary"],
        passkey_summary=records["passkey_result"]["summary"],
        parent_repair_summary=records["parent_controlled"]["summary"],
        parent_temporal=records["parent_temporal"],
        checkpoint_temporal=records["checkpoint_temporal"],
        bindings={key: expected_bindings[key] for key in _HASH_FIELDS},
        evidence_files=raw_evidence,
    )
    _require_json_equal(gate, expected, label="gate report")
    return gate


def check_gate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = validate_repair_parent(args.checkpoint_adapter)
    repair_manifest_sha = sha256_file(args.data_dir / "manifest.json")
    known = {
        "stage": args.stage,
        "segment": int(args.segment),
        "factor": get_stage(args.stage).factor,
        "checkpoint_adapter_sha256": checkpoint["adapter_sha256"],
        "checkpoint_adapter_receipt_sha256": checkpoint["adapter_receipt_sha256"],
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "longalpaca_manifest_sha256": sha256_file(args.longalpaca_manifest),
        "repair_manifest_sha256": repair_manifest_sha,
        "evaluator_code_sha256": evaluator_code_sha256(),
    }
    gate = validate_gate_file(args.gate)
    bindings = gate.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("gate has no bindings")
    expected = dict(bindings)
    expected.update(known)
    return validate_gate_transition(
        args.gate,
        allowed_statuses={args.status},
        expected_bindings=expected,
    )


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    spec = get_stage(args.stage)
    provenance, canonical, _ = _load_eval_identity(args)
    runtime = _runtime_contract_for_length(canonical, spec.seq_len)
    validation_record, _ = _bundle_record(args.data_dir, f"validation_{args.stage}.pt")
    passkey_filename, evaluation_split, _ = PASSKEY_BUNDLES[spec.seq_len]
    passkey_record, passkey_bundle = _bundle_record(args.data_dir, passkey_filename)
    if passkey_bundle.get("evaluation_split") != evaluation_split:
        raise ValueError("preflight passkey split mismatch")
    paths = _resolve_temporal_domains(args)
    selection, domains = _load_temporal_selection(
        paths, args.stage, max_packs_per_domain=1
    )
    _validate_temporal_tokenizers(domains, args.model_name)
    return {
        "status": "cpu_preflight_valid",
        "stage": args.stage,
        **provenance,
        "validation_bundle_sha256": validation_record["sha256"],
        "passkey_bundle_sha256": passkey_record["sha256"],
        "runtime_tensor_sha256": runtime["runtime_tensor_sha256"],
        "temporal_selection_sha256": _json_sha256(selection),
        "evaluation_budget": evaluation_budget(args.stage),
    }


def _model_hash_receipt(model_name: str, model_manifest: Path) -> dict[str, Any]:
    model_dir = Path(model_name).expanduser()
    manifest_path = Path(model_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_model_manifest(model_dir, manifest, verify_hashes=True)
    return {
        "format_version": 1,
        "purpose": "evq_seed42_retrieval_repair_model_hash_receipt",
        "status": "full_hashes_verified",
        "model": model_dir.name,
        "model_manifest_sha256": sha256_file(manifest_path),
        "files": [
            {"name": row["name"], "size_bytes": int(row["size_bytes"]), "sha256": row["sha256"]}
            for row in manifest["files"]
        ],
    }


def run_model_hash(args: argparse.Namespace) -> dict[str, Any]:
    """Spend CPU time once to verify every model byte before any GPU phase."""
    receipt = _model_hash_receipt(args.model_name, args.model_manifest)
    if args.output.exists():
        recorded = json.loads(args.output.read_text(encoding="utf-8"))
        _require_json_equal(recorded, receipt, label="model hash receipt")
    else:
        _atomic_json_dump(receipt, args.output)
    return receipt


def run_model_receipt_check(args: argparse.Namespace) -> dict[str, Any]:
    """Cheaply bind a prior full-hash receipt to the current manifest/stat identity."""
    expected = _model_hash_receipt(args.model_name, args.model_manifest) if args.rehash else None
    recorded = json.loads(args.receipt.read_text(encoding="utf-8"))
    if expected is not None:
        _require_json_equal(recorded, expected, label="model hash receipt")
    else:
        identity = load_model_identity(args.model_name, args.model_manifest)
        if (
            recorded.get("purpose") != "evq_seed42_retrieval_repair_model_hash_receipt"
            or recorded.get("status") != "full_hashes_verified"
            or recorded.get("model_manifest_sha256") != identity["manifest_sha256"]
            or recorded.get("files") != identity["files"]
        ):
            raise ValueError("model hash receipt differs from current model identity")
    return recorded


def run_capability_preflight(args: argparse.Namespace) -> dict[str, Any]:
    receipt, _ = _capability_budget_receipt(args.capability_dir)
    if args.output.exists():
        recorded = json.loads(args.output.read_text(encoding="utf-8"))
        _require_json_equal(recorded, receipt, label="capability budget receipt")
    else:
        _atomic_json_dump(receipt, args.output)
    return receipt


def run_preflight_complete(args: argparse.Namespace) -> dict[str, Any]:
    """Create or revalidate the receipt that alone authorizes paid GPU phases."""
    model_receipt = run_model_receipt_check(
        argparse.Namespace(
            model_name=args.model_name,
            model_manifest=args.model_manifest,
            receipt=args.model_receipt,
            rehash=False,
        )
    )
    capability_receipt, _ = _capability_budget_receipt(args.capability_dir)
    recorded_capability = json.loads(
        args.capability_receipt.read_text(encoding="utf-8")
    )
    _require_json_equal(
        recorded_capability,
        capability_receipt,
        label="capability budget receipt",
    )
    provenance, canonical, _ = _load_eval_identity(args)
    paths = _temporal_domains_from_collection(args.temporal_root)
    temporal = {}
    runtime = {}
    for stage in ("r8", "r16"):
        selection, domains = _load_temporal_selection(
            paths, stage, max_packs_per_domain=1
        )
        _validate_temporal_tokenizers(domains, args.model_name)
        temporal[stage] = _json_sha256(selection)
        runtime[stage] = _runtime_contract_for_length(
            canonical, get_stage(stage).seq_len
        )["runtime_tensor_sha256"]
    receipt = {
        "format_version": 1,
        "purpose": PREFLIGHT_COMPLETE_PURPOSE,
        "status": "all_cpu_gates_passed",
        "model_hash_receipt_sha256": sha256_file(args.model_receipt),
        "capability_budget_receipt_sha256": sha256_file(args.capability_receipt),
        "evaluator_code_sha256": evaluator_code_sha256(),
        "adapter_receipt_sha256": provenance["adapter_receipt_sha256"],
        "model_manifest_sha256": provenance["model_manifest_sha256"],
        "longalpaca_manifest_sha256": provenance["longalpaca_manifest_sha256"],
        "repair_manifest_sha256": provenance["repair_manifest_sha256"],
        "temporal_selection_sha256": temporal,
        "runtime_tensor_sha256": runtime,
        "capability_manifest_sha256": capability_receipt[
            "capability_manifest_sha256"
        ],
        "model_receipt_status": model_receipt["status"],
    }
    if args.output.exists():
        recorded = json.loads(args.output.read_text(encoding="utf-8"))
        _require_json_equal(recorded, receipt, label="complete preflight receipt")
    else:
        _atomic_json_dump(receipt, args.output)
    return receipt


def run_verify_result(args: argparse.Namespace) -> dict[str, Any]:
    """Validate an existing result before the launcher reuses it."""
    result = validate_result_file(args.result, kind=args.kind)
    provenance, _, _ = _load_eval_identity(args)
    for field, expected in (
        ("adapter_sha256", provenance["adapter_sha256"]),
        ("adapter_receipt_sha256", provenance["adapter_receipt_sha256"]),
        ("model_manifest_sha256", provenance["model_manifest_sha256"]),
        ("longalpaca_manifest_sha256", provenance["longalpaca_manifest_sha256"]),
        ("repair_manifest_sha256", provenance["repair_manifest_sha256"]),
    ):
        if result.get(field) != expected:
            raise ValueError(f"existing {args.kind} result differs on {field}")
    if args.kind in {"controlled", "temporal"}:
        if args.stage is None or result.get("stage") != args.stage:
            raise ValueError(f"existing {args.kind} result stage mismatch")
    if args.kind == "temporal":
        if args.temporal_root is None or args.max_packs_per_domain is None:
            raise ValueError(
                "temporal verification requires --temporal-root and "
                "--max-packs-per-domain"
            )
        max_packs = int(args.max_packs_per_domain)
        expected_selection, _ = _load_temporal_selection(
            _temporal_domains_from_collection(args.temporal_root),
            args.stage,
            max_packs_per_domain=None if max_packs == 0 else max_packs,
        )
        _require_json_equal(
            result.get("temporal_selection"),
            expected_selection,
            label="existing temporal selection",
        )
    if args.kind == "controlled":
        if args.split is None or result.get("split") != args.split:
            raise ValueError("existing controlled result split mismatch")
        record, _ = _bundle_record(
            args.data_dir, f"{args.split}_{args.stage}.pt"
        )
        if result.get("bundle_sha256") != record["sha256"]:
            raise ValueError("existing controlled result bundle hash mismatch")
    if args.kind in {"passkey", "capability"}:
        if args.target_length is None or int(result.get("target_length", 0)) != int(
            args.target_length
        ):
            raise ValueError(f"existing {args.kind} result target length mismatch")
    if args.kind == "passkey":
        filename, _, _ = PASSKEY_BUNDLES[int(args.target_length)]
        record, _ = _bundle_record(args.data_dir, filename)
        if result.get("bundle_sha256") != record["sha256"]:
            raise ValueError("existing passkey result bundle hash mismatch")
    if args.kind == "capability":
        if args.capability_dir is None:
            raise ValueError("capability verification requires --capability-dir")
        receipt, _ = _capability_budget_receipt(args.capability_dir)
        if (
            result.get("capability_manifest_sha256")
            != receipt["capability_manifest_sha256"]
        ):
            raise ValueError("existing capability result manifest hash mismatch")
    return {
        "status": "valid_for_reuse",
        "kind": args.kind,
        "result_sha256": sha256_file(args.result),
        "adapter_receipt_sha256": provenance["adapter_receipt_sha256"],
    }


def _build_final_report(args: argparse.Namespace) -> dict[str, Any]:
    result_dir = Path(args.result_dir)
    r8_gate = validate_gate_file(args.r8_gate)
    r16_gate = validate_gate_file(args.r16_gate)
    capability_receipt, _ = _capability_budget_receipt(args.capability_dir)
    specifications = {
        "test_r8": ("final_test_r8.json", "controlled"),
        "test_r16": ("final_test_r16.json", "controlled"),
        "passkey_32768": ("final_passkey_32768.json", "passkey"),
        "capability_8192": ("final_capability_8192.json", "capability"),
        "capability_16384": ("final_capability_16384.json", "capability"),
        "capability_32768": ("final_capability_32768.json", "capability"),
        "parent_temporal_r16": ("final_parent_temporal_full_r16.json", "temporal"),
        "checkpoint_temporal_r16": (
            "final_checkpoint_temporal_full_r16.json",
            "temporal",
        ),
    }
    results = {}
    for name, (filename, kind) in specifications.items():
        path = result_dir / filename
        record = validate_result_file(path, kind=kind)
        if kind == "capability" and (
            record.get("capability_manifest_sha256")
            != capability_receipt["capability_manifest_sha256"]
        ):
            raise ValueError(f"final capability manifest mismatch for {filename}")
        results[name] = {
            "file": filename,
            "sha256": sha256_file(path),
            "schema": record["schema"],
            "stage": record["stage"],
            "factor": record["factor"],
            "adapter_receipt_sha256": record["adapter_receipt_sha256"],
            "summary": record["summary"],
            **({"budget": record["budget"]} if kind == "capability" else {}),
        }
    final_receipts = {
        results[name]["adapter_receipt_sha256"]
        for name in (
            "test_r8",
            "test_r16",
            "passkey_32768",
            "capability_8192",
            "capability_16384",
            "capability_32768",
            "checkpoint_temporal_r16",
        )
    }
    if final_receipts != {
        r16_gate["bindings"]["checkpoint_adapter_receipt_sha256"]
    }:
        raise ValueError("final results do not share the selected passing r16 checkpoint")
    if (
        results["parent_temporal_r16"]["adapter_receipt_sha256"]
        != r8_gate["bindings"]["checkpoint_adapter_receipt_sha256"]
    ):
        raise ValueError("final temporal parent is not the selected passing r8 checkpoint")
    parent_temporal = validate_result_file(
        result_dir / specifications["parent_temporal_r16"][0], kind="temporal"
    )
    checkpoint_temporal = validate_result_file(
        result_dir / specifications["checkpoint_temporal_r16"][0], kind="temporal"
    )
    full_selection, _ = _load_temporal_selection(
        _temporal_domains_from_collection(args.temporal_root),
        "r16",
        max_packs_per_domain=None,
    )
    for name, record in (
        ("parent", parent_temporal),
        ("checkpoint", checkpoint_temporal),
    ):
        _require_json_equal(
            record.get("temporal_selection"),
            full_selection,
            label=f"final {name} full temporal selection",
        )
    return {
        "format_version": 1,
        "purpose": FINAL_REPORT_PURPOSE,
        "status": "complete_uninterpreted",
        "evaluator_code_sha256": evaluator_code_sha256(),
        "selected_gates": {
            "r8": {"sha256": sha256_file(args.r8_gate), "status": r8_gate["status"]},
            "r16": {"sha256": sha256_file(args.r16_gate), "status": r16_gate["status"]},
        },
        "capability_budget_receipt": capability_receipt,
        "full_temporal_guardrail": merge_temporal_guardrail(
            "r16", parent_temporal, checkpoint_temporal
        ),
        "results": results,
    }


def run_final_report(args: argparse.Namespace) -> dict[str, Any]:
    report = _build_final_report(args)
    if args.output.exists():
        recorded = json.loads(args.output.read_text(encoding="utf-8"))
        _require_json_equal(recorded, report, label="final report")
    else:
        _atomic_json_dump(report, args.output)
    return report


def _add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--longalpaca-manifest", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--adapter-kind", choices=("legacy", "repair"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    controlled = subparsers.add_parser("controlled")
    _add_common_eval_args(controlled)
    controlled.add_argument("--stage", choices=("r8", "r16"), required=True)
    controlled.add_argument("--split", choices=("validation", "test"), default="validation")
    controlled.add_argument("--output", type=Path, required=True)

    passkey = subparsers.add_parser("passkey")
    _add_common_eval_args(passkey)
    passkey.add_argument("--target-length", choices=(8192, 16384, 32768), type=int, required=True)
    passkey.add_argument("--output", type=Path, required=True)

    capability = subparsers.add_parser("capability")
    _add_common_eval_args(capability)
    capability.add_argument("--capability-dir", type=Path, required=True)
    capability.add_argument(
        "--target-length", choices=(8192, 16384, 32768), type=int, required=True
    )
    capability.add_argument("--mode", choices=("pilot", "full"), default="full")
    capability.add_argument("--output", type=Path, required=True)

    temporal = subparsers.add_parser("temporal")
    _add_common_eval_args(temporal)
    temporal.add_argument("--stage", choices=("r8", "r16"), required=True)
    temporal_source = temporal.add_mutually_exclusive_group(required=True)
    temporal_source.add_argument("--temporal-root", type=Path)
    temporal_source.add_argument("--temporal-domain", action="append")
    temporal.add_argument("--lm-head-chunk-tokens", type=int, default=256)
    temporal.add_argument("--max-packs-per-domain", type=int, default=1)
    temporal.add_argument("--output", type=Path, required=True)

    gate = subparsers.add_parser("gate")
    gate.add_argument("--stage", choices=("r8", "r16"), required=True)
    gate.add_argument("--segment", choices=(1, 2), type=int, required=True)
    gate.add_argument("--parent-controlled", type=Path, required=True)
    gate.add_argument("--checkpoint-controlled", type=Path, required=True)
    gate.add_argument("--passkey-result", type=Path, required=True)
    gate.add_argument("--parent-temporal", type=Path, required=True)
    gate.add_argument("--checkpoint-temporal", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)

    checker = subparsers.add_parser("check-gate")
    checker.add_argument("--gate", type=Path, required=True)
    checker.add_argument("--status", choices=("pass", "rescue_allowed"), required=True)
    checker.add_argument("--stage", choices=("r8", "r16"), required=True)
    checker.add_argument("--segment", choices=(1, 2), type=int, required=True)
    checker.add_argument("--checkpoint-adapter", type=Path, required=True)
    checker.add_argument("--model-manifest", type=Path, required=True)
    checker.add_argument("--longalpaca-manifest", type=Path, required=True)
    checker.add_argument("--data-dir", type=Path, required=True)

    preflight = subparsers.add_parser("preflight")
    _add_common_eval_args(preflight)
    preflight.add_argument("--stage", choices=("r8", "r16"), required=True)
    preflight_source = preflight.add_mutually_exclusive_group(required=True)
    preflight_source.add_argument("--temporal-root", type=Path)
    preflight_source.add_argument("--temporal-domain", action="append")

    model_hash = subparsers.add_parser("model-hash")
    model_hash.add_argument("--model-name", required=True)
    model_hash.add_argument("--model-manifest", type=Path, required=True)
    model_hash.add_argument("--output", type=Path, required=True)

    model_receipt = subparsers.add_parser("model-receipt")
    model_receipt.add_argument("--model-name", required=True)
    model_receipt.add_argument("--model-manifest", type=Path, required=True)
    model_receipt.add_argument("--receipt", type=Path, required=True)
    model_receipt.add_argument("--rehash", action="store_true")

    capability_preflight = subparsers.add_parser("capability-preflight")
    capability_preflight.add_argument("--capability-dir", type=Path, required=True)
    capability_preflight.add_argument("--output", type=Path, required=True)

    preflight_complete = subparsers.add_parser("preflight-complete")
    _add_common_eval_args(preflight_complete)
    preflight_complete.add_argument("--capability-dir", type=Path, required=True)
    preflight_complete.add_argument("--temporal-root", type=Path, required=True)
    preflight_complete.add_argument("--model-receipt", type=Path, required=True)
    preflight_complete.add_argument(
        "--capability-receipt", type=Path, required=True
    )
    preflight_complete.add_argument("--output", type=Path, required=True)

    verifier = subparsers.add_parser("verify-result")
    _add_common_eval_args(verifier)
    verifier.add_argument(
        "--kind", choices=("controlled", "passkey", "temporal", "capability"), required=True
    )
    verifier.add_argument("--result", type=Path, required=True)
    verifier.add_argument("--stage", choices=("r8", "r16"))
    verifier.add_argument("--split", choices=("validation", "test"))
    verifier.add_argument("--target-length", choices=(8192, 16384, 32768), type=int)
    verifier.add_argument("--capability-dir", type=Path)
    verifier.add_argument("--temporal-root", type=Path)
    verifier.add_argument("--max-packs-per-domain", type=int)

    final_report = subparsers.add_parser("final-report")
    final_report.add_argument("--result-dir", type=Path, required=True)
    final_report.add_argument("--r8-gate", type=Path, required=True)
    final_report.add_argument("--r16-gate", type=Path, required=True)
    final_report.add_argument("--capability-dir", type=Path, required=True)
    final_report.add_argument("--temporal-root", type=Path, required=True)
    final_report.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "controlled":
        output = run_controlled(args)
    elif args.command == "passkey":
        output = run_passkey(args)
    elif args.command == "capability":
        output = run_capability(args)
    elif args.command == "temporal":
        output = run_temporal(args)
    elif args.command == "gate":
        output = run_gate(args)
    elif args.command == "check-gate":
        output = check_gate(args)
    elif args.command == "preflight":
        output = run_preflight(args)
    elif args.command == "model-hash":
        output = run_model_hash(args)
    elif args.command == "model-receipt":
        output = run_model_receipt_check(args)
    elif args.command == "capability-preflight":
        output = run_capability_preflight(args)
    elif args.command == "preflight-complete":
        output = run_preflight_complete(args)
    elif args.command == "verify-result":
        output = run_verify_result(args)
    elif args.command == "final-report":
        output = run_final_report(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(output if args.command in {
        "gate", "check-gate", "preflight", "model-hash", "model-receipt",
        "capability-preflight", "preflight-complete"
        , "verify-result", "final-report"
    } else {
        "status": "complete_uninterpreted",
        "schema": output["schema"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
