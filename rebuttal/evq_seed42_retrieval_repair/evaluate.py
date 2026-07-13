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
from rebuttal.frequency_adaptation_8b.train import load_model_identity
from scripts.lib.rope.official_yarn import official_yarn_on_inv_freq

from .prepare_data import (
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
GATE_PURPOSE = "evq_seed42_retrieval_repair_gate"

_HASH_FIELDS = {
    "parent_adapter_sha256",
    "checkpoint_adapter_sha256",
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
    project_root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("protocol.py").resolve(),
        Path(__file__).with_name("train.py").resolve(),
        (project_root / "scripts/lib/rope/official_yarn.py").resolve(),
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
    return {
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
        "evaluation_budget": evaluation_budget(args.stage),
        "results": rows,
        "summary": summary,
    }
    _atomic_json_dump(output, args.output)
    return output


def run_passkey(args: argparse.Namespace) -> dict[str, Any]:
    target_length = int(args.target_length)
    factor = registered_factor_for_length(target_length)
    filename = f"passkey_{target_length}.pt"
    record, bundle = _bundle_record(args.data_dir, filename)
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
    finite = all(bool(row["finite"]) for row in rows)
    summary = {
        "rows": len(rows),
        "passkey_strict_exact": sum(bool(row["strict_exact"]) for row in rows) / len(rows),
        "passkey_first_value_exact": sum(bool(row["first_value_exact"]) for row in rows) / len(rows),
        "passkey_containment": sum(bool(row["gold_containment"]) for row in rows) / len(rows),
        "eos_fraction": sum(bool(row["eos_terminated"]) for row in rows) / len(rows),
        "mean_nll": sum(float(row["nll_sum"]) for row in rows)
        / sum(int(row["answer_tokens"]) for row in rows),
        "finite": finite,
    }
    output = {
        "schema": PASSKEY_SCHEMA,
        "stage": {8192: "r8", 16384: "r16", 32768: "x4"}[target_length],
        "factor": factor,
        "target_length": target_length,
        **provenance,
        "bundle_file": filename,
        "bundle_sha256": record["sha256"],
        "runtime": runtime,
        "runtime_tensor_sha256": runtime["runtime_tensor_sha256"],
        "evaluator_code_sha256": evaluator_code_sha256(),
        "results": rows,
        "summary": summary,
    }
    _atomic_json_dump(output, args.output)
    return output


def run_capability(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate the frozen RULER/downstream suite at one registered length."""
    target_length = int(args.target_length)
    factor = registered_factor_for_length(target_length)
    manifest, all_rows = load_capability_suite(args.capability_dir)
    validate_tokenizer_identity(
        manifest.get("tokenizer", {}), tokenizer_source_fingerprint(args.model_name)
    )
    length_rows = [
        row for row in all_rows if int(row["target_length"]) == target_length
    ]
    rows = select_rows(length_rows, mode=args.mode) if length_rows else []
    provenance, canonical, _ = _load_eval_identity(args)
    if not rows:
        output = {
            "schema": "evq_cosh.seed42_retrieval_repair_capability.v1",
            "stage": {8192: "r8", 16384: "r16", 32768: "x4"}[target_length],
            "factor": factor,
            "target_length": target_length,
            **provenance,
            "capability_manifest_sha256": sha256_file(
                args.capability_dir / "manifest.json"
            ),
            "evaluator_code_sha256": evaluator_code_sha256(),
            "status": "no_registered_rows_at_length",
            "results": [],
            "summary": {},
        }
        _atomic_json_dump(output, args.output)
        return output
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
        "schema": "evq_cosh.seed42_retrieval_repair_capability.v1",
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
    mean_nll = sum(float(row["mean_nll"]) for row in domain_results.values()) / len(domain_results)
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
        "summary": {"mean_nll": mean_nll, "finite": math.isfinite(mean_nll)},
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


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    parent_controlled = _read_json(args.parent_controlled, schema=CONTROLLED_SCHEMA)
    checkpoint_controlled = _read_json(args.checkpoint_controlled, schema=CONTROLLED_SCHEMA)
    passkey = _read_json(args.passkey_result, schema=PASSKEY_SCHEMA)
    parent_temporal = _read_json(args.parent_temporal, schema=TEMPORAL_SCHEMA)
    checkpoint_temporal = _read_json(args.checkpoint_temporal, schema=TEMPORAL_SCHEMA)
    spec = get_stage(args.stage)
    records = (parent_controlled, checkpoint_controlled, passkey, parent_temporal, checkpoint_temporal)
    if any(record.get("stage") != args.stage for record in records):
        raise ValueError("gate input stage mismatch")
    if any(not math.isclose(float(record.get("factor", math.nan)), spec.factor, abs_tol=1e-12) for record in records):
        raise ValueError("gate input factor mismatch")
    current_code_sha = evaluator_code_sha256()
    if any(record.get("evaluator_code_sha256") != current_code_sha for record in records):
        raise ValueError("gate input evaluator code hash mismatch")
    shared = (checkpoint_controlled, passkey, checkpoint_temporal)
    for field in (
        "adapter_sha256",
        "model_manifest_sha256",
        "longalpaca_manifest_sha256",
        "repair_manifest_sha256",
        "runtime_tensor_sha256",
    ):
        if len({record.get(field) for record in shared}) != 1:
            raise ValueError(f"checkpoint gate inputs disagree on {field}")
    if parent_controlled.get("adapter_sha256") != parent_temporal.get("adapter_sha256"):
        raise ValueError("parent controlled/temporal adapter mismatch")
    if parent_controlled.get("model_manifest_sha256") != checkpoint_controlled.get("model_manifest_sha256"):
        raise ValueError("parent/checkpoint model manifest mismatch")
    bindings = {
        "parent_adapter_sha256": parent_controlled["adapter_sha256"],
        "checkpoint_adapter_sha256": checkpoint_controlled["adapter_sha256"],
        "model_manifest_sha256": checkpoint_controlled["model_manifest_sha256"],
        "longalpaca_manifest_sha256": checkpoint_controlled["longalpaca_manifest_sha256"],
        "repair_manifest_sha256": checkpoint_controlled["repair_manifest_sha256"],
        "validation_bundle_sha256": checkpoint_controlled["bundle_sha256"],
        "passkey_bundle_sha256": passkey["bundle_sha256"],
        "operator_tensor_sha256": checkpoint_controlled["runtime_tensor_sha256"],
        "evaluator_code_sha256": current_code_sha,
        "parent_retrieval_result_sha256": sha256_file(args.parent_controlled),
        "retrieval_result_sha256": sha256_file(args.checkpoint_controlled),
        "passkey_result_sha256": sha256_file(args.passkey_result),
        "parent_temporal_sha256": sha256_file(args.parent_temporal),
        "checkpoint_temporal_sha256": sha256_file(args.checkpoint_temporal),
    }
    report = build_gate_report(
        stage=args.stage,
        segment=args.segment,
        repair_summary=checkpoint_controlled["summary"],
        passkey_summary=passkey["summary"],
        parent_repair_summary=parent_controlled["summary"],
        parent_temporal=parent_temporal,
        checkpoint_temporal=checkpoint_temporal,
        bindings=bindings,
    )
    _atomic_json_dump(report, args.output)
    return report


def check_gate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = validate_repair_parent(args.checkpoint_adapter)
    repair_manifest_sha = sha256_file(args.data_dir / "manifest.json")
    known = {
        "stage": args.stage,
        "segment": int(args.segment),
        "factor": get_stage(args.stage).factor,
        "checkpoint_adapter_sha256": checkpoint["adapter_sha256"],
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "longalpaca_manifest_sha256": sha256_file(args.longalpaca_manifest),
        "repair_manifest_sha256": repair_manifest_sha,
        "evaluator_code_sha256": evaluator_code_sha256(),
    }
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
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
    passkey_record, _ = _bundle_record(args.data_dir, f"passkey_{spec.seq_len}.pt")
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
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(output if args.command in {"gate", "check-gate", "preflight"} else {
        "status": "complete_uninterpreted",
        "schema": output["schema"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
