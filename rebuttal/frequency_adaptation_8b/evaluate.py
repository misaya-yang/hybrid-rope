#!/usr/bin/env python3
"""Evaluate source-dependent capability for the 8B frequency curriculum."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as functional

from experiments.lora_evq_v2.train_evq_lora import (
    inject_inv_freq,
    load_frequency_artifact,
    public_model_identifier,
    verify_model_inv_freq,
)

from .curriculum import get_phase
from .prepare_data import sha256_file, validate_bundle
from .train import _capture_native_inv_freq, load_model_identity, require_tail_logits_support


@dataclass(frozen=True)
class ScoredRecord:
    group_id: str
    variant: str
    task_type: str
    distance: int
    nll: float
    exact: bool
    teacher_forced_exact: Optional[bool] = None


def teacher_forced_answer_metrics(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    answer_start: int,
    answer_end: int,
) -> Dict[str, Any]:
    """Score answer tokens using the correct one-token causal shift."""
    if logits.ndim != 2 or input_ids.ndim != 1:
        raise ValueError("logits/input_ids must have shapes [sequence,vocab] and [sequence]")
    answer_start = int(answer_start)
    answer_end = int(answer_end)
    if not 1 <= answer_start < answer_end <= input_ids.numel():
        raise ValueError("answer span must be non-empty, in range, and start after a prompt token")
    answer_length = answer_end - answer_start
    if logits.shape[0] == input_ids.numel():
        answer_logits = logits[answer_start - 1 : answer_end - 1]
    elif logits.shape[0] == answer_length + 1 and answer_end == input_ids.numel():
        answer_logits = logits[-answer_length - 1 : -1]
    else:
        raise ValueError("logit sequence axis does not cover the requested answer predictors")
    targets = input_ids[answer_start:answer_end].to(answer_logits.device, dtype=torch.long)
    nll = functional.cross_entropy(answer_logits.float(), targets, reduction="mean")
    exact = bool(torch.equal(answer_logits.argmax(dim=-1), targets))
    return {"nll": float(nll.item()), "exact": exact, "tokens": int(answer_length)}


def _summarize_groups(records: Sequence[ScoredRecord]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, ScoredRecord]] = {}
    for record in records:
        group = grouped.setdefault(record.group_id, {})
        if record.variant in group:
            raise ValueError(f"duplicate {record.variant} row in group {record.group_id}")
        group[record.variant] = record
    required = {"original", "swapped", "source_removed"}
    for group_id, variants in grouped.items():
        if set(variants) != required:
            raise ValueError(f"counterfactual group {group_id} does not contain {sorted(required)}")
    originals = [variants["original"] for variants in grouped.values()]
    swapped = [variants["swapped"] for variants in grouped.values()]
    removed = [variants["source_removed"] for variants in grouped.values()]
    removal_deltas = [absent.nll - present.nll for absent, present in zip(removed, originals)]
    count = len(grouped)
    return {
        "groups": count,
        "original_exact": sum(record.exact for record in originals) / count,
        "swapped_exact": sum(record.exact for record in swapped) / count,
        "pair_consistency": sum(original.exact and changed.exact for original, changed in zip(originals, swapped))
        / count,
        "original_nll_mean": sum(record.nll for record in originals) / count,
        "swapped_nll_mean": sum(record.nll for record in swapped) / count,
        "source_removed_nll_mean": sum(record.nll for record in removed) / count,
        "removal_nll_increase_mean": sum(removal_deltas) / count,
        "removal_positive_fraction": sum(delta > 0 for delta in removal_deltas) / count,
    }


def summarize_counterfactual_scores(
    records: Sequence[ScoredRecord],
    distance_bounds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Aggregate only within matched original/swap/removal groups."""
    if not records:
        raise ValueError("cannot summarize an empty score sequence")
    summary = _summarize_groups(records)
    tasks = sorted({record.task_type for record in records})
    summary["by_task"] = {
        task: _summarize_groups([record for record in records if record.task_type == task]) for task in tasks
    }
    if distance_bounds is not None:
        if len(distance_bounds) != 2:
            raise ValueError("distance_bounds must contain [minimum, maximum]")
        lower, upper = (int(distance_bounds[0]), int(distance_bounds[1]))
        if not 0 <= lower < upper:
            raise ValueError("distance bounds must be non-negative and increasing")
        width = upper - lower + 1
        edges = [lower, lower + width // 3, lower + (2 * width) // 3, upper + 1]
        buckets = {}
        for start, end in zip(edges[:-1], edges[1:]):
            selected = [record for record in records if start <= record.distance < end]
            if selected:
                buckets[f"{start}-{end - 1}"] = _summarize_groups(selected)
        summary["by_distance_bucket"] = buckets
    return summary


def _atomic_json_dump(value: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"evaluation output already exists: {path.name}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_eval_bundle(path: Path, phase_name: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"evaluation bundle not found: {path.name}")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    validate_bundle(
        bundle,
        expected_phase=phase_name,
        expected_split="eval",
        expected_seq_len=get_phase(phase_name).seq_len,
    )
    variants = {row.get("variant") for row in bundle["metadata"]}
    if variants != {"original", "swapped", "source_removed"}:
        raise ValueError("evaluation bundle must contain counterfactual triplets")
    return bundle


def _adapter_protocol(adapter_dir: Path, phase_name: str, arm: str, seed: int) -> Dict[str, Any]:
    path = adapter_dir / "run_protocol.json"
    if not path.is_file():
        raise FileNotFoundError("adapter is missing run_protocol.json")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "complete_uninterpreted":
        raise RuntimeError("adapter phase is not marked complete_uninterpreted")
    if protocol.get("phase") != phase_name or protocol.get("arm") != arm:
        raise RuntimeError("adapter phase/arm does not match evaluation request")
    if int(protocol.get("seed", -1)) != int(seed):
        raise RuntimeError("adapter seed does not match evaluation request")
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate counterfactual retrieval for one frequency-adaptation checkpoint"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter-dir", type=Path)
    source.add_argument("--base-only", action="store_true")
    parser.add_argument(
        "--phase",
        required=True,
        choices=("warmup", "transition", "exact_8k", "exact_16k"),
    )
    parser.add_argument("--arm", required=True, choices=("geo", "evq"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.base_only and args.arm != "geo":
        raise ValueError("base-only evaluation is native Geo")
    bundle = _load_eval_bundle(args.data, args.phase)
    phase = get_phase(args.phase)
    adapter_protocol = None
    if args.adapter_dir is not None:
        adapter_protocol = _adapter_protocol(args.adapter_dir, args.phase, args.arm, args.seed)
    model_identity = load_model_identity(args.model_name, args.model_manifest)
    if adapter_protocol is not None and adapter_protocol.get("model") != model_identity:
        raise RuntimeError("adapter and evaluator do not use the same model bytes")
    require_tail_logits_support()
    if not torch.cuda.is_available():
        raise RuntimeError("8B evaluation requires CUDA")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=Path(args.model_name).expanduser().is_dir(),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        local_files_only=Path(args.model_name).expanduser().is_dir(),
    )
    frequency_provenance: Dict[str, Any]
    if args.adapter_dir is None:
        inv_freq = _capture_native_inv_freq(model)
        frequency_provenance = {"method": "native_geo", "artifact": None}
    else:
        method = "evq_cosh" if args.arm == "evq" else "native_geo"
        inv_freq, frequency_artifact, frequency_provenance = load_frequency_artifact(
            args.adapter_dir / "custom_inv_freq.pt",
            expected_method=method,
        )
        if frequency_artifact.get("phase") != args.phase or frequency_artifact.get("arm") != args.arm:
            raise RuntimeError("frequency artifact phase/arm does not match evaluation request")
        model = PeftModel.from_pretrained(model, str(args.adapter_dir), is_trainable=False)
    inject_inv_freq(model, inv_freq)
    verify_model_inv_freq(model, inv_freq)
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device

    forward_target = model.get_base_model() if hasattr(model, "get_base_model") else model
    if "logits_to_keep" not in inspect.signature(forward_target.forward).parameters:
        raise RuntimeError("loaded model does not expose the required logits_to_keep contract")
    scored: List[ScoredRecord] = []
    started = time.time()
    for index, metadata in enumerate(bundle["metadata"]):
        input_ids = bundle["input_ids"][index].to(device=device, dtype=torch.long)
        answer_start = int(bundle["answer_start"][index])
        answer_end = int(bundle["answer_end"][index])
        if answer_end != input_ids.numel():
            raise ValueError("v1 evaluator requires the supervised answer at sequence end")
        answer_length = answer_end - answer_start
        forward_kwargs: Dict[str, Any] = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": torch.ones_like(input_ids).unsqueeze(0),
            "use_cache": False,
        }
        forward_kwargs["logits_to_keep"] = answer_length + 1
        with torch.inference_mode():
            logits = model(**forward_kwargs).logits[0]
        teacher_metrics = teacher_forced_answer_metrics(
            logits,
            input_ids,
            answer_start=answer_start,
            answer_end=answer_end,
        )
        del logits

        exact = bool(teacher_metrics["exact"])
        if not args.skip_generation and metadata["variant"] != "source_removed":
            prompt = input_ids[:answer_start].unsqueeze(0)
            with torch.inference_mode():
                generated = model.generate(
                    prompt,
                    attention_mask=torch.ones_like(prompt),
                    max_new_tokens=answer_length,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )[0, answer_start:]
            expected = input_ids[answer_start:answer_end]
            exact = bool(generated.numel() == expected.numel() and torch.equal(generated, expected))

        scored.append(
            ScoredRecord(
                group_id=str(metadata["group_id"]),
                variant=str(metadata["variant"]),
                task_type=str(metadata["task_type"]),
                distance=int(metadata["distance"]),
                nll=float(teacher_metrics["nll"]),
                exact=exact,
                teacher_forced_exact=bool(teacher_metrics["exact"]),
            )
        )

    distance_bounds = (phase.min_distance, phase.max_distance)
    summary = summarize_counterfactual_scores(scored, distance_bounds=distance_bounds)
    teacher_scored = [replace(record, exact=bool(record.teacher_forced_exact)) for record in scored]
    result = {
        "format_version": 1,
        "purpose": "llama8b_rope_frequency_adaptation_evaluation",
        "status": "measured_uninterpreted",
        "model": public_model_identifier(args.model_name),
        "model_fingerprint": model_identity,
        "phase": args.phase,
        "arm": args.arm,
        "seed": int(args.seed),
        "checkpoint": "base" if args.adapter_dir is None else args.adapter_dir.name,
        "data": {
            "name": args.data.name,
            "sha256": sha256_file(args.data),
            "generation_seed": int(bundle["seed"]),
        },
        "frequency": frequency_provenance,
        "exact_metric": ("teacher_forced_argmax" if args.skip_generation else "autoregressive_token_sequence"),
        "summary": summary,
        "teacher_forced_summary": summarize_counterfactual_scores(teacher_scored, distance_bounds=distance_bounds),
        "records": [asdict(record) for record in scored],
        "elapsed_seconds": time.time() - started,
        "adapter_protocol_sha256": (
            sha256_file(args.adapter_dir / "run_protocol.json") if args.adapter_dir is not None else None
        ),
        "adapter_protocol_status": (adapter_protocol.get("status") if adapter_protocol is not None else None),
    }
    _atomic_json_dump(result, args.output)
    print(
        f"wrote {args.phase}/{args.arm} counterfactual evaluation to {args.output.name}; "
        "interpret only after the registered gate comparison"
    )


if __name__ == "__main__":
    main()
