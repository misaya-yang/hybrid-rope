#!/usr/bin/env python3
"""Generation-only 16K-max LongBench QA comparison for the seed-42 LoRA pair."""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.lora_evq_v2.eval_official_yarn_capability import (
    load_capability_suite,
    score_capability_prediction,
    score_generation_metrics,
    sha256_file,
)
from experiments.lora_evq_v2.eval_sparse_conversion import (
    _load_arm_model,
    _tokenizer_identity_matches,
    _validate_arm,
    _validate_config,
)
from experiments.lora_evq_v2.prepare_legacy_model_manifest import validate_model_manifest
from experiments.lora_evq_v2.prepare_positional_distill_data import (
    tokenizer_source_fingerprint,
)
from experiments.lora_evq_v2.prepare_seed42_capability_data import (
    _local_source_revisions,
    _tokenizer_identity,
    build_longbench_examples,
    write_suite_atomic,
)


SCHEMA = "evq_cosh.lora_qa16k_generation.v1"
SUMMARY_SCHEMA = "evq_cosh.lora_qa16k_three_arm_summary.v1"
TARGET_LENGTH = 16_384
MAX_SOURCE_PROMPT = 32_768
EXPECTED_TASKS = {"qasper": 200, "narrativeqa": 103}
LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
SOURCE_RECEIPTS = {
    "qasper.jsonl": "29aa07d2a63f36f4fb8e8cd200a3428ee3126d750bde6af1c8f9bc41c2366854",
    "narrativeqa.jsonl": "0fb8d08ba5cdad4b74244224b0dc2e8b41ee6b850d954a13eb2d282621ce2f71",
}
ARM_SUBSTRATE = {
    "base_native": "native_geo",
    "native_lora": "native_geo",
    "evq_lora": "evq_cosh",
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validate_rows(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    counts = Counter(str(row["task"]) for row in rows)
    if dict(counts) != EXPECTED_TASKS or len(rows) != sum(EXPECTED_TASKS.values()):
        raise ValueError(f"QA suite task counts differ from {EXPECTED_TASKS}: {dict(counts)}")
    if manifest.get("task_counts") != EXPECTED_TASKS:
        raise ValueError("QA manifest task counts differ from the loaded rows")
    if len({str(row["example_id"]) for row in rows}) != len(rows):
        raise ValueError("QA suite contains duplicate example IDs")
    for row in rows:
        if row.get("suite") != "longbench" or row.get("metric") != "qa_f1":
            raise ValueError("QA suite contains a non-LongBench/non-F1 row")
        if int(row["target_length"]) != TARGET_LENGTH:
            raise ValueError("QA suite contains another registered target length")
        if not 0 < len(row["prompt_ids"]) <= TARGET_LENGTH:
            raise ValueError("QA prompt is empty or exceeds 16K")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    source_paths = [args.source_dir / name for name in SOURCE_RECEIPTS]
    for path in source_paths:
        if not path.is_file() or sha256_file(path) != SOURCE_RECEIPTS[path.name]:
            raise ValueError(f"LongBench source receipt mismatch: {path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), trust_remote_code=True, use_fast=True, local_files_only=True
    )
    candidates = build_longbench_examples(
        tokenizer,
        source_paths,
        max_prompt_tokens=MAX_SOURCE_PROMPT,
        min_complete_prompt_tokens=1,
        diagnostic_lengths=(TARGET_LENGTH,),
    )
    selected = []
    for row in candidates:
        source = row["source"]
        selection = source["selection"]
        untruncated = int(source.get("untruncated_prompt_tokens", len(row["prompt_ids"])))
        keep_complete = selection == "complete" and len(row["prompt_ids"]) <= TARGET_LENGTH
        keep_diagnostic = (
            selection == "fixed_diagnostic"
            and TARGET_LENGTH < untruncated <= MAX_SOURCE_PROMPT
        )
        if not (keep_complete or keep_diagnostic):
            continue
        item = copy.deepcopy(row)
        item["task"] = str(item["task"]).removesuffix("_fixed_diagnostic")
        item["target_length"] = TARGET_LENGTH
        item["source"]["original_selection"] = selection
        item["source"]["selection"] = "longbench_max_context_16k"
        item["source"]["untruncated_prompt_tokens"] = untruncated
        item["source"]["length_semantics"] = "complete_prompt_or_document_only_truncation"
        selected.append(item)

    provisional = {"task_counts": dict(Counter(row["task"] for row in selected))}
    _validate_rows(provisional, selected)
    manifest = write_suite_atomic(
        output_dir=args.output_dir,
        records_by_file={"longbench.jsonl": selected},
        tokenizer_identity=_tokenizer_identity(tokenizer, args.tokenizer),
        source_revisions={
            "longbench": {
                "repository": "THUDM/LongBench",
                "revision": LONGBENCH_REVISION,
                "files": _local_source_revisions(source_paths),
            },
            "selection": {
                "source_complete_prompt_max_tokens": MAX_SOURCE_PROMPT,
                "evaluation_max_tokens": TARGET_LENGTH,
                "truncation": "document_only",
            },
        },
    )
    _validate_rows(manifest, selected)
    return manifest


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if torch.cuda.is_available():
        raise RuntimeError("preflight must run before the paid GPU is enabled")
    if not args.gpu_command.is_file():
        raise FileNotFoundError(args.gpu_command)
    config = _validate_config(str(args.model))
    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    validate_model_manifest(args.model, model_manifest, verify_hashes=True)
    manifest, rows = load_capability_suite(args.data_root)
    _validate_rows(manifest, rows)
    expected_tokenizer = tokenizer_source_fingerprint(str(args.model))
    if not _tokenizer_identity_matches(manifest.get("tokenizer", {}), expected_tokenizer):
        raise ValueError("QA suite tokenizer differs from the model tokenizer")
    geo = _validate_arm(
        adapter_dir=args.geo_adapter,
        substrate="native_geo",
        training_manifest=args.training_manifest,
        model_manifest=args.model_manifest,
    )
    evq = _validate_arm(
        adapter_dir=args.evq_adapter,
        substrate="evq_cosh",
        training_manifest=args.training_manifest,
        model_manifest=args.model_manifest,
    )
    for key in ("model_manifest_sha256", "training_manifest_sha256", "code_sha256"):
        if geo[key] != evq[key]:
            raise ValueError(f"matched adapters differ at {key}")
    receipt = {
        "schema": "evq_cosh.lora_qa16k_ready.v1",
        "status": "ready",
        "model_contract": config,
        "data": {
            "manifest_sha256": sha256_file(args.data_root / "manifest.json"),
            "rows": len(rows),
            "task_counts": dict(Counter(row["task"] for row in rows)),
            "prompt_tokens": {
                "min": min(len(row["prompt_ids"]) for row in rows),
                "max": max(len(row["prompt_ids"]) for row in rows),
                "sum": sum(len(row["prompt_ids"]) for row in rows),
            },
        },
        "arms": {
            "base_native": {"adapter_enabled": False, "substrate": "native_geo"},
            "native_lora": {key: value for key, value in geo.items() if key != "metadata"},
            "evq_lora": {key: value for key, value in evq.items() if key != "metadata"},
        },
        "script_sha256": _script_sha256(),
        "gpu_command": str(args.gpu_command),
        "gpu_command_sha256": sha256_file(args.gpu_command),
    }
    _atomic_json(args.output, receipt)
    return receipt


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(row)
    tasks = {
        task: {
            "examples": len(cell),
            "f1": statistics.fmean(float(row["f1"]) for row in cell),
            "strict_exact": statistics.fmean(float(row["strict_exact"]) for row in cell),
            "empty_output_fraction": statistics.fmean(
                float(not str(row["prediction"]).strip()) for row in cell
            ),
        }
        for task, cell in sorted(grouped.items())
    }
    return {
        "tasks": tasks,
        "task_macro_f1": statistics.fmean(cell["f1"] for cell in tasks.values()),
        "pooled_f1": statistics.fmean(float(row["f1"]) for row in rows),
    }


@torch.inference_mode()
def _generate_dense(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device="cuda")
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0, input_ids.shape[1] :].tolist()
    eos = tokenizer.eos_token_id
    eos_terminated = eos is not None and bool(generated) and generated[-1] == int(eos)
    content = generated[:-1] if eos_terminated else generated
    return {
        "prediction": tokenizer.decode(content, skip_special_tokens=True).strip(),
        "generated_token_count": len(content),
        "eos_terminated": eos_terminated,
    }


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    expected_substrate = ARM_SUBSTRATE[args.arm]
    args.substrate = expected_substrate
    manifest, rows = load_capability_suite(args.data_root)
    _validate_rows(manifest, rows)
    expected_tokenizer = tokenizer_source_fingerprint(str(args.model_name))
    if not _tokenizer_identity_matches(manifest.get("tokenizer", {}), expected_tokenizer):
        raise ValueError("QA suite tokenizer differs from the model tokenizer")

    model, tokenizer, identity = _load_arm_model(args)
    context = model.disable_adapter() if args.arm == "base_native" else nullcontext()
    rows = sorted(rows, key=lambda row: (str(row["task"]), str(row["example_id"])))
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    results = []
    with context:
        for index, row in enumerate(rows, start=1):
            generation = _generate_dense(
                model,
                tokenizer,
                row["prompt_ids"],
                int(row["generation_tokens"]),
            )
            prediction = str(generation["prediction"])
            metrics = score_generation_metrics(
                prediction,
                row["answers"],
                eos_terminated=bool(generation["eos_terminated"]),
                generated_token_count=int(generation["generated_token_count"]),
            )
            result = {
                "example_id": row["example_id"],
                "task": row["task"],
                "prompt_sha256": row["prompt_sha256"],
                "prompt_tokens": len(row["prompt_ids"]),
                "references": list(row["answers"]),
                "prediction": prediction,
                "f1": score_capability_prediction(
                    "qa_f1", prediction, row["answers"], source=row.get("source")
                ),
                **metrics,
            }
            results.append(result)
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "progress": f"{index}/{len(rows)}",
                        "task": row["task"],
                        "f1": result["f1"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    torch.cuda.synchronize()
    output = {
        "schema": SCHEMA,
        "status": "complete",
        "arm": args.arm,
        "substrate": expected_substrate,
        "adapter_enabled": args.arm != "base_native",
        "single_seed_supporting": True,
        "paper_claim": False,
        "raw_extrapolation": True,
        "protocol": {
            "benchmark": "LongBench v1 Qasper + NarrativeQA",
            "max_context_tokens": TARGET_LENGTH,
            "truncation": "document_only",
            "decoding": "greedy",
            "teacher_forced_nll": False,
        },
        "data_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
        "adapter_identity": identity,
        "aggregate": _aggregate(results),
        "results": results,
        "script_sha256": _script_sha256(),
        "runtime": {
            "seconds": time.time() - started,
            "input_tokens": sum(int(row["prompt_tokens"]) for row in results),
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        },
    }
    _atomic_json(args.output, output)
    return output


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))]


def _comparison(
    left: Mapping[tuple[str, str], Mapping[str, Any]],
    right: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    seed: int = 42,
    trials: int = 10_000,
) -> dict[str, Any]:
    by_task: defaultdict[str, list[float]] = defaultdict(list)
    for key in sorted(left):
        by_task[key[0]].append(float(left[key]["f1"]) - float(right[key]["f1"]))
    task_delta = {task: statistics.fmean(values) for task, values in sorted(by_task.items())}
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(trials):
        sampled_task_means = []
        for values in by_task.values():
            sampled_task_means.append(
                statistics.fmean(values[rng.randrange(len(values))] for _ in values)
            )
        bootstrap.append(statistics.fmean(sampled_task_means))
    return {
        "task_delta_f1": task_delta,
        "task_macro_delta_f1": statistics.fmean(task_delta.values()),
        "paired_task_macro_bootstrap_95ci": [
            _percentile(bootstrap, 0.025),
            _percentile(bootstrap, 0.975),
        ],
        "bootstrap_trials": trials,
        "bootstrap_seed": seed,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "base_native": args.base_native,
        "native_lora": args.native_lora,
        "evq_lora": args.evq_lora,
    }
    documents = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    indexed = {}
    for name, document in documents.items():
        if document.get("schema") != SCHEMA or document.get("status") != "complete":
            raise ValueError(f"{name} is not a complete QA result")
        if document.get("arm") != name:
            raise ValueError(f"{name} result has the wrong arm identity")
        indexed[name] = {
            (str(row["task"]), str(row["example_id"])): row for row in document["results"]
        }
    keys = set(indexed["base_native"])
    if any(set(rows) != keys for rows in indexed.values()):
        raise ValueError("three QA arms do not contain identical examples")
    for key in keys:
        prompt_hashes = {rows[key]["prompt_sha256"] for rows in indexed.values()}
        if len(prompt_hashes) != 1:
            raise ValueError(f"three QA arms differ at prompt {key}")

    comparisons = {
        "evq_minus_native_lora": _comparison(indexed["evq_lora"], indexed["native_lora"]),
        "evq_minus_base_native": _comparison(indexed["evq_lora"], indexed["base_native"]),
        "native_lora_minus_base_native": _comparison(
            indexed["native_lora"], indexed["base_native"]
        ),
    }
    main_ci = comparisons["evq_minus_native_lora"]["paired_task_macro_bootstrap_95ci"]
    gate = "positive" if main_ci[0] > 0 else "negative" if main_ci[1] < 0 else "inconclusive"
    output = {
        "schema": SUMMARY_SCHEMA,
        "status": "complete",
        "single_seed_supporting": True,
        "paper_claim": False,
        "primary_question": "Does EVQ-LoRA outperform matched Native-RoPE-LoRA on 16K-max QA?",
        "gate": gate,
        "absolute": {name: document["aggregate"] for name, document in documents.items()},
        "comparisons": comparisons,
        "inputs": {name: sha256_file(path) for name, path in paths.items()},
    }
    _atomic_json(args.output, output)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prep = commands.add_parser("prepare")
    prep.add_argument("--tokenizer", type=Path, required=True)
    prep.add_argument("--source-dir", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)

    ready = commands.add_parser("preflight")
    ready.add_argument("--model", type=Path, required=True)
    ready.add_argument("--model-manifest", type=Path, required=True)
    ready.add_argument("--training-manifest", type=Path, required=True)
    ready.add_argument("--geo-adapter", type=Path, required=True)
    ready.add_argument("--evq-adapter", type=Path, required=True)
    ready.add_argument("--data-root", type=Path, required=True)
    ready.add_argument("--gpu-command", type=Path, required=True)
    ready.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run-arm")
    run.add_argument("--arm", choices=tuple(ARM_SUBSTRATE), required=True)
    run.add_argument("--model-name", type=Path, required=True)
    run.add_argument("--model-manifest", type=Path, required=True)
    run.add_argument("--training-data-manifest", type=Path, required=True)
    run.add_argument("--adapter-dir", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)

    summary = commands.add_parser("summarize")
    summary.add_argument("--base-native", type=Path, required=True)
    summary.add_argument("--native-lora", type=Path, required=True)
    summary.add_argument("--evq-lora", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "preflight":
        result = preflight(args)
    elif args.command == "run-arm":
        result = run_arm(args)
    else:
        result = summarize(args)
    print(json.dumps({key: result.get(key) for key in ("schema", "status", "gate")}, indent=2))


if __name__ == "__main__":
    main()
