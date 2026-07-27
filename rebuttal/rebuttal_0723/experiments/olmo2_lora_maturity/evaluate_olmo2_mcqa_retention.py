#!/usr/bin/env python3
"""Matched MMLU/ARC retention diagnostic for OLMo-2 QKVO-LoRA adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoTokenizer

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    install_adaptation,
    load_model,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_screen import (
    validate_adapter_metadata,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_data import (
    atomic_json,
    sha256_file,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_4k_stage_a import (
    ready_checkpoint_digest,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_screen import (
    apply_frequency,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_ood_factorial import (
    load_adapter,
)


READY_STATUS = "OLMO2_MCQA_RETENTION_READY_V1"
RESULT_STATUS = "OLMO2_MCQA_RETENTION_COMPLETE_V1"
GATE_STATUS = "OLMO2_MCQA_RETENTION_GATE_V1"
TASKS = ("mmlu", "arc_challenge")
ROLES = (
    "evq_parent",
    "evq_candidate",
    "native_parent",
    "native_candidate",
)


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bound_code_hashes() -> dict[str, str]:
    here = Path(__file__).resolve()
    root = here.parents[4]
    paths = {
        "evaluator": here,
        "adapter_loader": (
            root
            / "rebuttal/rebuttal_0723/experiments/olmo2_lora_ood_factorial.py"
        ),
        "model_loader": (
            root
            / "rebuttal/rebuttal_0723/experiments/olmo2_lora_conversion.py"
        ),
        "frequency": (
            root
            / "rebuttal/rebuttal_0723/experiments/"
            "olmo2_lora_maturity/train_screen.py"
        ),
    }
    return {
        name: sha256_file(path)
        for name, path in sorted(paths.items())
    }


def _parse_role_specs(values: Sequence[str]) -> dict[str, dict[str, Path | str]]:
    parsed: dict[str, dict[str, Path | str]] = {}
    for value in values:
        fields = value.split("=", 1)
        if len(fields) != 2 or fields[0] not in ROLES:
            raise ValueError(
                "--role-spec must be ROLE=FREQUENCY,ADAPTER,OUTPUT"
            )
        role, payload = fields
        pieces = payload.split(",", 2)
        if (
            len(pieces) != 3
            or pieces[0] not in {"native", "evq"}
            or not pieces[1]
            or not pieces[2]
        ):
            raise ValueError(
                "--role-spec must be ROLE=FREQUENCY,ADAPTER,OUTPUT"
            )
        if role in parsed:
            raise ValueError(f"duplicate role spec: {role}")
        parsed[role] = {
            "frequency": pieces[0],
            "adapter": Path(pieces[1]).resolve(),
            "output": Path(pieces[2]).resolve(),
        }
    if set(parsed) != set(ROLES):
        raise ValueError(
            f"role coverage mismatch: expected={list(ROLES)}, "
            f"observed={sorted(parsed)}"
        )
    return parsed


def _load_capability_suite(
    data_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "evq_cosh.seed42_capability_manifest.v2":
        raise RuntimeError("capability manifest schema drift")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("capability manifest has no files")
    rows: list[dict[str, Any]] = []
    for filename, record in sorted(files.items()):
        if Path(filename).name != filename or not filename.endswith(".jsonl"):
            raise RuntimeError(f"unsafe capability filename: {filename}")
        path = data_root / filename
        if (
            not path.is_file()
            or sha256_file(path) != record.get("sha256")
            or path.stat().st_size != int(record.get("size_bytes", -1))
        ):
            raise RuntimeError(f"capability file identity drift: {filename}")
        file_rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(file_rows) != int(record.get("row_count", -1)):
            raise RuntimeError(f"capability row-count drift: {filename}")
        rows.extend(file_rows)
    if len(rows) != int(manifest.get("row_count", -1)):
        raise RuntimeError("capability total row-count drift")
    return manifest, rows


def _answer_token_ids(tokenizer: Any, answer: str) -> list[int]:
    ids = tokenizer(
        answer,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    if not ids:
        raise RuntimeError("MCQA choice tokenization produced no tokens")
    return [int(token_id) for token_id in ids]


@torch.inference_mode()
def _score_answer(
    model: torch.nn.Module,
    *,
    prompt_ids: Sequence[int],
    answer_ids: Sequence[int],
    device: torch.device,
) -> dict[str, float | int]:
    full_ids = [
        *[int(token_id) for token_id in prompt_ids],
        *[int(token_id) for token_id in answer_ids],
    ]
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    answer_start = len(prompt_ids)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=True,
    ):
        hidden = model.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[:, answer_start - 1 : len(full_ids) - 1]
        labels = input_ids[:, answer_start:]
        logits = model.get_output_embeddings()(hidden).float()
    nll_sum = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="sum",
    )
    token_count = int(labels.numel())
    value = float(nll_sum.detach().double().cpu())
    return {
        "nll_sum": value,
        "answer_tokens": token_count,
        "mean_logprob": -value / token_count,
    }


def _score_mcqa_record(
    row: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
) -> dict[str, Any]:
    if row.get("metric") != "mcqa":
        raise RuntimeError("MCQA evaluator received a non-MCQA row")
    choices = [str(choice) for choice in row["choices"]]
    scores = [
        _score_answer(
            model,
            prompt_ids=[int(token_id) for token_id in row["prompt_ids"]],
            answer_ids=_answer_token_ids(tokenizer, choice),
            device=device,
        )
        for choice in choices
    ]
    predicted = max(
        range(len(scores)),
        key=lambda index: float(scores[index]["mean_logprob"]),
    )
    gold = int(row["answer_index"])
    wrong_best = max(
        float(scores[index]["mean_logprob"])
        for index in range(len(scores))
        if index != gold
    )
    return {
        "example_id": row["example_id"],
        "task": row["task"],
        "prompt_sha256": row["prompt_sha256"],
        "prompt_tokens": len(row["prompt_ids"]),
        "choices": choices,
        "answer_index": gold,
        "prediction_index": predicted,
        "metric_score": float(predicted == gold),
        "nll_sum": float(scores[gold]["nll_sum"]),
        "answer_tokens": int(scores[gold]["answer_tokens"]),
        "correct_minus_best_wrong_mean_logprob": (
            float(scores[gold]["mean_logprob"]) - wrong_best
        ),
        "choice_mean_logprobs": [
            float(score["mean_logprob"]) for score in scores
        ],
    }


def _validate_data(
    data_root: Path,
    checkpoint: Path,
    *,
    limit_per_task: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest, rows = _load_capability_suite(data_root)
    if not 1 <= limit_per_task <= 1_000:
        raise ValueError("limit-per-task must be in [1, 1000]")
    files = manifest.get("tokenizer", {}).get("files", {})
    for name, digest in files.items():
        path = checkpoint / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"tokenizer identity drift: {name}")
    selected: list[dict[str, Any]] = []
    task_receipts: dict[str, Any] = {}
    for task in TASKS:
        candidates = [
            row
            for row in rows
            if row.get("suite") == "mcqa"
            and row.get("task") == task
            and row.get("metric") == "mcqa"
        ]
        if len(candidates) < limit_per_task:
            raise RuntimeError(
                f"insufficient frozen {task} rows: "
                f"{len(candidates)} < {limit_per_task}"
            )
        candidates.sort(
            key=lambda row: (
                str(row["prompt_sha256"]),
                str(row["example_id"]),
            )
        )
        chosen = candidates[:limit_per_task]
        selected.extend(chosen)
        task_receipts[task] = {
            "available_rows": len(candidates),
            "selected_rows": len(chosen),
            "selection": (
                "lexicographically smallest prompt_sha256, then example_id"
            ),
            "selected_identity_sha256": _json_hash(
                [
                    {
                        "example_id": row["example_id"],
                        "prompt_sha256": row["prompt_sha256"],
                        "answer_index": row["answer_index"],
                    }
                    for row in chosen
                ]
            ),
        }
    return (
        {
            "root": str(data_root),
            "manifest_sha256": sha256_file(data_root / "manifest.json"),
            "row_count": int(manifest["row_count"]),
            "tasks": task_receipts,
            "selected_rows": len(selected),
        },
        selected,
    )


def _adapter_record(
    path: Path,
    *,
    checkpoint_digest: str,
    frequency: str,
    rank: int,
    alpha: float,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = dict(payload.get("metadata", {}))
    expected = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": frequency,
        "adaptation": "qkvo_answer",
        "rank": int(rank),
        "alpha": float(alpha),
        "training_sequence_length": 4_096,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"adapter metadata drift for {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "metadata": metadata,
    }


def run_preflight(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint.resolve()
    ready = args.checkpoint_ready_receipt.resolve()
    checkpoint_digest = ready_checkpoint_digest(checkpoint, ready)
    data, _ = _validate_data(
        args.data_root.resolve(),
        checkpoint,
        limit_per_task=int(args.limit_per_task),
    )
    specs = _parse_role_specs(args.role_spec)
    adapters: dict[str, Any] = {}
    outputs: dict[str, str] = {}
    for role, spec in specs.items():
        adapter = Path(spec["adapter"])
        output = Path(spec["output"])
        if output.exists():
            raise FileExistsError(output)
        adapters[role] = {
            **_adapter_record(
                adapter,
                checkpoint_digest=checkpoint_digest,
                frequency=str(spec["frequency"]),
                rank=int(args.rank),
                alpha=float(args.alpha),
            ),
            "frequency": str(spec["frequency"]),
        }
        outputs[role] = str(output)
    module = (
        "rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity."
        "evaluate_olmo2_mcqa_retention"
    )
    receipt_path = args.output.resolve()
    commands = {
        role: [
            sys.executable,
            "-m",
            module,
            "evaluate",
            "--ready-receipt",
            str(receipt_path),
            "--role",
            role,
        ]
        for role in ROLES
    }
    commands["gate"] = [
        sys.executable,
        "-m",
        module,
        "gate",
        "--ready-receipt",
        str(receipt_path),
        "--output",
        str(args.gate_output.resolve()),
    ]
    receipt = {
        "status": READY_STATUS,
        "purpose": (
            "matched general-capability retention on frozen MMLU and "
            "ARC-Challenge; not a long-context capability claim"
        ),
        "concern": (
            "verify that the 4K-only query-gap/EOS continuation does not "
            "cause broad general knowledge or reasoning degradation"
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "composite_sha256": checkpoint_digest,
            "ready_receipt": str(ready),
            "ready_receipt_sha256": sha256_file(ready),
        },
        "data": data,
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "limit_per_task": int(args.limit_per_task),
        "bound_code_sha256": _bound_code_hashes(),
        "adapters": adapters,
        "outputs": outputs,
        "gate_output": str(args.gate_output.resolve()),
        "commands": commands,
        "budget": {
            "arms": len(ROLES),
            "rows_per_arm": int(data["selected_rows"]),
            "total_rows": len(ROLES) * int(data["selected_rows"]),
            "no_training": True,
            "no_parameter_sweep": True,
        },
        "stop_condition": (
            "stop after all four matched arms and one fixed retention gate"
        ),
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def _read_ready(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != READY_STATUS
        or receipt.get("bound_code_sha256") != _bound_code_hashes()
    ):
        raise RuntimeError("MCQA READY receipt or bound code drift")
    return receipt


def _write_jsonl_atomic(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.incomplete"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_evaluate(args: argparse.Namespace) -> None:
    ready_path = args.ready_receipt.resolve()
    receipt = _read_ready(ready_path)
    role = str(args.role)
    if role not in ROLES:
        raise ValueError(f"invalid role: {role}")
    output = Path(receipt["outputs"][role])
    if output.exists():
        raise FileExistsError(output)
    checkpoint = Path(receipt["checkpoint"]["path"])
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint,
        Path(receipt["checkpoint"]["ready_receipt"]),
    )
    if checkpoint_digest != receipt["checkpoint"]["composite_sha256"]:
        raise RuntimeError("checkpoint digest drift")
    data_receipt, rows = _validate_data(
        Path(receipt["data"]["root"]),
        checkpoint,
        limit_per_task=int(receipt["limit_per_task"]),
    )
    if data_receipt != receipt["data"]:
        raise RuntimeError("MCQA data receipt drift")
    adapter_record = receipt["adapters"][role]
    adapter_path = Path(adapter_record["path"])
    if sha256_file(adapter_path) != adapter_record["sha256"]:
        raise RuntimeError("adapter SHA drift")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = load_model(checkpoint)
    frequency = apply_frequency(model, str(adapter_record["frequency"]))
    readout = install_adaptation(
        model,
        "qkvo_answer",
        rank=int(receipt["rank"]),
        alpha=float(receipt["alpha"]),
    )
    if readout is not None:
        raise RuntimeError("MCQA retention does not admit a readout")
    metadata = load_adapter(adapter_path, model, None)
    validate_adapter_metadata(
        metadata,
        checkpoint_digest=checkpoint_digest,
        frequency=frequency,
        frequency_name=str(adapter_record["frequency"]),
        rank=int(receipt["rank"]),
        alpha=float(receipt["alpha"]),
    )
    model.config.use_cache = False
    model.eval()
    model.to("cuda")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        scored = _score_mcqa_record(
            row,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        results.append(scored)
        print(
            f"{index}/{len(rows)} {row['task']} "
            f"correct={int(scored['metric_score'])}",
            flush=True,
        )
    by_task: dict[str, Any] = {}
    for task in TASKS:
        task_rows = [row for row in results if row["task"] == task]
        tokens = sum(int(row["answer_tokens"]) for row in task_rows)
        nll_sum = sum(float(row["nll_sum"]) for row in task_rows)
        by_task[task] = {
            "examples": len(task_rows),
            "accuracy": sum(
                float(row["metric_score"]) for row in task_rows
            )
            / len(task_rows),
            "gold_answer_mean_nll": nll_sum / tokens,
            "gold_answer_tokens": tokens,
        }
    summary = {
        "macro_accuracy": sum(
            float(by_task[task]["accuracy"]) for task in TASKS
        )
        / len(TASKS),
        "tasks": by_task,
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl_atomic(results, output / "examples.jsonl")
    result = {
        "status": RESULT_STATUS,
        "role": role,
        "metric_boundary": (
            "zero-shot choice-string mean-logprob accuracy on frozen "
            "MMLU and ARC-Challenge rows; retention diagnostic only"
        ),
        "ready_receipt": str(ready_path),
        "ready_receipt_sha256": sha256_file(ready_path),
        "checkpoint_sha256": checkpoint_digest,
        "adapter": adapter_record,
        "frequency": frequency,
        "data": data_receipt,
        "summary": summary,
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "examples": {
            "path": str(output / "examples.jsonl"),
            "sha256": sha256_file(output / "examples.jsonl"),
            "rows": len(results),
        },
    }
    atomic_json(output / "results.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _load_result(path: Path, role: str) -> dict[str, Any]:
    result_path = path / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != RESULT_STATUS or result.get("role") != role:
        raise RuntimeError(f"invalid MCQA result for {role}")
    return result


def run_gate(args: argparse.Namespace) -> None:
    ready_path = args.ready_receipt.resolve()
    receipt = _read_ready(ready_path)
    results = {
        role: _load_result(Path(receipt["outputs"][role]), role)
        for role in ROLES
    }
    comparisons: dict[str, Any] = {}
    for substrate in ("evq", "native"):
        parent = results[f"{substrate}_parent"]["summary"]
        candidate = results[f"{substrate}_candidate"]["summary"]
        tasks: dict[str, Any] = {}
        for task in TASKS:
            delta = (
                float(candidate["tasks"][task]["accuracy"])
                - float(parent["tasks"][task]["accuracy"])
            )
            tasks[task] = {
                "parent": parent["tasks"][task]["accuracy"],
                "candidate": candidate["tasks"][task]["accuracy"],
                "delta": delta,
                "passed": delta >= -0.10,
            }
        macro_delta = (
            float(candidate["macro_accuracy"])
            - float(parent["macro_accuracy"])
        )
        comparisons[substrate] = {
            "parent_macro_accuracy": parent["macro_accuracy"],
            "candidate_macro_accuracy": candidate["macro_accuracy"],
            "macro_delta": macro_delta,
            "tasks": tasks,
            "passed": (
                macro_delta >= -0.05
                and all(record["passed"] for record in tasks.values())
            ),
        }
    gate = {
        "status": (
            "PASS"
            if all(
                comparison["passed"]
                for comparison in comparisons.values()
            )
            else "STOP"
        ),
        "gate": GATE_STATUS,
        "scope": (
            "matched parent-to-final general-capability retention on "
            "frozen MMLU and ARC-Challenge choice-logprob diagnostics"
        ),
        "thresholds": {
            "macro_accuracy_delta_min": -0.05,
            "per_task_accuracy_delta_min": -0.10,
        },
        "comparisons": comparisons,
        "ready_receipt": str(ready_path),
        "ready_receipt_sha256": sha256_file(ready_path),
        "result_sha256": {
            role: sha256_file(Path(receipt["outputs"][role]) / "results.json")
            for role in ROLES
        },
    }
    atomic_json(args.output.resolve(), gate)
    print(json.dumps(gate, indent=2, sort_keys=True))
    if gate["status"] != "PASS":
        raise SystemExit(21)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--checkpoint", type=Path, required=True)
    preflight.add_argument(
        "--checkpoint-ready-receipt",
        type=Path,
        required=True,
    )
    preflight.add_argument("--data-root", type=Path, required=True)
    preflight.add_argument("--role-spec", action="append", required=True)
    preflight.add_argument("--limit-per-task", type=int, default=100)
    preflight.add_argument("--rank", type=int, default=64)
    preflight.add_argument("--alpha", type=float, default=128.0)
    preflight.add_argument("--gate-output", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--ready-receipt", type=Path, required=True)
    evaluate.add_argument("--role", choices=ROLES, required=True)

    gate = subparsers.add_parser("gate")
    gate.add_argument("--ready-receipt", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        run_preflight(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "gate":
        run_gate(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
