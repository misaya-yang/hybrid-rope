#!/usr/bin/env python3
"""Small official-RULER smoke for target-free OLMo integration."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
    configure_cuda,
    configure_ruler_flash_attention,
    greedy_generate,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_transfer import (
    _validate_data,
    official_string_match_all,
    official_task_score,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_instruct_ruler_transfer import (
    TASK_CONFIGS,
    chat_input_ids,
)
from scripts.analysis.export_uniqueness_budgeted_tables import (
    build_default_tables,
    causal_distance_measure,
    conditional_pair_uniqueness,
    float32_sha256,
)
from scripts.eval.target_free_formal_eval import (
    METHODS,
    load_method_model,
    set_target_aware_factor,
    sha256_file,
    validate_checkpoint,
)
from scripts.lib.rope.length_conditioned_budgeted import matched_attention_scaling


STATUS = "TARGET_FREE_RULER_SMOKE_COMPLETE"
DEFAULT_TASKS = ("niah_single_1", "niah_multikey_2", "niah_multikey_3", "vt")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".incomplete",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--lengths", type=int, nargs="+", default=(8192, 16384))
    parser.add_argument("--limit-per-cell", type=int, default=20)
    parser.add_argument("--native-context-length", type=int, default=4096)
    parser.add_argument("--expected-weight-sha256")
    parser.add_argument("--expected-active-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def generic_weight_sha256(checkpoint: Path, expected: str | None) -> str:
    weight = checkpoint / "model.safetensors"
    if not weight.is_file():
        raise FileNotFoundError(weight)
    digest = sha256_file(weight)
    if expected is None or digest != str(expected):
        raise RuntimeError("cross-model checkpoint requires its exact expected weight SHA-256")
    return digest


def generic_yarn_config(config: Any, native_context_length: int) -> Any:
    native_rope = dict(getattr(config, "rope_parameters", None) or {})
    native_theta = getattr(config, "rope_theta", None)
    if native_theta is None:
        native_theta = native_rope.get("rope_theta")
    if native_theta is None:
        raise RuntimeError("cross-model Native RoPE theta is unavailable")
    native_theta = float(native_theta)
    parameters = {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": int(native_context_length),
        "rope_theta": native_theta,
    }
    config.rope_scaling = dict(parameters)
    config.rope_parameters = dict(parameters)
    config.max_position_embeddings = int(native_context_length * 4)
    return config


def load_cross_model_method(
    checkpoint: Path,
    method: str,
    *,
    native_context_length: int,
    expected_active_sha256: str | None,
) -> tuple[Any, dict[str, Any]]:
    if method not in {"native", "official_yarn", "session_binary_s4"}:
        raise RuntimeError("cross-model smoke supports only Native, YaRN-4, and binary s4")
    config = AutoConfig.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    if int(config.max_position_embeddings) != int(native_context_length):
        raise RuntimeError("cross-model Native context length drift")
    if method == "official_yarn":
        config = generic_yarn_config(config, native_context_length)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
    )
    rotary = model.model.rotary_emb
    native = rotary.inv_freq.detach().cpu().float().numpy()
    native_hash = float32_sha256(native)
    if method == "official_yarn":
        expected_inv, expected_scaling = ROPE_INIT_FUNCTIONS["yarn"](
            config,
            torch.device("cpu"),
        )
        realized = rotary.inv_freq.detach().cpu().float()
        if not torch.equal(realized, expected_inv.detach().cpu().float()):
            raise RuntimeError("cross-model official YaRN frequency drift")
        if float(rotary.attention_scaling) != float(expected_scaling):
            raise RuntimeError("cross-model official YaRN scaling drift")
        receipt = {
            "method": "official_transformers_yarn_factor4",
            "model_type": str(config.model_type),
            "native_context_length": int(native_context_length),
            "active_sha256_float32": float32_sha256(realized.numpy()),
            "attention_scaling": float(rotary.attention_scaling),
            "rope_scaling": dict(config.rope_scaling),
        }
    elif method == "session_binary_s4":
        support, weight = causal_distance_measure(
            length=int(native_context_length),
            max_points=2048,
        )
        uniqueness = conditional_pair_uniqueness(
            native.astype(np.float64),
            support,
            weight,
        )
        span = float(uniqueness.max()) - float(uniqueness.min())
        normalized = (
            np.zeros_like(uniqueness)
            if span <= 0.0
            else (uniqueness - float(uniqueness.min())) / span
        )
        movement = (1.0 - normalized) ** 2.0
        table = np.ascontiguousarray(
            native * (1.0 - movement) + (native / 4.0) * movement,
            dtype="<f4",
        )
        if not np.isfinite(table).all() or not (table > 0.0).all():
            raise RuntimeError("literal cross-model table is non-finite or non-positive")
        crossing_indices = np.flatnonzero(table[:-1] <= table[1:]).tolist()
        active_hash = float32_sha256(table)
        if expected_active_sha256 is None or active_hash != str(expected_active_sha256):
            raise RuntimeError(
                "literal cross-model table hash drift: "
                f"{active_hash} != {expected_active_sha256}"
            )
        with torch.no_grad():
            rotary.inv_freq.copy_(torch.from_numpy(table).to(rotary.inv_freq))
        if hasattr(rotary, "original_inv_freq"):
            rotary.original_inv_freq = rotary.inv_freq.detach().clone()
        rotary.attention_scaling = matched_attention_scaling(4.0)
        model.config.max_position_embeddings = int(native_context_length * 4)
        receipt = {
            "method": "cross_model_native_or_frozen_s4",
            "model_type": str(config.model_type),
            "selection": "Native within L_native; frozen s4 beyond L_native",
            "requires_L_target": False,
            "native_context_length": int(native_context_length),
            "native_sha256_float32": native_hash,
            "active_sha256_float32": active_hash,
            "attention_scaling": float(rotary.attention_scaling),
            "exponent": 2.0,
            "factor": 4.0,
            "gain_coefficient": 0.1,
            "support_points": 2048,
            "order_crossing_indices": crossing_indices,
            "order_crossings_retained": True,
            "transfer_policy": "literal frozen formula; no sorting or projection",
        }
    else:
        model.config.max_position_embeddings = int(native_context_length * 4)
        receipt = {
            "method": "native",
            "model_type": str(config.model_type),
            "native_context_length": int(native_context_length),
            "active_sha256_float32": native_hash,
            "attention_scaling": float(getattr(rotary, "attention_scaling", 1.0)),
        }
    configure_ruler_flash_attention(model)
    model.config.use_cache = True
    model.eval().to("cuda")
    receipt["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    return model, receipt


def main() -> int:
    args = parse_args()
    configure_cuda()
    checkpoint = args.checkpoint.resolve()
    config_probe = AutoConfig.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    is_olmo = str(config_probe.model_type) == "olmo2"
    checkpoint_sha = (
        validate_checkpoint(checkpoint)
        if is_olmo
        else generic_weight_sha256(checkpoint, args.expected_weight_sha256)
    )
    native_context_length = int(args.native_context_length)
    tasks = tuple(str(value) for value in args.tasks)
    lengths = tuple(sorted(int(value) for value in args.lengths))
    allowed_lengths = {native_context_length * 2, native_context_length * 4}
    if not lengths or any(value not in allowed_lengths for value in lengths):
        raise ValueError(f"smoke lengths must be in {sorted(allowed_lengths)}")
    data_receipt, rows = _validate_data(
        root=args.data_root.resolve(),
        checkpoint=checkpoint,
        requested_tasks=tasks,
        requested_lengths=lengths,
        limit_per_cell=int(args.limit_per_cell),
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    examples_path = output / "examples.jsonl"
    run_manifest = {
        "status": "TARGET_FREE_RULER_SMOKE_FROZEN",
        "checkpoint_sha256": checkpoint_sha,
        "data_manifest_sha256": data_receipt["manifest_sha256"],
        "method": str(args.method),
        "tasks": list(tasks),
        "lengths": list(lengths),
        "limit_per_cell": int(args.limit_per_cell),
        "model_type": str(config_probe.model_type),
        "native_context_length": native_context_length,
        "expected_active_sha256": args.expected_active_sha256,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    run_manifest_path = output / "run_manifest.json"
    if run_manifest_path.is_file():
        if json.loads(run_manifest_path.read_text(encoding="utf-8")) != run_manifest:
            raise RuntimeError("output belongs to a different smoke run")
    elif examples_path.exists():
        raise RuntimeError("examples exist without a run manifest")
    else:
        atomic_json(run_manifest_path, run_manifest)

    model, method_receipt = (
        load_method_model(
            checkpoint,
            str(args.method),
            native_context_length=native_context_length,
        )
        if is_olmo
        else load_cross_model_method(
            checkpoint,
            str(args.method),
            native_context_length=native_context_length,
            expected_active_sha256=args.expected_active_sha256,
        )
    )
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    native_inv = (
        model.model.rotary_emb.inv_freq.detach().cpu().float().clone()
        if hasattr(model.model.rotary_emb, "inv_freq")
        else None
    )
    tables = (
        build_default_tables()
        if is_olmo and args.method in {"target_aware", "session_binary_s4"}
        else {}
    )
    completed: dict[tuple[str, int, int], dict[str, Any]] = {}
    if examples_path.is_file():
        for line in examples_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["task"]), int(row["nominal_length"]), int(row["local_index"]))
            if key in completed:
                raise RuntimeError(f"duplicate completed row: {key}")
            completed[key] = row
    with examples_path.open("a", encoding="utf-8") as handle:
        for ordinal, row in enumerate(rows, start=1):
            task = str(row["_task"])
            length = int(row["_nominal_length"])
            local_index = int(row["_local_index"])
            key = (task, length, local_index)
            if key in completed:
                continue
            active_receipt = None
            if is_olmo and args.method in {"target_aware", "session_binary_s4"}:
                assert native_inv is not None
                active_receipt = set_target_aware_factor(
                    model,
                    4 if args.method == "session_binary_s4" else length // native_context_length,
                    tables,
                    native_inv,
                )
            chat_ids = chat_input_ids(tokenizer.apply_chat_template(
                [{"role": "user", "content": row["input"]}],
                add_generation_prompt=True,
                return_tensors="pt",
            ))
            prefix_ids = tokenizer(
                row.get("answer_prefix", ""),
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids
            input_ids = torch.cat((chat_ids, prefix_ids), dim=1).to("cuda")
            generation_tokens = int(row["_generation_tokens"])
            if input_ids.shape[1] + generation_tokens > length:
                raise RuntimeError(f"{task} exceeds L{length}")
            started = time.perf_counter()
            output_ids = greedy_generate(
                model,
                input_ids,
                max_new_tokens=generation_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            prediction = tokenizer.decode(
                output_ids[0].detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            references = [str(value) for value in row["outputs"]]
            metric = str(TASK_CONFIGS[task]["official_metric"])
            score = official_task_score(prediction, references, metric)
            result = {
                "task": task,
                "nominal_length": length,
                "local_index": local_index,
                "prediction": prediction,
                "references": references,
                "official_metric": metric,
                "official_task_score": score,
                "reference_recall": official_string_match_all(prediction, references),
                "elapsed_seconds": time.perf_counter() - started,
                "active_profile": active_receipt,
            }
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed[key] = result
            if ordinal == 1 or ordinal % 20 == 0 or ordinal == len(rows):
                print(f"{ordinal}/{len(rows)} {task} L{length} score={score:.3f}", flush=True)

    cells: dict[str, Any] = {}
    for task in tasks:
        cells[task] = {}
        for length in lengths:
            selected = [
                row for (row_task, row_length, _), row in completed.items()
                if row_task == task and row_length == length
            ]
            if selected:
                cells[task][str(length)] = {
                    "rows": len(selected),
                    "official_task_score": float(
                        np.mean([row["official_task_score"] for row in selected])
                    ),
                }
    macro = float(np.mean([
        cell["official_task_score"]
        for task_cells in cells.values()
        for cell in task_cells.values()
    ]))
    if not math.isfinite(macro):
        raise RuntimeError("non-finite RULER macro")
    receipt = {
        "status": STATUS,
        "method": method_receipt,
        "protocol": run_manifest,
        "data": data_receipt,
        "results": {
            "cells": cells,
            "macro_official_task_score": macro,
            "examples": len(completed),
            "examples_sha256": sha256_file(examples_path),
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
    }
    atomic_json(output / "results.json", receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
