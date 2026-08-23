#!/usr/bin/env python3
"""Formal real-context evaluation for the frozen target-free OLMo retrofit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
    configure_cuda,
    configure_ruler_flash_attention,
    greedy_generate,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import load_model
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_transfer import (
    official_yarn_config,
    verify_official_yarn,
)
from scripts.analysis.export_uniqueness_budgeted_tables import (
    build_default_tables,
    causal_distance_measure,
    conditional_pair_uniqueness,
    float32_sha256,
)
from scripts.lib.rope.length_conditioned_budgeted import (
    matched_attention_scaling,
    select_observed_session_factor,
)
from scripts.lib.rope.target_free import (
    ModelRoPEProfile,
    install_target_free_olmo2,
)
from scripts.eval.longbench_metrics import (
    TASK_METRIC_MAP,
    post_process_prediction,
    score_prediction,
)


STATUS = "TARGET_FREE_REAL_CONTEXT_EVALUATION_COMPLETE"
EXPECTED_CHECKPOINT_SHA256 = (
    "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
)
METHODS = (
    "native",
    "official_yarn",
    "target_aware",
    "session_adaptive",
    "session_binary_s4",
    "target_free_anchored",
)
BUCKET_TO_MULTIPLIER = {"retention": 1, "near": 2, "far": 4}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


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


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--multipliers", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--factor", type=float, default=4.0)
    parser.add_argument("--factors", type=float, nargs="+", default=(2.0, 4.0))
    parser.add_argument("--label")
    parser.add_argument("--limit-per-cell", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    multipliers = tuple(int(value) for value in args.multipliers)
    if not multipliers or any(value not in {1, 2, 4} for value in multipliers):
        raise ValueError("multipliers must be a nonempty subset of 1,2,4")
    if len(set(multipliers)) != len(multipliers):
        raise ValueError("multipliers must be unique")
    if int(args.limit_per_cell) < 0:
        raise ValueError("limit-per-cell cannot be negative")
    if args.method == "official_yarn" and float(args.factor) != 4.0:
        raise ValueError("formal one-deployment YaRN control is frozen at factor four")
    if args.method == "target_aware" and tuple(float(v) for v in args.factors) != (2.0, 4.0):
        raise ValueError("target-aware oracle is frozen to factors two and four")


def validate_checkpoint(checkpoint: Path) -> str:
    weight = checkpoint.resolve() / "model.safetensors"
    if not weight.is_file():
        raise FileNotFoundError(weight)
    digest = sha256_file(weight)
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("released OLMo checkpoint hash drift")
    return digest


def selected_longbench_rows(
    manifest: dict[str, Any],
    *,
    manifest_root: Path,
    tasks: set[str],
    multipliers: set[int],
    limit_per_cell: int,
) -> list[dict[str, Any]]:
    cells = manifest["longbench"]["cells"]
    rows: list[dict[str, Any]] = []
    for task in sorted(tasks - {"pg19"}):
        preferred = f"longbench_e:{task}"
        fallback = f"longbench:{task}"
        cell_name = preferred if preferred in cells else fallback
        if cell_name not in cells:
            raise RuntimeError(f"token manifest has no LongBench cell for {task}")
        cell_rows = load_jsonl(manifest_root / cells[cell_name]["rows_path"])
        for multiplier in sorted(multipliers):
            selected = [
                row
                for row in cell_rows
                if BUCKET_TO_MULTIPLIER[str(row["bucket"])] == multiplier
            ]
            if limit_per_cell:
                selected = selected[:limit_per_cell]
            if not selected:
                print(
                    f"SKIP unavailable natural cell: {task} x{multiplier}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            rows.extend(selected)
    return rows


def selected_pg19_rows(
    manifest: dict[str, Any],
    *,
    manifest_root: Path,
    multipliers: set[int],
    limit_per_cell: int,
) -> list[dict[str, Any]]:
    path = manifest_root / manifest["pg19"]["rows_path"]
    source = load_jsonl(path)
    rows = []
    for multiplier in sorted(multipliers):
        selected = [row for row in source if int(row["multiplier"]) == multiplier]
        if limit_per_cell:
            selected = selected[:limit_per_cell]
        if not selected:
            raise RuntimeError(f"no PG-19 rows for multiplier {multiplier}")
        rows.extend(selected)
    return rows


def observed_session_factor(
    row: dict[str, Any],
    native_length: int,
    *,
    supported_factors: tuple[int, ...] = (1, 2, 4),
) -> int:
    max_new_tokens = 0 if "nll_target_start" in row else int(row["generation_reserve"])
    return select_observed_session_factor(
        prefill_tokens=len(row["input_ids"]),
        max_new_tokens=max_new_tokens,
        native_context_length=native_length,
        supported_factors=supported_factors,
    )


def current_olmo_profile(model: Any, native_context_length: int) -> ModelRoPEProfile:
    native = model.model.rotary_emb.inv_freq.detach().cpu().float().numpy()
    support, weight = causal_distance_measure(length=native_context_length, max_points=2048)
    uniqueness = conditional_pair_uniqueness(native.astype(np.float64), support, weight)
    normalized = (uniqueness - uniqueness.min()) / (uniqueness.max() - uniqueness.min())
    movement = (1.0 - normalized) ** 2
    head_dim = int(model.config.hidden_size) // int(model.config.num_attention_heads)
    return ModelRoPEProfile.from_native(
        native,
        native_context_length=native_context_length,
        native_context_length_source="pinned released OLMo-2 config and owner",
        head_dim=head_dim,
        rotary_dim=head_dim,
        movement_coefficients=movement,
        native_rope_config={
            "model_type": "olmo2",
            "rope_theta": float(getattr(model.config, "rope_theta", 500000.0)),
            "rope_type": "default",
        },
        native_scaling_config={"attention_scaling": 1.0},
        gain_coefficient=0.1,
        gain_coefficient_source="frozen matched OLMo profile; not universal",
        pair_layout="half_split",
    )


def load_method_model(
    checkpoint: Path,
    method: str,
    *,
    native_context_length: int,
) -> tuple[Any, dict[str, Any]]:
    config = (
        official_yarn_config(
            checkpoint,
            factor=4.0,
            original_max_position_embeddings=native_context_length,
        )
        if method == "official_yarn"
        else None
    )
    model = load_model(checkpoint, config=config)
    configure_ruler_flash_attention(model)
    model.config.use_cache = True
    if method == "official_yarn":
        receipt = verify_official_yarn(model, config)
    elif method == "target_free_anchored":
        receipt = install_target_free_olmo2(
            model,
            current_olmo_profile(model, native_context_length),
        )
    else:
        receipt = {
            "method": method,
            "native_inv_freq_sha256": float32_sha256(
                model.model.rotary_emb.inv_freq.detach().cpu().float().numpy()
            ),
        }
    model.eval().to("cuda")
    return model, receipt


def set_target_aware_factor(
    model: Any,
    multiplier: int,
    tables: dict[float, np.ndarray],
    native: torch.Tensor,
) -> dict[str, Any]:
    rotary = model.model.rotary_emb
    if multiplier == 1:
        active = native
        scaling = 1.0
        identity = "native"
    else:
        active = torch.from_numpy(tables[float(multiplier)]).float()
        scaling = matched_attention_scaling(float(multiplier))
        identity = f"budgeted_s{multiplier}_p2"
    with torch.no_grad():
        rotary.inv_freq.copy_(active.to(rotary.inv_freq))
    if hasattr(rotary, "original_inv_freq"):
        rotary.original_inv_freq = rotary.inv_freq.detach().clone()
    rotary.attention_scaling = float(scaling)
    return {
        "identity": identity,
        "multiplier": multiplier,
        "table_sha256_float32": float32_sha256(active.cpu().numpy()),
        "attention_scaling": scaling,
    }


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.is_file():
        return set()
    result = set()
    for row in load_jsonl(path):
        result.add((str(row["family"]), str(row["row_sha256"]), int(row["multiplier"])))
    return result


def pg19_nll(model: Any, row: dict[str, Any]) -> tuple[float, int]:
    input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
    target_start = int(row["nll_target_start"])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
    selected_logits = logits[:, target_start - 1 : -1].float()
    labels = input_ids[:, target_start:]
    if selected_logits.shape[1] != labels.shape[1] or labels.shape[1] != int(row["nll_target_tokens"]):
        raise RuntimeError("PG-19 target/logit alignment drift")
    loss = F.cross_entropy(
        selected_logits.reshape(-1, selected_logits.shape[-1]),
        labels.reshape(-1),
        reduction="mean",
    )
    return float(loss), int(labels.numel())


def main() -> int:
    args = parse_args()
    validate_args(args)
    checkpoint = args.checkpoint.resolve()
    token_manifest_path = args.token_manifest.resolve()
    token_manifest = load_json(token_manifest_path)
    if token_manifest.get("tokenization_executed") is not True:
        raise RuntimeError("complete token manifest required")
    native_length = int(token_manifest["native_context_length"])
    if native_length != 4096:
        raise RuntimeError("current pinned OLMo checkpoint requires L_native=4096")
    tasks = {str(value) for value in args.tasks}
    multipliers = {int(value) for value in args.multipliers}
    rows = selected_longbench_rows(
        token_manifest,
        manifest_root=token_manifest_path.parent,
        tasks=tasks,
        multipliers=multipliers,
        limit_per_cell=int(args.limit_per_cell),
    )
    if "pg19" in tasks:
        rows.extend(
            selected_pg19_rows(
                token_manifest,
                manifest_root=token_manifest_path.parent,
                multipliers=multipliers,
                limit_per_cell=int(args.limit_per_cell),
            )
        )
    if args.preflight_only:
        counts: dict[str, int] = {}
        session_factor_matches = 0
        selected_session_factors: dict[str, int] = {}
        for row in rows:
            task = "pg19" if "nll_target_start" in row else str(row["task"])
            multiplier = int(
                row.get(
                    "multiplier",
                    BUCKET_TO_MULTIPLIER.get(str(row.get("bucket"))),
                )
            )
            key = f"{task}:x{multiplier}"
            counts[key] = counts.get(key, 0) + 1
            if args.method in {"session_adaptive", "session_binary_s4"}:
                supported = (1, 4) if args.method == "session_binary_s4" else (1, 2, 4)
                selected_factor = observed_session_factor(
                    row,
                    native_length,
                    supported_factors=supported,
                )
                expected_factor = (
                    1 if multiplier == 1 else 4
                    if args.method == "session_binary_s4"
                    else multiplier
                )
                if selected_factor != expected_factor:
                    raise RuntimeError(
                        f"observed session factor drift for {task}: "
                        f"selected x{selected_factor}, expected x{expected_factor}"
                    )
                session_factor_matches += 1
                factor_key = str(selected_factor)
                selected_session_factors[factor_key] = (
                    selected_session_factors.get(factor_key, 0) + 1
                )
        print(json.dumps({
            "status": "TARGET_FREE_FORMAL_PREFLIGHT_COMPLETE",
            "method": str(args.method),
            "token_manifest_sha256": sha256_file(token_manifest_path),
            "rows": len(rows),
            "cells": counts,
            "cuda_initialized": False,
            "checkpoint_loaded": False,
            "session_factor_matches": session_factor_matches,
            "selected_session_factors": selected_session_factors,
        }, indent=2, sort_keys=True))
        return 0
    configure_cuda()
    checkpoint_sha = validate_checkpoint(checkpoint)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    examples_path = output / "examples.jsonl"
    run_manifest = {
        "status": "TARGET_FREE_REAL_CONTEXT_RUN_FROZEN",
        "checkpoint_sha256": checkpoint_sha,
        "token_manifest_sha256": sha256_file(token_manifest_path),
        "method": str(args.method),
        "tasks": sorted(tasks),
        "multipliers": sorted(multipliers),
        "factor": float(args.factor),
        "factors": [float(value) for value in args.factors],
        "label": args.label,
        "limit_per_cell": int(args.limit_per_cell),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    run_manifest_path = output / "run_manifest.json"
    if run_manifest_path.is_file():
        if load_json(run_manifest_path) != run_manifest:
            raise RuntimeError("output belongs to a different formal run")
    elif examples_path.exists():
        raise RuntimeError("examples exist without a run manifest")
    else:
        atomic_json(run_manifest_path, run_manifest)

    model, method_receipt = load_method_model(
        checkpoint,
        str(args.method),
        native_context_length=native_length,
    )
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
    target_tables = (
        build_default_tables()
        if args.method in {"target_aware", "session_adaptive", "session_binary_s4"}
        else {}
    )
    completed = completed_keys(examples_path)
    eos_token_id = tokenizer.eos_token_id
    for ordinal, row in enumerate(rows, start=1):
        family = "pg19" if "nll_target_start" in row else "longbench"
        multiplier = int(row.get("multiplier", BUCKET_TO_MULTIPLIER.get(str(row.get("bucket")))))
        key = (family, str(row["row_sha256"]), multiplier)
        if key in completed:
            continue
        active_receipt = None
        if args.method in {"target_aware", "session_adaptive", "session_binary_s4"}:
            assert native_inv is not None
            active_multiplier = (
                observed_session_factor(
                    row,
                    native_length,
                    supported_factors=(1, 4) if args.method == "session_binary_s4" else (1, 2, 4),
                )
                if args.method in {"session_adaptive", "session_binary_s4"}
                else multiplier
            )
            active_receipt = set_target_aware_factor(
                model,
                active_multiplier,
                target_tables,
                native_inv,
            )
            if args.method in {"session_adaptive", "session_binary_s4"}:
                expected_multiplier = (
                    1 if multiplier == 1 else 4
                    if args.method == "session_binary_s4"
                    else multiplier
                )
                if active_multiplier != expected_multiplier:
                    raise RuntimeError("session policy disagrees with its frozen route")
                active_receipt.update({
                    "selection": (
                        "Native within L_native; frozen s4 beyond L_native"
                        if args.method == "session_binary_s4"
                        else "smallest profile covering prefill_tokens plus max_new_tokens"
                    ),
                    "requires_L_target": False,
                    "cache_policy": "profile fixed before prefill for the KV-cache lifetime",
                    "observed_prefill_tokens": len(row["input_ids"]),
                    "requested_max_new_tokens": (
                        0 if family == "pg19" else int(row["generation_reserve"])
                    ),
                })
        started = time.perf_counter()
        if family == "pg19":
            nll, target_tokens = pg19_nll(model, row)
            result = {
                "family": family,
                "task": "pg19",
                "row_sha256": str(row["row_sha256"]),
                "multiplier": multiplier,
                "nll": nll,
                "target_tokens": target_tokens,
            }
        else:
            input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
            generated = greedy_generate(
                model,
                input_ids,
                max_new_tokens=int(row["generation_reserve"]),
                eos_token_id=eos_token_id,
            )[0].detach().cpu().tolist()
            prediction = tokenizer.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            task = str(row["task"])
            metric = TASK_METRIC_MAP[task]
            processed = post_process_prediction(task, prediction)
            score = score_prediction(
                task,
                metric,
                processed,
                [str(value) for value in row["references"]],
                [str(value) for value in row.get("all_classes", [])],
            )
            result = {
                "family": family,
                "task": task,
                "row_sha256": str(row["row_sha256"]),
                "multiplier": multiplier,
                "metric": metric,
                "score": float(score),
                "prediction": prediction,
                "generated_token_ids": [int(value) for value in generated],
                "generated_tokens": len(generated),
            }
        result["elapsed_seconds"] = time.perf_counter() - started
        result["active_profile"] = active_receipt
        append_jsonl(examples_path, result)
        completed.add(key)
        if ordinal == 1 or ordinal % 20 == 0 or ordinal == len(rows):
            print(f"{ordinal}/{len(rows)} {family} x{multiplier}", flush=True)

    results = load_jsonl(examples_path)
    cells: dict[str, Any] = {}
    for task in sorted({str(row["task"]) for row in results}):
        cells[task] = {}
        for multiplier in sorted(multipliers):
            selected = [
                row for row in results
                if str(row["task"]) == task and int(row["multiplier"]) == multiplier
            ]
            if not selected:
                continue
            if task == "pg19":
                cells[task][str(multiplier)] = {
                    "rows": len(selected),
                    "mean_tail_nll": float(np.mean([row["nll"] for row in selected])),
                    "target_tokens": sum(int(row["target_tokens"]) for row in selected),
                }
            else:
                cells[task][str(multiplier)] = {
                    "rows": len(selected),
                    "official_metric": str(selected[0]["metric"]),
                    "mean_score": float(np.mean([row["score"] for row in selected])),
                }
    receipt = {
        "status": STATUS,
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "method": method_receipt,
        "protocol": run_manifest,
        "results": {
            "cells": cells,
            "examples": len(results),
            "examples_sha256": sha256_file(examples_path),
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
    }
    atomic_json(output / "results.json", receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
