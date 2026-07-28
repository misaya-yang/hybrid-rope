#!/usr/bin/env python3
"""Evaluate frozen Native/EVQ adapters on held-out RULER tasks."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from scripts.lib.rope.official_yarn import repo_fixed_ramp_inv_freq
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
    configure_cuda,
    configure_ruler_flash_attention,
    greedy_generate,
    row_sha256,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    endpoint_evq_inv_freq,
    endpoint_geo_inv_freq,
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    install_adaptation,
    load_model,
)
from .evaluate_instruct_ruler_screen import validate_adapter_metadata
from .evaluate_instruct_ruler_screen import apply_screen_frequency
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_data import (
    atomic_json,
    sha256_file,
)
from .prepare_instruct_ruler_transfer import (
    DATA_STATUS,
    TASK_CONFIGS,
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
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evq_attention_restoration import (
    install_qkv_lora,
)


RESULT_STATUS = "OLMO2_INSTRUCT_RULER_TRANSFER_COMPLETE"
RETENTION_READY_STATUS = "OLMO2_4K_RETENTION_READY_V1"
RUN_MANIFEST_STATUS = "OLMO2_4K_RETENTION_EVAL_RUN_V1"
LENGTH = 4_096


def bound_code_sha256() -> dict[str, str]:
    evaluator = Path(__file__).resolve()
    maturity_root = evaluator.parent
    experiments_root = maturity_root.parent
    paths = {
        "evaluator": evaluator,
        "screen_frequency": (
            maturity_root / "evaluate_instruct_ruler_screen.py"
        ),
        "hybrid_import_dependency": (
            maturity_root / "olmo2_exact_method.py"
        ),
        "data_contract": (
            maturity_root / "prepare_instruct_ruler_transfer.py"
        ),
        "checkpoint_contract": maturity_root / "train_4k_stage_a.py",
        "frequency_application": maturity_root / "train_screen.py",
        "greedy_generation_and_row_identity": (
            experiments_root / "olmo2_1b_evq" / "evaluate_ruler.py"
        ),
        "evq_contract": experiments_root / "olmo2_1b_evq" / "contract.py",
        "lora_conversion": experiments_root / "olmo2_lora_conversion.py",
        "adapter_loader": (
            experiments_root / "olmo2_lora_ood_factorial.py"
        ),
    }
    return {
        name: sha256_file(path)
        for name, path in sorted(paths.items())
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--experiment-ready-receipt", type=Path)
    parser.add_argument(
        "--experiment-role",
        choices=(
            "candidate",
            "immediate_parent",
            "pre_query_gap_parent",
        ),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frequency",
        choices=(
            "native",
            "evq",
            "official_yarn",
            "evq_official_yarn",
            "repo_fixed_ramp",
            "evq_repo_fixed_ramp",
            "hybrid_evq_low4",
            "hybrid_evq_low8",
            "hybrid_evq_low12",
            "hybrid_evq_low16",
            "hybrid_evq_low24",
            "hybrid_evq_low32",
            "hybrid_evq_low40",
        ),
        required=True,
    )
    parser.add_argument("--yarn-factor", type=float, default=4.0)
    parser.add_argument(
        "--yarn-original-max-position-embeddings",
        type=int,
        default=4_096,
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--adaptation",
        choices=(
            "qkvo_answer",
            "qk_answer",
            "qkv_attention_restoration",
        ),
        default="qkvo_answer",
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--lengths", type=int, nargs="+")
    parser.add_argument("--limit-per-cell", type=int, default=20)
    return parser.parse_args()


def official_string_match_all(
    prediction: str,
    references: list[str],
) -> float:
    if not references:
        raise RuntimeError("RULER row has no references")
    lowered = prediction.lower()
    return sum(
        float(str(reference).lower() in lowered)
        for reference in references
    ) / len(references)


def official_string_match_part(
    prediction: str,
    references: list[str],
) -> float:
    if not references:
        raise RuntimeError("RULER row has no references")
    lowered = prediction.lower()
    return max(
        float(str(reference).lower() in lowered)
        for reference in references
    )


def official_task_score(
    prediction: str,
    references: list[str],
    metric: str,
) -> float:
    if metric == "string_match_all":
        return official_string_match_all(prediction, references)
    if metric == "string_match_part":
        return official_string_match_part(prediction, references)
    raise RuntimeError(f"unsupported official RULER metric: {metric}")


def official_yarn_config(
    checkpoint: Path,
    *,
    factor: float,
    original_max_position_embeddings: int,
) -> Any:
    if factor <= 1.0:
        raise ValueError("YaRN factor must be greater than one")
    if original_max_position_embeddings <= 0:
        raise ValueError(
            "YaRN original max position embeddings must be positive"
        )
    config = AutoConfig.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    if config.model_type != "olmo2":
        raise RuntimeError("official YaRN requires OLMo-2 config")
    if int(config.max_position_embeddings) != int(
        original_max_position_embeddings
    ):
        raise RuntimeError("native max position embedding drift")
    config.rope_scaling = {
        "rope_type": "yarn",
        "factor": float(factor),
        "original_max_position_embeddings": int(
            original_max_position_embeddings
        ),
    }
    config.max_position_embeddings = int(
        round(original_max_position_embeddings * factor)
    )
    return config


def verify_official_yarn(model: Any, config: Any) -> dict[str, Any]:
    expected_inv_freq, expected_attention_scaling = (
        ROPE_INIT_FUNCTIONS["yarn"](config, torch.device("cpu"))
    )
    rotary = model.model.rotary_emb
    realized_inv_freq = rotary.inv_freq.detach().cpu().float()
    expected_inv_freq = expected_inv_freq.detach().cpu().float()
    if not torch.equal(realized_inv_freq, expected_inv_freq):
        raise RuntimeError(
            "official Transformers YaRN realized inv_freq drift"
        )
    realized_attention_scaling = float(rotary.attention_scaling)
    if not math.isclose(
        realized_attention_scaling,
        float(expected_attention_scaling),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError(
            "official Transformers YaRN attention scaling drift"
        )
    return {
        "active_frequency": "official_transformers_yarn",
        "rope_scaling": dict(config.rope_scaling),
        "max_position_embeddings": int(
            config.max_position_embeddings
        ),
        "active_sha256_float32": tensor_sha256(realized_inv_freq),
        "independent_expected_sha256_float32": tensor_sha256(
            expected_inv_freq
        ),
        "attention_scaling": realized_attention_scaling,
        "independent_expected_attention_scaling": float(
            expected_attention_scaling
        ),
        "transformers_rope_initializer": (
            "transformers.modeling_rope_utils."
            "ROPE_INIT_FUNCTIONS['yarn']"
        ),
    }


def apply_evq_official_yarn(
    model: Any,
    config: Any,
    official_yarn_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Apply Transformers' realized YaRN index scaler to the EVQ substrate."""
    rotary = model.model.rotary_emb
    official_native_yarn = rotary.inv_freq.detach().cpu().float().clone()
    native = endpoint_geo_inv_freq()
    evq = endpoint_evq_inv_freq()
    if (
        official_native_yarn.shape != native.shape
        or native.shape != evq.shape
        or not torch.isfinite(official_native_yarn).all()
        or not torch.isfinite(evq).all()
    ):
        raise RuntimeError("EVQ plus official YaRN frequency shape drift")
    per_index_scaler = official_native_yarn / native
    if (
        not torch.isfinite(per_index_scaler).all()
        or torch.any(per_index_scaler <= 0)
        or torch.any(per_index_scaler > 1)
    ):
        raise RuntimeError("official YaRN per-index scaler escaped (0, 1]")
    active = evq * per_index_scaler
    with torch.no_grad():
        rotary.inv_freq.copy_(
            active.to(
                device=rotary.inv_freq.device,
                dtype=rotary.inv_freq.dtype,
            )
        )
    if hasattr(rotary, "original_inv_freq"):
        rotary.original_inv_freq = rotary.inv_freq.detach().clone()
    if not math.isclose(
        float(rotary.attention_scaling),
        float(official_yarn_receipt["attention_scaling"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("EVQ plus official YaRN attention scaling drift")
    return {
        **official_yarn_receipt,
        "active_frequency": (
            "evq_endpoint_cosh_plus_official_transformers_yarn"
        ),
        "active_sha256_float32": tensor_sha256(active),
        "training_substrate": "evq_endpoint_cosh",
        "training_substrate_sha256_float32": tensor_sha256(evq),
        "native_substrate_sha256_float32": tensor_sha256(native),
        "official_native_yarn_sha256_float32": tensor_sha256(
            official_native_yarn
        ),
        "per_index_scaler_sha256_float32": tensor_sha256(
            per_index_scaler
        ),
        "composition": (
            "EVQ inverse frequencies multiplied by the exact per-index "
            "scaler realized by the installed Transformers YaRN "
            "initializer; the official YaRN attention scaling is retained"
        ),
        "rope_scaling": dict(config.rope_scaling),
    }


def apply_repo_fixed_ramp(
    model: Any,
    *,
    substrate: str,
    factor: float,
) -> dict[str, Any]:
    """Apply the repository's legacy fixed-index smooth-ramp scaler."""
    if substrate == "native":
        base_inv_freq = endpoint_geo_inv_freq()
    elif substrate == "evq":
        base_inv_freq = endpoint_evq_inv_freq()
    else:
        raise ValueError(f"unsupported fixed-ramp substrate: {substrate}")
    active, attention_scaling, metadata = repo_fixed_ramp_inv_freq(
        base_inv_freq,
        scale=float(factor),
    )
    active = active.float()
    if (
        active.shape != base_inv_freq.shape
        or not torch.isfinite(active).all()
        or torch.any(active <= 0)
        or float(attention_scaling) != 1.0
    ):
        raise RuntimeError("repository fixed-ramp frequency drift")
    rotary = model.model.rotary_emb
    with torch.no_grad():
        rotary.inv_freq.copy_(
            active.to(
                device=rotary.inv_freq.device,
                dtype=rotary.inv_freq.dtype,
            )
        )
    if hasattr(rotary, "original_inv_freq"):
        rotary.original_inv_freq = rotary.inv_freq.detach().clone()
    rotary.attention_scaling = 1.0
    scaler = active / base_inv_freq
    return {
        **metadata,
        "active_frequency": (
            f"{substrate}_plus_repo_fixed_index_smooth_ramp"
        ),
        "training_substrate": substrate,
        "training_substrate_sha256_float32": tensor_sha256(
            base_inv_freq
        ),
        "active_sha256_float32": tensor_sha256(active),
        "per_index_scaler_sha256_float32": tensor_sha256(scaler),
        "attention_scaling": 1.0,
        "canonical_implementation": (
            "scripts.lib.rope.official_yarn."
            "repo_fixed_ramp_inv_freq"
        ),
        "composition": (
            "training substrate inverse frequencies transformed by the "
            "repository-defined 20%-90% fixed-index smoothstep ramp"
        ),
    }


def validate_adapter_training_substrate(
    metadata: dict[str, Any],
    *,
    checkpoint_digest: str,
    frequency_name: str,
    frequency_sha256: str,
    adaptation: str,
    rank: int,
    alpha: float,
) -> None:
    expected = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": frequency_name,
        "frequency_sha256_float32": frequency_sha256,
        "adaptation": adaptation,
        "rank": int(rank),
        "alpha": float(alpha),
        "training_sequence_length": 4_096,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise RuntimeError(
                f"adapter training-substrate drift for {name}: "
                f"{metadata.get(name)!r} != {value!r}"
            )


def _validate_data(
    *,
    root: Path,
    checkpoint: Path,
    requested_tasks: tuple[str, ...] | None,
    requested_lengths: tuple[int, ...] | None,
    limit_per_cell: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != DATA_STATUS:
        raise RuntimeError("RULER transfer data status drift")
    if Path(manifest["checkpoint"]).resolve() != checkpoint.resolve():
        raise RuntimeError("RULER transfer checkpoint drift")
    tokenizer_digest = sha256_file(checkpoint / "tokenizer.json")
    if manifest.get("tokenizer_sha256") != tokenizer_digest:
        raise RuntimeError("RULER transfer tokenizer hash drift")
    manifest_tasks = tuple(str(task) for task in manifest["tasks"])
    manifest_lengths = tuple(
        int(length) for length in manifest["lengths"]
    )
    tasks = requested_tasks or manifest_tasks
    lengths = requested_lengths or manifest_lengths
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or any(task not in manifest_tasks for task in tasks)
    ):
        raise RuntimeError("invalid requested task set")
    if (
        not lengths
        or tuple(sorted(set(lengths))) != lengths
        or any(length not in manifest_lengths for length in lengths)
    ):
        raise RuntimeError("invalid requested length set")
    if not 1 <= limit_per_cell <= int(manifest["samples_per_cell"]):
        raise RuntimeError("limit-per-cell exceeds prepared rows")

    selected: list[dict[str, Any]] = []
    cells: dict[str, dict[str, Any]] = {}
    for task in tasks:
        cells[task] = {}
        for length in lengths:
            entry = manifest["cells"][task][str(length)]
            path = root / entry["relative_path"]
            digest = sha256_file(path)
            if digest != entry["sha256"]:
                raise RuntimeError(
                    f"RULER data hash drift for {task}/L{length}"
                )
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(rows) != int(entry["rows"]):
                raise RuntimeError(
                    f"RULER row-count drift for {task}/L{length}"
                )
            for local_index, row in enumerate(rows[:limit_per_cell]):
                row["_task"] = task
                row["_nominal_length"] = length
                row["_local_index"] = local_index
                row["_generation_tokens"] = int(
                    entry["generation_tokens"]
                )
                selected.append(row)
            cells[task][str(length)] = {
                "path": str(path),
                "sha256": digest,
                "rows": len(rows),
                "selected_rows": limit_per_cell,
            }
    return (
        {
            "manifest_sha256": sha256_file(manifest_path),
            "ruler_commit": manifest["ruler_commit"],
            "tokenizer_sha256": tokenizer_digest,
            "seed": int(manifest["seed"]),
            "cells": cells,
        },
        selected,
    )


def _load_completed(
    path: Path,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    completed: dict[tuple[str, int, int], dict[str, Any]] = {}
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (
            str(row["task"]),
            int(row["nominal_length"]),
            int(row["local_index"]),
        )
        if key in completed:
            raise RuntimeError(f"duplicate completed row: {key}")
        completed[key] = row
    return completed


def _validate_or_create_run_manifest(
    path: Path,
    expected: dict[str, Any],
) -> None:
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError(
                "retention output belongs to a different adapter, "
                "dataset, or protocol"
            )
        return
    atomic_json(path, expected)


def main() -> None:
    args = parse_args()
    tasks = (
        tuple(str(task) for task in args.tasks)
        if args.tasks is not None
        else None
    )
    lengths = (
        tuple(int(length) for length in args.lengths)
        if args.lengths is not None
        else None
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    examples_path = output / "examples.jsonl"
    run_manifest_path = output / "run_manifest.json"
    checkpoint = args.checkpoint.resolve()
    ready_receipt = args.ready_receipt.resolve()
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, ready_receipt
    )
    data_receipt, rows = _validate_data(
        root=args.data_root.resolve(),
        checkpoint=checkpoint,
        requested_tasks=tasks,
        requested_lengths=lengths,
        limit_per_cell=int(args.limit_per_cell),
    )
    selected_tasks = tuple(
        dict.fromkeys(str(row["_task"]) for row in rows)
    )
    selected_lengths = tuple(
        sorted(set(int(row["_nominal_length"]) for row in rows))
    )
    adapter_path = (
        None if args.adapter is None else args.adapter.resolve()
    )
    experiment_ready_path = (
        None
        if args.experiment_ready_receipt is None
        else args.experiment_ready_receipt.resolve()
    )
    experiment_ready_sha256 = (
        None
        if experiment_ready_path is None
        else sha256_file(experiment_ready_path)
    )
    if experiment_ready_path is not None:
        experiment_ready = json.loads(
            experiment_ready_path.read_text(encoding="utf-8")
        )
        role = str(args.experiment_role)
        if (
            experiment_ready.get("status") != RETENTION_READY_STATUS
            or role not in experiment_ready["registered_outputs"]
            or output
            != Path(
                experiment_ready["registered_outputs"][role]
            ).resolve()
            or experiment_ready["evaluator"]["sha256"]
            != sha256_file(Path(__file__).resolve())
            or experiment_ready.get("evaluator_bound_code_sha256")
            != bound_code_sha256()
            or experiment_ready["inputs"]["checkpoint"][
                "composite_sha256"
            ]
            != checkpoint_digest
            or experiment_ready["inputs"]["checkpoint"][
                "ready_receipt_sha256"
            ]
            != sha256_file(ready_receipt)
            or experiment_ready["inputs"]["data"]["manifest_sha256"]
            != data_receipt["manifest_sha256"]
            or {
                key: value["sha256"]
                for key, value in experiment_ready["inputs"]["data"][
                    "cells"
                ].items()
            }
            != {
                task: data_receipt["cells"][task][str(LENGTH)][
                    "sha256"
                ]
                for task in selected_tasks
            }
            or adapter_path is None
            or sha256_file(adapter_path)
            != experiment_ready["registered_adapters"][role]["sha256"]
        ):
            raise RuntimeError("retention experiment READY drift")
    elif args.experiment_role is not None:
        raise RuntimeError("retention role requires an experiment READY")

    run_manifest = {
        "status": RUN_MANIFEST_STATUS,
        "checkpoint_sha256": checkpoint_digest,
        "ready_receipt_sha256": sha256_file(ready_receipt),
        "experiment_ready_receipt_sha256": experiment_ready_sha256,
        "experiment_role": args.experiment_role,
        "bound_code_sha256": bound_code_sha256(),
        "data_manifest_sha256": data_receipt["manifest_sha256"],
        "data_cells": {
            task: {
                str(length): data_receipt["cells"][task][str(length)][
                    "sha256"
                ]
                for length in selected_lengths
            }
            for task in selected_tasks
        },
        "frequency": str(args.frequency),
        "yarn_factor": (
            float(args.yarn_factor)
            if "yarn" in str(args.frequency)
            or "fixed_ramp" in str(args.frequency)
            else None
        ),
        "yarn_original_max_position_embeddings": (
            int(args.yarn_original_max_position_embeddings)
            if "official_yarn" in str(args.frequency)
            else None
        ),
        "adaptation": (
            str(args.adaptation) if adapter_path is not None else None
        ),
        "adapter_sha256": (
            None if adapter_path is None else sha256_file(adapter_path)
        ),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "tasks": list(selected_tasks),
        "lengths": list(selected_lengths),
        "limit_per_cell": int(args.limit_per_cell),
        "greedy": True,
        "decode_skip_special_tokens": True,
        "decode_cleanup": False,
    }
    if examples_path.exists() != run_manifest_path.exists():
        raise RuntimeError(
            "retention examples and run manifest must be resumed together"
        )
    _validate_or_create_run_manifest(run_manifest_path, run_manifest)
    completed = _load_completed(examples_path)
    expected_rows = {
        (
            str(row["_task"]),
            int(row["_nominal_length"]),
            int(row["_local_index"]),
        ): row
        for row in rows
    }
    for key, result in completed.items():
        if key not in expected_rows:
            raise RuntimeError(f"completed retention row is outside run: {key}")
        source = expected_rows[key]
        if (
            result.get("row_sha256")
            != row_sha256(
                {
                    name: value
                    for name, value in source.items()
                    if not name.startswith("_")
                }
            )
            or result.get("references")
            != [str(value) for value in source["outputs"]]
            or result.get("official_metric")
            != str(TASK_CONFIGS[str(source["_task"])]["official_metric"])
        ):
            raise RuntimeError(f"completed retention row identity drift: {key}")

    configure_cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    model_config = None
    if args.frequency in {"official_yarn", "evq_official_yarn"}:
        model_config = official_yarn_config(
            checkpoint,
            factor=float(args.yarn_factor),
            original_max_position_embeddings=int(
                args.yarn_original_max_position_embeddings
            ),
        )
    model = load_model(checkpoint, config=model_config)
    if args.frequency in {"official_yarn", "evq_official_yarn"}:
        official_yarn = verify_official_yarn(model, model_config)
        frequency = (
            official_yarn
            if args.frequency == "official_yarn"
            else apply_evq_official_yarn(
                model, model_config, official_yarn
            )
        )
    elif args.frequency in {
        "repo_fixed_ramp",
        "evq_repo_fixed_ramp",
    }:
        frequency = apply_repo_fixed_ramp(
            model,
            substrate=(
                "native"
                if args.frequency == "repo_fixed_ramp"
                else "evq"
            ),
            factor=float(args.yarn_factor),
        )
    elif str(args.frequency).startswith("hybrid_"):
        frequency = apply_screen_frequency(
            model,
            str(args.frequency),
            custom_evq_head_indices=(),
        )
    else:
        frequency = apply_frequency(model, str(args.frequency))
    adapter_receipt = None
    if args.adapter is not None:
        if args.adaptation == "qkv_attention_restoration":
            install_qkv_lora(
                model,
                rank=int(args.rank),
                alpha=float(args.alpha),
            )
            readout = None
        else:
            readout = install_adaptation(
                model,
                str(args.adaptation),
                rank=int(args.rank),
                alpha=float(args.alpha),
            )
        if readout is not None:
            raise RuntimeError("RULER transfer does not admit a readout")
        adapter_path = args.adapter.resolve()
        adapter_metadata = load_adapter(adapter_path, model, None)
        if args.frequency in {
            "official_yarn",
            "repo_fixed_ramp",
        }:
            validate_adapter_training_substrate(
                adapter_metadata,
                checkpoint_digest=checkpoint_digest,
                frequency_name="native",
                frequency_sha256=tensor_sha256(
                    endpoint_geo_inv_freq()
                ),
                adaptation=str(args.adaptation),
                rank=int(args.rank),
                alpha=float(args.alpha),
            )
        elif args.frequency in {
            "evq_official_yarn",
            "evq_repo_fixed_ramp",
        }:
            validate_adapter_training_substrate(
                adapter_metadata,
                checkpoint_digest=checkpoint_digest,
                frequency_name="evq",
                frequency_sha256=tensor_sha256(
                    endpoint_evq_inv_freq()
                ),
                adaptation=str(args.adaptation),
                rank=int(args.rank),
                alpha=float(args.alpha),
            )
        elif str(args.frequency).startswith("hybrid_"):
            expected_metadata = {
                "base_checkpoint_sha256": checkpoint_digest,
                "frequency": "native",
                "frequency_sha256_float32": frequency[
                    "geo_sha256_float32"
                ],
                "adaptation": str(args.adaptation),
                "rank": int(args.rank),
                "alpha": float(args.alpha),
                "training_sequence_length": 4_096,
            }
            for name, expected in expected_metadata.items():
                if adapter_metadata.get(name) != expected:
                    raise RuntimeError(
                        f"Native adapter metadata drift for hybrid {name}"
                    )
        else:
            validate_adapter_metadata(
                adapter_metadata,
                checkpoint_digest=checkpoint_digest,
                frequency=frequency,
                frequency_name=str(args.frequency),
                rank=int(args.rank),
                alpha=float(args.alpha),
                adaptation=str(args.adaptation),
            )
        adapter_receipt = {
            "path": str(adapter_path),
            "sha256": sha256_file(adapter_path),
            "metadata": adapter_metadata,
        }

    configure_ruler_flash_attention(model)
    model.config.use_cache = True
    model.eval()
    model.to("cuda")
    expected = len(rows)
    torch.cuda.reset_peak_memory_stats()

    with examples_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            task = str(row["_task"])
            length = int(row["_nominal_length"])
            local_index = int(row["_local_index"])
            generation_tokens = int(row["_generation_tokens"])
            key = (task, length, local_index)
            if key in completed:
                continue
            chat_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["input"]}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            prefix_ids = tokenizer(
                row.get("answer_prefix", ""),
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids
            input_ids = torch.cat((chat_ids, prefix_ids), dim=1).to(
                "cuda"
            )
            if input_ids.shape[1] + generation_tokens > length:
                raise RuntimeError(
                    f"{task} exceeds L{length}: "
                    f"{input_ids.shape[1]}+{generation_tokens}"
                )
            started = time.perf_counter()
            output_ids = greedy_generate(
                model,
                input_ids,
                max_new_tokens=generation_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            generated_token_ids = [
                int(value)
                for value in output_ids[0].detach().cpu().tolist()
            ]
            prediction = tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            references = [str(value) for value in row["outputs"]]
            metric = str(TASK_CONFIGS[task]["official_metric"])
            official_score = official_task_score(
                prediction,
                references,
                metric,
            )
            reference_recall = official_string_match_all(
                prediction, references
            )
            result = {
                "task": task,
                "task_role": TASK_CONFIGS[task]["role"],
                "nominal_length": length,
                "local_index": local_index,
                "source_row_index": int(row["index"]),
                "row_sha256": row_sha256(
                    {
                        name: value
                        for name, value in row.items()
                        if not name.startswith("_")
                    }
                ),
                "input_tokens": int(input_ids.numel()),
                "maximum_generation_tokens": generation_tokens,
                "generated_tokens": int(output_ids.numel()),
                "generated_token_ids": generated_token_ids,
                "prediction": prediction,
                "references": references,
                "official_metric": metric,
                "official_task_score": official_score,
                "reference_recall": reference_recall,
                "all_references_found": float(
                    reference_recall == 1.0
                ),
                "elapsed_seconds": elapsed,
            }
            handle.write(
                json.dumps(result, ensure_ascii=False, sort_keys=True)
                + "\n"
            )
            handle.flush()
            completed[key] = result
            print(
                f"{len(completed)}/{expected} {task} L={length} "
                f"official={official_score:.3f} "
                f"all={result['all_references_found']:.0f} "
                f"seconds={elapsed:.2f}",
                flush=True,
            )

    relevant = [
        completed[
            (
                str(row["_task"]),
                int(row["_nominal_length"]),
                int(row["_local_index"]),
            )
        ]
        for row in rows
    ]
    if len(relevant) != expected:
        raise RuntimeError("RULER transfer result-count drift")
    cells: dict[str, dict[str, Any]] = {}
    for task in selected_tasks:
        cells[task] = {}
        for length in selected_lengths:
            cell_rows = [
                row
                for row in relevant
                if row["task"] == task
                and int(row["nominal_length"]) == length
            ]
            if not cell_rows:
                continue
            cells[task][str(length)] = {
                "examples": len(cell_rows),
                "official_metric": str(
                    TASK_CONFIGS[task]["official_metric"]
                ),
                "official_task_score": sum(
                    float(row["official_task_score"])
                    for row in cell_rows
                )
                / len(cell_rows),
                "reference_recall": sum(
                    float(row["reference_recall"])
                    for row in cell_rows
                )
                / len(cell_rows),
                "all_references_found": sum(
                    float(row["all_references_found"])
                    for row in cell_rows
                )
                / len(cell_rows),
                "mean_elapsed_seconds": sum(
                    float(row["elapsed_seconds"])
                    for row in cell_rows
                )
                / len(cell_rows),
            }
    macro = sum(
        float(cell["official_task_score"])
        for task_cells in cells.values()
        for cell in task_cells.values()
    ) / sum(len(task_cells) for task_cells in cells.values())
    if not math.isfinite(macro):
        raise RuntimeError("non-finite RULER transfer result")

    receipt = {
        "status": RESULT_STATUS,
        "metric_boundary": (
            "Official RULER autoregressive task-specific scoring: "
            "string_match_all for synthetic retrieval/counting tasks and "
            "string_match_part for QA. Reference recall and "
            "all_references_found are auxiliary and do not penalize "
            "unrelated extra text."
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "bound_code_sha256": bound_code_sha256(),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "ready_receipt_sha256": sha256_file(ready_receipt),
        "experiment_ready_receipt_sha256": experiment_ready_sha256,
        "experiment_role": args.experiment_role,
        "frequency": frequency,
        "adapter": adapter_receipt,
        "data": data_receipt,
        "protocol": {
            "tasks": list(selected_tasks),
            "lengths": list(selected_lengths),
            "limit_per_cell": int(args.limit_per_cell),
            "greedy": True,
            "decode_skip_special_tokens": True,
            "decode_cleanup": False,
            "training_length": (
                4_096 if adapter_receipt is not None else None
            ),
            "adaptation": (
                str(args.adaptation)
                if adapter_receipt is not None
                else None
            ),
            "yarn_factor": (
                float(args.yarn_factor)
                if "yarn" in str(args.frequency)
                or "fixed_ramp" in str(args.frequency)
                else None
            ),
            "yarn_original_max_position_embeddings": (
                int(args.yarn_original_max_position_embeddings)
                if "official_yarn" in str(args.frequency)
                else None
            ),
            "attention": "flash_only_custom_kv_cache",
            "precision": "bf16_weights_and_autocast",
            "task_generation_tokens": {
                task: int(TASK_CONFIGS[task]["tokens_to_generate"])
                for task in selected_tasks
            },
        },
        "runtime": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            "mem_efficient_sdp_enabled": (
                torch.backends.cuda.mem_efficient_sdp_enabled()
            ),
            "peak_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
            "peak_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved()
            ),
        },
        "results": {
            "cells": cells,
            "macro_official_task_score": macro,
            "examples": len(relevant),
            "examples_sha256": sha256_file(examples_path),
        },
    }
    atomic_json(output / "results.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output / "results.json"),
                "cells": cells,
                "macro_official_task_score": macro,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
