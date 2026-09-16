#!/usr/bin/env python3
"""Prepare portable source-order, unpadded RULER-13 transfer panels.

This is a CPU-only wrapper around the existing Plan-B source generator and
clean panel converter.  It deliberately does not inspect model generations or
scores.  Checkpoint-specific identity is recorded from public configuration
files, while artifact paths in the frozen manifests are relative to ``--out``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable, Sequence


RULER_REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
CONTRACT_REVISION = "strong-clean-transfer-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _parse_lengths(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--lengths must be comma-separated positive integers") from error
    if not lengths or len(lengths) != len(set(lengths)) or any(length <= 128 for length in lengths):
        raise ValueError("--lengths must contain unique context caps greater than 128")
    return lengths


def _scale_label(scale: float) -> str:
    return str(int(scale)) if float(scale).is_integer() else format(scale, ".12g").replace(".", "p")


def _validate_model_id(model_id: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", model_id):
        raise ValueError("--model-id must be a portable lowercase logical identifier")
    return model_id


def _portable_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact is outside --out and cannot be frozen portably: {path}") from error


def _checkpoint_identity(model: Path) -> dict:
    config_path = model / "config.json"
    tokenizer_path = model / "tokenizer.json"
    if not config_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError("--model must contain config.json and tokenizer.json")
    config = json.loads(config_path.read_text())
    if not config.get("model_type") or int(config.get("max_position_embeddings", 0)) <= 0:
        raise ValueError("checkpoint config lacks model_type or max_position_embeddings")
    tokenizer_files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = model / name
        if path.is_file():
            tokenizer_files[name] = _sha256(path)
    rope_parameters = config.get("rope_parameters") or {}
    attention_head_dim = int(config.get("head_dim") or config.get("hidden_size") // config.get("num_attention_heads"))
    partial_rotary_factor = float(config.get("partial_rotary_factor") or rope_parameters.get("partial_rotary_factor") or 1.0)
    return {
        "artifact_name": model.name,
        "config_sha256": _sha256(config_path),
        "model_type": config["model_type"],
        "architectures": list(config.get("architectures") or []),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "rope_theta": config.get("rope_theta") or rope_parameters.get("rope_theta"),
        "attention_head_dim": attention_head_dim,
        "partial_rotary_factor": partial_rotary_factor,
        "rotary_pairs": int(attention_head_dim * partial_rotary_factor) // 2,
        "native_length": int(config["max_position_embeddings"]),
        "rope_scaling": config.get("rope_scaling"),
        "tokenizer_files_sha256": tokenizer_files,
    }


def _detect_ruler_revision(data_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(data_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        revision = result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        revision = RULER_REVISION if RULER_REVISION in data_root.name else ""
    if revision != RULER_REVISION:
        raise ValueError(
            f"--data-root must be RULER revision {RULER_REVISION}; observed {revision or 'unknown'}"
        )
    return revision


def _default_planb_main(argv: Sequence[str]) -> int | None:
    from experiments.llama3_60dir_20260911.prepare_planb_panel import main
    return main(list(argv))


def _default_converter_main(argv: Sequence[str]) -> int | None:
    from experiments.fixed_rope_three_interfaces_20260913 import (
        prepare_tailspline_llama_32k_ruler200_clean as converter,
    )
    previous = sys.argv
    try:
        sys.argv = [str(Path(converter.__file__).name), *argv]
        return converter.main()
    finally:
        sys.argv = previous


def _sanitize_source_manifest(path: Path, *, model_id: str, model_name: str) -> None:
    """Remove host-specific model paths left by the historical preparer."""
    manifest = json.loads(path.read_text())
    manifest.pop("model", None)
    manifest["model_id"] = model_id
    manifest["model_artifact_name"] = model_name
    manifest["portable_paths"] = True
    _atomic_json(path, manifest)


def _validate_panel_rows(
    path: Path, *, length: int, rows_per_task: int, tasks: Sequence[str] = TASKS,
) -> list[dict]:
    rows = _read_jsonl(path)
    expected = len(tasks) * rows_per_task
    counts = Counter(row.get("task") for row in rows)
    if len(rows) != expected or counts != Counter({task: rows_per_task for task in tasks}):
        label = "Full-13" if tuple(tasks) == TASKS else f"requested {len(tasks)} tasks"
        raise ValueError(f"panel must contain {label} x {rows_per_task}; observed {len(rows)} rows")
    if len({row.get("row_id") for row in rows}) != expected:
        raise ValueError("panel row IDs are not unique")
    if len({row.get("prompt_sha256") for row in rows}) != expected:
        raise ValueError("panel prompt identities are not unique")
    for row in rows:
        prompt_ids = row.get("prompt_ids")
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError("panel row lacks nonempty prompt_ids")
        if (
            row.get("selection_mode") != "source-order"
            or row.get("selection_uses_model_outputs") is not False
            or int(row.get("irrelevant_padding_tokens", -1)) != 0
            or int(row.get("length_cap", -1)) != length
            or int(row.get("input_tokens", -1)) != len(prompt_ids)
            or int(row.get("actual_length", -1)) != len(prompt_ids)
            or len(prompt_ids) + int(row.get("max_new_tokens", 0)) > length
        ):
            raise ValueError("panel violates the source-order unpadded clean contract")
    return rows


def _freeze_panel_manifest(
    *, panel_dir: Path, out: Path, model_id: str, identity: dict,
    scale: float, length: int, rows_per_task: int, tasks: Sequence[str] = TASKS,
) -> dict:
    inputs_path = panel_dir / "inputs.jsonl"
    core_manifest_path = panel_dir / "manifest.json"
    rows = _validate_panel_rows(
        inputs_path, length=length, rows_per_task=rows_per_task, tasks=tasks,
    )
    core = json.loads(core_manifest_path.read_text())
    sources = {}
    for task, records in core.get("sources", {}).items():
        normalized = []
        for record in records:
            if record.get("artifact"):
                artifact = Path(str(record["artifact"]))
                if artifact.is_absolute():
                    raise ValueError("frozen source artifact must remain relative to --out")
                artifact_value = artifact.as_posix()
            elif record.get("path"):
                artifact_value = _portable_relative(Path(record["path"]), out)
            else:
                raise ValueError("source record lacks path or portable artifact")
            normalized.append({"artifact": artifact_value, "sha256": record["sha256"]})
        sources[task] = normalized
    contract = (
        f"strong_clean_{model_id}_s{_scale_label(scale)}_{length}_"
        f"ruler{len(tasks)}_{rows_per_task}_source_order_unpadded_v1"
    )
    manifest = {
        "status": "COMPLETE",
        "contract": contract,
        "contract_revision": CONTRACT_REVISION,
        "upstream_revision": RULER_REVISION,
        "model_id": model_id,
        "model_identity": identity,
        "scale": scale,
        "rows": len(rows),
        "rows_per_task": rows_per_task,
        "tasks": list(tasks),
        "length_cap": length,
        "selection_mode": "source-order",
        "selection_uses_model_outputs": False,
        "depth_balancing": False,
        "multi_evidence_profile_selection": False,
        "content_padding": False,
        "batching_scope": "batch=1 with exact unpadded prompt_ids; no runtime padding",
        "minimum_input_tokens": min(row["input_tokens"] for row in rows),
        "maximum_input_tokens": max(row["input_tokens"] for row in rows),
        "inputs_sha256": _sha256(inputs_path),
        "inputs_artifact": _portable_relative(inputs_path, out),
        "sources": sources,
        "scope": (
            f"RULER source-order sample over {len(tasks)} tasks, {rows_per_task}/task at the "
            f"{length}-token cap; no model-output selection and no content padding."
        ),
        "portable_paths": True,
    }
    if "LLAMA" in contract.upper() and "llama" not in model_id.lower():
        raise ValueError("non-Llama clean contract inherited a Llama label")
    _atomic_json(core_manifest_path, manifest)
    return manifest


def _validate_complete_root(path: Path, request: dict) -> dict | None:
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "COMPLETE":
        return None
    observed_request = {key: manifest.get(key) for key in request}
    if observed_request != request:
        raise ValueError("existing frozen clean-transfer manifest has a different request")
    root = path.parent
    for record in manifest.get("panels", {}).values():
        inputs = root / record["inputs"]
        panel_manifest = root / record["manifest"]
        if (
            not inputs.is_file()
            or not panel_manifest.is_file()
            or _sha256(inputs) != record["inputs_sha256"]
            or _sha256(panel_manifest) != record["manifest_sha256"]
        ):
            raise ValueError("frozen clean-transfer artifact is missing or has drifted")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--lengths", required=True, help="comma-separated token caps")
    parser.add_argument("--rows-per-task", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--qa-offset", "--qa-base-offset", dest="qa_offset", type=int, required=True)
    parser.add_argument("--tasks", default=",".join(TASKS),
                        help="comma-separated unique subset of RULER-13 tasks")
    return parser


def prepare(
    argv: Sequence[str] | None = None,
    *,
    planb_main: Callable[[Sequence[str]], int | None] | None = None,
    converter_main: Callable[[Sequence[str]], int | None] | None = None,
) -> dict:
    args = build_parser().parse_args(argv)
    model_id = _validate_model_id(args.model_id)
    lengths = _parse_lengths(args.lengths)
    selected_tasks = tuple(value.strip() for value in args.tasks.split(",") if value.strip())
    if not selected_tasks or len(selected_tasks) != len(set(selected_tasks)) or not set(selected_tasks).issubset(TASKS):
        raise ValueError("--tasks must be a unique non-empty subset of RULER-13")
    if not math.isfinite(args.scale) or args.scale <= 1.0:
        raise ValueError("--scale must be finite and greater than one")
    if args.rows_per_task <= 0 or args.seed < 0 or args.qa_offset < 0:
        raise ValueError("rows-per-task must be positive; seed and QA offset must be nonnegative")

    model = args.model.resolve()
    data_root = args.data_root.resolve()
    out = args.out.resolve()
    identity = _checkpoint_identity(model)
    _detect_ruler_revision(data_root)
    request = {
        "contract_revision": CONTRACT_REVISION,
        "model_id": model_id,
        "model_identity": identity,
        "scale": args.scale,
        "lengths": list(lengths),
        "rows_per_task": args.rows_per_task,
        "seed": args.seed,
        "qa_offset": args.qa_offset,
        "upstream_revision": RULER_REVISION,
        "tasks": list(selected_tasks),
    }
    existing = _validate_complete_root(out / "manifest.json", request)
    if existing is not None:
        print(json.dumps({"status": "SKIP_COMPLETE", "rows": existing["rows"]}, sort_keys=True))
        return existing

    out.mkdir(parents=True, exist_ok=True)
    planb = planb_main or _default_planb_main
    convert = converter_main or _default_converter_main
    caps = ",".join(str(length) for length in lengths)
    counts = ",".join(f"{length}:{args.rows_per_task}" for length in lengths)
    for task_index, task in enumerate(selected_tasks):
        source_part = out / "source_parts" / task
        result = planb([
            "--model", str(model),
            "--model-contract", "generic",
            "--upstream", str(data_root),
            "--out", str(source_part),
            "--stage", "H",
            "--contract", "planb",
            "--tasks", task,
            "--caps", caps,
            "--counts-by-cap", counts,
            "--selection-mode", "source-order",
            "--source-only",
            "--qa-base-offset", str(args.qa_offset),
            "--seed", str(args.seed + task_index * 100),
        ])
        if result not in (None, 0):
            raise RuntimeError(f"Plan-B source preparation failed for {task}: {result}")
        _sanitize_source_manifest(
            source_part / "manifest.json", model_id=model_id, model_name=model.name,
        )

    panels = {}
    for length in lengths:
        panel_dir = out / "panels" / str(length)
        result = convert([
            "--source-parts", str(out / "source_parts"),
            "--model", str(model),
            "--out", str(panel_dir),
            "--length", str(length),
            "--rows-per-task", str(args.rows_per_task),
            "--tasks", ",".join(selected_tasks),
        ])
        if result not in (None, 0):
            raise RuntimeError(f"clean converter failed for {length}: {result}")
        panel_manifest = _freeze_panel_manifest(
            panel_dir=panel_dir, out=out, model_id=model_id, identity=identity,
            scale=args.scale, length=length, rows_per_task=args.rows_per_task,
            tasks=selected_tasks,
        )
        panels[str(length)] = {
            "inputs": _portable_relative(panel_dir / "inputs.jsonl", out),
            "manifest": _portable_relative(panel_dir / "manifest.json", out),
            "rows": panel_manifest["rows"],
            "inputs_sha256": panel_manifest["inputs_sha256"],
            "manifest_sha256": _sha256(panel_dir / "manifest.json"),
        }

    manifest = {
        "status": "COMPLETE",
        **request,
        "rows": len(selected_tasks) * args.rows_per_task * len(lengths),
        "selection_mode": "source-order",
        "selection_uses_model_outputs": False,
        "content_padding": False,
        "panels": panels,
        "portable_paths": True,
    }
    _atomic_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def main() -> int:
    prepare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
