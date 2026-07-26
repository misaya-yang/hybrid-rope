#!/usr/bin/env python3
"""Merge prepared per-task RULER manifests without regenerating examples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .prepare_data import atomic_json, sha256_file
from .prepare_instruct_ruler_transfer import DATA_STATUS, TASK_CONFIGS


MERGED_STATUS = DATA_STATUS
LENGTHS = (4_096, 8_192, 16_384)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fwe-root", type=Path, required=True)
    parser.add_argument("--qa2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def source_root(
    task: str,
    length: int,
    *,
    data_root: Path,
    fwe_root: Path,
    qa2_root: Path,
) -> Path:
    if task == "fwe":
        return fwe_root
    if task != "qa_2":
        return data_root / task
    suffix = "_fixed" if length == 4_096 else ""
    return qa2_root / f"qa_2_L{length}{suffix}"


def load_source(
    root: Path,
    *,
    task: str,
    length: int,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != DATA_STATUS
        or task not in manifest.get("tasks", [])
        or length not in {
            int(value) for value in manifest.get("lengths", [])
        }
    ):
        raise RuntimeError(f"source manifest coverage drift: {root}")
    entry = dict(manifest["cells"][task][str(length)])
    source_file = root / entry["relative_path"]
    if sha256_file(source_file) != entry["sha256"]:
        raise RuntimeError(f"source cell hash drift: {source_file}")
    return manifest, source_file, entry


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    data_root = args.data_root.resolve()
    fwe_root = args.fwe_root.resolve()
    qa2_root = args.qa2_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    tokenizer_sha256 = sha256_file(checkpoint / "tokenizer.json")
    output.mkdir(parents=True)

    cells: dict[str, dict[str, Any]] = {}
    task_configs: dict[str, Any] = {}
    source_manifests: dict[str, Any] = {}
    ruler_commit = None
    seed = None
    samples_per_cell = None
    for task in TASK_CONFIGS:
        cells[task] = {}
        for length in LENGTHS:
            root = source_root(
                task,
                length,
                data_root=data_root,
                fwe_root=fwe_root,
                qa2_root=qa2_root,
            )
            manifest, source_file, source_entry = load_source(
                root, task=task, length=length
            )
            if manifest["tokenizer_sha256"] != tokenizer_sha256:
                raise RuntimeError(f"tokenizer hash drift: {root}")
            for name, observed in (
                ("ruler_commit", manifest["ruler_commit"]),
                ("seed", int(manifest["seed"])),
                ("samples_per_cell", int(manifest["samples_per_cell"])),
            ):
                current = {
                    "ruler_commit": ruler_commit,
                    "seed": seed,
                    "samples_per_cell": samples_per_cell,
                }[name]
                if current is not None and current != observed:
                    raise RuntimeError(f"source {name} drift: {root}")
                if name == "ruler_commit":
                    ruler_commit = observed
                elif name == "seed":
                    seed = observed
                else:
                    samples_per_cell = observed
            task_configs[task] = manifest["task_configs"][task]
            source_key = str(root)
            source_manifests.setdefault(
                source_key,
                {
                    "path": str(root / "manifest.json"),
                    "sha256": sha256_file(root / "manifest.json"),
                },
            )

            target = (
                output / "cells" / task / f"L{length}" / "test.jsonl"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(
                os.path.relpath(source_file, start=target.parent)
            )
            if sha256_file(target) != source_entry["sha256"]:
                raise RuntimeError(f"merged symlink hash drift: {target}")
            source_entry["relative_path"] = str(target.relative_to(output))
            cells[task][str(length)] = source_entry

    result = {
        "format_version": 1,
        "status": MERGED_STATUS,
        "kind": "manifest_only_merged_ruler_view",
        "checkpoint": str(checkpoint),
        "tokenizer_sha256": tokenizer_sha256,
        "ruler_commit": ruler_commit,
        "seed": seed,
        "samples_per_cell": samples_per_cell,
        "tasks": list(TASK_CONFIGS),
        "lengths": list(LENGTHS),
        "cells": cells,
        "task_configs": task_configs,
        "source_manifests": source_manifests,
        "examples_regenerated": False,
        "cell_files_are_relative_symlinks": True,
    }
    atomic_json(output / "manifest.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
