#!/usr/bin/env python3
"""Freeze the CPU-only OLMo-2 n=100 source-gap diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def copy_checked(
    source: Path,
    destination: Path,
    relative_path: str,
) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = sha256_file(source)
    if sha256_file(destination) != digest:
        raise RuntimeError(f"copy digest drift: {relative_path}")
    return {
        "relative_path": relative_path,
        "bytes": destination.stat().st_size,
        "sha256": digest,
    }


def make_read_only(root: Path) -> None:
    for current_root, directories, files in os.walk(root):
        for name in files:
            os.chmod(
                Path(current_root) / name,
                stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH,
            )
        for name in directories:
            os.chmod(
                Path(current_root) / name,
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH,
            )
    os.chmod(
        root,
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--analysis-script", type=Path, required=True)
    parser.add_argument("--regression-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    analysis = read_json(args.analysis_json)
    if analysis.get("status") != "POST_HOC_CPU_ONLY_GAP_STRUCTURE_AUDIT":
        raise RuntimeError("analysis JSON has the wrong status")
    if not analysis.get("row_alignment", {}).get("all_equal"):
        raise RuntimeError("analysis JSON did not pass row alignment")
    if analysis["training_gap_support"]["maximum_tokens"] != 3933:
        raise RuntimeError("training-gap support drift")
    expected = {
        "native": (0, 0),
        "evq_seed_a": (48, 21),
        "evq_seed_b": (48, 19),
    }
    for arm, (within, beyond) in expected.items():
        result = analysis["arms"][arm]["by_gap_support"]
        actual = (
            result["within_observed_training_gap"]["strict_exact"],
            result["beyond_observed_training_gap"]["strict_exact"],
        )
        if actual != (within, beyond):
            raise RuntimeError(
                f"frozen gap result drift for {arm}: {actual}"
            )

    sources = (
        (
            args.analysis_json.resolve(),
            output / "diagnostics" / args.analysis_json.name,
            f"diagnostics/{args.analysis_json.name}",
        ),
        (
            args.analysis_script.resolve(),
            output / "code" / args.analysis_script.name,
            f"code/{args.analysis_script.name}",
        ),
        (
            args.regression_test.resolve(),
            output / "tests" / args.regression_test.name,
            f"tests/{args.regression_test.name}",
        ),
    )
    inventory = [
        copy_checked(source, destination, relative_path)
        for source, destination, relative_path in sources
    ]
    inventory_payload = {
        "status": "IMMUTABLE_N100_GAP_AUDIT_INVENTORY",
        "files": sorted(
            inventory,
            key=lambda item: item["relative_path"],
        ),
    }
    write_json(output / "inventory.json", inventory_payload)
    inventory_sha = sha256_file(output / "inventory.json")
    write_json(
        output / "FREEZE_RECEIPT.json",
        {
            "status": "OLMO2_N100_GAP_AUDIT_FROZEN",
            "file_count": len(inventory),
            "inventory_sha256": inventory_sha,
            "result_sha256": sha256_file(args.analysis_json),
            "script_sha256": sha256_file(args.analysis_script),
            "test_sha256": sha256_file(args.regression_test),
        },
    )
    make_read_only(output)
    print(
        json.dumps(
            {
                "status": "OLMO2_N100_GAP_AUDIT_FROZEN",
                "file_count": len(inventory),
                "inventory_sha256": inventory_sha,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
