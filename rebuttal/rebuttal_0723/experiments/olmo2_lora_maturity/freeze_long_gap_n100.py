#!/usr/bin/env python3
"""Freeze the fresh all-long-gap 8K n=100 evaluation and its provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tarfile
from pathlib import Path
from typing import Any


PARENT_PREFIX = "instruct_native_evq_s20260725_v2"
REPLICATION_PREFIX = "instruct_evq_seed20260726_replication_v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def extract_parent_member(
    archive: tarfile.TarFile,
    member_name: str,
    output: Path,
) -> dict[str, Any]:
    extracted = archive.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(f"missing parent member: {member_name}")
    payload = extracted.read()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return {
        "relative_path": str(output),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "source_member": member_name,
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
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--parent-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    analysis_path = args.analysis_json.resolve()
    repo_root = args.repo_root.resolve()
    parent_archive = args.parent_archive.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    analysis = read_json(analysis_path)
    if (
        analysis.get("status")
        != "OLMO2_FRESH_ALL_LONG_GAP_N100_ANALYSIS_COMPLETE"
    ):
        raise RuntimeError("analysis JSON has the wrong status")
    if not analysis["data"]["all_rows_beyond_training_gap"]:
        raise RuntimeError("analysis did not establish all-long-gap data")
    expected = {
        "native": 0,
        "evq_seed_a": 49,
        "evq_seed_b": 48,
    }
    for arm, correct in expected.items():
        observed = analysis["arms"][arm]["summary"]["strict_exact"]
        if observed != correct:
            raise RuntimeError(
                f"frozen strict exact drift for {arm}: {observed}"
            )

    artifact_relatives = (
        "data/manifest.json",
        "data/candidates/niah_single_1/test.jsonl",
        "data/L8192/niah_single_1/test.jsonl",
        "instruct_long_gap_n100_ready_s20260727.json",
        "runs/native_seed20260725/results.json",
        "runs/native_seed20260725/examples.jsonl",
        "runs/evq_seed20260725/results.json",
        "runs/evq_seed20260725/examples.jsonl",
        "runs/evq_seed20260726/results.json",
        "runs/evq_seed20260726/examples.jsonl",
    )
    inventory = [
        copy_checked(
            base / relative,
            output / "artifacts" / relative,
            f"artifacts/{relative}",
        )
        for relative in artifact_relatives
    ]
    inventory.append(
        copy_checked(
            analysis_path,
            output / "analysis" / analysis_path.name,
            f"analysis/{analysis_path.name}",
        )
    )

    code_relatives = (
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
        "prepare_instruct_ruler_long_gap.py",
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
        "evaluate_instruct_ruler_screen.py",
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
        "preflight_instruct_long_gap_eval.py",
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
        "analyze_long_gap_n100.py",
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
        "freeze_long_gap_n100.py",
        "tests/test_olmo2_long_gap_screen.py",
        "tests/test_olmo2_long_gap_n100_analysis.py",
    )
    for relative in code_relatives:
        inventory.append(
            copy_checked(
                repo_root / relative,
                output / "code" / relative,
                f"code/{relative}",
            )
        )

    parent_members = (
        f"{PARENT_PREFIX}/formal_summary.json",
        f"{PARENT_PREFIX}/training_order.json",
        f"{PARENT_PREFIX}/value_query_overlap.json",
        f"{PARENT_PREFIX}/FREEZE_RECEIPT.json",
        f"{REPLICATION_PREFIX}/replication_summary.json",
        f"{REPLICATION_PREFIX}/training_order.json",
        f"{REPLICATION_PREFIX}/FREEZE_RECEIPT.json",
    )
    extracted_parent: dict[str, dict[str, Any]] = {}
    with tarfile.open(parent_archive, "r:gz") as archive:
        for member in parent_members:
            short_name = member.replace("/", "__", 1)
            destination = output / "parent_evidence" / short_name
            receipt = extract_parent_member(
                archive,
                member,
                destination,
            )
            receipt["relative_path"] = (
                f"parent_evidence/{short_name}"
            )
            inventory.append(receipt)
            extracted_parent[member] = read_json(destination)

    parent_summary = extracted_parent[f"{PARENT_PREFIX}/formal_summary.json"]
    if (
        parent_summary["arms"]["native"]["final_adapter_sha256"]
        != analysis["arms"]["native"]["adapter_sha256"]
    ):
        raise RuntimeError("Native adapter differs from parent freeze")
    if (
        parent_summary["arms"]["evq"]["final_adapter_sha256"]
        != analysis["arms"]["evq_seed_a"]["adapter_sha256"]
    ):
        raise RuntimeError("EVQ seed-A adapter differs from parent freeze")
    replication = extracted_parent[
        f"{REPLICATION_PREFIX}/replication_summary.json"
    ]
    replication_text = json.dumps(replication, sort_keys=True)
    if analysis["arms"]["evq_seed_b"]["adapter_sha256"] not in replication_text:
        raise RuntimeError("EVQ seed-B adapter is absent from parent freeze")

    optimizer = parent_summary["optimizer_protocol"]
    if optimizer.get("optimizer_state_saved") is not False:
        raise RuntimeError("unexpected optimizer-state provenance")
    parent_reference = {
        "status": "PARENT_ADAPTER_AND_TRAINING_PROVENANCE_REFERENCED",
        "parent_archive_sha256": sha256_file(parent_archive),
        "parent_archive_bytes": parent_archive.stat().st_size,
        "adapters": {
            arm: {
                "sha256": analysis["arms"][arm]["adapter_sha256"],
                "seed": analysis["arms"][arm]["adapter_seed"],
            }
            for arm in expected
        },
        "frequency_hashes": {
            arm: analysis["arms"][arm]["frequency_sha256_float32"]
            for arm in expected
        },
        "optimizer_protocol": optimizer,
        "optimizer_state_boundary": (
            "Adam moments were not persisted by the completed training runs. "
            "The exact optimizer configuration, seeds, code, data, and sampled "
            "training order are frozen; no optimizer-state claim is made."
        ),
        "training_order_members": [
            f"{PARENT_PREFIX}/training_order.json",
            f"{REPLICATION_PREFIX}/training_order.json",
        ],
        "value_overlap_member": (
            f"{PARENT_PREFIX}/value_query_overlap.json"
        ),
    }
    write_json(output / "parent_evidence_reference.json", parent_reference)
    inventory.append(
        {
            "relative_path": "parent_evidence_reference.json",
            "bytes": (
                output / "parent_evidence_reference.json"
            ).stat().st_size,
            "sha256": sha256_file(
                output / "parent_evidence_reference.json"
            ),
        }
    )

    inventory_payload = {
        "status": "IMMUTABLE_FRESH_ALL_LONG_GAP_N100_INVENTORY",
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
            "status": "OLMO2_FRESH_ALL_LONG_GAP_N100_FROZEN",
            "file_count": len(inventory),
            "inventory_sha256": inventory_sha,
            "analysis_sha256": sha256_file(analysis_path),
            "parent_archive_sha256": sha256_file(parent_archive),
            "strict_exact": expected,
        },
    )
    make_read_only(output)
    print(
        json.dumps(
            {
                "status": "OLMO2_FRESH_ALL_LONG_GAP_N100_FROZEN",
                "file_count": len(inventory),
                "inventory_sha256": inventory_sha,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
