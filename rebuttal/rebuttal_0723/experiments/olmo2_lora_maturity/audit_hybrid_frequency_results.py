#!/usr/bin/env python3
"""Audit whether recorded hybrid RULER runs used their declared frequencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def expected_frequency(frequency: dict[str, Any]) -> torch.Tensor:
    native = torch.tensor(frequency["geo"], dtype=torch.float32)
    evq = torch.tensor(frequency["evq"], dtype=torch.float32)
    if torch.equal(native, evq):
        raise RuntimeError("receipt Native and EVQ arrays unexpectedly match")

    if frequency.get("hybrid_axis") == "attention_head":
        native_heads = {
            int(value)
            for value in frequency["hybrid_native_head_indices"]
        }
        evq_heads = {
            int(value)
            for value in frequency["hybrid_evq_head_indices"]
        }
        head_count = int(frequency["hybrid_native_head_count"]) + int(
            frequency["hybrid_evq_head_count"]
        )
        if (
            native_heads & evq_heads
            or native_heads | evq_heads != set(range(head_count))
        ):
            raise RuntimeError("invalid head-hybrid partition in receipt")
        expected = native.repeat(head_count, 1)
        expected[list(sorted(evq_heads))] = evq
        return expected

    blend_weight = frequency.get("hybrid_log_frequency_evq_weight")
    if blend_weight is not None:
        weight = float(blend_weight)
        if not 0.0 < weight < 1.0:
            raise RuntimeError("invalid hybrid blend weight")
        return torch.exp(
            (1.0 - weight) * torch.log(native)
            + weight * torch.log(evq)
        )

    native_indices = [
        int(value)
        for value in frequency["hybrid_native_pair_indices"]
    ]
    evq_indices = [
        int(value) for value in frequency["hybrid_evq_pair_indices"]
    ]
    if (
        set(native_indices) & set(evq_indices)
        or set(native_indices) | set(evq_indices)
        != set(range(native.numel()))
    ):
        raise RuntimeError("invalid pair-hybrid partition in receipt")
    expected = evq.clone()
    expected[native_indices] = native[native_indices]
    return expected


def aliased_frequency(frequency: dict[str, Any]) -> torch.Tensor:
    """Reconstruct the output when the saved Native tensor aliases EVQ."""
    evq = torch.tensor(frequency["evq"], dtype=torch.float32)
    if frequency.get("hybrid_axis") == "attention_head":
        head_count = int(frequency["hybrid_native_head_count"]) + int(
            frequency["hybrid_evq_head_count"]
        )
        return evq.repeat(head_count, 1)
    blend_weight = frequency.get("hybrid_log_frequency_evq_weight")
    if blend_weight is not None:
        weight = float(blend_weight)
        return torch.exp(
            (1.0 - weight) * torch.log(evq)
            + weight * torch.log(evq)
        )
    return evq


def audit_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frequency = payload["frequency"]
    name = str(frequency["active_frequency"])
    if not name.startswith("hybrid_"):
        raise RuntimeError(f"not a hybrid receipt: {path}")
    expected = expected_frequency(frequency)
    expected_sha = tensor_sha256(expected)
    aliased_sha = tensor_sha256(aliased_frequency(frequency))
    recorded_sha = str(frequency["active_sha256_float32"])
    cells = payload["results"]["cells"]
    return {
        "relative_path": str(path),
        "result_sha256": sha256_file(path),
        "active_frequency": name,
        "recorded_active_sha256": recorded_sha,
        "expected_active_sha256": expected_sha,
        "expected_alias_failure_sha256": aliased_sha,
        "recorded_matches_declared_hybrid": recorded_sha == expected_sha,
        "recorded_matches_alias_failure": recorded_sha == aliased_sha,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.runs_root.resolve()
    paths = sorted(root.glob("instruct_hybrid*/results.json"))
    if not paths:
        raise RuntimeError("no hybrid result receipts found")
    rows = [audit_result(path) for path in paths]
    invalid = [
        row
        for row in rows
        if not row["recorded_matches_declared_hybrid"]
    ]
    aliased = [
        row for row in rows if row["recorded_matches_alias_failure"]
    ]
    receipt = {
        "status": "HYBRID_FREQUENCY_IDENTITY_AUDIT_COMPLETE",
        "claim_boundary": (
            "Method-identity audit only. Invalidated runs cannot establish "
            "the performance of any partial-pair, blended, or per-head "
            "Native/EVQ hybrid."
        ),
        "runs_root": str(root),
        "summary": {
            "receipts": len(rows),
            "declared_hybrid_identity_matches": len(rows) - len(invalid),
            "invalid_identity_receipts": len(invalid),
            "receipts_matching_buffer_alias_failure": len(aliased),
        },
        "rows": rows,
    }
    if len(invalid) != len(rows) or len(aliased) != len(rows):
        raise RuntimeError(
            "historical hybrid receipts do not share the expected aliasing "
            "failure; inspect individually before changing conclusions"
        )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output),
                "output_sha256": sha256_file(output),
                "summary": receipt["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
