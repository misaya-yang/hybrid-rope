#!/usr/bin/env python3
"""Audit frozen Qwen K32 packed-NLL targets without model or parquet access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.prepare_qwen_k32_natural_nll import (
    DOCUMENTS,
    GRID,
    TARGET_TOKENS,
    ids_hash,
    sha256_file,
)

STATUS = "QWEN_K32_PACKED_NATURAL_TARGET_BOUNDARY_SAFE_V1"


def audit(data_root: Path) -> dict:
    manifest_path = data_root / "manifest.json"
    rows_path = data_root / "rows.jsonl"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("model_evaluation_status") != "NOT_RUN"
        or manifest.get("model_outcomes_read") is not False
        or manifest.get("grid") != list(GRID)
        or manifest.get("natural_streams") != DOCUMENTS
        or manifest.get("target_tokens") != TARGET_TOKENS
        or manifest.get("file", {}).get("path") != rows_path.name
        or manifest.get("file", {}).get("sha256") != sha256_file(rows_path)
    ):
        raise ValueError("packed-natural manifest identity or lifecycle mismatch")
    eos = manifest.get("tokenizer_eos_token_id")
    if type(eos) is not int or eos < 0:
        raise ValueError("packed-natural manifest has no valid EOS token")

    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    expected = {
        (f"qwen-k32-natural-{index:03d}", length)
        for index in range(DOCUMENTS)
        for length in GRID
    }
    cells = {}
    for row in rows:
        key = row.get("sample_id"), row.get("length")
        ids = row.get("input_ids")
        if (
            key not in expected
            or key in cells
            or not isinstance(ids, list)
            or len(ids) != key[1]
            or row.get("target_start") != key[1] - TARGET_TOKENS
            or row.get("target_tokens") != TARGET_TOKENS
            or row.get("prompt_ids_sha256") != ids_hash(ids)
            or row.get("target_ids_sha256") != ids_hash(ids[-TARGET_TOKENS:])
        ):
            raise ValueError("invalid or duplicated packed-natural row")
        cells[key] = row
    if set(cells) != expected:
        raise ValueError("packed-natural paired grid is incomplete")

    receipts = []
    context_tokens = 2 * TARGET_TOKENS
    for index in range(DOCUMENTS):
        sample_id = f"qwen-k32-natural-{index:03d}"
        short = cells[sample_id, GRID[0]]
        long = cells[sample_id, GRID[1]]
        short_ids, long_ids = short["input_ids"], long["input_ids"]
        suffix_exact = short_ids == long_ids[-GRID[0]:]
        target_pair_exact = short["target_ids_sha256"] == long["target_ids_sha256"]
        tail = long_ids[-context_tokens:]
        eos_offsets = [offset for offset, token in enumerate(tail) if int(token) == eos]
        boundary_safe = suffix_exact and target_pair_exact and not eos_offsets
        receipts.append(
            {
                "sample_id": sample_id,
                "suffix_exact": suffix_exact,
                "target_pair_exact": target_pair_exact,
                "boundary_safe": boundary_safe,
                "eos_offsets_in_last_512": eos_offsets,
                "tail_512_sha256": ids_hash(tail),
                "target_ids_sha256": long["target_ids_sha256"],
            }
        )

    failures = [item["sample_id"] for item in receipts if not item["boundary_safe"]]
    return {
        "status": STATUS if not failures else "QWEN_K32_PACKED_NATURAL_TARGET_BOUNDARY_UNSAFE",
        "model_evaluation_status": "NOT_RUN",
        "streams": DOCUMENTS,
        "paired_lengths": list(GRID),
        "target_tokens": TARGET_TOKENS,
        "required_same_document_tail_tokens": context_tokens,
        "safe_streams": DOCUMENTS - len(failures),
        "unsafe_streams": failures,
        "data_manifest_sha256": sha256_file(manifest_path),
        "data_rows_sha256": sha256_file(rows_path),
        "audit_script_sha256": sha256_file(Path(__file__)),
        "per_stream": receipts,
        "claim": (
            "The 32K row is the exact suffix of its 64K pair and the final 256 targets plus "
            "their preceding 256-token local context contain no packed-document EOS boundary."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = audit(args.data_root)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError):
        report = {"status": "INVALID_OR_INCOMPLETE_BOUNDARY_AUDIT"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report["status"])
    return 0 if report["status"] == STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
