#!/usr/bin/env python3
"""Assemble the evaluator candidate manifest for the success-first tournament.

Turns the frozen portfolio (F1 morph grid + Native control) plus any
development-stage representative receipts (F2/F3/F4) into the generic candidate
manifest consumed by ``scripts/eval/eval_zero_training_tournament.py``.

CPU only.  Every emitted entry carries the identity fields the evaluator and the
selection rule need: table path, float32 hash, pair count, family, construction
label, calibrated degrees of freedom, whether construction read long-range
outcomes, chord displacement, and support factor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TIEBREAK_DOF = {"F1_PC_MORPH": 1, "F2_PC_RETENTION_PROJECT": 1,
                "F3_Z5_BEHAVIOUR": 5, "F4_SR_Z5": 6}
USES_LONG_RANGE = {"F1_PC_MORPH": False, "F2_PC_RETENTION_PROJECT": False,
                   "F3_Z5_BEHAVIOUR": True, "F4_SR_Z5": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-manifest", type=Path, required=True)
    parser.add_argument("--dev-receipt", type=Path, action="append", default=[],
                        help="development receipt whose representative joins the manifest")
    parser.add_argument("--families", nargs="+", default=None,
                        help="restrict to a subset of families (e.g. F1 only)")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def float32_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value, dtype="<f4")).tobytes()
    ).hexdigest()


def chord_rms(table_path: Path, native_table: np.ndarray) -> float:
    table = np.load(table_path, allow_pickle=False).astype(np.float64)
    displacement = np.log(table) - np.log(native_table.astype(np.float64))
    return float(np.sqrt(np.mean(np.square(displacement))))


def main() -> int:
    args = parse_args()
    portfolio = json.loads(args.portfolio_manifest.resolve().read_text(encoding="utf-8"))
    f1 = portfolio["families"]["F1_PC_MORPH"]
    wanted = set(args.families) if args.families else None

    native_entry = next(e for e in f1["tables"] if e["is_bitwise_native"])
    native_table = np.load(native_entry["path"], allow_pickle=False)
    pair_count = int(native_table.size)

    candidates: list[dict[str, Any]] = []

    def add(entry: dict[str, Any], family: str, *, is_native: bool, support: float,
            path: str, table_hash: str, dof: int, uses_long: bool) -> None:
        if wanted is not None and family not in wanted and not is_native:
            return
        candidates.append({
            "name": entry["name"],
            "family": family,
            "path": path,
            "table_sha256_float32": table_hash,
            "pair_count": pair_count,
            "is_bitwise_native": bool(is_native),
            "calibrated_dof": int(dof),
            "used_long_range_in_construction": bool(uses_long),
            "chord_displacement_rms": float(entry.get("chord_displacement_rms", 0.0)),
            "support_factor": float(support),
        })

    # Native control + F1 grid (always present from the frozen portfolio).
    for entry in f1["tables"]:
        add(
            entry, "F1_PC_MORPH",
            is_native=bool(entry["is_bitwise_native"]),
            support=1.0,
            path=entry["path"],
            table_hash=entry["inv_freq_float32_sha256"],
            dof=0 if entry["is_bitwise_native"] else TIEBREAK_DOF["F1_PC_MORPH"],
            uses_long=USES_LONG_RANGE["F1_PC_MORPH"],
        )

    # Development representatives (F2/F3/F4), when their receipts are supplied.
    for receipt_path in args.dev_receipt:
        receipt = json.loads(receipt_path.expanduser().resolve().read_text(encoding="utf-8"))
        representative = receipt.get("representative")
        if not representative or receipt.get("family_stop"):
            continue
        family = receipt["family"]
        table_path = Path(representative["path"]).resolve()
        table_hash = representative["table_sha256_float32"]
        support = float(receipt.get("report", {}).get("support_factor", 1.0))
        name = f"{family}_rep"
        candidates.append({
            "name": name,
            "family": family,
            "path": str(table_path),
            "table_sha256_float32": table_hash,
            "pair_count": pair_count,
            "is_bitwise_native": False,
            "calibrated_dof": int(TIEBREAK_DOF.get(family, 0)),
            "used_long_range_in_construction": bool(USES_LONG_RANGE.get(family, False)),
            "chord_displacement_rms": chord_rms(table_path, native_table),
            "support_factor": support,
        })

    names = [c["name"] for c in candidates]
    if len(set(names)) != len(names):
        raise RuntimeError(f"candidate names are not unique: {names}")
    payload = {
        "status": "SUCCESS_FIRST_CANDIDATES_ASSEMBLED_V1",
        "pair_count": pair_count,
        "portfolio_content_sha256": portfolio.get("content_sha256"),
        "candidates": candidates,
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"candidates": len(candidates), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
