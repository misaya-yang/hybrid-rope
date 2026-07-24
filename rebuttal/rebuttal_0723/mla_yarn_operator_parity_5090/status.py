#!/usr/bin/env python3
"""Compact artifact-only progress snapshot for slow polling."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rebuttal.rebuttal_0723.mla_yarn_operator_parity_5090.protocol import (
    FREQUENCY_PAIRS,
    GATE_SEED,
    SEEDS,
    SPEC,
    TRAINING_ARMS,
)


def _load_status(
    path: Path, accepted: tuple[str, ...]
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return value if value.get("status") in accepted else None


def _last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        buffer = b""
        while position > 0:
            position -= 1
            handle.seek(position)
            byte = handle.read(1)
            if byte == b"\n" and buffer:
                break
            buffer = byte + buffer
    try:
        return json.loads(buffer.decode().strip())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _run_dir(
    work_dir: Path, pairs: int, arm: str, seed: int
) -> Path:
    return work_dir / "runs" / f"k{pairs}" / arm / f"seed{seed}"


def build_status(work_dir: Path) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    base_ready = _load_status(
        work_dir / "ready_receipt.json", ("READY",)
    )
    parity_ready = _load_status(
        work_dir / "operator_parity_ready.json", ("READY",)
    )
    gate_path = work_dir / "operator_parity_gate.json"
    gate = (
        json.loads(gate_path.read_text())
        if gate_path.is_file()
        else None
    )
    summary = _load_status(
        work_dir / "summary_mla_yarn_operator_parity.json",
        ("PASS",),
    )

    completed: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    for seed in SEEDS:
        for pairs in FREQUENCY_PAIRS:
            for arm in TRAINING_ARMS:
                run_dir = _run_dir(work_dir, pairs, arm, seed)
                result = _load_status(
                    run_dir / "train_result.json", ("PASS",)
                )
                if result is not None:
                    completed.append(
                        {
                            "seed": seed,
                            "frequency_pairs": pairs,
                            "arm": arm,
                            "elapsed_seconds": result.get(
                                "elapsed_seconds"
                            ),
                            "final_loss": result.get("final_loss"),
                        }
                    )
                    continue
                metadata = run_dir / "metadata.json"
                progress = _last_jsonl(run_dir / "train_log.jsonl")
                if metadata.is_file() or progress is not None:
                    record: dict[str, Any] = {
                        "seed": seed,
                        "frequency_pairs": pairs,
                        "arm": arm,
                    }
                    if progress is not None:
                        step = int(progress.get("step", 0))
                        total = int(
                            progress.get(
                                "optimizer_steps",
                                SPEC.train_tokens
                                // (32 * SPEC.train_length),
                            )
                        )
                        rate = float(
                            progress.get("tokens_per_second", 0.0)
                        )
                        remaining_tokens = max(
                            0,
                            (total - step)
                            * (SPEC.train_tokens // total),
                        )
                        record.update(progress)
                        record["eta_seconds"] = (
                            remaining_tokens / rate if rate > 0 else None
                        )
                    active.append(record)

    raw_evaluations = [
        path
        for path in work_dir.glob(
            "runs/k*/*/seed*/eval_*_*_raw.json"
        )
        if _load_status(path, ("PASS",)) is not None
    ]
    parity_evaluations = [
        path
        for path in work_dir.glob(
            "runs/k*/*/seed*/eval_parity_*_*_*.json"
        )
        if _load_status(path, ("PASS",)) is not None
    ]
    checkpoints = list(
        work_dir.glob("runs/k*/*/seed*/checkpoint_*.pt")
    )
    incomplete = list(work_dir.rglob("*.incomplete"))

    gate_status = gate.get("status") if isinstance(gate, dict) else None
    if summary is not None:
        phase = "COMPLETE"
        terminal = True
    elif gate_status == "STOP":
        phase = "GATE_STOP"
        terminal = True
    elif base_ready is None or parity_ready is None:
        phase = "OFFLINE_NOT_READY"
        terminal = False
    elif gate_status == "PASS":
        if active or len(completed) > 4:
            phase = "CONFIRM_RUNNING"
        else:
            phase = "READY_FOR_CONFIRM"
        terminal = False
    elif active or completed or raw_evaluations or parity_evaluations:
        phase = "GATE_RUNNING"
        terminal = False
    else:
        phase = "READY_FOR_GATE"
        terminal = False

    return {
        "schema_version": 1,
        "status": "PASS",
        "artifact_status_only": True,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "work_dir": str(work_dir),
        "phase": phase,
        "terminal": terminal,
        "ready": {
            "base": base_ready is not None,
            "operator_parity": parity_ready is not None,
        },
        "gate_status": gate_status,
        "claim_gate": (
            summary.get("claim_gate") if summary is not None else None
        ),
        "training": {
            "completed": len(completed),
            "active": active,
            "gate_expected": 4,
            "full_expected": 12,
            "completed_records": completed,
        },
        "evaluation": {
            "raw_completed": len(raw_evaluations),
            "parity_completed": len(parity_evaluations),
            "gate_expected": {"raw": 8, "parity": 24},
            "confirm_additional": {"raw": 24, "parity": 72},
            "terminal_total_expected": {"raw": 32, "parity": 96},
        },
        "storage": {
            "checkpoint_count": len(checkpoints),
            "checkpoint_bytes": sum(
                path.stat().st_size for path in checkpoints
            ),
            "incomplete_count": len(incomplete),
        },
        "monitoring_boundary": (
            "This snapshot reads artifacts only. It does not prove a live "
            "PID, GPU utilization, or server state."
        ),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--exit-if-terminal", action="store_true")
    args = parser.parse_args()
    result = build_status(args.work_dir)
    if args.output is not None:
        _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    if args.exit_if_terminal and not result["terminal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
