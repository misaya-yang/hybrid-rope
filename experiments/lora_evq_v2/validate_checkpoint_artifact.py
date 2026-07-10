#!/usr/bin/env python3
"""Fail closed before reusing a completed LoRA comparison checkpoint."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch

from train_evq_lora import build_training_inv_freq, load_frequency_artifact
from train_positional_distill import fingerprint_model_source


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _named_hash(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"claim evidence is missing: {path.name}")
    return {"name": path.name, "sha256": sha256_file(path)}


def _read_invocation_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid invocation ledger JSON at line {line_number}"
                ) from error
    return entries


def _latest_recorded_global_step(checkpoint_dir: Path) -> int:
    metadata_path = checkpoint_dir / "experiment_meta.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_step = int(
            metadata.get("invocation", {}).get("ending_global_step", -1)
        )
        root_state_path = checkpoint_dir / "trainer_state.json"
        if not root_state_path.is_file():
            raise FileNotFoundError("final experiment metadata lacks trainer_state.json")
        root_state = json.loads(root_state_path.read_text(encoding="utf-8"))
        root_step = int(root_state.get("global_step", -1))
        if metadata_step != root_step:
            raise RuntimeError("final metadata and trainer state global_step mismatch")
        return metadata_step
    steps = []
    for candidate in checkpoint_dir.glob("checkpoint-*"):
        state_path = candidate / "trainer_state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            steps.append(int(state.get("global_step", -1)))
    return max(steps, default=0)


def append_invocation_ledger(
    checkpoint_dir: Path,
    claim_log_dir: Path,
    *,
    telemetry: Path,
    hardware_record: Path,
    train_log: Path,
    process_status: int,
) -> Optional[dict]:
    checkpoint_dir = Path(checkpoint_dir)
    claim_log_dir = Path(claim_log_dir)
    run_protocol_path = checkpoint_dir / "run_protocol.json"
    if not run_protocol_path.is_file():
        raise FileNotFoundError("cannot ledger an invocation without run_protocol.json")
    ledger_path = checkpoint_dir / "invocation_ledger.jsonl"
    entries = _read_invocation_ledger(ledger_path)
    starting_step = int(entries[-1]["ending_global_step"]) if entries else 0
    ending_step = _latest_recorded_global_step(checkpoint_dir)
    if ending_step < starting_step:
        raise RuntimeError("invocation ledger global steps moved backwards")
    if ending_step == starting_step:
        return None
    evidence_paths = {
        "telemetry": Path(telemetry),
        "hardware_record": Path(hardware_record),
        "train_log": Path(train_log),
    }
    for path in evidence_paths.values():
        if path.parent.resolve() != claim_log_dir.resolve():
            raise ValueError("invocation evidence files must live in claim_log_dir")
    entry = {
        "format_version": 1,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "starting_global_step": starting_step,
        "ending_global_step": ending_step,
        "process_status": int(process_status),
        "run_protocol_sha256": sha256_file(run_protocol_path),
        "evidence": {
            name: _named_hash(path) for name, path in evidence_paths.items()
        },
    }
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return entry


def validate_invocation_ledger(checkpoint_dir: Path, claim_log_dir: Path) -> list[dict]:
    checkpoint_dir = Path(checkpoint_dir)
    claim_log_dir = Path(claim_log_dir)
    ledger_path = checkpoint_dir / "invocation_ledger.jsonl"
    entries = _read_invocation_ledger(ledger_path)
    if not entries:
        raise RuntimeError("claim invocation ledger is empty")
    run_protocol_path = checkpoint_dir / "run_protocol.json"
    run_protocol_hash = sha256_file(run_protocol_path)
    run_protocol = json.loads(run_protocol_path.read_text(encoding="utf-8"))
    required_end = int(run_protocol.get("scientific", {}).get("max_steps", -1))
    expected_start = 0
    for entry in entries:
        start = int(entry.get("starting_global_step", -1))
        end = int(entry.get("ending_global_step", -1))
        if start != expected_start or end <= start:
            raise RuntimeError("claim invocation ledger step coverage has a gap")
        if entry.get("run_protocol_sha256") != run_protocol_hash:
            raise RuntimeError("claim invocation ledger protocol mismatch")
        evidence = entry.get("evidence", {})
        if set(evidence) != {"telemetry", "hardware_record", "train_log"}:
            raise RuntimeError("claim invocation ledger evidence set is incomplete")
        for label, record in evidence.items():
            name = record.get("name")
            if not isinstance(name, str) or Path(name).name != name:
                raise RuntimeError(f"unsafe invocation evidence name: {label}")
            if sha256_file(claim_log_dir / name) != record.get("sha256"):
                raise RuntimeError(f"claim invocation evidence mismatch: {label}")
        expected_start = end
    if expected_start != required_end:
        raise RuntimeError(
            f"claim invocation ledger ends at step {expected_start}, "
            f"expected {required_end}"
        )
    return entries


def finalize_claim_ready(
    checkpoint_dir: Path,
    claim_log_dir: Path,
    *,
    telemetry: Path,
    hardware_record: Path,
    train_log: Path,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    claim_log_dir = Path(claim_log_dir)
    marker_path = checkpoint_dir / "claim_ready.json"
    if marker_path.exists():
        raise FileExistsError("claim_ready.json already exists")
    evidence_paths = {
        "telemetry": Path(telemetry),
        "hardware_record": Path(hardware_record),
        "train_log": Path(train_log),
    }
    for path in evidence_paths.values():
        if path.parent.resolve() != claim_log_dir.resolve():
            raise ValueError("claim evidence files must live in claim_log_dir")
    checkpoint_names = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "custom_inv_freq.pt",
        "experiment_meta.json",
        "run_protocol.json",
        "trainer_state.json",
        "invocation_ledger.jsonl",
    )
    marker = {
        "format_version": 1,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint": {
            name: _named_hash(checkpoint_dir / name)["sha256"]
            for name in checkpoint_names
        },
        "evidence": {
            name: _named_hash(path) for name, path in evidence_paths.items()
        },
    }
    temporary = marker_path.with_name(
        f".{marker_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(marker, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, marker_path)
    finally:
        temporary.unlink(missing_ok=True)
    return marker_path


def validate_claim_ready(checkpoint_dir: Path, claim_log_dir: Path) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    claim_log_dir = Path(claim_log_dir)
    marker_path = checkpoint_dir / "claim_ready.json"
    if not marker_path.is_file():
        raise FileNotFoundError("claim_ready.json is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if int(marker.get("format_version", -1)) != 1:
        raise RuntimeError("claim_ready.json format mismatch")
    for name, expected_hash in marker.get("checkpoint", {}).items():
        if Path(name).name != name:
            raise RuntimeError("unsafe checkpoint evidence name")
        if sha256_file(checkpoint_dir / name) != expected_hash:
            raise RuntimeError(f"claim checkpoint evidence mismatch: {name}")
    required_checkpoint = {
        "adapter_model.safetensors",
        "adapter_config.json",
        "custom_inv_freq.pt",
        "experiment_meta.json",
        "run_protocol.json",
        "trainer_state.json",
        "invocation_ledger.jsonl",
    }
    if set(marker.get("checkpoint", {})) != required_checkpoint:
        raise RuntimeError("claim checkpoint evidence set is incomplete")
    evidence = marker.get("evidence", {})
    if set(evidence) != {"telemetry", "hardware_record", "train_log"}:
        raise RuntimeError("claim runtime evidence set is incomplete")
    for label, record in evidence.items():
        name = record.get("name")
        if not isinstance(name, str) or Path(name).name != name:
            raise RuntimeError(f"unsafe claim evidence name: {label}")
        if sha256_file(claim_log_dir / name) != record.get("sha256"):
            raise RuntimeError(f"claim runtime evidence mismatch: {label}")
    validate_invocation_ledger(checkpoint_dir, claim_log_dir)
    return marker


def validate_checkpoint_artifact(
    checkpoint_dir: Path,
    expected_method: str,
    expected_objective: Optional[str] = None,
    expected_data_manifest: Optional[Path] = None,
    expected_model: Optional[str] = None,
    claim_log_dir: Optional[Path] = None,
    require_claim_ready: bool = False,
) -> Dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    metadata_path = checkpoint_dir / "experiment_meta.json"
    if not adapter_path.is_file():
        raise FileNotFoundError("adapter_model.safetensors is missing")
    if not metadata_path.is_file():
        raise FileNotFoundError("experiment_meta.json is missing")

    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    recorded_method = metadata.get("rope_method", metadata.get("method"))
    if recorded_method != expected_method:
        raise RuntimeError(
            f"Checkpoint method mismatch: expected {expected_method}, found {recorded_method!r}"
        )

    if expected_objective is not None:
        if metadata.get("objective") != expected_objective:
            raise RuntimeError(
                f"Checkpoint objective mismatch: expected {expected_objective}, "
                f"found {metadata.get('objective')!r}"
            )
        actual_adapter_sha = sha256_file(adapter_path)
        if metadata.get("adapter_sha256") != actual_adapter_sha:
            raise RuntimeError("Checkpoint adapter SHA-256 mismatch")
        if expected_data_manifest is None:
            raise ValueError("expected_data_manifest is required for objective validation")
        expected_manifest_sha = sha256_file(expected_data_manifest)
        if metadata.get("data_manifest_sha256") != expected_manifest_sha:
            raise RuntimeError("Checkpoint data-manifest SHA-256 mismatch")
        if expected_model is None:
            raise ValueError("expected_model is required for objective validation")
        if metadata.get("base_model_fingerprint") != fingerprint_model_source(expected_model):
            raise RuntimeError("Checkpoint base-model fingerprint mismatch")
        run_protocol_path = checkpoint_dir / "run_protocol.json"
        if not run_protocol_path.is_file():
            raise FileNotFoundError("run_protocol.json is missing")
        if metadata.get("run_protocol_sha256") != sha256_file(run_protocol_path):
            raise RuntimeError("Checkpoint run-protocol SHA-256 mismatch")
        run_protocol = json.loads(run_protocol_path.read_text(encoding="utf-8"))
        if (
            run_protocol.get("objective") != expected_objective
            or run_protocol.get("student_method") != expected_method
            or run_protocol.get("data_manifest_sha256") != expected_manifest_sha
            or run_protocol.get("base_model_fingerprint")
            != metadata.get("base_model_fingerprint")
        ):
            raise RuntimeError("Checkpoint immutable run protocol mismatch")

        expected_steps = 1 if expected_method == "native_geo" else 300
        required = {
            "seed": 42,
            "max_steps": expected_steps,
            "lora_r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "lora_targets": ["q_proj", "k_proj"],
            "effective_batch_size": 8,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"Checkpoint protocol {key} mismatch: expected {expected!r}, "
                    f"found {metadata.get(key)!r}"
                )
        performance = run_protocol.get("performance", {})
        if (
            performance.get("per_device_batch_size")
            != metadata.get("per_device_batch_size")
            or performance.get("gradient_accumulation_steps")
            != metadata.get("gradient_accumulation_steps")
            or performance.get("gradient_checkpointing")
            != metadata.get("gradient_checkpointing")
            or performance.get("compile")
            != metadata.get("compile", {}).get("enabled")
        ):
            raise RuntimeError("Checkpoint performance protocol mismatch")

        adapter_config_path = checkpoint_dir / "adapter_config.json"
        if not adapter_config_path.is_file():
            raise FileNotFoundError("adapter_config.json is missing")
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        for key in ("r", "lora_alpha", "lora_dropout"):
            metadata_key = "lora_r" if key == "r" else key
            if adapter_config.get(key) != metadata.get(metadata_key):
                raise RuntimeError(f"adapter_config {key} disagrees with experiment metadata")
        if sorted(adapter_config.get("target_modules", [])) != ["k_proj", "q_proj"]:
            raise RuntimeError("adapter_config target_modules must be q_proj,k_proj")
        if not (checkpoint_dir / "trainer_state.json").is_file():
            raise FileNotFoundError("trainer_state.json is missing")

    provenance = None
    if expected_method in {"evq_cosh", "native_geo"}:
        inv_freq, frequency_data, provenance = load_frequency_artifact(
            checkpoint_dir / "custom_inv_freq.pt",
            expected_method=expected_method,
        )
        if expected_objective is not None:
            if int(frequency_data.get("head_dim", -1)) != 128 or float(
                frequency_data.get("base", float("nan"))
            ) != 500_000.0:
                raise RuntimeError("Checkpoint frequency geometry mismatch")
            if expected_method == "evq_cosh" and (
                frequency_data.get("tau") != 1.414
                or frequency_data.get("midpoint") is not True
            ):
                raise RuntimeError("Checkpoint EVQ frequency metadata mismatch")
            if expected_method == "native_geo" and (
                frequency_data.get("tau") is not None
                or frequency_data.get("midpoint") is not False
            ):
                raise RuntimeError("Checkpoint Geo frequency metadata mismatch")
            canonical, _ = build_training_inv_freq(
                rope_method=expected_method,
                head_dim=int(frequency_data.get("head_dim", 2 * inv_freq.numel())),
                base=float(frequency_data["base"]),
                tau=float(frequency_data.get("tau") or 0.0),
            )
            if not torch.allclose(
                inv_freq.to(torch.float64),
                canonical.to(torch.float64),
                rtol=1e-7,
                atol=1e-12,
            ):
                raise RuntimeError("Checkpoint frequency artifact is not canonical")
    claim_ready = None
    if require_claim_ready:
        if claim_log_dir is None:
            raise ValueError("claim_log_dir is required for claim-ready validation")
        claim_ready = validate_claim_ready(checkpoint_dir, claim_log_dir)
    return {
        "checkpoint": checkpoint_dir.name,
        "method": recorded_method,
        "frequency_provenance": provenance,
        "claim_ready": claim_ready is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-method",
        choices=["native_geo", "evq_cosh", "yarn"],
        required=True,
    )
    parser.add_argument("--expected-objective", default=None)
    parser.add_argument("--expected-data-manifest", type=Path, default=None)
    parser.add_argument("--expected-model", default=None)
    parser.add_argument("--claim-log-dir", type=Path, default=None)
    parser.add_argument("--require-claim-ready", action="store_true")
    parser.add_argument("--finalize-claim-ready", action="store_true")
    parser.add_argument("--telemetry", type=Path, default=None)
    parser.add_argument("--hardware-record", type=Path, default=None)
    parser.add_argument("--train-log", type=Path, default=None)
    parser.add_argument("--append-invocation-ledger", action="store_true")
    parser.add_argument("--process-status", type=int, default=0)
    args = parser.parse_args()
    if args.append_invocation_ledger:
        if None in (
            args.claim_log_dir,
            args.telemetry,
            args.hardware_record,
            args.train_log,
        ):
            parser.error(
                "ledger append requires claim-log-dir, telemetry, "
                "hardware-record, and train-log"
            )
        entry = append_invocation_ledger(
            args.checkpoint,
            args.claim_log_dir,
            telemetry=args.telemetry,
            hardware_record=args.hardware_record,
            train_log=args.train_log,
            process_status=args.process_status,
        )
        print("no durable checkpoint progress" if entry is None else json.dumps(entry))
        return
    result = validate_checkpoint_artifact(
        args.checkpoint,
        args.expected_method,
        expected_objective=args.expected_objective,
        expected_data_manifest=args.expected_data_manifest,
        expected_model=args.expected_model,
        claim_log_dir=args.claim_log_dir,
        require_claim_ready=args.require_claim_ready,
    )
    if args.finalize_claim_ready:
        if None in (
            args.claim_log_dir,
            args.telemetry,
            args.hardware_record,
            args.train_log,
        ):
            parser.error(
                "claim finalization requires claim-log-dir, telemetry, "
                "hardware-record, and train-log"
            )
        finalize_claim_ready(
            args.checkpoint,
            args.claim_log_dir,
            telemetry=args.telemetry,
            hardware_record=args.hardware_record,
            train_log=args.train_log,
        )
        validate_claim_ready(args.checkpoint, args.claim_log_dir)
    print(f"validated {result['checkpoint']} ({result['method']})")


if __name__ == "__main__":
    main()
