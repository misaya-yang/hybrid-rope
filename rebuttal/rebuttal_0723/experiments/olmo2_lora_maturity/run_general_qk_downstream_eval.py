#!/usr/bin/env python3
"""Run one registered benchmark stream from the general-QK READY matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    append_jsonl,
    sha256_file,
)

from .preflight_general_qk_downstream_eval import STATUS as READY_STATUS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("ruler", "2wiki"), required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--evq-run", type=Path, required=True)
    parser.add_argument("--two-wiki-data", type=Path, required=True)
    parser.add_argument("--ruler-data", type=Path, required=True)
    return parser.parse_args()


def adapter_for(
    role: str | None, native_run: Path, evq_run: Path
) -> Path | None:
    if role is None:
        return None
    if role == "native":
        return native_run.resolve() / "adapter.pt"
    if role == "evq":
        return evq_run.resolve() / "adapter.pt"
    raise RuntimeError(f"unknown adapter role: {role}")


def command_for(
    *,
    job: dict[str, Any],
    checkpoint: Path,
    checkpoint_ready: Path,
    native_run: Path,
    evq_run: Path,
    two_wiki_data: Path,
    ruler_data: Path,
    output: Path,
) -> list[str]:
    adapter = adapter_for(job["adapter_role"], native_run, evq_run)
    if job["benchmark"] == "ruler":
        command = [
            sys.executable,
            "-m",
            (
                "rebuttal.rebuttal_0723.experiments."
                "olmo2_lora_maturity.evaluate_instruct_ruler_transfer"
            ),
            "--checkpoint",
            str(checkpoint),
            "--ready-receipt",
            str(checkpoint_ready),
            "--data-root",
            str(ruler_data),
            "--output",
            str(output),
            "--frequency",
            str(job["frequency"]),
            "--lengths",
            *[str(value) for value in job["lengths"]],
            "--limit-per-cell",
            str(job["limit_per_cell"]),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            (
                "rebuttal.rebuttal_0723.experiments."
                "olmo2_lora_maturity.evaluate_2wiki_phase_adaptation"
            ),
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-ready-receipt",
            str(checkpoint_ready),
            "--data-root",
            str(two_wiki_data),
            "--output",
            str(output),
            "--role",
            str(job["name"]),
            "--frequency",
            str(job["frequency"]),
            "--budgets",
            *[str(value) for value in job["lengths"]],
            "--limit",
            str(job["limit_per_cell"]),
        ]
        if job["fill_to_budget"]:
            command.append("--fill-to-budget")
    if adapter is not None:
        command.extend(
            [
                "--adapter",
                str(adapter),
                "--adaptation",
                "qk_answer",
                "--rank",
                "64",
                "--alpha",
                "128",
            ]
        )
    if job["yarn_factor"] is not None:
        command.extend(
            ["--yarn-factor", str(job["yarn_factor"])]
        )
    return command


def main() -> None:
    args = parse_args()
    receipt_path = args.ready_receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != READY_STATUS:
        raise RuntimeError("general-QK downstream READY status drift")
    code_root = Path(__file__).resolve().parent
    if sha256_file(Path(__file__).resolve()) != receipt[
        "bound_code_sha256"
    ]["launcher"]:
        raise RuntimeError("general-QK downstream launcher code drift")
    evaluator_name = (
        "evaluate_instruct_ruler_transfer.py"
        if args.benchmark == "ruler"
        else "evaluate_2wiki_phase_adaptation.py"
    )
    expected_key = (
        "ruler_evaluator"
        if args.benchmark == "ruler"
        else "two_wiki_evaluator"
    )
    if sha256_file(code_root / evaluator_name) != receipt[
        "bound_code_sha256"
    ][expected_key]:
        raise RuntimeError("general-QK downstream evaluator code drift")

    checkpoint = Path(receipt["checkpoint"]).resolve()
    registered = receipt["registered_paths"]
    supplied_paths = {
        "native_run": args.native_run.resolve(),
        "evq_run": args.evq_run.resolve(),
        "two_wiki_data": args.two_wiki_data.resolve(),
        "ruler_data": args.ruler_data.resolve(),
    }
    for name, supplied in supplied_paths.items():
        if supplied != Path(registered[name]).resolve():
            raise RuntimeError(
                f"general-QK downstream registered path drift: {name}"
            )
    if (
        sha256_file(args.checkpoint_ready_receipt.resolve())
        != receipt["inputs"]["checkpoint_ready_receipt_sha256"]
        or sha256_file(args.native_run.resolve() / "results.json")
        != receipt["inputs"]["native_results_sha256"]
        or sha256_file(args.evq_run.resolve() / "results.json")
        != receipt["inputs"]["evq_results_sha256"]
        or sha256_file(args.two_wiki_data.resolve() / "manifest.json")
        != receipt["inputs"]["two_wiki_manifest_sha256"]
        or sha256_file(args.ruler_data.resolve() / "manifest.json")
        != receipt["inputs"]["ruler_manifest_sha256"]
    ):
        raise RuntimeError("general-QK downstream registered input hash drift")
    output_root = Path(receipt["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stream_log = output_root / f"{args.benchmark}_launcher.jsonl"
    jobs = [
        value
        for value in receipt["jobs"]
        if value["benchmark"] == args.benchmark
    ]
    if not jobs:
        raise RuntimeError("general-QK downstream stream has no jobs")

    environment = dict(os.environ)
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    for job in jobs:
        output = output_root / str(job["name"])
        if (output / "results.json").is_file():
            append_jsonl(
                stream_log,
                {
                    "event": "skip_completed",
                    "job_name": job["name"],
                    "results_sha256": sha256_file(
                        output / "results.json"
                    ),
                },
            )
            continue
        command = command_for(
            job=job,
            checkpoint=checkpoint,
            checkpoint_ready=args.checkpoint_ready_receipt.resolve(),
            native_run=args.native_run.resolve(),
            evq_run=args.evq_run.resolve(),
            two_wiki_data=args.two_wiki_data.resolve(),
            ruler_data=args.ruler_data.resolve(),
            output=output,
        )
        log_path = output_root / f"{job['name']}.log"
        started = time.time()
        append_jsonl(
            stream_log,
            {
                "event": "start",
                "job": job,
                "command": command,
                "started_unix": started,
            },
        )
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        append_jsonl(
            stream_log,
            {
                "event": "complete",
                "job_name": job["name"],
                "returncode": int(completed.returncode),
                "elapsed_seconds": time.time() - started,
                "log": str(log_path),
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"downstream job failed: {job['name']}; see {log_path}"
            )


if __name__ == "__main__":
    main()
