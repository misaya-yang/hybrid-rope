#!/usr/bin/env python3
"""Run X4's clean matched-dose control without touching the frozen T/P arms.

The default mode only prints the resolved plan.  GPU evaluation is possible only
with ``--execute``.  This is intentionally a thin wrapper around the existing
analytic table builder, frozen-model evaluator, and matched-generation reporter.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Iterable

import numpy as np

from experiments.fixed_rope_three_interfaces_20260913.tables import (
    find_table,
    tensor_sha256,
    validate_table,
)


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
NON_QA_TASKS = TASKS[:11]
QA_TASKS = TASKS[11:]
SUPPORTED_LENGTHS = (16384, 32768)
DEFAULT_ROWS_PER_TASK = {16384: 50, 32768: 200}
RUNTIME_KEYS = (
    "generation_length_caps",
    "limit_per_cell",
    "prefill_chunk_size",
    "generation_prefill_strategy",
    "batch_size",
    "left_pad_batches",
    "runtime_versions",
)


@dataclass(frozen=True)
class Job:
    length: int
    shard: str
    tasks: tuple[str, ...]
    rows_per_task: int
    panel: Path
    tailspline_run: Path
    mrpro_run: Path
    control_run: Path

    @property
    def expected_rows(self) -> int:
        return len(self.tasks) * self.rows_per_task


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def cli_tokens(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    for raw in values:
        if isinstance(raw, (list, tuple)):
            result.extend(cli_tokens(raw))
        else:
            result.extend(item.strip() for item in str(raw).split(",") if item.strip())
    return result


def parse_lengths(values: Iterable[object]) -> tuple[int, ...]:
    parsed: list[int] = []
    parsed.extend(int(item) for item in cli_tokens(values))
    if not parsed:
        parsed = list(SUPPORTED_LENGTHS)
    if len(set(parsed)) != len(parsed) or any(value not in SUPPORTED_LENGTHS for value in parsed):
        raise ValueError("X4 supports each of clean16K and clean32K at most once")
    return tuple(parsed)


def parse_rows_per_task(values: Iterable[object], lengths: tuple[int, ...]) -> dict[int, int]:
    tokens = cli_tokens(values)
    if not tokens:
        return {length: DEFAULT_ROWS_PER_TASK[length] for length in lengths}
    result: dict[int, int] = {}
    bare: list[int] = []
    for token in tokens:
        if ":" in token:
            raw_length, raw_count = token.split(":", 1)
            length, count = int(raw_length), int(raw_count)
            if length in result:
                raise ValueError("duplicate --rows-per-task length")
            result[length] = count
        else:
            bare.append(int(token))
    remaining = [length for length in lengths if length not in result]
    if bare:
        if len(bare) == 1 and len(remaining) == 1:
            result[remaining[0]] = bare[0]
        elif len(bare) == len(remaining):
            result.update(zip(remaining, bare))
        else:
            raise ValueError("bare row counts must align one-to-one with --lengths")
    if set(result) != set(lengths) or any(result[length] != DEFAULT_ROWS_PER_TASK[length] for length in lengths):
        raise ValueError("X4 must reuse the frozen clean contracts: 16K:50 and 32K:200")
    return result


def build_jobs(
    *, data_root: Path, out: Path, lengths: tuple[int, ...], rows_per_task: dict[int, int],
) -> list[Job]:
    jobs: list[Job] = []
    if 16384 in lengths:
        root = data_root / "tailspline_llama_s4_16k_ruler50_clean"
        for shard, tasks in (("nonqa11", NON_QA_TASKS), ("qa2", QA_TASKS)):
            jobs.append(Job(
                length=16384,
                shard=shard,
                tasks=tasks,
                rows_per_task=rows_per_task[16384],
                panel=root / "assets" / shard / "inputs.jsonl",
                tailspline_run=root / "runs" / f"tailspline_{shard}",
                mrpro_run=root / "runs" / f"mrpro_{shard}",
                control_run=out / "runs" / "16384" / f"dose_control_c_{shard}",
            ))
    if 32768 in lengths:
        root = data_root / "tailspline_llama_s4_32k_ruler200_clean"
        jobs.append(Job(
            length=32768,
            shard="full13",
            tasks=TASKS,
            rows_per_task=rows_per_task[32768],
            panel=root / "assets" / "inputs.jsonl",
            tailspline_run=root / "runs" / "tailspline",
            mrpro_run=root / "runs" / "mrpro",
            control_run=out / "runs" / "32768" / "dose_control_c",
        ))
    return jobs


def receipt_table(receipt_path: Path) -> tuple[dict, np.ndarray, float]:
    receipt = read_json(receipt_path)
    values, gain = validate_table(find_table(receipt), pairs=64)
    recorded_hash = receipt.get("table_sha256_float32")
    if recorded_hash and tensor_sha256(values) != recorded_hash:
        raise ValueError(f"table receipt hash drift: {receipt_path}")
    return receipt, values, gain


def validate_table_contract(contract: dict, receipt_path: Path) -> None:
    _, expected_values, expected_gain = receipt_table(receipt_path)
    static = contract.get("static_table")
    if not static:
        raise ValueError("run contract lacks its frozen static table")
    actual_values, actual_gain = validate_table(static, pairs=64)
    if not np.array_equal(actual_values, expected_values) or actual_gain != expected_gain:
        raise ValueError("run contract table/gain differs from its receipt")


def validate_panel(job: Job) -> tuple[list[dict], list[str]]:
    if not job.panel.is_file():
        raise FileNotFoundError(job.panel)
    rows = read_jsonl(job.panel)
    expected_cells = Counter({task: job.rows_per_task for task in job.tasks})
    if len(rows) != job.expected_rows or Counter(row.get("task") for row in rows) != expected_cells:
        raise ValueError(f"clean panel coverage drift: {job.panel}")
    prompts = [str(row.get("prompt_sha256") or "") for row in rows]
    if any(not value for value in prompts) or len(set(prompts)) != len(prompts):
        raise ValueError(f"clean panel prompt identity drift: {job.panel}")
    if any(int(row.get("length_cap", -1)) != job.length for row in rows):
        raise ValueError(f"clean panel length drift: {job.panel}")
    return rows, prompts


def validate_complete_run(
    run: Path, job: Job, *, receipt: Path, expected_prompts: list[str],
) -> dict:
    required = (run / "status.json", run / "contract.json", run / "generations.jsonl")
    if any(not path.is_file() for path in required):
        raise ValueError(f"incomplete clean arm: {run}")
    status = read_json(run / "status.json")
    if status != {"status": "COMPLETE", "rows": job.expected_rows, "lm_rows": 0}:
        raise ValueError(f"unexpected clean arm status {run}: {status}")
    generations = read_jsonl(run / "generations.jsonl")
    observed_prompts = [str(row.get("prompt_sha256") or "") for row in generations]
    if observed_prompts != expected_prompts:
        raise ValueError(f"arm is not paired in clean source order: {run}")
    if any("ruler_official_score" not in row for row in generations):
        raise ValueError(f"arm lacks official RULER scores: {run}")
    contract = read_json(run / "contract.json")
    validate_table_contract(contract, receipt)
    if contract.get("generation_length_caps") != [job.length]:
        raise ValueError(f"run length contract drift: {run}")
    if int(contract.get("batch_size", -1)) != 1:
        raise ValueError("X4 forbids the old batch2 C path and requires clean batch1")
    return contract


def runtime_projection(contract: dict) -> dict:
    return {key: contract.get(key) for key in RUNTIME_KEYS}


def validate_runtime_match(reference: dict, other: dict, *, label: str) -> None:
    if runtime_projection(reference) != runtime_projection(other):
        raise ValueError(f"runtime contract differs for {label}")


def validate_matched_dose(
    tailspline_receipt: Path, control_receipt: Path, mrpro_receipt: Path, *, scale: float,
) -> None:
    tailspline, tail_values, tail_gain = receipt_table(tailspline_receipt)
    control, control_values, control_gain = receipt_table(control_receipt)
    mrpro, _, mrpro_gain = receipt_table(mrpro_receipt)
    expected_gain = 1.0 + 0.1 * math.log(scale)
    if any(value.get("band_envelope") != [18, 35] for value in (tailspline, control, mrpro)):
        raise ValueError("X4 receipts do not share the canonical Llama band")
    if any(value.get("scale") != scale for value in (tailspline, control, mrpro)):
        raise ValueError("X4 receipt scale drift")
    if tail_gain != control_gain or tail_gain != mrpro_gain or tail_gain != expected_gain:
        raise ValueError("X4 receipts do not share the declared gain")
    if abs(float(tailspline["sum_m"]) - float(control["sum_m"])) > 2e-6:
        raise ValueError("TailSpline and C are not matched in deployed FP32 exponent dose")
    if np.array_equal(tail_values, control_values):
        raise ValueError("TailSpline and C unexpectedly use one table")
    method = (control.get("table") or {}).get("construction", {}).get("method")
    if method != "tailspline_same_log_displacement_control":
        raise ValueError("C is not the existing analytic tailspline_dose_control")


def run_command(command: list[str], *, repo: Path, log_path: Path | None = None) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if log_path is None:
        subprocess.run(command, check=True, cwd=repo, env=env)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as stream:
        subprocess.run(command, check=True, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)


def table_command(args: argparse.Namespace, table: Path) -> list[str]:
    return [
        str(args.python), "-m", "experiments.fixed_rope_three_interfaces_20260913.tables",
        "analytic", "--config", str(args.model / "config.json"),
        "--method", "tailspline_dose_control", "--scale", str(args.scale),
        "--candidate-id", "llama3_8b_s4_clean_tailspline_dose_control_c",
        "--model-id", args.model_id, "--role", "control",
        "--changed-variable", "interior_allocation_shape_at_fixed_total_log_displacement",
        "--out", str(table),
    ]


def evaluation_command(
    args: argparse.Namespace, job: Job, table: Path, reference_contract: dict,
) -> list[str]:
    command = [
        str(args.python), "-m", "experiments.olmo_recovery_20260912.recovery_v2_eval",
        "--data", str(args.data_manifest), "--model", str(args.model), "--arm", "Native",
        "--extra-panel", str(job.panel), "--only-extra-panels", "--skip-lm",
        "--length-cap", str(job.length),
        "--prefill-chunk-size", str(int(reference_contract.get("prefill_chunk_size", 0))),
        "--batch-size", "1", "--static-table-json", str(table),
        "--table-label", f"llama3_8b_s4_clean_{job.length}_{job.shard}_dose_control_c",
        "--out", str(job.control_run), "--execute",
    ]
    if reference_contract.get("left_pad_batches"):
        command.append("--left-pad-batches")
    return command


def report_commands(
    args: argparse.Namespace, jobs: list[Job], reports: Path,
) -> tuple[list[str], list[str]]:
    common = [str(args.python), "-m", "experiments.fixed_rope_three_interfaces_20260913.matched_generation_report"]
    tailspline_sources = [["--source", f"tailspline={job.tailspline_run / 'generations.jsonl'}"] for job in jobs]
    control_sources = [["--source", f"dose_control_c={job.control_run / 'generations.jsonl'}"] for job in jobs]
    mrpro_sources = [["--source", f"mrpro={job.mrpro_run / 'generations.jsonl'}"] for job in jobs]
    flatten = lambda groups: [item for group in groups for item in group]
    lengths = [item for length in sorted({job.length for job in jobs}) for item in ("--length", str(length))]
    primary = (
        common + flatten(tailspline_sources + control_sources + mrpro_sources)
        + ["--candidate", "tailspline", "--baseline", "dose_control_c", "--baseline", "mrpro"]
        + lengths + ["--out", str(reports / "tailspline_vs_control_and_mrpro.json")]
    )
    control = (
        common + flatten(control_sources + mrpro_sources)
        + ["--candidate", "dose_control_c", "--baseline", "mrpro"]
        + lengths + ["--out", str(reports / "control_vs_mrpro.json")]
    )
    return primary, control


def ensure_report(command: list[str], *, repo: Path) -> None:
    output = Path(command[command.index("--out") + 1])
    if output.exists():
        report = read_json(output)
        candidate = command[command.index("--candidate") + 1]
        baselines = [command[index + 1] for index, value in enumerate(command) if value == "--baseline"]
        lengths = [int(command[index + 1]) for index, value in enumerate(command) if value == "--length"]
        if (
            report.get("status") != "MATCHED_GENERATION_RANGE_REPORT_V1"
            or report.get("candidate") != candidate
            or report.get("baselines") != baselines
            or report.get("lengths") != lengths
        ):
            raise ValueError(f"invalid existing X4 report: {output}")
        return
    run_command(command, repo=repo)


def resolved_plan(args: argparse.Namespace) -> tuple[dict, list[Job], Path, Path, Path, Path]:
    lengths = parse_lengths(args.lengths)
    rows_per_task = parse_rows_per_task(args.rows_per_task, lengths)
    out = args.out or args.data_root / "strong_evidence" / "llama_s4_clean_matched_dose_c"
    classic = args.data_root / "tailspline_llama_s4_classic"
    tailspline_receipt = classic / "tables" / "tailspline.json"
    mrpro_receipt = classic / "tables" / "mrpro.json"
    control_receipt = out / "tables" / "tailspline_dose_control.json"
    jobs = build_jobs(data_root=args.data_root, out=out, lengths=lengths, rows_per_task=rows_per_task)
    report_commands_value = report_commands(args, jobs, out / "reports")
    plan = {
        "status": "PLAN_ONLY" if not args.execute else "EXECUTION_REQUESTED",
        "experiment": "X4_clean_matched_dose_control",
        "gpu_started": False,
        "execute_required": True,
        "method": "analytic:tailspline_dose_control",
        "scale": args.scale,
        "lengths": list(lengths),
        "rows_per_task": {str(key): value for key, value in rows_per_task.items()},
        "reuse_arms": ["tailspline", "mrpro"],
        "new_arm_only": "dose_control_c",
        "batch_size": 1,
        "forbidden_launcher": "run_tailspline_llama_s4_matched_dose_c.sh",
        "out": str(out),
        "table_command": table_command(args, control_receipt),
        "jobs": [{
            "length": job.length,
            "shard": job.shard,
            "tasks": list(job.tasks),
            "expected_rows": job.expected_rows,
            "panel": str(job.panel),
            "reuse_tailspline": str(job.tailspline_run),
            "reuse_mrpro": str(job.mrpro_run),
            "new_control": str(job.control_run),
        } for job in jobs],
        "reports": [command[command.index("--out") + 1] for command in report_commands_value],
    }
    return plan, jobs, out, tailspline_receipt, control_receipt, mrpro_receipt


def execute(args: argparse.Namespace) -> dict:
    plan, jobs, out, tailspline_receipt, control_receipt, mrpro_receipt = resolved_plan(args)
    source_roots = {
        args.data_root / "tailspline_llama_s4_classic",
        args.data_root / "tailspline_llama_s4_16k_ruler50_clean",
        args.data_root / "tailspline_llama_s4_32k_ruler200_clean",
        args.data_root / "tailspline_llama_s4_matched_dose_c",
    }
    resolved_out = out.resolve()
    resolved_sources = {path.resolve() for path in source_roots}
    if any(resolved_out == path or path in resolved_out.parents for path in resolved_sources):
        raise ValueError("X4 requires a new output root and cannot overwrite any prior experiment")
    if args.scale != 4.0:
        raise ValueError("X4 reuses the frozen S4 T/P arms and therefore requires --scale 4")
    for path in (args.model / "config.json", args.data_manifest, tailspline_receipt, mrpro_receipt):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Validate every frozen input and both reused arms before starting the only new GPU arm.
    validated: dict[tuple[int, str], tuple[list[str], dict]] = {}
    for job in jobs:
        _, prompts = validate_panel(job)
        tailspline_contract = validate_complete_run(
            job.tailspline_run, job, receipt=tailspline_receipt, expected_prompts=prompts,
        )
        mrpro_contract = validate_complete_run(
            job.mrpro_run, job, receipt=mrpro_receipt, expected_prompts=prompts,
        )
        validate_runtime_match(tailspline_contract, mrpro_contract, label=f"T/P {job.length}/{job.shard}")
        validated[(job.length, job.shard)] = (prompts, tailspline_contract)

    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "reports").mkdir(exist_ok=True)
    if not control_receipt.exists():
        run_command(table_command(args, control_receipt), repo=args.repo)
    validate_matched_dose(tailspline_receipt, control_receipt, mrpro_receipt, scale=args.scale)

    for job in jobs:
        prompts, reference_contract = validated[(job.length, job.shard)]
        status_path = job.control_run / "status.json"
        complete = status_path.is_file() and read_json(status_path).get("status") == "COMPLETE"
        if complete:
            control_contract = validate_complete_run(
                job.control_run, job, receipt=control_receipt, expected_prompts=prompts,
            )
        else:
            # Resume-compatible evaluator output is allowed; a partially populated run is not deleted.
            job.control_run.parent.mkdir(parents=True, exist_ok=True)
            run_command(
                evaluation_command(args, job, control_receipt, reference_contract),
                repo=args.repo,
                log_path=out / "logs" / f"dose_control_c_{job.length}_{job.shard}.log",
            )
            control_contract = validate_complete_run(
                job.control_run, job, receipt=control_receipt, expected_prompts=prompts,
            )
        validate_runtime_match(reference_contract, control_contract, label=f"T/C {job.length}/{job.shard}")

    primary, control = report_commands(args, jobs, out / "reports")
    ensure_report(primary, repo=args.repo)
    ensure_report(control, repo=args.repo)
    receipt = {
        **plan,
        "status": "X4_CLEAN_C_COMPLETE_V1",
        "gpu_started": True,
        "completed_control_rows": sum(job.expected_rows for job in jobs),
    }
    temporary = out / "execution_receipt.json.incomplete"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(out / "execution_receipt.json")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/root/autodl-tmp/hybrid-rope"))
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--model-id", default="meta_llama3_8b_instruct")
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/today_rope_plan_20260914"))
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument(
        "--lengths", action="append", nargs="+", default=[],
        help="16384 32768, comma-separated values, or repeated values",
    )
    parser.add_argument(
        "--rows-per-task", action="append", nargs="+", default=[],
        help="frozen LENGTH:COUNT mapping; defaults to 16384:50,32768:200",
    )
    parser.add_argument("--python", type=Path, default=Path("/root/miniconda3/bin/python"))
    parser.add_argument("--execute", action="store_true", help="start the new C GPU arm after all validations")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.data_manifest is None:
        args.data_manifest = args.data_root / "tailspline_llama_s4_classic" / "assets" / "ppl46" / "manifest.json"
    try:
        result = execute(args) if args.execute else resolved_plan(args)[0]
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
