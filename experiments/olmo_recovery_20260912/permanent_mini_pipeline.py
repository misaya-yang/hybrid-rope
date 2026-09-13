#!/usr/bin/env python3
"""Freeze, complete, and score reusable tiered RULER mini panels.

The module is deliberately model-agnostic.  It never launches a model itself:
``plan-run`` prints the existing ``recovery_v2_eval`` command for the rows that
are actually missing.  Historical and new outputs are merged by prompt hash,
then rescored from raw text with one scorer before any comparison is made.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys
from typing import Iterable

import numpy as np


CORE6 = (
    "niah_single_2",
    "niah_multikey_2",
    "niah_multiquery",
    "vt",
    "fwe",
    "qa_1",
)
IDENTITY_FIELDS = (
    "model_revision",
    "tokenizer_template",
    "table",
    "decoder",
    "scorer",
    "precision_arithmetic",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_panel(panel_path: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "FROZEN":
        raise ValueError("mini panel manifest is not frozen")
    actual_hash = sha256_file(panel_path)
    if actual_hash != manifest.get("panel_sha256"):
        raise ValueError("mini panel bytes differ from the frozen manifest")
    rows = read_jsonl(panel_path)
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("mini panel row count differs from the frozen manifest")
    validate_panel(
        rows, tasks=manifest["tasks"], lengths=manifest["lengths"],
        rows_per_cell=int(manifest["rows_per_cell"]),
    )
    return rows, manifest


def parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw_path)


def explicit_semantic_id(row: dict) -> str | None:
    """Return an explicit source-case identity, never an inferred answer key."""
    for field in ("semantic_group_id", "source_semantic_id", "semantic_id", "group_id"):
        value = row.get(field)
        if value is not None and str(value):
            return f"{field}:{value}"
    value = row.get("mini_semantic_id")
    if value and value != row.get("prompt_sha256"):
        return f"mini_semantic_id:{value}"
    return None


def normalized_panel_row(source: str, row: dict) -> dict:
    required = (
        "row_id", "task", "length_cap", "prompt_ids", "prompt_sha256",
        "references", "max_new_tokens",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"panel row lacks {missing}")
    prompt = str(row["prompt_sha256"])
    result = {
        **row,
        "row_id": f"{source}:{row['row_id']}",
        "source_panel": source,
        "source_row_id": str(row["row_id"]),
        "prompt_sha256": prompt,
        "mini_semantic_id": explicit_semantic_id(row) or prompt,
    }
    return result


def validate_panel(
    rows: list[dict], *, tasks: Iterable[str], lengths: Iterable[int], rows_per_cell: int,
) -> None:
    tasks = tuple(tasks)
    lengths = tuple(lengths)
    expected = {(task, length) for task in tasks for length in lengths}
    counts = Counter((row["task"], int(row["length_cap"])) for row in rows)
    if set(counts) != expected or any(counts[cell] != rows_per_cell for cell in expected):
        raise ValueError(f"panel cell counts differ from {rows_per_cell}: {dict(counts)}")
    prompts = [row["prompt_sha256"] for row in rows]
    if len(prompts) != len(set(prompts)):
        raise ValueError("panel contains duplicate prompt hashes")
    row_ids = [row["row_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("panel contains duplicate source-prefixed row IDs")
    semantic_cells = [
        (row["task"], int(row["length_cap"]), row["mini_semantic_id"])
        for row in rows
    ]
    if len(semantic_cells) != len(set(semantic_cells)):
        raise ValueError("panel contains a duplicate semantic case within a task-length cell")


def freeze_panel(
    sources: list[tuple[str, Path]], *, tasks: list[str], lengths: list[int],
    rows_per_cell: int, panel_id: str,
) -> tuple[list[dict], dict]:
    if rows_per_cell <= 0 or not tasks or not lengths:
        raise ValueError("tasks, lengths, and a positive rows-per-cell are required")
    selected: list[dict] = []
    counts: Counter = Counter()
    seen_prompts: set[str] = set()
    seen_semantic_cells: set[tuple[str, int, str]] = set()
    source_receipts = []
    duplicate_prompts = 0
    duplicate_semantics = 0
    for label, path in sources:
        accepted = 0
        source_rows = read_jsonl(path)
        for raw in source_rows:
            if raw.get("task") not in tasks or int(raw.get("length_cap", -1)) not in lengths:
                continue
            row = normalized_panel_row(label, raw)
            cell = (row["task"], int(row["length_cap"]))
            if counts[cell] >= rows_per_cell:
                continue
            prompt = row["prompt_sha256"]
            semantic_cell = (*cell, row["mini_semantic_id"])
            if prompt in seen_prompts:
                duplicate_prompts += 1
                continue
            if semantic_cell in seen_semantic_cells:
                duplicate_semantics += 1
                continue
            selected.append(row)
            counts[cell] += 1
            accepted += 1
            seen_prompts.add(prompt)
            seen_semantic_cells.add(semantic_cell)
        source_receipts.append({"source": label, "path": str(path), "accepted_rows": accepted})
    validate_panel(selected, tasks=tasks, lengths=lengths, rows_per_cell=rows_per_cell)
    manifest = {
        "status": "FROZEN",
        "panel_id": panel_id,
        "rows": len(selected),
        "tasks": tasks,
        "lengths": lengths,
        "rows_per_cell": rows_per_cell,
        "cell_counts": {f"{task}/{length}": counts[(task, length)] for task in tasks for length in lengths},
        "sources": source_receipts,
        "deduplication": {
            "key": "global prompt_sha256 OR explicit semantic identity within task-length cell",
            "duplicate_prompts_skipped": duplicate_prompts,
            "duplicate_semantics_skipped": duplicate_semantics,
        },
    }
    return selected, manifest


def expand_result_specs(specs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for spec in specs:
        matches = [Path(path) for path in glob.glob(spec)]
        if not matches and Path(spec).is_file():
            matches = [Path(spec)]
        if not matches:
            raise FileNotFoundError(spec)
        for path in sorted(matches):
            if path.is_dir():
                paths.extend(sorted(path.glob("*.json")))
            else:
                paths.append(path)
    return paths


def read_result_records(specs: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in expand_result_specs(specs):
        if path.suffix == ".jsonl":
            rows.extend(read_jsonl(path))
        else:
            value = json.loads(path.read_text())
            if isinstance(value, list):
                rows.extend(value)
            elif isinstance(value, dict) and "row_id" in value:
                rows.append(value)
    return rows


def raw_output_text(row: dict) -> str:
    for field in ("output_text", "raw_text", "output"):
        value = row.get(field)
        if isinstance(value, str):
            return value
    raise ValueError("generation record has no raw output text")


def score_output(source: dict, output: str) -> float:
    from scripts.experiments.olmo_fast_screen.ruler_bench import score

    return float(score(source, output))


def merge_arm(
    panel_rows: list[dict], result_specs: list[str], *, arm_label: str, identity: dict,
) -> tuple[list[dict], list[dict], dict]:
    missing_identity = [field for field in IDENTITY_FIELDS if not identity.get(field)]
    if missing_identity:
        raise ValueError(f"arm identity lacks {missing_identity}")
    panel = {row["prompt_sha256"]: row for row in panel_rows}
    if len(panel) != len(panel_rows):
        raise ValueError("panel prompt identities are not unique")
    found: dict[str, dict] = {}
    duplicate_records = 0
    foreign_records = 0
    for record in read_result_records(result_specs):
        prompt = record.get("prompt_sha256") or record.get("prompt_input_ids_sha256")
        if prompt not in panel:
            foreign_records += 1
            continue
        source = panel[prompt]
        for field in ("task", "length_cap"):
            if field in record and record[field] != source[field]:
                raise ValueError(f"generation {prompt} differs at {field}")
        if "references" in record and record["references"] != source["references"]:
            raise ValueError(f"generation {prompt} differs at references")
        output = raw_output_text(record)
        generated = record.get("generated_ids", record.get("output_ids", []))
        ended_eos = bool(record.get("ended_eos", record.get("eos_seen", False)))
        hit_cap = bool(record.get(
            "hit_cap",
            record.get("cap_hit", len(generated) >= int(source["max_new_tokens"]) and not ended_eos),
        ))
        canonical = {
            "arm": arm_label,
            "row_id": source["row_id"],
            "source_panel": source["source_panel"],
            "source_row_id": source["source_row_id"],
            "task": source["task"],
            "family": source.get("family"),
            "length_cap": int(source["length_cap"]),
            "prompt_sha256": prompt,
            "mini_semantic_id": source["mini_semantic_id"],
            "references": source["references"],
            "output_text": output,
            "generated_ids": generated,
            "ended_eos": ended_eos,
            "hit_cap": hit_cap,
            "official_score": score_output(source, output),
        }
        previous = found.get(prompt)
        if previous is not None:
            duplicate_records += 1
            for field in ("output_text", "generated_ids", "ended_eos", "official_score"):
                if previous[field] != canonical[field]:
                    raise ValueError(f"conflicting duplicate result for prompt {prompt}")
            continue
        found[prompt] = canonical
    merged = [found[row["prompt_sha256"]] for row in panel_rows if row["prompt_sha256"] in found]
    missing = [row for row in panel_rows if row["prompt_sha256"] not in found]
    coverage = {
        "status": "COMPLETE" if not missing else "MISSING_ROWS",
        "arm": arm_label,
        "panel_rows": len(panel_rows),
        "covered_rows": len(merged),
        "missing_rows": len(missing),
        "coverage_by_cell": dict(sorted(Counter(
            f"{row['task']}/{row['length_cap']}" for row in merged
        ).items())),
        "missing_by_cell": dict(sorted(Counter(
            f"{row['task']}/{row['length_cap']}" for row in missing
        ).items())),
        "duplicate_records_ignored": duplicate_records,
        "foreign_records_ignored": foreign_records,
        "identity": identity,
        "result_sources": result_specs,
        "scoring": "all old and new raw outputs rescored with one current official scorer",
    }
    return merged, missing, coverage


def log_auc(length_scores: dict[int, float]) -> float:
    lengths = sorted(length_scores)
    if len(lengths) < 2 or lengths[0] <= 0:
        raise ValueError("log-AUC needs at least two positive lengths")
    numerator = sum(
        0.5 * (length_scores[left] + length_scores[right])
        * (math.log(right) - math.log(left))
        for left, right in zip(lengths, lengths[1:])
    )
    return numerator / (math.log(lengths[-1]) - math.log(lengths[0]))


def summarize_arm(rows: list[dict], panel_rows: list[dict], tasks: list[str], lengths: list[int]) -> dict:
    if [row["prompt_sha256"] for row in rows] != [row["prompt_sha256"] for row in panel_rows]:
        raise ValueError("merged arm is not a complete panel-order match")
    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        cells[(row["task"], int(row["length_cap"]))].append(row)
    by_length = {}
    for length in lengths:
        by_task = {}
        for task in tasks:
            values = cells[(task, length)]
            if not values:
                raise ValueError(f"empty scoring cell {task}/{length}")
            by_task[task] = {
                "rows": len(values),
                "official": float(np.mean([row["official_score"] for row in values])),
                "eos_rate": float(np.mean([row["ended_eos"] for row in values])),
                "cap_rate": float(np.mean([row["hit_cap"] for row in values])),
            }
        by_length[str(length)] = {
            "task_macro_official": float(np.mean([by_task[task]["official"] for task in tasks])),
            "task_macro_eos_rate": float(np.mean([by_task[task]["eos_rate"] for task in tasks])),
            "task_macro_cap_rate": float(np.mean([by_task[task]["cap_rate"] for task in tasks])),
            "tasks": by_task,
        }
    curve = {length: by_length[str(length)]["task_macro_official"] for length in lengths}
    task_auc = {
        task: log_auc({length: by_length[str(length)]["tasks"][task]["official"] for length in lengths})
        for task in tasks
    }
    return {
        "by_length": by_length,
        "log_length_auc": log_auc(curve),
        "interval_min": min(curve.values()),
        "task_log_length_auc": task_auc,
    }


def bootstrap_contrast(
    candidate: list[dict], baseline: list[dict], *, tasks: list[str], lengths: list[int],
    draws: int, seed: int,
) -> dict:
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    candidate_by_prompt = {row["prompt_sha256"]: row for row in candidate}
    baseline_by_prompt = {row["prompt_sha256"]: row for row in baseline}
    if set(candidate_by_prompt) != set(baseline_by_prompt):
        raise ValueError("bootstrap arms are not row paired")
    cells: dict[tuple[str, int], list[str]] = defaultdict(list)
    semantic_by_cell: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for row in candidate:
        cell = (row["task"], int(row["length_cap"]))
        prompt = row["prompt_sha256"]
        cells[cell].append(prompt)
        semantic = row.get("mini_semantic_id")
        if semantic and semantic != prompt:
            semantic_by_cell[cell][semantic] = prompt
    joint_semantic: dict[str, list[str]] = {}
    for task in tasks:
        mappings = [semantic_by_cell[(task, length)] for length in lengths]
        if mappings and all(mappings) and all(set(mapping) == set(mappings[0]) for mapping in mappings[1:]):
            joint_semantic[task] = sorted(mappings[0])
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        candidate_curve = {}
        baseline_curve = {}
        joint_draws = {
            task: rng.integers(0, len(semantic_ids), size=len(semantic_ids))
            for task, semantic_ids in joint_semantic.items()
        }
        for length in lengths:
            candidate_tasks = []
            baseline_tasks = []
            for task in tasks:
                if task in joint_semantic:
                    semantic_ids = joint_semantic[task]
                    selected = [
                        semantic_by_cell[(task, length)][semantic_ids[int(index)]]
                        for index in joint_draws[task]
                    ]
                else:
                    prompts = cells[(task, length)]
                    take = rng.integers(0, len(prompts), size=len(prompts))
                    selected = [prompts[int(index)] for index in take]
                candidate_tasks.append(float(np.mean([
                    candidate_by_prompt[prompt]["official_score"] for prompt in selected
                ])))
                baseline_tasks.append(float(np.mean([
                    baseline_by_prompt[prompt]["official_score"] for prompt in selected
                ])))
            candidate_curve[length] = float(np.mean(candidate_tasks))
            baseline_curve[length] = float(np.mean(baseline_tasks))
        samples[draw] = log_auc(candidate_curve) - log_auc(baseline_curve)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "draws": draws,
        "seed": seed,
        "resampling": "paired semantic clusters across length when complete; otherwise paired within task-length cell; tasks are not resampled",
        "joint_semantic_tasks": sorted(joint_semantic),
        "cell_paired_tasks": sorted(set(tasks) - set(joint_semantic)),
        "delta_log_auc_mean": float(samples.mean()),
        "delta_log_auc_interval95": [float(low), float(high)],
        "probability_delta_gt_zero": float(np.mean(samples > 0.0)),
    }


def compare_arms(
    panel_rows: list[dict], arms: dict[str, list[dict]], *, candidate: str,
    baselines: list[str], tasks: list[str], lengths: list[int], draws: int, seed: int,
) -> dict:
    if candidate not in arms or any(label not in arms for label in baselines):
        raise ValueError("candidate or baseline arm is missing")
    summaries = {
        label: summarize_arm(rows, panel_rows, tasks, lengths)
        for label, rows in arms.items()
    }
    contrasts = {}
    for offset, baseline in enumerate(baselines):
        candidate_summary = summaries[candidate]
        baseline_summary = summaries[baseline]
        contrasts[f"{candidate}_minus_{baseline}"] = {
            "delta_log_length_auc": (
                candidate_summary["log_length_auc"] - baseline_summary["log_length_auc"]
            ),
            "delta_by_length": {
                str(length): (
                    candidate_summary["by_length"][str(length)]["task_macro_official"]
                    - baseline_summary["by_length"][str(length)]["task_macro_official"]
                )
                for length in lengths
            },
            "delta_task_log_length_auc": {
                task: (
                    candidate_summary["task_log_length_auc"][task]
                    - baseline_summary["task_log_length_auc"][task]
                )
                for task in tasks
            },
            "bootstrap": bootstrap_contrast(
                arms[candidate], arms[baseline], tasks=tasks, lengths=lengths,
                draws=draws, seed=seed + offset,
            ),
        }
    return {
        "status": "COMPLETE",
        "candidate": candidate,
        "baselines": baselines,
        "tasks": tasks,
        "lengths": lengths,
        "summaries": summaries,
        "contrasts": contrasts,
        "metric_contract": "cell mean -> task-equal length macro -> trapezoidal log-length AUC",
        "decision_contract": "bootstrap is uncertainty evidence, not an automatic candidate gate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--source", action="append", type=parse_labeled_path, required=True)
    freeze.add_argument("--task", action="append", default=[])
    freeze.add_argument("--length", action="append", type=int, required=True)
    freeze.add_argument("--rows-per-cell", type=int, default=18)
    freeze.add_argument("--panel-id", required=True)
    freeze.add_argument("--out", type=Path, required=True)

    coverage = subparsers.add_parser("coverage")
    coverage.add_argument("--panel", type=Path, required=True)
    coverage.add_argument("--manifest", type=Path, required=True)
    coverage.add_argument("--result", action="append", required=True)
    coverage.add_argument("--arm-label", required=True)
    for field in IDENTITY_FIELDS:
        coverage.add_argument("--" + field.replace("_", "-"), required=True)
    coverage.add_argument("--merged-out", type=Path, required=True)
    coverage.add_argument("--missing-out", type=Path, required=True)
    coverage.add_argument("--receipt-out", type=Path, required=True)

    plan = subparsers.add_parser("plan-run")
    plan.add_argument("--missing-panel", type=Path, required=True)
    plan.add_argument("--data-manifest", type=Path, required=True)
    plan.add_argument("--model", type=Path, required=True)
    plan.add_argument("--base-arm", required=True)
    plan.add_argument("--label", required=True)
    plan.add_argument("--out-run", type=Path, required=True)
    plan.add_argument("--static-table-json", type=Path)
    plan.add_argument("--prefill-chunk-size", type=int, default=0)

    score = subparsers.add_parser("score")
    score.add_argument("--panel", type=Path, required=True)
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--arm", action="append", type=parse_labeled_path, required=True)
    score.add_argument("--candidate", required=True)
    score.add_argument("--baseline", action="append", required=True)
    score.add_argument("--bootstrap-draws", type=int, default=20_000)
    score.add_argument("--bootstrap-seed", type=int, default=20260913)
    score.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "freeze":
        if args.out.exists():
            raise FileExistsError(args.out)
        rows, manifest = freeze_panel(
            args.source, tasks=args.task or list(CORE6), lengths=args.length,
            rows_per_cell=args.rows_per_cell, panel_id=args.panel_id,
        )
        args.out.mkdir(parents=True)
        write_jsonl(args.out / "screen.jsonl", rows)
        manifest["panel_sha256"] = sha256_file(args.out / "screen.jsonl")
        write_json(args.out / "manifest.json", manifest)
        print(json.dumps({"status": "FROZEN", "rows": len(rows), "out": str(args.out)}, sort_keys=True))
    elif args.command == "coverage":
        panel_rows, manifest = load_frozen_panel(args.panel, args.manifest)
        identity = {field: getattr(args, field) for field in IDENTITY_FIELDS}
        identity.update(panel_id=manifest["panel_id"], panel_sha256=manifest["panel_sha256"])
        merged, missing, receipt = merge_arm(
            panel_rows, args.result, arm_label=args.arm_label, identity=identity,
        )
        write_jsonl(args.merged_out, merged)
        write_jsonl(args.missing_out, missing)
        receipt.update(
            panel_id=manifest["panel_id"], panel_sha256=manifest["panel_sha256"],
            merged_out=str(args.merged_out), missing_out=str(args.missing_out),
        )
        write_json(args.receipt_out, receipt)
        print(json.dumps({"status": receipt["status"], "covered": len(merged), "missing": len(missing)}, sort_keys=True))
    elif args.command == "plan-run":
        rows = read_jsonl(args.missing_panel)
        if not rows:
            print(json.dumps({"status": "COMPLETE_NOTHING_MISSING", "command": []}, sort_keys=True))
            return
        command = [
            sys.executable, "-m", "experiments.olmo_recovery_20260912.recovery_v2_eval",
            "--data", str(args.data_manifest), "--model", str(args.model),
            "--arm", args.base_arm, "--extra-panel", str(args.missing_panel),
            "--only-extra-panels", "--skip-lm", "--prefill-chunk-size",
            str(args.prefill_chunk_size),
        ]
        if args.static_table_json:
            command.extend(("--static-table-json", str(args.static_table_json), "--table-label", args.label))
        command.extend(("--out", str(args.out_run), "--execute"))
        print(json.dumps({
            "status": "READY_GPU", "missing_rows": len(rows), "argv": command,
            "shell": shlex.join(command),
        }, sort_keys=True))
    else:
        panel_rows, manifest = load_frozen_panel(args.panel, args.manifest)
        arms = {label: read_jsonl(path) for label, path in args.arm}
        result = compare_arms(
            panel_rows, arms, candidate=args.candidate, baselines=args.baseline,
            tasks=list(manifest["tasks"]), lengths=list(map(int, manifest["lengths"])),
            draws=args.bootstrap_draws, seed=args.bootstrap_seed,
        )
        result.update(panel_id=manifest["panel_id"], panel_sha256=manifest["panel_sha256"])
        write_json(args.out, result)
        print(json.dumps({
            "status": result["status"],
            "auc": {label: summary["log_length_auc"] for label, summary in result["summaries"].items()},
        }, sort_keys=True))


if __name__ == "__main__":
    main()
