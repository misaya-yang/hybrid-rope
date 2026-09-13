#!/usr/bin/env python3
"""Freeze, queue, resume, and score the range-optimal fixed-RoPE pipeline."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from . import PIPELINE_FORMAT, TABLE_FORMAT
from .tables import (
    find_table, model_geometry, read_json, runtime_native_inv_freq, tensor_sha256,
    validate_table,
)


DRAFT_FORMAT = "DRAFT_FIXED_ROPE_PIPELINE_V1"
STAGE_ORDER_REQUIRED_FIRST = "paper_confirm"
RESULT_TEXT_FIELDS = ("output_text", "raw_text", "output")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


@contextmanager
def exclusive_lock(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"another executor owns lock {path}; inspect before removing it") from error
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps({"pid": os.getpid(), **payload}, sort_keys=True) + "\n")
        yield
    finally:
        path.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def resolve_specs(specs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for spec in specs:
        matches = [Path(value) for value in glob.glob(str(spec), recursive=True)]
        if not matches and Path(spec).is_file():
            matches = [Path(spec)]
        for path in sorted(matches):
            if path.is_dir():
                generation = path / "generations.jsonl"
                if generation.is_file():
                    paths.append(generation)
            elif path.is_file():
                paths.append(path)
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            result.append(resolved)
            seen.add(resolved)
    return result


def result_records(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if path.suffix == ".jsonl":
            rows.extend(read_jsonl(path))
            continue
        value = json.loads(path.read_text())
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
        elif isinstance(value, dict) and any(key in value for key in ("row_id", "prompt_sha256", "prompt_input_ids_sha256")):
            rows.append(value)
    return rows


def prompt_hash(row: dict) -> str:
    value = row.get("prompt_sha256") or row.get("prompt_input_ids_sha256")
    if not value:
        raise ValueError("row lacks prompt hash")
    return str(value)


def output_text(row: dict) -> str:
    for field in RESULT_TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str):
            return value
    raise ValueError("generation row lacks raw output text")


def validate_panel_rows(rows: list[dict], panel_id: str) -> list[dict]:
    required = ("row_id", "task", "length_cap", "prompt_ids", "references", "max_new_tokens")
    seen: set[str] = set()
    normalized = []
    for row in rows:
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(f"panel {panel_id} row lacks {missing}")
        identity = prompt_hash(row)
        computed_identity = hashlib.sha256(json.dumps(
            list(row["prompt_ids"]), separators=(",", ":"),
        ).encode()).hexdigest()
        if identity != computed_identity:
            raise ValueError(f"panel {panel_id} prompt hash does not match prompt_ids")
        if identity in seen:
            raise ValueError(f"panel {panel_id} repeats prompt {identity}")
        if not row["prompt_ids"] or not row["references"] or int(row["max_new_tokens"]) <= 0:
            raise ValueError(f"panel {panel_id} has an invalid prompt/reference/budget")
        if len(row["prompt_ids"]) + int(row["max_new_tokens"]) > int(row["length_cap"]):
            raise ValueError(f"panel {panel_id} prompt exceeds its physical cap")
        seen.add(identity)
        normalized.append({
            **row,
            "prompt_sha256": identity,
            "source_panel": str(row.get("source_panel", panel_id)),
            "source_row_id": str(row.get("source_row_id", row["row_id"])),
            "mini_semantic_id": str(
                row.get("mini_semantic_id")
                or row.get("semantic_group_id")
                or row.get("source_semantic_id")
                or identity
            ),
        })
    return normalized


def covered_prompts(
    panel: list[dict], paths: Iterable[Path], *, allowed_arms: set[str] | None = None,
) -> tuple[set[str], dict]:
    allowed = {prompt_hash(row): row for row in panel}
    found: dict[str, dict] = {}
    foreign = 0
    duplicates = 0
    for row in result_records(paths):
        arm = row.get("arm") or row.get("table_id")
        if allowed_arms is not None and arm is not None and str(arm) not in allowed_arms:
            raise ValueError(f"generation row carries unexpected arm identity {arm}")
        identity = prompt_hash(row)
        if identity not in allowed:
            foreign += 1
            continue
        source = allowed[identity]
        for field in ("task", "length_cap", "references"):
            if field in row and row[field] != source[field]:
                raise ValueError(f"generation {identity} differs from its panel at {field}")
        text = output_text(row)
        previous = found.get(identity)
        if previous is not None:
            duplicates += 1
            if output_text(previous) != text:
                raise ValueError(f"conflicting duplicate generation for {identity}")
            continue
        found[identity] = row
    return set(found), {"foreign_records": foreign, "duplicate_records": duplicates}


def _resolved(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def verify_frozen_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError(f"frozen {label} changed or disappeared: {path}")


def verify_panel_file(panel: dict) -> Path:
    path = Path(panel["path"])
    verify_frozen_file(path, panel["panel_sha256"], "panel")
    return path


def verified_historical_sources(job: dict) -> list[Path]:
    paths = [Path(value) for value in job.get("result_sources", [])]
    hashes = job.get("result_source_sha256", {})
    if set(hashes) != {str(path) for path in paths}:
        raise ValueError(f"job {job['job_id']} lacks a closed historical source ledger")
    for path in paths:
        verify_frozen_file(path, hashes[str(path)], "historical result")
    for audit in job.get("reuse_audit", []):
        verify_frozen_file(Path(audit["path"]), audit["sha256"], "reuse receipt")
    return paths


def validate_result_identity(
    rows: list[dict], *, table_id: str, table: dict,
    allowed_arms: set[str], require_deployment_fields: bool,
) -> None:
    for row in rows:
        arm = row.get("arm") or row.get("table_id")
        if arm is None:
            if require_deployment_fields:
                raise ValueError("live result row lacks an arm identity")
        elif str(arm) not in allowed_arms:
            raise ValueError(f"result row carries unexpected arm identity {arm}")
        observed_hash = row.get("table_sha256_float32")
        observed_gain = row.get("gain")
        if require_deployment_fields and (observed_hash is None or observed_gain is None):
            raise ValueError("live result row lacks table hash or gain")
        if observed_hash is not None and str(observed_hash) != table["table_sha256_float32"]:
            raise ValueError(f"result row for {table_id} carries another frequency table")
        if observed_gain is not None and float(observed_gain) != float(table["gain"]):
            raise ValueError(f"result row for {table_id} carries another gain")


def validate_live_output(
    job: dict, contract: dict, run_rows: list[dict], *, contract_sha256: str,
    require_complete: bool,
) -> tuple[list[dict], bool]:
    output_dir = Path(job["output_dir"])
    generation_path = output_dir / "generations.jsonl"
    if not generation_path.exists():
        if require_complete:
            raise ValueError(f"live output is not post-run verified COMPLETE: {output_dir}")
        return [], False
    run_contract_path = output_dir / "contract.json"
    if not run_contract_path.is_file():
        raise ValueError(f"nonempty live output lacks a run contract: {output_dir}")
    run_contract = read_json(run_contract_path)
    table = contract["tables"][job["table_id"]]
    model = contract["models"][job["model_id"]]
    expected = {
        "status": "FIXED_ROPE_RESIDENT_RUN_V1",
        "pipeline_contract_sha256": contract_sha256,
        "job_id": job["job_id"],
        "stage": job["stage"],
        "model_id": job["model_id"],
        "model_config_sha256": model["config_sha256"],
        "panel_id": job["panel_id"],
        "panel_sha256": contract["panels"][job["panel_id"]]["panel_sha256"],
        "table_id": job["table_id"],
        "table_receipt_sha256": table["receipt_sha256"],
        "table_sha256_float32": table["table_sha256_float32"],
        "gain": table["gain"],
        "decoder": job["decoder"],
        "scorer": job["scorer"],
        "precision_arithmetic": model["precision_arithmetic"],
        "same_table_all_layers_and_lengths": True,
        "runtime_table_switching": False,
    }
    for field, value in expected.items():
        if run_contract.get(field) != value:
            raise ValueError(f"live run contract differs at {field}: {output_dir}")
    expected_prompts = [prompt_hash(row) for row in run_rows]
    if run_contract.get("row_prompt_sha256") != expected_prompts:
        raise ValueError(f"live run contract has another prompt sequence: {output_dir}")
    rows = read_jsonl(generation_path)
    if [prompt_hash(row) for row in rows] != expected_prompts[:len(rows)]:
        raise ValueError(f"live output is not the expected resumable prefix: {output_dir}")
    validate_result_identity(
        rows, table_id=job["table_id"], table=table,
        allowed_arms={job["table_id"]}, require_deployment_fields=True,
    )
    status_path = output_dir / "status.json"
    complete = False
    if status_path.is_file():
        status = read_json(status_path)
        complete = (
            status.get("status") == "COMPLETE"
            and int(status.get("rows", -1)) == len(run_rows) == len(rows)
            and status.get("generations_sha256") == file_sha256(generation_path)
            and status.get("run_contract_sha256") == file_sha256(run_contract_path)
            and status.get("table_sha256_float32") == table["table_sha256_float32"]
            and float(status.get("gain", float("nan"))) == float(table["gain"])
        )
        if status.get("status") == "COMPLETE" and not complete:
            raise ValueError(f"live COMPLETE receipt is inconsistent: {output_dir}")
    if require_complete and not complete:
        raise ValueError(f"live output is not post-run verified COMPLETE: {output_dir}")
    return rows, complete


def audit_reuse_receipts(
    receipt_paths: list[str], *, model: dict, panel: dict, table: dict, job: dict,
) -> list[dict]:
    """Check every identity field that a heterogeneous historical receipt exposes."""
    from .tables import tensor_sha256
    import numpy as np

    audits = []
    for raw_path in receipt_paths:
        path = Path(raw_path)
        receipt = read_json(path)
        identity = receipt.get("identity") if isinstance(receipt.get("identity"), dict) else receipt
        observed_panel = receipt.get("panel_sha256") or identity.get("panel_sha256")
        if (
            observed_panel and str(observed_panel) != panel["panel_sha256"]
            and not job.get("allow_source_panel_superset", False)
        ):
            raise ValueError(f"reuse receipt {path} belongs to another panel")
        comparisons = {
            "model_revision": model.get("revision"),
            "tokenizer_template": model.get("tokenizer_template"),
            "decoder": job.get("decoder"),
            "precision_arithmetic": model.get("precision_arithmetic"),
            "prefill_chunk_size": model.get("prefill_chunk_size"),
        }
        for field, expected in comparisons.items():
            observed = identity.get(field)
            if observed is not None and expected is not None and str(observed) != str(expected):
                raise ValueError(f"reuse receipt {path} differs at {field}")
        static_table = receipt.get("static_table")
        observed_table = identity.get("table")
        observed_gain = identity.get("gain")
        table_check = "unavailable_in_legacy_receipt"
        if isinstance(static_table, dict) and static_table.get("values_float32"):
            actual = tensor_sha256(np.asarray(static_table["values_float32"], dtype=np.float32))
            if actual != table["table_sha256_float32"]:
                raise ValueError(f"reuse receipt {path} embeds another static table")
            if float(static_table.get("gain", float("nan"))) != float(table["gain"]):
                raise ValueError(f"reuse receipt {path} embeds another attention gain")
            table_check = "exact_embedded_float32_hash"
        elif observed_table:
            short_hash = str(observed_table).rsplit(":", 1)[-1]
            if not table["table_sha256_float32"].startswith(short_hash):
                raise ValueError(f"reuse receipt {path} names another table")
            table_check = "legacy_table_hash_prefix"
        if observed_gain is not None and float(observed_gain) != float(table["gain"]):
            raise ValueError(f"reuse receipt {path} names another attention gain")
        if receipt.get("unadapted") is not None and receipt.get("unadapted") is not True:
            raise ValueError(f"reuse receipt {path} is not an unadapted frozen checkpoint run")
        if receipt.get("checkpoint_arm") is not None:
            raise ValueError(f"reuse receipt {path} includes an adapter checkpoint")
        if isinstance(static_table, dict) and receipt.get("base_arm") not in (None, "Native"):
            raise ValueError(f"reuse receipt {path} stacked a static table on a non-Native base arm")
        gain_check = "explicit"
        if not isinstance(static_table, dict) and observed_gain is None:
            if not job.get("allow_unverified_legacy_gain", False):
                raise ValueError(f"reuse receipt {path} has no verifiable gain identity")
            if not job.get("legacy_gain_justification"):
                raise ValueError("legacy gain exception requires a written justification")
            gain_check = "legacy_exception_declared"
        audits.append({
            "path": str(path),
            "sha256": file_sha256(path),
            "table_check": table_check,
            "gain_check": gain_check,
            "legacy_gain_justification": job.get("legacy_gain_justification"),
            "source_panel_superset_allowed": bool(job.get("allow_source_panel_superset", False)),
            "exposed_identity_fields": sorted(
                field for field in comparisons if identity.get(field) is not None
            ),
        })
    return audits


def freeze_contract(draft_path: Path, out: Path) -> dict:
    draft = read_json(draft_path)
    if draft.get("status") != DRAFT_FORMAT:
        raise ValueError(f"draft status must be {DRAFT_FORMAT}")
    stage_order = list(draft.get("stage_order", []))
    if not stage_order or stage_order[0] != STAGE_ORDER_REQUIRED_FIRST or len(stage_order) != len(set(stage_order)):
        raise ValueError("paper_confirm must be the first unique stage")
    models = draft.get("models") or {}
    panels = draft.get("panels") or {}
    tables = draft.get("tables") or {}
    jobs = list(draft.get("jobs") or [])
    comparisons = list(draft.get("comparisons") or [])
    if not models or not panels or not tables or not jobs or not comparisons:
        raise ValueError("models, panels, tables, jobs, and comparisons are required")

    frozen_models = {}
    for model_id, record in models.items():
        required_model_fields = ("tokenizer_template", "precision_arithmetic")
        missing_model_fields = [field for field in required_model_fields if not record.get(field)]
        if missing_model_fields:
            raise ValueError(f"model {model_id} lacks {missing_model_fields}")
        model_path = Path(record["model_path"]).expanduser().resolve()
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = read_json(config_path)
        frozen_models[model_id] = {
            **record,
            "model_path": str(model_path),
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "model_geometry": model_geometry(config),
            "prefill_chunk_size": int(record.get("prefill_chunk_size", 0)),
        }

    frozen_panels = {}
    for panel_id, record in panels.items():
        if record["model_id"] not in frozen_models:
            raise ValueError(f"panel {panel_id} names an unknown model")
        path = Path(record["path"]).expanduser().resolve()
        manifest_path = Path(record["manifest_path"]).expanduser().resolve()
        if not path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"panel or manifest missing for {panel_id}")
        rows = validate_panel_rows(read_jsonl(path), panel_id)
        manifest = read_json(manifest_path)
        recorded = manifest.get("panel_sha256")
        actual = file_sha256(path)
        if recorded and recorded != actual:
            raise ValueError(f"panel {panel_id} differs from its frozen manifest")
        if manifest.get("rows") is not None and int(manifest["rows"]) != len(rows):
            raise ValueError(f"panel {panel_id} row count differs from manifest")
        frozen_panels[panel_id] = {
            **record,
            "path": str(path),
            "manifest_path": str(manifest_path),
            "panel_sha256": actual,
            "rows": len(rows),
            "tasks": sorted({str(row["task"]) for row in rows}),
            "lengths": sorted({int(row["length_cap"]) for row in rows}),
        }

    frozen_tables = {}
    for table_id, raw_path in tables.items():
        path = Path(raw_path).expanduser().resolve()
        receipt = read_json(path)
        if receipt.get("status") != TABLE_FORMAT or receipt.get("candidate_id") != table_id:
            raise ValueError(f"table {table_id} is not a matching frozen receipt")
        model_id = str(receipt.get("model_id"))
        if model_id not in frozen_models:
            raise ValueError(f"table {table_id} names an unknown model")
        if receipt.get("model_geometry") != frozen_models[model_id]["model_geometry"]:
            raise ValueError(f"table {table_id} has another model geometry")
        nested = find_table(receipt)
        values, gain = validate_table(
            nested, pairs=int(frozen_models[model_id]["model_geometry"]["pairs"]),
        )
        if tensor_sha256(values) != receipt.get("table_sha256_float32"):
            raise ValueError(f"table {table_id} self-reported frequency hash is wrong")
        if float(gain) != float(receipt.get("gain", float("nan"))):
            raise ValueError(f"table {table_id} top-level and nested gain differ")
        if receipt.get("role") == "native" and (
            not np.array_equal(
                values, runtime_native_inv_freq(frozen_models[model_id]["model_geometry"])
            )
            or float(gain) != 1.0
        ):
            raise ValueError(f"native table {table_id} is not exact Native at gain 1")
        frozen_tables[table_id] = {
            "path": str(path),
            "receipt_sha256": file_sha256(path),
            "table_sha256_float32": receipt["table_sha256_float32"],
            "model_id": model_id,
            "role": receipt["role"],
            "gain": float(gain),
            "depth": receipt["depth"],
            "band_envelope": receipt["band_envelope"],
        }

    seen_jobs: set[str] = set()
    seen_job_pairs: set[tuple[str, str]] = set()
    seen_output_dirs: set[Path] = set()
    seen_deployments: dict[tuple, str] = {}
    frozen_jobs = []
    for record in jobs:
        job_id = str(record["job_id"])
        if job_id in seen_jobs:
            raise ValueError(f"duplicate job {job_id}")
        seen_jobs.add(job_id)
        stage = str(record["stage"])
        model_id = str(record["model_id"])
        panel_id = str(record["panel_id"])
        table_id = str(record["table_id"])
        if stage not in stage_order or model_id not in frozen_models or panel_id not in frozen_panels or table_id not in frozen_tables:
            raise ValueError(f"job {job_id} references an unknown object")
        if frozen_panels[panel_id]["model_id"] != model_id or frozen_tables[table_id]["model_id"] != model_id:
            raise ValueError(f"job {job_id} crosses model identities")
        output_dir = Path(record["output_dir"]).expanduser().resolve()
        pair = (panel_id, table_id)
        if pair in seen_job_pairs:
            raise ValueError(f"duplicate panel/table job pair {pair}")
        if output_dir in seen_output_dirs:
            raise ValueError(f"multiple jobs share output directory {output_dir}")
        seen_job_pairs.add(pair)
        seen_output_dirs.add(output_dir)
        source_specs = [_resolved(value) for value in record.get("result_sources", [])]
        resolved_sources = resolve_specs(source_specs)
        sources = [str(path) for path in resolved_sources]
        reuse_receipts = [_resolved(value) for value in record.get("reuse_receipts", [])]
        if bool(sources) != bool(reuse_receipts):
            raise ValueError(f"job {job_id} must pair reused results with explicit receipts")
        if any(not Path(value).is_file() for value in reuse_receipts):
            raise FileNotFoundError(f"job {job_id} has a missing reuse receipt")
        if source_specs and not sources:
            raise FileNotFoundError(f"job {job_id} result source patterns match no files")
        if len(sources) != len(reuse_receipts):
            raise ValueError(f"job {job_id} must map each historical result to one receipt")
        for source, receipt in zip(map(Path, sources), map(Path, reuse_receipts)):
            if source.parent != receipt.parent and source.parent.parent != receipt.parent:
                raise ValueError(f"job {job_id} result/receipt parents do not correspond")
        source_arm_labels = {str(value) for value in record.get("source_arm_labels", [])}
        if sources and not source_arm_labels:
            raise ValueError(f"job {job_id} must declare accepted historical arm labels")
        reuse_audit = audit_reuse_receipts(
            reuse_receipts,
            model=frozen_models[model_id], panel=frozen_panels[panel_id],
            table=frozen_tables[table_id], job=record,
        )
        deployment = (
            model_id, panel_id, frozen_tables[table_id]["table_sha256_float32"],
            float(frozen_tables[table_id]["gain"]).hex(), str(record["decoder"]),
            frozen_models[model_id]["precision_arithmetic"],
        )
        previous_deployment = seen_deployments.get(deployment)
        if previous_deployment is not None:
            raise ValueError(
                f"jobs {previous_deployment} and {job_id} are duplicate deployments on one panel"
            )
        seen_deployments[deployment] = job_id
        frozen_jobs.append({
            **record,
            "job_id": job_id,
            "stage": stage,
            "model_id": model_id,
            "panel_id": panel_id,
            "table_id": table_id,
            "priority": int(record.get("priority", 0)),
            "result_sources": sources,
            "result_source_sha256": {
                str(path): file_sha256(path) for path in resolved_sources
            },
            "reuse_receipts": reuse_receipts,
            "reuse_pairs": [
                {"result": source, "receipt": receipt}
                for source, receipt in zip(sources, reuse_receipts)
            ],
            "reuse_audit": reuse_audit,
            "source_arm_labels": sorted(source_arm_labels),
            "output_dir": str(output_dir),
        })

    seen_comparisons: set[str] = set()
    for record in comparisons:
        comparison_id = str(record["comparison_id"])
        if comparison_id in seen_comparisons:
            raise ValueError(f"duplicate comparison {comparison_id}")
        seen_comparisons.add(comparison_id)
        panel_id = str(record["panel_id"])
        named_tables = [
            str(record["candidate"]), *map(str, record["baselines"]),
            *map(str, record.get("diagnostic_controls", [])),
        ]
        if panel_id not in frozen_panels or any(table_id not in frozen_tables for table_id in named_tables):
            raise ValueError(f"comparison {comparison_id} references an unknown object")
        required_pairs = {(panel_id, table_id) for table_id in named_tables}
        available_pairs = {(job["panel_id"], job["table_id"]) for job in frozen_jobs}
        if not required_pairs <= available_pairs:
            raise ValueError(f"comparison {comparison_id} lacks one or more jobs")

    frozen = {
        **draft,
        "status": PIPELINE_FORMAT,
        "source_draft": str(Path(draft_path).resolve()),
        "stage_order": stage_order,
        "models": frozen_models,
        "panels": frozen_panels,
        "tables": frozen_tables,
        "jobs": frozen_jobs,
        "comparisons": comparisons,
        "scientific_contract": {
            "primary_goal": "one fixed table maximizes useful quality across [L,S L]",
            "paper_confirmation_before_mechanism_sweeps": True,
            "three_interfaces_are_identification_coordinates": True,
            "score_gates_disabled": True,
            "baseline_reuse_by_exact_benchmark_identity": True,
        },
    }
    if out.exists():
        existing = read_json(out)
        if existing == frozen:
            return existing
        raise ValueError(f"refusing to overwrite a different frozen pipeline contract: {out}")
    atomic_json(out, frozen)
    return frozen


def plan_queue(contract_path: Path, out: Path) -> dict:
    contract = read_json(contract_path)
    if contract.get("status") != PIPELINE_FORMAT:
        raise ValueError("pipeline contract is not frozen")
    contract_sha256 = file_sha256(Path(contract_path))
    out.mkdir(parents=True, exist_ok=True)
    stage_rank = {stage: index for index, stage in enumerate(contract["stage_order"])}
    queue = []
    coverage = []
    for job in contract["jobs"]:
        panel_id = job["panel_id"]
        panel_record = contract["panels"][panel_id]
        panel_path = verify_panel_file(panel_record)
        model_record = contract["models"][job["model_id"]]
        table_record = contract["tables"][job["table_id"]]
        verify_frozen_file(Path(model_record["config_path"]), model_record["config_sha256"], "model config")
        verify_frozen_file(Path(table_record["path"]), table_record["receipt_sha256"], "table receipt")
        panel = validate_panel_rows(read_jsonl(panel_path), panel_id)
        sources = verified_historical_sources(job)
        historical_arms = set(job.get("source_arm_labels", []))
        historical_rows = result_records(sources)
        validate_result_identity(
            historical_rows, table_id=job["table_id"], table=table_record,
            allowed_arms=historical_arms, require_deployment_fields=False,
        )
        historically_covered, historical_diagnostics = covered_prompts(
            panel, sources, allowed_arms=historical_arms if sources else None,
        )
        run_rows = [row for row in panel if prompt_hash(row) not in historically_covered]
        live = Path(job["output_dir"]) / "generations.jsonl"
        live_rows, live_complete = validate_live_output(
            job, contract, run_rows, contract_sha256=contract_sha256, require_complete=False,
        )
        job_complete = not run_rows or live_complete
        coverage_sources = list(sources)
        if live.is_file():
            coverage_sources.append(live.resolve())
        covered, diagnostics = covered_prompts(
            panel, coverage_sources,
            allowed_arms=historical_arms | {job["table_id"]},
        )
        missing = [row for row in panel if prompt_hash(row) not in covered]
        missing_path = out / "missing" / job["model_id"] / f"{job['job_id']}.jsonl"
        atomic_jsonl(missing_path, run_rows)
        record = {
            "job_id": job["job_id"],
            "stage": job["stage"],
            "model_id": job["model_id"],
            "panel_id": panel_id,
            "table_id": job["table_id"],
            "covered_rows": len(covered),
            "missing_rows": len(missing),
            "resident_input_rows": len(run_rows),
            "resident_saved_prefix_rows": len(live_rows),
            "post_run_verified_complete": job_complete,
            "panel_rows": len(panel),
            "coverage_by_cell": dict(sorted(Counter(
                f"{row['task']}/{row['length_cap']}" for row in panel if prompt_hash(row) in covered
            ).items())),
            "missing_by_cell": dict(sorted(Counter(
                f"{row['task']}/{row['length_cap']}" for row in missing
            ).items())),
            "result_sources": [str(path) for path in coverage_sources],
            "reuse_receipts": job.get("reuse_receipts", []),
            "historical_coverage": {
                "covered_rows": len(historically_covered),
                **historical_diagnostics,
            },
            **diagnostics,
        }
        coverage.append(record)
        if missing or not job_complete:
            queue.append({
                **job,
                "missing_panel": str(missing_path.resolve()),
                "resident_input_sha256": file_sha256(missing_path),
                "missing_rows": len(run_rows),
                "remaining_rows": len(missing),
                "saved_prefix_rows": len(live_rows),
                "table_path": contract["tables"][job["table_id"]]["path"],
                "panel_sha256": panel_record["panel_sha256"],
            })
    queue.sort(key=lambda row: (
        stage_rank[row["stage"]], int(row["priority"]), row["model_id"], row["job_id"]
    ))
    result = {
        "status": "READY_GPU" if queue else "COMPLETE_NOTHING_MISSING",
        "contract_path": str(Path(contract_path).resolve()),
        "contract_sha256": contract_sha256,
        "queue": queue,
        "coverage": coverage,
        "rule": "only missing prompt hashes are queued; job order starts with paper confirmation",
    }
    atomic_json(out / "queue.json", result)
    coverage_before = out / "coverage_before.json"
    if not coverage_before.exists():
        atomic_json(coverage_before, {"coverage": coverage})
    return result


def _jobs_for_comparison(contract: dict, comparison: dict) -> dict[str, dict]:
    panel_id = comparison["panel_id"]
    wanted = {
        comparison["candidate"], *comparison["baselines"],
        *comparison.get("diagnostic_controls", []),
    }
    matches = {
        job["table_id"]: job for job in contract["jobs"]
        if job["panel_id"] == panel_id and job["table_id"] in wanted
    }
    if set(matches) != wanted:
        raise ValueError("comparison jobs are incomplete")
    return matches


def summarize_point(rows: list[dict], tasks: list[str], length: int) -> dict:
    import numpy as np

    cells = {
        task: [row for row in rows if row["task"] == task and int(row["length_cap"]) == length]
        for task in tasks
    }
    if any(not values for values in cells.values()):
        raise ValueError("point comparison has an empty task cell")
    by_task = {
        task: {
            "rows": len(values),
            "official": float(np.mean([row["official_score"] for row in values])),
            "eos_rate": float(np.mean([row["ended_eos"] for row in values])),
            "cap_rate": float(np.mean([row["hit_cap"] for row in values])),
        }
        for task, values in cells.items()
    }
    macro = float(np.mean([by_task[task]["official"] for task in tasks]))
    return {
        "by_length": {
            str(length): {
                "task_macro_official": macro,
                "task_macro_eos_rate": float(np.mean([by_task[task]["eos_rate"] for task in tasks])),
                "task_macro_cap_rate": float(np.mean([by_task[task]["cap_rate"] for task in tasks])),
                "tasks": by_task,
            }
        },
        "task_macro_official": macro,
        "interval_min": macro,
        "task_log_length_auc": {task: by_task[task]["official"] for task in tasks},
    }


def bootstrap_point_contrast(
    candidate: list[dict], baseline: list[dict], *, tasks: list[str], draws: int, seed: int,
) -> dict:
    import numpy as np

    candidate_by_prompt = {row["prompt_sha256"]: row for row in candidate}
    baseline_by_prompt = {row["prompt_sha256"]: row for row in baseline}
    if set(candidate_by_prompt) != set(baseline_by_prompt):
        raise ValueError("point-comparison arms are not row paired")
    task_prompts = {
        task: [row["prompt_sha256"] for row in candidate if row["task"] == task]
        for task in tasks
    }
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        task_deltas = []
        for task, prompts in task_prompts.items():
            indices = rng.integers(0, len(prompts), size=len(prompts))
            selected = [prompts[int(index)] for index in indices]
            task_deltas.append(float(np.mean([
                candidate_by_prompt[prompt]["official_score"]
                - baseline_by_prompt[prompt]["official_score"]
                for prompt in selected
            ])))
        samples[draw] = float(np.mean(task_deltas))
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "draws": draws,
        "seed": seed,
        "resampling": "paired rows within task at one registered length; tasks fixed",
        "delta_task_macro_mean": float(samples.mean()),
        "delta_task_macro_interval95": [float(low), float(high)],
        "probability_delta_gt_zero": float(np.mean(samples > 0.0)),
    }


def bootstrap_range_contrast(
    candidate: list[dict], baseline: list[dict], *, tasks: list[str], lengths: list[int],
    task_families: dict[str, str], draws: int, seed: int,
) -> dict:
    import math
    import numpy as np

    def auc(curve: dict[int, float]) -> float:
        numerator = sum(
            0.5 * (curve[left] + curve[right]) * (math.log(right) - math.log(left))
            for left, right in zip(lengths, lengths[1:])
        )
        return numerator / (math.log(lengths[-1]) - math.log(lengths[0]))

    candidate_by_prompt = {row["prompt_sha256"]: row for row in candidate}
    baseline_by_prompt = {row["prompt_sha256"]: row for row in baseline}
    if set(candidate_by_prompt) != set(baseline_by_prompt):
        raise ValueError("range-comparison arms are not row paired")
    cells: dict[tuple[str, int], list[str]] = {}
    semantics: dict[tuple[str, int], dict[str, str]] = {}
    for task in tasks:
        for length in lengths:
            rows = [
                row for row in candidate
                if row["task"] == task and int(row["length_cap"]) == length
            ]
            cells[(task, length)] = [row["prompt_sha256"] for row in rows]
            semantics[(task, length)] = {
                row["mini_semantic_id"]: row["prompt_sha256"]
                for row in rows
                if row.get("mini_semantic_id") and row["mini_semantic_id"] != row["prompt_sha256"]
            }
    joint_semantic = {}
    for task in tasks:
        mappings = [semantics[(task, length)] for length in lengths]
        if mappings and all(mappings) and all(set(value) == set(mappings[0]) for value in mappings[1:]):
            joint_semantic[task] = sorted(mappings[0])
    families: dict[str, list[str]] = {}
    for task in tasks:
        families.setdefault(task_families[task], []).append(task)
    rng = np.random.default_rng(seed)
    delta_auc = np.empty(draws)
    candidate_worst = np.empty(draws)
    baseline_worst = np.empty(draws)
    candidate_argmin = np.empty(draws, dtype=np.int64)
    baseline_argmin = np.empty(draws, dtype=np.int64)
    delta_by_length = np.empty((draws, len(lengths)))
    delta_family_auc = {family: np.empty(draws) for family in families}
    for draw in range(draws):
        joint_draws = {
            task: rng.integers(0, len(ids), size=len(ids))
            for task, ids in joint_semantic.items()
        }
        candidate_tasks: dict[str, dict[int, float]] = {task: {} for task in tasks}
        baseline_tasks: dict[str, dict[int, float]] = {task: {} for task in tasks}
        for task in tasks:
            for length in lengths:
                if task in joint_semantic:
                    selected = [
                        semantics[(task, length)][joint_semantic[task][int(index)]]
                        for index in joint_draws[task]
                    ]
                else:
                    prompts = cells[(task, length)]
                    indices = rng.integers(0, len(prompts), size=len(prompts))
                    selected = [prompts[int(index)] for index in indices]
                candidate_tasks[task][length] = float(np.mean([
                    candidate_by_prompt[prompt]["official_score"] for prompt in selected
                ]))
                baseline_tasks[task][length] = float(np.mean([
                    baseline_by_prompt[prompt]["official_score"] for prompt in selected
                ]))
        candidate_curve = {
            length: float(np.mean([candidate_tasks[task][length] for task in tasks]))
            for length in lengths
        }
        baseline_curve = {
            length: float(np.mean([baseline_tasks[task][length] for task in tasks]))
            for length in lengths
        }
        delta_auc[draw] = auc(candidate_curve) - auc(baseline_curve)
        candidate_worst[draw] = min(candidate_curve.values())
        baseline_worst[draw] = min(baseline_curve.values())
        candidate_argmin[draw] = lengths[int(np.argmin([candidate_curve[length] for length in lengths]))]
        baseline_argmin[draw] = lengths[int(np.argmin([baseline_curve[length] for length in lengths]))]
        delta_by_length[draw] = [
            candidate_curve[length] - baseline_curve[length] for length in lengths
        ]
        for family, family_tasks in families.items():
            candidate_family = {
                length: float(np.mean([candidate_tasks[task][length] for task in family_tasks]))
                for length in lengths
            }
            baseline_family = {
                length: float(np.mean([baseline_tasks[task][length] for task in family_tasks]))
                for length in lengths
            }
            delta_family_auc[family][draw] = auc(candidate_family) - auc(baseline_family)

    def interval(values) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    observed_delta_by_length = []
    for length in lengths:
        task_deltas = []
        for task in tasks:
            prompts = cells[(task, length)]
            task_deltas.append(float(np.mean([
                candidate_by_prompt[prompt]["official_score"]
                - baseline_by_prompt[prompt]["official_score"]
                for prompt in prompts
            ])))
        observed_delta_by_length.append(float(np.mean(task_deltas)))
    centered = delta_by_length - np.asarray(observed_delta_by_length)[None, :]
    simultaneous_halfwidth = float(np.quantile(np.max(np.abs(centered), axis=1), 0.95))
    return {
        "draws": draws,
        "seed": seed,
        "resampling": "paired semantic clusters across length when complete; otherwise paired within task-length; tasks fixed",
        "joint_semantic_tasks": sorted(joint_semantic),
        "delta_log_auc": {"mean": float(delta_auc.mean()), "interval95": interval(delta_auc)},
        "candidate_worst_length_score": {
            "mean": float(candidate_worst.mean()), "interval95": interval(candidate_worst),
            "recomputed_inside_each_draw": True,
        },
        "baseline_worst_length_score": {
            "mean": float(baseline_worst.mean()), "interval95": interval(baseline_worst),
            "recomputed_inside_each_draw": True,
        },
        "delta_worst_length_score": {
            "mean": float((candidate_worst - baseline_worst).mean()),
            "interval95": interval(candidate_worst - baseline_worst),
        },
        "candidate_argmin_length_frequency": {
            str(length): float(np.mean(candidate_argmin == length)) for length in lengths
        },
        "baseline_argmin_length_frequency": {
            str(length): float(np.mean(baseline_argmin == length)) for length in lengths
        },
        "delta_by_length_interval95": {
            str(length): interval(delta_by_length[:, index])
            for index, length in enumerate(lengths)
        },
        "delta_curve_simultaneous95_halfwidth": simultaneous_halfwidth,
        "delta_family_log_auc": {
            family: {"mean": float(values.mean()), "interval95": interval(values)}
            for family, values in delta_family_auc.items()
        },
        "probability_delta_auc_gt_zero": float(np.mean(delta_auc > 0.0)),
    }


def score_comparison(
    contract_path: Path, comparison_id: str, out: Path, *, draws: int, seed: int,
) -> dict:
    import experiments.olmo_recovery_20260912.permanent_mini_pipeline as mini_scoring
    import scripts.experiments.olmo_fast_screen.ruler_bench as ruler_scoring

    merge_arm = mini_scoring.merge_arm
    summarize_arm = mini_scoring.summarize_arm

    contract = read_json(contract_path)
    if contract.get("status") != PIPELINE_FORMAT:
        raise ValueError("pipeline contract is not frozen")
    contract_sha256 = file_sha256(Path(contract_path))
    comparison = next(
        (item for item in contract["comparisons"] if item["comparison_id"] == comparison_id),
        None,
    )
    if comparison is None:
        raise ValueError(f"unknown comparison {comparison_id}")
    panel_id = comparison["panel_id"]
    panel_record = contract["panels"][panel_id]
    panel_path = verify_panel_file(panel_record)
    panel = validate_panel_rows(read_jsonl(panel_path), panel_id)
    jobs = _jobs_for_comparison(contract, comparison)
    arms = {}
    coverages = {}
    input_ledger = {}
    for table_id, job in jobs.items():
        table = contract["tables"][table_id]
        model = contract["models"][job["model_id"]]
        verify_frozen_file(Path(model["config_path"]), model["config_sha256"], "model config")
        verify_frozen_file(Path(table["path"]), table["receipt_sha256"], "table receipt")
        historical_paths = verified_historical_sources(job)
        historical_rows = result_records(historical_paths)
        historical_arms = set(job.get("source_arm_labels", []))
        validate_result_identity(
            historical_rows, table_id=table_id, table=table,
            allowed_arms=historical_arms, require_deployment_fields=False,
        )
        historically_covered, _ = covered_prompts(
            panel, historical_paths, allowed_arms=historical_arms if historical_paths else None,
        )
        run_rows = [row for row in panel if prompt_hash(row) not in historically_covered]
        live_rows, live_complete = validate_live_output(
            job, contract, run_rows, contract_sha256=contract_sha256,
            require_complete=bool(run_rows),
        )
        paths = list(historical_paths)
        if run_rows:
            paths.append((Path(job["output_dir"]) / "generations.jsonl").resolve())
        validate_result_identity(
            [*historical_rows, *live_rows], table_id=table_id, table=table,
            allowed_arms=historical_arms | {table_id}, require_deployment_fields=False,
        )
        input_ledger[table_id] = {
            "historical": {str(path): file_sha256(path) for path in historical_paths},
            "live": (
                {str(paths[-1]): file_sha256(paths[-1])} if run_rows and live_complete else {}
            ),
        }
        identity = {
            "model_revision": model.get("revision", model["config_sha256"]),
            "tokenizer_template": model["tokenizer_template"],
            "table": table["table_sha256_float32"],
            "decoder": job["decoder"],
            "scorer": job["scorer"],
            "precision_arithmetic": model["precision_arithmetic"],
        }
        merged, missing, coverage = merge_arm(
            panel, [str(path) for path in paths], arm_label=table_id, identity=identity,
        )
        coverages[table_id] = coverage
        if missing:
            raise ValueError(f"comparison {comparison_id} is missing {len(missing)} rows for {table_id}")
        arms[table_id] = merged
    tasks = panel_record["tasks"]
    lengths = panel_record["lengths"]
    task_families = {}
    for task in tasks:
        values = {str(row.get("family") or task) for row in panel if row["task"] == task}
        if len(values) != 1:
            raise ValueError(f"task {task} has inconsistent family labels")
        task_families[task] = values.pop()
    score_request = {
        "pipeline_contract_sha256": contract_sha256,
        "comparison_id": comparison_id,
        "panel_sha256": panel_record["panel_sha256"],
        "input_ledger": input_ledger,
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "scoring_source_sha256": {
            "permanent_mini_pipeline.py": file_sha256(Path(mini_scoring.__file__)),
            "ruler_bench.py": file_sha256(Path(ruler_scoring.__file__)),
            "fixed_rope_pipeline.py": file_sha256(Path(__file__)),
        },
    }
    existing_score = out / "comparison.json"
    if existing_score.exists():
        existing = read_json(existing_score)
        required_artifacts = (out / "cell_scores.csv", out / "interval_summary.csv")
        artifact_hashes = existing.get("artifact_sha256", {})
        artifacts_hold = all(
            path.is_file() and artifact_hashes.get(path.name) == file_sha256(path)
            for path in required_artifacts
        )
        if existing.get("score_request") == score_request and artifacts_hold:
            return existing
        raise ValueError(f"refusing to overwrite a score with another frozen input identity: {out}")
    if len(lengths) == 1:
        summaries = {label: summarize_point(rows, tasks, lengths[0]) for label, rows in arms.items()}
        contrasts = {}
        contrast_arms = [*comparison["baselines"], *comparison.get("diagnostic_controls", [])]
        for offset, baseline in enumerate(contrast_arms):
            candidate_summary = summaries[comparison["candidate"]]
            baseline_summary = summaries[baseline]
            contrasts[f"{comparison['candidate']}_minus_{baseline}"] = {
                "delta_task_macro_official": (
                    candidate_summary["task_macro_official"]
                    - baseline_summary["task_macro_official"]
                ),
                "delta_by_task": {
                    task: (
                        candidate_summary["task_log_length_auc"][task]
                        - baseline_summary["task_log_length_auc"][task]
                    ) for task in tasks
                },
                "bootstrap": bootstrap_point_contrast(
                    arms[comparison["candidate"]], arms[baseline], tasks=tasks,
                    draws=draws, seed=seed + offset,
                ),
            }
        result = {
            "status": "COMPLETE",
            "candidate": comparison["candidate"],
            "baselines": list(comparison["baselines"]),
            "diagnostic_controls": list(comparison.get("diagnostic_controls", [])),
            "tasks": tasks,
            "lengths": lengths,
            "summaries": summaries,
            "contrasts": contrasts,
            "metric_contract": "one registered length: cell mean -> task-equal macro",
            "decision_contract": "bootstrap is uncertainty evidence, not an automatic gate",
        }
    else:
        summaries = {
            label: summarize_arm(rows, panel, tasks, lengths)
            for label, rows in arms.items()
        }
        for summary in summaries.values():
            summary["family_log_length_auc"] = {
                family: sum(summary["task_log_length_auc"][task] for task in family_tasks)
                / len(family_tasks)
                for family, family_tasks in {
                    name: [task for task in tasks if task_families[task] == name]
                    for name in sorted(set(task_families.values()))
                }.items()
            }
        contrasts = {}
        contrast_arms = [*comparison["baselines"], *comparison.get("diagnostic_controls", [])]
        for offset, baseline in enumerate(contrast_arms):
            candidate_summary = summaries[comparison["candidate"]]
            baseline_summary = summaries[baseline]
            contrasts[f"{comparison['candidate']}_minus_{baseline}"] = {
                "delta_log_length_auc": candidate_summary["log_length_auc"] - baseline_summary["log_length_auc"],
                "delta_by_length": {
                    str(length): (
                        candidate_summary["by_length"][str(length)]["task_macro_official"]
                        - baseline_summary["by_length"][str(length)]["task_macro_official"]
                    ) for length in lengths
                },
                "delta_task_log_length_auc": {
                    task: candidate_summary["task_log_length_auc"][task] - baseline_summary["task_log_length_auc"][task]
                    for task in tasks
                },
                "delta_family_log_length_auc": {
                    family: candidate_summary["family_log_length_auc"][family] - baseline_summary["family_log_length_auc"][family]
                    for family in candidate_summary["family_log_length_auc"]
                },
                "bootstrap": bootstrap_range_contrast(
                    arms[comparison["candidate"]], arms[baseline], tasks=tasks,
                    lengths=lengths, task_families=task_families,
                    draws=draws, seed=seed + offset,
                ),
            }
        result = {
            "status": "COMPLETE",
            "candidate": comparison["candidate"],
            "baselines": list(comparison["baselines"]),
            "diagnostic_controls": list(comparison.get("diagnostic_controls", [])),
            "tasks": tasks,
            "task_families": task_families,
            "lengths": lengths,
            "summaries": summaries,
            "contrasts": contrasts,
            "metric_contract": "cell mean -> task-equal length macro -> trapezoidal log-length AUC",
            "decision_contract": "bootstrap is uncertainty evidence, not an automatic candidate gate",
        }
    result.update(
        comparison_id=comparison_id,
        panel_id=panel_id,
        panel_sha256=panel_record["panel_sha256"],
        coverage=coverages,
        input_ledger=input_ledger,
        pipeline_contract_sha256=contract_sha256,
        scope="finite registered task/length grid; one fixed table per arm",
        score_request=score_request,
    )
    baselines = list(comparison["baselines"])
    envelope = {
        length: max(
            result["summaries"][baseline]["by_length"][str(length)]["task_macro_official"]
            for baseline in baselines
        )
        for length in lengths
    }
    candidate_summary = result["summaries"][comparison["candidate"]]
    candidate_curve = {
        length: candidate_summary["by_length"][str(length)]["task_macro_official"]
        for length in lengths
    }
    result["candidate_range_diagnostics"] = {
        "worst_regret_vs_pointwise_baseline_envelope": max(
            envelope[length] - candidate_curve[length] for length in lengths
        ),
        "regret_by_length": {
            str(length): envelope[length] - candidate_curve[length] for length in lengths
        },
        "native_point": candidate_curve[lengths[0]],
        "endpoint": candidate_curve[lengths[-1]],
        "worst_length_tokens": min(lengths, key=lambda length: candidate_curve[length]),
        "worst_length_score": min(candidate_curve.values()),
    }
    out.mkdir(parents=True, exist_ok=True)
    cell_path = out / "cell_scores.csv"
    cell_temporary = cell_path.with_name(cell_path.name + ".incomplete")
    with cell_temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "length", "task", "rows", "official", "eos_rate", "cap_rate"])
        for arm, summary in result["summaries"].items():
            for length in lengths:
                for task, cell in summary["by_length"][str(length)]["tasks"].items():
                    writer.writerow([
                        arm, length, task, cell["rows"], cell["official"],
                        cell["eos_rate"], cell["cap_rate"],
                    ])
    cell_temporary.replace(cell_path)
    interval_path = out / "interval_summary.csv"
    interval_temporary = interval_path.with_name(interval_path.name + ".incomplete")
    with interval_temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "log_length_auc", "worst_length_score", "native_point", "endpoint"])
        for arm, summary in result["summaries"].items():
            writer.writerow([
                arm, summary.get("log_length_auc"), summary["interval_min"],
                summary["by_length"][str(lengths[0])]["task_macro_official"],
                summary["by_length"][str(lengths[-1])]["task_macro_official"],
            ])
    interval_temporary.replace(interval_path)
    result["artifact_sha256"] = {
        cell_path.name: file_sha256(cell_path),
        interval_path.name: file_sha256(interval_path),
    }
    atomic_json(out / "comparison.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--draft", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--contract", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--contract", type=Path, required=True)
    score.add_argument("--comparison-id", required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--bootstrap-draws", type=int, default=20_000)
    score.add_argument("--bootstrap-seed", type=int, default=20260913)

    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_contract(args.draft, args.out)
        print(json.dumps({"status": result["status"], "out": str(args.out)}, sort_keys=True))
    elif args.command == "plan":
        result = plan_queue(args.contract, args.out)
        print(json.dumps({
            "status": result["status"], "jobs": len(result["queue"]),
            "remaining_rows": sum(job["remaining_rows"] for job in result["queue"]),
        }, sort_keys=True))
    else:
        result = score_comparison(
            args.contract, args.comparison_id, args.out,
            draws=args.bootstrap_draws, seed=args.bootstrap_seed,
        )
        print(json.dumps({
            "status": result["status"], "comparison_id": args.comparison_id,
            "candidate": result["candidate"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
