#!/usr/bin/env python3
"""Run queued fixed RoPE tables sequentially with one resident model."""
from __future__ import annotations

import argparse
import atexit
import json
from pathlib import Path
import time

from . import PIPELINE_FORMAT, TABLE_FORMAT
from .pipeline import (
    atomic_json, exclusive_lock, file_sha256, prompt_hash, read_jsonl, validate_live_output,
    validate_panel_rows, verify_frozen_file,
)
from .tables import find_table, model_geometry, read_json, tensor_sha256, validate_table


def normalized(text: str) -> str:
    return text.strip().lower().strip(" .,!;:\"'`\n\t")


def validate_saved_prefix(saved: list[dict], rows: list[dict]) -> None:
    if len(saved) > len(rows):
        raise ValueError("saved output is longer than the queued missing panel")
    for index, record in enumerate(saved):
        if prompt_hash(record) != prompt_hash(rows[index]):
            raise ValueError("saved output is not the expected prompt prefix")


def jobs_for_model(queue: dict, model_id: str, stages: set[str] | None) -> list[dict]:
    jobs = [job for job in queue.get("queue", []) if job["model_id"] == model_id]
    if stages:
        jobs = [job for job in jobs if job["stage"] in stages]
    return jobs


def load_static_receipt(path: Path, *, model_id: str, pairs: int) -> tuple[dict, dict]:
    receipt = read_json(path)
    if receipt.get("status") != TABLE_FORMAT or receipt.get("model_id") != model_id:
        raise ValueError("static table receipt has the wrong format or model identity")
    table = find_table(receipt)
    values, gain = validate_table(table, pairs=pairs)
    if tensor_sha256(values) != receipt.get("table_sha256_float32"):
        raise ValueError("static table bytes differ from their frozen receipt")
    return receipt, {"values_float32": values, "gain": gain, "construction": table.get("construction", {})}


def deployment_cache_key(
    *, table_hash: str, gain: float, decoder: str, prompt: str, max_new_tokens: int,
) -> tuple[str, str, str, str, int]:
    return table_hash, float(gain).hex(), decoder, prompt, int(max_new_tokens)


def cache_record(
    cache: dict[tuple[str, str, str, str, int], dict], record: dict, *,
    expected_table_hash: str, expected_gain: float, decoder: str, max_new_tokens: int,
) -> None:
    observed_hash = record.get("table_sha256_float32")
    observed_gain = record.get("gain")
    if observed_hash != expected_table_hash:
        raise ValueError("saved generation carries another table hash")
    if observed_gain is None or float(observed_gain) != float(expected_gain):
        raise ValueError("saved generation carries another or missing gain")
    key = deployment_cache_key(
        table_hash=expected_table_hash, gain=expected_gain, decoder=decoder,
        prompt=prompt_hash(record), max_new_tokens=max_new_tokens,
    )
    previous = cache.get(key)
    if previous is not None:
        comparable = ("generated_ids", "output_text", "ended_eos", "ruler_official_score")
        if any(previous.get(field) != record.get(field) for field in comparable):
            raise ValueError("same table/prompt has conflicting resident outputs")
    else:
        cache[key] = record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    queue = read_json(args.queue)
    contract_path = Path(queue["contract_path"])
    if file_sha256(contract_path) != queue.get("contract_sha256"):
        raise ValueError("frozen pipeline contract changed after queue planning")
    contract = read_json(contract_path)
    if contract.get("status") != PIPELINE_FORMAT:
        raise ValueError("pipeline contract is not frozen")
    if args.model_id not in contract["models"]:
        raise ValueError(f"unknown model {args.model_id}")
    all_model_jobs = jobs_for_model(queue, args.model_id, None)
    jobs = jobs_for_model(queue, args.model_id, set(args.stage) if args.stage else None)
    plan = {
        "status": "READY_GPU" if jobs else "COMPLETE_NOTHING_MISSING",
        "model_id": args.model_id,
        "jobs": [{
            "job_id": job["job_id"], "stage": job["stage"],
            "table_id": job["table_id"], "resident_input_rows": job["missing_rows"],
            "saved_prefix_rows": job.get("saved_prefix_rows", 0),
            "remaining_rows": job.get("remaining_rows", job["missing_rows"]),
        } for job in jobs],
        "one_model_load": True,
        "sequential_static_tables": True,
    }
    if not args.execute or not jobs:
        print(json.dumps(plan, sort_keys=True))
        return

    lock = exclusive_lock(
        args.queue.resolve().parent / f".resident_{args.model_id}.lock",
        {"model_id": args.model_id, "queue": str(args.queue.resolve())},
    )
    lock.__enter__()
    release_lock = lambda: lock.__exit__(None, None, None)
    atexit.register(release_lock)

    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from experiments.olmo_recovery_20260912.recovery_v2_eval import greedy_tokens
    from experiments.olmo_recovery_20260912.recovery_v2_runtime import load_model
    from experiments.olmo_recovery_20260912.runtime import validate_cuda
    from scripts.eval.longbench_metrics import qa_f1_score
    from scripts.experiments.cross_audit.tables import install_static, verify_static
    from scripts.experiments.olmo_fast_screen.ruler_bench import score as ruler_score

    validate_cuda()
    model_record = contract["models"][args.model_id]
    model_path = Path(model_record["model_path"])
    verify_frozen_file(
        Path(model_record["config_path"]), model_record["config_sha256"], "model config",
    )
    model, wrapper, _ = load_model(model_path, "Native", checkpoint=None, training=False)
    if wrapper is not None:
        raise RuntimeError("frozen resident evaluation unexpectedly created an adapter")
    model.eval()
    if model.training:
        raise RuntimeError("resident evaluator failed to enter eval mode")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    runtime_geometry = model_geometry(model.config.to_dict())
    if runtime_geometry != model_geometry(read_json(Path(model_record["config_path"]))):
        raise RuntimeError("loaded model RoPE geometry differs from the frozen config")
    eos_value = model.generation_config.eos_token_id
    eos_ids = set(eos_value if isinstance(eos_value, list) else [eos_value])
    eos_ids.discard(None)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = next(iter(eos_ids), 0)
    pairs = int(model.config.hidden_size // model.config.num_attention_heads // 2)
    device = next(model.parameters()).device
    generation_cache: dict[tuple[str, str, str, str, int], dict] = {}
    for queued_job in all_model_jobs:
        input_path = Path(queued_job["missing_panel"])
        verify_frozen_file(input_path, queued_job["resident_input_sha256"], "resident input")
        input_rows = validate_panel_rows(read_jsonl(input_path), queued_job["panel_id"])
        existing_rows, complete = validate_live_output(
            queued_job, contract, input_rows,
            contract_sha256=queue["contract_sha256"], require_complete=False,
        )
        if not complete:
            continue
        expected_hash = contract["tables"][queued_job["table_id"]]["table_sha256_float32"]
        expected_gain = contract["tables"][queued_job["table_id"]]["gain"]
        rows_by_prompt = {prompt_hash(row): row for row in input_rows}
        for record in existing_rows:
            source = rows_by_prompt[prompt_hash(record)]
            cache_record(
                generation_cache, record, expected_table_hash=expected_hash,
                expected_gain=expected_gain, decoder=queued_job["decoder"],
                max_new_tokens=int(source["max_new_tokens"]),
            )

    for job in jobs:
        missing_path = Path(job["missing_panel"])
        verify_frozen_file(missing_path, job["resident_input_sha256"], "resident input")
        rows = validate_panel_rows(read_jsonl(missing_path), job["panel_id"])
        if len(rows) != int(job["missing_rows"]):
            raise ValueError(f"job {job['job_id']} missing-row count changed")
        receipt, table = load_static_receipt(
            Path(job["table_path"]), model_id=args.model_id, pairs=pairs,
        )
        table_record = contract["tables"][job["table_id"]]
        verify_frozen_file(
            Path(job["table_path"]), table_record["receipt_sha256"], "table receipt",
        )
        if (
            receipt["table_sha256_float32"] != table_record["table_sha256_float32"]
            or float(table["gain"]) != float(table_record["gain"])
        ):
            raise RuntimeError(f"table {job['table_id']} differs from the frozen contract")
        if receipt["model_geometry"] != runtime_geometry:
            raise RuntimeError(f"table {job['table_id']} belongs to another RoPE geometry")
        values = np.asarray(table["values_float32"], dtype=np.float32)
        output_dir = Path(job["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        generations_path = output_dir / "generations.jsonl"
        saved = read_jsonl(generations_path) if generations_path.exists() else []
        validate_saved_prefix(saved, rows)
        run_contract = {
            "status": "FIXED_ROPE_RESIDENT_RUN_V1",
            "pipeline_contract_sha256": queue["contract_sha256"],
            "job_id": job["job_id"],
            "stage": job["stage"],
            "model_id": args.model_id,
            "model_config_sha256": model_record["config_sha256"],
            "panel_id": job["panel_id"],
            "panel_sha256": job["panel_sha256"],
            "table_id": job["table_id"],
            "table_receipt_sha256": file_sha256(Path(job["table_path"])),
            "table_sha256_float32": receipt["table_sha256_float32"],
            "gain": table["gain"],
            "decoder": job["decoder"],
            "scorer": job["scorer"],
            "precision_arithmetic": model_record["precision_arithmetic"],
            "row_prompt_sha256": [prompt_hash(row) for row in rows],
            "same_table_all_layers_and_lengths": True,
            "runtime_table_switching": False,
        }
        contract_file = output_dir / "contract.json"
        if saved and not contract_file.is_file():
            raise ValueError(f"nonempty output has no prior run contract: {output_dir}")
        if contract_file.exists() and read_json(contract_file) != run_contract:
            raise ValueError(f"output directory belongs to another run: {output_dir}")
        atomic_json(contract_file, run_contract)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        install_static(model, values, table["gain"])
        verify_static(model, values, table["gain"])
        started = time.perf_counter()
        last_length = None
        with generations_path.open("a") as stream, torch.inference_mode():
            for row in rows[len(saved):]:
                length = int(row["length_cap"])
                if length != last_length:
                    verify_static(model, values, table["gain"])
                    last_length = length
                row_started = time.perf_counter()
                cache_key = deployment_cache_key(
                    table_hash=receipt["table_sha256_float32"], gain=float(table["gain"]),
                    decoder=job["decoder"], prompt=prompt_hash(row),
                    max_new_tokens=int(row["max_new_tokens"]),
                )
                cached = generation_cache.get(cache_key)
                if cached is not None:
                    tokens = list(cached["generated_ids"])
                    text = str(cached["output_text"])
                    ended_eos = bool(cached["ended_eos"])
                    references = list(row["references"])
                    literal = any(text.strip() == str(reference).strip() for reference in references)
                    exact = any(normalized(text) == normalized(str(reference)) for reference in references)
                    record = {
                        key: row.get(key) for key in (
                            "row_id", "source_panel", "source_row_id", "mini_semantic_id",
                            "semantic_group_id", "source_document_id", "task", "family",
                            "length_cap", "input_tokens", "references",
                        )
                    }
                    record.update(
                        job_id=job["job_id"], arm=job["table_id"],
                        prompt_sha256=prompt_hash(row), generated_ids=tokens,
                        output_text=text, ruler_official_score=float(ruler_score(row, text)),
                        whole_response_f1=float(qa_f1_score(text, references)),
                        literal_exact=literal, literal_exact_plus_eos=literal and ended_eos,
                        normalized_exact=exact, exact_plus_eos=exact and ended_eos,
                        ended_eos=ended_eos, empty=not text.strip(),
                        hit_cap=len(tokens) == int(row["max_new_tokens"]) and not ended_eos,
                        elapsed_seconds=0.0, table_sha256_float32=receipt["table_sha256_float32"],
                        gain=float(table["gain"]), reused_resident_generation=True,
                    )
                else:
                    input_ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
                    tokens = greedy_tokens(
                        model, input_ids, max_new_tokens=int(row["max_new_tokens"]),
                        eos_ids=eos_ids, pad_token_id=pad_token_id,
                        prefill_chunk_size=int(model_record.get("prefill_chunk_size", 0)),
                    )
                    ended_eos = bool(tokens and tokens[-1] in eos_ids)
                    decoded_ids = tokens[:-1] if ended_eos else tokens
                    text = tokenizer.decode(
                        decoded_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
                    )
                    references = list(row["references"])
                    literal = any(text.strip() == str(reference).strip() for reference in references)
                    exact = any(normalized(text) == normalized(str(reference)) for reference in references)
                    record = {
                        key: row.get(key) for key in (
                            "row_id", "source_panel", "source_row_id", "mini_semantic_id",
                            "semantic_group_id", "source_document_id", "task", "family",
                            "length_cap", "input_tokens", "references",
                        )
                    }
                    record.update(
                        job_id=job["job_id"], arm=job["table_id"],
                        prompt_sha256=prompt_hash(row), generated_ids=tokens,
                        output_text=text, ruler_official_score=float(ruler_score(row, text)),
                        whole_response_f1=float(qa_f1_score(text, references)),
                        literal_exact=literal, literal_exact_plus_eos=literal and ended_eos,
                        normalized_exact=exact, exact_plus_eos=exact and ended_eos,
                        ended_eos=ended_eos, empty=not text.strip(),
                        hit_cap=len(tokens) == int(row["max_new_tokens"]) and not ended_eos,
                        elapsed_seconds=time.perf_counter() - row_started,
                        table_sha256_float32=receipt["table_sha256_float32"],
                        gain=float(table["gain"]),
                        reused_resident_generation=False,
                    )
                    del input_ids
                    cache_record(
                        generation_cache, record,
                        expected_table_hash=receipt["table_sha256_float32"],
                        expected_gain=float(table["gain"]), decoder=job["decoder"],
                        max_new_tokens=int(row["max_new_tokens"]),
                    )
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                saved.append(record)
                atomic_json(output_dir / "live.json", {
                    "status": "RUNNING", "job_id": job["job_id"],
                    "completed": len(saved), "total": len(rows),
                    "last_prompt_sha256": prompt_hash(row),
                })
        verify_static(model, values, table["gain"])
        atomic_json(output_dir / "status.json", {
            "status": "COMPLETE", "job_id": job["job_id"],
            "rows": len(saved), "wall_seconds_this_invocation": time.perf_counter() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            "table_sha256_float32": receipt["table_sha256_float32"],
            "gain": float(table["gain"]),
            "generations_sha256": file_sha256(generations_path),
            "run_contract_sha256": file_sha256(contract_file),
        })
        for record, source in zip(saved, rows):
            cache_record(
                generation_cache, record,
                expected_table_hash=receipt["table_sha256_float32"],
                expected_gain=float(table["gain"]), decoder=job["decoder"],
                max_new_tokens=int(source["max_new_tokens"]),
            )
        torch.cuda.empty_cache()

    atexit.unregister(release_lock)
    release_lock()
    print(json.dumps({
        "status": "COMPLETE", "model_id": args.model_id,
        "jobs_completed": [job["job_id"] for job in jobs],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
