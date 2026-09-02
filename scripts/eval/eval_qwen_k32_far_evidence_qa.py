#!/usr/bin/env python3
"""Evaluate the frozen Qwen K32 profiles on the derived 64K far-evidence QA panel."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import time

from scripts.analysis import summarize_qwen_k32_natural_nll as nll_summary
from scripts.data.prepare_qwen_k32_far_evidence_qa import (
    EVIDENCE_END_CEILING,
    MIN_EVIDENCE_TO_QUERY_TOKENS,
    EXPECTED_DATA_ZIP_SHA256,
    EXPECTED_MAIN_ZIP_SHA256,
    EXPECTED_PACKED_MANIFEST_SHA256,
    EXPECTED_PACKED_ROWS_SHA256,
    ROWS_PER_TASK,
    STATUS as DATA_STATUS,
    TARGET_LENGTH,
    TASKS,
    ids_sha256,
)
from scripts.data.prepare_qwen_k32_natural_nll import sha256_file, tokenizer_file_receipts
from scripts.eval.eval_qwen_k32_natural_nll import (
    ARM_ORDER,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_NATIVE_SHA256,
    EXPECTED_WEIGHT_SHA256,
    load_profiles,
    tensor_hash,
)
from scripts.eval.longbench_metrics import TASK_METRIC_MAP, post_process_prediction, score_prediction
from scripts.lib.checkpoint_identity import safetensors_weight_set_sha256

RESULT_STATUS = {
    "canary": "QWEN_K32_FAR_EVIDENCE_QA_CANARY_COMPLETE",
    "full": "QWEN_K32_FAR_EVIDENCE_QA_COMPLETE",
}


def load_nll_receipt(path: Path) -> dict:
    receipt = json.loads(path.read_text())
    identity = receipt.get("identity") or {}
    if (
        receipt.get("status") != "QWEN_K32_PACKED_NATURAL_NLL_STAGED_SUMMARIZED"
        or receipt.get("classification", {}).get("resolver") != "PASS"
        or receipt.get("classification", {}).get("index_vs_yarn") == "YARN_FAVORED"
        or identity.get("checkpoint_weight_sha256") != EXPECTED_WEIGHT_SHA256
        or identity.get("config_sha256") != EXPECTED_CONFIG_SHA256
        or identity.get("profiles") != {
            name: {"tensor_sha256": values[0], "file_sha256": values[1],
                   "attention_scaling": float(values[2])}
            for name, values in nll_summary.PROFILE.items()
        }
    ):
        raise ValueError("packed-natural NLL receipt does not authorize far-evidence QA")
    return receipt


def load_data(root: Path, checkpoint: Path) -> tuple[dict, list[dict]]:
    manifest_path, rows_path = root / "manifest.json", root / "rows.jsonl"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != DATA_STATUS
        or manifest.get("model_evaluation_status") != "NOT_RUN"
        or manifest.get("selection_uses_model_outcomes") is not False
        or manifest.get("prior_invalid_run_exists") is not True
        or manifest.get("tasks") != list(TASKS)
        or manifest.get("rows_per_task") != ROWS_PER_TASK
        or manifest.get("rows") != ROWS_PER_TASK * len(TASKS)
        or manifest.get("target_length") != TARGET_LENGTH
        or manifest.get("evidence_end_ceiling") != EVIDENCE_END_CEILING
        or manifest.get("minimum_evidence_to_query_tokens") != MIN_EVIDENCE_TO_QUERY_TOKENS
        or manifest.get("filler_special_token_policy") != (
            "replace every tokenizer special id one-for-one with token 198 newline")
        or manifest.get("filler_replacement_token_id") != 198
        or manifest.get("filler_source_disjoint_from_nll_by_row_range") is not True
        or manifest.get("source_text_sha256_intersection_count") != 0
        or manifest.get("filler_source_start_row") != 652019
        or manifest.get("prior_invalid_data_manifest_sha256") != (
            "b2c5bf42487fb3394cee6b1f66393b4f824849d778f3871c10d3b2ca1f8bc457")
        or manifest.get("question_panel_frozen_from_rows_sha256") != (
            "fca23e32d8019e245635175ab4d951d255270cc9e98840c745be6c383cdf82d6")
        or manifest.get("longbench_main_zip_sha256") != EXPECTED_MAIN_ZIP_SHA256
        or manifest.get("longbench_data_zip_sha256") != EXPECTED_DATA_ZIP_SHA256
        or manifest.get("packed_manifest_sha256") != EXPECTED_PACKED_MANIFEST_SHA256
        or manifest.get("packed_rows_sha256") != EXPECTED_PACKED_ROWS_SHA256
        or manifest.get("config_sha256") != sha256_file(checkpoint / "config.json")
        or manifest.get("tokenizer_files") != tokenizer_file_receipts(checkpoint)
        or manifest.get("file") != {"path": rows_path.name, "sha256": sha256_file(rows_path)}
        or manifest.get("script_sha256") != sha256_file(
            Path(__file__).resolve().parents[1] / "data" / "prepare_qwen_k32_far_evidence_qa.py")
    ):
        raise ValueError("far-evidence data contract drift")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    seen, counts = set(), {task: 0 for task in TASKS}
    for row in rows:
        task, row_hash, ids = row.get("task"), row.get("row_sha256"), row.get("input_ids")
        if (
            task not in counts or row_hash in seen or not isinstance(ids, list)
            or not ids or any(type(value) is not int or value < 0 for value in ids)
            or row.get("input_ids_sha256") != ids_sha256(ids)
            or type(row.get("generation_tokens")) is not int
            or len(ids) + row["generation_tokens"] != TARGET_LENGTH
            or row.get("evidence_end_token_exclusive", TARGET_LENGTH) > EVIDENCE_END_CEILING
            or row.get("evidence_to_query_lower_bound", 0) < MIN_EVIDENCE_TO_QUERY_TOKENS
            or row.get("filler_selected_special_tokens") != 0
            or row.get("filler_replacement_token_id") != 198
            or not isinstance(row.get("filler_sanitized_ids_sha256"), str)
            or row.get("evidence_special_ids") != [151644, 151645, 151644]
            or row.get("label_special_ids") != []
            or row.get("query_special_ids") != [151645, 151644]
            or type(row.get("filler_source_special_tokens_replaced")) is not int
            or row["filler_source_special_tokens_replaced"] < 0
            or not isinstance(row.get("references"), list) or not row["references"]
        ):
            raise ValueError("invalid or duplicated far-evidence row")
        seen.add(row_hash); counts[task] += 1
    if counts != {task: ROWS_PER_TASK for task in TASKS}:
        raise ValueError("far-evidence task grid is incomplete")
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest, sorted(rows, key=lambda row: (row["task"], row["row_sha256"]))


def main() -> int:
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index-table", type=Path, required=True)
    parser.add_argument("--yarn-table", type=Path, required=True)
    parser.add_argument("--nll-receipt", type=Path, required=True)
    parser.add_argument("--stage", choices=("canary", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("far-evidence evaluation output must be fresh")
    if sha256_file(args.checkpoint / "config.json") != EXPECTED_CONFIG_SHA256:
        raise ValueError("Qwen K32 config hash mismatch")
    weights = safetensors_weight_set_sha256(args.checkpoint)
    if weights != EXPECTED_WEIGHT_SHA256:
        raise ValueError("Qwen K32 weight hash mismatch")
    nll_receipt = load_nll_receipt(args.nll_receipt)
    data, rows = load_data(args.data_root, args.checkpoint)
    profiles = load_profiles(args.index_table, args.yarn_table)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
        configure_cuda, configure_ruler_flash_attention, greedy_generate)

    configure_cuda()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa").eval().to("cuda")
    model.requires_grad_(False); model.config.use_cache = True
    configure_ruler_flash_attention(model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False)
    rotary = model.model.rotary_emb
    native = rotary.inv_freq.detach().cpu().float().numpy().copy()
    if tensor_hash(native) != EXPECTED_NATIVE_SHA256 or float(rotary.attention_scaling) != 1.0:
        raise RuntimeError("runtime Native table/gain mismatch")
    profiles[0]["values"] = native
    for profile in profiles:
        profile["active"] = torch.from_numpy(profile["values"]).to(
            device=rotary.inv_freq.device, dtype=rotary.inv_freq.dtype)

    active_profiles = profiles if args.stage == "full" else profiles[:1]
    if args.stage == "canary":
        active_rows = [next(row for row in rows if row["generation_tokens"] == budget)
                       for budget in (32, 128)]
    else:
        active_rows = rows
    args.output.mkdir(parents=True)
    manifest = {
        "status": "QWEN_K32_FAR_EVIDENCE_QA_FROZEN",
        "stage": args.stage,
        "checkpoint_weight_sha256": weights,
        "config_sha256": data["config_sha256"],
        "data_manifest_sha256": data["manifest_sha256"],
        "data_rows_sha256": data["file"]["sha256"],
        "nll_receipt_sha256": sha256_file(args.nll_receipt),
        "nll_classification": nll_receipt["classification"],
        "tasks": list(TASKS),
        "rows_per_task": ROWS_PER_TASK,
        "arm_order": [profile["name"] for profile in active_profiles],
        "declared_arm_order": list(ARM_ORDER),
        "profiles": [{key: value for key, value in profile.items() if key not in {"values", "active"}}
                     for profile in profiles],
        "script_sha256": sha256_file(Path(__file__)),
        "model_source_sha256": sha256_file(Path(inspect.getsourcefile(type(model)))),
        "attention_source_sha256": sha256_file(Path(inspect.getsourcefile(configure_ruler_flash_attention))),
        "use_cache": True,
        "compile": False,
        "model_updates": 0,
        "profile_selection": False,
        "all_profiles_loaded_before_inference": True,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    raw_path = args.output / "examples.jsonl"
    results, started = [], time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with raw_path.open("x") as handle:
        for profile in active_profiles:
            with torch.no_grad():
                rotary.inv_freq.copy_(profile["active"])
                if hasattr(rotary, "original_inv_freq"):
                    rotary.original_inv_freq = rotary.inv_freq.detach().clone()
                rotary.attention_scaling = profile["attention_scaling"]
            for row in active_rows:
                input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
                batch_started = time.perf_counter()
                generated = greedy_generate(
                    model, input_ids, max_new_tokens=int(row["generation_tokens"]),
                    eos_token_id=None if args.stage == "canary" else tokenizer.eos_token_id)[0]
                torch.cuda.synchronize()
                if not torch.isfinite(generated.float()).all():
                    raise RuntimeError("nonfinite generated token ids")
                if (not torch.equal(rotary.inv_freq, profile["active"])
                        or not math.isclose(float(rotary.attention_scaling),
                                            profile["attention_scaling"], rel_tol=0, abs_tol=0)):
                    raise RuntimeError("static QA profile mutated during generation")
                record = {
                    "arm": profile["name"], "task": row["task"],
                    "row_sha256": row["row_sha256"],
                    "table_sha256_float32": profile["tensor_sha256"],
                    "attention_scaling": profile["attention_scaling"],
                    "generation_tokens_budget": row["generation_tokens"],
                    "generated_tokens": int(generated.numel()),
                    "batch_seconds": time.perf_counter() - batch_started,
                }
                if args.stage == "canary":
                    record["metrics_exposed"] = False
                else:
                    token_ids = [int(value) for value in generated.detach().cpu().tolist()]
                    prediction = tokenizer.decode(
                        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    processed = post_process_prediction(row["task"], prediction)
                    record.update(
                        prediction=prediction,
                        generated_token_ids=token_ids,
                        score=score_prediction(
                            row["task"], TASK_METRIC_MAP[row["task"]], processed,
                            row["references"], row.get("all_classes", [])))
                results.append(record); handle.write(json.dumps(record, ensure_ascii=False) + "\n"); handle.flush()
                if len(results) == 1 or len(results) % 10 == 0:
                    print(f"{args.stage}: {len(results)}/{len(active_profiles) * len(active_rows)}", flush=True)
                del input_ids, generated
        os.fsync(handle.fileno())

    result = {
        "status": RESULT_STATUS[args.stage], "stage": args.stage, "rows": len(results),
        "elapsed_seconds": time.perf_counter() - started,
        "process_elapsed_seconds": time.perf_counter() - process_started,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "examples_sha256": sha256_file(raw_path),
        "run_manifest_sha256": sha256_file(manifest_path),
    }
    if args.stage == "canary":
        result.update(
            metrics_exposed=False,
            batch_seconds={str(row["generation_tokens_budget"]): row["batch_seconds"] for row in results},
            evidence_limit="Timing/finite/identity canary only; generated content and scores are not stored.")
    else:
        result["means"] = {
            arm: {task: float(sum(row["score"] for row in results
                                  if row["arm"] == arm and row["task"] == task) / ROWS_PER_TASK)
                  for task in TASKS}
            for arm in ARM_ORDER}
        result["evidence_limit"] = "Derived far-evidence LongBench QA; not official LongBench or broad SOTA."
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
