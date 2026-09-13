#!/usr/bin/env python3
"""Measure each frozen OLMo baseline once on range-solver answer NLL."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np


ARMS = ("Native", "BM_g4", "MrPro_g4", "C42V24_g4")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source-cf", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    rows = [row for row in read_rows(args.data / "rows.jsonl") if row["split"] == "fit"]
    cf_manifest = json.loads((args.source_cf / "manifest.json").read_text())
    cf_rows = [row for row in read_rows(args.source_cf / "rows.jsonl") if row["split"] == "fit"]
    if manifest.get("status") != "RANGE_SOLVER_DATA_READY" or len(rows) != 168:
        raise ValueError("range fit data is incomplete")
    if cf_manifest.get("status") != "RANGE_SOURCE_COUNTERFACTUAL_READY" or len(cf_rows) != 128:
        raise ValueError("range source-counterfactual fit data is incomplete")
    if not args.execute:
        print(json.dumps({
            "status": "PLAN_ONLY", "arms": ARMS, "fit_rows_per_arm": len(rows),
            "source_cf_rows_per_arm": len(cf_rows),
        }))
        return

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from experiments.olmo_recovery_20260912.recovery_v2_runtime import table_for_config
    from experiments.olmo_recovery_20260912.runtime import validate_cuda
    from scripts.experiments.cross_audit.tables import install_static

    environment = validate_cuda()
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda"}, attn_implementation="sdpa",
    ).eval().requires_grad_(False)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summaries = {}
    with torch.inference_mode():
        for arm in ARMS:
            table = table_for_config(config, arm)
            install_static(model, np.asarray(table["values_float32"], dtype=np.float32), table["gain"])
            path = args.out / f"{arm}.jsonl"
            records = read_rows(path) if path.exists() else []
            if records:
                expected_ids = [row["row_id"] for row in rows]
                if len(records) != len(rows) or [row["row_id"] for row in records] != expected_ids or any(row["arm"] != arm for row in records):
                    raise ValueError(f"existing baseline arm is partial or mismatched: {arm}")
            else:
                with path.open("x") as stream:
                    for index, row in enumerate(rows):
                        target = row["target_ids"]
                        ids = torch.tensor([row["prompt_ids"] + target[:-1]], dtype=torch.long, device="cuda")
                        labels = torch.tensor(target, dtype=torch.long, device="cuda")
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(target), return_dict=True).logits[0].float()
                            nll = torch.nn.functional.cross_entropy(logits, labels)
                        record = {
                            "row_id": row["row_id"], "task": row["task"],
                            "length_cap": row["length_cap"], "arm": arm,
                            "answer_tokens_including_eos": len(target), "nll": float(nll),
                        }
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                        stream.flush()
                        records.append(record)
                        if index % 8 == 7:
                            write(args.out / "live.json", {"arm": arm, "completed": index + 1, "total": len(rows)})
                        del ids, labels, logits, nll
            cells = defaultdict(list)
            for record in records:
                cells[(record["length_cap"], record["task"])].append(record["nll"])
            summaries[arm] = {
                "table": table,
                "cells": {
                    f"{cap}/{task}": {"rows": len(values), "mean_nll": sum(values) / len(values)}
                    for (cap, task), values in sorted(cells.items())
                },
            }
            cf_path = args.out / f"source_cf_{arm}.jsonl"
            cf_records = read_rows(cf_path) if cf_path.exists() else []
            for index, record in enumerate(cf_records):
                if (
                    index >= len(cf_rows)
                    or record["row_id"] != cf_rows[index]["row_id"]
                    or record["arm"] != arm
                ):
                    raise ValueError(f"existing source-counterfactual prefix is mismatched: {arm}")
            with cf_path.open("a") as stream:
                for index, row in enumerate(cf_rows[len(cf_records):], start=len(cf_records)):
                    correct, wrong = row["correct_target_ids"], row["wrong_target_ids"]
                    if len(correct) != len(wrong):
                        raise ValueError("counterfactual target lengths differ")
                    inputs = torch.tensor(
                        [row["prompt_ids"] + correct[:-1], row["prompt_ids"] + wrong[:-1]],
                        dtype=torch.long, device="cuda",
                    )
                    targets = torch.tensor([correct, wrong], dtype=torch.long, device="cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(
                            input_ids=inputs, use_cache=False,
                            logits_to_keep=len(correct), return_dict=True,
                        ).logits.float()
                        logprob = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1).sum(-1)
                        margin = logprob[0] - logprob[1]
                        hinge = torch.relu(torch.tensor(0.5, device="cuda") - margin)
                    record = {
                        "row_id": row["row_id"], "family": row["family"],
                        "length_cap": row["length_cap"], "source_seed": row["source_seed"],
                        "world": row["world"], "arm": arm,
                        "target_tokens_including_eos": len(correct),
                        "correct_minus_wrong_logprob": float(margin),
                        "margin_half_hinge": float(hinge),
                    }
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    stream.flush()
                    cf_records.append(record)
                    if index % 8 == 7:
                        write(args.out / "live.json", {
                            "arm": arm, "source_cf_completed": index + 1,
                            "source_cf_total": len(cf_rows),
                        })
                    del inputs, targets, logits, logprob, margin, hinge
            cf_cells = defaultdict(list)
            for record in cf_records:
                cf_cells[(record["length_cap"], record["family"])].append(record["margin_half_hinge"])
            summaries[arm]["source_cf_cells"] = {
                f"{cap}/{family}": {"rows": len(values), "mean_margin_half_hinge": sum(values) / len(values)}
                for (cap, family), values in sorted(cf_cells.items())
            }
    result = {
        "status": "COMPLETE",
        "arms": summaries,
        "fit_rows_per_arm": len(rows),
        "source_cf_rows_per_arm": len(cf_rows),
        "baseline_reuse": "first and only teacher-forced measurement on this data contract",
        "environment": environment,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    write(args.out / "summary.json", result)
    write(args.out / "status.json", {
        "status": "COMPLETE", "arms": list(ARMS), "rows_per_arm": len(rows),
        "source_cf_rows_per_arm": len(cf_rows),
    })
    print(json.dumps({
        "status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
        "peak_memory_allocated_bytes": result["peak_memory_allocated_bytes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
