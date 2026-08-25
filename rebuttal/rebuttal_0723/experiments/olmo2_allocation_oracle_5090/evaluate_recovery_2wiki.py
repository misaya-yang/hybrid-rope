#!/usr/bin/env python3
"""Matched 20-row 2Wiki screen for the two natural-recovery models."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
    configure_ruler_flash_attention,
    greedy_generate,
)
from rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.evaluate_frozen_2wiki import (
    build_2wiki_jobs,
    load_official_2wiki_rows,
    normalized_exact,
    token_f1,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    configure_cuda,
)

from .oracle import sha256_file
from .train import _load_model
from .train_natural_recovery import STATUS as RECOVERY_STATUS


STATUS = "OLMO2_ALLOCATION_RECOVERY_2WIKI_SCREEN_COMPLETE_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--longbench-zip", type=Path, required=True)
    parser.add_argument("--native-recovery", type=Path, required=True)
    parser.add_argument("--learned-recovery", type=Path, required=True)
    parser.add_argument("--oracle-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--authorize", action="store_true")
    return parser.parse_args()


def _load_run(path: Path, *, expected_table: str) -> tuple[dict[str, Any], Path]:
    result_path = path.resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    adapter = result_path.parent / "adapter" / "adapter_model.safetensors"
    expected_allocation = {
        "native": "frozen native fixed-support table",
        "learned": "frozen learned fixed-support oracle table",
    }[expected_table]
    if (
        result.get("status") != RECOVERY_STATUS
        or result["protocol"]["allocation"] != expected_allocation
        or sha256_file(adapter) != result["adapter"]["model_sha256"]
    ):
        raise RuntimeError(f"{expected_table} recovery receipt drift")
    return result, adapter


def _evaluate_role(
    *,
    role: str,
    checkpoint: Path,
    adapter: Path,
    table_logits: torch.Tensor,
    jobs: list[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    model, allocation = _load_model(checkpoint)
    loaded = set_peft_model_state_dict(model, load_file(adapter))
    if loaded.unexpected_keys:
        raise RuntimeError(f"unexpected {role} adapter keys: {loaded.unexpected_keys}")
    with torch.no_grad():
        allocation.gap_delta_logits.copy_(table_logits)
    configure_ruler_flash_attention(model)
    model.config.use_cache = True
    model.eval().to("cuda")
    cells: dict[str, Any] = {}
    for length in (4096, 8192, 16384):
        selected = [job for job in jobs if job["nominal_length"] == length]
        rows = []
        for job in selected:
            ids = torch.tensor([job["input_ids"]], dtype=torch.long, device="cuda")
            generated = greedy_generate(
                model,
                ids,
                max_new_tokens=int(job["max_new_tokens"]),
                eos_token_id=int(tokenizer.eos_token_id),
            )[0].detach().cpu().tolist()
            prediction = tokenizer.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            rows.append(
                {
                    "row_id": job["row_id"],
                    "row_sha256": job["row_sha256"],
                    "input_tokens": len(job["input_ids"]),
                    "truncated": bool(job["truncated"]),
                    "prediction": prediction,
                    "references": job["references"],
                    "token_f1": token_f1(prediction, job["references"]),
                    "normalized_exact": normalized_exact(
                        prediction, job["references"]
                    ),
                }
            )
        cells[str(length)] = {
            "examples": len(rows),
            "mean_token_f1": float(np.mean([row["token_f1"] for row in rows])),
            "normalized_exact": float(
                np.mean([row["normalized_exact"] for row in rows])
            ),
            "mean_input_tokens": float(
                np.mean([row["input_tokens"] for row in rows])
            ),
            "truncated_examples": int(sum(row["truncated"] for row in rows)),
            "rows": rows,
        }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return cells


def main() -> int:
    args = parse_args()
    if not args.authorize or os.environ.get("OLMO_ALLOCATION_ORACLE_GPU_AUTHORIZED") != "YES":
        raise PermissionError("2Wiki screen requires both authorization factors")
    if int(args.limit) not in (20, 200):
        raise ValueError("2Wiki evaluation is frozen to either 20 or 200 rows")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    checkpoint = args.checkpoint.resolve()
    native_result, native_adapter = _load_run(
        args.native_recovery, expected_table="native"
    )
    learned_result, learned_adapter = _load_run(
        args.learned_recovery, expected_table="learned"
    )
    oracle_path = args.oracle_result.resolve()
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    state_path = oracle_path.parent / "allocation_state.pt"
    if sha256_file(state_path) != oracle["artifacts"]["allocation_state_sha256"]:
        raise RuntimeError("oracle allocation receipt drift")
    saved = torch.load(state_path, map_location="cpu", weights_only=True)

    package = load_official_2wiki_rows(args.longbench_zip.resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    all_jobs = build_2wiki_jobs(
        tokenizer, package, lengths=(4096, 8192, 16384)
    )
    jobs = [
        job
        for length in (4096, 8192, 16384)
        for job in [item for item in all_jobs if item["nominal_length"] == length][
            : int(args.limit)
        ]
    ]
    environment = configure_cuda()
    results = {
        "native_table__native_trained_lora": _evaluate_role(
            role="native",
            checkpoint=checkpoint,
            adapter=native_adapter,
            table_logits=torch.zeros_like(saved["gap_delta_logits"]),
            jobs=jobs,
            tokenizer=tokenizer,
        ),
        "learned_table__learned_trained_lora": _evaluate_role(
            role="learned",
            checkpoint=checkpoint,
            adapter=learned_adapter,
            table_logits=saved["gap_delta_logits"],
            jobs=jobs,
            tokenizer=tokenizer,
        ),
    }
    atomic_json(
        output,
        {
            "status": STATUS,
            "script_sha256": sha256_file(Path(__file__)),
            "metric_boundary": (
                f"{int(args.limit)}-row-per-length official LongBench 2Wiki "
                "prompt comparison; greedy token-F1/exact"
            ),
            "data": {
                "zip_sha256": package["zip_sha256"],
                "rows_sha256": package["rows_sha256"],
            },
            "environment": environment,
            "owners": {
                "native_recovery_sha256": sha256_file(args.native_recovery.resolve()),
                "learned_recovery_sha256": sha256_file(args.learned_recovery.resolve()),
            },
            "results": results,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
