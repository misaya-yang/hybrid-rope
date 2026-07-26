#!/usr/bin/env python3
"""Evaluate fixed teacher-forced retrieval pairs without full-sequence logits."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    sha256_file,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate import (
    load_checkpoint,
    write_json,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.train import (
    Backbone,
    configure_cuda,
)


def score_last_token(
    backbone: torch.nn.Module,
    lm_head_weight: torch.Tensor,
    tokens: np.ndarray,
    *,
    gold: int,
    distractors: list[int],
) -> dict[str, Any]:
    input_ids = torch.from_numpy(
        np.asarray(tokens, dtype=np.int64)
    ).unsqueeze(0).to("cuda", non_blocking=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = backbone(input_ids)
        logits = F.linear(hidden[:, -1, :], lm_head_weight).float()[0]
    gold_logit = logits[gold]
    nll = torch.logsumexp(logits, dim=0) - gold_logit
    rank = 1 + int((logits > gold_logit).sum())
    best_distractor = (
        logits[torch.tensor(distractors, device=logits.device)].max()
        if distractors
        else torch.tensor(float("-inf"), device=logits.device)
    )
    return {
        "answer_token_nll": float(nll),
        "answer_token_rank": rank,
        "gold_minus_best_distractor_logit": float(
            gold_logit - best_distractor
        ),
        "next_token_exact_match": int(int(torch.argmax(logits)) == gold),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, float, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["length"]),
            float(row["source_fraction"]),
            int(row["distractor_count"]),
        )
        groups.setdefault(key, []).append(row)

    def metrics(local: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(local),
            "mean_answer_token_nll": float(
                np.mean([row["sourced"]["answer_token_nll"] for row in local])
            ),
            "median_answer_token_rank": float(
                np.median(
                    [row["sourced"]["answer_token_rank"] for row in local]
                )
            ),
            "mean_gold_minus_best_distractor_logit": float(
                np.mean(
                    [
                        row["sourced"][
                            "gold_minus_best_distractor_logit"
                        ]
                        for row in local
                    ]
                )
            ),
            "next_token_exact_match": float(
                np.mean(
                    [
                        row["sourced"]["next_token_exact_match"]
                        for row in local
                    ]
                )
            ),
            "mean_source_deletion_nll_gap": float(
                np.mean([row["source_deletion_nll_gap"] for row in local])
            ),
        }

    return {
        "overall": metrics(rows),
        "cells": {
            f"L{length}_P{fraction:.1f}_D{density}": metrics(local)
            for (length, fraction, density), local in sorted(groups.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--schedule", choices=("geo", "evq"), required=True)
    parser.add_argument("--retrieval-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--compile-mode", default="max-autotune-no-cudagraphs"
    )
    args = parser.parse_args()

    configure_cuda()
    model, frequency, checkpoint_sha = load_checkpoint(
        args.base_model.resolve(),
        args.checkpoint.resolve(),
        schedule=args.schedule,
        data_manifest=args.data_manifest.resolve(),
        device=torch.device("cuda", 0),
    )
    backbone: torch.nn.Module = Backbone(model.model)
    if args.compile:
        backbone = torch.compile(
            backbone,
            fullgraph=True,
            dynamic=False,
            mode=args.compile_mode,
        )
    manifest_path = args.retrieval_manifest.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "RETRIEVAL_DATA_VERIFIED":
        raise RuntimeError("retrieval manifest is not verified")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for length in manifest["lengths"]:
        files = {
            row["variant"]: row
            for row in manifest["arrays"]
            if row["length"] == length
        }
        for row in files.values():
            if sha256_file(root / row["path"]) != row["sha256"]:
                raise RuntimeError("retrieval artifact hash drift")
        sourced = np.load(
            root / files["sourced"]["path"],
            mmap_mode="r",
            allow_pickle=False,
        )
        deleted = np.load(
            root / files["source_deleted"]["path"],
            mmap_mode="r",
            allow_pickle=False,
        )
        metadata = json.loads(
            (root / files["metadata"]["path"]).read_text(encoding="utf-8")
        )
        for index, row in enumerate(metadata):
            sourced_score = score_last_token(
                backbone,
                model.lm_head.weight,
                sourced[index],
                gold=int(row["gold_token_id"]),
                distractors=[int(value) for value in row["distractor_token_ids"]],
            )
            deleted_score = score_last_token(
                backbone,
                model.lm_head.weight,
                deleted[index],
                gold=int(row["gold_token_id"]),
                distractors=[int(value) for value in row["distractor_token_ids"]],
            )
            rows.append(
                {
                    **row,
                    "sourced": sourced_score,
                    "source_deleted": deleted_score,
                    "source_deletion_nll_gap": (
                        deleted_score["answer_token_nll"]
                        - sourced_score["answer_token_nll"]
                    ),
                }
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if any(
        not math.isfinite(row["sourced"]["answer_token_nll"])
        or not math.isfinite(row["source_deleted"]["answer_token_nll"])
        for row in rows
    ):
        raise RuntimeError("retrieval evaluation produced non-finite NLL")
    receipt = {
        "status": "RETRIEVAL_EVALUATION_COMPLETE",
        "schedule": args.schedule,
        "checkpoint_sha256": checkpoint_sha,
        "retrieval_manifest_sha256": sha256_file(manifest_path),
        "frequency": frequency,
        "compile": {"enabled": args.compile, "mode": args.compile_mode},
        "elapsed_seconds": elapsed,
        "rows": rows,
        "summary": summarize(rows),
        "metric_boundary": (
            "teacher-forced next-token probability/rank probe; "
            "not autoregressive retrieval accuracy"
        ),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "results.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
