#!/usr/bin/env python3
"""Evaluate one S1 checkpoint on shared-endpoint 2K/4K/8K/16K NLL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from protocol import ARMS, SEEDS, SPEC, SUPPORTS
from run import atomic_json, build_model, sha256_file


LENGTHS = (2_048, 4_096, 8_192, 16_384)
BATCH_SIZE = {2_048: 8, 4_096: 4, 8_192: 2, 16_384: 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--support", type=int, choices=SUPPORTS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.original_root.resolve()))
    from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m import (  # noqa: PLC0415
        run_experiment as old,
    )

    old.configure_cuda_kernels()
    eval_manifest = json.loads(args.eval_manifest.resolve().read_text())
    if eval_manifest.get("status") != "READY":
        raise ValueError("evaluation manifest is not READY")
    data_path = Path(eval_manifest["documents"])
    if sha256_file(data_path) != eval_manifest["documents_sha256"]:
        raise ValueError("evaluation document payload hash changed")
    documents = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if documents.shape != (512, 16_385) or documents.dtype != np.uint16:
        raise ValueError(f"unexpected evaluation payload {documents.shape}/{documents.dtype}")

    model, full_z = build_model(old, args.arm, args.support, args.seed)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval().cuda()
    checkpoint_sha = sha256_file(args.checkpoint)
    result_path = args.output / "rows.jsonl"
    status = {
        "status": "RUNNING",
        "arm": args.arm,
        "support": int(args.support),
        "seed": int(args.seed),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_completed_updates": int(payload.get("completed_updates", payload.get("metadata", {}).get("completed_updates", 0))),
        "eval_manifest": str(args.eval_manifest.resolve()),
        "eval_manifest_sha256": sha256_file(args.eval_manifest),
        "lengths": list(LENGTHS),
        "tail_tokens": 128,
        "metric": "per-document mean next-token NLL; whole window and last 128 targets",
        "started_at": time.time(),
    }
    atomic_json(args.output / "status.json", status)
    torch.cuda.reset_peak_memory_stats()
    summaries = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for length in LENGTHS:
            whole = []
            tail = []
            batch_size = BATCH_SIZE[length]
            for start in range(0, len(documents), batch_size):
                indices = list(range(start, min(start + batch_size, len(documents))))
                windows = np.stack(
                    [np.asarray(documents[index, -(length + 1) :], dtype=np.int64) for index in indices]
                )
                batch = torch.from_numpy(windows).cuda(non_blocking=True)
                logits = model(batch[:, :-1])
                losses = F.cross_entropy(
                    logits.transpose(1, 2).float(), batch[:, 1:], reduction="none"
                )
                for offset, document_index in enumerate(indices):
                    row = {
                        "arm": args.arm,
                        "support": int(args.support),
                        "seed": int(args.seed),
                        "checkpoint_sha256": checkpoint_sha,
                        "checkpoint_completed_updates": status["checkpoint_completed_updates"],
                        "document_index": document_index,
                        "source_document_id": eval_manifest["document_ids"][document_index],
                        "context_length": length,
                        "whole_sum_nll": float(losses[offset].sum()),
                        "whole_target_tokens": length,
                        "tail_sum_nll": float(losses[offset, -128:].sum()),
                        "tail_target_tokens": 128,
                        "target_sha256": hashlib.sha256(windows[offset, -128:].astype("<i8").tobytes()).hexdigest(),
                    }
                    whole.append(row["whole_sum_nll"] / length)
                    tail.append(row["tail_sum_nll"] / 128)
                    with result_path.open("a") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                del batch, logits, losses
            summaries[str(length)] = {
                "documents": len(whole),
                "mean_document_whole_nll": float(np.mean(whole)),
                "mean_document_tail128_nll": float(np.mean(tail)),
            }
            status.update({"current_length": length, "summaries": summaries})
            atomic_json(args.output / "live.json", status)
    status.update(
        {
            "status": "COMPLETE",
            "summaries": summaries,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "finished_at": time.time(),
        }
    )
    if full_z is not None:
        status["allocation"] = model.blocks[0].attention.rope.receipt()
    atomic_json(args.output / "status.json", status)
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
