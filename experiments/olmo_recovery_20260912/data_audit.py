#!/usr/bin/env python3
"""Fail-closed split, token-target, and source-overlap audit for recovery data."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.evq_recovery.acquire import file_hash
from experiments.evq_recovery.data import JsonlIndex


FILES = tuple(f"{kind}_{split}.jsonl" for kind in ("sft", "native", "qa") for split in ("train", "dev", "test"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    args = p.parse_args()
    root = args.data.resolve()
    manifest = json.loads((root / "data_manifest.json").read_text())
    if manifest.get("status") != "CPU_DATA_READY_GPU_NOT_RUN":
        raise ValueError("data manifest is not ready")
    for name in FILES:
        if name != "qa_train.jsonl" and (not (root / name).is_file() or not (root / name).stat().st_size):
            raise ValueError(f"required nonempty split is absent: {name}")
    for name, receipt in manifest["files"].items():
        if file_hash(root / name) != receipt["sha256"]:
            raise ValueError(f"prepared file drift: {name}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model.resolve(), local_files_only=True)
    if file_hash(args.model.resolve() / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("tokenizer identity drift")

    groups = defaultdict(set)
    counts = defaultdict(int)
    for name in FILES:
        path = root / name
        if not path.exists():
            continue
        kind, split = name.removesuffix(".jsonl").split("_", 1)
        index = JsonlIndex(path)
        for row_index in range(len(index)):
            row = index[row_index]
            if row.get("split") != split or not row.get("source_id"):
                raise ValueError(f"missing split/source identity: {name}:{row_index}")
            groups[kind, split].add(row["source_id"])
            counts[kind, split] += 1
            if (kind == "sft" or (kind == "native" and row.get("task") != "text")) and "input_ids" in row:
                ids, start = row["input_ids"], int(row["target_start"])
                if not 0 <= start < len(ids) or ids[-1] != tokenizer.eos_token_id:
                    raise ValueError(f"answer/EOS target contract drift: {name}:{row_index}")
            if kind == "native" and split == "train":
                positions = row.get("kl_positions", [])
                if not positions or len(positions) > 128 or len(positions) != len(set(positions)):
                    raise ValueError(f"KL position contract drift: {name}:{row_index}")
                if not all(0 <= p < len(row["input_ids"]) - 1 for p in positions):
                    raise ValueError(f"KL index out of range: {name}:{row_index}")
                if row.get("task") != "text" and not all(p + 1 < row["target_start"] for p in positions):
                    raise ValueError(f"KL position overlaps gold answer: {name}:{row_index}")
            if kind == "qa":
                if not row.get("references") or len(row["prompt_ids"]) != row["input_tokens"]:
                    raise ValueError(f"QA prompt/reference drift: {name}:{row_index}")
                if row["input_tokens"] + row["generation_budget"] > row["length_bucket"]:
                    raise ValueError(f"QA physical cap drift: {name}:{row_index}")
    for kind in ("sft", "native", "qa"):
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            overlap = groups[kind, left] & groups[kind, right]
            if overlap:
                raise ValueError(f"{kind} source leakage {left}/{right}: {len(overlap)} groups")
    for name in ("cpt_train.npy", "lm_validation.npy", "lm_test.npy"):
        array = np.load(root / name, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.shape[1] not in (16_385, 32_769) or array.dtype.kind not in "iu":
            raise ValueError(f"invalid contiguous LM payload: {name}/{array.shape}/{array.dtype}")
    report = {"status": "PASS", "counts": {f"{kind}/{split}": value for (kind, split), value in counts.items()},
              "source_groups": {f"{kind}/{split}": len(value) for (kind, split), value in groups.items()},
              "claims": "split and target integrity only; no GPU readiness, learning, or FFN-necessity result"}
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
