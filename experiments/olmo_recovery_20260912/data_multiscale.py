"""Streaming multiscale data builders used only by this recovery experiment."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.evq_recovery.data import write_json


def prepare_pg19_multiscale(sources: Path, out: Path, tokenizer) -> dict:
    """Assign disjoint 16K prediction blocks to either 2x or 4x training."""
    books = json.loads((sources / "pg19_books.json").read_text())
    train = {8192: [], 16384: []}
    heldout = defaultdict(list)
    rows = defaultdict(list)
    seen = {}
    for book_index, book in enumerate(books):
        path = sources / "pg19" / book["key"]
        source_identity = book["key"]
        split = book["split"]
        if source_identity in seen and seen[source_identity] != split:
            raise ValueError("PG19 source identity crosses official splits")
        seen[source_identity] = split
        tokens = tokenizer.encode(path.read_text(), add_special_tokens=False)
        if split == "train":
            block_index = 0
            for offset in range(0, len(tokens) - 16384, 16384):
                if block_index % 2 == 0:
                    for half in (0, 8192):
                        local_offset = offset + half
                        array = np.asarray(tokens[local_offset:local_offset + 8193], dtype="<i4")
                        train[8192].append(array)
                        rows["train_8192"].append({"source_id": book["key"],
                            "parent_block": block_index, "offset": local_offset,
                            "prediction_span": [local_offset + 1, local_offset + 8193]})
                else:
                    array = np.asarray(tokens[offset:offset + 16385], dtype="<i4")
                    train[16384].append(array)
                    rows["train_16384"].append({"source_id": book["key"],
                        "parent_block": block_index, "offset": offset,
                        "prediction_span": [offset + 1, offset + 16385]})
                block_index += 1
        elif len(tokens) >= 32769:
            offset = int(hashlib.sha256(book["key"].encode()).hexdigest()[:12], 16) % (len(tokens) - 32768)
            array = np.asarray(tokens[offset:offset + 32769], dtype="<i4")
            heldout[split].append(array)
            rows[split].append({"source_id": book["key"], "offset": offset})
    if not train[8192] or not train[16384] or min(len(heldout[name]) for name in ("validation", "test")) < 1:
        raise ValueError("insufficient source-isolated PG19 material")
    np.save(out / "cpt_train_8192.npy", np.stack(train[8192]), allow_pickle=False)
    np.save(out / "cpt_train.npy", np.stack(train[16384]), allow_pickle=False)
    for split, values in heldout.items():
        np.save(out / f"lm_{split}.npy", np.stack(values), allow_pickle=False)
    write_json(out / "pg19_manifest.json", {"rows": rows,
        "training_prediction_tokens_by_length": {str(length): len(values) * length for length, values in train.items()},
        "train_window_policy": "Within each book, even 16K source blocks become two contiguous 8K windows and odd blocks remain one contiguous 16K window. Prediction spans never overlap; only the causal boundary token is shared.",
        "source_split_policy": "Official PG19 train/validation/test books; no source crosses a split.",
        "padding": "none", "position_jumps": "none", "eval_window_length": 32768})
    return {"train_windows_by_length": {str(length): len(values) for length, values in train.items()},
            "train_prediction_tokens_by_length": {str(length): len(values) * length for length, values in train.items()},
            "heldout_books": {split: len(values) for split, values in heldout.items()}}


def partition_long_prompt_sft(out: Path, eos_token_id: int) -> dict:
    """Keep intact answer+EOS rows whose prompt itself occupies its long bucket."""
    source = out / "sft_train.jsonl"
    destinations = {8192: out / "sft_train_8192.jsonl", 16384: out / "sft_train_16384.jsonl"}
    handles = {length: path.open("x") for length, path in destinations.items()}
    union_tmp = out / "sft_train.multiscale.incomplete"
    counts = defaultdict(int)
    source_ids = defaultdict(set)
    try:
        with source.open() as stream, union_tmp.open("x") as union:
            for line in stream:
                row = json.loads(line)
                bucket = int(row["length_bucket"])
                if bucket not in handles:
                    counts["unsupported_bucket"] += 1
                    continue
                prompt_tokens = int(row["prompt_tokens"])
                full_tokens = len(row["input_ids"])
                if prompt_tokens < math.ceil(0.875 * bucket) or full_tokens > bucket:
                    counts[f"excluded_{bucket}_not_true_long_prompt"] += 1
                    continue
                if row["input_ids"][-1] != eos_token_id or not 0 < row["target_start"] < full_tokens:
                    raise ValueError("LongAlign answer/EOS supervision boundary is invalid")
                row["training_length"] = bucket
                row["length_contract"] = "prompt_tokens>=0.875*bucket; intact prompt+answer+EOS<=bucket; no padding or concatenated short instructions"
                rendered = json.dumps(row) + "\n"
                handles[bucket].write(rendered)
                union.write(rendered)
                counts[f"train_{bucket}"] += 1
                source_ids[bucket].add(row["source_id"])
    finally:
        for handle in handles.values():
            handle.close()
    if any(counts[f"train_{bucket}"] == 0 for bucket in destinations):
        raise ValueError("both 8K and 16K true-long LongAlign training pools must be nonempty")
    union_tmp.replace(source)
    return {"counts": dict(counts), "source_groups_by_length": {str(k): len(v) for k, v in source_ids.items()},
            "distance_limitation": "LongAlign natural long prompts do not by themselves prove the distance from decisive evidence to the supervised answer."}
