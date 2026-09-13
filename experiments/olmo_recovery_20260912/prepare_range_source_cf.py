#!/usr/bin/env python3
"""Prepare source-counterfactual pairs for frozen range-table optimization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.rope_fast_5090_20260912.source_counterfactual import (
    FAMILIES,
    LENGTHS,
    SEEDS,
    build_group,
    equal_token_labels,
)


def split_for_seed(seed: int) -> str:
    index = SEEDS.index(seed)
    return "fit" if index < 8 else "select" if index < 12 else "internal_confirm"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    from transformers import AutoTokenizer

    model = args.model.resolve()
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "olmo2" or config.get("num_hidden_layers") != 16:
        raise ValueError("source pairs require the retained OLMo-2-1B checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    labels = equal_token_labels(tokenizer)
    rows = []
    for seed in SEEDS:
        for length in LENGTHS:
            for family in FAMILIES:
                pair = build_group(tokenizer, family, length, seed, labels)
                for row in pair:
                    correct = tokenizer.encode(" " + row["answer"], add_special_tokens=False)
                    wrong_answer = row["counterfactual_answers"][1 - row["world"]]
                    wrong = tokenizer.encode(" " + wrong_answer, add_special_tokens=False)
                    if len(correct) != len(wrong) or not correct:
                        raise ValueError("counterfactual labels do not have matched token length")
                    row.update(
                        split=split_for_seed(seed),
                        correct_target_ids=correct + [int(tokenizer.eos_token_id)],
                        wrong_target_ids=wrong + [int(tokenizer.eos_token_id)],
                    )
                    rows.append(row)
    if len(rows) != 256 or len({row["row_id"] for row in rows}) != 256:
        raise RuntimeError("source-counterfactual matrix is incomplete")
    args.out.mkdir(parents=True)
    with (args.out / "rows.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "status": "RANGE_SOURCE_COUNTERFACTUAL_READY",
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "lengths": list(LENGTHS),
        "families": list(FAMILIES),
        "source_seeds": list(SEEDS),
        "split_seed_counts": {name: len({row["source_seed"] for row in rows if row["split"] == name}) for name in ("fit", "select", "internal_confirm")},
        "target": "paired correct-vs-counterfactual answer log-probability margin; final pair-follow generation remains separate",
        "scope": "constructed development mechanism panel, not full official RULER",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
