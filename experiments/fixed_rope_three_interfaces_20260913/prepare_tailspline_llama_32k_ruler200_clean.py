#!/usr/bin/env python3
"""Build a plain source-order RULER-13×200 panel without content padding."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parts", type=Path, required=True)
    parser.add_argument("--qa1-tail10", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    output = args.out.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "COMPLETE" and manifest.get("rows") == 2600:
            print(json.dumps({"status": "SKIP_COMPLETE", "rows": 2600}))
            return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model.resolve(), local_files_only=True)
    rows = []
    sources = {}
    for task in TASKS:
        part = args.source_parts / task
        main_source = part / "source/32768" / task / "validation.jsonl"
        raw = read_jsonl(main_source)
        source_files = [main_source]
        if task == "qa_1":
            tail = args.qa1_tail10 / "source/32768/qa_1/validation.jsonl"
            raw.extend(read_jsonl(tail))
            source_files.append(tail)
        raw = raw[:200]
        if len(raw) != 200:
            raise ValueError(f"source-order panel lacks 200 rows for {task}: {len(raw)}")
        prepared_rows = read_jsonl(part / "rows.jsonl")
        if not prepared_rows:
            raise ValueError(f"missing task generation budget for {task}")
        budget = int(prepared_rows[0]["max_new_tokens"])
        for index, source in enumerate(raw):
            text = source["input"] + source.get("answer_prefix", "")
            prompt_ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
            references = source.get("outputs") or []
            if not prompt_ids or not references or len(prompt_ids) + budget > 32768:
                raise ValueError(f"invalid unpadded source row: {task}/{index}")
            rows.append({
                "row_id": f"clean_{task}_32768_{index:04d}",
                "task": task,
                "family": "ruler_full13_source_order",
                "length_cap": 32768,
                "prompt_ids": prompt_ids,
                "prompt_sha256": digest(prompt_ids),
                "input_tokens": len(prompt_ids),
                "actual_length": len(prompt_ids),
                "references": [str(value) for value in references],
                "max_new_tokens": budget,
                "source_order_index": index,
                "source_document_id": f"clean:{task}:{index}",
                "document_cluster_id": f"clean:{task}:{index}",
                "irrelevant_padding_tokens": 0,
                "selection_mode": "source-order",
                "selection_uses_model_outputs": False,
                "scorer_revision": "ruler-upstream-string-match-v1",
            })
        sources[task] = [{"path": str(path.resolve()), "sha256": sha256(path)} for path in source_files]

    counts = Counter(row["task"] for row in rows)
    prompts = {row["prompt_sha256"] for row in rows}
    if len(rows) != 2600 or counts != Counter({task: 200 for task in TASKS}) or len(prompts) != 2600:
        raise ValueError("clean RULER-200 task or prompt coverage drift")
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "inputs.jsonl"
    temporary = rows_path.with_name(rows_path.name + ".incomplete")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, rows_path)
    manifest = {
        "status": "COMPLETE",
        "contract": "TAILSPLINE_LLAMA_32K_RULER13_200_SOURCE_ORDER_UNPADDED_V1",
        "upstream_revision": "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
        "rows": 2600,
        "rows_per_task": 200,
        "tasks": list(TASKS),
        "length_cap": 32768,
        "selection_mode": "source-order",
        "depth_balancing": False,
        "multi_evidence_profile_selection": False,
        "content_padding": False,
        "batch_padding_scope": "masked left pad at runtime only; absent from prompt_ids and scorer inputs",
        "minimum_input_tokens": min(row["input_tokens"] for row in rows),
        "maximum_input_tokens": max(row["input_tokens"] for row in rows),
        "inputs_sha256": sha256(rows_path),
        "sources": sources,
        "scope": "RULER upstream generated Full-13 source-order sample, 200/task at the 32K cap; not the upstream default 500/task.",
    }
    temporary = manifest_path.with_name(manifest_path.name + ".incomplete")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
