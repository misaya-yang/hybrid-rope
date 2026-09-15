#!/usr/bin/env python3
"""Build a plain source-order RULER-13 panel without content padding.

The historical default remains 32K×200.  ``--length`` and
``--rows-per-task`` allow a separately frozen light panel without changing the
completed 32K identity.
"""

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
    parser.add_argument("--qa1-tail10", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--rows-per-task", type=int, default=200)
    parser.add_argument("--tasks", default=",".join(TASKS))
    args = parser.parse_args()
    if args.length <= 0 or args.rows_per_task <= 0:
        raise ValueError("length and rows-per-task must be positive")
    selected_tasks = tuple(value.strip() for value in args.tasks.split(",") if value.strip())
    if (
        not selected_tasks
        or len(set(selected_tasks)) != len(selected_tasks)
        or not set(selected_tasks).issubset(TASKS)
    ):
        raise ValueError("tasks must be a unique non-empty subset of RULER-13")

    output = args.out.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("status") == "COMPLETE"
            and manifest.get("rows") == len(selected_tasks) * args.rows_per_task
            and manifest.get("rows_per_task") == args.rows_per_task
            and manifest.get("length_cap") == args.length
            and manifest.get("tasks") == list(selected_tasks)
            and manifest.get("batching_scope") == "batch=1 with exact unpadded prompt_ids; no runtime padding"
        ):
            print(json.dumps({"status": "SKIP_COMPLETE", "rows": manifest["rows"]}))
            return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model.resolve(), local_files_only=True)
    rows = []
    sources = {}
    for task in selected_tasks:
        part = args.source_parts / task
        main_source = part / "source" / str(args.length) / task / "validation.jsonl"
        raw = read_jsonl(main_source)
        source_files = [main_source]
        if task == "qa_1" and len(raw) < args.rows_per_task and args.qa1_tail10 is not None:
            tail = args.qa1_tail10 / "source" / str(args.length) / "qa_1" / "validation.jsonl"
            raw.extend(read_jsonl(tail))
            source_files.append(tail)
        raw = raw[:args.rows_per_task]
        if len(raw) != args.rows_per_task:
            raise ValueError(
                f"source-order panel lacks {args.rows_per_task} rows for {task}: {len(raw)}"
            )
        prepared_rows = read_jsonl(part / "rows.jsonl")
        if not prepared_rows:
            raise ValueError(f"missing task generation budget for {task}")
        budget = int(prepared_rows[0]["max_new_tokens"])
        for index, source in enumerate(raw):
            text = source["input"] + source.get("answer_prefix", "")
            prompt_ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
            references = source.get("outputs") or []
            if not prompt_ids or not references or len(prompt_ids) + budget > args.length:
                raise ValueError(f"invalid unpadded source row: {task}/{index}")
            rows.append({
                "row_id": f"clean_{task}_{args.length}_{index:04d}",
                "task": task,
                "family": "ruler_full13_source_order",
                "length_cap": args.length,
                "prompt_ids": prompt_ids,
                "prompt_sha256": digest(prompt_ids),
                "input_tokens": len(prompt_ids),
                "actual_length": len(prompt_ids),
                "references": [str(value) for value in references],
                "max_new_tokens": budget,
                "source_order_index": index,
                "source_document_id": f"clean:{args.length}:{task}:{index}",
                "document_cluster_id": f"clean:{args.length}:{task}:{index}",
                "irrelevant_padding_tokens": 0,
                "selection_mode": "source-order",
                "selection_uses_model_outputs": False,
                "scorer_revision": "ruler-upstream-string-match-v1",
            })
        sources[task] = [{"path": str(path.resolve()), "sha256": sha256(path)} for path in source_files]

    counts = Counter(row["task"] for row in rows)
    prompts = {row["prompt_sha256"] for row in rows}
    expected_rows = len(selected_tasks) * args.rows_per_task
    if (
        len(rows) != expected_rows
        or counts != Counter({task: args.rows_per_task for task in selected_tasks})
        or len(prompts) != expected_rows
    ):
        raise ValueError("clean RULER task or prompt coverage drift")
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "inputs.jsonl"
    temporary = rows_path.with_name(rows_path.name + ".incomplete")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, rows_path)
    manifest = {
        "status": "COMPLETE",
        "contract": (
            f"TAILSPLINE_LLAMA_{args.length}_RULER{len(selected_tasks)}_{args.rows_per_task}_"
            "SOURCE_ORDER_UNPADDED_V1"
        ),
        "upstream_revision": "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
        "rows": expected_rows,
        "rows_per_task": args.rows_per_task,
        "tasks": list(selected_tasks),
        "length_cap": args.length,
        "selection_mode": "source-order",
        "depth_balancing": False,
        "multi_evidence_profile_selection": False,
        "content_padding": False,
        "batching_scope": "batch=1 with exact unpadded prompt_ids; no runtime padding",
        "minimum_input_tokens": min(row["input_tokens"] for row in rows),
        "maximum_input_tokens": max(row["input_tokens"] for row in rows),
        "inputs_sha256": sha256(rows_path),
        "sources": sources,
        "scope": (
            "RULER upstream generated Full-13 source-order sample, "
            f"{args.rows_per_task}/task at the {args.length}-token cap; "
            "not the upstream default 500/task."
        ),
    }
    temporary = manifest_path.with_name(manifest_path.name + ".incomplete")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
