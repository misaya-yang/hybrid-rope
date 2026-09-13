#!/usr/bin/env python3
"""Freeze accepted teacher generations as specified margin target paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--teacher-generations", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--fit-rows", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.fit_rows < 1:
        raise ValueError("invalid fit count or pre-existing output")
    panel = {
        row["prompt_sha256"]: row
        for row in read_jsonl(args.panel)
        if row.get("task") == args.task and int(row.get("length_cap", -1)) == args.length
    }
    teachers = {
        row["prompt_sha256"]: row
        for row in read_jsonl(args.teacher_generations)
        if row.get("task") == args.task and int(row.get("length_cap", -1)) == args.length
        and row.get("prompt_sha256") in panel
    }
    if set(teachers) != set(panel) or not args.fit_rows < len(panel):
        raise ValueError("teacher and panel coverage must match with nonempty fit/select splits")
    selected = []
    for index, prompt_hash in enumerate(sorted(panel, key=lambda value: str(panel[value]["row_id"]))):
        source, teacher = panel[prompt_hash], teachers[prompt_hash]
        target = list(teacher.get("generated_ids", []))
        if (
            not target or not teacher.get("ended_eos") or teacher.get("hit_cap")
            or float(teacher.get("ruler_official_score", 0.0)) <= 0.0
        ):
            raise ValueError(f"teacher path is not a terminated accepted output: {source['row_id']}")
        selected.append({
            **source,
            "target_ids": target,
            "target_includes_eos": True,
            "split": "fit" if index < args.fit_rows else "select",
            "teacher_arm": teacher.get("arm"),
            "teacher_official_score": float(teacher["ruler_official_score"]),
            "teacher_output_text": teacher.get("output_text"),
            "teacher_generation_row_id": teacher.get("row_id"),
        })
    args.out.mkdir(parents=True)
    rows_path = args.out / "rows.jsonl"
    with rows_path.open("w") as stream:
        for row in selected:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "status": "FROZEN_ACCEPTED_TEACHER_MARGIN_PATHS_V1",
        "task": args.task, "length": args.length,
        "rows": len(selected), "fit_rows": args.fit_rows,
        "select_rows": len(selected) - args.fit_rows,
        "panel": str(args.panel), "teacher_generations": str(args.teacher_generations),
        "rows_sha256": digest(rows_path),
        "target_contract": "accepted Native free generation including EOS; answer-token margin excludes EOS by default",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
