"""Reject incomplete, drifting, or cross-stage-leaking Plan B data panels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


TASKS = (
    "niah_single_2", "niah_multikey_2", "niah_multivalue", "niah_multiquery",
    "vt", "fwe", "qa_1", "qa_2",
)
CAPS = (8192, 16384, 32768)
COUNTS = {"P": (4, 4), "S": (8, 4), "V": (16, 8), "H": (64, 32)}
SINGLE = {"niah_single_2", "niah_multikey_2"}
MULTI = {"niah_multivalue", "niah_multiquery"}
DEPTH_TARGETS = (0.10, 0.35, 0.65, 0.90)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_digest(ids):
    return hashlib.sha256(json.dumps(
        list(ids), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def load_stage(root, stage):
    directory = Path(root) / "data" / stage
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text().splitlines()
            if line]
    if manifest.get("status") != "COMPLETE" or manifest.get("stage") != stage:
        raise ValueError(f"{stage}: manifest is not COMPLETE for this stage")
    if manifest.get("rows_sha256") != sha_file(directory / "rows.jsonl"):
        raise ValueError(f"{stage}: rows hash differs from manifest")
    long_count, guard_count = COUNTS[stage]
    expected = {(task, cap): guard_count if cap == 8192 else long_count
                for task in TASKS for cap in CAPS}
    got = Counter((row["task"], int(row["length_cap"])) for row in rows)
    if got != Counter(expected):
        raise ValueError(f"{stage}: task/cap count mismatch")
    if len(rows) != len({row["row_id"] for row in rows}):
        raise ValueError(f"{stage}: duplicate row_id")
    for identity_field in ("source_document_id", "semantic_group_id", "prompt_sha256"):
        if len(rows) != len({row.get(identity_field) for row in rows}):
            raise ValueError(f"{stage}: missing or duplicate {identity_field}")
    for row in rows:
        cap = int(row["length_cap"])
        if len(row["prompt_ids"]) != cap - 128:
            raise ValueError(f"{stage}/{row['row_id']}: prompt is not cap-128")
        if row["prompt_sha256"] != ids_digest(row["prompt_ids"]):
            raise ValueError(f"{stage}/{row['row_id']}: prompt hash mismatch")
        if not row.get("references"):
            raise ValueError(f"{stage}/{row['row_id']}: no references")
        if str(row["task"]).startswith("qa_") and row.get("qa_source_index") is None:
            raise ValueError(f"{stage}/{row['row_id']}: no stable QA source index")
        if row["task"] in SINGLE:
            if row.get("depth_profile") != "point" or \
                    row.get("depth_error_mean_abs", 1.0) > 0.10:
                raise ValueError(f"{stage}/{row['row_id']}: invalid point-depth selection")
        if row["task"] in MULTI:
            if row.get("depth_profile") not in {"dispersed", "middle_cluster"} or \
                    row.get("depth_error_mean_abs", 1.0) > 0.18:
                raise ValueError(f"{stage}/{row['row_id']}: invalid multi-depth selection")
    # Exact balance of registered target/profile labels per task/cap.
    for task in SINGLE:
        for cap in CAPS:
            selected = [tuple(row["depth_target"]) for row in rows
                        if row["task"] == task and row["length_cap"] == cap]
            want = [(DEPTH_TARGETS[i % 4],) for i in range(len(selected))]
            if Counter(selected) != Counter(want):
                raise ValueError(f"{stage}/{task}/{cap}: depth targets are not balanced")
    for task in MULTI:
        for cap in CAPS:
            labels = Counter(row["depth_profile"] for row in rows
                             if row["task"] == task and row["length_cap"] == cap)
            if abs(labels["dispersed"] - labels["middle_cluster"]) > 1:
                raise ValueError(f"{stage}/{task}/{cap}: multi-depth profiles are not balanced")
    return manifest, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/root/autodl-tmp/llama3_planb_20260911")
    ap.add_argument("--stages", default="P,S")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    stages = [x.strip() for x in args.stages.split(",") if x.strip()]
    loaded = {stage: load_stage(args.root, stage) for stage in stages}
    for i, left in enumerate(stages):
        lrows = loaded[left][1]
        for right in stages[i + 1:]:
            rrows = loaded[right][1]
            for field in ("source_document_id", "semantic_group_id", "prompt_sha256"):
                overlap = {r[field] for r in lrows} & {r[field] for r in rrows}
                if overlap:
                    raise ValueError(f"{left}/{right}: cross-stage {field} overlap")
            for task in ("qa_1", "qa_2"):
                li = {r["qa_source_index"] for r in lrows if r["task"] == task}
                ri = {r["qa_source_index"] for r in rrows if r["task"] == task}
                if li & ri:
                    raise ValueError(f"{left}/{right}/{task}: reused QA questions")
    report = {
        "all_pass": True, "stages": stages,
        "rows": {stage: len(loaded[stage][1]) for stage in stages},
        "rows_sha256": {stage: loaded[stage][0]["rows_sha256"] for stage in stages},
        "checks": [
            "complete exact task-cap counts", "cap-minus-128 prompts",
            "row and prompt identity", "balanced registered depth targets",
            "cross-stage semantic/source/prompt isolation", "QA question isolation",
        ],
        "validator_sha256": sha_file(__file__),
    }
    atomic_json(args.out, report)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
