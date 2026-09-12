"""Migrate erroneous NIAH character-offset cluster IDs without rerunning logits.

The original data builder used upstream ``raw['index']`` as a source identity.
NVIDIA RULER's NIAH generator overwrites that field with the character offset
of the first answer, so independent prompts can collide.  This migration is a
metadata-only correction keyed by the already-frozen prompt hash.  It never
changes prompt IDs, row IDs, outputs, scores, operators, or generation config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


REVISION = "planb-semantic-v2-prompt-bound"
QA_STAGE_ROW_OFFSET = {"P": 0, "S": 1_000, "V": 2_500, "H": 5_000}
STAGE_COUNTS = {"P": (4, 4), "S": (8, 4), "V": (16, 8), "H": (64, 32)}


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def atomic_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    tmp.replace(path)


def new_cluster(row, stage):
    return digest([
        REVISION, stage, row["task"], int(row["length_cap"]),
        int(row["generator_seed"]), row["row_id"], row["prompt_sha256"],
    ])


def qa_source_index(row, stage):
    if not str(row["task"]).startswith("qa_"):
        return None
    long_count, guard_count = STAGE_COUNTS[stage]
    cap = int(row["length_cap"])
    prior = 0 if cap == 8192 else guard_count if cap == 16384 \
        else guard_count + long_count
    local_index = int(str(row["row_id"]).rsplit("_", 1)[1])
    return QA_STAGE_ROW_OFFSET[stage] + prior + local_index


def migrate_stage(root, stage, repair_results=False):
    directory = Path(root) / "data" / stage
    manifest_path, rows_path = directory / "manifest.json", directory / "rows.jsonl"
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    if manifest.get("status") != "COMPLETE":
        raise ValueError(f"{stage}: refusing to migrate incomplete data")
    qa_complete = all(not str(row["task"]).startswith("qa_") or
                      row.get("qa_source_index") is not None for row in rows)
    if manifest.get("source_identity_revision") == REVISION and qa_complete:
        return {"stage": stage, "status": "ALREADY_MIGRATED",
                "rows_sha256": sha_file(rows_path)}
    old_rows_sha = sha_file(rows_path)
    old_ids = {row["row_id"]: (row.get("source_document_id"),
                                row.get("semantic_group_id")) for row in rows}
    for row in rows:
        cluster = new_cluster(row, stage)
        row.update(source_id=cluster, source_document_id=cluster,
                   group_id=cluster, semantic_group_id=cluster,
                   source_identity_revision=REVISION)
        if str(row["task"]).startswith("qa_"):
            row["qa_source_index"] = qa_source_index(row, stage)
    if len(rows) != len({row["source_document_id"] for row in rows}):
        raise ValueError(f"{stage}: migrated source IDs are not unique")
    stamp = str(int(time.time()))
    shutil.copy2(rows_path, rows_path.with_name(rows_path.name + f".pre_{REVISION}_{stamp}"))
    shutil.copy2(manifest_path,
                 manifest_path.with_name(manifest_path.name + f".pre_{REVISION}_{stamp}"))
    atomic_jsonl(rows_path, rows)
    new_rows_sha = sha_file(rows_path)
    manifest.update({
        "rows_sha256": new_rows_sha,
        "source_identity_revision": REVISION,
        "qa_identity_revision": "stage-offset-plus-local-row-v1",
        "metadata_only_repair": {
            "reason": "upstream NIAH raw index is an answer character offset, not sample ID",
            "old_rows_sha256": old_rows_sha,
            "new_rows_sha256": new_rows_sha,
            "prompts_outputs_operators_scores_changed": False,
            "repair_script_sha256": sha_file(__file__),
        },
        "independence": "distinct seed and prompt-bound semantic/source id per row",
    })
    atomic_json(manifest_path, manifest)

    migrated_results = []
    if repair_results:
        for result_dir in (Path(root) / "results" / "P_native",
                           Path(root) / "results" / "P_l0"):
            if not result_dir.exists():
                continue
            for path in sorted(result_dir.glob("*.jsonl")):
                result_rows = [json.loads(line) for line in path.read_text().splitlines() if line]
                for result in result_rows:
                    panel = next((row for row in rows if row["row_id"] == result["row_id"]), None)
                    if panel is None:
                        raise ValueError(f"{path}: result row absent from repaired panel")
                    for field in ("task", "length_cap", "prompt_sha256", "references"):
                        if result.get(field) != panel.get(field):
                            raise ValueError(f"{path}/{result['row_id']}: {field} drift")
                    cluster = panel["source_document_id"]
                    result.update(source_id=cluster, source_document_id=cluster,
                                  group_id=cluster, semantic_group_id=cluster,
                                  source_identity_revision=REVISION)
                    if str(result["task"]).startswith("qa_"):
                        result["qa_source_index"] = panel["qa_source_index"]
                backup = path.with_name(path.name + f".pre_{REVISION}_{stamp}")
                shutil.copy2(path, backup)
                atomic_jsonl(path, result_rows)
                migrated_results.append(str(path))
            summary_path = result_dir / "run_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                backup = summary_path.with_name(
                    summary_path.name + f".pre_{REVISION}_{stamp}")
                shutil.copy2(summary_path, backup)
                summary["panel_sha256"] = new_rows_sha
                summary["source_identity_revision"] = REVISION
                summary["metadata_only_repair"] = manifest["metadata_only_repair"]
                atomic_json(summary_path, summary)
    return {"stage": stage, "status": "MIGRATED", "rows": len(rows),
            "old_rows_sha256": old_rows_sha, "new_rows_sha256": new_rows_sha,
            "old_duplicate_clusters": len(rows) - len(set(old_ids.values())),
            "migrated_result_files": migrated_results}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/root/autodl-tmp/llama3_planb_20260911")
    ap.add_argument("--stages", default="P,S")
    ap.add_argument("--repair-results", default="P")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    repair = {x.strip() for x in args.repair_results.split(",") if x.strip()}
    reports = [migrate_stage(args.root, stage.strip(), stage.strip() in repair)
               for stage in args.stages.split(",") if stage.strip()]
    report = {"status": "COMPLETE", "revision": REVISION, "stages": reports,
              "completed_at": time.time()}
    atomic_json(args.out, report)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
