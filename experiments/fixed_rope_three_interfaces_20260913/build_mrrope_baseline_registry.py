#!/usr/bin/env python3
"""Build a non-destructive server registry for reusable MrRoPE baselines."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re


FORMAT = "MRROPE_BASELINE_REGISTRY_V1"
SKIP_PARTS = {
    ".git", "models", "huggingface", "hub", "wandb", "mrrope_baselines",
}
RAW_NAMES = (
    "generations.jsonl", "lm_rows.jsonl", "outputs.jsonl", "rows.jsonl",
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_rows(path: Path) -> list[dict]:
    result = []
    try:
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    result.append(value)
    except (OSError, json.JSONDecodeError):
        return []
    return result


def source_id(root: Path, source: Path) -> str:
    relative = str(source.relative_to(root))
    slug = re.sub(r"[^a-z0-9]+", "-", relative.lower()).strip("-")[-96:]
    suffix = hashlib.sha256(relative.encode()).hexdigest()[:10]
    return f"{slug}-{suffix}"


def ensure_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise ValueError(f"registry link drift: {link} -> {link.resolve()} != {target}")
        return
    if link.exists():
        raise FileExistsError(link)
    link.symlink_to(target, target_is_directory=target.is_dir())


def raw_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix == ".jsonl" else []
    paths = [source / name for name in RAW_NAMES if (source / name).is_file()]
    paths.extend(
        path for path in source.glob("*.jsonl")
        if path.stem.lower() in {"mr", "mrpro", "mrrope"} and path not in paths
    )
    return paths


def summarize_source(root: Path, source: Path) -> dict:
    rows = []
    raws = raw_paths(source)
    for path in raws:
        rows.extend(read_rows(path))
    tasks = sorted({str(row["task"]) for row in rows if row.get("task") is not None})
    lengths = sorted({
        int(row.get("length_cap", row.get("length")))
        for row in rows
        if row.get("length_cap", row.get("length")) is not None
    })
    status = {}
    if source.is_dir():
        for name in ("status.json", "summary.json", "manifest.json", "contract.json"):
            if (source / name).is_file():
                status[name] = read_json(source / name)
    status_values = [str(value.get("status", "")).upper() for value in status.values()]
    complete = "COMPLETE" in status_values
    name = (source.stem if source.is_file() else source.name).lower()
    pure_name = bool(re.fullmatch(r"(?:run[_-])?(?:mr|mrpro|mrrope)(?:[_-].*)?", name))
    return {
        "id": source_id(root, source),
        "source_path": str(source),
        "kind": "directory" if source.is_dir() else "file",
        "raw_paths": [str(path) for path in raws],
        "rows_observed": len(rows),
        "tasks": tasks,
        "lengths": lengths,
        "status_receipts": status,
        "complete_receipt_found": complete,
        "name_looks_like_pure_mrrope": pure_name,
        "reuse_class": (
            "raw_reusable_with_exact_matching_contract"
            if complete and raws and pure_name
            else "catalog_only_identity_check_required"
        ),
    }


def discover(root: Path) -> list[Path]:
    named_dirs = []
    named_files = []
    for current, directory_names, file_names in os.walk(root):
        current_path = Path(current)
        directory_names[:] = [
            name for name in directory_names
            if name not in SKIP_PARTS and not any(part in SKIP_PARTS for part in current_path.parts)
        ]
        for name in directory_names:
            lowered = name.lower()
            path = current_path / name
            if ("mrpro" in lowered or "mrrope" in lowered or lowered == "mr") and any(
                (path / receipt).is_file()
                for receipt in (*RAW_NAMES, "status.json", "summary.json", "contract.json")
            ):
                named_dirs.append(path)
        for name in file_names:
            lowered = name.lower()
            path = current_path / name
            if (
                ("mrpro" in lowered or "mrrope" in lowered or Path(name).stem.lower() == "mr")
                and path.suffix in {".json", ".jsonl"}
                and not any(parent in SKIP_PARTS for parent in path.parts)
            ):
                named_files.append(path)
    covered = [path.resolve() for path in named_dirs]
    unique = list(named_dirs)
    for path in named_files:
        resolved = path.resolve()
        if not any(parent == resolved or parent in resolved.parents for parent in covered):
            unique.append(path)
    return sorted(set(unique), key=str)


def curated_bundles(root: Path, out: Path) -> list[dict]:
    specs = [
        {
            "id": "llama3_8b_s4_full13_ppl46_mrpro",
            "model": "Meta-Llama-3-8B-Instruct",
            "scale": 4,
            "band": [18, 35],
            "gain": 1.138629436111989,
            "protocol": "Full-13 x 8K/16K/32K x 10 rows plus PPL46 x 3 lengths",
            "expected_generation_rows": 390,
            "expected_lm_rows": 138,
            "links": {
                "run": root / "today_rope_plan_20260914/tailspline_llama_s4_classic/runs/mrpro",
                "table.json": root / "today_rope_plan_20260914/tailspline_llama_s4_classic/tables/mrpro.json",
                "full13_rows.jsonl": root / "today_rope_plan_20260914/tailspline_llama_s4_classic/assets/full13/rows.jsonl",
                "full13_manifest.json": root / "today_rope_plan_20260914/tailspline_llama_s4_classic/assets/full13/manifest.json",
                "ppl46_lm.npy": root / "today_rope_plan_20260914/tailspline_llama_s4_classic/assets/ppl46/lm.npy",
                "ppl46_manifest.json": root / "today_rope_plan_20260914/tailspline_llama_s4_classic/assets/ppl46/manifest.json",
            },
        },
        {
            "id": "llama3_8b_s4_core6_8k16k32k6_mrpro",
            "model": "Meta-Llama-3-8B-Instruct",
            "scale": 4,
            "band": [18, 35],
            "gain": 1.138629436111989,
            "protocol": "Core-6 x 8K/16K/32K x 6 rows",
            "expected_generation_rows": 108,
            "expected_lm_rows": 0,
            "links": {
                "run_8k32k": root / "today_rope_plan_20260914/tailspline_llama_s4_first/runs/mrpro",
                "run_16k": root / "today_rope_plan_20260914/tailspline_llama_s4_complete108/runs/mrpro_16k",
                "table.json": root / "today_rope_plan_20260914/tailspline_llama_s4_first/tables/mrpro.json",
                "panel_8k32k.jsonl": root / "fixed_rope_three_interfaces_20260913/panels/llama_low108/screen.jsonl",
                "panel_16k.jsonl": root / "fixed_rope_three_interfaces_20260913/panels/llama_s4_core6_fresh6_16k32k_seed20260917_v2/screen.jsonl",
            },
        },
        {
            "id": "olmo2_1b_s4_full13_ppl46_mrpro",
            "model": "OLMo-2-0425-1B-Instruct",
            "scale": 4,
            "band": [14, 32],
            "gain": 1.138629436111989,
            "protocol": "Full-13 x 4K/8K/16K x 10 rows plus PPL46 x 3 lengths",
            "expected_generation_rows": 390,
            "expected_lm_rows": 138,
            "links": {
                "run": root / "today_rope_plan_20260914/tailspline_olmo_s4_classic/runs/mrpro",
                "table.json": root / "today_rope_plan_20260914/tailspline_olmo_s4_classic/tables/mrpro.json",
                "full13_rows.jsonl": root / "today_rope_plan_20260914/tailspline_olmo_s4_classic/assets/full13/rows.jsonl",
                "full13_manifest.json": root / "today_rope_plan_20260914/tailspline_olmo_s4_classic/assets/full13/manifest.json",
                "ppl46_lm.npy": root / "today_rope_plan_20260914/tailspline_olmo_s4_classic/assets/ppl46/lm.npy",
                "ppl46_manifest.json": root / "today_rope_plan_20260914/tailspline_olmo_s4_classic/assets/ppl46/manifest.json",
            },
        },
    ]
    bundles = []
    for spec in specs:
        bundle_dir = out / "current" / spec["id"]
        present = {}
        for name, target in spec.pop("links").items():
            present[name] = target.exists()
            if target.exists():
                ensure_link(bundle_dir / name, target)
        run = bundle_dir / "run"
        run_status = read_json(run / "status.json") if run.is_dir() else {}
        if spec["id"] == "llama3_8b_s4_core6_8k16k32k6_mrpro":
            status_8k32k = read_json(bundle_dir / "run_8k32k/status.json")
            status_16k = read_json(bundle_dir / "run_16k/status.json")
            ready = (
                status_8k32k == {"status": "COMPLETE", "rows": 72, "lm_rows": 0}
                and status_16k == {"status": "COMPLETE", "rows": 36, "lm_rows": 0}
            )
            run_status = {"components": {"8k32k": status_8k32k, "16k": status_16k}}
        else:
            ready = (
                run_status.get("status") == "COMPLETE"
                and run_status.get("rows") == spec["expected_generation_rows"]
                and run_status.get("lm_rows") == spec["expected_lm_rows"]
            )
        record = {
            **spec,
            "linked_assets": present,
            "run_status": run_status,
            "ready_for_score_reuse": ready,
            "reuse_rule": "exact checkpoint, prompt hashes, table/gain, decoder and scorer must match",
        }
        atomic_json(bundle_dir / "manifest.json", record)
        bundles.append(record)
    return bundles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--out", type=Path, default=Path("/root/autodl-tmp/mrrope_baselines"))
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    legacy = []
    for source in discover(root):
        record = summarize_source(root, source)
        ensure_link(out / "legacy" / record["id"], source)
        legacy.append(record)
    bundles = curated_bundles(root, out)
    counts = Counter(record["reuse_class"] for record in legacy)
    registry = {
        "status": FORMAT,
        "root": str(root),
        "curated_bundles": bundles,
        "legacy_entries": legacy,
        "counts": {"curated": len(bundles), "legacy": len(legacy), **dict(counts)},
        "policy": (
            "Reuse a score only when checkpoint, prompt/data identity, table, gain, decoder, "
            "precision and scorer match. Otherwise reuse the asset for provenance only."
        ),
    }
    atomic_json(out / "registry.json", registry)
    atomic_json(out / "reusable.json", {
        "status": FORMAT,
        "curated_bundles": [
            record for record in bundles if record["ready_for_score_reuse"]
        ],
        "legacy_raw_candidates": [
            record for record in legacy
            if record["reuse_class"] == "raw_reusable_with_exact_matching_contract"
        ],
        "warning": registry["policy"],
    })
    (out / "README.md").write_text(
        "# MrRoPE baseline registry\n\n"
        "`current/` contains strict reusable bundles. `legacy/` is a non-destructive symlink "
        "catalog of every discovered MrPro/MrRoPE-named raw asset. Read `registry.json` before "
        "reuse: a score is reusable only under an exact scientific identity match. Refreshing "
        "the registry never moves or deletes source evidence.\n"
    )
    print(json.dumps({"status": FORMAT, **registry["counts"], "out": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
