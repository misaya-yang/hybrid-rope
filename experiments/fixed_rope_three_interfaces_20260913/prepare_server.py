#!/usr/bin/env python3
"""Freeze the current 32 GiB server pipeline without launching a model.

The prepared order deliberately confirms the existing Llama interval candidate
before spending GPU time on conditional band/depth/shape interventions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .pipeline import (
    DRAFT_FORMAT,
    atomic_json,
    atomic_jsonl,
    file_sha256,
    freeze_contract,
    plan_queue,
    read_jsonl,
    validate_panel_rows,
)
from .tables import (
    build_analytic,
    build_depth_control,
    find_table,
    make_receipt,
    model_geometry,
    read_json,
    validate_table,
)


SCORER = "RULER-official-complete-output-contains-current"
NEW_DECODER = "greedy-row-budget-terminal-eos-use-cache-chunked-prefill"
PRECISION = "bf16-sdpa-chunked-prefill-fp32-rope"


def write_once_or_equal(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"refusing to replace a different frozen artifact: {path}")
        return
    atomic_json(path, value)


def write_jsonl_once_or_equal(path: Path, rows: list[dict]) -> None:
    if path.exists():
        if read_jsonl(path) != rows:
            raise ValueError(f"refusing to replace a different frozen panel: {path}")
        return
    atomic_jsonl(path, rows)


def receipt_from_saved(
    *, source: Path, config_path: Path, candidate_id: str, model_id: str,
    role: str, scale: float, changed_variables: list[str] | None = None,
) -> dict:
    config = read_json(config_path)
    geometry = model_geometry(config)
    payload = read_json(source)
    table = find_table(payload)
    values, gain = validate_table(table, pairs=int(geometry["pairs"]))
    construction = {
        **table.get("construction", {}),
        "method": "wrapped_exact_saved_table",
        "wrapped_source_status": payload.get("status"),
        "wrapped_source_label": payload.get("label"),
    }
    return make_receipt(
        candidate_id=candidate_id, model_id=model_id, role=role, scale=scale,
        geometry=geometry, values=values, gain=gain, construction=construction,
        source=str(source), changed_variables=changed_variables or [],
    )


def receipt_analytic(
    *, config_path: Path, candidate_id: str, model_id: str, role: str,
    method: str, scale: float, low: int | None = None, high: int | None = None,
    depth: float = 1.0, parent_candidate_id: str | None = None,
    changed_variables: list[str] | None = None,
) -> dict:
    config = read_json(config_path)
    geometry = model_geometry(config)
    values, gain, construction = build_analytic(
        config, method=method, scale=scale, low=low, high=high,
        depth=depth, gain=None,
    )
    return make_receipt(
        candidate_id=candidate_id, model_id=model_id, role=role, scale=scale,
        geometry=geometry, values=values, gain=gain, construction=construction,
        source=f"analytic:{method}", parent_candidate_id=parent_candidate_id,
        changed_variables=changed_variables or [],
    )


def receipt_depth(
    *, config_path: Path, parent: dict, candidate_id: str, model_id: str,
    scale: float, depth: float,
) -> dict:
    config = read_json(config_path)
    geometry = model_geometry(config)
    values, gain, construction = build_depth_control(
        config, parent, scale=scale, depth=depth, gain=None,
    )
    return make_receipt(
        candidate_id=candidate_id, model_id=model_id, role="control", scale=scale,
        geometry=geometry, values=values, gain=gain, construction=construction,
        source=f"normalized depth control of {parent['candidate_id']}",
        parent_candidate_id=parent["candidate_id"], changed_variables=["profile_depth"],
    )


def freeze_subset(
    *, source_panel: Path, source_manifest: Path, out_dir: Path,
    panel_id: str, rows_per_cell: int, lengths: set[int] | None = None,
    start_per_cell: int = 0, tasks: set[str] | None = None,
) -> tuple[Path, Path]:
    source_rows = validate_panel_rows(read_jsonl(source_panel), panel_id + ":source")
    source_meta = read_json(source_manifest)
    selected = []
    counts: Counter[tuple[str, int]] = Counter()
    selected_tasks = [
        str(value) for value in source_meta["tasks"]
        if tasks is None or str(value) in tasks
    ]
    source_lengths = source_meta.get("lengths", source_meta.get("physical_caps", []))
    selected_lengths = [
        int(value) for value in source_lengths
        if lengths is None or int(value) in lengths
    ]
    visited: Counter[tuple[str, int]] = Counter()
    for row in source_rows:
        cell = (str(row["task"]), int(row["length_cap"]))
        if cell[0] not in selected_tasks or cell[1] not in selected_lengths:
            continue
        index_in_cell = visited[cell]
        visited[cell] += 1
        if index_in_cell < start_per_cell:
            continue
        if counts[cell] >= rows_per_cell:
            continue
        selected.append(row)
        counts[cell] += 1
    expected = {(task, length) for task in selected_tasks for length in selected_lengths}
    if set(counts) != expected or any(counts[cell] != rows_per_cell for cell in expected):
        raise ValueError(f"cannot freeze balanced subset {panel_id}: {dict(counts)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / "screen.jsonl"
    manifest_path = out_dir / "manifest.json"
    write_jsonl_once_or_equal(panel_path, selected)
    manifest = {
        "status": "FROZEN",
        "panel_id": panel_id,
        "rows": len(selected),
        "tasks": selected_tasks,
        "lengths": selected_lengths,
        "rows_per_cell": rows_per_cell,
        "cell_counts": {
            f"{task}/{length}": counts[(task, length)]
            for task in selected_tasks for length in selected_lengths
        },
        "panel_sha256": file_sha256(panel_path),
        "source_panel": str(source_panel),
        "source_panel_sha256": file_sha256(source_panel),
        "selection": (
            f"rows {start_per_cell}:{start_per_cell + rows_per_cell} in frozen source order "
            "within every task/length cell"
        ),
        "strict_subset": True,
    }
    write_once_or_equal(manifest_path, manifest)
    return panel_path, manifest_path


def job(
    *, job_id: str, stage: str, priority: int, model_id: str,
    panel_id: str, table_id: str, run_root: Path,
    result_sources: list[str] | None = None,
    reuse_receipts: list[str] | None = None,
    source_arm_labels: list[str] | None = None,
    source_panel_superset: bool = False,
    decoder: str = NEW_DECODER,
    allow_unverified_legacy_gain: bool = False,
    legacy_gain_justification: str | None = None,
) -> dict:
    return {
        "job_id": job_id,
        "stage": stage,
        "priority": priority,
        "model_id": model_id,
        "panel_id": panel_id,
        "table_id": table_id,
        "output_dir": str(run_root / job_id),
        "result_sources": result_sources or [],
        "reuse_receipts": reuse_receipts or [],
        "source_arm_labels": source_arm_labels or [],
        "allow_source_panel_superset": source_panel_superset,
        "allow_unverified_legacy_gain": allow_unverified_legacy_gain,
        "legacy_gain_justification": legacy_gain_justification,
        "decoder": decoder,
        "scorer": SCORER,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--repo-root", type=Path, default=Path("/root/autodl-tmp/hybrid-rope"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("/root/autodl-tmp/fixed_rope_three_interfaces_20260913"),
    )
    args = parser.parse_args()
    data = args.data_root.resolve()
    out = args.out.resolve()
    tables_dir = out / "tables"
    panels_dir = out / "panels"
    run_root = out / "runs"
    queue_dir = out / "queue"
    for directory in (tables_dir, panels_dir, run_root, queue_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_paths = {
        "llama3_8b_s8": data / "models/Meta-Llama-3-8B-Instruct",
        "olmo2_1b_s4": data / "olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct",
        "qwen25_1p5b_s2": data / "qwen25_1p5b_32k",
    }
    config_paths = {key: value / "config.json" for key, value in model_paths.items()}
    missing_configs = [str(path) for path in config_paths.values() if not path.is_file()]
    if missing_configs:
        raise FileNotFoundError(f"missing model configs: {missing_configs}")

    source_panels = {
        "llama_mini": data / "band_mini_20260913/llama_s8/frozen_324",
        "olmo_mini": data / "band_mini_20260913/olmo_s4/frozen_324",
        "qwen_mini": data / "band_mini_20260913/qwen_s2/frozen_324",
    }
    panel_specs = {}
    for key, directory in source_panels.items():
        panel_specs[key] = (directory / "screen.jsonl", directory / "manifest.json")
        if not all(path.is_file() for path in panel_specs[key]):
            raise FileNotFoundError(f"missing frozen panel {key}")

    llama_low = freeze_subset(
        source_panel=panel_specs["llama_mini"][0], source_manifest=panel_specs["llama_mini"][1],
        out_dir=panels_dir / "llama_low108", panel_id="llama3_8b_s8_core6_low6_20260913",
        rows_per_cell=6,
    )
    llama_native = freeze_subset(
        source_panel=panel_specs["llama_mini"][0], source_manifest=panel_specs["llama_mini"][1],
        out_dir=panels_dir / "llama_native108", panel_id="llama3_8b_native_core6_18_20260913",
        rows_per_cell=18, lengths={8192},
    )
    olmo_low = freeze_subset(
        source_panel=panel_specs["olmo_mini"][0], source_manifest=panel_specs["olmo_mini"][1],
        out_dir=panels_dir / "olmo_low108", panel_id="olmo2_1b_s4_core6_low6_20260913",
        rows_per_cell=6,
    )
    qwen_low = freeze_subset(
        source_panel=panel_specs["qwen_mini"][0], source_manifest=panel_specs["qwen_mini"][1],
        out_dir=panels_dir / "qwen_low108", panel_id="qwen25_1p5b_s2_core6_low6_20260913",
        rows_per_cell=6,
    )

    table_receipts: dict[str, dict] = {}
    table_receipts["llama_solver_s8"] = receipt_from_saved(
        source=data / "fixed_table_interval_20260913/SolverProfile_g8_table.json",
        config_path=config_paths["llama3_8b_s8"], candidate_id="llama_solver_s8",
        model_id="llama3_8b_s8", role="candidate", scale=8.0,
    )
    table_receipts["llama_c42_s8"] = receipt_from_saved(
        source=data / "fixed_table_interval_20260913/C42FullProfile_g8_table.json",
        config_path=config_paths["llama3_8b_s8"], candidate_id="llama_c42_s8",
        model_id="llama3_8b_s8", role="control", scale=8.0,
    )
    for table_id, method, role in (
        ("llama_native", "native", "native"),
        ("llama_bm_s8", "bm", "baseline"),
        ("llama_mrpro_s8", "mrpro", "baseline"),
        ("llama_static_yarn_s8", "yarn", "control"),
    ):
        table_receipts[table_id] = receipt_analytic(
            config_path=config_paths["llama3_8b_s8"], candidate_id=table_id,
            model_id="llama3_8b_s8", role=role, method=method, scale=8.0,
        )
    table_receipts["llama_c42_dlog_s8"] = receipt_depth(
        config_path=config_paths["llama3_8b_s8"], parent=table_receipts["llama_c42_s8"],
        candidate_id="llama_c42_dlog_s8", model_id="llama3_8b_s8", scale=8.0,
        depth=math.log(4.5) / math.log(8.0),
    )

    table_receipts["olmo_c42_s4"] = receipt_from_saved(
        source=data / "olmo_minimal_band_s4_20260913/axial/tables/C42Band14_31_s4.json",
        config_path=config_paths["olmo2_1b_s4"], candidate_id="olmo_c42_s4",
        model_id="olmo2_1b_s4", role="candidate", scale=4.0,
    )
    for table_id, method, role, low, high in (
        ("olmo_bm_s4", "bm", "baseline", None, None),
        ("olmo_mrpro_s4", "mrpro", "baseline", None, None),
        ("olmo_bm_band14_31_s4", "bm", "control", 14, 31),
    ):
        table_receipts[table_id] = receipt_analytic(
            config_path=config_paths["olmo2_1b_s4"], candidate_id=table_id,
            model_id="olmo2_1b_s4", role=role, method=method, scale=4.0,
            low=low, high=high,
        )
    table_receipts["olmo_c42_dlog_s4"] = receipt_depth(
        config_path=config_paths["olmo2_1b_s4"], parent=table_receipts["olmo_c42_s4"],
        candidate_id="olmo_c42_dlog_s4", model_id="olmo2_1b_s4", scale=4.0,
        depth=math.log(2.5) / math.log(4.0),
    )
    table_receipts["olmo_bm_band14_31_dlog_s4"] = receipt_depth(
        config_path=config_paths["olmo2_1b_s4"],
        parent=table_receipts["olmo_bm_band14_31_s4"],
        candidate_id="olmo_bm_band14_31_dlog_s4", model_id="olmo2_1b_s4", scale=4.0,
        depth=math.log(2.5) / math.log(4.0),
    )

    qwen_table_root = data / "qwen15_minimal_band_s2_20260913/tables"
    for table_id, filename, role in (
        ("qwen_c42_band22_39_s2", "C42Band22_39_s2.json", "candidate"),
        ("qwen_bm_s2", "BM_s2.json", "baseline"),
        ("qwen_mrpro_s2", "MrPro_s2.json", "baseline"),
    ):
        table_receipts[table_id] = receipt_from_saved(
            source=qwen_table_root / filename, config_path=config_paths["qwen25_1p5b_s2"],
            candidate_id=table_id, model_id="qwen25_1p5b_s2", role=role, scale=2.0,
        )
    table_receipts["qwen_c42_dlog_s2"] = receipt_depth(
        config_path=config_paths["qwen25_1p5b_s2"],
        parent=table_receipts["qwen_c42_band22_39_s2"],
        candidate_id="qwen_c42_dlog_s2", model_id="qwen25_1p5b_s2", scale=2.0,
        depth=math.log(1.5) / math.log(2.0),
    )

    table_paths = {}
    for table_id, receipt in table_receipts.items():
        path = tables_dir / f"{table_id}.json"
        write_once_or_equal(path, receipt)
        table_paths[table_id] = str(path)

    olmo_cache_root = data / "band_mini_20260913/olmo_s4/cache"
    olmo_reuse = {
        "olmo_c42_s4": (
            [str(olmo_cache_root / "C42Band14_31_s4.jsonl")],
            [str(olmo_cache_root / "C42Band14_31_s4_coverage.json")],
            ["C42Band14_31_s4"],
        ),
        "olmo_bm_s4": (
            [str(olmo_cache_root / "BM_g4.jsonl")],
            [str(olmo_cache_root / "BM_g4_coverage.json")],
            ["BM_g4"],
        ),
        "olmo_mrpro_s4": (
            [str(olmo_cache_root / "MrPro_g4.jsonl")],
            [str(olmo_cache_root / "MrPro_g4_coverage.json")],
            ["MrPro_g4"],
        ),
    }

    qwen_history = data / "qwen15_minimal_band_s2_20260913"
    qwen_gap = data / "band_mini_20260913/qwen_s2"
    qwen_labels = {
        "qwen_c42_band22_39_s2": "C42Band22_39_s2",
        "qwen_bm_s2": "BM_s2",
        "qwen_mrpro_s2": "MrPro_s2",
    }
    qwen_reuse = {}
    for table_id, label in qwen_labels.items():
        gap_label = f"{label}_mini_gap"
        gap_dir = {
            "BM_s2": "run_BM_s2_missing270",
            "MrPro_s2": "run_MrPro_s2_missing270",
            "C42Band22_39_s2": "run_C42Band22_39_s2_missing270",
        }[label]
        sources = [
            str(qwen_history / f"runs/{label}/generation/generations.jsonl"),
            str(qwen_history / f"confirm_remaining/{label}/generation/generations.jsonl"),
            str(qwen_gap / f"{gap_dir}/generations.jsonl"),
        ]
        receipts = [str(Path(path).parent / "contract.json") for path in sources]
        qwen_reuse[table_id] = (sources, receipts, [label, gap_label])

    models = {
        "llama3_8b_s8": {
            "model_path": str(model_paths["llama3_8b_s8"]),
            "revision": "Meta-Llama-3-8B-Instruct-local-frozen",
            "tokenizer_template": "frozen-Llama3-prompt-ids; local tokenizer exact decode",
            "precision_arithmetic": PRECISION,
            "prefill_chunk_size": 8192,
        },
        "olmo2_1b_s4": {
            "model_path": str(model_paths["olmo2_1b_s4"]),
            "revision": "olmo2-1b-48d788e",
            "tokenizer_template": "olmo2-tokenizer-chat-v1",
            "precision_arithmetic": "bf16-sdpa-flash-stock-generate",
            "prefill_chunk_size": 0,
        },
        "qwen25_1p5b_s2": {
            "model_path": str(model_paths["qwen25_1p5b_s2"]),
            "revision": "qwen25-1p5b-32k-local-frozen",
            "tokenizer_template": "frozen-Qwen2.5-prompt-ids; local tokenizer exact decode",
            "precision_arithmetic": PRECISION,
            "prefill_chunk_size": 8192,
        },
    }
    panels = {
        "llama_mini324": {"model_id": "llama3_8b_s8", "path": str(panel_specs["llama_mini"][0]), "manifest_path": str(panel_specs["llama_mini"][1])},
        "llama_low108": {"model_id": "llama3_8b_s8", "path": str(llama_low[0]), "manifest_path": str(llama_low[1])},
        "llama_native108": {"model_id": "llama3_8b_s8", "path": str(llama_native[0]), "manifest_path": str(llama_native[1])},
        "olmo_low108": {"model_id": "olmo2_1b_s4", "path": str(olmo_low[0]), "manifest_path": str(olmo_low[1])},
        "qwen_mini324": {"model_id": "qwen25_1p5b_s2", "path": str(panel_specs["qwen_mini"][0]), "manifest_path": str(panel_specs["qwen_mini"][1])},
        "qwen_low108": {"model_id": "qwen25_1p5b_s2", "path": str(qwen_low[0]), "manifest_path": str(qwen_low[1])},
    }

    jobs = []
    for priority, table_id in enumerate((
        "llama_solver_s8", "llama_bm_s8", "llama_mrpro_s8", "llama_static_yarn_s8",
    )):
        jobs.append(job(
            job_id=f"paper_{table_id}_mini324", stage="paper_confirm", priority=priority,
            model_id="llama3_8b_s8", panel_id="llama_mini324", table_id=table_id,
            run_root=run_root,
        ))
    for priority, table_id in enumerate(("llama_solver_s8", "llama_native"), start=10):
        jobs.append(job(
            job_id=f"paper_{table_id}_native108", stage="paper_confirm", priority=priority,
            model_id="llama3_8b_s8", panel_id="llama_native108", table_id=table_id,
            run_root=run_root,
        ))

    for priority, table_id in enumerate((
        "olmo_c42_s4", "olmo_c42_dlog_s4", "olmo_bm_band14_31_s4",
        "olmo_bm_band14_31_dlog_s4", "olmo_bm_s4", "olmo_mrpro_s4",
    )):
        reuse = olmo_reuse.get(table_id, ([], [], []))
        jobs.append(job(
            job_id=f"depth_{table_id}_low108", stage="olmo_depth_shape", priority=priority,
            model_id="olmo2_1b_s4", panel_id="olmo_low108", table_id=table_id,
            run_root=run_root, result_sources=reuse[0], reuse_receipts=reuse[1],
            source_arm_labels=reuse[2], source_panel_superset=bool(reuse[0]),
            decoder="greedy-row-budget-eos100257-pad100277-use-cache",
            allow_unverified_legacy_gain=bool(reuse[0]),
            legacy_gain_justification=(
                "permanent OLMo mini cache was frozen from the named built-in arm; its legacy "
                "coverage receipt hashes frequencies but did not serialize gain separately"
                if reuse[0] else None
            ),
        ))

    for priority, table_id in enumerate((
        "qwen_c42_band22_39_s2", "qwen_bm_s2", "qwen_mrpro_s2",
    )):
        reuse = qwen_reuse[table_id]
        jobs.append(job(
            job_id=f"qwen_confirm_{table_id}_mini324", stage="qwen_confirm_depth",
            priority=priority, model_id="qwen25_1p5b_s2", panel_id="qwen_mini324",
            table_id=table_id, run_root=run_root, result_sources=reuse[0],
            reuse_receipts=reuse[1], source_arm_labels=reuse[2],
        ))
    for priority, table_id in enumerate((
        "qwen_c42_band22_39_s2", "qwen_c42_dlog_s2", "qwen_bm_s2", "qwen_mrpro_s2",
    ), start=10):
        reuse = qwen_reuse.get(table_id, ([], [], []))
        jobs.append(job(
            job_id=f"qwen_depth_{table_id}_low108", stage="qwen_confirm_depth",
            priority=priority, model_id="qwen25_1p5b_s2", panel_id="qwen_low108",
            table_id=table_id, run_root=run_root, result_sources=reuse[0],
            reuse_receipts=reuse[1], source_arm_labels=reuse[2],
            source_panel_superset=bool(reuse[0]),
        ))

    for priority, table_id in enumerate((
        "llama_c42_s8", "llama_c42_dlog_s8", "llama_solver_s8",
        "llama_bm_s8", "llama_mrpro_s8",
    )):
        jobs.append(job(
            job_id=f"llama_depth_{table_id}_low108", stage="llama_depth", priority=priority,
            model_id="llama3_8b_s8", panel_id="llama_low108", table_id=table_id,
            run_root=run_root,
        ))

    comparisons = [
        {
            "comparison_id": "llama_solver_range_vs_baselines",
            "panel_id": "llama_mini324", "candidate": "llama_solver_s8",
            "baselines": ["llama_bm_s8", "llama_mrpro_s8"],
            "diagnostic_controls": ["llama_static_yarn_s8"],
            "purpose": "paper-level same-runner interval confirmation; static YaRN is diagnostic, not a matched SFT claim",
        },
        {
            "comparison_id": "llama_solver_native_retention",
            "panel_id": "llama_native108", "candidate": "llama_solver_s8",
            "baselines": ["llama_native"],
            "purpose": "separate Native-window preservation from long-window AUC",
        },
        {
            "comparison_id": "olmo_depth_shape_conditional",
            "panel_id": "olmo_low108", "candidate": "olmo_c42_dlog_s4",
            "baselines": [
                "olmo_c42_s4", "olmo_bm_band14_31_dlog_s4",
                "olmo_bm_band14_31_s4", "olmo_bm_s4", "olmo_mrpro_s4",
            ],
            "purpose": "conditional depth signal plus matched-band shape interaction; low is not a winner gate",
        },
        {
            "comparison_id": "qwen_c42_range_vs_baselines",
            "panel_id": "qwen_mini324", "candidate": "qwen_c42_band22_39_s2",
            "baselines": ["qwen_bm_s2", "qwen_mrpro_s2"],
            "purpose": "complete the current cross-model mini on its frozen development panel",
        },
        {
            "comparison_id": "qwen_depth_conditional",
            "panel_id": "qwen_low108", "candidate": "qwen_c42_dlog_s2",
            "baselines": ["qwen_c42_band22_39_s2", "qwen_bm_s2", "qwen_mrpro_s2"],
            "purpose": "first real-task intervention on profile depth at S=2; retrospective development evidence",
        },
        {
            "comparison_id": "llama_depth_conditional",
            "panel_id": "llama_low108", "candidate": "llama_c42_dlog_s8",
            "baselines": ["llama_c42_s8", "llama_solver_s8", "llama_bm_s8", "llama_mrpro_s8"],
            "purpose": "test depth without retuning band or gain; low is diagnostic only",
        },
    ]

    draft = {
        "status": DRAFT_FORMAT,
        "name": "range-optimal fixed RoPE confirmation and three-interface interventions",
        "repo_root": str(args.repo_root.resolve()),
        "stage_order": [
            "paper_confirm", "olmo_depth_shape", "qwen_confirm_depth",
            "llama_depth", "checkpoint_replay",
        ],
        "models": models,
        "panels": panels,
        "tables": table_paths,
        "jobs": jobs,
        "comparisons": comparisons,
        "execution_policy": {
            "launch_now": False,
            "one_model_resident_sequential_tables": True,
            "no_score_based_automatic_stop": True,
            "low_is_diagnostic_not_a_gate": True,
            "replay_is_not_authorized_to_filter_task_candidates": True,
            "static_yarn_scope": "diagnostic zero-training table only; not matched to official YaRN SFT",
        },
    }
    draft_path = out / "draft.json"
    write_once_or_equal(draft_path, draft)
    contract_path = out / "contract.json"
    frozen = freeze_contract(draft_path, contract_path)
    queue = plan_queue(contract_path, queue_dir)
    print(json.dumps({
        "status": queue["status"],
        "contract": str(contract_path),
        "jobs": len(frozen["jobs"]),
        "queued_jobs": len(queue["queue"]),
        "remaining_rows": sum(job["remaining_rows"] for job in queue["queue"]),
        "launch_now": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
