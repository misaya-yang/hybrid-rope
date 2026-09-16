#!/usr/bin/env python3
"""CPU-only, world-clustered paired report for the native mechanism panel.

Example: python -m experiments.native_enhancement_oral_20260915.report \
    --panel assets/inputs.jsonl --run native=runs/native --run ncp=runs/ncp \
    --out reports/ncp_vs_native.json

The second --run is the candidate and all effects are candidate minus baseline.
Scores use literal whole-output equality after trimming outer whitespace only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


TASKS = ("native_binding", "native_chain")
LENGTHS = (1024, 2048, 4096)
CONDITIONS = {"native_binding": ("layout", "near", "far"),
              "native_chain": ("graph", "base", "rewired")}
PAIR_FIELDS = ("task", "length_cap", "group_id", "prompt_sha256", "references", "input_tokens")
RUNTIME_FIELDS = ("base_arm", "split", "unadapted", "checkpoint_arm", "generation_length_caps",
                  "lm_enabled", "limit_per_cell", "prefill_chunk_size",
                  "generation_prefill_strategy", "batch_size", "runtime_versions", "row_split")
METRICS = ("exact_match", "all_four_correct", "query_pair_both_correct", "condition_0_exact",
           "condition_1_exact", "condition_1_minus_0", "ended_eos", "empty", "hit_cap")


def _finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite value in input")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    _finite(value)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        _finite(row)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSONL objects: {path}")
    return rows


def _index(rows: list[dict], source: str) -> dict[str, dict]:
    mapping = {}
    for row in rows:
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"missing row_id: {source}")
        if row_id in mapping:
            raise ValueError(f"duplicate row_id: {source}/{row_id}")
        mapping[row_id] = row
    return mapping


def load_panel(path: Path) -> dict[str, dict]:
    panel = _index(read_jsonl(path), "panel")
    groups = defaultdict(list)
    prompt_owners, context_owners = {}, {}
    for row in panel.values():
        task, cap = row.get("task"), row.get("length_cap")
        if task not in TASKS or type(cap) is not int or cap not in LENGTHS:
            raise ValueError("unknown panel task or length_cap")
        for key in ("group_id", "context_id", "prompt_sha256"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"missing panel {key}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["prompt_sha256"]):
            raise ValueError("invalid prompt_sha256")
        if row["prompt_sha256"] in prompt_owners:
            raise ValueError("duplicate prompt content in panel")
        prompt_owners[row["prompt_sha256"]] = row["row_id"]
        previous_group = context_owners.setdefault(row["context_id"], row["group_id"])
        if previous_group != row["group_id"]:
            raise ValueError("same context assigned to multiple world clusters")
        refs = row.get("references")
        if not isinstance(refs, list) or len(refs) != 1 or not re.fullmatch(r"[A-Za-z]{5}", str(refs[0])):
            raise ValueError("references must contain one five-letter ID")
        if type(row.get("input_tokens")) is not int or not 0 < row["input_tokens"] <= cap - 32:
            raise ValueError("invalid input_tokens or generation reserve")
        if row.get("max_new_tokens") != 32:
            raise ValueError("panel requires max_new_tokens=32")
        axis, first, second = CONDITIONS[task]
        intervention = row.get("intervention", {})
        if not isinstance(intervention, dict) or set(intervention) != {axis, "query"}:
            raise ValueError("invalid intervention fields")
        if intervention[axis] not in (first, second) or intervention["query"] not in ("a", "b"):
            raise ValueError("invalid intervention value")
        groups[row["group_id"]].append(row)
    strata = set()
    for group_id, rows in groups.items():
        identities = {(r["task"], r["length_cap"]) for r in rows}
        if len(identities) != 1 or len(rows) != 4:
            raise ValueError(f"group must have exactly four rows in one stratum: {group_id}")
        task, cap = next(iter(identities))
        strata.add((task, cap))
        axis, first, second = CONDITIONS[task]
        cells = {(r["intervention"][axis], r["intervention"]["query"]): r for r in rows}
        if set(cells) != {(c, q) for c in (first, second) for q in ("a", "b")}:
            raise ValueError(f"missing or duplicate intervention cell: {group_id}")
        for condition in (first, second):
            a, b = cells[condition, "a"], cells[condition, "b"]
            if a["context_id"] != b["context_id"]:
                raise ValueError(f"context drift within query pair: {group_id}")
            if a["references"] == b["references"]:
                raise ValueError(f"query pair needs distinct answers: {group_id}")
        if cells[first, "a"]["context_id"] == cells[second, "a"]["context_id"]:
            raise ValueError(f"intervention must change context identity: {group_id}")
        if task == "native_binding" and any(cells[first, q]["references"] != cells[second, q]["references"] for q in ("a", "b")):
            raise ValueError(f"binding layout changed the target answer: {group_id}")
        if task == "native_chain" and any(cells[first, q]["references"] == cells[second, q]["references"] for q in ("a", "b")):
            raise ValueError(f"chain rewiring must change each query answer: {group_id}")
    if strata != {(task, cap) for task in TASKS for cap in LENGTHS}:
        raise ValueError("panel must contain both tasks at every frozen length")
    return panel


def load_run(directory: Path, panel: dict[str, dict]) -> tuple[dict[str, dict], dict]:
    status = read_json(directory / "status.json")
    if status.get("status") != "COMPLETE" or status.get("rows") != len(panel) or status.get("lm_rows") != 0:
        raise ValueError(f"run is not COMPLETE for the full panel: {directory}")
    raw = read_jsonl(directory / "generations.jsonl")
    rows = _index(raw, str(directory))
    if set(rows) != set(panel):
        raise ValueError(f"missing or extra run rows: {directory}")
    contract = read_json(directory / "contract.json")
    for key in RUNTIME_FIELDS:
        if key not in contract:
            raise ValueError(f"missing runtime contract field: {key}")
    if contract["unadapted"] is not True or contract["checkpoint_arm"] is not None or contract["lm_enabled"] is not False:
        raise ValueError("report requires a frozen-model generation-only run")
    if contract["generation_length_caps"] != list(LENGTHS):
        raise ValueError("contract length coverage drift")
    if not isinstance(contract["runtime_versions"], dict) or not all(k in contract["runtime_versions"] for k in ("torch", "transformers", "model_dtype", "attention_backend")):
        raise ValueError("missing runtime version identity")
    eval_ids = [row.get("eval_id") for row in raw]
    if any(not isinstance(value, str) or not value for value in eval_ids) or len(set(eval_ids)) != len(raw):
        raise ValueError("missing or duplicate eval_id")
    if contract.get("row_ids") != eval_ids:
        raise ValueError("contract row_ids differ from generation order")
    for row_id, row in rows.items():
        expected = panel[row_id]
        for key in PAIR_FIELDS:
            if row.get(key) != expected[key]:
                raise ValueError(f"panel/run identity drift: {row_id}/{key}")
        for key in ("context_id", "intervention", "max_new_tokens"):
            if key in row and row[key] != expected[key]:
                raise ValueError(f"panel/run context drift: {row_id}/{key}")
        if row.get("arm") != contract.get("arm") or not isinstance(row.get("output_text"), str):
            raise ValueError(f"invalid arm or output_text: {row_id}")
        tokens = row.get("generated_ids")
        if not isinstance(tokens, list) or len(tokens) > 32 or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError(f"invalid generated_ids: {row_id}")
        for flag in ("ended_eos", "empty", "hit_cap"):
            if type(row.get(flag)) is not bool:
                raise ValueError(f"missing boolean generation diagnostic: {row_id}/{flag}")
        if row["ended_eos"] and not tokens:
            raise ValueError(f"EOS without generated token: {row_id}")
        if row["empty"] != (not row["output_text"].strip()):
            raise ValueError(f"empty diagnostic disagrees with output: {row_id}")
        if row["hit_cap"] != (len(tokens) == 32 and not row["ended_eos"]):
            raise ValueError(f"hit_cap diagnostic disagrees with tokens: {row_id}")
    return rows, contract


def _validate_preparation(directory: Path, contract: dict, panel_sha256: str) -> dict:
    receipt = read_json(directory / "preparation_run_identity.json")
    if not isinstance(receipt.get("model_path"), str) or not receipt["model_path"]:
        raise ValueError("missing prepared model_path")
    if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("model_config_sha256", ""))):
        raise ValueError("missing prepared model_config_sha256")
    if receipt.get("panel_sha256") != panel_sha256 or receipt.get("arm") != contract.get("arm"):
        raise ValueError("preparation receipt panel or arm identity drift")
    if "static_table" not in receipt or "static_table" not in contract:
        raise ValueError("missing prepared static_table identity")
    prepared, installed = receipt["static_table"], contract["static_table"]
    if (prepared is None) != (installed is None):
        raise ValueError("preparation receipt table identity drift")
    if prepared is not None:
        if not isinstance(prepared, dict) or not isinstance(installed, dict):
            raise ValueError("invalid static_table identity")
        for key in ("values_float32", "gain"):
            if key not in prepared or prepared[key] != installed.get(key):
                raise ValueError(f"preparation receipt table identity drift: {key}")
        values = prepared["values_float32"]
        if not isinstance(values, list) or len(values) < 2 or any(type(v) not in (int, float) or v <= 0 for v in values):
            raise ValueError("invalid static_table frequency values")
        if any(a <= b for a, b in zip(values, values[1:])) or prepared["gain"] != 1.0:
            raise ValueError("native frequency table must be ordered with gain=1")
    return receipt


def _world_metrics(panel: dict[str, dict], rows: dict[str, dict]) -> dict[tuple[str, int], np.ndarray]:
    groups = defaultdict(list)
    for row in panel.values():
        groups[row["group_id"]].append(row)
    strata = defaultdict(list)
    for group_id in sorted(groups):
        group = groups[group_id]
        task, cap = group[0]["task"], group[0]["length_cap"]
        axis, first, second = CONDITIONS[task]
        order = {(condition, query): index for index, (condition, query) in enumerate((c, q) for c in (first, second) for q in ("a", "b"))}
        group = sorted(group, key=lambda r: order[r["intervention"][axis], r["intervention"]["query"]])
        outputs = [rows[r["row_id"]] for r in group]
        correct = np.array([r["output_text"].strip() == p["references"][0] for p, r in zip(group, outputs)], dtype=float)
        condition0, condition1 = correct[:2].mean(), correct[2:].mean()
        values = [correct.mean(), correct.prod(), (correct[:2].prod() + correct[2:].prod()) / 2,
                  condition0, condition1, condition1 - condition0]
        values.extend(np.mean([r[flag] for r in outputs]) for flag in ("ended_eos", "empty", "hit_cap"))
        strata[task, cap].append(values)
    return {key: np.asarray(value, dtype=float) for key, value in strata.items()}


def _summary(left: np.ndarray, right: np.ndarray, samples: np.ndarray, names: tuple[str, str]) -> dict:
    return {metric: {names[0]: float(left[i]), names[1]: float(right[i]),
                     "effect": float(right[i] - left[i]),
                     "paired_world_bootstrap_ci95": [float(v) for v in np.quantile(samples[:, i], [0.025, 0.975])]}
            for i, metric in enumerate(METRICS)}


def build_report(panel_path: Path, run_paths: dict[str, Path], *, draws: int = 20_000, seed: int = 20260915) -> dict:
    """Validate real run artifacts, then return exact effects and clustered CIs."""
    if len(run_paths) != 2 or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) for name in run_paths):
        raise ValueError("provide exactly two distinct named arms")
    if type(draws) is not int or draws < 100:
        raise ValueError("bootstrap draws must be an integer >=100")
    panel = load_panel(panel_path)
    panel_sha256 = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    loaded = {name: load_run(Path(path), panel) for name, path in run_paths.items()}
    preparations = {name: _validate_preparation(Path(run_paths[name]), loaded[name][1], panel_sha256)
                    for name in run_paths}
    names = tuple(run_paths)
    baseline, candidate = (loaded[name][1] for name in names)
    for key in ("model_path", "model_config_sha256"):
        if preparations[names[0]][key] != preparations[names[1]][key]:
            raise ValueError(f"between-arm prepared model identity drift: {key}")
    for key in (*RUNTIME_FIELDS, "left_pad_batches", "model_id", "model_path", "checkpoint_identity"):
        if baseline.get(key, False if key == "left_pad_batches" else None) != candidate.get(key, False if key == "left_pad_batches" else None):
            raise ValueError(f"between-arm runtime identity drift: {key}")
    metric_arrays = {name: _world_metrics(panel, loaded[name][0]) for name in names}
    rng = np.random.default_rng(seed)
    strata_report, bootstrap, points = {}, {}, {}
    for cap in LENGTHS:
        for task in TASKS:
            key = task, cap
            left, right = (metric_arrays[name][key] for name in names)
            if len(left) < 2:
                raise ValueError(f"at least two independent worlds required per stratum: {key}")
            indices = rng.integers(len(left), size=(draws, len(left)))
            delta = right - left
            sampled = delta[indices].mean(axis=1)
            bootstrap[key] = sampled
            points[key] = left.mean(axis=0), right.mean(axis=0)
            strata_report[f"{task}:{cap}"] = {
                "worlds": len(left), "rows": 4 * len(left),
                "condition_0": CONDITIONS[task][1], "condition_1": CONDITIONS[task][2],
                "metrics": _summary(*points[key], sampled, names),
            }
    by_length = {}
    for cap in LENGTHS:
        left, right = (np.mean([points[task, cap][side] for task in TASKS], axis=0) for side in (0, 1))
        sampled = np.mean([bootstrap[task, cap] for task in TASKS], axis=0)
        by_length[str(cap)] = {"metrics": _summary(left, right, sampled, names)}
    overall = {}
    for i, metric in enumerate(METRICS):
        left, right = (float(np.mean([point[side][i] for point in points.values()])) for side in (0, 1))
        overall[metric] = {names[0]: left, names[1]: right, "effect": right - left}
    return {
        "status": "NATIVE_MECHANISM_PAIRED_REPORT_V1", "baseline": names[0], "candidate": names[1],
        "panel": {"path": str(panel_path), "sha256": panel_sha256, "rows": len(panel)},
        "runs": {name: str(path) for name, path in run_paths.items()},
        "runtime_identity": {key: baseline.get(key) for key in (*RUNTIME_FIELDS, "model_id", "model_path")},
        "prepared_model_identity": {key: preparations[names[0]][key] for key in ("model_path", "model_config_sha256")},
        "preparation_receipts": preparations,
        "arm_contracts": {name: loaded[name][1] for name in names},
        "strata": strata_report, "task_equal_by_length": by_length,
        "task_and_length_equal_descriptive": overall,
        "statistics": {
            "draws": draws, "seed": seed, "units": "score fractions; multiply effects and intervals by 100 for percentage points",
            "exact_match": "1[output_text.strip() == sole reference]; no lowercasing, punctuation removal, substring scoring or EOS requirement",
            "world_metrics": "exact=mean four correctness indicators; all_four=product four; query_pair=mean of the two within-condition products",
            "contrast": "condition_1_minus_0=(second-condition query mean)-(first-condition query mean); its candidate-minus-baseline effect is the difference-in-differences",
            "bootstrap": "resample group_id worlds with replacement within each task x length; keep all four rows and both arms together; reuse draws across metrics; 2.5/97.5 percentile interval",
            "point_estimate": "exact observed candidate mean minus baseline mean; never a bootstrap mean",
            "length_macro": "equal mean of the two fixed task families, regardless of world counts; independent resampling within each family",
            "cross_length": "equal mean of fixed lengths and tasks is descriptive only; no joint CI because independence across lengths is not assumed",
        },
        "claim_boundary": "Synthetic native-window mechanism panel. This report does not establish natural-task utility, LM health or cross-model generalization. Model identity comes from the launch preparation receipt (model path and config identity); checkpoint weights are not rehashed. Runtime equality is checked for recorded contract fields.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="NAME=RUN_DIRECTORY; baseline first, candidate second")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    paths = {}
    for spec in args.run:
        name, separator, path = spec.partition("=")
        if not separator or not path or name in paths:
            raise ValueError("each --run must be a unique NAME=RUN_DIRECTORY")
        paths[name] = Path(path)
    report = build_report(args.panel, paths, draws=args.draws, seed=args.seed)
    if args.out.exists():
        raise ValueError("use a new report path; completed reports are not overwritten")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": report["status"], "rows": report["panel"]["rows"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
