#!/usr/bin/env python3
"""Freeze the 96-prompt stratified Q/K/V mechanism capture panel.

The selection is output-blind: four complete counterfactual worlds are chosen
per task-family and length cell by a stable group hash.  Every selected world
contributes all four counterfactual rows, giving 2 x 3 x 4 x 4 = 96 prompts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .prepare import CONTRACT, LENGTHS, audit_group


CAPTURE_CONTRACT = "native-counterfactual-qkv-capture96-v1"
FAMILIES = ("native_binding", "native_chain")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _stable_groups(rows: list[dict], worlds_per_cell: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    selected = []
    cells: dict[tuple[str, int], list[tuple[str, list[dict]]]] = defaultdict(list)
    for group_id, group in grouped.items():
        audit_group(group)
        cell = (str(group[0]["task"]), int(group[0]["length_cap"]))
        cells[cell].append((group_id, group))
    expected_cells = {(task, length) for task in FAMILIES for length in LENGTHS}
    if set(cells) != expected_cells:
        raise ValueError("mechanism panel task/length cells drifted")
    for cell in sorted(cells):
        ordered = sorted(
            cells[cell],
            key=lambda item: (hashlib.sha256(item[0].encode()).hexdigest(), item[0]),
        )
        if len(ordered) < worlds_per_cell:
            raise ValueError(f"not enough worlds in capture cell {cell}")
        for _, group in ordered[:worlds_per_cell]:
            selected.extend(sorted(group, key=lambda row: row["row_id"]))
    return selected


def _correct_evidence(row: dict) -> list[dict]:
    positions = list(row.get("evidence_positions") or [])
    query = str(row["query_node"])
    if row["task"] == "native_binding":
        result = [item for item in positions if item["source"] == query]
        if len(result) != 1:
            raise ValueError("binding row must have one query-matched evidence record")
        return result
    mapping = {str(left): str(right) for left, right in row["source_records"]}
    needed = []
    cursor = query
    for _ in range(3):
        destination = mapping[cursor]
        match = [item for item in positions
                 if item["source"] == cursor and item["destination"] == destination]
        if len(match) != 1:
            raise ValueError("chain row lacks a uniquely annotated path edge")
        needed.extend(match)
        cursor = destination
    if cursor != row["references"][0]:
        raise ValueError("annotated chain path does not reach the frozen answer")
    return needed


def annotate_capture_queries(row: dict) -> dict:
    evidence = _correct_evidence(row)
    final_evidence = sorted({
        token
        for item in evidence
        for token in range(int(item["token_start"]), int(item["token_end"]))
    })
    final_position = int(row["input_tokens"]) - 1
    if not final_evidence or max(final_evidence) >= final_position:
        raise ValueError("final-readout evidence must be causally visible")
    queries = [{
        "role": "final_readout",
        "position": final_position,
        "evidence_token_indices": final_evidence,
    }]
    for index, item in enumerate(evidence, start=1):
        position = int(item["destination_token_end"]) - 1
        visible = list(range(int(item["token_start"]), min(int(item["token_end"]), position + 1)))
        if not visible or max(visible) > position:
            raise ValueError("relation-write evidence is not causally visible")
        queries.append({
            "role": f"relation_write_{index}",
            "position": position,
            "evidence_token_indices": visible,
            "source": item["source"],
            "destination": item["destination"],
        })
    positions = [item["position"] for item in queries]
    if len(positions) != len(set(positions)):
        raise ValueError("capture query positions must be unique within a prompt")
    return {**row, "capture_queries": queries}


def layer_quartiles(num_hidden_layers: int) -> list[int]:
    if num_hidden_layers < 4:
        raise ValueError("four-layer mechanism capture needs at least four layers")
    return sorted({
        (num_hidden_layers * numerator + 3) // 4 - 1
        for numerator in (1, 2, 3, 4)
    })


def intervention_subset(rows: list[dict], count: int = 64) -> list[dict]:
    """Output-blind round-robin sample with near-equal task/length counts."""
    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        cells[(str(row["task"]), int(row["length_cap"]))].append(row)
    ordered = {
        cell: sorted(values, key=lambda row: (
            hashlib.sha256(str(row["row_id"]).encode()).hexdigest(), row["row_id"],
        ))
        for cell, values in cells.items()
    }
    selected = []
    offset = 0
    while len(selected) < count:
        added = False
        for cell in sorted(ordered):
            if offset < len(ordered[cell]):
                selected.append(ordered[cell][offset])
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise ValueError("not enough capture rows for the intervention subset")
        offset += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds-per-cell", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("capture assets are immutable; use a new --out directory")
    manifest = json.loads(args.panel_manifest.read_text())
    if manifest.get("contract") != CONTRACT or manifest.get("rows") != 288:
        raise ValueError("input is not the frozen 288-row mechanism panel")
    if sha256(args.panel) != manifest.get("inputs_sha256", sha256(args.panel)):
        raise ValueError("mechanism panel hash differs from its manifest")
    config = json.loads((args.model / "config.json").read_text())
    rows = read_jsonl(args.panel)
    selected = [annotate_capture_queries(row)
                for row in _stable_groups(rows, args.worlds_per_cell)]
    layers = layer_quartiles(int(config["num_hidden_layers"]))
    expected = len(FAMILIES) * len(LENGTHS) * args.worlds_per_cell * 4
    if len(selected) != expected or len({row["row_id"] for row in selected}) != expected:
        raise ValueError("capture panel cardinality drift")
    args.out.mkdir(parents=True)
    inputs = args.out / "inputs.jsonl"
    with inputs.open("w") as stream:
        for row in selected:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    intervention = intervention_subset(selected)
    intervention_inputs = args.out / "intervention_inputs.jsonl"
    with intervention_inputs.open("w") as stream:
        for row in intervention:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    canary_inputs = args.out / "override_canary_inputs.jsonl"
    with canary_inputs.open("w") as stream:
        for row in intervention[:2]:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    counts = Counter((row["task"], row["length_cap"]) for row in selected)
    num_hidden_layers = int(config["num_hidden_layers"])
    final_quarter = list(range((3 * num_hidden_layers) // 4, num_hidden_layers))
    receipt = {
        "status": "CPU_PREPARED",
        "contract": CAPTURE_CONTRACT,
        "rows": len(selected),
        "worlds_per_task_length": args.worlds_per_cell,
        "rows_by_task_length": {f"{task}:{length}": count
                                for (task, length), count in sorted(counts.items())},
        "layers_zero_based": layers,
        "layer_rule": "ceil(num_hidden_layers * fraction) - 1 for 1/4,1/2,3/4,1",
        "model_config_sha256": sha256(args.model / "config.json"),
        "source_panel_sha256": sha256(args.panel),
        "inputs_sha256": sha256(inputs),
        "intervention_rows": len(intervention),
        "intervention_inputs_sha256": sha256(intervention_inputs),
        "override_canary_rows": 2,
        "override_canary_inputs_sha256": sha256(canary_inputs),
        "intervention_rows_by_task_length": {
            f"{task}:{length}": count
            for (task, length), count in sorted(Counter(
                (row["task"], row["length_cap"]) for row in intervention
            ).items())
        },
        "intervention_layer_block_zero_based": final_quarter,
        "intervention_layer_rule": "final quarter of the checkpoint, frozen before outputs",
        "capture_storage": (
            "Q/K/V arrays preserve exact BF16 payload bits when the runtime tensors are BF16; "
            "analysis decodes them to FP32. Other arrays retain their native NumPy dtype."
        ),
        "selection_uses_model_outputs": False,
        "query_contract": (
            "Every row captures the final prompt token plus each correct assignment/connection "
            "destination token; evidence indices are fixed before model execution."
        ),
        "gpu_execution": False,
    }
    (args.out / "manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
