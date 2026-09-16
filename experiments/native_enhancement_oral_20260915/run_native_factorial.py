"""Run the two prepared Qwen1.5B gain cells; default invocation is plan-only.

This entry never schedules itself or alters another queue. Existing generations
are paired by prompt identity, not historical row_id (which repeats by block).
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .evidence_review import raw


ROOT = Path("/root/autodl-tmp/today_rope_plan_20260914/native_enhancement_oral_cpu/qwen15_native_gain_factorial")
NEW_ARMS = ("mix075_gain1", "native_gain_mid")
CELLS = ("native", "mix075_gain1", "native_gain_mid", "candidate")
EFFECTS = {
    "frequency_at_gain1": (-1, 1, 0, 0),
    "gain_at_native": (-1, 0, 1, 0),
    "interaction": (1, -1, -1, 1),
    "combined": (-1, 0, 0, 1),
}


def commands(root: Path, model: Path, python: str) -> dict[str, list[str]]:
    return {arm: [
        python, "-m", "experiments.olmo_recovery_20260912.recovery_v2_eval",
        "--data", str(root / "data.json"), "--model", str(model), "--arm", "Native",
        "--extra-panel", str(root / "inputs.jsonl"), "--only-extra-panels", "--skip-lm",
        "--length-cap", "32768", "--batch-size", "1", "--prefill-chunk-size", "8192",
        "--static-table-json", str(root / "tables" / f"{arm}.json"),
        "--table-label", arm, "--out", str(root / "runs" / arm),
    ] for arm in NEW_ARMS}


def panel_key(row: dict) -> tuple:
    return row["task"], int(row["length_cap"]), row["prompt_sha256"]


def load_panel(root: Path) -> list[dict]:
    rows = [json.loads(line) for line in (root / "inputs.jsonl").read_text().splitlines() if line.strip()]
    counts = Counter(row["task"] for row in rows)
    if (len(rows) != 108 or len({panel_key(row) for row in rows}) != 108
            or len(counts) != 6 or set(counts.values()) != {18}
            or Counter(row["historical_block"] for row in rows) != {"development6": 36, "additional12": 72}
            or any(row["length_cap"] != 32768 for row in rows)):
        raise ValueError("prepared historical Core6 x 18 identity differs")
    return rows


def paired_cell_scores(panel: list[dict], cells: dict[str, dict]) -> np.ndarray:
    expected = {panel_key(row) for row in panel}
    if set(cells) != set(CELLS) or any(set(mapping) != expected for mapping in cells.values()):
        raise ValueError("four cells must cover every prepared prompt; no intersection filtering")
    result = np.empty((len(panel), len(CELLS)), dtype=np.float64)
    for index, source in enumerate(panel):
        for column, arm in enumerate(CELLS):
            row = cells[arm][panel_key(source)]
            if any(row.get(field) != source.get(field) for field in (
                "task", "length_cap", "input_tokens", "prompt_sha256", "references",
            )):
                raise ValueError(f"factorial prompt identity differs: {arm}/{source['row_id']}")
            score = float(row["ruler_official_score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid recorded RULER score")
            result[index, column] = score
    return result


def summarize(panel: list[dict], scores: np.ndarray, *, draws: int = 10000, seed: int = 20260916) -> dict:
    if scores.shape != (len(panel), 4) or not len(panel) or draws <= 0:
        raise ValueError("four paired cells and positive bootstrap draws are required")
    tasks = sorted({row["task"] for row in panel})
    rng = np.random.default_rng(seed)
    point = np.zeros(4)
    sampled = np.zeros((draws, 4))
    for task in tasks:
        indices = [i for i, row in enumerate(panel) if row["task"] == task]
        values = scores[indices]
        point += values.mean(axis=0) / len(tasks)
        # One draw index is shared by all four cells, so interactions retain
        # within-prompt covariance. Tasks keep their predeclared equal weight.
        sample_indices = rng.integers(len(values), size=(draws, len(values)))
        sampled += values[sample_indices].mean(axis=1) / len(tasks)
    effects = {}
    for name, weights in EFFECTS.items():
        coef = np.asarray(weights)
        effects[name] = {
            "delta": float(point @ coef),
            "ci95": [float(x) for x in np.quantile(sampled @ coef, [0.025, 0.975])],
        }
    return {"rows": len(panel), "tasks": tasks,
            "cell_task_macro": dict(zip(CELLS, map(float, point))), "effects": effects,
            "bootstrap": {"draws": draws, "seed": seed, "unit": "paired rows within task; all four cells jointly sampled"}}


def build_report(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    panel = load_panel(root)
    paths = {arm: [Path(value) for value in manifest["reusable_arms"][arm]]
             for arm in ("native", "candidate")}
    for arm in NEW_ARMS:
        run = root / "runs" / arm
        if json.loads((run / "status.json").read_text()) != {"status": "COMPLETE", "rows": 108, "lm_rows": 0}:
            raise ValueError(f"incomplete new factorial cell: {arm}")
        paths[arm] = [run / "generations.jsonl"]
    cells = {arm: raw(files, 32768) for arm, files in paths.items()}
    scores = paired_cell_scores(panel, cells)
    contracts = {arm: [json.loads((path.parent / "contract.json").read_text()) for path in files]
                 for arm, files in paths.items()}
    for arm, records in contracts.items():
        if any(record.get("batch_size") != 1 or record.get("prefill_chunk_size") != 8192
               or record.get("base_arm") != "Native" or record.get("unadapted") is not True
               for record in records):
            raise ValueError(f"known historical execution fields differ: {arm}")
    source_table = contracts["candidate"][0]["static_table"]
    if any(record.get("static_table") != source_table for record in contracts["candidate"]):
        raise ValueError("historical mix table differs between blocks")
    for arm in NEW_ARMS:
        frozen = json.loads((root / "tables" / f"{arm}.json").read_text())
        active = contracts[arm][0].get("static_table") or {}
        if any(active.get(key) != frozen.get(key) for key in ("values_float32", "gain")):
            raise ValueError(f"new cell installed a different table: {arm}")
        if arm == "mix075_gain1" and (
            active.get("values_float32") != source_table.get("values_float32")
            or active.get("gain") != 1.0
        ):
            raise ValueError("mix gain-one cell changed the historical frequencies")
        if arm == "native_gain_mid" and active.get("gain") != source_table.get("gain"):
            raise ValueError("Native gain cell does not use the historical midpoint gain")
    missing = {arm: [str(path.parent / "contract.json") for path, record in zip(paths[arm], contracts[arm])
                     if not record.get("runtime_versions")] for arm in CELLS}
    blocks = {}
    for block in ("development6", "additional12"):
        selected = [i for i, row in enumerate(panel) if row["historical_block"] == block]
        blocks[block] = summarize([panel[i] for i in selected], scores[selected])
    return {
        "status": "QWEN15_NATIVE_GAIN_FACTORIAL_RECORDED_SCORES_COMPLETE_V1",
        "model": manifest["model"], "native_length": 32768,
        "source_gain": manifest["source_gain"], "cumulative": summarize(panel, scores),
        "historical_blocks": blocks,
        "source_files": {arm: [str(path) for path in files] for arm, files in paths.items()},
        "missing_runtime_versions": {arm: values for arm, values in missing.items() if values},
        "runtime_qualification": "Known fields checked; historical unrecorded runtime versions are not reconstructed or certified here.",
        "scope": "Retrospective four-cell attribution on historical inputs; recorded scores, not independent confirmation or a claim of fixed-support pure-z.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/qwen25_1p5b_32k"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.execute and args.report_only:
        raise ValueError("choose execute or report-only")
    planned = commands(args.root, args.model, args.python)
    if not args.execute and not args.report_only:
        print(json.dumps({"status": "PLAN_ONLY", "new_generations": 216,
                          "queue_position": "after the already-authorized InfiniteBench and Qwen 256K work; never auto-scheduled",
                          "commands_without_execute": planned}, indent=2))
        return
    if args.execute:
        import fcntl
        panel = load_panel(args.root)
        manifest = json.loads((args.root / "manifest.json").read_text())
        if args.model.resolve() != Path(manifest["model"]).resolve():
            raise ValueError("use the checkpoint recorded by the existing factorial preparation")
        repo = Path(__file__).resolve().parents[2]
        env = dict(os.environ, PYTHONPATH=str(repo))
        with open("/tmp/hybrid-rope-gpu0.lock", "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            for arm, command in planned.items():
                state = args.root / "runs" / arm / "status.json"
                if state.exists():
                    if json.loads(state.read_text()) != {"status": "COMPLETE", "rows": len(panel), "lm_rows": 0}:
                        raise ValueError(f"invalid completed status: {arm}")
                    continue
                logs = args.root / "logs"
                logs.mkdir(exist_ok=True)
                with (logs / f"{arm}.log").open("a") as output:
                    subprocess.run(command + ["--execute"], cwd=repo, env=env,
                                   stdout=output, stderr=subprocess.STDOUT, check=True)
    result = build_report(args.root)
    target = args.root / "reports" / "frequency_gain_factorial.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.incomplete")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    print(json.dumps({"status": result["status"], "report": str(target)}))


if __name__ == "__main__":
    main()
