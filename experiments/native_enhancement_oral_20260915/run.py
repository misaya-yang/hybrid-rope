"""Print an executable mechanism comparison plan; GPU execution is opt-in.

This entry point never changes another queue. The present preparation task does
not authorize --execute. A future authorized operator can use it after existing
GPU jobs have finished.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

PLAN = Path("/root/autodl-tmp/today_rope_plan_20260914")
ROOT = PLAN / "native_enhancement_oral_cpu"
MODEL = Path("/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct")
TABLES = {
    "native": None,
    "halfturn": PLAN / "olmo_native_halfturn_phase/tables/contract.json",
    "ncp": PLAN / "olmo_native_contrastive_proximal/tables/ncp.json",
    "v1": PLAN / "olmo_native_z5_enhancement/optimization/table.json",
    "ncp_dose_control": ROOT / "tables/ncp_dose_control.json",
    "ncp_phase_reflection": ROOT / "tables/ncp_phase_reflection.json",
}


def normalize_static_table(payload: dict | None) -> dict | None:
    """Record the same FP32 table values that recovery_v2_eval will install."""
    if payload is None:
        return None
    table = payload.get("table", payload)
    values = [struct.unpack("f", struct.pack("f", value))[0] for value in table["values_float32"]]
    gain = float(table["gain"])
    if len(values) != 64 or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("invalid native static table frequencies")
    if any(a <= b for a, b in zip(values, values[1:])) or gain != 1.0:
        raise ValueError("native static table must be ordered with gain=1")
    return {"values_float32": values, "gain": gain}


def build_command(*, python: str, model: Path, panel: Path, data: Path,
                  out: Path, arm: str, table: Path | None) -> list[str]:
    command = [python, "-m", "experiments.olmo_recovery_20260912.recovery_v2_eval",
               "--data", str(data), "--model", str(model), "--arm", "Native",
               "--extra-panel", str(panel), "--only-extra-panels", "--skip-lm",
               "--prefill-chunk-size", "4096", "--batch-size", "1", "--out", str(out)]
    if table is not None:
        command += ["--static-table-json", str(table), "--table-label", f"olmo_native_mechanism_{arm}"]
    return command


def frozen_json(path: Path, payload: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"existing preparation identity differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/root/autodl-tmp/hybrid-rope"))
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--panel", type=Path, default=ROOT / "assets/mechanism/inputs.jsonl")
    parser.add_argument("--data", type=Path, default=PLAN / "tailspline_olmo_s4_classic/assets/ppl46/manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/mechanism")
    parser.add_argument("--arm", action="append", choices=tuple(TABLES))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    arms = args.arm or ["native", "halfturn", "ncp", "v1"]
    if len(set(arms)) != len(arms):
        raise ValueError("duplicate arm")
    commands = {arm: build_command(python=args.python, model=args.model, panel=args.panel,
                                   data=args.data, out=args.out / arm, arm=arm, table=TABLES[arm])
                for arm in arms}
    if not args.execute:
        print(json.dumps({"status": "PLAN_ONLY", "model_loaded": False, "gpu_execution": False,
                          "commands_without_execute": commands,
                          "expected_rows_per_arm": 288,
                          "execution_instruction": "Only a separately authorized --execute invocation runs these commands under the global GPU lock."}, indent=2))
        return

    # Importing this module or asking for a plan cannot load a model.
    import fcntl
    from .prepare import CONTRACT, audit_group
    with open("/tmp/hybrid-rope-gpu0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"
        ], text=True).strip()
        if active:
            raise RuntimeError("a GPU process is active; this entry point does not co-run or stop it")
        manifest = json.loads((args.panel.parent / "manifest.json").read_text())
        if manifest.get("contract") != CONTRACT or manifest.get("rows") != 288:
            raise ValueError("mechanism panel contract differs")
        if Path(manifest.get("model", "")).resolve() != args.model.resolve():
            raise ValueError("panel tokenizer checkpoint differs from the requested model")
        rows = [json.loads(line) for line in args.panel.read_text().splitlines() if line.strip()]
        if len(rows) != 288 or len({r['row_id'] for r in rows}) != 288:
            raise ValueError("mechanism panel row count differs")
        groups = {}
        for row in rows:
            groups.setdefault(row["group_id"], []).append(row)
        for group in groups.values():
            audit_group(group)
        config = json.loads((args.model / "config.json").read_text())
        if config.get("model_type") != "olmo2" or config.get("max_position_embeddings") != 4096:
            raise ValueError("this frozen mechanism launcher is for OLMo native 4K; other models require their own tokenized panel and table mapping")
        config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        panel_sha256 = hashlib.sha256(args.panel.read_bytes()).hexdigest()
        for arm, command in commands.items():
            run = args.out / arm
            table_path = TABLES[arm]
            table = json.loads(table_path.read_text()) if table_path else None
            identity = {"contract": CONTRACT, "model_id": "olmo2_1b_instruct_native4096",
                        "model_path": str(args.model.resolve()), "model_config_sha256": config_id,
                        "checkpoint_identity_basis": "same trusted local checkpoint path and config; weights are not rehashed",
                        "panel_path": str(args.panel.resolve()), "panel_sha256": panel_sha256,
                        "panel_row_ids": [r["row_id"] for r in rows],
                        "arm": f"olmo_native_mechanism_{arm}" if table_path else "Native",
                        "logical_arm": arm, "static_table": normalize_static_table(table), "frozen_weights": True,
                        "selection_uses_model_outputs": False}
            frozen_json(run / "preparation_run_identity.json", identity)
            state = run / "status.json"
            if state.exists() and json.loads(state.read_text()) == {"status": "COMPLETE", "rows": 288, "lm_rows": 0}:
                generated = [json.loads(line) for line in (run / "generations.jsonl").read_text().splitlines() if line.strip()]
                if [r["row_id"] for r in generated] != [r["row_id"] for r in rows]:
                    raise ValueError("completed arm generation identity differs")
                continue
            subprocess.run(command + ["--execute"], cwd=args.repo, check=True)


if __name__ == "__main__":
    main()
