"""Run a frozen broader panel through the existing PM or NOSA reader.

Defaults to a concrete dry run. Execution uses the existing server queue lock;
this entry point never interrupts a current experiment or changes its data.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys

from .prepare import read_rows, write_json, sha
from .scoring import score, score_nosa, VERSION


def build_command(args):
    engine = args.engine
    data = args.panel / (engine + ".jsonl")
    frozen = json.loads((args.panel / "manifest.json").read_text())
    if sha(data.read_bytes()) != frozen["models"][engine]["sha256"]:
        raise ValueError("frozen broader data hash changed")
    rows = [r for r in read_rows(data) if r["split"] == args.split
            and (not args.tasks or r["task"] in args.tasks)]
    if not rows:
        raise ValueError("empty requested broader panel")
    if args.tasks and set(args.tasks) - {r["task"] for r in rows}:
        raise ValueError("one or more requested tasks are unavailable for this model/length")
    model = args.model or (args.root / "runs/pm_gpu_ready_20260909_v3/model_view" if engine == "pm"
                           else Path("/root/autodl-tmp/NOSA-1B"))
    cache = args.root / "baselines" / ("broad_" + engine + "_" + VERSION)
    command = ["--model", str(model), "--data", str(data), "--output", str(args.output),
               "--baseline-cache", str(cache), "--split", args.split,
               "--device", args.device, "--dtype", args.dtype]
    if args.tasks:
        command += ["--tasks", *args.tasks]
    if engine == "pm":
        command += ["--arms", *args.arms, "--horizon", "128", "--samples", "256",
                    "--query-policy", "uniform_prefix", "--seed", "20260909",
                    "--keep-fraction", str(args.keep_fraction)]
    else:
        command += ["--selectors", *args.selectors, "--topk", str(args.topk),
                    "--chunk-size", "128", "--attention-query-chunk-size", "64"]
    return command, rows, frozen, model


def install(engine, sampling, topk=64):
    if engine == "pm":
        from experiments.pm_keep import run as base
        from experiments.pm_keep.balanced_queries import BalancedConfig, BalancedValueSession
        from experiments.pm_keep.adapter import AdapterConfig
        from experiments.pm_keep.run_followup import ValueAwareSession

        @dataclass(frozen=True)
        class UniformConfig(AdapterConfig):
            value_objective: str = "raw_norm"
            broad_scoring_version: str = VERSION

        @dataclass(frozen=True)
        class PanelBalancedConfig(BalancedConfig):
            broad_scoring_version: str = VERSION

        base.score = score
        base.BASELINE_VERSION += "_" + VERSION
        base.AdapterConfig = PanelBalancedConfig if sampling == "balanced" else UniformConfig
        base.PrefillSession = BalancedValueSession if sampling == "balanced" else ValueAwareSession
    else:
        from experiments.nosa_position import run as base
        from experiments.nosa_position.exact_probe import ExactBlockSelector
        from experiments.nosa_position import exact_probe
        previous = base.source_hashes
        base.source_hashes = lambda: {**previous(), "exact_probe.py": sha(Path(exact_probe.__file__).read_bytes())}
        base.BlockSummarySelector = ExactBlockSelector
        base.score_output = partial(score_nosa, topk=topk)
    return base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=("pm", "pc2"), required=True)
    p.add_argument("--sampling", choices=("balanced", "uniform"), default="balanced")
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/position_overnight_20260909"))
    p.add_argument("--model", type=Path)
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--arms", nargs="+", choices=("F", "E", "E_author_policy", "K", "P", "C", "U"),
                   default=["F", "E", "E_author_policy", "K", "P", "C", "U"])
    p.add_argument("--selectors", nargs="+", default=["native", "cobs_rank2", "pc2_rank1", "exact_mass", "dense"])
    p.add_argument("--keep-fraction", type=float, default=.25)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--wait-for-lock", action="store_true")
    args = p.parse_args()
    command, rows, frozen, model = build_command(args)
    print(json.dumps({"engine": args.engine, "sampling": args.sampling if args.engine == "pm" else None,
                      "rows": len(rows), "independent_units": len({r['material_cluster_id'] for r in rows}),
                      "command_arguments": command, "execute": args.execute}, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    if sha((model / "tokenizer.json").read_bytes()) != frozen["models"][args.engine]["tokenizer_sha256"]:
        raise ValueError("model tokenizer differs from the frozen panel")
    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__).parent / name for name in ("prepare.py", "scoring.py", "run.py")]
    receipt = {"engine": args.engine, "sampling": args.sampling if args.engine == "pm" else None,
               "data_sha256": frozen["models"][args.engine]["sha256"], "argv": command,
               "scoring_version": VERSION, "sources": {p.name: sha(p.read_bytes()) for p in source_paths},
               "scope": "Complete generation on new source families; metric types and exact reference stay distinct"}
    receipt_path = args.output / "broad_execution.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("broader execution contract changed; use a new output directory")
    write_json(receipt_path, receipt)
    snapshot = args.output / "broad_code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in source_paths:
        (snapshot / path.name).write_bytes(path.read_bytes())
    with (args.root / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_for_lock else fcntl.LOCK_NB))
        if (args.root / "STOP").exists():
            raise RuntimeError("user STOP exists; no experiment launched")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ["PM_KEEP_KVPRESS_ROOT"] = str(args.root / "vendor/kvpress")
        base = install(args.engine, args.sampling, args.topk)
        sys.argv = [sys.argv[0], *command]
        base.main()


if __name__ == "__main__":
    main()
