#!/usr/bin/env python3
"""Score the frozen Native/NCP full-versus-recent same-target LM panel."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .lm_context import CONDITIONS, analyze_four_conditions, build_context_pair, target_digest


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score(model, context: dict) -> dict:
    import torch
    import torch.nn.functional as F

    ids = torch.tensor(context["input_ids"], device="cuda", dtype=torch.long).unsqueeze(0)
    positions = torch.tensor(context["position_ids"], device="cuda", dtype=torch.long).unsqueeze(0)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(input_ids=ids, position_ids=positions, use_cache=False).last_hidden_state[0]
    loss_positions = torch.tensor(context["loss_positions"], device="cuda", dtype=torch.long)
    targets = torch.tensor(context["target_ids"], device="cuda", dtype=torch.long)
    selected = hidden.index_select(0, loss_positions)
    total = 0.0
    for start in range(0, len(targets), 64):
        stop = min(start + 64, len(targets))
        logits = F.linear(selected[start:stop], model.lm_head.weight).float()
        total += float(F.cross_entropy(logits, targets[start:stop], reduction="sum"))
    return {"nll_sum": total, "target_count": int(len(targets))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--ncp-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=("native", "ncp", "both"), default="both")
    parser.add_argument("--allow-paired-parallel", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    matrix = np.load(args.tokens, mmap_mode="r", allow_pickle=False)
    if list(matrix.shape) != manifest["array_shape"] or len(matrix) != len(manifest["samples"]):
        raise ValueError("LM token matrix differs from the frozen manifest")
    selected_conditions = CONDITIONS if args.arm == "both" else tuple(
        condition for condition in CONDITIONS if condition[0] == args.arm
    )
    if args.allow_paired_parallel and args.arm == "both":
        raise ValueError("paired-parallel mode requires one explicit arm")
    plan = {
        "status": "PLAN_ONLY", "rows": len(matrix),
        "conditions": [list(value) for value in selected_conditions],
        "forwards": len(matrix) * len(selected_conditions), "model_loaded": False,
        "arm": args.arm, "allow_paired_parallel": bool(args.allow_paired_parallel),
    }
    if not args.execute:
        print(json.dumps(plan, sort_keys=True))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    contract = {**plan, "status": "NATIVE_SAME_TARGET_CONTEXT_NLL_RUN_V1",
                "manifest": str(args.manifest.resolve()), "tokens": str(args.tokens.resolve()),
                "model": str(args.model.resolve()), "ncp_table": str(args.ncp_table.resolve())}
    contract_path = args.out / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("LM output root belongs to another contract")
    atomic_json(contract_path, contract)
    lock_path = (
        f"/tmp/hybrid-rope-native-lm-{args.arm}.lock"
        if args.allow_paired_parallel else "/tmp/hybrid-rope-gpu0.lock"
    )
    with open(lock_path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.allow_paired_parallel:
            active = subprocess.check_output([
                "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
            ], text=True).strip()
            if active:
                raise RuntimeError("another GPU process is active")
        import torch
        from experiments.olmo_recovery_20260912.recovery_v2_runtime import load_model
        from experiments.olmo_recovery_20260912.runtime import validate_cuda
        from scripts.experiments.cross_audit.tables import install_static, native_table

        validate_cuda()
        model, wrapper, _ = load_model(args.model, "Native", checkpoint=None, training=False)
        if wrapper is not None:
            raise RuntimeError("native LM study requires the unadapted checkpoint")
        dim = int(model.config.hidden_size // model.config.num_attention_heads)
        base = float(model.config.rope_theta)
        native = native_table(dim, base).astype(np.float32)
        ncp_payload = json.loads(args.ncp_table.read_text())
        ncp = ncp_payload.get("table", ncp_payload)
        records_path = args.out / ("scores.jsonl" if args.arm == "both" else f"scores_{args.arm}.jsonl")
        saved = read_jsonl(records_path)
        expected = [(sample, arm, context) for sample in manifest["samples"]
                    for arm, context in selected_conditions]
        if len(saved) > len(expected):
            raise ValueError("LM output contains too many records")
        for index, row in enumerate(saved):
            sample, arm, context = expected[index]
            if (row["pair_id"], row["arm"], row["context"]) != (sample["pair_id"], arm, context):
                raise ValueError("LM output is not the expected resumable prefix")
        with records_path.open("a") as stream:
            for sample, arm, context_name in expected[len(saved):]:
                if arm == "native":
                    install_static(model, native, 1.0)
                else:
                    install_static(model, np.asarray(ncp["values_float32"], dtype=np.float32), float(ncp["gain"]))
                pair = build_context_pair(
                    matrix[int(sample["source_row"])], native_length=manifest["native_length"],
                    recent_history=manifest["recent_history"], target_tokens=manifest["target_tokens"],
                    window_start=manifest["window_start"],
                )
                context = pair[context_name]
                row = {
                    "pair_id": sample["pair_id"], "arm": arm, "context": context_name,
                    "target_sha256": target_digest(context["target_ids"]), **score(model, context),
                }
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                saved.append(row)
                atomic_json(args.out / "live.json", {"completed": len(saved), "total": len(expected)})
                torch.cuda.empty_cache()
    if args.arm == "both":
        analysis = analyze_four_conditions(manifest, saved)
        atomic_json(args.out / "report.json", analysis)
    atomic_json(args.out / "status.json", {
        "status": "COMPLETE", "arm": args.arm, "rows": len(saved),
        "report_complete": args.arm == "both",
    })
    print(json.dumps({"status": "COMPLETE", "arm": args.arm, "records": len(saved)}))


if __name__ == "__main__":
    main()
