#!/usr/bin/env python3
"""E1 summary + Pro §4.3 branch table. CPU-only; run after driver_e1.sh.

Reads:
  e1/cross/{w_on_t_z,w_zf_t_0}/{evaluation.json,examples.jsonl}
  e1/fit/{ZF,ON}/manifest_*.json + per-stage summaries
Diagonals (round-11, reference):
  T0xW0  = out/task128            (native table, native-trained adapter? no:
           T0 diagnostic = W_T0 at step128 under native runtime)
  ZF diag= out_zf/task128         (W_ZF under T_Z)
  ON diag= out_on/task128         (W_ON under T_N)
Outputs e1/summary.json and prints the §4.3 decision rows:
  train-far fits (positive worst margin, strict>0 at some step) but
    validation-far fails      -> did-not-transfer
  train-far never fits        -> did-not-learn (learning-signal problem)
  cross cells localize whether the weight change or the table change
    carries the far-side behavior.
"""
from __future__ import annotations

import json
from pathlib import Path

B12 = Path("/root/autodl-tmp/claude_round12_20260906")
R11 = Path("/root/autodl-tmp/claude_round11_olmo_20260905")
E1 = B12 / "e1"


def load_diag(path: Path):
    ev = path / "evaluation.json"
    if not ev.exists():
        return None
    data = json.loads(ev.read_text())
    out = {"path": str(path)}
    cells = data.get("cells") or data
    if isinstance(cells, dict):
        for key, cell in cells.items():
            if isinstance(cell, dict) and "both_worlds_exact_eos" in cell:
                out[key] = cell["both_worlds_exact_eos"]
    if "both_worlds_exact_eos" in data:
        out["overall"] = data["both_worlds_exact_eos"]
    return out


def fit_stages(arm: str):
    man = E1 / "fit" / arm / f"manifest_{arm}.json"
    if not man.exists():
        candidates = list((E1 / "fit" / arm).glob("manifest*.json")) if (E1 / "fit" / arm).is_dir() else []
        if not candidates:
            return None, None
        man = candidates[0]
    data = json.loads(man.read_text())
    stages = [s for s in data.get("stages", []) if "strict" in s]
    return data, stages


def main():
    summary = {"cross_cells": {}, "diagonals": {}, "fit": {}, "branch_table": {}}

    for name in ["w_on_t_z", "w_zf_t_0"]:
        summary["cross_cells"][name] = load_diag(E1 / "cross" / name)
    summary["diagonals"]["T0"] = load_diag(R11 / "out" / "task128")
    summary["diagonals"]["ZF"] = load_diag(R11 / "out_zf" / "task128")
    summary["diagonals"]["ON"] = load_diag(R11 / "out_on" / "task128")

    for arm in ["ZF", "ON"]:
        man, stages = fit_stages(arm)
        if not stages:
            summary["fit"][arm] = {"status": "MISSING"}
            continue
        traj = {}
        for s in stages:
            traj.setdefault(s["cell"], []).append(
                {k: s.get(k) for k in ["step", "strict", "lenient", "rows",
                                       "groups_both", "groups", "eos_rate",
                                       "mean_CE", "min_worst_margin",
                                       "mean_worst_margin", "frac_positive_worst_margin"]})
        for cell in traj:
            traj[cell].sort(key=lambda r: r["step"])
        summary["fit"][arm] = {"trajectory": traj,
                               "contract_check": man.get("diagonal_contract_check"),
                               "status": man.get("status")}

    # §4.3 branch logic on ZF (the arm with far-side signal).
    zf = summary["fit"].get("ZF", {})
    train_far = zf.get("trajectory", {}).get("train:far", [])
    val_far = zf.get("trajectory", {}).get("validation:far", [])
    if train_far:
        best_train = max(train_far, key=lambda r: (r["strict"], r["mean_worst_margin"]))
        train_fits = best_train["strict"] > 0 and best_train["mean_worst_margin"] > 0
        best_val = max(val_far, key=lambda r: r["strict"]) if val_far else None
        val_fails = best_val is None or best_val["strict"] <= max(2, int(0.1 * best_val["rows"]))
        if not train_fits:
            verdict = "DID_NOT_LEARN: train-far content never robustly fitted -> learning-signal problem (Pro §4.4 gradient trap); intervention must change what supervision reaches, not just table/weights"
        elif val_fails:
            verdict = "DID_NOT_TRANSFER: train-far fitted but validation-far fails -> generalization/transfer problem; intervention targets regularization or data coverage"
        else:
            verdict = "BOTH_FIT: check decoding contract and native-compatibility readouts"
        summary["branch_table"] = {
            "best_train_far_step": best_train,
            "best_val_far_step": best_val,
            "verdict": verdict,
            "cross_cell_note": ("W_ONxT_Z vs ZF-diag isolates table effect under ON weights; "
                                "W_ZFxT_0 vs ON-diag isolates weight effect under native table"),
        }
    (E1 / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["branch_table"], indent=1, ensure_ascii=False))
    for arm in ["ZF", "ON"]:
        traj = summary["fit"].get(arm, {}).get("trajectory", {})
        for cell, rows in traj.items():
            line = " ".join(f"s{r['step']}:{r['strict']}/{r['rows']} m={r['mean_worst_margin']:.2f}"
                            for r in rows)
            print(f"[{arm} {cell}] {line}")
    print("E1_SUMMARY_WRITTEN", E1 / "summary.json")


if __name__ == "__main__":
    main()
