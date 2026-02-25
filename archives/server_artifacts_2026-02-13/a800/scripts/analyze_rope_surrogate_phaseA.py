#!/usr/bin/env python3
"""Phase-A surrogate analysis for RoPE frequency spectra.

Given candidate spectra (omega_k), compute:
- collision similarity sidelobe S(Delta)
- OOD increment per channel DeltaV = max(0, V_target - V_train)
- risk summaries R_coll_all / R_coll_ood / R_ood

Optionally join with PPL results and report risk-vs-PPL correlations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def sinc(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-12
    out[nz] = np.sin(x[nz]) / x[nz]
    return out


def similarity_curve(omega: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    phase = np.outer(deltas, omega)
    return np.cos(phase).mean(axis=1)


def V_of_omega(omega: np.ndarray, length: int) -> np.ndarray:
    x = omega * (0.5 * float(length))
    return 1.0 - sinc(x) ** 2


def compute_risks(omega: np.ndarray, l_train: int, l_target: int) -> Dict[str, Any]:
    delta_all = np.arange(1, l_target + 1, dtype=np.float64)
    delta_ood = np.arange(l_train + 1, l_target + 1, dtype=np.float64)
    s_all = similarity_curve(omega, delta_all)
    s_ood = similarity_curve(omega, delta_ood)

    v_train = V_of_omega(omega, l_train)
    v_target = V_of_omega(omega, l_target)
    dv = np.clip(v_target - v_train, a_min=0.0, a_max=None)

    return {
        "s_all_abs": np.abs(s_all),
        "s_ood_abs": np.abs(s_ood),
        "v_train": v_train,
        "v_target": v_target,
        "dv": dv,
        "r_coll_all": float(np.max(np.abs(s_all))),
        "r_coll_ood": float(np.max(np.abs(s_ood))),
        "r_ood": float(np.mean(dv)),
    }


def parse_names(spec: str) -> List[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def load_candidates(path: Path) -> Dict[str, Dict[str, Any]]:
    raw = json.loads(path.read_text())
    out: Dict[str, Dict[str, Any]] = {}
    for c in raw["candidates"]:
        out[c["name"]] = c
    return out


def load_ppl_map(path: Path | None, target_length: int) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: Dict[str, float] = {}
    key = str(target_length)

    # stage1/stage2-per-seed format
    if isinstance(raw.get("variants"), dict):
        for name, data in raw["variants"].items():
            if "ppl" in data and key in data["ppl"]:
                out[name] = float(data["ppl"][key]["mean"])
        return out

    # stage2 aggregate format
    if isinstance(raw.get("aggregate"), dict):
        for name, by_len in raw["aggregate"].items():
            if key in by_len:
                out[name] = float(by_len[key]["mean"])
        return out

    return out


def pearson(x: List[float], y: List[float]) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 2:
        return float("nan")
    xv = xa - xa.mean()
    yv = ya - ya.mean()
    den = math.sqrt(float((xv * xv).sum() * (yv * yv).sum()))
    if den <= 0:
        return float("nan")
    return float((xv * yv).sum() / den)


def rankdata(vals: List[float]) -> List[float]:
    arr = list(vals)
    idx = sorted(range(len(arr)), key=lambda i: arr[i])
    ranks = [0.0] * len(arr)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[idx[j + 1]] == arr[idx[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = r
        i = j + 1
    return ranks


def spearman(x: List[float], y: List[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def select_default_names(candidates: Dict[str, Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    names.extend(sorted([n for n in candidates if n.startswith("geom_theta_")]))
    if "bimodal_heuristic" in candidates:
        names.append("bimodal_heuristic")
    cosd = sorted([n for n in candidates if n.startswith("cosd_")])
    names.extend(cosd[:2])
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates_json", type=str, default="/opt/dfrope/results/cosd_search/latest.json")
    ap.add_argument("--results_json", type=str, default="/opt/dfrope/results/cosd_stage1/results.json")
    ap.add_argument("--out_dir", type=str, default="/opt/dfrope/results/cosd_phaseA_surrogate")
    ap.add_argument("--L_train", type=int, default=2048)
    ap.add_argument("--L_target", type=int, default=16384)
    ap.add_argument("--ppl_length", type=int, default=12288)
    ap.add_argument("--names", type=str, default="")
    ap.add_argument("--skip_plots", action="store_true")
    args = ap.parse_args()

    cand_path = Path(args.candidates_json)
    res_path = Path(args.results_json) if args.results_json else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(cand_path)
    names = parse_names(args.names) if args.names else select_default_names(candidates)
    names = [n for n in names if n in candidates]
    if not names:
        raise ValueError("no candidate names selected")

    ppl = load_ppl_map(res_path, args.ppl_length)

    rows = []
    for name in names:
        omega = np.asarray(candidates[name]["omega"], dtype=np.float64)
        risk = compute_risks(omega, args.L_train, args.L_target)
        rows.append(
            {
                "name": name,
                "source": candidates[name]["source"],
                "r_coll_all": risk["r_coll_all"],
                "r_coll_ood": risk["r_coll_ood"],
                "r_ood": risk["r_ood"],
                "combined_0p5": 0.5 * risk["r_coll_ood"] + 0.5 * risk["r_ood"],
                "ppl": ppl.get(name),
                "s_all_abs": risk["s_all_abs"].tolist(),
                "dv": risk["dv"].tolist(),
            }
        )

    rows_sorted = sorted(rows, key=lambda r: r["combined_0p5"])

    corr = {}
    rows_with_ppl = [r for r in rows_sorted if r["ppl"] is not None]
    if len(rows_with_ppl) >= 3:
        x_coll = [r["r_coll_ood"] for r in rows_with_ppl]
        x_ood = [r["r_ood"] for r in rows_with_ppl]
        x_comb = [r["combined_0p5"] for r in rows_with_ppl]
        y = [r["ppl"] for r in rows_with_ppl]
        corr = {
            "n": len(rows_with_ppl),
            "pearson_coll_vs_ppl": pearson(x_coll, y),
            "pearson_ood_vs_ppl": pearson(x_ood, y),
            "pearson_comb_vs_ppl": pearson(x_comb, y),
            "spearman_comb_vs_ppl": spearman(x_comb, y),
        }

    payload = {
        "config": {
            "candidates_json": str(cand_path),
            "results_json": str(res_path) if res_path else None,
            "L_train": args.L_train,
            "L_target": args.L_target,
            "ppl_length": args.ppl_length,
            "selected_names": names,
        },
        "rows_sorted_by_combined_0p5": rows_sorted,
        "correlation": corr,
    }
    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps(payload, indent=2))

    if not args.skip_plots:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            print("[warn] matplotlib not installed; skip plot generation")
            plt = None

        if plt is not None:
            deltas = np.arange(1, args.L_target + 1, dtype=np.float64)
            plt.figure(figsize=(10, 5))
            for r in rows_sorted:
                plt.plot(deltas, r["s_all_abs"], label=r["name"])
            plt.axvline(args.L_train, linestyle="--", linewidth=1, color="gray")
            plt.xlabel("Delta")
            plt.ylabel("|S(Delta)|")
            plt.ylim(0.0, 1.0)
            plt.xlim(1, args.L_target)
            plt.title("Collision Sidelobe Curves")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / "sidelobe_curves_selected.png", dpi=150)
            plt.close()

            K = len(rows_sorted[0]["dv"])
            ch = np.arange(K)
            plt.figure(figsize=(10, 5))
            for r in rows_sorted:
                plt.plot(ch, r["dv"], marker="o", markersize=2, linewidth=1.1, label=r["name"])
            plt.xlabel("Channel index k")
            plt.ylabel("DeltaV = max(0, V_target - V_train)")
            plt.title("OOD Increment by Channel")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / "ood_increment_selected.png", dpi=150)
            plt.close()

            if len(rows_with_ppl) >= 3:
                plt.figure(figsize=(6, 5))
                for r in rows_with_ppl:
                    plt.scatter(r["combined_0p5"], r["ppl"], s=40)
                    plt.annotate(r["name"], (r["combined_0p5"], r["ppl"]), fontsize=8, xytext=(4, 2), textcoords="offset points")
                plt.xlabel("0.5*R_coll_ood + 0.5*R_ood")
                plt.ylabel(f"PPL@{args.ppl_length}")
                plt.title("Surrogate Risk vs PPL")
                plt.tight_layout()
                plt.savefig(out_dir / f"risk_vs_ppl_{args.ppl_length}.png", dpi=150)
                plt.close()

    print("[done] wrote", out_json)
    print("Top by combined_0p5:")
    for r in rows_sorted[: min(10, len(rows_sorted))]:
        ppl_txt = f"{r['ppl']:.3f}" if r["ppl"] is not None else "na"
        print(
            f"- {r['name']:28s} src={r['source']:18s} "
            f"Rcoll_ood={r['r_coll_ood']:.4f} Rood={r['r_ood']:.4f} "
            f"comb={r['combined_0p5']:.4f} ppl={ppl_txt}"
        )
    if corr:
        print("Correlation:", json.dumps(corr))


if __name__ == "__main__":
    main()
