#!/usr/bin/env python3
"""P0 diagnostic A — content-fork E/B decomposition (Pro synthesis §11.2 / §6.1 / §6.2).

Read-only, teacher-forced. For each legal two-world counterfactual pair
(transport_views, worlds 0/1, same semantic_id/layout/length) we locate the
first differing target token (the content fork), run ONE forward pass per
world with the shared target prefix teacher-forced, take FULL-VOCAB logits at
the fork position, and compute:

  d0 = z0[a0] - z0[a1]          (world-0 log-odds of the two answers)
  d1 = z1[a0] - z1[a1]
  E  = (d0 - d1)/2              (evidence-direction component)
  B  = (d0 + d1)/2              (common bias component)
  pair_order_correct <=> E > |B|   (both worlds rank own answer above the other)

plus full-vocab margins (gold vs best competitor, gold vs best NON-candidate
competitor) and the exact paired-CE decomposition of §6.2:

  pair_ce = [softplus(-d0) + softplus(d1)]/2  -  [log P0 + log P1]/2
            (conditional discrimination)          (mass allocation)

Registered strict generation scores from existing per-row receipts are joined
UNCHANGED (no rescoring, no candidate reranking, no decoding changes). This is
a teacher-forced diagnostic, not a generation claim.

Usage (one system per run; single GPU process):
  python diag_fork_eb.py --model <1B> --tables <tables_dir> \
      --system-id ZF --arm Z --adapter <round11 ZF step_128 dir> \
      --views /root/autodl-tmp/claude_round11_olmo_20260905/olmo_tasks/transport_views.jsonl \
      --receipts /root/.../out_zf/task128/examples.jsonl \
      --family single_evidence --layouts compact far near \
      --out $B12/diag/fork_eb_ZF
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from track_a_eval import load_and_install, sha256_bytes


def first_fork(t0: list[int], t1: list[int]) -> int:
    f = 0
    m = min(len(t0), len(t1))
    while f < m and t0[f] == t1[f]:
        f += 1
    return f


def softplus(x: float) -> float:
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


@torch.inference_mode()
def fork_logits(model, ids: list[int], device: str) -> torch.Tensor:
    """Full-vocab fp32 logits at the LAST position, teacher-forced."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(x, use_cache=False, logits_to_keep=1)
    return out.logits[0, -1].float()


def margins(z: torch.Tensor, gold: int, cand: set[int]) -> dict:
    zg = float(z[gold])
    zc = z.clone()
    zc[gold] = float("-inf")
    best_other = int(zc.argmax())
    gap_all = zg - float(zc.max())
    zc2 = zc.clone()
    for c in cand:
        if c != gold:
            zc2[c] = float("-inf")
    if torch.isfinite(zc2).any():
        best_outside = int(zc2.argmax())
        gap_outside = zg - float(zc2.max())
    else:
        best_outside, gap_outside = -1, float("nan")
    return {"gold": gold, "gap_vs_best_other": gap_all,
            "best_other": best_other,
            "gap_vs_best_non_candidate": gap_outside,
            "best_non_candidate": best_outside}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--system-id", required=True,
                    help="label for this product, e.g. T0/Z0/ZC/ZF/ON")
    ap.add_argument("--arm", required=True, choices=["N", "Z"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--views", required=True, help="transport_views.jsonl")
    ap.add_argument("--receipts", default=None,
                    help="examples.jsonl with per-row registered strict receipts")
    ap.add_argument("--family", default="single_evidence")
    ap.add_argument("--layouts", nargs="*", default=["compact", "far", "near"])
    ap.add_argument("--max-groups", type=int, default=10**9)
    ap.add_argument("--split", default=None,
                    help="optional split filter (default: all splits present)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("output complete already; preserved, not overwritten")

    model, tok, identity = load_and_install(args.model, Path(args.tables), args.arm,
                                            adapter_dir=args.adapter,
                                            extra_path=args.extra)
    model.to(args.device)

    # ---- registered strict receipts (joined unchanged) -------------------
    reg = {}
    if args.receipts:
        for line in Path(args.receipts).read_text().splitlines():
            r = json.loads(line)
            reg[(r["semantic_id"], int(r["world"]), r["layout"], int(r["length_cap"]))] = \
                bool(r["full_exact_eos"])

    # ---- stream legal two-world pairs ------------------------------------
    groups: dict = {}
    order: list = []
    for line in Path(args.views).read_text().splitlines():
        r = json.loads(line)
        if r["family"] != args.family or r["layout"] not in set(args.layouts):
            continue
        if args.split and r.get("split") != args.split:
            continue
        key = (r["semantic_id"], r["layout"], int(r["length_cap"]))
        if key not in groups:
            groups[key] = {}
            order.append(key)
        groups[key][int(r["world"])] = r

    rows_path = out / "fork_eb_rows.jsonl"
    t0 = time.time()
    n = 0
    with rows_path.open("w") as fh:
        for key in order:
            if n >= args.max_groups:
                break
            g = groups[key]
            if 0 not in g or 1 not in g:
                continue
            r0, r1 = g[0], g[1]
            tgt0, tgt1 = r0["target_ids"], r1["target_ids"]
            f = first_fork(tgt0, tgt1)
            if f >= min(len(tgt0), len(tgt1)):
                continue  # identical targets: no content fork
            a0, a1 = tgt0[f], tgt1[f]
            shared = tgt0[:f]

            z = {}
            for w, r in ((0, r0), (1, r1)):
                ids = list(r["prompt_ids"]) + list(shared)
                z[w] = fork_logits(model, ids, args.device)

            d0 = float(z[0][a0] - z[0][a1])
            d1 = float(z[1][a0] - z[1][a1])
            E = (d0 - d1) / 2.0
            B = (d0 + d1) / 2.0

            # §6.2 exact paired-CE decomposition (full vocab)
            p = {w: torch.softmax(z[w], dim=-1) for w in (0, 1)}
            P0 = float(p[0][a0] + p[0][a1])
            P1 = float(p[1][a0] + p[1][a1])
            cond = (softplus(-d0) + softplus(d1)) / 2.0
            mass = (math.log(P0) + math.log(P1)) / 2.0 if P0 > 0 and P1 > 0 else float("nan")
            pair_ce = (-math.log(max(float(p[0][a0]), 1e-40))
                       - math.log(max(float(p[1][a1]), 1e-40))) / 2.0

            rec = {
                "system": args.system_id,
                "semantic_id": key[0], "layout": key[1], "length_cap": key[2],
                "split": r0.get("split"),
                "fork_index": f, "shared_prefix_len": f,
                "a0": a0, "a1": a1,
                "d0": d0, "d1": d1, "E": E, "B": B,
                "pair_order_correct": bool(E > abs(B)),
                "w0_own_first": bool(d0 > 0.0), "w1_own_first": bool(d1 < 0.0),
                "margins_w0": margins(z[0], a0, {a0, a1}),
                "margins_w1": margins(z[1], a1, {a0, a1}),
                "P0_mass": P0, "P1_mass": P1,
                "ce_conditional_discrimination": cond,
                "ce_mass_allocation": mass,
                "ce_pair_actual": pair_ce,
                "ce_identity_residual": (cond - mass) - pair_ce,
                "prompt_lens": [len(r0["prompt_ids"]), len(r1["prompt_ids"])],
                "registered_world_strict": [
                    reg.get((key[0], 0, key[1], key[2])),
                    reg.get((key[0], 1, key[1], key[2]))],
            }
            rec["registered_group_strict"] = (
                bool(rec["registered_world_strict"][0] and rec["registered_world_strict"][1])
                if all(v is not None for v in rec["registered_world_strict"]) else None)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            if n % 8 == 0:
                print(f"[{n}] {key[1]} L{key[2]} E={E:.3f} B={B:.3f} "
                      f"ok={rec['pair_order_correct']} wall={time.time()-t0:.0f}s", flush=True)

    # ---- summary ----------------------------------------------------------
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    cells = {}
    for lay in sorted({r["layout"] for r in rows}):
        sub = [r for r in rows if r["layout"] == lay]
        cells[lay] = {
            "pairs": len(sub),
            "pair_order_correct": sum(r["pair_order_correct"] for r in sub),
            "mean_E": sum(r["E"] for r in sub) / len(sub),
            "mean_B": sum(r["B"] for r in sub) / len(sub),
            "mean_gap_w0": sum(r["margins_w0"]["gap_vs_best_other"] for r in sub) / len(sub),
            "mean_gap_w1": sum(r["margins_w1"]["gap_vs_best_other"] for r in sub) / len(sub),
            "registered_group_strict": (
                sum(1 for r in sub if r["registered_group_strict"] is True)),
            "registered_groups_joined": sum(
                1 for r in sub if r["registered_group_strict"] is not None),
            "max_ce_identity_residual": max(abs(r["ce_identity_residual"]) for r in sub),
        }
    manifest = {
        "status": "DIAG_FORK_EB_COMPLETE",
        "diagnostic": "pro_synthesis_11.2_content_fork_EB",
        "system_id": args.system_id,
        "identity": identity,
        "views_path": args.views,
        "family": args.family, "layouts": sorted({r["layout"] for r in rows}),
        "n_pairs": len(rows),
        "per_layout": cells,
        "wall_seconds": round(time.time() - t0, 1),
        "notes": [
            "teacher-forced diagnostic at first differing target token; NOT a generation score",
            "registered strict scores joined unchanged from existing receipts (if provided)",
            "no new table/gain/head selection; no candidate reranking; strict definition unchanged",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
