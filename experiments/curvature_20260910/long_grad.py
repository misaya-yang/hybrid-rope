#!/usr/bin/env python3
"""The long-range side of the allocation problem: g_L = d(Long loss)/d(log freq).

At a fixed base table (MrRoPE by default) this measures, per slot, how the long
loss responds to slowing that slot down further.  g_j < 0 means slot j still has
extrapolation left to buy; g_j > 0 means it is already over-compressed and
protecting it is free native budget.

Measured forward-only by central differences, so nothing depends on autograd
surviving a 128K graph and no activation is stored.  The per-token loss
difference is taken before averaging, which cancels the O(1) common part
exactly and leaves a quantity set by delta^2 -- the reason this is usable in
bf16 at all.

Two long objectives:

  nll   tail-token NLL on a long natural-text prefix.  This is the screen the
        project already uses (screen target-length PPL before releasing a
        downstream test), so g_L points at the thing the protocol optimises.
  gold  teacher-forced NLL of the reference answer on the prepared benchmark
        rows.  Task-level rather than corpus-level: it scores the binding
        margin instead of the average token.  Slower per slot (one row per
        forward) and only as good as the prepared references.

Usage:

  python -m experiments.curvature_20260910.long_grad \
      --model /root/autodl-tmp/rope_qwen_baseline_20260907/model \
      --npy   /root/autodl-tmp/bm_transfer_20260908/prepared_nll_01/doc_00.npy \
      --length 131072 --base-table mrpro_n17 --slots bridge \
      --out runs/gl_mrpro_128k.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from . import tables as T
from .model import FrozenRoPE

BRIDGE = list(range(23, 41))                      # MrPro / YaRN transition
HIGH = [0, 4, 8, 12, 16, 20]                      # samples of the fast band
LOW = [44, 48, 52, 56, 60, 63]                    # samples of the slow band
SLOT_SETS = {
    "all": list(range(T.K)),
    "bridge": BRIDGE,
    "wide": sorted(set(BRIDGE + HIGH + LOW)),
}


def perturb(base_values, j, eps):
    v = np.asarray(base_values, dtype=np.float64).copy()
    v[j] = v[j] * math.exp(eps)
    return v


def load_nll_ids(args, model):
    arr = np.load(args.npy)
    need = args.offset + args.length
    if need > arr.shape[0]:
        raise ValueError(f"{args.npy} has {arr.shape[0]} tokens, need {need}")
    return torch.tensor(arr[args.offset:need].astype(np.int64))[None].to(model.device)


def load_bind_rows(args, model):
    """Rows from a prepared screen.jsonl: prompt ids plus the gold answer text."""
    rows = []
    with open(args.screen) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if args.length and r.get("length_cap") != args.length:
                continue
            refs = r.get("references") or []
            if not refs:
                continue
            prompt = torch.tensor(r["prompt_ids"], dtype=torch.long)
            ans = model.tokenizer("\n" + str(refs[0]), add_special_tokens=False).input_ids
            rows.append(dict(row_id=r["row_id"], task=r["task"],
                             ids=torch.cat([prompt, torch.tensor(ans)]).to(model.device),
                             n_answer=len(ans)))
    if not rows:
        raise ValueError(f"no usable rows in {args.screen} for length_cap={args.length}")
    return rows


def mean_answer_nll(model, row):
    """NLL of the answer tokens only, teacher-forced after the prompt."""
    ids = row["ids"]
    keep = row["n_answer"] + 1
    lg = model.logits(ids[None], keep)
    tgt = ids[-row["n_answer"]:]
    return float(torch.nn.functional.cross_entropy(lg[:-1], tgt, reduction="mean"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npy", default=None, help="long natural-text prefix")
    ap.add_argument("--screen", default=None, help="prepared screen.jsonl for --long-loss gold")
    ap.add_argument("--long-loss", default="nll", choices=["nll", "gold"])
    ap.add_argument("--length", type=int, default=131072)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--keep", type=int, default=512, help="tail tokens scored in nll mode")
    ap.add_argument("--base-table", default="mrpro_n17", choices=sorted(T.CONSTRUCTIONS))
    ap.add_argument("--gain", type=float, default=T.GAIN_YARN)
    ap.add_argument("--slots", default="wide", choices=sorted(SLOT_SETS))
    ap.add_argument("--eps", type=float, default=2e-2, help="log-frequency step for the differences")
    ap.add_argument("--one-sided", action="store_true", help="half the forwards, O(eps) bias")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = FrozenRoPE(args.model, dtype=args.dtype)
    base = T.build(args.base_table, gain=args.gain)
    slots = SLOT_SETS[args.slots]

    if args.long_loss == "nll":
        if not args.npy:
            ap.error("--long-loss nll needs --npy")
        ids = load_nll_ids(args, model)
        model.install_table(base)
        with torch.no_grad():
            nll0 = model.nll_per_token(ids, args.keep).cpu().numpy()

        def loss_with(eps_map):
            v = base["values_float32"].copy()
            for j, e in eps_map.items():
                v[j] = v[j] * math.exp(e)
            model.install(v, base["gain"])
            with torch.no_grad():
                return model.nll_per_token(ids, args.keep).cpu().numpy()
        ctx = dict(tokens=int(ids.shape[1]), keep=args.keep)
    else:
        if not args.screen:
            ap.error("--long-loss gold needs --screen")
        rows = load_bind_rows(args, model)
        model.install_table(base)
        nll0 = np.array([mean_answer_nll(model, r) for r in rows])
        keep = args.keep

        def loss_with(eps_map):
            v = base["values_float32"].copy()
            for j, e in eps_map.items():
                v[j] = v[j] * math.exp(e)
            model.install(v, base["gain"])
            return np.array([mean_answer_nll(model, r) for r in rows])
        ctx = dict(rows=[r["row_id"] for r in rows], n_rows=len(rows))

    started = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Per-unit loss difference, then average.  Differencing before averaging is
    # what keeps the O(1) part from swamping the O(eps^2) signal in bf16.
    grad, curv, raw = {}, {}, {}
    for j in slots:
        d_up = float((loss_with({j: +args.eps}) - nll0).mean())
        if args.one_sided:
            grad[j], curv[j], raw[j] = d_up / args.eps, None, dict(d_up=d_up)
        else:
            d_dn = float((loss_with({j: -args.eps}) - nll0).mean())
            grad[j] = (d_up - d_dn) / (2.0 * args.eps)
            # second difference of the long loss; the local model drops it, so
            # it is the cheapest honest check on how far the linear term reaches
            curv[j] = (d_up + d_dn) / (args.eps * args.eps)
            raw[j] = dict(d_up=d_up, d_dn=d_dn)
        print(json.dumps(dict(slot=j, g=grad[j], curvature=curv[j])), flush=True)

    rec = dict(base_table=args.base_table, gain=args.gain, long_loss=args.long_loss,
               length=args.length, offset=args.offset, eps=args.eps,
               one_sided=bool(args.one_sided), slots=slots,
               grad={str(j): grad[j] for j in slots},
               curvature={str(j): curv[j] for j in slots},
               raw={str(j): raw[j] for j in slots},
               base_loss=float(nll0.mean()), m=[float(x) for x in base["m"]],
               inv_freq=[float(x) for x in base["values_float32"]],
               model_path=args.model, dtype=args.dtype,
               elapsed_seconds=time.monotonic() - started, **ctx)
    if torch.cuda.is_available():
        rec["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"wrote {args.out}  base_loss={rec['base_loss']:.4f}  "
          f"g range [{min(grad.values()):+.4e}, {max(grad.values()):+.4e}]  "
          f"{rec['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
