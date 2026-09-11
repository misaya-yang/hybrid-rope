#!/usr/bin/env python3
"""Native-preservation metric: output KL and the Fisher it implies.

The constraint side of the allocation problem is not "how far did the
frequencies move" but "how far did the model's output distribution move on the
distribution it was trained for".  Write it as

    D_N(d) = E_u KL( p_{x_b}(.|u) || p_{x_b+d}(.|u) )

for a step d in the log-frequency table.  Then D_N(0) = 0, the first-order
term is absent by construction, and

    D_N(d) = (1/2) d^T F_N d + O(||d||^3),   F_N = E[ J^T (diag p - p p^T) J ] >= 0

with J = d log p / d x.  F_N is the exact Gauss-Newton/Fisher of the output;
because sum_v p_v grad log p_v = 0 identically, the usual indefinite
cross-entropy Hessian term is gone and F_N is genuinely PSD.  This is the
quantity that decides which frequencies a frozen checkpoint can afford to move,
and it makes no assumption about which band is which.

Three estimators, cheapest first:

  diag   one forward per slot.  D_N(delta e_j) = 0.5 F_jj delta^2 + O(delta^3)
         and D_N(0) = 0 exactly, so a single one-sided forward isolates F_jj.
         No backward at any length.  64 forwards.
  cross  adds one forward per named pair, which is what makes a specific
         coupling claim (e.g. slots 28/29) testable for a handful of forwards.
  mc     one forward plus T backwards on the retained graph.  Each backward
         gives grad log p_{v_t}(u_t) for one sampled token and position, and
         E[grad grad^T] is F_N exactly.  Gives the full 64x64 PSD matrix, so
         the solver keeps the cross-frequency coupling instead of dropping it.

Usage (run from the repository root, on the GPU host):

  python -m experiments.curvature_20260910.local_probe \
      --model /root/autodl-tmp/rope_qwen_baseline_20260907/model \
      --npy   /root/autodl-tmp/bm_transfer_20260908/prepared_nll_01/doc_00.npy \
      --length 32768 --base-table mrpro_n17 --fisher diag --out runs/fisher_mrpro_32k.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import tables as T
from .model import (FrozenRoPE, fisher_diagonal, fisher_cross,
                    fisher_scaling, fisher_all_at_once)


def load_ids(args, model):
    if args.npy:
        arr = np.load(args.npy)
        if args.offset + args.length > arr.shape[0]:
            raise ValueError(f"corpus {args.npy} has {arr.shape[0]} tokens, need {args.offset + args.length}")
        ids = torch.tensor(arr[args.offset:args.offset + args.length].astype(np.int64))
    else:
        with open(args.text, encoding="utf-8", errors="ignore") as f:
            ids = model.tokenizer(f.read(), return_tensors="pt", add_special_tokens=False).input_ids[0]
        if ids.shape[0] < args.offset + args.length:
            raise ValueError(f"corpus has {ids.shape[0]} tokens, need {args.offset + args.length}")
        ids = ids[args.offset:args.offset + args.length]
    return ids[None].to(model.device)


def mc_fisher(model, ids, keep, base_table, n_samples, seed=20260910, batch_report=16):
    """Full PSD Fisher from one forward and `n_samples` backwards on its graph.

    Sampling (position, token) pairs and averaging grad log p_v(u) grad log p_v(u)^T
    estimates E[J^T(diag p - pp^T)J] with the second term vanishing identically --
    not approximately, since sum_v p_v grad log p_v = grad sum_v p_v = 0.

    The frequencies must be installed with track_grad=True or the graph is
    detached at the rotation and the backward raises; that is why this does not
    go through install_table.
    """
    model.install(np.asarray(base_table["values_float32"], dtype=np.float64),
                  base_table["gain"], track_grad=True)
    lg = model.logits(ids, keep, want_grad=True)          # (keep, V), float32, graph alive
    with torch.enable_grad():
        logp = torch.log_softmax(lg, dim=-1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = torch.randint(0, logp.shape[0], (n_samples,), generator=g).to(logp.device)
    with torch.no_grad():
        tok = torch.multinomial(logp[pos].detach().exp().cpu(), 1, generator=g).to(logp.device).squeeze(-1)
    v = model.rotary.inv_freq
    F = torch.zeros((T.K, T.K), dtype=torch.float64, device=logp.device)
    for i in range(n_samples):
        gr, = torch.autograd.grad(logp[pos[i], tok[i]], v, retain_graph=(i + 1 < n_samples))
        gr = gr.detach().float().double()
        F += torch.outer(gr, gr)
        if (i + 1) % batch_report == 0:
            print(f"  fisher mc {i+1}/{n_samples}", flush=True)
    F /= n_samples
    F = F.cpu().numpy()
    # symmetrise away numerical drift, then report the spectrum
    F = 0.5 * (F + F.T)
    ev = np.linalg.eigvalsh(F)
    return F, dict(eigen_min=float(ev.min()), eigen_max=float(ev.max()),
                   condition_number=float(ev.max() / max(ev.min(), 1e-300)),
                   trace=float(np.trace(F)),
                   top8_share=float(ev[-8:].sum() / max(ev.sum(), 1e-300)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npy", default=None)
    ap.add_argument("--text", default=None)
    ap.add_argument("--length", type=int, default=32768)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--keep", type=int, default=512,
                    help="positions scored; the native window is long, the tail is enough")
    ap.add_argument("--base-table", default="mrpro_n17", choices=sorted(T.CONSTRUCTIONS))
    ap.add_argument("--gain", type=float, default=T.GAIN_YARN)
    ap.add_argument("--fisher", default="diag", choices=["diag", "cross", "mc", "none"])
    ap.add_argument("--delta", type=float, default=2e-2, help="log-frequency step for the KL probes")
    ap.add_argument("--slots", default="all")
    ap.add_argument("--pairs", default="28:29,24:39,0:63,31:32,35:39,23:24,39:40,50:51")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--scaling-check", default="", dest="scaling_check",
                    help="slots to probe at delta/2delta/4delta; 'auto' picks 6 spread slots. "
                         "Empty disables.  Default is on for --fisher diag|cross.")
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp32"],
                    help="fp32 by default and it matters: bf16 rounds the logits at ~4e-3 "
                         "relative, which is the size of the whole perturbation response for "
                         "a 2e-2 step on some slots, so a bf16 Fisher measures the rounding. "
                         "32K in fp32 fits comfortably with logits_to_keep; the 128K long "
                         "gradient stays bf16 for memory and uses per-token differencing "
                         "instead.  Override only to show the difference.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = FrozenRoPE(args.model, dtype=args.dtype)
    ids = load_ids(args, model)
    base = T.build(args.base_table, gain=args.gain)

    if args.slots == "all":
        slots = list(range(T.K))
    elif ":" in args.slots:
        a, b = args.slots.split(":")
        slots = list(range(int(a), int(b)))
    else:
        slots = [int(s) for s in args.slots.split(",")]
    pairs = [tuple(int(x) for x in p.split(":")) for p in args.pairs.split(",") if p]

    started = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Baseline distribution of the reference table: everything below is a KL
    # against this, so it is computed once and reused.
    model.install_table(base)
    base_logp = model.log_probs(ids, args.keep)
    rec = dict(table=args.base_table, gain=args.gain, m=[float(x) for x in base["m"]],
               inv_freq=[float(x) for x in base["values_float32"]],
               length=int(ids.shape[1]), keep=args.keep, fisher=args.fisher,
               delta=args.delta, model_path=args.model, dtype=args.dtype,
               npy=args.npy, text=args.text, offset=args.offset)

    if args.fisher in ("diag", "cross"):
        t0 = time.monotonic()
        rec["fisher_diag"] = {str(k): v for k, v in
                              fisher_diagonal(model, ids, args.keep, base_logp, base, args.delta, slots).items()}
        rec["fisher_diag_seconds"] = time.monotonic() - t0
    if args.fisher == "cross":
        t0 = time.monotonic()
        cross, diag = fisher_cross(model, ids, args.keep, base_logp, base, args.delta, pairs)
        rec["fisher_cross"] = {f"{j}:{k}": v for (j, k), v in cross.items()}
        rec["fisher_cross_seconds"] = time.monotonic() - t0
        rec["fisher_diag"] = {str(k): v for k, v in diag.items()}
    if args.fisher == "mc":
        t0 = time.monotonic()
        F, spec = mc_fisher(model, ids, args.keep, base, args.samples)
        rec["fisher_matrix"] = F.tolist()
        rec["fisher_spectrum"] = spec
        rec["fisher_mc_seconds"] = time.monotonic() - t0
        rec["fisher_diag"] = {str(j): float(F[j, j]) for j in range(T.K)}

    # Diagonal dominance: one extra forward.  A large joint/diag ratio is
    # expected for any correlated metric (64*63 off-diagonal terms against 64
    # diagonal ones) so this is a warning line, not a verdict; the precise
    # comparison happens on the solver's own step in solve_kkt.py.
    d0 = rec.get("fisher_diag") or {}
    if d0 and args.fisher != "none":
        rec["all_at_once"] = fisher_all_at_once(
            model, ids, args.keep, base_logp, base, args.delta,
            {int(k): v for k, v in d0.items()})

    # Is the response quadratic at this delta?  Everything downstream assumes
    # yes; this is the cheapest direct test of it.
    want = args.scaling_check or ("auto" if args.fisher in ("diag", "cross") else "")
    if want and args.fisher != "none":
        probe_slots = ([0, 16, 23, 28, 40, 63] if want == "auto"
                       else [int(s) for s in want.split(",")])
        probe_slots = [j for j in probe_slots if j in slots]
        if probe_slots:
            rec["scaling"] = fisher_scaling(model, ids, args.keep, base_logp, base,
                                            probe_slots, args.delta)

    # Native NLL of the reference table, for the record and for the receipts.
    rec["base_nll"] = float(model.nll(ids, args.keep))

    # Slot order must stay non-increasing in nu for the table to be a rotation
    # grid at all; report it rather than assume it.
    nu = np.asarray(base["values_float32"])
    rec["ordered"] = bool(np.all(np.diff(nu) <= 0))
    rec["elapsed_seconds"] = time.monotonic() - started
    if torch.cuda.is_available():
        rec["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("fisher_matrix", "m", "inv_freq")}, indent=1)[:2000])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
