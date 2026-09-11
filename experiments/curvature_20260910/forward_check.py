#!/usr/bin/env python3
"""Real forwards on the solved step: does the local model survive contact?

solve_kkt.py predicts two numbers from a quadratic model built on in-window
data: an output-KL cost and a long-loss gain.  Neither is allowed to be
believed.  This file spends the GPU time to check both, by walking the step
itself -- alpha * d* for alpha in {1/4, 1/2, 1} -- and reading the real losses.

Three things are measured, and each falsifies a different assumption:

1. THE NATIVE COST SCALES.  D_N(alpha d*) should equal alpha^2 * D_N(d*).  The
   quadratic model says the exponent is exactly 2.  A fitted exponent far from
   2 means the step left the region where F_N describes the model, and the
   predicted gain is meaningless no matter how good it looks.

2. THE LONG GAIN IS LINEAR, AND THE RIGHT SIZE.  The realised drop at step
   alpha should be alpha * g^T d*.  The pre-registered number is
   predicted = sqrt(2 eps G); the trust ratio is

       rho_trust = (measured drop at alpha=1) / predicted

   and the gate is 0.5 <= rho_trust <= 1.5.  Below the band the local model
   over-promised, above it the measurement or the baseline is off.  This ratio
   was fixed before any of it was run -- it is a prediction, not a summary.

3. THE DIRECTION IS NOT DECORATION.  Three budget-matched controls spend the
   SAME native KL: a random feasible direction, a uniform feasible direction,
   and the naive per-slot |g_j| / F_jj direction.  All four are projected onto
   A d = 0 and rescaled to 0.5 d^T F_N d = eps, so the only difference is where
   the budget goes.

   solver > controls   the Fisher whitening and the cross-frequency coupling
                       carry real information: the theory BEATS the field.
   solver ~ controls   direction is irrelevant and only the budget matters.
                       That is a finding too -- and it says every geometric
                       family is a budget choice wearing a shape.
   control > solver    the local model is wrong here, and the failure is in
                       F_N or in g_L, not in the optimizer.

`--mode ppl` is the project's own protocol: screen target-length PPL before
releasing a downstream test.  Tail-token NLL at each of several lengths for the
candidate against the base table, in one model load.

Five.6-pro's stopping rule applies verbatim: if the local response cannot be
reproduced by forward passes at all, stop -- the local model of the frequency
response is wrong and no later GPU work is worth releasing.

  python -m experiments.curvature_20260910.forward_check \
      --model /root/autodl-tmp/rope_qwen_baseline_20260907/model \
      --kkt runs/kkt_mrpro.json --npy .../doc_00.npy \
      --native-length 32768 --long-length 131072 \
      --screen-lengths 8192,32768,65536,131072 \
      --out runs/forward_check_mrpro.json
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
from .model import FrozenRoPE, output_kl
from .long_grad import SLOT_SETS, load_bind_rows, mean_answer_nll
from .solve_kkt import constraint_matrix


# ---------------------------------------------------------------------------
def load_step(path, slots):
    rec = json.load(open(path))
    if rec.get("degenerate"):
        raise ValueError(f"{path} is a degenerate receipt: the solver found no payable "
                         "direction (G <= 0).  There is no step to forward-check; report "
                         "the EXPLAINS branch instead.")
    d = np.zeros(T.K)
    for k, v in rec["d"].items():
        d[int(k)] = float(v)
    if not any(abs(d[j]) > 0 for j in slots):
        raise ValueError(f"{path}: step is zero on every slot in {slots}")
    return d, rec


def budget_matched(d_raw, F, diag, A, eps):
    """Project onto A d = 0, then scale to 0.5 d^T F d = eps.

    Controls must spend the same native KL as the solver step or the comparison
    is between budgets, not between directions.  Projection first, so the
    scaling is done inside the feasible subspace the solver also lives in.
    """
    K = T.K
    if A.shape[0]:
        # least-squares projection onto the null space of A (F-agnostic, and the
        # solver's own step already lies in it up to numerical noise)
        Q, _ = np.linalg.qr(A.T)
        d_p = d_raw - Q @ (Q.T @ d_raw)
    else:
        d_p = np.asarray(d_raw, dtype=np.float64).copy()
    if not np.isfinite(d_p).all() or np.linalg.norm(d_p) == 0:
        return None
    M = F if F is not None else np.diag(diag)
    q = float(d_p @ M @ d_p)
    if q <= 0:
        return None
    return d_p * math.sqrt(2.0 * eps / q)


def controls(step, F, diag, A, eps, slots, seed=20260910):
    """Four directions, one budget."""
    d_star = budget_matched(step, F, diag, A, eps)
    rng = np.random.default_rng(seed)

    d_rand = np.zeros(T.K)
    d_rand[slots] = rng.standard_normal(len(slots))
    d_rand = budget_matched(d_rand, F, diag, A, eps)

    d_unif = np.zeros(T.K)
    d_unif[slots] = 1.0
    d_unif = budget_matched(d_unif, F, diag, A, eps)

    # the naive benefit/cost direction, the one a per-slot reading would give
    d_ratio = np.zeros(T.K)
    d_ratio[slots] = -step[slots] / np.clip(np.abs(diag[slots]), 1e-300, None)
    d_ratio = budget_matched(d_ratio, F, diag, A, eps)

    out = {"solver": d_star}
    for name, d in (("random", d_rand), ("uniform", d_unif), ("per_slot", d_ratio)):
        if d is not None:
            out[name] = d
    return out


# ---------------------------------------------------------------------------
def fit_exponent(alphas, values):
    """log-log slope through the origin; values are KL costs, expected ~alpha^2."""
    a = np.asarray(alphas, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    ok = (a > 0) & (v > 0)
    if ok.sum() < 2:
        return None
    return float(np.sum(np.log(v[ok]) * np.log(a[ok])) / np.sum(np.log(a[ok]) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kkt", required=True, help="solve_kkt.py receipt")
    ap.add_argument("--npy", default=None, help="long natural-text corpus")
    ap.add_argument("--screen", default=None, help="prepared screen.jsonl (gold mode)")
    ap.add_argument("--long-loss", default="nll", choices=["nll", "gold"])
    ap.add_argument("--base-table", default=None, help="default: the receipt's base_table")
    ap.add_argument("--native-length", type=int, default=32768)
    ap.add_argument("--long-length", type=int, default=131072)
    ap.add_argument("--keep", type=int, default=512)
    ap.add_argument("--fracs", default="0.25,0.5,1.0")
    ap.add_argument("--slots", default="wide", choices=sorted(SLOT_SETS))
    ap.add_argument("--screen-lengths", default="8192,32768,65536,131072")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d, krec = load_step(args.kkt, SLOT_SETS[args.slots])
    base_table = args.base_table or krec["base_table"]
    gain = float(krec.get("gain", T.GAIN_YARN))
    eps = float(krec["eps"])
    slots = krec["slots"]

    model = FrozenRoPE(args.model, dtype=args.dtype)
    base = T.build(base_table, gain=gain)

    # ---- native side: in-window data, the distribution the checkpoint knows
    if not args.npy:
        ap.error("--npy is required (in-window corpus for the native KL)")
    arr = np.load(args.npy)
    if args.native_length > arr.shape[0]:
        raise ValueError(f"{args.npy} has {arr.shape[0]} tokens, need {args.native_length}")
    nid = torch.tensor(arr[:args.native_length].astype(np.int64))[None].to(model.device)
    model.install_table(base)
    base_logp = model.log_probs(nid, args.keep)

    # ---- long side
    if args.long_loss == "nll":
        if args.long_length > arr.shape[0]:
            raise ValueError(f"{args.npy} has {arr.shape[0]} tokens, need {args.long_length}")
        lid = torch.tensor(arr[:args.long_length].astype(np.int64))[None].to(model.device)
        lctx = dict(ids=lid, keep=args.keep)
    else:
        if not args.screen:
            ap.error("--long-loss gold needs --screen")
        lctx = dict(rows=load_bind_rows(args, model))

    def long_loss(table):
        model.install_table(table)
        if args.long_loss == "nll":
            with torch.no_grad():
                return float(model.nll_per_token(lctx["ids"], lctx["keep"]).mean())
        return float(np.mean([mean_answer_nll(model, r) for r in lctx["rows"]]))

    def native_kl(table):
        return output_kl(model, nid, args.keep, base_logp, table)

    started = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    base_long = long_loss(base)
    print(f"base {base_table}: long loss {base_long:.6f}  (length {args.long_length})", flush=True)

    dirs = {"solver": d}
    if not args.no_controls:
        if "fisher_diag" not in krec:
            raise ValueError("the KKT receipt has no fisher_diag; controls cannot be "
                             "budget-matched and the comparison would be meaningless")
        diag = np.array([float(krec["fisher_diag"][str(j)]) for j in range(T.K)])
        A = constraint_matrix(pin_ends=True, fix_sum=False)
        dirs.update(controls(d, None, diag, A, eps, slots))

    alphas = [float(x) for x in args.fracs.split(",")]
    rows = {}
    for name, dd in dirs.items():
        rows[name] = []
        for a in [0.0] + alphas:
            t = T.from_eps(a * dd, base_name=base_table, gain=gain)
            kl = float(native_kl(t))
            ll = long_loss(t) if a > 0 else base_long
            rows[name].append(dict(alpha=a, native_kl=kl, long_loss=ll,
                                   drop=base_long - ll))
            print(json.dumps(dict(direction=name, alpha=a, native_kl=kl,
                                  long_loss=ll, drop=base_long - ll)), flush=True)

    solver = rows["solver"]
    kl_curve = [r["native_kl"] for r in solver[1:]]
    drop_curve = [r["drop"] for r in solver[1:]]
    predicted = float(krec["pred_long_gain"])
    measured = drop_curve[-1]
    rho = measured / predicted if predicted else None
    exponent = fit_exponent(alphas, kl_curve)

    # control comparison at full step, all at the same native budget
    comp = {}
    for name, rs in rows.items():
        if rs[-1]["alpha"] != alphas[-1]:
            continue
        comp[name] = dict(native_kl=rs[-1]["native_kl"], drop=rs[-1]["drop"],
                          gain_per_kl=rs[-1]["drop"] / rs[-1]["native_kl"]
                          if rs[-1]["native_kl"] > 0 else None)
    if "solver" in comp:
        s = comp["solver"]["drop"]
        for name in list(comp):
            if name == "solver":
                comp[name]["vs_solver"] = 1.0
            else:
                comp[name]["vs_solver"] = comp[name]["drop"] / s if s else None

    gates = {}
    if rho is not None:
        gates["trust_ratio"] = dict(value=rho, band=[0.5, 1.5],
                                    passed=bool(0.5 <= rho <= 1.5))
    if exponent is not None:
        gates["native_kl_exponent"] = dict(value=exponent, band=[1.7, 2.3],
                                           passed=bool(1.7 <= exponent <= 2.3))
    if len(comp) > 1:
        best_control = max((v["drop"] for k, v in comp.items() if k != "solver"), default=None)
        gates["beats_best_control"] = dict(value=measured, control=best_control,
                                           passed=bool(best_control is not None and measured > best_control))
    gates["ALL_PASS"] = bool(gates) and all(g["passed"] for g in gates.values())

    rec = dict(base_table=base_table, gain=gain, eps=eps, slots=slots,
               predicted_long_gain=predicted, predicted_native_kl=float(krec["pred_native_kl"]),
               measured_long_drop=measured, trust_ratio=rho,
               native_kl_exponent=exponent, base_long_loss=base_long,
               long_length=args.long_length, native_length=args.native_length,
               long_loss=args.long_loss, keep=args.keep,
               alphas=alphas, curves=rows, comparison=comp, gates=gates,
               model_path=args.model, dtype=args.dtype,
               elapsed_seconds=time.monotonic() - started)
    if torch.cuda.is_available():
        rec["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()

    # ---- the screen: the project's own gate before any downstream test
    if args.screen_lengths and args.npy:
        screen = {}
        lengths = [int(x) for x in args.screen_lengths.split(",") if x]
        cand = T.from_eps(d, base_name=base_table, gain=gain)
        for L in lengths:
            if L > arr.shape[0]:
                screen[str(L)] = dict(skipped=f"corpus has {arr.shape[0]} tokens")
                continue
            ids = torch.tensor(arr[:L].astype(np.int64))[None].to(model.device)
            model.install_table(base)
            with torch.no_grad():
                b = float(model.nll_per_token(ids, args.keep).mean())
            model.install_table(cand)
            with torch.no_grad():
                c = float(model.nll_per_token(ids, args.keep).mean())
            screen[str(L)] = dict(base=b, candidate=c, delta=c - b, rel=c / b - 1.0)
            print(json.dumps(dict(screen=L, base=b, candidate=c, rel=c / b - 1.0)), flush=True)
            del ids
        rec["screen"] = screen
        tgt = str(lengths[-1])
        if tgt in screen and "delta" in screen[tgt]:
            rec["screen_target_length"] = lengths[-1]
            rec["screen_gate"] = dict(
                target=tgt, delta=screen[tgt]["delta"], passed=bool(screen[tgt]["delta"] < 0),
                in_window_ok=all(screen[str(L)]["rel"] < 5e-3
                                 for L in lengths if L <= 32768 and "rel" in screen[str(L)]),
                note="target-length NLL must improve; in-window (<=32768) within +0.5%")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)

    print("\n== gates ==")
    for k, v in gates.items():
        print(f"  {k:24s} {v}")
    if "screen" in rec:
        print("\n== screen (tail-token NLL, keep=%d) ==" % args.keep)
        for L, v in rec["screen"].items():
            if "delta" in v:
                print(f"  {L:>7s}  base {v['base']:.4f}  cand {v['candidate']:.4f}  "
                      f"{v['rel']*100:+.3f}%")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
