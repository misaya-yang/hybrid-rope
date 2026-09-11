#!/usr/bin/env python3
"""THE KKT CONDITION, MEASURED DIRECTLY ON BOTH SIDES.

THE PROBLEM, AS STATED.  Minimise the in-window loss (no catastrophic forgetting)
while maximising the extrapolation range.  Written as a constrained problem with
the in-window loss as the constraint and the long-range loss as the objective,

    minimise   R(m)     (long-range tail-NLL at 4x the window)
    s.t.       D(m) <= D0   (in-window tail-NLL at the native window)

the KKT stationarity condition for the active constraint is

    dR/dm_j  =  lambda * dD/dm_j      for every slot j in the band

i.e. THE EXCHANGE RATE  rho_j = (dR/dm_j) / (dD/dm_j)  IS CONSTANT ACROSS SLOTS.

WHAT THIS FILE DOES.  It measures both gradients by central differences at a
table you name, and reports the exchange-rate profile rho_j.

  * rho_j FLAT over the band  ->  that table satisfies the KKT condition, and the
    campaign's search is over: the incumbent is the optimum of the stated problem.
  * rho_j TILTED             ->  the tilt IS the direction of improvement, and the
    gradient is measured, not guessed.  Slots with a high rho are expensive in
    long-range terms per unit of in-window damage; move compression OFF them.

WHY THIS REPLACES THE MODELLED ROUTES.  Every table the campaign has derived came
from a MODEL of one side or the other, and each model was a local expansion used
28x outside its fitting range:
  * F_jj measured at the native table with delta=0.05, used to price m_j ~ 1
    (a log-frequency displacement of ln4 = 1.386).
  * the long-range side was modelled as a saturating coverage min(1, 4^(m-1)).
Between them they produced a hard step at slot 25 (RULER 0.1121 vs 0.4167) and a
fixed point that returns the deployed table (a=0.993, b=1.061).
Neither side here is modelled.  Both gradients are finite differences of
quantities that are actually measured, at the table actually under test.

WHAT IT CANNOT SAY.  It tests stationarity of the constrained problem as posed.
If that problem is not the right one -- e.g. if retrieval is not monotone in
long-range NLL, which this campaign has one measured instance of -- the test is
silent about it. The RULER panel remains the only arbiter of the objective.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OLMO = dict(theta=500_000.0, window=4096, head_dim=128, K=64, low=14, n=18)
GAIN = 1.138629436111989


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nll-dir", required=True)
    ap.add_argument("--start", default="bm")
    ap.add_argument("--window", type=int, default=4096, help="the in-window length")
    ap.add_argument("--far", type=int, default=16384, help="the long-range length")
    ap.add_argument("--tail", type=int, default=512)
    ap.add_argument("--docs", type=int, default=16)
    ap.add_argument("--band", default="15,32")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--shuffle", action="store_true",
                    help="control: perturb slots in a shuffled order to catch "
                         "drift/order artifacts")
    args = ap.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda"}, attn_implementation="sdpa").eval()
    model.requires_grad_(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)

    need = max(args.window, args.far)
    files = sorted(Path(args.nll_dir).glob("doc_*.npy"))
    data = []
    for f in files:
        a = np.load(f)
        if len(a) >= need + 1:
            data.append(a)
        if len(data) >= args.docs:
            break
    if not data:
        print(f"REFUSING: no document long enough for {need} tokens under "
              f"{args.nll_dir}", file=sys.stderr)
        return 2

    rot = model.model.rotary_emb
    dev = next(model.parameters()).device
    nu0 = OLMO["theta"] ** (-np.arange(64) / 64.0)
    LN4 = float(np.log(4.0))

    def per_doc_nll(values, L):
        rot.inv_freq = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(dev)
        rot.attention_scaling = float(GAIN)
        out = []
        with torch.inference_mode():
            for arr in data:
                ids = torch.tensor(arr[:L].astype(np.int64), device=dev)[None]
                tgt = torch.tensor(arr[L - args.tail + 1: L + 1].astype(np.int64),
                                   device=dev)
                lg = model(ids, use_cache=False,
                           logits_to_keep=args.tail).logits[0].float()
                out.append(float(torch.nn.functional.cross_entropy(lg, tgt)))
                del ids, tgt, lg
        return np.array(out)

    def values_of(m):
        return nu0 * np.power(4.0, -np.asarray(m, dtype=float))

    from experiments.curvature_20260910.tables import m_incr_beta
    lo, n = OLMO["low"], OLMO["n"]
    if args.start == "bm":
        m0 = np.asarray(m_incr_beta(1.0, n=n, low=lo), dtype=float)
    elif args.start == "b3_lo14":
        m0 = np.asarray(m_incr_beta(3.0, n=n, low=lo), dtype=float)
    elif args.start == "c42v24":
        from experiments.curvature_20260910.tables import m_C42V24
        m0 = np.asarray(m_C42V24(), dtype=float)
    else:
        m0 = np.asarray(m_incr_beta(float(args.start), n=n, low=lo), dtype=float)

    blo, bhi = (int(x) for x in args.band.split(","))
    band = list(range(blo, bhi + 1))

    t0 = time.monotonic()
    win0 = per_doc_nll(values_of(m0), args.window)
    far0 = per_doc_nll(values_of(m0), args.far)
    print(json.dumps({"phase": "BASELINE", "start": args.start,
                      "sum_m": float(m0.sum()),
                      "in_window_nll": float(win0.mean()),
                      "long_range_nll": float(far0.mean()),
                      "docs": len(data),
                      "seconds": round(time.monotonic() - t0, 1)}), flush=True)

    order = band[:]
    if args.shuffle:
        rng = np.random.default_rng(20260911)
        rng.shuffle(order)

    rows = []
    for j in order:
        t1 = time.monotonic()
        d = args.delta
        mp = m0.copy(); mp[j] = min(1.0, m0[j] + d)
        mm = m0.copy(); mm[j] = max(0.0, m0[j] - d)
        hp = mp[j] - m0[j]
        hm = m0[j] - mm[j]

        dWp = per_doc_nll(values_of(mp), args.window) - win0
        dWm = per_doc_nll(values_of(mm), args.window) - win0
        dFp = per_doc_nll(values_of(mp), args.far) - far0
        dFm = per_doc_nll(values_of(mm), args.far) - far0

        # CENTRAL difference, not the sum of the two one-sided quotients.
        # (f(m+d) - f(m))/d + (f(m) - f(m-d))/d is TWICE the central difference;
        # the ratio rho is unaffected by that factor but the reported gradient
        # magnitudes are, and they are what kkt_step.py steps along.
        gW = (((dWp.mean() - dWm.mean()) / (hp + hm))
              if (hp + hm) > 0 else 0.0)
        gF = (((dFp.mean() - dFm.mean()) / (hp + hm))
              if (hp + hm) > 0 else 0.0)
        # paired SE of the long-range change, for scale
        seF = float(np.concatenate([dFp, dFm]).std(ddof=1) / np.sqrt(2 * len(data)))
        rows.append(dict(slot=j, m=float(m0[j]), gW=float(gW), gF=float(gF),
                         se_gF=float(seF),
                         rho=(float(gF / gW) if abs(gW) > 1e-12 else None)))
        print(json.dumps({"slot": j, "m": round(float(m0[j]), 4),
                          "gW": round(float(gW), 6), "gF": round(float(gF), 6),
                          "rho": (round(float(gF / gW), 4)
                                  if abs(gW) > 1e-12 else None),
                          "seconds": round(time.monotonic() - t1, 1)}), flush=True)

    out = dict(start=args.start, window=args.window, far=args.far,
               band=[blo, bhi], delta=args.delta, shuffled=bool(args.shuffle),
               baseline=dict(in_window=float(win0.mean()),
                             long_range=float(far0.mean()),
                             sum_m=float(m0.sum()),
                             m=[float(x) for x in m0]),
               rows=rows,
               scope=("KKT stationarity for  minimise R(m) s.t. D(m)<=D0 : the "
                      "exchange rate rho_j = (dR/dm_j)/(dD/dm_j) must be flat "
                      "across the band. Both gradients are measured central "
                      "differences at the named table -- no cost model, no "
                      "quadratic expansion, no coverage functional. A TILT in "
                      "rho is the measured direction of improvement."))
    (root / "kkt.json").write_text(json.dumps(out, indent=1))

    print("\n=== exchange rate rho_j = (dR/dm_j) / (dD/dm_j) ===")
    print("  %5s %8s %13s %13s %10s" % ("slot", "m", "dD/dm", "dR/dm", "rho"))
    for r in sorted(rows, key=lambda z: z["slot"]):
        print("  %5d %8.3f %13.6f %13.6f %10s"
              % (r["slot"], r["m"], r["gW"], r["gF"],
                 ("%.4f" % r["rho"]) if r["rho"] is not None else "-"))
    rr = np.array([r["rho"] for r in rows if r["rho"] is not None])
    if len(rr):
        print("\n  rho: mean %+.4f  std %.4f  min %+.4f  max %+.4f  (n=%d)"
              % (rr.mean(), rr.std(ddof=1), rr.min(), rr.max(), len(rr)))
        cv = rr.std(ddof=1) / abs(rr.mean()) if abs(rr.mean()) > 0 else float("inf")
        print("  coefficient of variation = %.3f" % cv)
        print("  FLAT (small CV)   -> this table satisfies the KKT condition.")
        print("  TILTED (large CV) -> the tilt is the direction of improvement;")
        print("  move compression OFF the high-rho slots and ONTO the low-rho ones.")
    print("\n  band table:  " + " ".join("%d:%.3f" % (r["slot"], r["m"])
                                        for r in sorted(rows, key=lambda z: z["slot"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
