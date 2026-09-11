#!/usr/bin/env python3
"""Projected gradient descent ON THE MEASURED OBJECTIVE -- no model of the cost.

WHY THIS EXISTS.  Every analytic route this campaign has tried replaces the
measured cost with a model of it, and every one has failed for the same
structural reason: the model is a LOCAL expansion evaluated at a displacement
28x outside its fitting range.  F_jj is measured at the native table with
delta = 0.05; the tables derived from it are evaluated at m_j ~ 1, i.e. a
log-frequency displacement of ln4 = 1.386.

  * value = const            -> a hard step at slot 25  -> RULER 0.1121 (deployed 0.4167)
  * value = coverage marginal -> the deployed BM itself (a=0.993, b=1.061)

So instead of modelling the cost, MEASURE the gradient of the cost directly, at
the table you actually care about, and step along it.

THE OBJECTIVE.  Under the constraint/objective split the campaign has settled on
-- in-window NLL is the CONSTRAINT, long-range retrieval is the OBJECTIVE -- the
thing to minimise is the in-window tail-NLL at a FIXED budget.  So:

    minimise   D(m)   subject to   sum(m) = S0

and the projected gradient is  g~ = g - mean(g)  with  g_j = dD/dm_j  measured by
a central difference at the CURRENT table (not at native).  The step is

    m <- clip(m - eta * g~, 0, 1) ,  then rescale to sum(m) = S0

WHAT MAKES THIS NOT A SCAN.  The direction is a measured gradient, not a grid
axis; there is no table chosen by hand.  If the deployed table is already optimal
under this criterion the measured gradient is flat and the iteration does not
move -- that is a result too, and the honest one to report first.

WHAT IT CANNOT SAY.  It optimises the CONSTRAINT, not the objective.  A table
that is better in-window need not retrieve better, and this campaign has already
measured one case where the two instruments disagree (ctl_C42: tied on RULER,
significantly worse in-window).  So the output is a CANDIDATE for the RULER
panel, never a verdict.
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
    ap.add_argument("--start", default="bm", help="bm | b3_lo14 | <b value>")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--tail", type=int, default=512)
    ap.add_argument("--docs", type=int, default=16)
    ap.add_argument("--band", default="15,32")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--eta", type=float, default=0.35)
    ap.add_argument("--iters", type=int, default=6)
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

    files = sorted(Path(args.nll_dir).glob("doc_*.npy"))[: args.docs]
    if not files:
        print(f"REFUSING: no doc_*.npy under {args.nll_dir}", file=sys.stderr)
        return 2
    data = [np.load(f)[: args.length + 1] for f in files]

    rot = model.model.rotary_emb
    dev = next(model.parameters()).device
    nu0 = OLMO["theta"] ** (-np.arange(64) / 64.0)

    def per_doc_nll(values):
        rot.inv_freq = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(dev)
        rot.attention_scaling = float(GAIN)
        out = []
        with torch.inference_mode():
            for arr in data:
                ids = torch.tensor(arr[: args.length].astype(np.int64), device=dev)[None]
                tgt = torch.tensor(arr[args.length - args.tail + 1:].astype(np.int64),
                                   device=dev)
                lg = model(ids, use_cache=False, logits_to_keep=args.tail).logits[0].float()
                out.append(float(torch.nn.functional.cross_entropy(lg, tgt)))
                del ids, tgt, lg
        return np.array(out)

    def values_of(m):
        return nu0 * np.power(4.0, -np.asarray(m, dtype=float))

    def nll_of(m):
        d = per_doc_nll(values_of(m))
        return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), d

    # ---- the starting table -------------------------------------------------
    from experiments.curvature_20260910.tables import m_incr_beta
    lo, n = OLMO["low"], OLMO["n"]
    if args.start == "bm":
        m0 = np.asarray(m_incr_beta(1.0, n=n, low=lo), dtype=float)
        name0 = "deployed_BM"
    elif args.start == "b3_lo14":
        m0 = np.asarray(m_incr_beta(3.0, n=n, low=lo), dtype=float)
        name0 = "b3_lo14"
    else:
        m0 = np.asarray(m_incr_beta(float(args.start), n=n, low=lo), dtype=float)
        name0 = f"beta_b{args.start}"
    S0 = float(m0.sum())

    blo, bhi = (int(x) for x in args.band.split(","))
    band = list(range(blo, bhi + 1))

    t0 = time.monotonic()
    base_nll, base_se, base_doc = nll_of(m0)
    print(json.dumps({"phase": "START", "arm": name0, "sum_m": S0,
                      "nll": base_nll, "se": base_se,
                      "seconds": round(time.monotonic() - t0, 1)}), flush=True)

    hist = [dict(iter=0, arm=name0, nll=base_nll, se=base_se,
                 m=[float(x) for x in m0], sum_m=S0)]

    m = m0.copy()
    for it in range(1, args.iters + 1):
        t0 = time.monotonic()
        grad = {}
        for j in band:
            d = args.delta
            mp = m.copy(); mp[j] = min(1.0, m[j] + d)
            mm = m.copy(); mm[j] = max(0.0, m[j] - d)
            # clip at the plateau: a slot already at 1 can only move down
            hi = mp[j] - m[j]
            lo_ = m[j] - mm[j]
            dp = per_doc_nll(values_of(mp))
            dm = per_doc_nll(values_of(mm))
            # central difference in the actual step sizes used
            g = ((dp.mean() - base_doc.mean()) / hi if hi > 0 else 0.0) \
                - ((dm.mean() - base_doc.mean()) / lo_ if lo_ > 0 else 0.0)
            grad[j] = float(g)
        g = np.zeros(64)
        for j in band:
            g[j] = grad[j]
        gt = g[band]
        gproj = gt - gt.mean()                      # project onto sum(m) = S0
        # scale the step so the largest move is eta * delta
        scale = np.abs(gproj).max()
        step = np.zeros(64)
        if scale > 0:
            step[band] = -args.eta * args.delta * (gproj / scale)
        m_new = np.clip(m + step, 0.0, 1.0)
        # rescale the band to restore the budget exactly
        cur = m_new[band].sum()
        want = m[band].sum()
        if cur > 0:
            m_new[band] = np.clip(m_new[band] * (want / cur), 0.0, 1.0)
        nll, se, doc = nll_of(m_new)
        print(json.dumps({"iter": it, "nll": round(nll, 6), "se": round(se, 6),
                          "sum_m": round(float(m_new.sum()), 4),
                          "d_vs_base": round(nll - base_nll, 6),
                          "max_move": round(float(np.abs(m_new - m).max()), 5),
                          "seconds": round(time.monotonic() - t0, 1)}), flush=True)
        hist.append(dict(iter=it, nll=nll, se=se, sum_m=float(m_new.sum()),
                         m=[float(x) for x in m_new], grad=grad,
                         d_vs_base=nll - base_nll))
        # accept if it improves beyond one paired SE, else stop
        if nll < base_nll - se:
            m = m_new
            base_nll, base_se, base_doc = nll, se, doc
        else:
            print(json.dumps({"stop": "no improvement beyond one SE",
                              "iter": it}), flush=True)
            break

    out = dict(start=name0, band=[blo, bhi], delta=args.delta, eta=args.eta,
               history=hist,
               scope=("Projected gradient descent on the MEASURED in-window "
                      "tail-NLL at fixed sum(m). The gradient is a central "
                      "difference at the CURRENT table, so no quadratic model "
                      "enters. This optimises the CONSTRAINT; the output is a "
                      "candidate for the RULER panel, never a verdict."))
    (root / "gradient.json").write_text(json.dumps(out, indent=1))

    print("\n=== projected gradient descent on the measured cost ===")
    print("  start %s  sum(m)=%.3f  nll=%.6f" % (name0, S0, hist[0]["nll"]))
    for h in hist[1:]:
        print("  iter %d   nll=%.6f (se %.6f)   d=%+0.6f   sum(m)=%.3f"
              % (h["iter"], h["nll"], h["se"], h["d_vs_base"], h["sum_m"]))
    print("\n=== the measured gradient at the starting table ===")
    g = hist[1]["grad"] if len(hist) > 1 else None
    if g:
        print("  %5s %8s %14s" % ("slot", "m_start", "dD/dm"))
        for j in band:
            print("  %5d %8.3f %14.6f" % (j, m0[j], g[str(j)]))
        gv = np.array([g[str(j)] for j in band])
        print("\n  grad: mean %+.6f  std %.6f  min %+.6f  max %+.6f"
              % (gv.mean(), gv.std(), gv.min(), gv.max()))
        print("  A FLAT profile means the starting table is already optimal "
              "under this criterion.\n  A TILT is the direction of improvement, "
              "and the iterations above walk it.")
    print("\n=== final table (m on the band) ===")
    mf = np.array(hist[-1]["m"])
    print("  slot  m  :  " + " ".join("%d:%.3f" % (j, mf[j]) for j in band))
    print("  sum(m) = %.4f   m_63 = %.4f" % (mf.sum(), mf[63]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
