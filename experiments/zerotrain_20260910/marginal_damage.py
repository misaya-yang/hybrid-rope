#!/usr/bin/env python3
"""The TRUE marginal cost of compressing one slot -- and whether the quadratic model survives.

WHY THIS EXISTS.  The analytic program prices a compression m_j with a LOCAL
quantity: D_j(m) ~= 0.5 * F_jj * (ln4)^2 * m^2, where F_jj is the in-window
output-Fisher diagonal measured at the NATIVE table with a step of delta = 0.05.

Every table the program has derived is then evaluated at m_j near 1.  m_j = 1 is
a factor-4 change in the slot's frequency, i.e. a log-frequency displacement of
ln4 = 1.386 -- 28x the delta the curvature was measured at.  A quadratic model
extrapolated 28x past its fitting point is not a model, and that single fact
would explain the two failures the program has actually had:

  * derive_tstar (value = const) returned a hard STEP at slot 25.  A step puts the
    largest possible displacement on the largest number of slots -- exactly where
    the quadratic is worst.  It scored RULER 0.1121 against the deployed 0.4167.
  * the value = coverage-marginal fixed point returns the deployed BM almost
    exactly (fitted exponents a=0.993, b=1.061).  That is what a smooth local
    approximation returning the smooth incumbent looks like.

WHAT IT MEASURES.  For a slot j, set m_j alone (all other slots native) and
measure the in-window tail-NLL.  D_j(0) = 0 by construction, so the curve is
directly readable.  Two things follow:

  (1) THE QUADRATIC TEST.  Compare D_j(m) against 0.5 F_jj (ln4)^2 m^2 at the
      same delta.  If the ratio drifts with m, the Fisher is a curvature and not
      a cost, and every table the program derived from it is retired -- cleanly,
      with a measured reason rather than a shrug.

  (2) THE EQUIMARGINAL PROFILE.  For a separable cost with uniform value, the
      optimum under a budget sets D'_j(m_j) EQUAL across slots (water-filling).
      So evaluate the measured D'_j at the DEPLOYED table's own m_j.  If the
      deployed table is the equimarginal optimum, that profile is flat.  If it is
      tilted, the tilt is the direction to move -- and this is a measurement of
      the objective's gradient, not a scan over tables.

THE MEASUREMENT IS PAIRED AND IN-PROCESS.  One model load; every arm's logits are
compared against the native reference measured in the same process with the same
batch policy, so no archived offset enters.
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
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--tail", type=int, default=512)
    ap.add_argument("--docs", type=int, default=16)
    ap.add_argument("--slots", default="14,16,18,20,22,24,26,28,30,32,34,36")
    ap.add_argument("--grid", default="0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--fine-slot", type=int, default=24)
    ap.add_argument("--fine-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
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

    t0 = time.monotonic()
    base = per_doc_nll(nu0)
    print(json.dumps({"phase": "BASELINE", "nll": float(base.mean()),
                      "docs": len(data), "seconds": round(time.monotonic() - t0, 1)}),
          flush=True)

    def damage(slot, mval):
        v = nu0.copy()
        v[slot] = v[slot] * (4.0 ** (-mval))
        d = per_doc_nll(v) - base
        return d

    # ---- (1) the fine sweep on one slot: is the quadratic real? --------------
    from derive_tstar import measure_fisher  # reuse the measured F_jj

    fine = {}
    fslot = args.fine_slot
    for mval in [float(x) for x in args.fine_grid.split(",")]:
        d = damage(fslot, mval)
        se = d.std(ddof=1) / np.sqrt(len(d))
        fine[mval] = dict(delta=float(d.mean()), se=float(se), per_doc=d.tolist())
        print(json.dumps({"fine": fslot, "m": mval, "delta": round(float(d.mean()), 6),
                          "se": round(float(se), 6)}), flush=True)

    # ---- (2) the coarse grid over the band: the equimarginal profile ---------
    slots = [int(s) for s in args.slots.split(",")]
    grid = [float(x) for x in args.grid.split(",")]
    curves = {}
    for j in slots:
        curves[str(j)] = {}
        for mval in grid:
            d = damage(j, mval)
            curves[str(j)][str(mval)] = dict(
                delta=float(d.mean()),
                se=float(d.std(ddof=1) / np.sqrt(len(d))))
            print(json.dumps({"slot": j, "m": mval,
                              "delta": round(float(d.mean()), 6)}), flush=True)

    # ---- the deployed table's own m, read from the archived builder ----------
    from experiments.curvature_20260910.tables import m_incr_beta
    m_bm = np.asarray(m_incr_beta(1.0, n=OLMO["n"], low=OLMO["low"]), dtype=float)

    # slope of D_j at m_j = the deployed value, by local finite difference
    equi = {}
    for j in slots:
        c = curves[str(j)]
        mb = float(m_bm[j])
        gs = sorted(float(k) for k in c)
        lo = max([g for g in gs if g <= mb], default=gs[0])
        hi = min([g for g in gs if g >= mb], default=gs[-1])
        if hi == lo:
            slope = float("nan")
        else:
            slope = (c[str(hi)]["delta"] - c[str(lo)]["delta"]) / (hi - lo)
        equi[str(j)] = dict(m_bm=mb, slope=slope, curvature=(mb > 0))

    out = dict(length=args.length, docs=len(data), baseline=float(base.mean()),
               fine_slot=fslot, fine=fine, curves=curves, equimarginal=equi,
               scope=("In-window tail-NLL cost of compressing ONE slot, paired "
                      "against the native reference measured in this process. "
                      "D_j(0)=0 by construction. The fine sweep tests whether "
                      "0.5 F_jj (ln4)^2 m^2 describes D_j(m) out to m=1; the "
                      "coarse grid gives D'_j at the DEPLOYED table's own m_j, "
                      "which is the objective's gradient in table space."))
    (root / "marginal.json").write_text(json.dumps(out, indent=1))

    print("\n=== (1) is the quadratic real?  slot %d ===" % fslot)
    from derive_tstar import native_nu
    print("  %6s %12s %10s   %s" % ("m", "measured D", "SE", "0.5 F(ln4)^2 m^2"))
    for mval in sorted(fine):
        meas = fine[mval]["delta"]
        print("  %6.2f %12.6f %10.6f" % (mval, meas, fine[mval]["se"]))
    print("\n  Compare against F_jj from the same checkpoint (see tstar.json);")
    print("  the ratio measured/quadratic is printed below once F is loaded.")
    try:
        Fj = json.load(open(Path(args.root).parent / "tstar" / "tstar.json"))["F_diag"][str(fslot)]
        import math
        print("  %6s %12s %12s %8s" % ("m", "measured", "quadratic", "ratio"))
        for mval in sorted(fine):
            q = 0.5 * Fj * (math.log(4.0) ** 2) * mval ** 2
            meas = fine[mval]["delta"]
            r = (meas / q) if q > 0 else float("nan")
            print("  %6.2f %12.6f %12.6f %8.3f" % (mval, meas, q, r))
        print("  A ratio that DRIFTS with m means the Fisher is a curvature, not "
              "a cost.")
    except Exception as e:
        print("  (could not load F_jj for the ratio: %s)" % e)

    print("\n=== (2) equimarginal profile: D'_j at the deployed table's own m_j ===")
    print("  %5s %8s %14s" % ("slot", "m_BM", "dD/dm at m_BM"))
    for j in slots:
        e = equi[str(j)]
        print("  %5d %8.3f %14.6f" % (j, e["m_bm"], e["slope"]))
    print("\n  A FLAT profile means the deployed table is the equimarginal optimum")
    print("  under uniform value. A TILT is the direction to move -- it is a")
    print("  measurement of the objective's gradient, not a scan over tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
