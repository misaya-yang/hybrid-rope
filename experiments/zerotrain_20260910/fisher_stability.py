#!/usr/bin/env python3
"""Is the measured Fisher diagonal a real curvature, or a fit artifact?

WHY THIS EXISTS.  `derive_tstar.py` measures F_jj -- the diagonal of the
output-Fisher of the in-window loss w.r.t. each slot's log-frequency -- and then
solves for the band edge t* at which a hard step's damage, in that measured
metric, equals the deployed BM's.  It returned hi* = 25, and the step table at
hi* scored RULER 0.1121 against the deployed table's 0.4167.  The route failed.

The printed F_jj has isolated spikes: slot 24 = 7842, slot 35 = 1.21e4, slot 12
= 7293, while their immediate neighbours sit at 400-1100.  A 20x single-slot
spike in a quantity estimated as 2*KL/delta^2 from 8 documents is exactly what a
noise-dominated estimator looks like, and if it is noise then so is every t*
derived from it -- which would be a complete, cheap explanation of the failure,
and would retire the route instead of leaving it half-open.

WHAT IT DOES.  One model load, three probes:

  (1) DELTA.  F_jj is defined by D_N(delta e_j) = 0.5 F_jj delta^2 + O(delta^3).
      Measure at delta = 0.02 / 0.05 / 0.10.  Real curvature gives the same F_jj
      at every delta; a third-order contamination or a noise floor does not.

  (2) DOCS.  Re-measure at delta = 0.05 on a DISJOINT set of documents.  A real
      per-slot curvature is a property of the checkpoint and the slot, not of
      which 8 documents were sampled.

  (3) SPIKES.  Report the spiky slots explicitly and whether each one survives
      probes (1) and (2).  A spike that is gone at another delta or on other
      documents was an estimator artifact.

The decisive readout is the derived edge: hi* is recomputed under every probe.
If hi* moves, the analytic program's output was an artifact of its own
measurement, and that is the finding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from derive_tstar import GAIN, LN4, OLMO, measure_fisher, native_nu, turns  # noqa: E402


def derive_hi(Fv, nu0, m_bm):
    """The same solve as derive_tstar, as a function of the measured F."""
    Fv = np.asarray(Fv, dtype=np.float64)
    suffix = np.cumsum((Fv * LN4 * LN4 / 2.0)[::-1])[::-1]
    dam_bm = float(np.sum(0.5 * Fv * (LN4 ** 2) * m_bm ** 2))
    feasible = [j for j in range(64) if suffix[j] <= dam_bm]
    return (min(feasible) if feasible else None), dam_bm, suffix


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nll-dir", required=True)
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--docs", type=int, default=8)
    ap.add_argument("--deltas", default="0.02,0.05,0.10")
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

    files = sorted(Path(args.nll_dir).glob("doc_*.npy"))
    if len(files) < 2 * args.docs:
        print(f"REFUSING: need {2 * args.docs} documents for the disjoint split, "
              f"found {len(files)}", file=sys.stderr)
        return 2
    docsA = [np.load(f) for f in files[: args.docs]]
    docsB = [np.load(f) for f in files[args.docs: 2 * args.docs]]

    from experiments.curvature_20260910.tables import m_incr_beta
    nu0 = native_nu()
    m_bm = np.asarray(m_incr_beta(1.0, n=OLMO["n"], low=OLMO["low"]),
                      dtype=np.float64)
    t = turns(nu0)

    probes = {}
    for d in [float(x) for x in args.deltas.split(",")]:
        key = f"delta{d:g}_docsA"
        probes[key] = (measure_fisher(model, docsA, args.length, nu0, d,
                                      list(range(64))), d)
        print(json.dumps({"probe": key, "done": True}), flush=True)
    key = "delta0.05_docsB"
    probes[key] = (measure_fisher(model, docsB, args.length, nu0, 0.05,
                                  list(range(64))), 0.05)
    print(json.dumps({"probe": key, "done": True}), flush=True)

    vecs = {k: np.array([v[0][j] for j in range(64)]) for k, v in probes.items()}
    out = dict(probes={k: v.tolist() for k, v in vecs.items()},
               deltas={k: v[1] for k, v in probes.items()})

    hi = {}
    for k, v in vecs.items():
        h, dam, _ = derive_hi(v, nu0, m_bm)
        hi[k] = dict(hi=h, t_star=(float(t[h]) if h is not None else None),
                     damage_bm=dam)
    out["derived"] = hi

    keys = list(vecs)
    corr = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = vecs[keys[i]], vecs[keys[j]]
            corr[f"{keys[i]} vs {keys[j]}"] = float(np.corrcoef(a, b)[0, 1])
    out["correlation"] = corr

    # the spiky slots: those whose F exceeds 3x the median of their neighbours
    med = np.array([np.median(np.delete(vecs["delta0.05_docsA"], k))
                    for k in range(64)])
    ratio = vecs["delta0.05_docsA"] / np.maximum(med, 1e-9)
    top = np.argsort(-ratio)[:8]
    spikes = {}
    for j in sorted(int(x) for x in top):
        spikes[str(j)] = {k: float(vecs[k][j]) for k in keys}
        spikes[str(j)]["ratio_to_median"] = float(ratio[j])
    out["spike_slots"] = spikes

    (root / "stability.json").write_text(json.dumps(out, indent=1))

    print("\n=== F_jj under every probe ===")
    print("%5s %11s %11s %11s %11s" % ("slot", *keys))
    for j in range(64):
        print("%5d %11.1f %11.1f %11.1f %11.1f"
              % (j, *(vecs[k][j] for k in keys)))
    print("\n=== pairwise correlation of the measured F vectors ===")
    for k, v in corr.items():
        print("   %-38s r = %+.4f" % (k, v))
    print("\n=== derived band edge under every probe ===")
    for k in keys:
        d = hi[k]
        print("   %-22s hi* = %-5s  t* = %s"
              % (k, d["hi"], ("%.3f" % d["t_star"]) if d["t_star"] else "-"))
    print("\n=== the spiky slots, and whether they survive ===")
    for j, rec in sorted(spikes.items(), key=lambda z: int(z[0])):
        vals = [rec[k] for k in keys]
        spread = max(vals) / max(min(vals), 1e-9)
        print("   slot %3s  ratio-to-median %6.2f   %s   spread %.2fx %s"
              % (j, rec["ratio_to_median"],
                 " ".join("%9.1f" % v for v in vals), spread,
                 "STABLE" if spread < 1.6 else "**UNSTABLE**"))
    print("\n  If the F vector at one delta does not correlate with itself at "
          "another,\n  or the derived hi* moves between probes, the analytic "
          "route was reading\n  its own estimator's noise. That retires the "
          "route; it does not leave it open.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
