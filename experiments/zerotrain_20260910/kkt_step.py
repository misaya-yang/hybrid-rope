#!/usr/bin/env python3
"""Turn the measured KKT gradient into tables, and score them on the cheap instrument.

WHY THIS EXISTS.  `kkt_gradient.py` measures, at a named table and by central
differences on real forwards, two gradients:

    gD_j = d(in-window tail-NLL)/dm_j        the constraint side
    gR_j = d(long-range tail-NLL)/dm_j       the objective side

Nothing in that measurement is a model: no Fisher quadratic, no coverage
functional, no saturating benefit curve.  This file USES the result -- it takes
the measured gradient and walks it, producing candidate tables that are then
scored on the continuous long-range instrument (about 25 s per arm), and written
out so the winners can go to the RULER panel.

WHAT IT BUILDS.

  (1) THE OBJECTIVE STEP.  If the in-window constraint is slack -- and the first
      measurement says it is, gD_15 = -0.060, i.e. compressing slot 15 IMPROVES
      the in-window NLL -- then the problem is just "minimise R(m)", and the
      measured gradient of R is the direction.  Step m <- m - eta * gR~ with gR~
      projected onto sum(m) = const, so the budget is held and only the SHAPE
      moves.  That is the cleanest available reading of "where should the
      compression sit".

  (2) THE KKT STEP.  Under the constrained reading, the exchange rate rho_j =
      gR_j/gD_j must be flat at an optimum; a tilt is the direction.  Step along
      the tilt of rho rather than the raw gR.

  (3) THE CONSTRAINT-ONLY STEP.  A control: walk gD alone, which is what the
      abandoned analytic routes were trying to model.

  All three keep sum(m) fixed, clip to [0,1], and are scored with the identical
  paired instrument, so the comparison between them is exact.

A POSITIVE RESULT HERE IS A CANDIDATE, NOT A VERDICT.  The instrument is
long-range NLL, and this campaign has one measured case (ctl_C42) where the
instrument and the RULER panel disagree in significance.  Every table this file
emits is a proposal for the panel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OLMO = dict(theta=500_000.0, window=4096, head_dim=128, K=64, low=14, n=18)
GAIN = 1.138629436111989


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kkt", required=True, help="the kkt.json to read")
    ap.add_argument("--root", required=True, help="where to write the scored arms")
    ap.add_argument("--model", required=True)
    ap.add_argument("--nll-dir", required=True)
    ap.add_argument("--far", type=int, default=16385)
    ap.add_argument("--tail", type=int, default=512)
    ap.add_argument("--docs", type=int, default=16)
    ap.add_argument("--etas", default="0.05,0.15,0.4")
    ap.add_argument("--extra", default=None,
                    help="comma list of eta values to apply to the rho tilt too")
    args = ap.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    kk = json.loads(Path(args.kkt).read_text())
    band = list(range(kk["band"][0], kk["band"][1] + 1))
    m0 = np.array(kk["baseline"]["m"], dtype=float)
    S0 = float(m0.sum())
    gD = np.array([next(r["gW"] for r in kk["rows"] if r["slot"] == j)
                   for j in band], dtype=float)
    gR = np.array([next(r["gF"] for r in kk["rows"] if r["slot"] == j)
                   for j in band], dtype=float)

    def proj(g):
        return g - g.mean()

    def snapped(g, eta):
        step = np.zeros(64)
        sc = np.abs(proj(g)).max()
        if sc > 0:
            step[band] = -eta * S0 * proj(g) / sc / len(band)
        m = np.clip(m0 + step, 0.0, 1.0)
        cur = m[band].sum()
        if cur > 0:
            m[band] = np.clip(m[band] * (S0 - (m.sum() - cur)) / cur, 0.0, 1.0)
        m[band] *= (S0 - (m.sum() - m[band].sum())) / max(m[band].sum(), 1e-12)
        m[band] = np.clip(m[band], 0.0, 1.0)
        return m

    # the rho tilt, with zeros where gD is too small to define a ratio
    rho = np.where(np.abs(gD) > 1e-9, gR / np.where(np.abs(gD) > 1e-9, gD, 1), 0.0)

    cands = [("start", m0)]
    for e in [float(x) for x in args.etas.split(",")]:
        cands.append((f"objstep_e{e:g}", snapped(gR, e)))
        cands.append((f"dstep_e{e:g}", snapped(gD, e)))
    for e in [float(x) for x in (args.extra or "").split(",") if x.strip()]:
        cands.append((f"rhostep_e{e:g}", snapped(rho, e)))

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

    docs = []
    for f in sorted(Path(args.nll_dir).glob("doc_*.npy"))[: args.docs]:
        a = np.load(f)
        if len(a) >= args.far:
            docs.append(a)
    if not docs:
        print(f"REFUSING: no document of {args.far} tokens under {args.nll_dir}",
              file=sys.stderr)
        return 2

    rot = model.model.rotary_emb
    dev = next(model.parameters()).device
    nu0 = OLMO["theta"] ** (-np.arange(64) / 64.0)

    def per_doc_nll(values):
        rot.inv_freq = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(dev)
        rot.attention_scaling = float(GAIN)
        out = []
        with torch.inference_mode():
            for arr in docs:
                ids = torch.tensor(arr[: args.far - 1].astype(np.int64),
                                   device=dev)[None]
                tgt = torch.tensor(
                    arr[args.far - 1 - args.tail: args.far - 1].astype(np.int64),
                    device=dev)
                lg = model(ids, use_cache=False,
                           logits_to_keep=args.tail).logits[0].float()
                out.append(float(torch.nn.functional.cross_entropy(lg, tgt)))
                del ids, tgt, lg
        return np.array(out)

    ref = None
    rows = []
    for name, m in cands:
        v = nu0 * np.power(4.0, -np.asarray(m, dtype=float))
        d = per_doc_nll(v)
        if ref is None:
            ref = d
        dd = d - ref
        rec = dict(arm=name, sum_m=float(np.asarray(m).sum()),
                   nll=float(d.mean()), d_vs_start=float(dd.mean()),
                   se=float(dd.std(ddof=1) / np.sqrt(len(dd))),
                   m=[float(x) for x in m])
        rows.append(rec)
        print(json.dumps({k: (round(v_, 6) if isinstance(v_, float) else v_)
                          for k, v_ in rec.items() if k != "m"}), flush=True)

    (root / "rows.json").write_text(json.dumps(dict(
        source=args.kkt, band=kk["band"], start_sum_m=S0,
        gradient_gD={str(j): float(g) for j, g in zip(band, gD)},
        gradient_gR={str(j): float(g) for j, g in zip(band, gR)},
        rho={str(j): float(g) for j, g in zip(band, rho)},
        rows=rows,
        scope=("Candidate tables built from the MEASURED KKT gradient in "
               "kkt.json, scored on the continuous long-range instrument with "
               "the start table measured in the same process so every delta is "
               "paired. Candidates, not verdicts -- the RULER panel is the "
               "arbiter.")), indent=1))

    print("\n=== candidates from the measured gradient, scored on the "
          "continuous instrument ===")
    print("  %-16s %8s %12s %11s %9s" % ("arm", "sum_m", "nll", "d_vs_start", "se"))
    for r in sorted(rows, key=lambda z: z["d_vs_start"]):
        print("  %-16s %8.3f %12.6f %+11.6f %9.6f"
              % (r["arm"], r["sum_m"], r["nll"], r["d_vs_start"], r["se"]))
    best = min(rows, key=lambda z: z["d_vs_start"])
    print("\n  best: %s   d_vs_start %+.6f (se %.6f)  t = %+.2f"
          % (best["arm"], best["d_vs_start"], best["se"],
             best["d_vs_start"] / best["se"] if best["se"] > 0 else 0.0))
    print("\n  band table of the best:")
    mf = np.array(best["m"])
    print("  " + " ".join("%d:%.3f" % (j, mf[j]) for j in band))
    return 0


if __name__ == "__main__":
    sys.exit(main())
