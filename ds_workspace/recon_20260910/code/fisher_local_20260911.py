#!/usr/bin/env python3
"""Measure the in-window Fisher diagonal F_jj on OLMo-2-0425-1B locally.

Why this exists: tstar.json (the only F_jj measurement) lives on
/root/autodl-tmp/phase1_20260910/tstar/tstar.json on the 5090 box, and that
box is unreachable.  The checkpoint, the prepared corpus and fp32 CPU torch are
all present locally, and F_jj costs one forward per slot, so the measurement is
reproducible here instead of being quoted from a dead host.

    D_N(delta e_j) = 0.5 F_jj delta^2 + O(delta^3),   D_N(0) = 0 exactly,
    F_jj = 2 * D_N(delta e_j) / delta^2

Base table = the DEPLOYED OLMo table (m_incr_beta(1.0, low=14, n=18) == MrProBM),
matching derive_tstar.py, so F_jj is the local curvature of the in-window
distribution around the table the arms actually deploy.

Usage:
  python3 ds_workspace/recon_20260910/code/fisher_local_20260911.py \
      --docs 1 --out /tmp/fisher_olmo_1doc.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.curvature_20260910 import tables as T          # noqa: E402
from experiments.curvature_20260910.model import FrozenRoPE, fisher_diagonal  # noqa: E402

OLMO_MODEL = str(Path.home() /
                 ".cache/huggingface/hub/models--allenai--OLMo-2-0425-1B-Instruct")
NPY_DIR = ROOT / "results/olmo_fast_screen_20260908/prepared_nll_02"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=OLMO_MODEL)
    ap.add_argument("--npy-dir", default=str(NPY_DIR))
    ap.add_argument("--docs", type=int, default=1)
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--keep", type=int, default=512)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--slots", default="all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    slots = (list(range(T.K)) if args.slots == "all"
             else [int(s) for s in args.slots.split(",")])

    t0 = time.monotonic()
    model = FrozenRoPE(args.model, dtype="fp32", device="cpu")
    print(f"[load] {time.monotonic() - t0:.1f}s  native_inv_freq[0]="
          f"{model.native_inv_freq[0]:.6e}", flush=True)

    # the deployed OLMo table, and the checkpoint's own native table for the
    # construction check (the two must share the same inv_freq convention)
    m_bm = np.asarray(T.m_incr_beta(1.0, n=18, low=14), dtype=np.float64)
    bm = dict(m=m_bm, values_float32=T.m_to_inv_freq(m_bm, T.OLMO2_1B["theta"]),
              gain=1.0)
    native = dict(m=np.zeros(T.K), values_float32=model.native_inv_freq.copy(),
                  gain=1.0)
    dev = float(np.abs(np.asarray(bm["values_float32"])
                       / model.native_inv_freq - 1.0).max())
    print(f"[check] deployed-BM inv_freq vs checkpoint native ratio max dev "
          f"= {dev:.3e}  (small => same theta convention)", flush=True)

    Fs, per_doc = [], []
    for d in range(args.docs):
        ids = np.load(Path(args.npy_dir) / f"doc_{d:02d}.npy")[:args.length]
        import torch
        ids = torch.tensor(ids.astype(np.int64)).unsqueeze(0)
        model.install_table(native)                    # D_N is measured FROM native
        base_logp = model.log_probs(ids, args.keep)
        t1 = time.monotonic()
        F = fisher_diagonal(model, ids, args.keep, base_logp, native,
                            args.delta, slots)
        F = np.array([F[j] for j in range(T.K)], dtype=np.float64)
        Fs.append(F)
        per_doc.append(float(time.monotonic() - t1))
        print(f"[doc {d}] {per_doc[-1]:.1f}s  F[0]={F[0]:.4g} F[32]={F[32]:.4g} "
              f"F[63]={F[63]:.4g}", flush=True)

    F = np.mean(Fs, axis=0)
    T_W = 4096.0 * T.native_inv_freq(500_000.0) / (2 * np.pi)
    j = np.arange(T.K)
    rec = dict(source="local CPU fp32 re-measurement, this script",
               model=args.model, npy_dir=str(args.npy_dir), docs=args.docs,
               length=args.length, keep=args.keep, delta=args.delta,
               base_table="native (m=0)", per_doc_seconds=per_doc,
               fisher_diag={str(k): float(v) for k, v in enumerate(F)},
               turns_W={str(k): float(v) for k, v in enumerate(T_W)},
               inv_freq_ratio_maxdev=dev)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(f"[done] wrote {args.out}", flush=True)

    print("\n  j   F_jj        t_j(W)     F normalized   (t_j/t_14)^2  ratio")
    for p in (1.0, 2.0):
        w = (T_W / T_W[14]) ** p
        r = F / F[14] / w
        print(f"  --- F_jj / (t_j/t_14)^{p:g} :"
              f" {np.nanmin(r[F > 0]):.3g} .. {np.nanmax(r[F > 0]):.3g}"
              f"   (constant => that power law holds)")
    for k in range(0, T.K, 4):
        print(f"  {k:2d}  {F[k]:10.4g}  {T_W[k]:9.4f}   {F[k] / F[14]:10.4g}"
              f"   {((T_W / T_W[14]) ** 2)[k]:10.4g}  {F[k] / F[14] / ((T_W / T_W[14]) ** 2)[k]:8.3g}")


if __name__ == "__main__":
    main()
