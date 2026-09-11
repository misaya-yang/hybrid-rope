#!/usr/bin/env python3
"""The mid-band operating variable: derive, don't scan.

Run:  python3 ds_workspace/recon_20260910/code/midband_n.py    (cwd = repo root)

The 12 OLMo-2-0425-1B / 16K-RULER points.  Every table is rebuilt from the
repo's own constructions and every S is asserted against ds_workspace/
recon_20260910/LEDGER_20260911.md before any statistic is computed, so a
mismatch is a build error and not a finding.

Coordinates (OLMo geometry theta=5e5, W=4096, d=128, K=64):
    t_j = W * omega_j / (2 pi)                in-window turns (table-free)
    T_j = D * nu_j   / (2 pi) = 4 t_j 4^-m_j  turns at the TEST distance D = 4W
"""
from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.curvature_20260910 import tables as T  # noqa: E402

K = 64
LN4 = math.log(4.0)
OLMO = dict(theta=500_000.0, window=4096.0, head_dim=128)
WINDOW = 4096.0
D_TEST = 4.0 * WINDOW                      # 16384, the RULER length
LOW, N_INCR = 14, 18

m_native = T.native_inv_freq  # unused alias guard


# ---------------------------------------------------------------------------
# the 12 arms -- name -> (m, score).  Scores are accuracy over 350 RULER rows.
# ---------------------------------------------------------------------------
def knife(delta, a):
    """olmo_beta.py --knife: taper the plateau, then leak into the held slots."""
    m = np.asarray(T.m_taper(delta, hi=32, lo=LOW), dtype=np.float64).copy()
    if a:
        m[:LOW] = np.maximum(m[:LOW],
                             a * np.arange(1, LOW + 1, dtype=np.float64))
    return np.clip(m, 0.0, 1.0)


def turns(a, b, ramp="beta1"):
    return np.asarray(T.m_turns(a, b, OLMO["theta"], OLMO["window"],
                                OLMO["head_dim"], ramp=ramp), dtype=np.float64)


ARMS = {
    "MrPro":        (T.m_incr_beta(0.0, n=N_INCR, low=LOW), 24.80 / 350),
    "beta_b0p25":   (T.m_incr_beta(0.25, n=N_INCR, low=LOW), 0.144714),
    "turns_a0p5b32": (turns(0.5, 32.0), 0.160429),
    "beta_b0p5":    (T.m_incr_beta(0.5, n=N_INCR, low=LOW), 0.232000),
    "turns_a1b16":  (turns(1.0, 16.0), 0.265143),
    "knife_both":   (knife(0.0063, 0.0063), 0.3846),
    "knife_taper":  (knife(0.0063, 0.0), 0.3937),
    "knife_leak":   (knife(0.0, 0.0063), 0.3969),
    "MrProBM":      (T.m_incr_beta(1.0, n=N_INCR, low=LOW), 145.85 / 350),
    "beta_b2":      (T.m_incr_beta(2.0, n=N_INCR, low=LOW), 0.500143),
    "turns_a2b32":  (turns(2.0, 32.0), 0.510000),
    "turns_a1b64":  (turns(1.0, 64.0), 0.538429),
}
# S from LEDGER_20260911.md §1.3 (and the archived endpoints); asserted below.
S_LEDGER = {
    "MrPro": 37.666667, "beta_b0p25": 38.464958, "turns_a0p5b32": 39.000000,
    "beta_b0p5": 39.209117, "turns_a1b16": 38.500000, "knife_both": 38.037,
    "knife_taper": 37.375, "knife_leak": 41.162, "MrProBM": 40.500000,
    "beta_b2": 42.378947, "turns_a2b32": 42.000000, "turns_a1b64": 42.000000,
}


def omega():
    return T.native_inv_freq(OLMO["theta"], K)


def turns_in_window():
    return WINDOW * omega() / (2.0 * math.pi)


def turns_at_test(m, d_test=D_TEST):
    """T_j = d_test * nu_j / 2 pi, the turns slot j makes over the test distance."""
    return (d_test / (2.0 * math.pi)) * omega() * np.power(4.0, -np.asarray(m, float))


# ---------------------------------------------------------------------------
# the candidate statistic, stated once so every variant shares one code path
# ---------------------------------------------------------------------------
def N_of(m, lo=0.25, hi=16.0, d_test=D_TEST):
    Tj = turns_at_test(m, d_test)
    return int(np.sum((Tj >= lo) & (Tj <= hi)))


def collect():
    tW = turns_in_window()
    rows = []
    for name, (m, acc) in ARMS.items():
        m = np.asarray(m, dtype=np.float64)
        Tj = turns_at_test(m)
        eps = np.diff(np.concatenate(([0.0], m)))
        j = np.arange(K, dtype=np.float64)
        rows.append(dict(
            name=name, acc=acc, m=m, T=Tj,
            S=float(m.sum()), mu_eps=float((j * eps).sum()),
            N=N_of(m),
            n_held=int(np.sum(m < 1e-12)),
            n_plateau=int(np.sum(m > 1 - 1e-12)),
            n_ramp=int(np.sum((m > 1e-12) & (m < 1 - 1e-12))),
            C=float(np.mean(np.minimum(1.0, np.power(4.0, m - 1.0)))),
            drift=float(np.sum(4.0 * tW * (1.0 - np.power(4.0, -m)))),
            m63=float(m[-1]),
        ))
    return rows


# ---------------------------------------------------------------------------
# statistics: rank correlation, and a paired test that needs no linear model
# ---------------------------------------------------------------------------
def rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    r[order] = np.arange(len(a), dtype=float)
    # average ties
    out = r.copy()
    for v in np.unique(a):
        sel = a == v
        if sel.sum() > 1:
            out[sel] = r[sel].mean()
    return out


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx @ ry) / math.sqrt((rx @ rx) * (ry @ ry)))


def rms(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def linear_fit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef, A @ coef


def paired_verdicts(rows, key):
    """Count pairs where `key` says one arm is better and the score agrees.

    A pair is usable only if the key separates it at all; ties are discarded.
    This is a NEGATIVE instrument (LESSONS: negative results beat correlations).
    """
    win = lose = 0
    for a, b in combinations(rows, 2):
        dk = key(a) - key(b)
        if abs(dk) < 1e-12:
            continue
        if (dk > 0) == (a["acc"] > b["acc"]):
            win += 1
        else:
            lose += 1
    return win, lose


# ---------------------------------------------------------------------------
def main():
    rows = collect()

    bad = {r["name"]: (r["S"], S_LEDGER[r["name"]])
           for r in rows if abs(r["S"] - S_LEDGER[r["name"]]) > 5e-3}
    print("== build check vs LEDGER ==")
    print("  S mismatches:", bad if bad else "none (all 12 within 5e-3)")
    assert not bad, "table rebuilt wrong; every number below would be suspect"

    tW = turns_in_window()
    print(f"\n== geometry (theta={OLMO['theta']:.0f}, W={WINDOW:.0f}, "
          f"D={D_TEST:.0f}, head_dim=128, K=64) ==")
    print(f"  t_W[0]={tW[0]:.4f}  t_W[31]={tW[31]:.4f}  t_W[32]={tW[32]:.4f}  "
          f"t_W[63]={tW[63]:.6f}")
    print(f"  native slot whose t_W = 1: j = "
          f"{K * math.log(WINDOW / (2 * math.pi)) / (2 * math.log(OLMO['theta'])):.3f}")

    print("\n== the 12 points, sorted by score ==")
    print(f"{'arm':15s} {'acc':>7s} {'S':>8s} {'mu_e':>6s} {'N':>3s} "
          f"{'held':>4s} {'ramp':>4s} {'plat':>4s} {'C':>6s} {'drift':>8s} "
          f"{'m63':>5s}")
    for r in sorted(rows, key=lambda r: r["acc"]):
        print(f"{r['name']:15s} {r['acc']:7.4f} {r['S']:8.3f} {r['mu_eps']:6.2f} "
              f"{r['N']:3d} {r['n_held']:4d} {r['n_ramp']:4d} {r['n_plateau']:4d} "
              f"{r['C']:6.4f} {r['drift']:8.1f} {r['m63']:5.3f}")

    print("\n== N is the only stat with this few distinct values ==")
    for key in ("N", "n_held", "n_plateau", "n_ramp"):
        vals = sorted({r[key] for r in rows})
        print(f"  {key:10s} {len(vals):2d} distinct: {vals}")

    print("\n== group means by N ==")
    for n in sorted({r["N"] for r in rows}):
        grp = [r for r in rows if r["N"] == n]
        accs = [r["acc"] for r in grp]
        gm = math.exp(np.mean([math.log(a) for a in accs]))
        print(f"  N={n:2d}  k={len(grp)}  acc {min(accs):.4f}..{max(accs):.4f}  "
              f"geo-mean {gm:.4f}  arms {[r['name'] for r in grp]}")

    print("\n== Spearman / linear rms of each candidate ==")
    acc = np.array([r["acc"] for r in rows])
    cands = {
        "S": [r["S"] for r in rows],
        "mu_eps": [r["mu_eps"] for r in rows],
        "N": [float(r["N"]) for r in rows],
        "C": [r["C"] for r in rows],
        "drift": [r["drift"] for r in rows],
        "n_plateau": [float(r["n_plateau"]) for r in rows],
        "n_held": [float(r["n_held"]) for r in rows],
        "n_ramp": [float(r["n_ramp"]) for r in rows],
        "logC": [math.log(max(r["C"], 1e-9)) for r in rows],
    }
    for nm, x in sorted(cands.items(), key=lambda kv: -abs(spearman(kv[1], acc))):
        _, yhat = linear_fit(x, acc)
        print(f"  {nm:10s} spearman {spearman(x, acc):+.3f}   lin rms {rms(acc, yhat):.4f}")

    print("\n== paired verdicts (win/lose over all separating pairs) ==")
    for nm, x in cands.items():
        lut = {r["name"]: float(v) for r, v in zip(rows, x)}
        w, l = paired_verdicts(rows, lambda r: lut[r["name"]])
        print(f"  {nm:10s} {w:3d}W / {l:3d}L   ({w / (w + l):.1%})")

    np.save("/tmp/midband_rows.npy", np.array(rows, dtype=object),
            allow_pickle=True)
    return rows


if __name__ == "__main__":
    main()
