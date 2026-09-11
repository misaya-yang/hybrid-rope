"""§6, part 2: the C42 / C42-V24 controlled pair and the "one mechanism?" test.

The plan (§6) proposes using C42's two arms as a NEW prediction test of whether
S (the budget) is approximately sufficient at fixed band / endpoints / support.
This script:
  1. rebuilds C42 and C42V24 from their closed forms and tabulates every
     coordinate the campaign uses (S, centroid, band, plateaus, increment
     variance) so the "only the second moment moves" claim can be checked;
  2. runs the per-row tests that distinguish one mechanism from several:
     containment of the moved-row sets, sign agreement, and the rows where two
     candidates disagree with each other.
No GPU, no model, pure numpy + the existing jsonl.
"""
from __future__ import annotations

import itertools
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JL = os.path.join(HERE, "jsonl")
K = 64


# --- exact copies of the closed forms (experiments/curvature_20260910) ------
def m_C42(lo=14, n=18, k=K):
    kk = np.arange(1, int(n) + 1, dtype=np.float64)
    p = 6.0 * kk * (int(n) + 1 - kk) / (n * (n + 1) * (n + 2))
    eps = p * (1.0 - (10.0 / 119.0) * (kk - 19.0 / 2.0))
    m = np.zeros(k)
    m[lo + 1: lo + int(n) + 1] = np.cumsum(eps)
    m[lo + int(n) + 1:] = 1.0
    return m


def m_C42V24(lo=14, n=18, k=K):
    kk = np.arange(1, int(n) + 1, dtype=np.float64)
    p = 6.0 * kk * (int(n) + 1 - kk) / (n * (n + 1) * (n + 2))
    eps = p * (1.0 - (10.0 / 119.0) * (kk - 19.0 / 2.0)
               + (35.0 / 1496.0) * ((kk - 19.0 / 2.0) ** 2 - 357.0 / 20.0))
    m = np.zeros(k)
    m[lo + 1: lo + int(n) + 1] = np.cumsum(eps)
    m[lo + int(n) + 1:] = 1.0
    return m


def m_incr_beta(b, n, low, shape_a=1.0, k=K):
    kk = np.arange(1, int(n) + 1, dtype=np.float64)
    w = np.power(kk, float(shape_a)) * np.power(int(n) + 1 - kk, float(b))
    eps = w / w.sum()
    out = np.zeros(k)
    out[low:low + int(n) + 1] = np.concatenate([[0.0], np.cumsum(eps)])
    out[low + int(n) + 1:] = 1.0
    return out


def coords(m):
    e = np.diff(np.concatenate([[0.0], m]))
    kk = np.arange(0, K, dtype=np.float64)
    mu = float((kk * e).sum())
    var = float(((kk - mu) ** 2 * e).sum())
    held = int((m <= 1e-12).sum())
    plat = int((m >= 1 - 1e-12).sum())
    band = np.nonzero((m > 1e-12) & (m < 1 - 1e-12))[0]
    return dict(S=float(m.sum()), mu=mu, var=var, held=held, plat=plat,
                band=(int(band[0]), int(band[-1])) if len(band) else None,
                nband=len(band), min_eps=float(e[e > 0].min()) if (e > 0).any() else 0.0)


def load(rel):
    out = {}
    for line in open(os.path.join(JL, rel)):
        line = line.strip()
        if line:
            d = json.loads(line)
            out[d["row_id"]] = float(d["correct"])
    return out


def main():
    print("=" * 78)
    print("A.  THE C42 / C42-V24 CONTROLLED PAIR — what actually differs?")
    print("=" * 78)
    tabs = {
        "C42": m_C42(),
        "C42V24": m_C42V24(),
        "a1_b64": m_incr_beta(1.0, 21, 11),
        "b3_lo14": m_incr_beta(3.0, 18, 14),
        "wide_b4": m_incr_beta(4.0, 21, 11),
        "BM": m_incr_beta(1.0, 18, 14),
    }
    print(f"  {'arm':10s} {'S':>8s} {'centroid':>9s} {'var(eps)':>9s} "
          f"{'held':>5s} {'plat':>5s} {'band':>10s} {'n_band':>7s} {'min eps':>9s}")
    for n, t in tabs.items():
        c = coords(t)
        print(f"  {n:10s} {c['S']:8.4f} {c['mu']:9.4f} {c['var']:9.4f} "
              f"{c['held']:5d} {c['plat']:5d} {str(c['band']):>10s} "
              f"{c['nband']:7d} {c['min_eps']:9.6f}")

    print("\n  C42 vs C42V24 — the ONLY differences:")
    for n in ["S", "mu", "held", "plat", "band", "nband"]:
        a, b = coords(tabs["C42"])[n], coords(tabs["C42V24"])[n]
        print(f"    {n:8s} C42={a}  C42V24={b}   same={a == b}")
    print(f"    var(eps) C42={coords(tabs['C42'])['var']:.6f}  "
          f"C42V24={coords(tabs['C42V24'])['var']:.6f}   "
          f"(tables.py docstring says 15.6 -> 24)")

    print("\n  MEASURED (350-row 16K RULER, paired):")
    A = load("olmo_c42/ctl_C42.jsonl")
    B = load("olmo_c42/ctl_C42V24.jsonl")
    BM = load("archive/MrProBM.jsonl")
    ids = sorted(A)
    a = np.array([A[r] for r in ids])
    b = np.array([B[r] for r in ids])
    bm = np.array([BM[r] for r in ids])
    d = b - a
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    print(f"    C42    {a.mean():.4f}")
    print(f"    C42V24 {b.mean():.4f}")
    print(f"    delta  {d.mean():+.4f}  (paired SE {se:.4f}, "
          f"95% CI [{d.mean()-1.96*se:+.4f}, {d.mean()+1.96*se:+.4f}])")
    print(f"    D_pattern {np.abs(d).mean():.4f}   rows moved "
          f"{int((np.abs(d) > 1e-12).sum())}/{len(d)}")
    print(f"    SAME S=42, SAME band, SAME endpoints, SAME plateaus -> "
          f"{d.mean():+.4f} from the second moment alone")

    print("\n" + "=" * 78)
    print("B.  ONE MECHANISM OR SEVERAL?  (deterministic harness, no noise floor)")
    print("=" * 78)
    CAND = ["b3_lo14", "ctl_C42V24", "wide_b4", "turns_a1_b64"]
    src = {
        "b3_lo14": "olmo_b3/beta_b3p0.jsonl",
        "ctl_C42": "olmo_c42/ctl_C42.jsonl",
        "ctl_C42V24": "olmo_c42/ctl_C42V24.jsonl",
        "wide_b4": "olmo_wide/wide_b4p0.jsonl",
        "turns_a1_b64": "olmo/turns_a1_b64.jsonl",
        "MrPro": "archive/MrPro.jsonl",
        "step_hi25": "olmo_tstar/step_hi25.jsonl",
        "BM": "archive/MrProBM.jsonl",
    }
    data = {k: load(v) for k, v in src.items()}
    ids = sorted(data["BM"])
    X = {k: np.array([v[r] for r in ids]) for k, v in data.items()}
    tasks = [data["BM"] and json.loads(l)["task"]
             for l in open(os.path.join(JL, "archive/MrProBM.jsonl"))]
    tid = {}
    for l in open(os.path.join(JL, "archive/MrProBM.jsonl")):
        dd = json.loads(l)
        tid[dd["row_id"]] = dd["task"]
    tl = [tid[r] for r in ids]

    print("  B1. Is the moved-set NESTED?  (Jaccard + containment vs BM)")
    moved = {k: set(np.nonzero(np.abs(X[k] - X["BM"]) > 1e-12)[0]) for k in CAND}
    print(f"  {'A':14s} {'B':14s} {'Jaccard':>8s} {'A⊂B':>7s} {'B⊂A':>7s}")
    for p, q in itertools.combinations(CAND, 2):
        inter, uni = moved[p] & moved[q], moved[p] | moved[q]
        print(f"  {p:14s} {q:14s} {len(inter)/len(uni):8.3f} "
              f"{str(moved[p] <= moved[q]):>7s} {str(moved[q] <= moved[p]):>7s}")

    print("\n  B2. Rows where two candidates DISAGREE (one gains, the other loses,"
          " both vs BM)")
    print(f"  {'A':14s} {'B':14s} {'A>BM&B<BM':>10s} {'B>BM&A<BM':>10s} "
          f"{'A!=B':>7s}")
    for p, q in itertools.combinations(CAND, 2):
        dp = X[p] - X["BM"]
        dq = X[q] - X["BM"]
        n1 = int(((dp > 0) & (dq < 0)).sum())
        n2 = int(((dq > 0) & (dp < 0)).sum())
        ne = int((np.abs(X[p] - X[q]) > 1e-12).sum())
        print(f"  {p:14s} {q:14s} {n1:10d} {n2:10d} {ne:7d}")

    print("\n  B3. per-task profile vs BM (does the advantage live in the same tasks?)")
    tset = sorted(set(tl))
    print(f"  {'task':22s} {'n':>4s} " + " ".join(f"{k[:10]:>11s}"
                                                 for k in ["BM"] + CAND))
    for t in tset:
        idx = [i for i, x in enumerate(tl) if x == t]
        print(f"  {t:22s} {len(idx):4d} " +
              " ".join(f"{X[k][idx].mean():11.4f}" for k in ["BM"] + CAND))

    print("\n  B4. rank correlation of the per-row gain vs BM across candidates")
    G = {k: X[k] - X["BM"] for k in CAND}
    print(f"  {'A':14s} {'B':14s} {'Pearson':>9s} {'Spearman':>9s}")
    def spear(u, v):
        ru = np.argsort(np.argsort(u)).astype(float)
        rv = np.argsort(np.argsort(v)).astype(float)
        return float(np.corrcoef(ru, rv)[0, 1])
    for p, q in itertools.combinations(CAND, 2):
        print(f"  {p:14s} {q:14s} "
              f"{float(np.corrcoef(G[p], G[q])[0,1]):9.4f} {spear(G[p], G[q]):9.4f}")


if __name__ == "__main__":
    main()
