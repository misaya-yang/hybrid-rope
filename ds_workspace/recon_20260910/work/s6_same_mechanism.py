"""§6 verification: are the four "plateau" winners one mechanism or several?

The Pro plan (RESEARCH_PLAN.md §6) proposes two per-row statistics computable
from EXISTING results, no GPU:

    Delta_mean = mean(score_A - score_B)      (net effect)
    D_pattern  = mean(|score_A - score_B|)    (how many samples move at all)

and claims: "same mean but large D_pattern means the arms fix different samples".
This script computes both for every pair among the candidate arms, plus the
machinery needed to say whether the residual is signal or sampling noise:
paired SE, a sign test, and the "both-move" fraction (how often the two arms
disagree row-by-row while the aggregate ties).

READ-ONLY. Pure numpy. No model, no GPU.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JL = os.path.join(HERE, "jsonl")

# arm name -> (relative path, human label)
ARMS = {
    "MrPro":      ("archive/MrPro.jsonl",          "MrRoPE m_mrpro(n=18,low=14)"),
    "BM":         ("archive/MrProBM.jsonl",        "deployed BM m_incr_beta(1.0,n=18,low=14)"),
    "b3_lo14":    ("olmo_b3/beta_b3p0.jsonl",      "m_incr_beta(3.0,n=18,low=14)"),
    "ctl_C42":    ("olmo_c42/ctl_C42.jsonl",       "C42 control"),
    "ctl_C42V24": ("olmo_c42/ctl_C42V24.jsonl",    "C42-V24 control"),
    "wide_b4":    ("olmo_wide/wide_b4p0.jsonl",    "m_incr_beta(4.0,n=21,low=11)"),
    "turns_a1_b64": ("olmo/turns_a1_b64.jsonl",    "m_incr_beta(1.0,n=21,low=11)"),
    "step_hi25":  ("olmo_tstar/step_hi25.jsonl",   "step hi=25"),
}


def load(rel):
    out = {}
    with open(os.path.join(JL, rel)) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["row_id"]] = (float(d["correct"]), d.get("task"))
    return out


def main():
    data = {a: load(p) for a, (p, _) in ARMS.items()}

    # ---- alignment check -------------------------------------------------
    print("=" * 78)
    print("ROW-LEVEL ALIGNMENT")
    print("=" * 78)
    ids = {a: set(d) for a, d in data.items()}
    ref = ids["BM"]
    for a in ARMS:
        miss = len(ref - ids[a])
        extra = len(ids[a] - ref)
        print(f"  {a:14s} n={len(ids[a]):4d}  missing_vs_BM={miss:3d}  extra={extra:3d}")

    common = sorted(set.intersection(*ids.values()))
    print(f"  common row_ids across all {len(ARMS)} arms: {len(common)}")
    if len(common) != 350:
        print("  !! not 350 -- investigate before trusting the numbers")

    tasks = {}
    for rid in common:
        tasks.setdefault(data["BM"][rid][1], []).append(rid)
    print("\n  task composition (common rows):")
    for t, rs in sorted(tasks.items()):
        print(f"    {t:24s} {len(rs):4d}")

    S = {a: np.array([data[a][r][0] for r in common]) for a in ARMS}

    # ---- headline accuracies (sanity vs the stated measured facts) --------
    print("\n" + "=" * 78)
    print("HEADLINE SCORES (common rows)")
    print("=" * 78)
    for a in ARMS:
        print(f"  {a:14s} {S[a].mean():.4f}   sum={S[a].sum():8.2f}   {ARMS[a][1]}")

    # ---- the §6 table ----------------------------------------------------
    CAND = ["b3_lo14", "ctl_C42V24", "wide_b4", "turns_a1_b64"]
    print("\n" + "=" * 78)
    print("§6 CORE: Delta_mean and D_pattern for every pair")
    print("=" * 78)
    print(f"{'A':14s} {'B':14s} {'mean_A':>8s} {'mean_B':>8s} {'Delta':>9s} "
          f"{'D_pat':>8s} {'D_pat/|D|':>10s} {'SE_d':>7s} {'p_sign':>8s} "
          f"{'bothmove':>9s}")
    print("-" * 78)

    n = len(common)
    rows = []
    for a, b in itertools.combinations(CAND, 2):
        d = S[a] - S[b]
        dm = d.mean()
        dp = np.abs(d).mean()
        se = d.std(ddof=1) / math.sqrt(n)
        # two-sided sign test on the nonzero differences
        nz = d[d != 0]
        npos = int((nz > 0).sum())
        nneg = int((nz < 0).sum())
        p_sign = _binom_two_sided(npos, npos + nneg)
        both = float((np.abs(d) > 1e-12).mean())
        rows.append((a, b, dm, dp, se, p_sign, both, npos, nneg))
        ratio = "inf" if abs(dm) < 1e-12 else f"{dp/abs(dm):8.2f}"
        print(f"{a:14s} {b:14s} {S[a].mean():8.4f} {S[b].mean():8.4f} "
              f"{dm:+9.4f} {dp:8.4f} {ratio:>10s} {se:7.4f} {p_sign:8.4f} "
              f"{both:9.4f}")

    # ---- and against the two refusals (BM, C42) --------------------------
    print("\n" + "=" * 78)
    print("§6 EXTENDED: the four candidates vs the two non-plateau references")
    print("=" * 78)
    print(f"{'A':14s} {'B':14s} {'Delta':>9s} {'D_pat':>8s} {'SE_d':>7s} "
          f"{'p_sign':>8s} {'bothmove':>9s}  {'95% CI on Delta'}")
    print("-" * 78)
    for a in CAND:
        for b in ["BM", "ctl_C42"]:
            d = S[a] - S[b]
            dm, dp = d.mean(), np.abs(d).mean()
            se = d.std(ddof=1) / math.sqrt(n)
            nz = d[d != 0]
            p_sign = _binom_two_sided(int((nz > 0).sum()), len(nz))
            both = float((np.abs(d) > 1e-12).mean())
            ci = (dm - 1.96 * se, dm + 1.96 * se)
            print(f"{a:14s} {b:14s} {dm:+9.4f} {dp:8.4f} {se:7.4f} {p_sign:8.4f} "
                  f"{both:9.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    # ---- the decisive question: do the four move the SAME rows? ----------
    print("\n" + "=" * 78)
    print("§6 DECISIVE: do the four candidates move the same rows?")
    print("=" * 78)
    print("  Pairwise AGREEMENT on which rows moved (relative to BM).")
    print("  Jaccard of {rows where arm != BM}.  1.0 = identical support.")
    moved = {a: set(np.nonzero(np.abs(S[a] - S["BM"]) > 1e-12)[0]) for a in CAND}
    print(f"\n  {'A':14s} {'B':14s} {'nA':>5s} {'nB':>5s} {'both':>5s} "
          f"{'union':>6s} {'Jaccard':>8s}")
    for a, b in itertools.combinations(CAND, 2):
        inter = moved[a] & moved[b]
        uni = moved[a] | moved[b]
        j = len(inter) / len(uni) if uni else float("nan")
        print(f"  {a:14s} {b:14s} {len(moved[a]):5d} {len(moved[b]):5d} "
              f"{len(inter):5d} {len(uni):6d} {j:8.3f}")

    for a in CAND:
        print(f"    {a:14s} moves {len(moved[a]):3d}/{n} rows vs BM")

    # rows where all four agree in SIGN of delta vs BM
    print("\n  Sign agreement of delta-vs-BM across the four candidates:")
    sign_mat = {a: np.sign(S[a] - S["BM"]) for a in CAND}
    allsame = np.ones(n, dtype=bool)
    for a in CAND[1:]:
        allsame &= (sign_mat[a] == sign_mat[CAND[0]])
    allsame &= (sign_mat[CAND[0]] != 0)
    print(f"    rows where all four move the SAME direction: {int(allsame.sum())}")
    for a in CAND:
        print(f"    rows where {a:14s} moves and BM does not (up/down): "
              f"{int((sign_mat[a] > 0).sum())}/{int((sign_mat[a] < 0).sum())}")

    # ---- per-task breakdown: is the plateau one task or all tasks? -------
    print("\n" + "=" * 78)
    print("§6 PER-TASK: where the four candidates' advantage lives")
    print("=" * 78)
    tnames = sorted(tasks)
    hdr = f"  {'task':24s} {'n':>4s} " + " ".join(f"{a[:11]:>11s}" for a in
                                                  ["BM"] + CAND)
    print(hdr)
    for t in tnames:
        idx = [common.index(r) for r in tasks[t]]
        line = f"  {t:24s} {len(idx):4d} " + " ".join(
            f"{S[a][idx].mean():11.4f}" for a in ["BM"] + CAND)
        print(line)

    # ---- what the plan actually asserts -----------------------------------
    print("\n" + "=" * 78)
    print("§6 VERDICT INPUTS")
    print("=" * 78)
    dm_all = [r[2] for r in rows]
    dp_all = [r[3] for r in rows]
    print(f"  among the four candidates: |Delta_mean| ranges "
          f"{min(abs(x) for x in dm_all):.4f} .. {max(abs(x) for x in dm_all):.4f}")
    print(f"                             D_pattern  ranges "
          f"{min(dp_all):.4f} .. {max(dp_all):.4f}")
    print(f"  ratio D_pattern / |Delta_mean| ranges "
          f"{min(dp_all)/max(abs(x) for x in dm_all):.1f} .. "
          f"{max(dp_all)/min(abs(x) for x in dm_all):.1f}")

    # noise floor: split-half of BM against itself is not available, so use
    # the binomial SE of a single arm's score as the reference scale.
    se_single = math.sqrt(S["BM"].mean() * (1 - S["BM"].mean()) / n)
    print(f"  single-arm binomial SE at p={S['BM'].mean():.4f}, n={n}: {se_single:.4f}")
    print(f"  paired SE of a candidate-vs-BM difference (typical): "
          f"{np.mean([ (S[a]-S['BM']).std(ddof=1)/math.sqrt(n) for a in CAND]):.4f}")


def _binom_two_sided(k, m):
    """Exact two-sided sign-test p-value, ties already removed."""
    if m == 0:
        return 1.0
    from math import comb
    tail = sum(comb(m, i) for i in range(0, min(k, m - k) + 1)) / 2.0 ** m
    return min(1.0, 2.0 * tail)


if __name__ == "__main__":
    main()
