#!/usr/bin/env python3
"""Build the tables that SEPARATE S from n_plateau and from sum(nu).

The point of no return for the budget story is whether S is the cause or a
stand-in for one of its co-moving partners.  The 9 existing arms cannot answer
it because they were all built by moving the SAME knobs (band edge lo, band edge
hi, ramp shape), so every candidate moved together -- and 7 of the 9 arms share
n_held = 15 and n_plateau = 32 exactly.  This file CONSTRUCTS a factorial at
FIXED S that moves n_held and n_plateau independently, prices it, and reports
the exact geometry so the experiment is pre-registered before any GPU is
touched.

THE COORDINATES.  A three-band table is fixed by three integers/one real:

    n_held h    slots 0..h-1 stay at m = 0   (native frequency)
    n_ramp r    slots h..h+r-1 carry m in (0,1)
    n_plateau p slots h+r..63 carry m = 1    (relocated onto the 4x grid)
    h + r + p = 64
    S = L_ramp + p          L_ramp = sum of m over the r interior slots

so at FIXED S the only freedom is HOW the budget is held: p slots held exactly
at m=1 and L_ramp = S - p smeared over r slots.  Building a cell means solving
for a ramp shape that carries integral L_ramp on r slots, which this file does
by bisecting the exponent of m_i = (i/(r+1))^gamma; the family is monotone in
gamma, and monotonicity is ASSERTED on the bracket before the root is trusted.

WHY THIS SHAPE FAMILY IS LEGITIMATE HERE.  Round 1 measured 29 shape arms inside
5e-3 nats in-window (LESSONS "current state of the solution space"), so shape at
fixed integral is the one axis already shown inert.  n_held and n_plateau are
not shape -- they change WHICH slots are compressed -- and are what we are
testing.

PREDICTIONS, all at S = 42 (the budget where turns_a2_b32 = 0.5100 and
turns_a1_b64 = 0.5384 both sit):
    H1 budget-only          every cell lands at ~0.51-0.54
    H2 split matters        p up  => score up   (column contrast)
    H3 held-end matters     h up  => score down (row contrast)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.kkt_20260910.sep_cause import candidates, K  # noqa: E402

S_TARGET = 42.0
# Baselines already measured at S = 42 on OLMo, for the "budget-only" prediction.
MEASURED_AT_42 = {"turns_a2_b32": 0.5100, "turns_a1_b64": 0.5384}


def power_profile(r, gamma):
    """The r interior m values of a ramp spanning gamma:  m_i = (i/(r+1))^gamma.

    gamma = 1 is EXACTLY the linear ramp already in tables.py (`m_turns(...,
    ramp='linear')` fills k/(r+1)), so this family is not a new object dropped
    in for convenience -- it contains a table the project has already run.
    gamma -> 0 drives every interior value to 1 (L_ramp -> r); gamma -> inf
    drives them to 0 (L_ramp -> 0).  Strictly decreasing in gamma, so bisection
    is valid, and the monotonicity is ASSERTED rather than assumed.
    """
    i = np.arange(1, r + 1, dtype=np.float64)
    return np.power(i / (r + 1.0), gamma)


def ramp_integral(r, gamma):
    return float(power_profile(r, gamma).sum())


def solve_gamma(r, target):
    lo, hi = 1e-3, 60.0
    flo, fhi = ramp_integral(r, lo), ramp_integral(r, hi)
    if not (flo >= target >= fhi):
        raise ValueError(f"target {target:.4f} outside [{fhi:.4f}, {flo:.4f}] "
                         f"for r={r}")
    assert flo > fhi, "power family is not monotone -- bisection invalid"
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if ramp_integral(r, mid) < target:
            hi = mid
        else:
            lo = mid
    g = 0.5 * (lo + hi)
    return g, ramp_integral(r, g)


def build_table(h, p, L_target, k=K):
    """Three-band table: h held, r ramp slots carrying integral L, p at m=1."""
    r = k - h - p
    if r < 2:
        raise ValueError(f"ramp too short: held={h} plateau={p} ramp={r}")
    if not (0.0 < L_target < r):
        raise ValueError(f"L_ramp={L_target} infeasible for r={r}")
    g, got = solve_gamma(r, L_target)
    m = np.zeros(k)
    m[h:h + r] = power_profile(r, g)
    m[h + r:] = 1.0
    return m, g, got


def design_cells():
    """The factorial.  Axis 1 moves p at fixed h; axis 2 moves h at fixed p.

    (15, 32) is the PIVOT -- it lies on both axes, so one arm serves two
    contrasts.  (12, 32) is a built-in ANCHOR: it reproduces the geometry of the
    already-measured turns_a1_b64 (h=12, p=32, r=20, L=10, S=42) with a
    different ramp SHAPE, so re-running it in the same batch tests both the
    harness (LESSONS L6/L9) and the claim that shape at fixed integral is inert.
    """
    return {
        # --- axis 1: split of the budget, held-end FROZEN at h = 15 ---
        "split_p28_h15": (15, 28),
        "split_p32_h15": (15, 32),
        "split_p36_h15": (15, 36),
        # --- axis 2: held-end, plateau FROZEN at p = 32 ---
        "held_h12_p32": (12, 32),
        "held_h19_p32": (19, 32),
    }


def main():
    print("=" * 78)
    print("0.  WHERE sum(nu) ACTUALLY VARIES  (variance decomposition, 9 arms)")
    print("=" * 78)
    from experiments.zerotrain_20260910.why import MEASURED, table_of
    names = [n for n, _ in MEASURED]
    C = [candidates(np.asarray(table_of(n), dtype=float)) for n in names]
    parts = np.array([[c["nu_held"], c["nu_ramp"], c["nu_plateau"]] for c in C])
    tot = np.array([c["sum_nu"] for c in C])
    nh = np.array([c["n_held"] for c in C])
    print(f"\n  sum(nu): min={tot.min():.5f} max={tot.max():.5f} sd={tot.std():.5f}"
          f"  (total spread {(tot.max() - tot.min()) / tot.mean() * 100:.2f}%)")
    print(f"\n  {'component':12s} {'mean':>10s} {'sd':>10s} {'share of level':>16s}")
    for i, lbl in enumerate(["nu_held", "nu_ramp", "nu_plateau"]):
        print(f"  {lbl:12s} {parts[:, i].mean():10.5f} {parts[:, i].std():10.5f}"
              f" {parts[:, i].mean() / tot.mean() * 100:15.2f}%")
    vtot = tot.var()
    print(f"\n  Var(sum_nu) = {vtot:.3e}; share of VARIANCE by component:")
    comp = {}
    for i, lbl in enumerate(["nu_held", "nu_ramp", "nu_plateau"]):
        for j, lbl2 in enumerate(["nu_held", "nu_ramp", "nu_plateau"]):
            if j < i:
                continue
            v = float(np.cov(parts[:, i], parts[:, j])[0, 1])
            comp[f"{lbl}|{lbl2}"] = v / vtot * 100
    for kk, vv in comp.items():
        print(f"    {kk:26s} {vv:+9.2f}%")
    frac = (parts[:, 0] / tot)
    print(f"\n  nu_held / sum(nu): min={frac.min():.5f}  max={frac.max():.5f}"
          f"  => sum(nu) is >= {frac.min() * 100:.1f}% a deterministic function")
    print(f"     of ONE INTEGER (n_held), via nu_held = (1-r^n)/(1-r).")
    print(f"\n  n_held values across 9 arms: {sorted(set(nh.astype(int).tolist()))}"
          f"   ({int((nh == 15).sum())}/9 arms share n_held = 15)")
    npl = np.array([c["n_plateau"] for c in C])
    print(f"  n_plateau values:            {sorted(set(npl.astype(int).tolist()))}"
          f"   ({int((npl == 32).sum())}/9 arms share n_plateau = 32)")
    both = (nh == 15) & (npl == 32)
    print(f"  => {int(both.sum())}/9 arms are IDENTICAL on BOTH separating axes,"
          f" and {int((nh == 15).sum())}/9 and {int((npl == 32).sum())}/9 share")
    print("     each axis individually.  The collinearity is a property of the")
    print("     CHOSEN ARMS, not of the physics.")

    print("\n" + "=" * 78)
    print("1.  FEASIBILITY: hit S = 42 at chosen (n_held, n_plateau)")
    print("=" * 78)
    rows = []
    print(f"\n  {'cell':16s} {'h':>3s} {'p':>3s} {'r':>3s} {'L=S-p':>7s}"
          f" {'gamma':>9s} {'L got':>8s} {'S':>9s} {'S err':>9s}")
    for cell, (h, p) in design_cells().items():
        r = K - h - p
        L = S_TARGET - p
        try:
            m, g, got = build_table(h, p, L)
            cd = candidates(m)
            rows.append((cell, h, p, r, L, m, cd))
            print(f"  {cell:16s} {h:3d} {p:3d} {r:3d} {L:7.2f} {g:9.4f}"
                  f" {got:8.4f} {cd['S']:9.5f} {cd['S'] - S_TARGET:+9.1e}")
        except ValueError as exc:
            print(f"  {cell:16s} {h:3d} {p:3d} {r:3d} {L:7.2f}   INFEASIBLE: {exc}")

    if rows:
        print("\n  --- the design, realised (all with S pinned at 42) ---")
        print(f"\n  {'cell':16s} {'h':>3s} {'p':>3s} {'r':>3s} {'L_ramp':>7s}"
              f" {'sum_nu':>9s} {'nu_held':>9s} {'wt% held':>9s}")
        for cell, h, p, r, L, m, cd in rows:
            print(f"  {cell:16s} {cd['n_held']:3.0f} {cd['n_plateau']:3.0f}"
                  f" {cd['n_ramp']:3.0f} {cd['L_ramp']:7.2f} {cd['sum_nu']:9.6f}"
                  f" {cd['nu_held']:9.6f} {cd['nu_held'] / cd['sum_nu'] * 100:8.2f}%")
        print("\n  AND THE CONTROL THAT MATTERS: how far is each cell from the")
        print("  9 arms we already ran?  A design differing in <=2 slots tests")
        print("  nothing (LESSONS L8).")
        allm = [np.asarray(table_of(n), float) for n in names]
        for cell, h, p, r, L, m, cd in rows:
            d = sorted((int(np.sum(np.abs(m - mm) > 1e-9)), n)
                       for n, mm in zip(names, allm))
            print(f"    {cell:16s} nearest {d[0][1]:16s} ({d[0][0]:2d} slots),"
                  f"  next {d[1][1]:16s} ({d[1][0]:2d} slots)")
        print("\n  --- the contrasts and what each isolates ---")
        d = {cell: cd for cell, _, _, _, _, _, cd in rows}
        def show(tag, a, b):
            ca, cb = d[a], d[b]
            print(f"  {tag:40s} dS={ca['S'] - cb['S']:+.2e}"
                  f"  dp={ca['n_plateau'] - cb['n_plateau']:+.0f}"
                  f"  dh={ca['n_held'] - cb['n_held']:+.0f}"
                  f"  d(nu)={ca['sum_nu'] - cb['sum_nu']:+.5f}")
        for tag, a, b in [
            ("H2 read: p36 - p32  (h fixed 15)", "split_p36_h15", "split_p32_h15"),
            ("H2 read: p28 - p32  (h fixed 15)", "split_p28_h15", "split_p32_h15"),
            ("H3 read: h19 - h12  (p fixed 32)", "held_h19_p32", "held_h12_p32"),
            ("H3 read: h15 - h12  (p fixed 32)", "split_p32_h15", "held_h12_p32"),
            ("ANCHOR vs turns_a1_b64 geometry", "held_h12_p32", "held_h12_p32"),
        ]:
            if a in d and b in d:
                show(tag, a, b)

    print("\n" + "=" * 78)
    print("2.  PREDICTIONS  (baseline: the two MEASURED arms already at S = 42)")
    print("=" * 78)
    print(f"\n  measured at S=42: " + ", ".join(
        f"{n} = {v:.4f}" for n, v in MEASURED_AT_42.items()))
    print("  budget-only reading therefore predicts EVERY cell at ~0.51-0.54.")
    print(f"\n  {'hypothesis':30s} {'p28':>7s} {'p32':>7s} {'p36':>7s}"
          f"   {'h11':>7s} {'h13':>7s} {'h19':>7s}")
    print(f"  {'H1 budget only':30s} {'0.52':>7s} {'0.52':>7s} {'0.52':>7s}"
          f"   {'0.52':>7s} {'0.52':>7s} {'0.52':>7s}")
    print(f"  {'H2 more plateau better':30s} {'low':>7s} {'mid':>7s} {'high':>7s}"
          f"   {'-':>7s} {'-':>7s} {'-':>7s}")
    print(f"  {'H3 more held worse':30s} {'-':>7s} {'-':>7s} {'-':>7s}"
          f"   {'best':>7s} {'mid':>7s} {'worst':>7s}")
    print("\n  H2 and H3 are read off ORTHOGONAL contrasts (column vs row); a")
    print("  cell disagreeing with both voids the budget-only reading outright.")

    print("\n" + "=" * 78)
    print("3.  SEPARATING POWER: 9 arms vs 9 + this design")
    print("=" * 78)
    from analysis.kkt_20260910.sep_power import vif, ols
    newm = {cell: m for cell, h, p, r, L, m, cd in rows}
    base_C = {k: np.array([c[k] for c in C], dtype=float) for k in C[0]}
    new_C = {}
    for cell, m in newm.items():
        cd = candidates(m)
        for k, v in cd.items():
            new_C.setdefault(k, np.array([]))
            new_C[k] = np.append(new_C[k], v)
    for k in base_C:
        if k in new_C:
            new_C[k] = np.concatenate([base_C[k], new_C[k]])
    print(f"\n  {'design':16s} {'VIF(S)':>9s} {'VIF(n_plat)':>12s}"
          f" {'VIF(sum_nu)':>13s} {'R^2(S~others)':>14s}")
    for label, CC in (("9 arms", base_C), (f"9 + {len(newm)}", new_C)):
        vS = vif(CC, "S", ["n_plateau", "sum_nu"])[0]
        vP = vif(CC, "n_plateau", ["S", "sum_nu"])[0]
        vN = vif(CC, "sum_nu", ["S", "n_plateau"])[0]
        print(f"  {label:16s} {vS:9.2f} {vP:12.2f} {vN:13.2f} {1 - 1 / vS:14.4f}")
    print("\n  R^2 of each candidate regressed on the other two is the")
    print("  collinearity that destroys separation; VIF = 1/(1-R^2).")

    print("\n  --- the ramp the design actually buys (per-candidate contrast) ---")
    print(f"\n  {'candidate':14s} {'9-arm range':>22s} "
          f"{'9+design range':>22s} {'gain':>8s}")
    for k in ("S", "n_plateau", "n_held", "sum_nu"):
        b = (base_C[k].min(), base_C[k].max())
        a = (new_C[k].min(), new_C[k].max())
        print(f"  {k:14s} [{b[0]:9.4f}, {b[1]:9.4f}] "
              f"[{a[0]:9.4f}, {a[1]:9.4f}] {(a[1] - a[0]) / (b[1] - b[0]):7.2f}x")

    print("\n  --- cost ---")
    print("   RULER 16K panel: 1384.977 s / 700 generations = 1.979 s/gen,")
    print("                    so 350 rows = ~692 s = 11.5 min per arm")
    print("                    (results/olmo_fast_screen_20260908/"
          "run_ruler_newtasks_01/status.json)")
    print("   in-window NLL:   44.439 s for 112 forwards INCLUDING model load")
    print("                    = 0.40 s/forward (run_nll_01/status.json)")
    print(f"   => {len(newm) + 2} new arms RULER = "
          f"{(len(newm) + 2) * 692 / 60:.0f} min;")
    print(f"      in-window @4096 for {len(newm) + 2} arms x 16 docs = "
          f"{(len(newm) + 2) * 16} forwards = "
          f"{(len(newm) + 2) * 16 * 0.40:.0f} s")
    print("   Both regimes on ONE checkpoint for the price of one panel.")

    out = dict(
        S_target=S_TARGET,
        measured_at_S42=MEASURED_AT_42,
        cells=[dict(cell=cell, n_held=int(h), n_plateau=int(p), n_ramp=int(r),
                    L_ramp=float(cd["L_ramp"]), S=float(cd["S"]),
                    sum_nu=float(cd["sum_nu"])) for cell, h, p, r, L, m, cd in rows],
        m_vectors={cell: [float(x) for x in m] for cell, h, p, r, L, m, cd in rows},
        note=("Factorial at fixed S=42 on OLMo geometry.  Ramp shape at fixed "
              "integral is the axis round 1 showed inert in-window (29 arms "
              "inside 5e-3 nats)."))
    dest = Path(__file__).resolve().parent / "sep_design_tables.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\n  wrote {dest}")


if __name__ == "__main__":
    sys.exit(main())
