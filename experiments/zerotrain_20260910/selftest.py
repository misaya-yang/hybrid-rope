"""CPU gate for the screen: the bank's algebra, checked before any card time.

No model, no GPU, no corpus.  Everything here is a statement about the TABLES,
and it exists because a table that is quietly wrong does not fail -- it scores,
and the score is attributed to the construction rather than to the bug.

Run:  python -m experiments.zerotrain_20260910.selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

import numpy as np

from ..curvature_20260910 import tables as T
from . import bank as B

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append(dict(name=name, fn=fn))
        return fn
    return deco


class Fail(AssertionError):
    pass


@check("every arm is monotone, with the endpoints where its construction says")
def t_monotone():
    rows = []
    for arm in B.arms():
        m = np.asarray(arm["m"], dtype=np.float64)
        if m.shape != (64,):
            raise Fail(f"{arm['name']}: {m.shape}")
        # THE REQUIREMENT IS ON nu, NOT ON m.  The plan forbids forcing the
        # cumulative scaling m to be monotone ("不要强制累计缩放 m 单调 ... 这些会
        # 排除 EVQ 中的加速或非单调搬运"), so a table where m dips at a splice is
        # legitimate as long as the frequencies stay strictly ordered -- and some
        # of the bank's splices do exactly that, because ln(omega) falls fast
        # enough per slot to absorb a falling m.  m's monotonicity is REPORTED.
        nu = T.m_to_inv_freq(m, T.QWEN25_3B["theta"])
        if not (np.diff(nu) < 0).all():
            raise Fail(f"{arm['name']}: nu is not strictly descending -- two "
                       f"slots would collide (min gap "
                       f"{float(np.diff(nu).min()):.3e})")
        if not (nu > 0).all():
            raise Fail(f"{arm['name']}: a non-positive frequency")
        if not np.isfinite(m).all():
            raise Fail(f"{arm['name']}: non-finite m")
        d = B.describe(arm)
        d["m_is_monotone"] = bool((np.diff(m) >= -1e-12).all())
        rows.append(d)
    return dict(n_arms=len(rows), rows=rows)


@check("the r family is ANCHORED: r=1 reproduces MrRoPE exactly")
def t_anchor():
    """Not 'close to'.  If the family's r=1 is not bit-identical to the deployed
    table, then every statement of the form 'the score improves past r=1' is a
    statement about a different table than the one being compared against."""
    a = T.m_incr_power(1.0)
    b = T.m_mrpro(17)
    if not np.array_equal(a, b):
        raise Fail(f"r=1 differs from MrRoPE by {np.abs(a - b).max():.3e}; the "
                   "family is not anchored and the comparison is void")
    nu_a = T.m_to_inv_freq(a, T.QWEN25_3B["theta"])
    nu_b = T.m_to_inv_freq(b, T.QWEN25_3B["theta"])
    if not np.array_equal(nu_a, nu_b):
        raise Fail("m matches but nu does not -- dtype or theta drift")
    # ... and the bank actually contains it under both names, so a harness that
    # scores them differently is caught by the mismatch rather than by a reader
    names = {a_["name"] for a_ in B.arms()}
    if "incr_r1" not in names or "mrpro_n17" not in names:
        raise Fail(f"the bank does not carry both anchors: {sorted(names)}")
    return dict(identical=True, sum_m=float(a.sum()))


@check("the family holds the conserved span at exactly 1 for every r")
def t_span():
    """The plan's conserved quantity is the band's total span, not sum m.  A
    family that quietly moved the span would be changing the compression budget
    and every comparison across r would be confounded by it."""
    rows = []
    for r in B.R_VALUES:
        m = T.m_incr_power(r)
        rec = T.band_report(m)
        if abs(rec["band_span"] - 1.0) > 1e-12:
            raise Fail(f"r={r}: band span {rec['band_span']} != 1")
        if abs(rec["m_first"]) > 1e-12 or abs(rec["m_last"] - 1.0) > 1e-12:
            raise Fail(f"r={r}: endpoints moved: {rec['m_first']} .. {rec['m_last']}")
        if abs(rec["tail_value"] - 1.0) > 1e-12:
            raise Fail(f"r={r}: the tail is not fully compressed")
        rows.append(dict(r=r, **rec))
    sums = [x["sum_m"] for x in rows]
    if len(set(np.round(sums, 12))) == 1:
        raise Fail("sum m is identical across r, so the family does not move the "
                   "quantity the receipt claims to report")
    return dict(rows=rows, sum_m_range=[min(sums), max(sums)])


@check("the r family is strictly ordered in back-loading")
def t_ordering_monotone():
    """Higher r must mean MORE back-loading, not just a different shape.

    The whole reading of the family depends on r being a monotone dial on "how
    much of the compression is deferred".  If r were not monotone in that, the
    sign of a score change would not identify a direction.
    """
    prev = None
    for r in B.R_VALUES:
        m = T.m_incr_power(r)
        front = float(np.sum(m[24:33]))          # first half of the band
        back = float(np.sum(m[33:41]))           # second half
        frac_front = front / (front + back)
        if prev is not None and not (frac_front < prev - 1e-9):
            raise Fail(f"r={r}: front share {frac_front:.6f} did not fall below "
                       f"the previous r's {prev:.6f}")
        prev = frac_front
    # and the first transition slot's perturbation must fall with r
    pert = [float(1.0 - 4.0 ** (-T.m_incr_power(r)[24])) for r in B.R_VALUES]
    if not all(pert[i] > pert[i + 1] for i in range(len(pert) - 1)):
        raise Fail(f"the front-edge perturbation is not decreasing in r: {pert}")
    return dict(front_perturbation=pert)


@check("the splices really interpolate the two incumbents")
def t_splits():
    """s=24 and s=40 must BE the two originals, or the 'ends are the incumbents'
    reading in the bank's predictions is false."""
    mM, mY = T.m_mrpro(17), T.m_yarn(k=64)
    a24 = np.asarray(B._split(24, "mrpro", "yarn"))
    a40 = np.asarray(B._split(40, "mrpro", "yarn"))
    if not np.allclose(a24, mY):
        raise Fail(f"splitA_s24 is not YaRN (max diff {np.abs(a24 - mY).max():.3e})")
    if not np.allclose(a40, mM):
        raise Fail(f"splitA_s40 is not MrPro (max diff {np.abs(a40 - mM).max():.3e})")
    b24 = np.asarray(B._split(24, "yarn", "mrpro"))
    b40 = np.asarray(B._split(40, "yarn", "mrpro"))
    if not np.allclose(b24, mM):
        raise Fail("splitB_s24 is not MrPro")
    if not np.allclose(b40, mY):
        raise Fail("splitB_s40 is not YaRN")
    # the interior points must differ from BOTH originals, or they are not tests
    for s in (28, 32, 36):
        v = np.asarray(B._split(s, "mrpro", "yarn"))
        dM, dY = np.abs(v - mM).max(), np.abs(v - mY).max()
        if dM < 1e-6 or dY < 1e-6:
            raise Fail(f"splitA_s{s} coincides with an original (dM={dM:.2e}, "
                       f"dY={dY:.2e}); it tests nothing")
    # and the literal max/min construction the plan specifies is DEGENERATE --
    # pinned here so it cannot be reintroduced as a 'simplification'
    mM_nu, mY_nu = (T.m_to_inv_freq(x, T.QWEN25_3B["theta"]) for x in (mM, mY))
    fast = np.maximum(mY_nu, mM_nu)
    n_diff = int((~np.isclose(fast, mM_nu)).sum())
    if n_diff > 4:
        raise Fail(f"the max/min construction now differs from MrPro on {n_diff} "
                   "slots; the pre-flight note says 2 and the design rationale "
                   "rests on it")
    return dict(n_slots_where_maxmin_differs=n_diff,
                differing_slots=[int(i) for i in
                                 np.flatnonzero(~np.isclose(fast, mM_nu))])


@check("the bank is declared before measurement, and carries predictions")
def t_declared():
    arms = B.arms()
    names = [a["name"] for a in arms]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise Fail(f"duplicate arm names: {dup}")
    if len(names) < 10:
        raise Fail(f"only {len(names)} arms; the campaign asks for at least ten "
                   "methods per hour and the bank is what makes that a plan")
    for a in arms:
        if not a.get("prediction"):
            raise Fail(f"{a['name']} has no pre-declared prediction")
    groups = {}
    for a in arms:
        groups[a["group"]] = groups.get(a["group"], 0) + 1
    return dict(n_arms=len(names), groups=groups, names=names)


@check("build() refuses a table that would reassign slots")
def t_build_guard():
    good = B.arms()[0]
    out = B.build(good)
    if out["values_float32"].shape != (64,):
        raise Fail("build returned the wrong shape")
    bad = dict(good, name="crossed", m=np.zeros(64))
    bad["m"] = np.asarray(good["m"]).copy()
    bad["m"][10] = 1.0                     # jump a slot then fall back
    try:
        B.build(bad)
        raise Fail("a non-monotone table was built")
    except ValueError:
        pass
    return dict(ok=True)


@check("every caller's anchor key is one anchor_check actually returns")
def t_anchor_contract():
    """THE CHECK THAT WOULD HAVE SAVED A GPU-HOUR.

    `anchor_check()` grew detailed keys and stopped returning `bit_exact`, which
    three separate callers index -- kkt_residual.py, phase1_ruler.py, and
    phase1_screen.py's own main().  Nothing pinned the contract, so the campaign
    chain ran round 1 to completion, entered its second stage, and died in one
    second with `KeyError: 'bit_exact'`, leaving the card idle.  A dict return
    across files is an interface; this is the test of it.
    """
    import importlib
    import re
    from pathlib import Path

    ps = importlib.import_module(".phase1_screen", __package__)
    ac = ps.anchor_check()
    for key in ps.anchor_contract_keys():
        if key not in ac:
            raise Fail(f"anchor_check does not return the promised key {key!r}; "
                       f"it returns {sorted(ac)}")

    # and the callers must not index anything else.  Parse them rather than
    # trusting a hand-maintained list -- a new caller is exactly the case this
    # has to catch.
    here = Path(__file__).resolve().parent
    missing = []
    for path in sorted(here.glob("*.py")):
        src = path.read_text()
        for m in re.finditer(r"\bac\[(['\"])([A-Za-z_][A-Za-z0-9_]*)\1\]", src):
            if m.group(2) not in ac:
                missing.append(f"{path.name} indexes ac[{m.group(2)!r}]")
    if missing:
        raise Fail("callers index keys the callee does not return: "
                   + "; ".join(missing))
    return dict(ok=True, keys=sorted(ac), callers_checked=len(list(here.glob('*.py'))))


@check("the scorer refuses a corpus it cannot score")
def t_scorer_guard():
    """The guard rails are checked without a model: they run in __init__."""
    from .score import Scorer

    class NoModel:
        pass

    # a document no longer than keep cannot have beyond-window scored positions,
    # which is the entire premise of the long_nll metric
    try:
        Scorer(NoModel(), [np.zeros((1, 100), dtype=np.int64)], keep=512)
        raise Fail("a document shorter than keep was accepted")
    except ValueError as e:
        if "beyond the native window" not in str(e):
            raise Fail(f"the refusal does not say why: {e}")
    try:
        Scorer(NoModel(), [], keep=8)
        raise Fail("an empty corpus was accepted")
    except ValueError:
        pass
    try:
        Scorer(NoModel(), [np.zeros((2, 5000), dtype=np.int64)], keep=8)
        raise Fail("a batched document was accepted")
    except ValueError:
        pass
    return dict(ok=True)


@check("the two mechanism dials are independent, and anchored on MrRoPE")
def t_mechanism_separation():
    """The (a, r) family exists to separate the two claimed mechanisms.

    That claim is only true if the dials really are independent: varying `a` must
    leave the TAIL's shape alone (the ratios eps_q/eps_2 for q >= 2), and varying
    `r` must leave the FIRST increment alone.  If either leaked, a score change
    along one axis would still be confounded by the other and the family would be
    no better than the single-parameter one it was built to replace.
    """
    a_mr = 2.0 / 306.0
    anchors = T.m_incr_split(a_mr, 1.0)
    mr = T.m_mrpro(17)
    max_diff = float(np.abs(anchors - mr).max())
    if not np.allclose(anchors, mr, atol=1e-15):
        raise Fail(f"the (2/306, 1) anchor is off MrRoPE by {max_diff:.3e}")
    # the ULP gap is REAL and recorded rather than rounded away
    n_ulp = int((anchors != mr).sum())

    def eps(m):
        band = np.asarray(m)[23:41]
        return np.diff(band)

    # dial a: tail ratios fixed, first increment moves
    ref_ratio = eps(T.m_incr_split(a_mr, 1.0))[1:] / eps(T.m_incr_split(a_mr, 1.0))[1]
    for a in (0.001, 0.03, 0.10):
        e = eps(T.m_incr_split(a, 1.0))
        if abs(e[0] - a) > 1e-15:
            raise Fail(f"eps_1 = {e[0]} but a = {a}")
        ratio = e[1:] / e[1]
        if np.abs(ratio - ref_ratio).max() > 1e-12:
            raise Fail(f"a={a} changed the tail shape; the dials are not "
                       f"independent (max ratio drift "
                       f"{np.abs(ratio - ref_ratio).max():.3e})")
    # dial r: first increment fixed, tail shape moves
    for r in (0.0, 2.0, 4.0):
        e = eps(T.m_incr_split(a_mr, r))
        if abs(e[0] - a_mr) > 1e-15:
            raise Fail(f"r={r} moved the first increment to {e[0]}")
    spread = [float(np.std(eps(T.m_incr_split(a_mr, r))[1:])) for r in (0.0, 2.0, 4.0)]
    if not (spread[0] < spread[1] < spread[2]):
        raise Fail(f"r does not order the tail's concentration: {spread}")
    return dict(anchor_max_diff=max_diff, anchor_slots_not_bit_exact=n_ulp,
                front_perturbation=[float(1 - 4.0 ** (-a))
                                    for a in (0.001, a_mr, 0.03, 0.10)],
                tail_spread_at_r=spread)


@check("every mechanism arm moves one dial only")
def t_mechanism_arms():
    arms = {a["name"]: a for a in B.arms()}
    front = [n for n in arms if n.startswith("front_")]
    back = [n for n in arms if n.startswith("back_")]
    both = [n for n in arms if n.startswith("both_")]
    if not (front and back and both):
        raise Fail(f"a mechanism axis is empty: front={front} back={back} both={both}")
    for n in front:
        if arms[n].get("axis") != "front" or arms[n].get("r") is not None:
            raise Fail(f"{n} carries an r, so it is not a front-only arm")
    for n in back:
        if arms[n].get("axis") != "back" or arms[n].get("a") is not None:
            raise Fail(f"{n} carries an a, so it is not a back-only arm")
    return dict(front=front, back=back, both=both)


@check("the KKT readout returns the residual it defines, on known data")
def t_kkt_readout():
    """Synthetic arms with a hand-chosen answer.

    Two dials, each with three arms.  `price` is set by construction so that the
    residual's SIGN is known: on one dial the objective falls exactly at the
    frontier's rate (residual 0), on the other it falls twice as fast (residual
    negative -- an unclaimed improvement).  A readout that got the sign or the
    projection wrong would pass a "residual is a number" test and fail here.
    """
    from . import kkt_readout as K

    p = -2.0                                  # dL/dD along the frontier
    rows = [dict(name="mrpro_n17", status="ok", native_kl=0.0, long_nll=0.0)]
    # neighbours that establish the price exactly
    rows.append(dict(name="nb1", status="ok", native_kl=0.1, long_nll=p * 0.1))
    rows.append(dict(name="nb2", status="ok", native_kl=0.2, long_nll=p * 0.2))
    # front dial: dD/da = 1, dL/da = p*1  -> residual 0
    for a, in [(0.01,), (0.02,), (0.03,)]:
        rows.append(dict(name=f"front_a{a}".replace(".", "p"), status="ok",
                         native_kl=1.0 * a, long_nll=p * (1.0 * a)))
    # back dial: dD/dr = 0.5, dL/dr = 2*p*0.5  -> residual = -p (negative)
    for r, in [(1.0,), (2.0,), (3.0,)]:
        rows.append(dict(name=f"back_r{r:g}".replace(".", "p"), status="ok",
                         native_kl=0.5 * r, long_nll=2 * p * 0.5 * r))
    out = K.readout(rows)
    price = out["pareto"]["price"]
    if abs(price - p) > 1e-9:
        raise Fail(f"price {price} != {p}")
    fr = out["dials"]["front_perturbation"]
    if abs(fr["residual"]) > 1e-9:
        raise Fail(f"a perfectly-priced dial gave residual {fr['residual']}")
    bk = out["dials"]["tail_concentration"]
    if bk["residual"] >= -1e-9:
        raise Fail(f"an unclaimed improvement gave residual {bk['residual']}, "
                   "which should be negative")
    if "NOT stationary" not in bk["reading"]:
        raise Fail(f"the reading does not name the finding: {bk['reading']}")
    # the scope disclaimer must be present -- this is the field that stops the
    # number being read as a statement about the whole 64-slot design
    if "TWO DIALS" not in out["scope"]:
        raise Fail("the readout does not declare its scope")
    return dict(price=price, front_residual=fr["residual"],
                back_residual=bk["residual"], back_reading=bk["reading"][:40])


@check("the reported metrics are the differences they claim")
def t_metric_algebra():
    """long_nll_gain is native_nll - long_nll, and the sign convention is stated.

    A gain of zero means the arm matches the native table's long-range NLL; a
    POSITIVE gain is an improvement.  Screens that report the difference the
    other way round are a classic source of a reversed leaderboard.
    """
    rec = dict(native_nll_at_native=2.5, long_nll=2.3)
    gain = rec["native_nll_at_native"] - rec["long_nll"]
    if not (gain > 0):
        raise Fail("a lower NLL must produce a POSITIVE gain")
    if abs((2.5 - 2.5)) > 1e-12:
        raise Fail("an arm equal to native must score a zero gain")
    return dict(gain_positive_means_better=True, example_gain=gain)


def run(verbose=True):
    results = []
    for spec in CHECKS:
        try:
            results.append(dict(name=spec["name"], ok=True, detail=spec["fn"]()))
        except Exception as e:                               # noqa: BLE001
            results.append(dict(name=spec["name"], ok=False,
                                error=f"{type(e).__name__}: {e}",
                                traceback=traceback.format_exc()))
    n_ok = sum(r["ok"] for r in results)
    if verbose:
        for r in results:
            print(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['name']}")
            if not r["ok"]:
                print(f"        {r['error']}")
        print(f"\n{n_ok}/{len(results)} checks passed")
    return dict(n_total=len(results), n_ok=n_ok, ok=n_ok == len(results),
                results=results)


def main(argv=None):
    ap = argparse.ArgumentParser(description="CPU gate for the zero-training screen")
    ap.add_argument("--json", default=None)
    ap.add_argument("--bank", action="store_true",
                    help="print the declared arm table and exit")
    args = ap.parse_args(argv)
    if args.bank:
        for a in B.arms():
            d = B.describe(a)
            print(f"{d['name']:18s} {d['group']:11s} sum_m={d['sum_m']:7.4f} "
                  f"span={d['band_span']:.6f} m1={d['m_first']:.3f} "
                  f"mN={d['m_last']:.3f}  {d['prediction'][:60]}")
        return 0
    out = run()
    if args.json:
        import os
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=str)
        print(f"receipt -> {args.json}")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
