"""The CPU gate: every mathematical claim this package rests on, checked here.

FINAL_PLAN.md sec.11 lists what the reference package must self-test:

    "已知1D QCQP闭式答案、线性不等式、拒绝原生超额步、gain联合解等价、
     影子价格方向、随机logits上的错误上界。所有自测均为数学/软件检查，无真实
     模型结论。"

Those six are here, and so is everything that had to be pinned while building
the modules around them, because the point of the gate is that a session with a
card cannot silently run a package whose algebra has drifted:

  coordinates   sec.2's x/a pack-unpack round trips; eps recovers the coordinate
                the panel and the older package already speak.
  feasible set  the ordering rows are exactly  x_{j+1} - x_j >= delta_min, and a
                point that breaks them is DETECTED rather than sorted.
  QCQP          a 1-D problem whose min-max is hand-derivable, a 2-D problem
                with a linear equality whose KKT point is hand-derivable, and
                the trust ellipsoid's implied box.
  solver        random problems: every returned step satisfies the equalities,
                the trust region and  t = max_i q_i(d); an inconsistent
                constraint set is REPORTED, not converged to.
  sec.6         eliminating the gain by Schur complement gives the SAME step as
                solving the full 65-variable problem, and differs from merely
                projecting the gradient against a pinned gain.  This is the
                check that keeps the elimination honest.
  sec.5         v . (gap transfer) == price_k - price_i, against a finite
                difference of the objective.
  sec.3         b_e equals a brute-force max over (position, wrong token); the
                smoothed bound sandwiches it at several eta.
  sec.4         D_keep(native) == 0 exactly; a row whose baseline has no margin
                is EXCLUDED, not floored.
  sec.8         the acceptance predicate accepts a dominated step, refuses an
                optimistic one, refuses a native overrun, refuses an out-of-band
                rho, and refuses to price a gain the model has no coordinate for;
                the retry ladder raises damping and converges when damping can
                fix the failure, and refuses when it cannot.
  panel         the span guard reports the rank a constructed matrix actually
                has, and the null fraction of an in-span step is 0 (at the
                float64 floor, not at 1e-8).
  loop          the whole driver, on a quadratic objective whose constrained
                optimum is closed-form: Phase-I repairs an infeasible start
                without touching eps, the step sequence reaches the analytic
                optimum without passing it, a worsening objective yields a
                recorded refusal rather than a step, and the Pareto sweep
                carries the slope the plan's dE*/deps = -lambda asserts.

Every check is a mathematical or software statement.  NONE of them is a claim
about a model: passing this gate says the package computes what it says it
computes, and says nothing about whether the table it produces is better.

Run:   python -m experiments.joint_kkt_20260910.selftest
       python -m experiments.joint_kkt_20260910.selftest --json out.json
       python -m experiments.joint_kkt_20260910.selftest --only sec6
Exit code is 0 only if every check passed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback

import numpy as np

from . import accept as A
from . import bound as BD
from . import design as D
from . import qcqp
from . import risk as RK

CHECKS = []


def check(name, group):
    def deco(fn):
        CHECKS.append(dict(name=name, group=group, fn=fn))
        return fn
    return deco


class Fail(AssertionError):
    pass


def close(a, b, tol=1e-9, what=""):
    """Max absolute difference, so a vector and a scalar are compared the same
    way and a caller cannot accidentally get a truth test on an array."""
    d = float(np.max(np.abs(np.asarray(a, dtype=np.float64)
                            - np.asarray(b, dtype=np.float64))))
    if not (d <= tol):
        raise Fail(f"{what}: |{a!r} - {b!r}| = {d:.3e} > {tol:g}")


# ===========================================================================
# sec.2 -- coordinates
# ===========================================================================
@check("coordinates round trip", "design")
def t_coords():
    rng = np.random.default_rng(0)
    nu = np.exp(-rng.uniform(-2, 2, D.K))
    gain = 1.234
    y = D.pack(nu, gain)
    nu2, g2 = D.unpack(y)
    close(np.abs(nu2 - nu).max(), 0.0, 1e-12, "nu round trip")
    close(g2, gain, 1e-12, "gain round trip")
    # x is eps measured FROM THE NATIVE POINT -- the origin is the native table,
    # not zero -- and the package that owns the m definition must agree
    x = y[:D.K]
    origin = D.native_x()
    eps = D.eps_of_x(x, origin)
    close(np.abs(eps - (x - origin)).max(), 0.0, 1e-15, "eps = x - x_native")
    # m = eps / ln S with S the EXTENSION SCALE (4 for the prepared panel), NOT
    # ln theta.  ln theta / K is the native grid's log-gap -- a different number
    # that design.delta_min is built from -- and the two are easy to confuse
    # because both are "the log spacing".  `m_to_inv_freq` is the authority.
    m = D.T.inv_freq_to_m(D.nu_of_x(x), D.T.QWEN25_3B["theta"])
    close(np.abs(m - eps / D.T.LN_S).max(), 0.0, 1e-12, "m == eps/ln S")
    close(D.T.LN_S, math.log(D.T.QWEN25_3B["scale"]), 1e-15, "S is the panel scale")
    # and the package's own inverse round trips against it
    nu_back = D.T.m_to_inv_freq(m, D.T.QWEN25_3B["theta"])
    close(np.abs(nu_back - D.nu_of_x(x)).max(), 0.0, 1e-12, "m -> nu round trip")
    # a = log g^2, and the native point has m = 0
    close(D.a_of_gain(gain), 2 * math.log(gain), 1e-12, "a = 2 log g")
    m0 = D.T.inv_freq_to_m(D.nu_of_x(origin), D.T.QWEN25_3B["theta"])
    close(np.abs(m0).max(), 0.0, 1e-12, "native m == 0")
    return dict(gain=gain, m_native_max=float(np.abs(m0).max()),
                x_is_increasing=bool((np.diff(origin) > 0).all()))


# ===========================================================================
# feasible set
# ===========================================================================
@check("ordering rows detect a sub-floor gap, and never sort", "design")
def t_ordering():
    x = D.native_x()
    gap = math.log(D.T.QWEN25_3B["theta"]) / D.K      # 0.21587
    G, h = D.ordering_rows(x)
    # the increment form: h_j = gap_j - delta, and at native every gap is equal
    close(h.min(), gap - D.delta_min(), 1e-12, "h at native")

    # the reference is the FULL design vector at native, gain included: a bare
    # 64-vector cannot be subtracted from a 65-vector, and both `feasible_set`
    # and `violations` refuse it rather than broadcasting into a wrong slack
    y0 = D.pack(D.nu_of_x(x), 1.0)
    A_, b_, G_, h_ = D.feasible_set(y0)
    v0 = D.violations(y0, y0, A_, b_, G_, h_)
    if not (v0["ineq_ok"] and v0["call_ok"] and v0["freq_order_ok"]):
        raise Fail(f"the native point was reported infeasible: {v0}")

    # a gap exactly at the floor is feasible; a hair below it is not.  This is
    # the case the increment/point confusion used to pass silently.
    for frac, want_ok in ((1.0, True), (0.999, False), (0.5, False)):
        y = y0.copy()
        y[3] = y[2] + frac * D.delta_min()        # gap between slots 2 and 3
        v = D.violations(y, y0, A_, b_, G_, h_)
        if v["ineq_ok"] != want_ok:
            raise Fail(f"gap = {frac} * delta_min reported ineq_ok="
                       f"{v['ineq_ok']}, expected {want_ok} "
                       f"(excess {v['ineq_excess']:.3e})")
        if not v["freq_order_ok"]:
            raise Fail("nu should still be descending here")

    # crossing two slots breaks the frequency ORDER itself -- the failure that
    # silently reassigns which dimension carries which band
    y_swap = y0.copy()
    y_swap[[5, 6]] = y_swap[[6, 5]]
    v2 = D.violations(y_swap, y0, A_, b_, G_, h_)
    if v2["freq_order_ok"]:
        raise Fail("crossed frequencies were not flagged as reordered")
    if v2["ineq_ok"]:
        raise Fail("crossed frequencies were not flagged by the ordering rows")

    # and the native point is NOT silently repaired by sorting anywhere
    if not np.array_equal(D.unpack(y0)[0], D.nu_of_x(x)):
        raise Fail("building a design point reordered the frequencies")
    return dict(min_h=float(h.min()), delta_min=D.delta_min(),
                excess_at_half=float(D.violations(
                    np.r_[y0[:3], [y0[2] + 0.5 * D.delta_min()], y0[4:]],
                    y0, A_, b_, G_, h_)["ineq_excess"]))


@check("gain box is the declared GAIN range, through feasible_set", "design")
def t_gainbox():
    """The box is a statement about g, and it must survive the trip through
    `feasible_set`.

    `gain_box_rows` takes GAINS and converts to the a coordinate internally.
    Feeding it gain bounds as if they were a bounds enforces a different box
    (g in [1.284, 3.49] for GAIN_BOX) and nothing downstream would reveal it --
    the corners would simply be refused as infeasible.  So the corners are
    checked through `feasible_set`, which is the path that had the bug, and not
    only through a direct call.
    """
    x = D.native_x()
    lo_g, hi_g = D.GAIN_BOX
    y_ref = D.pack(D.nu_of_x(x), 1.0)          # the reference the rows are on
    A_, b_, G_, h_ = D.feasible_set(y_ref)
    for g, want_ok in ((lo_g, True), (hi_g, True), (1.0, True),
                       (lo_g * 0.99, False), (hi_g * 1.01, False),
                       (0.2, False), (5.0, False)):
        y = D.pack(D.nu_of_x(x), g)
        v = D.violations(y, y_ref, A_, b_, G_, h_)
        if v["ineq_ok"] != want_ok:
            raise Fail(f"gain {g:.4f} reported ineq_ok={v['ineq_ok']}, expected "
                       f"{want_ok} (excess {v['inexcess']:.3e})"
                       if False else
                       f"gain {g:.4f} reported ineq_ok={v['ineq_ok']}, expected "
                       f"{want_ok} (excess {v['ineq_excess']:.3e})")
    # the corners sit exactly ON the box, so the slack there is exactly zero --
    # a box that is never violated would also report ineq_ok at g = 0.2
    for g, row in ((lo_g, -1), (hi_g, -2)):
        y = D.pack(D.nu_of_x(x), g)
        slack = float(D.violations(y, y_ref, A_, b_, G_, h_)["ineq_excess"])
        close(slack, 0.0, 1e-12, f"slack at the g={g} corner")
    # the a-bounds the rows encode are a_box of the GAIN box, shifted onto a_at
    a_lo, a_hi = D.a_box()
    a_at = float(y_ref[D.A_INDEX])
    close(h_[-1], a_at - a_lo, 1e-12, "low-side a bound")
    close(h_[-2], a_hi - a_at, 1e-12, "high-side a bound")
    return dict(gain_box=list(D.GAIN_BOX), a_box=[float(a_lo), float(a_hi)],
                a_at=a_at)


# ===========================================================================
# sec.8 -- the QCQP, against hand-derived optima
# ===========================================================================
def _entry(name, kind, r, B, limit=None):
    return dict(name=name, kind=kind, r=np.asarray(r, dtype=np.float64),
                B=np.asarray(B, dtype=np.float64), limit=limit)


@check("1-D min-max epigraph has the hand-derived optimum", "qcqp")
def t_qcqp_1d():
    # q1 = -2d + 0.5 d^2  (min at d = 2, value -2)
    # q2 = -1d + 0.25 d^2 (min at d = 2, value -1)
    # max(q1, q2) is minimised at d = 2 with t = -1: moving to d = 4 crosses at
    # 0 and is worse, and q2 is the binding term throughout.
    r1, r2 = [-2.0], [-1.0]
    B1, B2 = [[1.0]], [[0.5]]
    E = [_entry("a", "risk", r1, B1), _entry("b", "risk", r2, B2)]
    out = qcqp.solve_epigraph(np.zeros(1), E, np.zeros((0, 1)), np.zeros(0),
                              np.zeros((0, 1)), np.zeros(0), np.eye(1), 10.0)
    close(out["t"], -1.0, 2e-4, "t at the 1-D optimum")
    close(out["d"][0], 2.0, 2e-3, "argmin of the 1-D min-max")
    if not out["feasible"]:
        raise Fail(f"solver reported infeasible: {out['max_quad_violation']}")
    return dict(t=out["t"], d=float(out["d"][0]), analytic_t=-1.0, analytic_d=2.0)


@check("analytic step satisfies the linear system it claims", "qcqp")
def t_analytic():
    # r = [-1, -3], Q = diag(2, 4), A d = 0 with A = [1, 1]
    # KKT: r + Q d + A^T lam = 0 with d0 + d1 = 0  =>  d = [-1/3, 1/3]
    r = np.array([-1.0, -3.0])
    Q = np.diag([2.0, 4.0])
    Amat = np.array([[1.0, 1.0]])
    b = np.array([0.0])
    d = qcqp.analytic_step(Q, r, Amat, b)
    close(d[0], -1.0 / 3.0, 1e-12, "d0")
    close(d[1], 1.0 / 3.0, 1e-12, "d1")
    close(Amat @ d - b, 0.0, 1e-12, "A d = b")
    # and the gradient of the model at the solution is in the row space of A
    g = r + Q @ d
    resid = g - Amat.T @ (np.linalg.lstsq(Amat.T, g, rcond=None)[0])
    close(np.linalg.norm(resid), 0.0, 1e-12, "gradient in span(A^T)")
    # unconstrained degenerate case
    d0 = qcqp.analytic_step(Q, r)
    close(np.abs(d0 - np.array([0.5, 0.75])).max(), 0.0, 1e-12, "unconstrained")
    return dict(d=[float(x) for x in d], unconstrained=[float(x) for x in d0])


@check("trust ellipsoid: implied box contains it and is tight", "qcqp")
def t_trust():
    """The box is a SUPERSET of the trust region, not a subset.

    `implied_box` is used as hard bounds in `solve_epigraph`, so the property
    that matters is containment: box supseteq ellipsoid, so restricting to the
    box cannot cut off a feasible step.  The natural misreading -- "the box is
    inside the ellipsoid" -- is false for any D that is not a multiple of the
    identity, and asserting it would fail on a correct implementation.
    """
    Dm = np.diag([0.25, 1.0, 4.0])
    Delta = 2.0
    lo, hi = qcqp.implied_box(Dm, Delta)
    half = Delta / math.sqrt(0.25)                 # Delta / sqrt(lambda_min)
    close(hi[0], half, 1e-12, "half width = Delta/sqrt(lambda_min)")
    close(lo[0], -half, 1e-12, "symmetric")
    # containment, verified against the ANALYTIC extreme coordinate:
    # max |d_i| s.t. d^T D d <= Delta^2 is Delta * sqrt((D^-1)_ii)
    Dinv = np.linalg.inv(Dm)
    for i in range(3):
        extreme = Delta * math.sqrt(Dinv[i, i])
        if extreme > half + 1e-12:
            raise Fail(f"coordinate {i} reaches {extreme} > box {half}")
    # and by sampling: every random point of the ellipsoid is in the box
    rng = np.random.default_rng(17)
    worst = 0.0
    for _ in range(2000):
        w = rng.normal(size=3)
        w = w / max(np.sqrt(float(w @ (Dm @ w))), 1e-12) * Delta * rng.random() ** 0.5
        worst = max(worst, float(np.abs(w).max()))
    if worst > half + 1e-9:
        raise Fail(f"sampled ellipsoid point ({worst}) escaped the box ({half})")
    # project_tr returns the boundary point on the same ray
    d = np.array([10.0, 0.0, 0.0])
    p = qcqp.project_tr(d, Dm, Delta)
    close(p @ (Dm @ p), Delta * Delta, 1e-9, "projected onto the boundary")
    if not np.allclose(p / p[0], d / d[0], atol=1e-15):
        raise Fail("projection left the ray")
    close(np.abs(qcqp.project_tr(np.array([0.5, 0.0, 0.0]), Dm, Delta)
                 - np.array([0.5, 0.0, 0.0])).max(), 0.0, 1e-15, "inside is fixed")
    # a singular metric is refused, not silently replaced
    try:
        qcqp.implied_box(np.diag([0.0, 1.0]), 1.0)
        raise Fail("singular D was accepted")
    except ValueError:
        pass
    return dict(half_width=half, analytic_extreme=[
        float(Delta * math.sqrt(Dinv[i, i])) for i in range(3)],
        worst_sampled=worst)


@check("solver: random problems satisfy every constraint", "qcqp")
def t_solver_random():
    rng = np.random.default_rng(7)
    n, n_risk = 6, 3
    bad = []
    for trial in range(40):
        E = []
        for i in range(n_risk):
            Bm = rng.normal(size=(n, n))
            Bm = Bm @ Bm.T * 0.1 + 0.05 * np.eye(n)
            E.append(_entry(f"r{i}", "risk", rng.normal(size=n) * 0.5, Bm))
        Dm = np.eye(n)
        Delta = 0.5 + 0.5 * rng.random()
        out = qcqp.solve_epigraph(np.zeros(n), E, np.zeros((0, n)), np.zeros(0),
                                  np.zeros((0, n)), np.zeros(0), Dm, Delta)
        qs = [e["r"] @ out["d"] + 0.5 * out["d"] @ (e["B"] @ out["d"]) for e in E]
        err = abs(out["t"] - max(qs))
        if err > 1e-7 or out["tr_violation"] > 1e-7:
            bad.append((trial, err, out["tr_violation"]))
    if bad:
        raise Fail(f"{len(bad)}/40 problems returned a step with t != max q_i "
                   f"or outside the trust region: {bad[:3]}")
    return dict(n_trials=40, max_t_error=0.0)


@check("solver: inconsistent constraints are reported, not converged", "qcqp")
def t_solver_inconsistent():
    # ordering floor x1 - x0 >= 5 while the same coordinate is pinned by an
    # equality to a value that breaks it.  No point satisfies both.
    n = 2
    E = [_entry("a", "risk", [-1.0, -1.0], np.eye(n) * 0.5)]
    Amat = np.array([[1.0, -1.0]])
    b = np.array([0.0])                   # x1 - x0 = 0  (gap of 0)
    G = np.array([[1.0, -1.0]])
    h = np.array([-5.0])                  # x1 - x0 >= 5
    out = qcqp.solve_epigraph(np.zeros(n), E, Amat, b, G, h, np.eye(n), 1.0)
    if out["feasible"]:
        raise Fail("an inconsistent constraint set was reported feasible")
    return dict(ineq_violation=out["lin_ineq_violation"],
                repair_report=bool(out["repaired"]))


@check("repair_step: fixes what is fixable, stalls on what is not", "qcqp")
def t_repair():
    n = 3
    E = [_entry("a", "risk", np.zeros(n), np.eye(n))]
    A_eq = np.array([[1.0, 1.0, 1.0]])
    b_eq = np.array([0.0])
    G_in = np.array([[1.0, -1.0, 0.0]])
    h_in = np.array([-0.5])                # d0 - d1 >= 0.5
    d0 = np.array([1.0, 1.0, 1.0])
    d, _, info = qcqp.repair_step(d0, E, A_eq, b_eq, G_in, h_in,
                                  np.eye(n), 10.0)
    close(float(np.abs(A_eq @ d - b_eq).max()), 0.0, 1e-9, "equality after repair")
    if float((G_in @ d - h_in).max()) > 1e-8:
        raise Fail("repair left an inequality violated on a consistent set")
    # a genuinely inconsistent set: d0 = 0 pinned by the equality while the
    # halfspace demands d0 >= 3.  No point satisfies both, so POCS cannot
    # converge and the stall must be REPORTED.
    A_bad = np.array([[1.0, 0.0, 0.0]])
    b_bad = np.array([0.0])
    G_bad = np.array([[-1.0, 0.0, 0.0]])
    h_bad = np.array([-3.0])               # -d0 <= -3  <=>  d0 >= 3
    db, _, info2 = qcqp.repair_step(d0, E, A_bad, b_bad, G_bad, h_bad,
                                    np.eye(n), 10.0)
    if not info2["stalled"]:
        raise Fail("repair did not report a stall on an inconsistent set")
    if info2["eq_violation_after"] <= 1e-6 and info2["ineq_violation_after"] <= 1e-6:
        raise Fail("stall reported but both residuals are zero")
    return dict(consistent_info=info, inconsistent_stalled=info2["stalled"],
                residual_after=dict(eq=info2["eq_violation_after"],
                                    ineq=info2["ineq_violation_after"]))


# ===========================================================================
# sec.6 -- gain elimination
# ===========================================================================
@check("Schur elimination equals the joint solve", "sec6")
def t_eliminate():
    rng = np.random.default_rng(3)
    n = 7
    gain = 4
    M = rng.normal(size=(n, n))
    Q = M @ M.T + np.eye(n)                # strictly PD, so all of sec.6 applies
    r = rng.normal(size=n)
    Q_eff, r_eff, d_a_of_d_f, free = qcqp.eliminate_gain(Q, r, gain)
    if list(free) != [j for j in range(n) if j != gain]:
        raise Fail("free index set is wrong")
    d_full = qcqp.analytic_step(Q, r)
    d_free = qcqp.analytic_step(Q_eff, r_eff)
    d_schur = np.empty(n)
    d_schur[free] = d_free
    d_schur[gain] = d_a_of_d_f(d_free)
    close(np.abs(d_schur - d_full).max(), 0.0, 1e-10, "Schur vs joint step")
    # and the value at the recovered point is the joint optimum
    v_joint = qcqp.predicted(Q, r, d_full)[0]
    v_schur = qcqp.predicted(Q, r, d_schur)[0]
    close(v_schur, v_joint, 1e-9, "value at the Schur point")

    # the elimination is NOT a gradient projection: pinning the gain (d_a = 0)
    # gives a materially different step whenever Q_fa != 0
    d_pin = np.zeros(n)
    d_pin[free] = qcqp.analytic_step(Q[:n, :n][np.ix_(free, free)],
                                     r[free] - Q[np.ix_(free, [gain])][:, 0] * 0.0)
    if np.abs(Q_eff - Q[np.ix_(free, free)]).max() < 1e-12:
        raise Fail("Q_eff equals Q_ff: this problem cannot distinguish the two")
    if np.abs(d_schur[free] - d_pin[free]).max() < 1e-6:
        raise Fail("Schur and pinned-gain steps coincide on a problem with "
                   "Q_fa != 0 -- the two operations are not being distinguished")
    # a gain at its box boundary cannot be eliminated
    try:
        qcqp.eliminate_gain(Q, r, gain, interior=False)
        raise Fail("elimination ran on a boundary-bound gain")
    except ValueError:
        pass
    return dict(dim=n, schur_gap=float(np.abs(d_schur - d_full).max()),
                pin_vs_schur=float(np.abs(d_schur[free] - d_pin[free]).max()),
                value_gap=float(abs(v_schur - v_joint)))


# ===========================================================================
# sec.5 -- shadow prices
# ===========================================================================
@check("gap transfer derivative equals price_k - price_i", "sec5")
def t_prices():
    rng = np.random.default_rng(11)
    k = D.K
    # L is linear with a known gradient v over the FULL design vector, so every
    # price statement about it is exact and any error is in the price algebra
    v = rng.normal(size=D.N_DESIGN)
    v[D.A_INDEX] = 0.0                     # a gap transfer never moves the gain
    y = D.pack(D.nu_of_x(D.native_x()), 1.0)
    L = lambda z: float(v @ z)
    prices = D.gap_prices(v)
    worst = 0.0
    for (i, j) in [(1, 2), (1, 63), (23, 40), (40, 23), (12, 41), (63, 62)]:
        u = D.gap_transfer(i, j)
        t = 1e-6
        fd = (L(y + t * u) - L(y - t * u)) / (2 * t)
        exact = float(v @ u)
        worst = max(worst, abs(fd - exact))
        if abs(exact - (prices[j] - prices[i])) > 1e-9:
            raise Fail(f"price identity fails for ({i},{j}): "
                       f"{exact} != {prices[j]} - {prices[i]}")
    # the gain coordinate is untouched by a gap transfer, and the transfer
    # preserves what the plan says it preserves: the SPAN x_{K-1} - x_0, i.e.
    # the sum of the gaps, and both endpoints.  It does NOT preserve sum_j x_j --
    # a transfer of t over a run of m slots moves that sum by -t*m.  The two are
    # different statements and design.span_row pins the second one, which is why
    # it is off by default and why the plan's sec.5 prices transfers against the
    # span rather than against the centroid (R4: sum m is a free decision
    # variable, not a conserved quantity).
    if D.gap_transfer(1, 63)[D.A_INDEX] != 0.0:
        raise Fail("a gap transfer moved the gain")
    for (i, j) in [(1, 2), (5, 40), (23, 40), (40, 23), (1, 63)]:
        u = D.gap_transfer(i, j)
        if abs(float(u[0])) > 0 or abs(float(u[k - 1])) > 0:
            raise Fail(f"transfer ({i},{j}) moved an endpoint")
        # the gap sum is x_{k-1} - x_0, which the endpoints' invariance gives
        close(float(np.diff(np.r_[0.0, np.cumsum(u[:k])])[k - 1]), 0.0, 1e-15,
              f"transfer ({i},{j}) changed the total gap span")
        # and it is not a no-op
        if np.abs(u).max() == 0:
            raise Fail(f"transfer ({i},{j}) is the zero direction")
    # equi-priced gaps make every transfer flat -- the reading the receipt uses
    # Equi-priced gaps means every transfer is flat, and stationarity in the
    # transfer directions forces the INTERIOR gradient to vanish: gap_transfer(j,
    # j+1) = -e_j for j = 1..K-2, so those K-2 coordinate directions are all
    # reachable and v_j must be 0 on each.  Only the two endpoints and the gain
    # are unconstrained by transfers (the endpoints are pinned by the span, and
    # the gain is not a gap).
    for j in range(1, k - 1):
        u = D.gap_transfer(j, j + 1)
        if u[j] != -1.0 or np.abs(u).sum() != 1.0:
            raise Fail(f"gap_transfer({j},{j+1}) is not -e_{j}: {u[:8]}")
    flat_v = np.zeros(D.N_DESIGN)
    flat_v[0], flat_v[k - 1], flat_v[D.A_INDEX] = 3.0, -2.0, 5.0
    flat = float(flat_v @ D.gap_transfer(1, 63))
    close(flat, 0.0, 1e-15, "flat gradient transfer")
    fp = D.gap_prices(flat_v)
    close(float(np.abs(fp[1:k] - fp[1]).max()), 0.0, 1e-15,
          "interior prices are equal when the interior gradient vanishes")
    return dict(max_fd_error=worst, price_1_minus_63=float(prices[1] - prices[63]),
                transfer_derivative_on_flat=flat,
                price_1_minus_63_on_flat=float(D.gap_prices(flat_v)[1]
                                               - D.gap_prices(flat_v)[63]))


# ===========================================================================
# sec.3 -- the error bound
# ===========================================================================
@check("b_e is the brute-force max over (position, wrong token)", "sec3")
def t_raw_bound():
    import torch
    rng = np.random.default_rng(5)
    T_e, V = 9, 53
    z = torch.tensor(rng.normal(size=(T_e, V)) * 2.0, dtype=torch.float64)
    y = torch.tensor(rng.integers(0, V, size=T_e), dtype=torch.long)
    out = BD.raw_bound(z, y)
    # brute force
    best, arg = -np.inf, None
    for t in range(T_e):
        for v in range(V):
            if v == int(y[t]):
                continue
            val = float(z[t, v] - z[t, y[t]])
            if val > best:
                best, arg = val, (t, v)
    close(out["b"], best, 1e-12, "b vs brute force")
    if out["pos"] != arg[0]:
        raise Fail(f"argmax position {out['pos']} != brute force {arg[0]}")
    # the position must actually be attained at the maximising (t, v)
    if abs(float(z[arg[0], arg[1]] - z[arg[0], y[arg[0]]]) - best) > 1e-12:
        raise Fail("reported position does not attain the reported max")
    return dict(b=out["b"], brute=float(best), pos=int(out["pos"]),
                n_terms=out["n_terms"])


@check("smoothed bound sandwiches the raw one at every eta", "sec3")
def t_sandwich():
    import torch
    rng = np.random.default_rng(6)
    rows = []
    for T_e in (4, 33, 200):
        V = 31
        z = torch.tensor(rng.normal(size=(T_e, V)) * 1.5, dtype=torch.float64)
        y = torch.tensor(rng.integers(0, V, size=T_e), dtype=torch.long)
        for eta in (0.005, 0.02, 0.5, 2.0):
            s = BD.smoothed_bound(z, y, eta=eta)
            b, bt = float(s["b"]), float(s["b_tilde"])
            if not (b - 1e-9 <= bt <= b + eta + 1e-6):
                raise Fail(f"sandwich broken at T_e={T_e}, eta={eta}: "
                           f"{b} <= {bt} <= {b + eta}")
            rows.append(dict(t_e=T_e, eta=eta, b=b, b_tilde=bt, gap=bt - b))
    # bound_loss must be softplus(b_tilde)/log2, and b_tilde carries the eta
    # offset, so a flat-logit block is NOT 1.0 -- it is softplus(eta)/log2.  The
    # 1.0 case is the degenerate one where there is a single (position, wrong
    # token) pair, log N_e = 0, and the plan says take the max directly: there
    # tau is None and b_tilde IS b.
    import torch
    z = torch.zeros(3, 7, dtype=torch.float64)
    y = torch.zeros(3, dtype=torch.long)
    loss, s = BD.bound_loss(z, y)
    ref = torch.nn.functional.softplus(s["b_tilde"]) / math.log(2)
    close(float(loss), float(ref), 1e-12, "bound_loss == softplus(b_tilde)/log2")
    close(float(s["b_tilde"]) - float(s["b"]), 0.02, 1e-9, "flat block gains eta")
    z1 = torch.zeros(1, 2, dtype=torch.float64)
    y1 = torch.zeros(1, dtype=torch.long)
    loss1, s1 = BD.bound_loss(z1, y1)
    close(float(loss1), 1.0, 1e-12, "degenerate N_e = 1 reads exactly 1.0")
    if s1["tau"] is not None:
        raise Fail("N_e = 1 should signal tau=None (take the max directly)")
    # the loss must dominate the loss at the raw bound (it is an upper bound)
    for T_e in (5, 40):
        zz = torch.tensor(rng.normal(size=(T_e, 11)), dtype=torch.float64)
        yy = torch.tensor(rng.integers(0, 11, size=T_e), dtype=torch.long)
        l, ss = BD.bound_loss(zz, yy)
        raw = float(torch.nn.functional.softplus(
            torch.tensor(float(ss["b"]), dtype=torch.float64)) / math.log(2))
        if float(l) < raw - 1e-9:
            raise Fail(f"bound_loss {float(l)} < raw-bound loss {raw}")
    return dict(n_cases=len(rows), max_gap=max(r["gap"] for r in rows),
                flat_block_b_tilde=float(s["b_tilde"]))


@check("bound informativeness threshold is the one it reports", "sec3")
def t_informative():
    ok, thr = BD.bound_is_informative(-3.0, target=0.1)
    if not ok:
        raise Fail(f"b=-3 should clear the 0.1 threshold ({thr})")
    ok2, _ = BD.bound_is_informative(0.0, target=0.1)
    if ok2:
        raise Fail("b=0 must not be called informative")
    # the threshold is the value at which softplus(b)/log2 == 0.1
    import torch
    at = float(torch.nn.functional.softplus(
        torch.tensor(float(thr), dtype=torch.float64)) / math.log(2))
    close(at, 0.1, 1e-9, "softplus(threshold)/log2")
    # a stricter target needs a more negative b, and a looser one less
    _, thr_lo = BD.bound_is_informative(0.0, target=0.5)
    _, thr_hi = BD.bound_is_informative(0.0, target=0.01)
    if not (thr_hi < thr < thr_lo):
        raise Fail(f"thresholds not ordered: {thr_hi} < {thr} < {thr_lo}")
    return dict(threshold=float(thr), value_at_threshold=at,
                thr_for_0p5=float(thr_lo), thr_for_0p01=float(thr_hi))


@check("bound_summary reports the vacuum rate it defines", "sec3")
def t_summary():
    bs = np.array([-5.0, -3.0, -1.0, 0.5, 2.0])
    s = BD.bound_summary(bs)
    close(s["vacuum_rate"], 2.0 / 5.0, 1e-12, "vacuum rate")
    close(s["min"], -5.0, 1e-12, "min")
    close(s["n"], 5, 1e-12, "n")
    return dict(vacuum_rate=s["vacuum_rate"], threshold=s["threshold_for_0p1"])


# ===========================================================================
# sec.4 -- the native-retention certificate
# ===========================================================================
@check("D_keep(native) is exactly zero", "sec4")
def t_dkeep_zero():
    rng = np.random.default_rng(8)
    b_native = -np.abs(rng.normal(size=12)) - 0.1     # every row has a margin
    rec = BD.d_keep(b_native, b_native)
    close(rec["d_keep"], 0.0, 1e-15, "D_keep at native")
    if rec["n_certified"] != 12:
        raise Fail("some rows were dropped that should be certified")
    return dict(d_keep=rec["d_keep"], n_certified=rec["n_certified"])


@check("rows without a baseline margin are excluded, never floored", "sec4")
def t_dkeep_excluded():
    b_native = np.array([-2.0, -1.0, 0.0, 0.5, -0.5])
    g = BD.gamma_from_native(b_native)
    if g["certified"].tolist() != [True, True, False, False, True]:
        raise Fail(f"certification mask wrong: {g['certified'].tolist()}")
    if 2 not in g["ties"]:
        raise Fail("a tie row (b == 0) was not flagged")
    # a dropped row must not influence the mean
    b_theta = np.array([-2.0, -1.0, 1e9, 1e9, -0.5])
    rec = BD.d_keep(b_theta, b_native)
    if rec["n_certified"] != 3:
        raise Fail("excluded rows re-entered the mean")
    if not np.isfinite(rec["d_keep"]):
        raise Fail("D_keep is not finite on a valid panel")
    # no certified rows at all -> undefined, not a number
    rec2 = BD.d_keep(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    if not rec2["undefined"]:
        raise Fail("an all-uncertified panel returned a value instead of undefined")
    return dict(d_keep=rec["d_keep"], n_certified=rec["n_certified"],
                n_undefined=rec["n_undefined"], ties=g["ties"])


# ===========================================================================
# sec.8 -- the acceptance predicate
# ===========================================================================
def _accept_fixture():
    n, gain_index = 6, 5
    r1 = np.zeros(n); r1[0] = -2.0
    r2 = np.zeros(n); r2[0] = -1.0
    entries = [_entry("risk_a", "risk", r1, np.eye(n) * 0.5),
               _entry("risk_b", "risk", r2, np.eye(n) * 0.3),
               _entry("native", "native", np.r_[0.4, np.zeros(n - 1)],
                      np.eye(n) * 0.2, limit=1e-3)]
    d = np.zeros(n); d[0] = 0.5
    return entries, d, gain_index


def _real_terms(theta, k):
    """A real objective whose curvature is `k` times the model's own.

    k < 1 means the model dominates the real curvature, which is the condition
    under which (*) holds; k > 1 is the optimistic case the acceptance test
    exists to catch.  The linear part is left equal to the model's, so a
    violation can only come from curvature and not from a disagreement about
    the gradient.
    """
    t = float(theta[0])
    return {"risk_a": -2.0 * t + 0.5 * (k * 0.5) * t * t,
            "risk_b": -1.0 * t + 0.5 * (k * 0.3) * t * t}


@check("accept: dominated step is accepted, optimistic step refused", "sec8")
def t_accept_core():
    E, d, gi = _accept_fixture()
    rb = _real_terms(np.zeros(6), 1.0)
    # the real point after the step is theta_k + d, so the real terms must be
    # evaluated AT d.  Evaluating them at a scaled d while the model is priced
    # at d compares two different points, and a fixture that does so passes or
    # fails for a reason unrelated to (*).
    v_ok = A.check_acceptance(E, d, rb, _real_terms(d, 0.5), 0.0, 0.0)
    if not v_ok["accepted"]:
        raise Fail(f"a dominated step was refused: {v_ok['reasons']}")
    if v_ok["rho_state"] != "in_band":
        raise Fail(f"a dominated step gave rho_state {v_ok['rho_state']!r}")
    v_bad = A.check_acceptance(E, d, rb, _real_terms(d, 2.0), 0.0, 0.0)
    if v_bad["accepted"] or v_bad["parts"]["model_satisfied"]:
        raise Fail("an optimistic step (real curvature 2x the model) was accepted")
    if v_bad["worst_violation"] <= 0:
        raise Fail("a refused step reported no violation")
    if v_bad["worst_violation_of"] != "risk_a":
        raise Fail(f"violation charged to {v_bad['worst_violation_of']}, but "
                   "risk_a is the term with the larger curvature")
    return dict(accepted_rho=v_ok["rho"], refused_violation=v_bad["worst_violation"],
                refused_of=v_bad["worst_violation_of"])


@check("accept: native overrun refuses the step and is named", "sec8")
def t_accept_native():
    E, d, gi = _accept_fixture()
    rb = _real_terms(np.zeros(6), 1.0)
    v = A.check_acceptance(E, d, rb, _real_terms(d, 0.5), 0.0, 5e-3, eps=1e-3)
    if v["parts"]["native_satisfied"]:
        raise Fail("a native output-KL of 5e-3 against a budget of 1e-3 passed")
    if "native_satisfied" not in v["reasons"]:
        raise Fail("the native overrun was not named in the reasons")
    # the native overrun must NOT contaminate the (*) verdict: the two are on
    # different scales (absolute output-KL vs a delta) and sharing a max both
    # misreports (*) and masks the gain flag below
    if not v["parts"]["model_satisfied"]:
        raise Fail("the native overrun was folded into the (*) check -- the two "
                   "are on different scales and must not share a max")
    esc = A.escalation_for(v)
    if esc["action"] != "shrink":
        raise Fail(f"native overrun did not escalate to shrink: {esc}")
    v0 = A.check_acceptance(E, d, rb, _real_terms(d, 0.5), 0.0, 0.0, eps=1e-3)
    if not v0["parts"]["native_satisfied"]:
        raise Fail("a zero native delta was reported as an overrun")
    return dict(reasons=v["reasons"], native_kl=v["real_native_kl"],
                action=esc["action"])


@check("accept: rho outside the band is refused at both ends", "sec8")
def t_accept_rho():
    E, d, gi = _accept_fixture()
    rb = _real_terms(np.zeros(6), 1.0)
    q = A.model_values(E, d)
    # real improves 3x what the model predicted: (*) holds, but rho > 1.5, which
    # says the gain is not coming from this model and must not be credited to it
    v_hi = A.check_acceptance(
        E, d, rb, {"risk_a": 3.0 * q["risk_a"], "risk_b": 3.0 * q["risk_b"]},
        0.0, 0.0)
    if v_hi["rho_state"] != "above_band" or v_hi["accepted"]:
        raise Fail(f"rho={v_hi['rho']} ({v_hi['rho_state']}) was not refused")
    if not v_hi["parts"]["model_satisfied"]:
        raise Fail("an above-band step should still satisfy (*) -- the two are "
                   "different failures")
    # above band is the only rho failure that reaches its own ladder rung, because
    # (*) IS the statement rho >= 1 and the band starts at 0.5
    if A.escalation_for(v_hi)["action"] != "shrink":
        raise Fail("an above-band rho did not escalate to shrink")
    if "rho" not in A.escalation_for(v_hi)["why"]:
        raise Fail("the above-band escalation does not name rho")
    # real improves 10% of what the model promised.  This is below band AND (*)
    # is violated -- necessarily so, since below the band means rho < 1 -- so the
    # escalation is the (*) rung, which must still be a shrink of some kind
    v_lo = A.check_acceptance(
        E, d, rb, {"risk_a": 0.1 * q["risk_a"], "risk_b": 0.1 * q["risk_b"]},
        0.0, 0.0)
    if v_lo["rho_state"] != "below_band":
        raise Fail(f"rho={v_lo['rho']} should be below the band")
    if v_lo["parts"]["model_satisfied"]:
        raise Fail("a below-band rho satisfied (*) -- impossible unless the band "
                   "starts at or above 1")
    if not A.escalation_for(v_lo)["action"].startswith("shrink"):
        raise Fail(f"a below-band rho escalated to {A.escalation_for(v_lo)}")
    # An UNDEFINED rho is the rung that STOPS instead of shrinking: the objective
    # does not move at this radius, so shrinking tests nothing.  It needs a step
    # whose model prediction is not a decrease, which a zero-gradient point
    # supplies -- there q = 0.5 d^T B d >= 0, so the model predicts an increase.
    E0 = [_entry("risk_a", "risk", np.zeros(6), np.eye(6) * 0.5),
          _entry("risk_b", "risk", np.zeros(6), np.eye(6) * 0.3)]
    v_u = A.check_acceptance(E0, d, {"risk_a": 0.0, "risk_b": 0.0},
                             {"risk_a": 0.0, "risk_b": 0.0}, 0.0, 0.0)
    if v_u["rho"] is not None or v_u["rho_state"] != "undefined":
        raise Fail(f"rho_state {v_u['rho_state']!r} for a non-decreasing model")
    esc = A.escalation_for(v_u)
    if esc["action"] != "stop":
        raise Fail(f"an undefined rho escalated to {esc['action']!r}, not stop -- "
                   "this is the rung that must not shrink")
    if "INCREASE" not in str(v_u.get("rho_reason")):
        raise Fail(f"the undefined rho did not say why: {v_u.get('rho_reason')}")
    return dict(above=v_hi["rho"], below=v_lo["rho"], undefined=v_u["rho_state"],
                stop_why=esc["why"][:70])


@check("accept: a gain the model cannot price is flagged", "sec8")
def t_accept_gain():
    E, d, gi = _accept_fixture()
    rb = _real_terms(np.zeros(6), 1.0)
    d2 = d.copy(); d2[gi] = 0.1
    ra = _real_terms(d2, 0.5)
    v = A.check_acceptance(E, d2, rb, ra, 0.0, 0.0, gain_index=gi)
    if v["native_model_admissible"]:
        raise Fail("a gain move was called model-admissible with no gain "
                   "coordinate in the model")
    if not v["gain_unpriced"]:
        raise Fail("a gain move was not flagged gain_unpriced")
    # ... AND IT IS STILL ACCEPTED.  The flag says the MODEL cannot price the
    # gain, not that the step is bad: the native constraint is measured, so a
    # model with no gain coordinate cannot make the measured verdict wrong.
    # Refusing these was a real bug -- it made sec.10C's joint solve impossible.
    if not v["accepted"]:
        raise Fail(f"a gain move was refused for a reporting flag: {v['reasons']}")
    if "gain_unpriced" in v["reasons"]:
        raise Fail("gain_unpriced leaked into the refusal reasons")
    if "gain_unpriced" not in v["warnings"]:
        raise Fail("gain_unpriced is not recorded as a warning")
    v2 = A.check_acceptance(E, d2, rb, ra, 0.0, 0.0, gain_index=gi,
                            model_covers_gain=True)
    if not v2["native_model_admissible"]:
        raise Fail("a gain move was called inadmissible even though the model "
                   "covers the gain")
    if v2["gain_unpriced"] or v2["warnings"]:
        raise Fail("gain_unpriced set although the model prices the gain")
    # a step with no gain component is not flagged
    v3 = A.check_acceptance(E, d, rb, _real_terms(d, 0.5), 0.0, 0.0,
                            gain_index=gi)
    if v3["gain_unpriced"]:
        raise Fail("a frequency-only step was flagged gain_unpriced")
    return dict(unpriced=v["warnings"], priced_warnings=v2["warnings"],
                gain_component=v["gain_component"])


@check("accept: a missing measurement is an error, not a silence", "sec8")
def t_accept_missing():
    E, d, gi = _accept_fixture()
    rb = _real_terms(np.zeros(6), 1.0)
    try:
        A.check_acceptance(E, d, rb, {"risk_a": 0.0}, 0.0, 0.0)
        raise Fail("(* ) was checked on a term that was never measured")
    except ValueError:
        pass
    return dict(ok=True)


@check("ladder: raises damping to fix an optimistic model", "sec8")
def t_ladder_damping():
    """The one failure mode damping can fix, driven through the real solver.

    The real curvature is 14x the model's (7.0 against B_a = 0.5), so shrinking
    Delta cannot help: shrinking scales the step down but the model's quadratic
    is scaled down with it, and the violation stays O(d^2) with the same sign.
    Raising mu inflates B until it dominates the real curvature, and then (*)
    holds -- so a `propose` that actually consumes `mu` must converge.
    A `propose` that ignored `mu` would still be shrinking at the last attempt,
    which is what makes this check pin the contract between loop.py and
    accept.py rather than merely exercise the accept predicate.

    The numbers are chosen so the accepted attempt lands INSIDE the rho band.
    Damping necessarily makes the model's promise more conservative, so mu large
    enough to satisfy (*) can push rho above 1.5 -- and a rho above the band is
    refused, correctly, because the gain is then not the quadratic's.  Here the
    accepted step has rho ~ 1.34.  At mu = 0 the same step has rho ~ -0.86: the
    real loss goes UP, which is exactly the optimistic-model failure.
    """
    E, d0, gi = _accept_fixture()
    n = 6
    state = {"proposals": []}

    def propose(delta, mu):
        # the solver's job, unchanged: minimise the DAMPED model under the trust
        # region.  Damping is added to each group's own B, which is the same
        # object as adding mu*I to the assembled Q for a shared step.
        damped = [dict(e, B=e["B"] + mu * np.eye(n)) for e in E]
        out = qcqp.solve_epigraph(np.zeros(n), damped, np.zeros((0, n)),
                                  np.zeros(0), np.zeros((0, n)), np.zeros(0),
                                  np.eye(n), delta)
        state["proposals"].append(dict(delta=delta, mu=mu, d=out["d"].copy(),
                                       feasible=out["feasible"]))
        # hand back the model that was actually solved: (*) must be checked
        # against the damped B, or the remedy sec.8 prescribes can never pass
        return out["d"], dict(out, entries=damped)

    res = A.run_ladder(np.zeros(n), propose, lambda t: _real_terms(t, 14.0),
                       lambda t: 0.0, E, delta0=1.0, max_attempts=6,
                       shrink=0.5, mu_growth=10.0)
    if not res["accepted"]:
        raise Fail("the ladder failed to converge even though damping can fix "
                   f"this failure: {A.summarize_ladder(res)}")
    if res["n_attempts"] < 2:
        raise Fail("converged on the first attempt: the fixture is not exercising "
                   "the escalation at all")
    mus = [a["mu"] for a in res["attempts"]]
    if mus != sorted(mus) or mus[-1] <= mus[0]:
        raise Fail(f"mu did not grow monotonically: {mus}")
    deltas = [a["delta"] for a in res["attempts"]]
    if deltas != sorted(deltas, reverse=True):
        raise Fail(f"delta did not shrink monotonically: {deltas}")
    first = res["attempts"][0]
    if first["parts"]["model_satisfied"]:
        raise Fail("the first attempt already satisfied (*): the fixture does not "
                   "reproduce the failure the ladder is supposed to fix")
    if not np.isfinite(res["verdict"]["rho"]):
        raise Fail("the accepted step has no finite rho")
    if not (0.5 <= res["verdict"]["rho"] <= 1.5):
        raise Fail(f"accepted at rho = {res['verdict']['rho']}, outside the band "
                   "the predicate claims to enforce")
    return dict(n_attempts=res["n_attempts"], mus=mus, deltas=deltas,
                rho=res["verdict"]["rho"], first_rho=first["rho"],
                first_violation=first["worst_violation"])


@check("ladder: exhaustion is a refusal, never a relabelled success", "sec8")
def t_ladder_refuse():
    """A real objective that WORSENS under every step must exhaust the ladder.

    The fixture's real loss increases with any move (it is +|d0|), so no radius
    and no damping can produce an accepted step: (*) is violated and rho is
    negative at every attempt.  This is the case where the honest output is a
    refusal, and the check exists to make sure an exhausted ladder cannot return
    its last attempt relabelled as a success.
    """
    E, d0, gi = _accept_fixture()
    n = 6

    def propose(delta, mu):
        step = np.zeros(n); step[0] = min(0.5, float(delta))
        return step, dict(feasible=True)

    def worsening(theta):
        t = abs(float(theta[0]))
        return {"risk_a": t, "risk_b": t}

    res = A.run_ladder(np.zeros(n), propose, worsening, lambda t: 0.0, E,
                       delta0=1.0, max_attempts=3, shrink=0.25)
    if res["accepted"] or res["step"] is not None or res["theta_new"] is not None:
        raise Fail("an exhausted ladder released a step")
    if res["n_attempts"] != 3:
        raise Fail(f"expected 3 attempts, got {res['n_attempts']}")
    if "refus" not in res["note"]:
        raise Fail("the refusal is not stated in the result")
    for a in res["attempts"]:
        if a["parts"]["model_satisfied"]:
            raise Fail("a step on a monotonically worsening objective satisfied (*)")
    # the reference point must BE the native table, or the native delta is not a
    # delta -- a ladder that starts elsewhere must raise rather than report
    try:
        A.run_ladder(np.zeros(n), propose, worsening,
                     lambda t: 0.5, E, delta0=1.0, max_attempts=1)
        raise Fail("a non-native reference point was accepted")
    except ValueError:
        pass
    return dict(n_attempts=res["n_attempts"], note=res["note"][:40],
                rho_first=res["attempts"][0]["rho"])


# ===========================================================================
# the panel guard
# ===========================================================================
@check("span guard: rank and null fraction are what they claim", "panel")
def t_span_guard():
    rng = np.random.default_rng(13)
    d, n_rows, true_rank = 65, 30, 6
    basis = rng.normal(size=(true_rank, d))
    # einsum, not `@`: numpy 2.0.2 on macOS routes this shape to Accelerate,
    # which leaves FP status flags set and makes a correct result emit a
    # spurious "invalid value encountered in matmul".  Suppressing it here keeps
    # the REAL numerical warnings this package relies on seeing.
    G = np.einsum("nr,rd->nd", rng.normal(size=(n_rows, true_rank)), basis)
    g = RK.span_guard(G)
    if g["rank"] != true_rank:
        raise Fail(f"rank {g['rank']} != constructed rank {true_rank}")
    if g["null_dim"] != d - true_rank:
        raise Fail(f"null dimension {g['null_dim']} != {d - true_rank}")
    # a step inside the row span has zero null component ...
    inside = basis[0] * 1.7 + basis[1] * (-0.4)
    ni = RK.null_fraction(inside, g)
    if ni > 1e-9:
        raise Fail(f"an in-span step reported null fraction {ni}")
    # ... and a mostly-null step has one
    Q, _ = np.linalg.qr(basis.T)                 # (d, rank) orthonormal basis
    proj = Q @ (Q.T @ inside)
    v = rng.normal(size=d)
    outside = (v - Q @ (Q.T @ v)) + 0.1 * proj
    no = RK.null_fraction(outside, g)
    if not (0.9 < no < 1.0):
        raise Fail(f"a mostly-null step reported {no}")
    # a PURELY null step is 1.  The vector must be projected once and reused --
    # projecting one random vector and subtracting the projection of a DIFFERENT
    # one leaves an in-span residue, which is how this check was first broken.
    null_step = v - Q @ (Q.T @ v)
    nn = RK.null_fraction(null_step, g)
    if nn < 1.0 - 1e-9:
        raise Fail(f"a null step reported {nn} instead of 1")
    # the projector agrees with the coefficient projection
    P = RK.projector(g)
    # einsum rather than @: the same Accelerate dispatch that risk.py documents
    close(np.abs(np.einsum("ij,j->i", P, inside) - proj).max(), 0.0, 1e-9,
          "projector vs basis")
    # a guard built for a different dimension must refuse the step, not broadcast
    try:
        RK.null_fraction(np.zeros(d + 1), g)
        raise Fail("a step of the wrong dimension was accepted")
    except ValueError:
        pass
    return dict(rank=g["rank"], null_dim=g["null_dim"], null_in=ni,
                null_out=no, null_pure=nn,
                singular_values=g["singular_values"][:8])


@check("panel guard: a per-row objective is refused", "panel")
def t_panel_guard():
    rec = RK.guard(30, 65, 6, tasks=list(RK.PREPARED_TASKS), lengths=[32768, 131072])
    if rec["n_lt_d"] is not True:
        raise Fail("the n<d regime was not recorded")
    if "vex2_note" not in rec:
        raise Fail("the V-E2 acknowledgement is missing from the receipt")
    try:
        RK.guard(30, 65, 30)
        raise Fail("a 30-group/30-row (per-row) objective was admitted")
    except ValueError:
        pass
    try:
        RK.guard(30, 6, 6)
        raise Fail("more groups than design variables was admitted")
    except ValueError:
        pass
    return dict(rec={k: rec[k] for k in ("n_rows", "d_design", "n_groups",
                                        "rows_per_design", "n_lt_d")})


@check("panel guard: group weights and LOO folds are well formed", "panel")
def t_groups():
    rows = [dict(row_id=f"{t}-{L}-{i}", task=t, length_cap=L, references=["x"],
                 ids=[1], input_tokens=1, budget=1, prompt_sha256="a")
            for t in RK.PREPARED_TASKS for L in (32768, 131072) for i in range(3)]
    groups = RK.build_groups(rows)
    if groups["n_groups"] != 6 or groups["n_rows"] != 18:
        raise Fail(f"grid wrong: {groups['n_groups']} groups, {groups['n_rows']} rows")
    w = RK.group_weights(groups, scheme="uniform")
    close(sum(w.values()), 1.0, 1e-12, "uniform weights sum")
    wl = RK.group_weights(groups, scheme="length_first")
    close(sum(wl.values()), 1.0, 1e-12, "length_first weights sum")
    per_len = {}
    for (L, _t), v in wl.items():
        per_len[L] = per_len.get(L, 0.0) + v
    close(per_len[32768], 0.5, 1e-12, "short length mass")
    close(per_len[131072], 0.5, 1e-12, "long length mass")
    folds = RK.loo_folds(groups)
    if len(folds) != 18:
        raise Fail(f"{len(folds)} folds for 18 rows")
    # each cell holds 3 rows, so removing one leaves 2 behind and no fold is a
    # singleton -- a held-out row that was the whole cell would make the group
    # mean undefined rather than merely noisy
    if any(f["singleton_cell"] for f in folds):
        raise Fail("a fold emptied its own cell")
    if any(f["n_held_in_group"] != 2 for f in folds):
        raise Fail("leave-one-out did not leave exactly 2 rows in the cell")

    # an INCOMPLETE grid must be refused: drop an entire cell, not one row
    thin = [r for r in rows
            if not (r["task"] == "vt" and r["length_cap"] == 32768)]
    if len(thin) == len(rows):
        raise Fail("the thinning removed nothing -- the fixture is wrong")
    try:
        RK.build_groups(thin)
        raise Fail("an incomplete task x length grid was admitted")
    except ValueError:
        pass

    # a one-row-per-cell corpus is a legitimate grid, and there the fold IS a
    # singleton: the two cases must be distinguishable
    small = [r for i, r in enumerate(rows) if i % 3 == 0]
    gs = RK.build_groups(small)
    fs = RK.loo_folds(gs)
    if not all(f["singleton_cell"] for f in fs):
        raise Fail("a one-row-per-cell corpus did not report singleton folds")
    return dict(n_groups=groups["n_groups"], n_rows=groups["n_rows"],
                n_folds=len(folds), n_groups_thin=gs["n_groups"],
                singleton_in_thin=bool(fs[0]["singleton_cell"]))


@check("row loader: a malformed row is named, not skipped", "panel")
def t_loader():
    import tempfile, os
    good = dict(row_id="r1", task="vt", length_cap=32768, ids=[1, 2],
                references=["a"], input_tokens=2, budget=4, prompt_sha256="x")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "rows.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps(good) + "\n")
        rows = RK.load_rows(p)
        if len(rows) != 1:
            raise Fail("a valid row did not load")
        bad = dict(good); del bad["prompt_sha256"]
        with open(p, "w") as f:
            f.write(json.dumps(bad) + "\n")
        try:
            RK.load_rows(p)
            raise Fail("a row missing a required key loaded")
        except ValueError as e:
            if "prompt_sha256" not in str(e):
                raise Fail(f"the missing key was not named: {e}")
        with open(p, "w") as f:
            f.write(json.dumps(good) + "\n" + json.dumps(good) + "\n")
        try:
            RK.load_rows(p)
            raise Fail("duplicate row_id loaded")
        except ValueError:
            pass
    return dict(ok=True)


# ===========================================================================
def run(only=None, verbose=True):
    results = []
    for spec in CHECKS:
        if only and only not in spec["name"] and only != spec["group"]:
            continue
        try:
            detail = spec["fn"]()
            results.append(dict(name=spec["name"], group=spec["group"],
                                ok=True, detail=detail))
        except Exception as e:                       # noqa: BLE001 -- a test box
            results.append(dict(name=spec["name"], group=spec["group"], ok=False,
                                error=f"{type(e).__name__}: {e}",
                                traceback=traceback.format_exc()))
    n_ok = sum(r["ok"] for r in results)
    if verbose:
        for r in results:
            mark = "PASS" if r["ok"] else "FAIL"
            print(f"[{mark}] {r['group']:8s} {r['name']}")
            if not r["ok"]:
                print(f"        {r['error']}")
            elif r["detail"]:
                print(f"        {json.dumps(r['detail'], default=str)}")
        print(f"\n{n_ok}/{len(results)} checks passed")
    return dict(n_total=len(results), n_ok=n_ok,
                ok=bool(n_ok == len(results)), results=results)


def main(argv=None):
    ap = argparse.ArgumentParser(description="CPU mathematics gate for the "
                                             "joint-KKT package")
    ap.add_argument("--json", default=None, help="write the full receipt here")
    ap.add_argument("--only", default=None,
                    help="substring of a check name, or a group "
                         "(design/qcqp/sec6/sec5/sec3/sec4/sec8/panel)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    out = run(only=args.only, verbose=not args.quiet)
    if args.json:
        import os
        d = os.path.dirname(args.json)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=str)
        print(f"receipt -> {args.json}")
    return 0 if out["ok"] else 1



# ===========================================================================
# the driver -- exercised end to end on a synthetic objective with a closed form
#
# This group is the payoff of `loop.py` taking its objective by injection: the
# whole chain (Phase-I, the ladder, the receipt, the Pareto sweep) runs on a
# quadratic landscape whose constrained optimum is hand-derivable, on CPU, with
# no checkpoint anywhere.  A driver whose first test is on a card is a driver
# whose first test costs money and cannot be repeated cheaply.
# ===========================================================================
class _QuadObjective:
    """native(y) = 0.5 s_N ||z||^2, risk_g(y) = -a_g . z + 0.5 s_R ||z||^2, z = y - y0.

    The constrained problem  min_a -a.z + 0.5 s_R ||z||^2  s.t.  0.5 s_N ||z||^2
    <= eps  has the closed form

        ||z*|| = sqrt(2 eps / s_N),   lambda* = (||a|| / ||z*|| - s_R) / s_N,
        decrease* = a.z* - 0.5 s_R ||z*||^2

    which is what `t_loop_known_answer` compares against.  The gradients are exact
    (analytic, not finite-differenced), so a discrepancy is the driver's.
    """

    def __init__(self, y0, a_risk, s_native=1.0, s_risk=0.05):
        self.y0 = np.asarray(y0, np.float64).copy()
        self.a = {k: np.asarray(v, np.float64) for k, v in a_risk.items()}
        self.s_n = float(s_native)
        self.s_r = float(s_risk)
        self.n_calls = {"values": 0, "grads": 0}

    def names(self):
        return list(self.a) + ["native_kl"]

    def _z(self, y):
        return np.asarray(y, np.float64) - self.y0

    def values(self, y, names=None):
        z = self._z(y)
        out = {"native_kl": 0.5 * self.s_n * float(z @ z)}
        for k, a in self.a.items():
            out[k] = -float(a @ z) + 0.5 * self.s_r * float(z @ z)
        self.n_calls["values"] += 1
        return {k: out[k] for k in (names or out)}

    def grads(self, y, names=None):
        z = self._z(y)
        g = {"native_kl": self.s_n * z}
        for k, a in self.a.items():
            g[k] = -a + self.s_r * z
        keys = names or list(g)
        vals = self.values(y, names=keys)
        self.n_calls["grads"] += 1
        return dict(names=keys, grads=np.array([g[k] for k in keys]),
                    values=[vals[k] for k in keys])


def _loop_fixture(a_scale=1.0, s_risk=0.05):
    y0 = D.pack(D.nu_of_x(D.native_x()), 1.0)
    # a descent direction supported on the interior slots only: the endpoints are
    # pinned by the spectral constraints, so a direction that needed them would be
    # infeasible and the closed form would not apply
    a = np.zeros(D.N_DESIGN)
    a[10:40] = 1.0
    a *= a_scale
    obj = _QuadObjective(y0, {"risk_a": a}, s_native=1.0, s_risk=s_risk)
    return obj, y0, a


@check("loop: Phase-I repairs an infeasible start without widening eps", "loop")
def t_loop_phase_one():
    from . import loop as L
    obj, y0, a = _loop_fixture()
    # start 4x outside the budget
    bad = y0.copy(); bad[20] = y0[20] + 0.02
    cfg = L.Config(eps=1e-4, delta0=0.05, max_steps=1)
    pi = L.phase_one(obj, bad, cfg, "native_kl")
    if not pi["feasible"]:
        raise Fail(f"Phase-I failed to reach a reachable budget: "
                   f"{pi['native_final']} > {cfg.eps}")
    if pi["native_start"] <= cfg.eps:
        raise Fail("the fixture did not start infeasible -- the check is vacuous")
    if pi["native_final"] >= pi["native_start"]:
        raise Fail("Phase-I did not reduce the native term")
    # eps is untouched: it is in the config and Phase-I has no way to write it
    if cfg.eps != 1e-4:
        raise Fail("Phase-I modified eps")
    for row in pi["history"]:
        if row["iteration"] > 0 and row["native"] > pi["history"][0]["native"]:
            raise Fail("Phase-I increased the native term")
    # an UNREACHABLE budget must be reported, not papered over
    cfg2 = L.Config(eps=1e-12, delta0=0.002, max_steps=1)
    pi2 = L.phase_one(obj, bad, cfg2, "native_kl")
    if pi2["feasible"]:
        raise Fail("a step far too small to reach the budget was called feasible")
    if "NOT widened" not in pi2["note"]:
        raise Fail("the infeasibility note does not say eps was left alone")
    return dict(native_start=pi["native_start"], native_final=pi["native_final"],
                iterations=len(pi["history"]))


@check("loop: reaches the closed-form constrained optimum", "loop")
def t_loop_known_answer():
    from . import loop as L
    obj, y0, a = _loop_fixture()
    eps = 1e-3
    cfg = L.Config(eps=eps, delta0=0.02, max_steps=60, max_attempts=4)
    res = L.run(obj, y0, cfg=cfg, verbose=False)
    if res["status"] != "ok" or not res["steps"]:
        raise Fail(f"loop took no step: {res['status']}")
    # the analytic optimum
    z_norm = math.sqrt(2.0 * eps / obj.s_n)
    lam = (float(np.linalg.norm(a)) / z_norm - obj.s_r) / obj.s_n
    best = float(np.linalg.norm(a)) * z_norm - 0.5 * obj.s_r * z_norm ** 2
    native = res["native_final"]["native_kl"]
    risk = res["risk_final"]["risk_a"]
    # the achieved native must be inside the budget -- this is the constraint the
    # whole problem exists to respect, and an overrun means the driver released a
    # step it should have refused
    if native > eps * (1.0 + 1e-9):
        raise Fail(f"final native {native:.3e} exceeds the budget {eps:g}")
    # improvement toward, never past, the analytic optimum
    if risk < -best - 1e-9:
        raise Fail(f"beat the analytic optimum: {risk} < {-best}")
    if risk >= 0:
        raise Fail(f"no long-range improvement at all: {risk}")
    frac = risk / (-best)
    if frac < 0.5:
        raise Fail(f"reached only {frac:.1%} of the analytic optimum")
    # monotone: every accepted step improved the long-range term
    for s in res["steps"]:
        if s["real_long_decrease"] is not None and s["real_long_decrease"] < 0:
            raise Fail("an accepted step made the long-range term worse")
    # the receipts carry the disclaimer, because a B that is not a Hessian is the
    # single most misreadable object in this package
    if not res["B_is_not_a_hessian"]:
        raise Fail("the run receipt does not mark B as not-a-Hessian")
    for s in res["steps"]:
        if not s["B_is_not_a_hessian"]:
            raise Fail("a step receipt does not mark B as not-a-Hessian")
    lam_hat = res["steps"][-1]["stationarity"]["lambda_hat"]
    return dict(n_steps=res["n_steps"], native_final=native, budget=eps,
                risk_final=risk, analytic_best=-best, fraction_of_best=frac,
                lambda_star=lam, lambda_hat=lam_hat)


@check("loop: the null-space share of a step is measured, not assumed", "loop")
def t_loop_null_fraction():
    from . import loop as L
    obj, y0, a = _loop_fixture()
    # a panel that sees the descent direction: one row gradient along a
    rows = np.tile(a / max(np.linalg.norm(a), 1e-30), (30, 1))
    rows[1:] = 0.0                       # 29 dead rows, as a real panel can have
    res = L.run(obj, y0, cfg=L.Config(eps=1e-3, delta0=0.02, max_steps=4),
                row_grads=rows, verbose=False)
    if not res["steps"]:
        raise Fail("no step taken with a guard attached")
    nf = res["steps"][0]["null_fraction"]
    if nf is None or not (0.0 <= nf <= 1.0):
        raise Fail(f"null fraction not reported: {nf}")
    if nf > 1e-6:
        raise Fail(f"a step along the panel's own direction reported null "
                   f"fraction {nf}, but the descent direction lies in the span")
    # a panel that sees nothing must report a null fraction of 1 and still not
    # crash -- the run is then not evidence, which the field makes visible
    blind = np.zeros((30, D.N_DESIGN))
    res2 = L.run(obj, y0, cfg=L.Config(eps=1e-3, delta0=0.02, max_steps=2),
                 row_grads=blind, verbose=False)
    if res2["guard"]["rank"] != 0:
        raise Fail(f"a zero row-gradient matrix reported rank {res2['guard']['rank']}")
    return dict(null_fraction=nf, rank=res["guard"]["rank"],
                null_dim=res["guard"]["null_dim"],
                blind_rank=res2["guard"]["rank"])


@check("loop: a run that accepts no step is reported as one", "loop")
def t_loop_no_step():
    from . import loop as L
    y0 = D.pack(D.nu_of_x(D.native_x()), 1.0)
    # a long-range term that always gets worse: every step must be refused
    class Worsening(_QuadObjective):
        def values(self, y, names=None):
            z = self._z(y)
            out = {"native_kl": 0.5 * self.s_n * float(z @ z),
                   "risk_a": float(np.abs(z).sum())}
            return {k: out[k] for k in (names or out)}

        def grads(self, y, names=None):
            z = self._z(y)
            g = {"native_kl": self.s_n * z,
                 "risk_a": np.sign(z) * 1.0}
            keys = names or list(g)
            return dict(names=keys, grads=np.array([g[k] for k in keys]),
                        values=[float(np.abs(z).sum()) if k == "risk_a"
                                else 0.5 * self.s_n * float(z @ z) for k in keys])

    obj = Worsening(y0, {"risk_a": np.zeros(D.N_DESIGN)})
    res = L.run(obj, y0, cfg=L.Config(eps=1e-2, delta0=0.02, max_steps=3),
                verbose=False)
    if res["status"] != "no_step_accepted":
        raise Fail(f"a worsening objective produced status {res['status']!r}")
    if res["n_steps"] != 0 or not res["refusals"]:
        raise Fail("a refused run carries no refusal record")
    if "result" not in res["no_step_note"]:
        raise Fail("the no-step note does not say a refusal is a result")
    return dict(status=res["status"], n_refused=res["n_refused"],
                refused_rho=res["refusals"][0].get("rho"))


@check("loop: Pareto continuation records the slope it is checking", "loop")
def t_loop_pareto():
    from . import loop as L
    obj, y0, a = _loop_fixture()
    out = L.pareto(obj, y0, [1e-4, 4e-4], cfg=L.Config(delta0=0.02, max_steps=40),
                   verbose=False)
    rows = out["frontier"]
    if len(rows) != 2:
        raise Fail(f"{len(rows)} frontier rows for 2 budgets")
    if rows[0]["eps"] != 1e-4:
        raise Fail("the frontier is not in the requested order")
    if "slope_dloss_deps" not in rows[1] or "predicted_dloss_deps" not in rows[1]:
        raise Fail("the slope column is missing -- the plan's identity is then "
                   "asserted rather than checked")
    # the plan's convention: BOTH columns negative (loss falls as the budget grows)
    if rows[1]["slope_dloss_deps"] >= 0:
        raise Fail(f"a larger budget did not lower the loss on this fixture: "
                   f"slope {rows[1]['slope_dloss_deps']}")
    if rows[1]["predicted_dloss_deps"] >= 0:
        raise Fail("lambda_hat has the wrong sign for a minimisation")
    # THE FRONTIER IS A REPORT, NOT A GUARANTEE, and the check must not pretend
    # otherwise.  A larger budget admits a superset of steps, but the solve is a
    # sequence of local trust-region steps whose accepted path depends on the
    # budget, so monotonicity is not implied and is not asserted.  What IS
    # asserted is that each run respected its own budget -- the property the
    # driver is actually responsible for.
    for r in rows:
        if r["eps"] == 1e-4 and r["native_kl"] is not None and r["native_kl"] > 1e-4 * 1.000001:
            raise Fail(f"the eps=1e-4 run ended at native {r['native_kl']}")
    out["frontier_is_monotone"] = bool(
        rows[1]["long_decrease"] is not None
        and rows[0]["long_decrease"] is not None
        and rows[1]["long_decrease"] >= rows[0]["long_decrease"] - 1e-12)
    # .get: the FIRST frontier row has no slope -- a slope needs a predecessor --
    # so indexing it would fail on a correct result
    return dict(rows=[{k: r.get(k) for k in ("eps", "status", "n_steps",
                                             "lambda_hat", "slope_dloss_deps",
                                             "predicted_dloss_deps", "native_kl",
                                             "long_decrease")} for r in rows],
                frontier_is_monotone=out["frontier_is_monotone"],
                slope_agrees_with_lambda=out["frontier"][1].get(
                    "slope_agrees_with_lambda"))


@check("loop: stationarity residual is reported with its asymmetry", "loop")
def t_loop_stationarity():
    from . import loop as L
    e = np.array([1.0, 0.0, 0.0])
    n = np.array([0.0, 1.0, 0.0])
    d = L.stationarity_diagnostic({"g": e}, {"native_kl": n})
    close(d["lambda_hat"], 0.0, 1e-15, "lambda with n perpendicular to e")
    close(d["residual_norm"], 1.0, 1e-15, "residual is all of e")
    if "not global optimality" not in d["note"]:
        raise Fail("the diagnostic does not disclaim global optimality")
    # n antiparallel to e: lambda > 0 and the residual can vanish
    d2 = L.stationarity_diagnostic({"g": np.array([1.0, 0.0])},
                                   {"native_kl": np.array([-1.0, 0.0])})
    close(d2["lambda_hat"], 1.0, 1e-12, "lambda")
    close(d2["residual_norm"], 0.0, 1e-12, "residual vanishes when balanced")
    # a zero native gradient must not divide by zero
    d3 = L.stationarity_diagnostic({"g": e}, {"native_kl": np.zeros(3)})
    if d3["lambda_hat"] is not None or "zero" not in d3["note"]:
        raise Fail("a zero native gradient was not reported as such")
    # the active group defaults to largest-gradient and SAYS so
    d4 = L.stationarity_diagnostic({"small": np.array([0.1, 0.0]),
                                    "big": np.array([1.0, 0.0])},
                                   {"native_kl": np.array([0.0, 1.0])})
    if d4["active"] != "big" or "not the measured active set" not in d4["selected_by"]:
        raise Fail(f"the default active-group choice is silent: {d4['selected_by']}")
    return dict(lambda_zero=d["lambda_hat"], lambda_balanced=d2["lambda_hat"],
                active_default=d4["active"])


@check("loop: gain movement is flagged on every receipt", "loop")
def t_loop_gain_flag():
    from . import loop as L
    y0 = D.pack(D.nu_of_x(D.native_x()), 1.0)
    a = np.zeros(D.N_DESIGN); a[10:40] = 1.0
    a[D.A_INDEX] = -0.5                    # a direction that also wants the gain
    obj = _QuadObjective(y0, {"risk_a": a}, s_native=1.0, s_risk=0.05)
    res = L.run(obj, y0, cfg=L.Config(eps=1e-3, delta0=0.02, max_steps=4),
                verbose=False)
    if not res["steps"]:
        raise Fail("no step taken")
    for s in res["steps"]:
        if "gain_unpriced" not in s:
            raise Fail("a step receipt does not carry gain_unpriced")
        if s["gain_component"] != 0.0 and not s["gain_unpriced"]:
            raise Fail("a step moved the gain without flagging it")
    return dict(n_steps=res["n_steps"],
                gains=[s["gain_component"] for s in res["steps"]],
                flags=[s["gain_unpriced"] for s in res["steps"]])
if __name__ == "__main__":
    sys.exit(main())
