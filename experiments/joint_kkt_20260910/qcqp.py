"""The sec.6 elimination, the sec.8 analytic step, and the sec.8 QCQP.

Three objects, deliberately kept in one file because they are three views of the
same subproblem and separating them is how the packages drift apart:

  * `build_quadratic` -- assemble Q and r = sum_i w_i (B_i, grad_i) from the
    multipliers of the Lagrangian in sec.5.  Nothing here knows what a "risk
    group" is; it takes weights and matrices.
  * `analytic_step` -- sec.8's closed form for the step when the active linear
    constraints are known:
        d = -Q^-1 r + Q^-1 A^T (A Q^-1 A^T)^-1 (b + A Q^-1 r)
    This is a linear solve, never a matrix inverse, exactly as the plan requires.
  * `solve_epigraph` -- the same step when the constraints are NOT known to be
    active: min t subject to the convex quadratics q_i(d) <= t, the spectral
    constraints, and the trust region.  The epigraph variable is what keeps the
    max smooth; the plan forbids putting a C2 Hessian across an active-set max.

WHAT B_i IS AND IS NOT.  sec.8 is explicit: this first implementation uses real
gradients and a positive-definite model B_i that is *not* the measured Hessian.
Every receipt this file's callers write must carry that sentence, because a
quasi-Newton model that has never been checked against a real curvature is a
numerical convenience, not evidence.  The acceptance check in accept.py is what
keeps that honest: a step is only taken if the real function stays under the
model's own prediction.

WHY THE GAIN IS ELIMINATED AND NOT JUST PROJECTED OUT.  sec.6 removes the gain
from the quadratic model in BOTH the curvature and the gradient:

    d_a   = -(r_a + Q_af d_f) / Q_aa
    Q_eff = Q_ff - Q_fa Q_af / Q_aa
    r_eff = r_f  - Q_fa r_a   / Q_aa

This is a Schur complement, and it is not the same operation as projecting the
gradient against a pinned gain.  Projection would keep Q_ff and only orthogonalise
r; the Schur form also lets the gain's curvature *soften* the frequency
directions.  They agree only when Q_fa = 0.  Running the elimination when the gain
is at its box boundary is wrong -- the complementarity condition says the gain
stops being a free variable there -- so `eliminate_gain` refuses unless the caller
asserts the gain is interior, and the caller is the one that checked.

WHY THE SOLVER IS HAND-ROLLED RATHER THAN HANDED TO SCIPY'S NLP INTERFACE.  This
was measured, not assumed.  Two attempts to express the problem as
scipy.optimize.minimize constraints both returned non-solutions:

  * SLSQP returned t = -6.8e30 and ||d|| = 4e28.  It linearises its constraints,
    and at d = 0 the linearisation of d^T D d <= Delta^2 is the vacuous
    0 <= Delta^2 with a zero Jacobian row, so the epigraph objective is unbounded
    below and nothing stops the step.
  * trust-constr returned points violating the trust region by ~1.0 in squared
    norm.  Its quasi-Newton model is built for a nonlinear objective, and
    `min t` is linear, so scipy itself warns "the function is linear ... define
    the Hessian as zero".

Neither is a bug in scipy; the problem is a small SOCP (the form astra03
identifies: epigraph + diagonal quadratic is second-order-cone representable) and
no SOCP solver is installed on either machine -- cvxpy, ecos, scs, clarabel and
osqp are all absent, checked on both.  So the solve here is an annealed smooth
penalty minimised by L-BFGS-B under a hard box, followed by an exact repair.

The repair step is what makes solver *quality* non-critical.  sec.8 accepts a step
only if the real functions stay under the model's prediction, so a merely-good
step is fine and a bad step is caught downstream; what must never happen is a step
that is silently infeasible.  `repair_step` therefore ends every solve by
projecting onto the affine constraints and the trust ball by alternating
projection, and `feasible` reports the post-repair truth.  A caller that sees
feasible=False is expected to shrink Delta and re-solve -- that is the sec.8
retry ladder, not an error path.
"""
from __future__ import annotations

import numpy as np

try:                                                    # scipy is present on both
    from scipy.optimize import minimize as _scipy_minimize
    HAVE_SCIPY = True
except Exception:                                       # pragma: no cover
    HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# two tiny linear-algebra helpers, so the hot paths do not go through `@`
#
# numpy 2.0.2 on macOS routes these shapes to Accelerate, which leaves the FP
# status flags set and makes a CORRECT result emit "divide by zero / overflow
# encountered in matmul".  The values are right -- verified against the same
# computation under einsum -- but a spurious warning is worse than a slow one
# here: this package's numerical checks rely on seeing the real ones, and a gate
# that prints four bogus warnings per solve is a gate nobody reads.  The server's
# OpenBLAS does not emit them, so this is a local-only concern that would
# otherwise hide a real one on the machine the code is edited on.
# ---------------------------------------------------------------------------
def _q(M, x):
    """x^T M x."""
    return float(np.einsum("i,ij,j->", x, M, x))


def _mv(M, x):
    """M x."""
    return np.einsum("ij,j->i", M, x)


# ---------------------------------------------------------------------------
# assembling the local model
# ---------------------------------------------------------------------------
def build_quadratic(terms, D=None, eta=0.0, n=None):
    """Q, r for  q(d) = r^T d + 0.5 d^T Q d.

    `terms` is an iterable of (weight, B_i, r_i).  A negative weight is allowed
    and is how a shadow-price term enters, but it can destroy positive
    definiteness -- `psd_floor` reports that rather than silently clipping it,
    because a non-convex local model is a modelling error and not a numerical
    nuisance.
    """
    terms = list(terms)
    if not terms and D is None:
        raise ValueError("nothing to assemble")
    n = n if n is not None else (terms[0][1].shape[0] if terms else D.shape[0])
    Q = np.zeros((n, n))
    r = np.zeros(n)
    for w, B, g in terms:
        Q += float(w) * np.asarray(B, dtype=np.float64)
        r += float(w) * np.asarray(g, dtype=np.float64)
    if D is not None and eta:
        Q += float(eta) * np.asarray(D, dtype=np.float64)
    return 0.5 * (Q + Q.T), r


def psd_floor(Q, rel=1e-9):
    """Smallest eigenvalue relative to the largest, and whether Q is usable as a
    convex model.  Reported, not repaired: the callers' retry ladder raises the
    damping itself so the escalation is visible in the receipt."""
    w = np.linalg.eigvalsh(0.5 * (np.asarray(Q) + np.asarray(Q).T))
    scale = max(float(np.abs(w).max()), 1e-300)
    return dict(lambda_min=float(w.min()), lambda_max=float(w.max()),
                rel_min=float(w.min() / scale), psd=bool(w.min() >= -rel * scale))


def damp(Q, mu):
    """Q + mu I.  The sec.8 escalation knob: raise mu, re-solve the same problem."""
    Q = np.asarray(Q, dtype=np.float64)
    return Q + float(mu) * np.eye(Q.shape[0])


# ---------------------------------------------------------------------------
# sec.6 -- joint gain elimination
# ---------------------------------------------------------------------------
def eliminate_gain(Q, r, gain_index, interior=True):
    """Schur-complement the gain out of the quadratic model.

    Returns (Q_eff, r_eff, d_a_of_d_f, free_index).  The third return is a
    closure, so a caller can recover the eliminated coordinate once it knows d_f:

        d_a = -(r_a + Q_af d_f) / Q_aa

    `interior` is the caller's assertion that the gain is off its box.  It
    defaults to True only so the numeric path can be unit tested; production
    callers pass the result of an actual box check, because eliminating a
    boundary-bound gain silently drops the complementary slackness that pins it
    there.
    """
    if not interior:
        raise ValueError("gain is not interior -- keep it in the problem and let "
                         "the QCQP handle the box, per sec.6's last line")
    Q = np.asarray(Q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    f = [j for j in range(Q.shape[0]) if j != gain_index]
    Q_ff = Q[np.ix_(f, f)]
    Q_fa = Q[np.ix_(f, [gain_index])][:, 0]
    Q_af = Q[np.ix_([gain_index], f)][0]
    Q_aa = float(Q[gain_index, gain_index])
    r_f, r_a = r[f], float(r[gain_index])
    if Q_aa <= 0:
        raise ValueError(f"Q_aa = {Q_aa:.3e} is not positive; the gain is not a "
                         "strictly convex coordinate here and cannot be eliminated")
    Q_eff = Q_ff - np.outer(Q_fa, Q_af) / Q_aa
    r_eff = r_f - (Q_fa * r_a) / Q_aa

    def d_a_of_d_f(d_f):
        return float(-(r_a + float(np.einsum("i,i->", Q_af, d_f))) / Q_aa)

    return 0.5 * (Q_eff + Q_eff.T), r_eff, d_a_of_d_f, f


# ---------------------------------------------------------------------------
# sec.8 -- closed form with known active set
# ---------------------------------------------------------------------------
def analytic_step(Q, r, A=None, b=None):
    """d = -Q^-1 r + Q^-1 A^T (A Q^-1 A^T)^-1 (b + A Q^-1 r), by linear solve.

    Note the sign convention: the plan writes the model as  r^T d + 0.5 d^T Q d,
    so the unconstrained minimiser is -Q^-1 r and the correction enforces A d = b.
    With A empty this returns the unconstrained minimiser, which is the right
    degenerate behaviour -- no constraints means nothing to project against.
    """
    Q = np.asarray(Q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    Qinv_r = np.linalg.solve(Q, r)
    if A is None or np.asarray(A).shape[0] == 0:
        return -Qinv_r
    A = np.asarray(A, dtype=np.float64)
    b = np.zeros(A.shape[0]) if b is None else np.asarray(b, dtype=np.float64)
    Qinv_At = np.linalg.solve(Q, A.T)
    M = A @ Qinv_At
    # pinv rather than solve: two endpoint rows can be near-degenerate if a
    # caller pins a slot twice, and a rank-deficient A is a caller error that
    # should degrade to "enforce what is enforceable", not raise
    y = np.linalg.pinv(M) @ (b + A @ Qinv_r)
    return -Qinv_r + Qinv_At @ y


def predicted(Q, r, d):
    """(value, gradient) of the local model at d -- what accept.py compares the
    real function against."""
    d = np.asarray(d, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    return float(np.einsum('i,i->', r, d)) + 0.5 * _q(Q, d), r + _mv(Q, d)


# ---------------------------------------------------------------------------
# the trust ellipsoid and its implied box
# ---------------------------------------------------------------------------
def implied_box(D, Delta, tol=1e-12):
    """The largest box the trust ellipsoid provably contains.

    For any PSD D,  d^T D d >= lambda_min(D) * ||d||^2,  so  ||d|| <= Delta/sqrt(l)
    is a CONSEQUENCE of the trust region, not a relaxation of it.  Intersecting
    with this box therefore leaves the feasible set unchanged -- it does not
    weaken the model -- while giving a gradient-based solver a bounded domain.

    lambda_min > 0 is required for the bound to exist; a singular D is reported
    rather than silently replaced, because a singular metric means the caller
    forgot to damp and the solve that follows would be meaningless.
    """
    D = np.asarray(D, dtype=np.float64)
    if D.shape[0] != D.shape[1]:
        raise ValueError("D must be square")
    w = np.linalg.eigvalsh(0.5 * (D + D.T))
    if w.min() <= tol * max(float(w.max()), 1.0):
        raise ValueError(f"D is singular or indefinite (lambda_min={w.min():.3e}); "
                         "the trust region is not an ellipsoid and the implied box "
                         "does not exist -- use a damped metric")
    r = Delta / np.sqrt(w.min())
    return -np.full(D.shape[0], r), np.full(D.shape[0], r)


def project_tr(d, D, Delta):
    """Nearest point of the trust ellipsoid to d, by the exact scaling that works
    for the D-metric: d stays on its own ray, so the affine properties of d are
    the only thing that can be broken and the caller re-projects after."""
    d = np.asarray(d, dtype=np.float64)
    q = _q(D, d)
    if q <= Delta * Delta:
        return d
    return d * (Delta / np.sqrt(q))


def repair_step(d, entries, A, b, G, h, D, Delta, iters=200, tol=1e-10,
                ineq_tol=None, stall=12):
    """Alternating projection (POCS) onto all three constraint sets at once:

        {A d = b}   affine, exact projection
        {G d <= h}  halfspaces, exact projection onto the most-violated row
        {d^T D d <= Delta^2}   ray scaling

    Each set is convex and closed, so alternating projection converges to a point
    of their intersection whenever that intersection is nonempty.  The reason for
    including the inequalities rather than only reporting them: the annealed
    penalty leaves them ~1e-4 outside, which is small but is still an infeasible
    step handed to an acceptance check, and driving the penalty weight up far
    enough to remove it destroys the conditioning of the stage that produces the
    direction in the first place.

    The honest failure mode is kept.  If the inequality rows are INCONSISTENT the
    intersection is empty, POCS does not converge, and the violation stops
    decreasing -- `stall` consecutive non-improving iterations end the loop and
    the still-violating rows are reported rather than quietly converged to a
    boundary.  A caller that sees them is expected to shrink Delta (the sec.8
    retry ladder): a smaller ball cannot make an inconsistent set consistent, but
    it can reveal that the ordering floor itself, not the step, is the problem.

    Returns (d, None, repair_info); repair_info is None when nothing was touched.
    """
    d = np.asarray(d, dtype=np.float64).copy()
    A = np.asarray(A, dtype=np.float64).reshape(-1, len(d))
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    G = np.asarray(G, dtype=np.float64).reshape(-1, len(d))
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    D = np.asarray(D, dtype=np.float64)
    if ineq_tol is None:
        ineq_tol = tol

    gn = np.linalg.norm(G, axis=1) if G.shape[0] else np.zeros(0)

    def eq_viol(x):
        return float(np.abs(np.einsum("ij,j->i", A, x) - b).max()) if A.shape[0] else 0.0

    def in_viol(x):
        return float((np.einsum("ij,j->i", G, x) - h).max()) if G.shape[0] else 0.0

    def tr_viol(x):
        return _q(D, x) - Delta * Delta

    if eq_viol(d) <= tol and in_viol(d) <= ineq_tol and tr_viol(d) <= tol:
        return d, None, None

    before = dict(eq=eq_viol(d), ineq=in_viol(d), tr=tr_viol(d))
    A_pinv = np.linalg.pinv(A) if A.shape[0] else None
    best = in_viol(d)
    stalled = 0
    it = 0
    for it in range(iters):
        if A.shape[0]:
            d = d - np.einsum("ij,j->i", A_pinv,
                              np.einsum("ij,j->i", A, d) - b)
        if G.shape[0]:
            s = np.einsum("ij,j->i", G, d) - h
            j = int(np.argmax(s))
            if s[j] > ineq_tol and gn[j] > 0:
                d = d - (s[j] / (gn[j] ** 2)) * G[j]
        d = project_tr(d, D, Delta)

        v = in_viol(d)
        if v <= ineq_tol and eq_viol(d) <= tol and tr_viol(d) <= tol:
            break
        if v < best - 1e-15:
            best, stalled = v, 0
        else:
            stalled += 1
            if stalled >= stall:
                break
    return d, None, dict(eq_violation_before=before["eq"], ineq_violation_before=before["ineq"],
                         tr_violation_before=before["tr"],
                         eq_violation_after=eq_viol(d), ineq_violation_after=in_viol(d),
                         tr_violation_after=tr_viol(d),
                         stalled=bool(stalled >= stall), iterations=it + 1)


# ---------------------------------------------------------------------------
# sec.8 -- the epigraph QCQP
# ---------------------------------------------------------------------------
def _lse(q, mu):
    """Stable mu * log sum exp(q / mu), plus its softmax weights."""
    z = np.asarray(q, dtype=np.float64) / mu
    m = z.max()
    e = np.exp(z - m)
    s = e.sum()
    return mu * (m + np.log(s)), e / s


def solve_epigraph(d0, entries, A, b, G, h, D, Delta, bounds=None,
                   mu_schedule=(0.5, 0.2, 0.05, 0.02),
                   w_schedule=(1.0, 10.0, 100.0, 1_000.0, 10_000.0),
                   maxiter=600):
    """min t  s.t.  q_i(d) <= t (risk),  q_i(d) <= limit_i (native),
                   A d = b,  G d <= h,  d^T D d <= Delta^2.

    `entries` are dicts: name, kind ('risk'|'native'), r (n,), B (n,n), limit.
    d0 is the step to start from (zero is fine; that is what the loop uses).

    The residual spectral constraints are passed already shifted to the step --
    A d = b - A theta_k -- because computing that shift is the caller's job and
    getting it wrong is invisible once inside a solver.

    Method.  The max over risk groups is smoothed by log-sum-exp with weight mu,
    the remaining constraints enter as squared hinges with weight w, and the whole
    thing is minimised by L-BFGS-B under the implied box (so the domain is
    bounded).  mu and w are annealed outward: the first stages see a smooth,
    well-conditioned landscape, later stages sharpen toward the true max and
    drive the hinge residuals down.  Then the linear equalities and the trust ball
    are enforced exactly by `repair_step`.

    Returns the step, the achieved t, per-entry values, a feasibility report, and
    the schedule that produced it (so a receipt says how hard the solver worked).
    """
    if not HAVE_SCIPY:
        raise RuntimeError("scipy is required for solve_epigraph")
    n = len(d0)
    d0 = np.asarray(d0, dtype=np.float64)
    # normalise once, here: downstream code indexes these arrays hundreds of
    # times and a list-typed `r` would only fail on the first `@`, inside a
    # solver callback where the traceback is unreadable
    entries = [dict(e, r=np.asarray(e["r"], dtype=np.float64),
                    B=np.asarray(e["B"], dtype=np.float64)) for e in entries]
    A = np.asarray(A, dtype=np.float64).reshape(-1, n)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    G = np.asarray(G, dtype=np.float64).reshape(-1, n)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    D = np.asarray(D, dtype=np.float64)

    lo, hi = implied_box(D, Delta)
    if bounds is not None:
        lo = np.maximum(lo, np.asarray(bounds[0], dtype=np.float64))
        hi = np.minimum(hi, np.asarray(bounds[1], dtype=np.float64))
        if (lo > hi).any():
            j = int(np.flatnonzero(lo > hi)[0])
            raise ValueError(f"caller box is empty against the trust region at "
                             f"index {j}: lo={lo[j]:.3e} > hi={hi[j]:.3e}")

    risk = [e for e in entries if e["kind"] == "risk"]
    native = [e for e in entries if e["kind"] != "risk"]
    if not risk:
        raise ValueError("no risk entries: min t has no meaning without them")

    def objectives(d):
        return np.array([float(np.einsum("i,i->", e["r"], d)) + 0.5 * _q(e["B"], d)
                         for e in risk])

    def grads_q(e, d):
        return e["r"] + _mv(e["B"], d)

    def make_phi(mu, w):
        def phi(d):
            q = objectives(d)
            val, alpha = _lse(q, mu)
            g = np.zeros(n)
            for a_i, e, q_i in zip(alpha, risk, q):
                g += a_i * grads_q(e, d)
            for e in native:
                c = (float(np.einsum('i,i->', e["r"], d))
                     + 0.5 * _q(e["B"], d) - float(e["limit"]))
                if c > 0:
                    val += w * c * c
                    g += 2.0 * w * c * grads_q(e, d)
            if A.shape[0]:
                res = np.einsum("ij,j->i", A, d) - b
                val += w * float(np.einsum('i,i->', res, res))
                g += 2.0 * w * np.einsum('ij,i->j', A, res)
            if G.shape[0]:
                s = np.einsum("ij,j->i", G, d) - h
                bad = s > 0
                if bad.any():
                    val += w * float(np.einsum("i,i->", s[bad], s[bad]))
                    g += 2.0 * w * np.einsum("ij,i->j", G[bad], s[bad])
            tr = _q(D, d) - Delta * Delta
            if tr > 0:
                val += w * tr * tr
                g += 4.0 * w * tr * _mv(D, d)
            return float(val), g
        return phi

    d = d0.copy()
    bnds = list(zip(lo, hi))
    path = []
    for mu in mu_schedule:
        for w in w_schedule:
            phi = make_phi(mu, w)
            res = _scipy_minimize(phi, d, jac=True, bounds=bnds,
                                  method="L-BFGS-B",
                                  options=dict(maxiter=maxiter, ftol=1e-14, gtol=1e-12))
            d = res.x
            path.append(dict(mu=float(mu), w=float(w), fun=float(res.fun),
                             nit=int(getattr(res, "nit", -1))))

    d_raw = d.copy()
    d, _, repair = repair_step(d, entries, A, b, G, h, D, Delta)
    vals = {e["name"]: (float(np.einsum('i,i->', e["r"], d))
                        + 0.5 * _q(e["B"], d)) for e in entries}
    t = max(vals[e["name"]] for e in risk)

    worst_name, worst = None, -np.inf
    for e in entries:
        v = vals[e["name"]] - (t if e["kind"] == "risk" else float(e["limit"]))
        if v > worst:
            worst_name, worst = e["name"], float(v)
    v_lin = float(np.abs(np.einsum("ij,j->i", A, d) - b).max()) if A.shape[0] else 0.0
    v_g = float(np.einsum('ij,j->i', G, d).max() - h.max()) if G.shape[0] else 0.0
    v_tr = _q(D, d) - Delta * Delta

    return dict(d=d, t=float(t), values=vals,
                max_quad_violation=float(worst), max_quad_violation_of=worst_name,
                lin_eq_violation=float(v_lin), lin_ineq_violation=float(v_g),
                tr_violation=float(v_tr),
                feasible=bool(max(worst, v_lin, v_g, v_tr) <= 1e-7),
                repaired=repair is not None,
                repair_moved=float(np.linalg.norm(d - d_raw)),
                path=path, method="annealed-penalty+L-BFGS-B+repair")
