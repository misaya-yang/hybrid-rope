"""The 65-dimensional design variable, its feasible set, and the two coordinates.

FINAL_PLAN.md sec.2 fixes the variable:

    y = (x, a),    x_j := -log(nu_j),    a := log(g^2)
    so             nu_j = exp(-x_j),      g = exp(a/2)

65 numbers for the whole model: 64 shared rotary frequencies and one shared
attention gain.  Every layer uses the same table -- the layers all participate in
the forward and in the backward, but they are not independently configured.  That
is the "shared table" model the panel was scored under, and it is why the design
dimension is 65 and not 65*36.

RELATION TO THE EXISTING eps-COORDINATE.  `curvature_20260910/tables.py` already
works in

    eps_j = ln(omega_j / nu_j)  =  x_j - x_j^0

which is this file's x shifted by the native point.  The two agree by
construction:  m_j = eps_j / ln S, and running the existing `from_eps` by hand
reproduces `m_new` here exactly.  Nothing is re-parameterised; x is the same
coordinate with the origin dropped, and eps is recovered by `eps_of`.

WHY THE NATIVE POINT IS THE ORIGIN OF THE METRIC, NOT OF THE DESIGN.  All the
curvature in this package (F_N, B_i) is measured AT native, because that is where
a frozen checkpoint's response is defined and where the probes in
`curvature_20260910` measured it.  But the solve does not have to start there --
the plan is explicit that MrPro is a starting point, not an optimum (sec.9), so
`origin` and `x0` are separate arguments.

CONSTRAINTS, AND WHICH ONES ARE DESIGN CHOICES.  The plan separates them and so
does this file:

  * ordering  x_{j+1} - x_j >= delta_min.  A hard feasibility requirement: two
    slots at the same frequency are one slot, and a cross-over silently reassigns
    which dimension carries which band.  Written as the plan states it, in
    increments: d_j - d_{j+1} <= x_{j+1} - x_j - delta_min.  Never repaired by
    sorting after the step -- sorting IS the reassignment.
  * endpoints and total span.  NOT identities.  `pin_ends` and `fix_span` are
    off-by-default-recalled design choices; the plan calls a fixed endpoint a
    matched-support experimental convention.  Both are offered because both are
    runnable arms, and a receipt must say which was on.
  * gain interval.  A declared finite box.  The plan requires it declared in
    advance rather than discovered by the solver at the boundary.
"""
from __future__ import annotations

import math

import numpy as np

from ..curvature_20260910 import tables as T

K = T.K
N_DESIGN = K + 1                     # 64 frequencies + 1 gain
A_INDEX = K                          # position of `a` in the design vector


# ---------------------------------------------------------------------------
# coordinates
# ---------------------------------------------------------------------------
def x_of_nu(nu):
    """x_j = -log nu_j."""
    nu = np.asarray(nu, dtype=np.float64)
    if (nu <= 0).any():
        raise ValueError("frequencies must be positive")
    return -np.log(nu)


def nu_of_x(x):
    return np.exp(-np.asarray(x, dtype=np.float64))


def a_of_gain(gain):
    """a = log(g^2) = 2 log g.  The square is not decoration: the rotary cos and
    sin are both multiplied by `attention_scaling`, so q.k picks up g^2."""
    return 2.0 * math.log(float(gain))


def gain_of_a(a):
    return math.exp(0.5 * float(a))


def eps_of_x(x, origin):
    """Back to the coordinate the existing package and the panel speak in."""
    return np.asarray(x, dtype=np.float64) - np.asarray(origin, dtype=np.float64)


def native_x(theta=T.QWEN25_3B["theta"], k=K):
    return x_of_nu(T.native_inv_freq(theta, k))


# ---------------------------------------------------------------------------
# assembling / disassembling a design vector
# ---------------------------------------------------------------------------
def pack(nu, gain):
    y = np.empty(N_DESIGN)
    y[:K] = x_of_nu(nu)
    y[A_INDEX] = a_of_gain(gain)
    return y


def unpack(y):
    y = np.asarray(y, dtype=np.float64)
    if y.shape != (N_DESIGN,):
        raise ValueError(f"expected {N_DESIGN} design entries, got {y.shape}")
    return nu_of_x(y[:K]), gain_of_a(y[A_INDEX])


def to_table(y, name="joint", theta=T.QWEN25_3B["theta"]):
    """The dict `FrozenRoPE.install_table` consumes."""
    nu, g = unpack(y)
    m = T.inv_freq_to_m(nu, theta)
    return dict(name=name, m=m, gain=float(g), theta=float(theta),
                values_float32=nu.astype(np.float32))


def from_table(table):
    return pack(np.asarray(table["values_float32"], dtype=np.float64), table["gain"])


# ---------------------------------------------------------------------------
# gain box and ordering floor
# ---------------------------------------------------------------------------
# The gain box is declared, not discovered.  YaRN's own factor 1 + 0.1 ln S is
# the historical anchor and sits near the middle; the half-decade either side is
# wide enough that the solver is never forced onto the boundary by the box alone,
# which is the property that makes "gain interior" checkable rather than assumed.
GAIN_BOX = (0.5, 2.5)


def a_box(gain_box=GAIN_BOX):
    return a_of_gain(gain_box[0]), a_of_gain(gain_box[1])


# delta_min as a fraction of the native log-gap.  The native grid's gap is
# ln(theta)/K = 0.2159 at Qwen; requiring a tenth of it is a floor no plausible
# optimum presses against (MrPro's tightest transition gap is ~0.5 native gaps)
# while still forbidding the exact collision that would reassign slots.
DELTA_MIN_FRAC = 0.1


def delta_min(theta=T.QWEN25_3B["theta"], k=K, frac=DELTA_MIN_FRAC):
    return frac * math.log(theta) / k


# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------
def ordering_rows(x, delta=None, k=K):
    """G x <= h for  x_{j+1} - x_j >= delta.

    Returns (G, h) with one row per adjacent pair, in the INCREMENT form the plan
    states:

        G (y - x) <= h        h_j = (x_{j+1} - x_j) - delta

    so a caller can hand the same object to any of the solvers here without
    re-deriving the sign -- `qcqp.solve_epigraph` and `qcqp.repair_step` both
    take the step directly and apply G to it.  A caller holding a POINT rather
    than a step must go through `violations`, which does the subtraction; the
    distinction is stated there at length because collapsing it is silent.

    Empty rows (h_j <= 0, i.e. the base point already sits at or below the
    floor) mean no positive step can restore that gap.  That is a Phase-I
    problem, and the caller is told rather than handed a silent zero row.
    """
    x = np.asarray(x, dtype=np.float64)
    d = delta if delta is not None else delta_min(k=k)
    gaps = x[1:] - x[:k - 1]
    G = np.zeros((k - 1, N_DESIGN))
    for j in range(k - 1):
        G[j, j] = 1.0
        G[j, j + 1] = -1.0
    h = gaps - d
    return G, h


def endpoint_rows(k=K):
    """x_0 and x_{k-1} pinned: two rows of A."""
    A = np.zeros((2, N_DESIGN))
    A[0, 0] = 1.0
    A[1, k - 1] = 1.0
    return A


def span_row(k=K):
    """Total log-span pinned: sum_j x_j fixed (the 'bank' the plan's sec.5 prices
    transfers against)."""
    A = np.zeros((1, N_DESIGN))
    A[0, :k] = 1.0
    return A


def gain_box_rows(a_at, gain_lo=None, gain_hi=None):
    """g in [gain_lo, gain_hi], as two rows in the INCREMENT form G (y-x) <= h.

    TWO THINGS ABOUT THIS FUNCTION ARE EASY TO GET WRONG AND BOTH WERE.

    (1) THE ARGUMENTS ARE GAINS, NOT a-VALUES.  The box is a statement about the
    attention scaling -- GAIN_BOX = (0.5, 2.5) -- and `a = 2 log g` is nonlinear,
    so feeding gain bounds through as if they were a bounds silently enforces
    g in [1.284, 3.49]: a different box, wrong at both ends, invisible in every
    receipt.  `feasible_set` did exactly that.  The conversion happens here, once.

    (2) THE BOX IS AN ABSOLUTE CONSTRAINT ON a, WHICH MAKES IT AN INCREMENT ROW
    ANYWAY -- and it cannot be dropped into the same array as the ordering rows
    in point form, because `violations` applies G to `y - x` uniformly.  A point
    row that is left in point form therefore gets evaluated against the increment
    and reads as satisfied everywhere near the reference, which is how the box
    stopped being enforced at all in the first place.  Shifting by `a_at` fixes
    that: (a - a_at) <= hi - a_at is the same statement as a <= hi, and at the
    reference the slack is exactly zero.

    `a_at` is required rather than defaulted, because a default of zero is right
    only when the caller's reference happens to have a = 0, and the failure mode
    of getting it wrong is a box that is never violated.

    The gain is never in A: it is a design variable with a box, not an equality
    the other variables are projected against.  Pinning it to YaRN's value is what
    the `--fix-gain` arm does, and that arm is a decomposition control, not the
    main run -- sec.10C is explicit that the main comparison is the joint solve.
    """
    lo, hi = a_box(GAIN_BOX if gain_lo is None or gain_hi is None
                   else (gain_lo, gain_hi))
    a_at = float(a_at)
    G = np.zeros((2, N_DESIGN))
    G[0, A_INDEX] = 1.0
    G[1, A_INDEX] = -1.0
    return G, np.array([hi - a_at, a_at - lo])


def feasible_set(y_at, pin_ends=False, fix_span=False, gain_box_=GAIN_BOX,
                 delta=None, k=K):
    """(A, b, G, h) for  A y = b  and  G (y - y_at) <= h.

    `y_at` is the FULL design point the rows are built around -- all 65 entries,
    gain included.  A 64-vector of frequencies is refused rather than sliced: the
    gain box is one of the rows and it is only meaningful relative to the gain
    coordinate of the reference, so a caller that has not decided where the gain
    starts has not decided what the box is.

    The two families are in the forms their solvers want, and they differ:

      * `A`, `b` are POINT rows, A y = b.  An endpoint at x_0 is the statement
        y_0 = x_0, which does not depend on where you measured from.
      * `G`, `h` are INCREMENT rows, G (y - y_at) <= h.  Both the ordering floor
        and the gain box are stated this way: the ordering floor because it is
        already about a difference, the gain box because an absolute bound is the
        same statement as a shifted difference and has to be in the same form for
        `violations` to evaluate the array uniformly.

    A is empty unless the caller asked for the matched-support conventions.  The
    ordering rows are ALWAYS present; the gain box is ALWAYS present.  Row counts
    are reported by the callers' receipts, because "which constraints were on" is
    part of what makes a solved table interpretable.
    """
    y_at = np.asarray(y_at, dtype=np.float64).reshape(-1)
    if y_at.shape != (N_DESIGN,):
        raise ValueError(f"feasible_set needs the full {N_DESIGN}-entry design "
                         f"vector as the reference point, got {y_at.shape}. The "
                         "gain box is built around its gain coordinate, so a "
                         "frequency-only reference cannot define the set.")
    x = y_at[:k]
    rows, rhs = [], []
    if pin_ends:
        rows.append(endpoint_rows(k))
        rhs.append(np.array([x[0], x[k - 1]]))
    if fix_span:
        rows.append(span_row(k))
        rhs.append(np.array([x[:k].sum()]))
    A = np.vstack(rows) if rows else np.zeros((0, N_DESIGN))
    b = np.concatenate(rhs) if rhs else np.zeros(0)

    G, h = ordering_rows(x, delta=delta, k=k)
    Gg, hg = gain_box_rows(y_at[A_INDEX], *gain_box_)
    G = np.vstack([G, Gg])
    h = np.concatenate([h, hg])
    return A, b, G, h


def step_rows(y_at, **kw):
    """(A, b, G, h) for the STEP solvers:  A d = b  and  G d <= h.

    `feasible_set` returns A, b in POINT form (A y = b), because an endpoint at
    x_0 is a statement about the point and does not depend on where you measured
    from.  The solvers take the step.  The conversion is one subtraction and
    getting it wrong is SILENT -- the solve returns a step satisfying the wrong
    equality, `G` still looks satisfied because it was already step-form, and the
    only symptom is a step that does not move the thing it was supposed to move.
    That is exactly what happened the first time `loop.py` was wired up, so the
    shift lives here, once, instead of in every caller.

    `G`, `h` need no shift: they are already stated on the increment.
    """
    A, b, G, h = feasible_set(y_at, **kw)
    y_at = np.asarray(y_at, dtype=np.float64).reshape(-1)
    if A.shape[0]:
        if A.shape[1] != y_at.size:
            raise ValueError(f"A is {A.shape}, reference point is {y_at.shape}")
        b = b - A @ y_at
    return A, b, G, h


# ---------------------------------------------------------------------------
# sec.5 -- the gap price structure
#
# The plan prices the design through its GAPS rather than its coordinates:
#
#     p_i = x_i - x_{i-1}        (i = 1 .. K-1)
#     price_i = sum_{j >= i} v_j            v = dL/dx
#
# and the reason is that a gap transfer is the operation the design can actually
# perform while holding the endpoints: moving t of gap budget from gap i to gap
# k leaves every other gap and both endpoints untouched.  In coordinates that
# transfer is
#
#     x_j -> x_j - t     for j in [min(i,k), max(i,k) - 1]
#
# so its directional derivative is  v . u = price_k - price_i.  That identity is
# exact and is checked against a finite difference in `selftest.py`; everything
# downstream (the gap-transfer table in the receipt, the "are the free gaps
# equi-priced at the optimum" reading) is this one line.
#
# `gap_prices` merely differences a gradient it is HANDED.  It does not compute
# an objective, and sec.5 is explicit that prices built from energy, cycle counts
# or the existing m-coordinate are not an identification of anything -- a price
# is a statement about the actual loss, so the gradient must come from one.
# ---------------------------------------------------------------------------
def gap_transfer(i, k, k_slots=K):
    """The direction of "move t of gap from gap i to gap k", for i != k.

    Indexing follows the gaps: gap i is the interval between x_{i-1} and x_i, so
    i runs over 1..K-1.  The gain coordinate is never touched -- a gap transfer
    holds the gain, and a caller that also moves the gain is doing sec.6, not
    sec.5.
    """
    if i == k:
        raise ValueError("a transfer from a gap to itself is the zero direction")
    n = N_DESIGN if k_slots >= K else k_slots + 1
    lo, hi = (i, k) if i < k else (k, i)
    if lo < 1 or hi > k_slots - 1:
        raise ValueError(f"gap indices must lie in 1..{k_slots - 1}, got ({i}, {k})")
    u = np.zeros(n)
    u[lo:hi] = -1.0 if i < k else 1.0
    return u


def gap_prices(v, k=K):
    """price_i = sum_{j >= i} v_j, with price_k = 0 by convention.

    price_j is the marginal value of moving the whole tail of gaps outward; the
    DIFFERENCES price_a - price_b are what a transfer sees, so the offset is
    arbitrary and the last entry is fixed at zero for readability rather than
    necessity.
    """
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.size < k:
        raise ValueError(f"gradient has {v.size} entries, need {k} frequencies")
    tail = np.cumsum(v[k - 1::-1])[::-1]          # tail[j] = sum_{m >= j} v_m
    return np.concatenate([tail, [0.0]])


# ---------------------------------------------------------------------------
def violations(y, x, A, b, G, h, k=K, tol=1e-9):
    """Which constraints the point y breaks, and by how much.  Used by the
    acceptance check (sec.8) and by Phase-I, which needs to know the shortfall
    before it can price it.

    THE TWO KINDS OF ROW ARE EVALUATED DIFFERENTLY, AND THIS IS THE ONE PLACE
    THAT CAN GO QUIETLY WRONG.  `feasible_set` returns them in the forms their
    solvers want, and those forms are not the same:

      * `A`, `b` are POINT rows:  A y = b.  An endpoint at x_0 is the statement
        y_0 = x_0, which does not depend on where you measured from.
      * `G`, `h` are INCREMENT rows:  G (y - x) <= h.  The ordering floor is
        stated on the increments --  d_j - d_{j+1} <= (x_{j+1} - x_j) - delta --
        because that is the form the step solvers consume directly, with h
        already evaluated at the current point.

    Applying `G` to `y` instead of to `y - x` is therefore not a near miss: it
    evaluates a constraint about the increment on the point itself, and the
    numbers come out negative for every point near x, so a sub-floor gap passes
    unnoticed.  That is exactly what this function did before `selftest.py`
    caught it.  The increment form is used below, and `t_ordering` pins it.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(
            f"violations needs the reference point at the same shape as y "
            f"({y.shape}), got {x.shape}. A bare 64-vector of frequencies is not "
            "a reference point: the increment G(y - x) mixes in the gain "
            "coordinate, and a shape mismatch here broadcasts into a silently "
            "wrong slack rather than raising.")
    out = dict(call_ok=True, call_excess=0.0)
    if A.shape[0]:
        v = A @ y - b
        out["call_ok"] = bool(np.abs(v).max() <= tol) if v.size else True
        out["call_excess"] = float(np.abs(v).max()) if v.size else 0.0
    if G.shape[0]:
        # einsum, not `@`: numpy 2.0.2 on macOS routes this shape to Accelerate,
        # which leaves FP status flags set and makes a correct result emit
        # "divide by zero / overflow encountered in matmul".  A spurious warning
        # is worse than a slow one here -- this package relies on seeing the real
        # numerical warnings.
        slack = np.einsum("rj,j->r", G, y - x) - h
        bad = np.flatnonzero(slack > tol)
        out["ineq_ok"] = bool(bad.size == 0)
        out["ineq_excess"] = float(slack.max()) if slack.size else 0.0
        out["ineq_violating"] = [int(j) for j in bad]
    else:
        out["ineq_ok"], out["ineq_excess"], out["ineq_violating"] = True, 0.0, []
    # a frequency ordering violation is worth naming separately: it is the one
    # failure that silently changes what the design vector MEANS
    nu, _ = unpack(y)
    ordered = bool((np.diff(nu) <= 0).all())
    out["freq_order_ok"] = ordered
    return out


def describe(y, theta=T.QWEN25_3B["theta"]):
    nu, g = unpack(y)
    m = T.inv_freq_to_m(nu, theta)
    d = np.diff(m)
    return dict(gain=float(g), a=float(a_of_gain(g)), m_sum=float(m.sum()),
                m_min=float(m.min()), m_max=float(m.max()),
                gap_min=float(d.min()), gap_max=float(d.max()),
                nu_min=float(nu.min()), nu_max=float(nu.max()))
