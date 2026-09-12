"""The nine C-free rules: D01 (M01-M03), D02 (M04-M06), D03 (M07-M09).

Plan section 7.2: "先构造9条无C规则，再一次性采集C、生成其余规则".
These nine need no calibration data, so they are exact, reproducible from the
geometry alone, and can be built and self-checked before any GPU minute.

Three construction classes, one per direction:

D01  integer winding promotion.  Pick the branch so that the phase at one
     anchor A is preserved exactly: exp(i A nu) = exp(i A omega/s).  Every
     solution is nu = omega/s + 2 pi k / A.  The rule takes the feasible
     integer set, then the k nearest to MR.  The plan is explicit that this is
     NOT a search over k against task scores.

D02  joint recurrence period.  Inside T only, replace each period by an
     integer satisfying the arm's arithmetic constraint, ascending in j and
     strictly increasing.  The whole modified subband then reproduces exactly
     at lcm(P_j).  No solution for any slot => INFEASIBLE, and the plan forbids
     patching that with an amplitude change.

D03  native dictionary reuse.  The dictionary is the shipped 64 native
     frequencies; nothing is extrapolated.  Each T slot may only take a
     dictionary node inside [omega_j/s, omega_j].

Section 5.4 governs the bookkeeping: failures are construction records
(INFEASIBLE / NO_CHANGE), never a task score of zero, and never a reason to
invent a fourth rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import core

LN_S = core.LN_S


@dataclass
class Construction:
    """A built frequency array plus the construction record section 5.4 wants."""

    method: str
    direction: str
    nu: np.ndarray
    status: str = "OK"                  # OK | INFEASIBLE | NO_CHANGE
    note: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """INFEASIBLE and NO_CHANGE are records, not runnable operators."""
        return self.status in ("OK", "NO_CHANGE")

    def m(self, geom):
        return geom.nu_to_m(self.nu)

    def sum_m(self, geom):
        nu = self.nu
        if not np.all(nu > 0):
            return None
        return float(np.sum(geom.nu_to_m(nu)))


# ---------------------------------------------------------------------------
# D01  integer winding promotion
# ---------------------------------------------------------------------------

D01_ANCHORS = {"M01": 1.0, "M02": 2.0, "M03": 4.0}


def d01(geom, method):
    """nu = omega/s + 2 pi k / A, k the feasible integer nearest to MR.

    Feasible means 0 <= k <= floor[A(omega - omega/s)/2pi], i.e. the result stays
    inside [omega/s, omega] and therefore never speeds a slot up.
    """
    A = D01_ANCHORS[method] * geom.window
    nu = geom.nu_mr.copy()
    kmin = np.full(geom.K, -1, dtype=np.int64)
    for j in geom.T:
        w = geom.omega[j]
        kmax = int(math.floor(A * (w - w / geom.scale) / (2.0 * math.pi) + 1e-12))
        if kmax < 0:
            continue
        ks = np.arange(0, kmax + 1)
        cand = w / geom.scale + 2.0 * math.pi * ks / A
        dev = np.abs(cand - geom.nu_mr[j])
        # nearest; ties go to the smaller |nu - nu0| (identical here) then smaller k
        best = int(np.argmin(dev * 1e12 + ks))
        nu[j] = cand[best]
        kmin[j] = int(ks[best])
    changed = [int(j) for j in geom.T if kmin[j] >= 0 and abs(nu[j] - geom.nu_mr[j]) > 0]
    if not changed:
        return Construction(method, "D01", nu, "NO_CHANGE", "no slot moved off its MR node")
    return Construction(
        method, "D01", nu, "OK",
        f"anchor A={int(A)}; {len(changed)} slots moved",
        {"A": float(A), "k": {int(j): int(kmin[j]) for j in geom.T if kmin[j] >= 0},
         "n_changed": len(changed)},
    )


# ---------------------------------------------------------------------------
# D02  joint recurrence period
# ---------------------------------------------------------------------------


def _d02_feasible(geom, j):
    """Integer periods in [2 pi/omega_j, 2 pi s / omega_j], ascending."""
    lo = 2.0 * math.pi / geom.omega[j]
    hi = 2.0 * math.pi * geom.scale / geom.omega[j]
    a = int(math.ceil(lo - 1e-12))
    b = int(math.floor(hi + 1e-12))
    return np.arange(max(a, 1), b + 1, dtype=np.int64)


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _prime_power(n):
    """Return (p, r) with n = p^r for PRIME p and r >= 2, else None.

    The primality of p is not a detail.  Without it 36 = 6^2 is accepted with
    base 6, and a table holding both 64 = 2^6 and 36 = 6^2 has gcd 4 -- so the
    "pairwise coprime" property M06 is built on silently fails.  Caught by
    selftest's coprimality assertion.
    """
    for p in range(2, int(math.isqrt(n)) + 1):
        if not _is_prime(p):
            continue
        r, v = 0, n
        while v % p == 0:
            v //= p
            r += 1
        if v == 1 and r >= 2:
            return p, r
    return None


def d02(geom, method):
    """Ascending in j, strictly increasing integer periods under an arithmetic rule."""
    nu = geom.nu_mr.copy()
    chosen = {}
    used_primes = set()
    used_bases = set()
    prev = 0
    for j in geom.T:
        feasible = _d02_feasible(geom, j)
        feasible = feasible[feasible > prev]
        if method == "M04":
            cand = np.array([p for p in feasible if _is_prime(int(p))], dtype=np.int64)
            cand = np.array([p for p in cand if int(p) not in used_primes], dtype=np.int64)
        elif method == "M05":
            cand = np.array([p for p in feasible
                             if all(math.gcd(int(p), q) == 1 for q in chosen.values())],
                            dtype=np.int64)
        else:  # M06
            cand = np.array([p for p in feasible if _prime_power(int(p)) is not None], dtype=np.int64)
            cand = np.array([p for p in cand if _prime_power(int(p))[0] not in used_bases],
                            dtype=np.int64)
        if cand.size == 0:
            return Construction(method, "D02", nu, "INFEASIBLE",
                                f"no feasible integer period at slot {j}",
                                {"infeasible_at": int(j)})
        p0 = 2.0 * math.pi / geom.nu_mr[j]
        dev = np.abs(np.log(cand.astype(float) / p0))
        best = int(np.argmin(dev * 1e12 + cand))     # ties -> smaller P
        P = int(cand[best])
        chosen[int(j)] = P
        prev = P
        nu[j] = 2.0 * math.pi / P
        if method == "M04":
            used_primes.add(P)
        elif method == "M06":
            used_bases.add(_prime_power(P)[0])

    if len(chosen) == 0:
        return Construction(method, "D02", nu, "NO_CHANGE", "no slot had a feasible period")
    # The exact joint recurrence is lcm(P_j), which for 16 pairwise-coprime
    # periods of order 1e4-1e5 is astronomically larger than int64.  numpy's
    # lcm.reduce wraps and returns a NEGATIVE number, so it is computed in
    # Python ints; the digit count is reported because the value is unusable
    # as a number but meaningful as "the subband never exactly repeats inside
    # any context we can deploy".
    lcm_exact = 1
    for p in chosen.values():
        lcm_exact = lcm_exact * int(p) // math.gcd(lcm_exact, int(p))
    return Construction(
        method, "D02", nu, "OK",
        f"{len(chosen)} integer periods, strictly increasing",
        {"periods": chosen,
         "lcm_exact_recurrence": str(lcm_exact),
         "lcm_digits": len(str(lcm_exact)),
         "lcm_exceeds_deployable_context": lcm_exact > 1_000_000},
    )


# ---------------------------------------------------------------------------
# D03  native dictionary reuse
# ---------------------------------------------------------------------------


def _dict_candidates(geom, j):
    """Indices k with omega_j/s <= omega_k <= omega_j.  No extrapolation."""
    lo = geom.omega[j] / geom.scale
    hi = geom.omega[j]
    idx = np.where((geom.omega >= lo - 1e-15) & (geom.omega <= hi + 1e-15))[0]
    return idx


def d03(geom, method):
    """Replace each T slot by a node from the shipped 64-frequency dictionary."""
    nu = geom.nu_mr.copy()
    picked = {}
    bracket = {}
    for j in geom.T:
        idx = _dict_candidates(geom, j)
        if idx.size == 0:
            return Construction(method, "D03", nu, "INFEASIBLE",
                                f"empty dictionary window at slot {j}", {"infeasible_at": int(j)})
        vals = geom.omega[idx]                       # decreasing in k
        target = geom.nu_mr[j]

        if method == "M07":
            best = int(idx[int(np.argmin(np.abs(vals - target)))])
            picked[int(j)] = best
            nu[j] = geom.omega[best]

        elif method == "M08":
            # bracketing pair in the dictionary's own order: lowering / raising node
            lower = idx[vals <= target]
            upper = idx[vals >= target]
            if lower.size and upper.size:
                l = int(lower[np.argmax(geom.omega[lower])])    # closest below
                u = int(upper[np.argmin(geom.omega[upper])])    # closest above
                # cumulative linear error: pick the sign that moves the running
                # error back toward zero
                acc = bracket.get("acc", 0.0)
                err_l = acc + (geom.omega[l] - target)
                err_u = acc + (geom.omega[u] - target)
                pick = l if abs(err_l) <= abs(err_u) else u
                bracket["acc"] = err_l if pick == l else err_u
                bracket["bracketed"] = bracket.get("bracketed", 0) + 1
            else:
                pick = int(idx[int(np.argmin(np.abs(vals - target)))])
                bracket["unbracketed"] = bracket.get("unbracketed", 0) + 1
            picked[int(j)] = pick
            nu[j] = geom.omega[pick]

        else:  # M09  unbiased dictionary rounding, fixed seed
            lower = idx[vals <= target]
            upper = idx[vals >= target]
            if lower.size and upper.size:
                l = int(lower[np.argmax(geom.omega[lower])])
                u = int(upper[np.argmin(geom.omega[upper])])
                wl, wu = geom.omega[l], geom.omega[u]
                frac = (target - wl) / (wu - wl) if wu != wl else 0.0
                rng = np.random.default_rng(20260911 + int(j))
                pick = u if rng.random() < frac else l
            else:
                pick = int(idx[int(np.argmin(np.abs(vals - target)))])
            picked[int(j)] = pick
            nu[j] = geom.omega[pick]

    counts = {}
    for v in picked.values():
        counts[v] = counts.get(v, 0) + 1
    collisions = {int(k): int(c) for k, c in counts.items() if c > 1}
    untouched = [int(j) for j in geom.T if abs(nu[j] - geom.nu_mr[j]) == 0]
    detail = {"picked": picked, "collisions": collisions, "n_collisions": len(collisions),
              "n_unchanged_slots": len(untouched)}
    if method == "M08":
        detail.update(bracket)
    if not picked:
        return Construction(method, "D03", nu, "NO_CHANGE", "no slot remapped")
    return Construction(method, "D03", nu, "OK",
                        f"{len(picked)} slots remapped, {len(collisions)} dictionary collisions",
                        detail)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

BUILDERS = {
    "M01": lambda g: d01(g, "M01"), "M02": lambda g: d01(g, "M02"),
    "M03": lambda g: d01(g, "M03"),
    "M04": lambda g: d02(g, "M04"), "M05": lambda g: d02(g, "M05"),
    "M06": lambda g: d02(g, "M06"),
    "M07": lambda g: d03(g, "M07"), "M08": lambda g: d03(g, "M08"),
    "M09": lambda g: d03(g, "M09"),
}

DIRECTION_OF = {m: m[:3].replace("M0", "D0").replace("M1", "D1") for m in BUILDERS}
CFREE_METHODS = tuple(BUILDERS)


def build_all(geom):
    return {m: BUILDERS[m](geom) for m in CFREE_METHODS}


def fingerprint(construction):
    """A stable identity for the resulting operator, for the section 5.4 dedup."""
    nu = np.asarray(construction.nu, dtype=np.float64)
    return {"method": construction.method, "status": construction.status,
            "nu_sha256": core_hash(nu)}


def core_hash(nu):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(nu.astype(np.float64)).tobytes()).hexdigest()
