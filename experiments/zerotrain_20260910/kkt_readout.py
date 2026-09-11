"""Turn the screen's (price, benefit) plane into the KKT reading it is run for.

The screen measures two numbers per arm: `native_kl` (the constraint side, what
the table costs in behavioural drift from native) and `long_nll` (the objective
side, what it buys beyond the window).  Those two are the plan's problem,

    minimise  L_long(nu)          subject to   D_N(nu) <= eps,

sampled at many points.  The KKT statement about such a problem is a statement
about DIRECTIONAL DERIVATIVES: at an optimum, the objective cannot fall any
further without paying more constraint, so along every admissible direction the
two derivatives are proportional, with the multiplier as the constant of
proportionality.  A leaderboard does not answer that; a slope does.

WHAT IS COMPUTED HERE, AND WHAT IT IS NOT.

  * `pareto_price` -- the local exchange rate at the incumbent, taken from the
    frontier the bank itself traces: how much `long_nll` falls per unit of
    `native_kl`, estimated from the arms nearest the anchor on the passage.

  * `slope_along_dial` -- for each named dial (the front-perturbation arm series
    and the tail-exponent series), the two derivatives dD/d(dial) and dL/d(dial)
    by finite difference over that series.

  * `residual` -- dL/d(dial) - price * dD/d(dial), which is the change in the
    LAGRANGIAN L + lambda*D along the dial with the multiplier lambda = -price.
    THE MINUS IS LOAD-BEARING AND WAS WRONG FIRST TIME ROUND.  `price` is
    dL*/d(eps), which is normally NEGATIVE (spending native drift buys a lower
    long_nll), while the multiplier in the KKT condition is lambda = -dL*/d(eps)
    and is non-negative.  Writing `+ price * dD` instead of `- price * dD` flips
    the residual's meaning on every dial; the synthetic known-answer test in
    `selftest.py` is what caught it, and it is the reason that test exists.
    A residual of zero means the dial is priced
    consistently with the frontier -- moving along it is neither a free
    improvement nor a waste.  A residual that is NEGATIVE means the objective
    falls faster than the constraint makes you pay, i.e. there is an unclaimed
    improvement in that direction and the incumbent is not a KKT point along it.
    A residual that is POSITIVE means the dial overspends.

  * `kkt_consistent` -- |residual| below a threshold taken from the measurement's
    own resolution, not chosen for comfort.  The threshold and how it was set are
    both in the receipt.

THE THREE THINGS THIS CANNOT SAY, ALL OF WHICH ARE IN THE RECEIPT.

  1. It is a statement about the TWO DIALS SAMPLED, not about the whole design
     space.  A zero residual on both dials means the incumbent is stationary
     along those two directions and along nothing else.  With 64 frequencies and
     two dials, that is a very thin claim and it is reported as one.
  2. The price is estimated from a handful of arms, so it carries the noise of
     the finite difference rather than a confidence interval.  The receipt
     carries the arm count behind each number.
  3. `long_nll` is held-out language modelling, not a task score.  A residual
     here is a statement about this objective on this corpus, and the panel jobs
     that would make it a statement about capability are a separate step.
"""
from __future__ import annotations

import numpy as np

# The dial series, by the prefix of the arm names the bank produces.  Fixed here
# rather than discovered from whatever ran, so a partially-completed screen
# cannot silently change which axis is being read.
DIALS = {
    "front_perturbation": dict(prefix="front_a", param="a", anchor=2.0 / 306.0),
    "tail_concentration": dict(prefix="back_r", param="r", anchor=1.0),
}


def _by_name(rows):
    return {r["name"]: r for r in rows if r.get("status") == "ok"}


def _arm_value(name, spec):
    """Recover the dial's value from the arm name, and check it against the bank."""
    tail = name.split(spec["prefix"], 1)[1].replace("p", ".")
    return float(tail)


def pareto_price(named, anchor="mrpro_n17", k=4):
    """The local exchange rate at the anchor, from its nearest neighbours.

    Estimated as the slope of long_nll against native_kl over the k arms closest
    to the anchor in native_kl -- a LOCAL estimate on purpose, because the
    frontier is not a line and a global fit would report the wrong price at the
    point the dials are anchored to.
    """
    if anchor not in named:
        return dict(price=None, n=0, reason=f"anchor {anchor} did not run")
    a = named[anchor]
    others = [r for n, r in named.items() if n != anchor]
    if len(others) < 2:
        return dict(price=None, n=len(others), reason="fewer than 2 other arms")
    others.sort(key=lambda r: abs(r["native_kl"] - a["native_kl"]))
    sel = others[:int(k)]
    x = np.array([r["native_kl"] - a["native_kl"] for r in sel])
    y = np.array([r["long_nll"] - a["long_nll"] for r in sel])
    if np.abs(x).max() < 1e-12:
        return dict(price=None, n=len(sel),
                    reason="the neighbours are at the same native_kl, so there is "
                           "no price to read")
    price = float(np.sum(x * y) / np.sum(x * x))
    return dict(price=price, n=len(sel),
                neighbours=[r["name"] for r in sel],
                note=("d(long_nll)/d(native_kl) at the anchor, least squares over "
                      "the nearest arms; a NEGATIVE price is the normal case -- "
                      "spending native drift buys a lower long_nll"))


def dial_readout(named, price, threshold=None):
    """The residual along each declared dial."""
    out = {}
    for label, spec in DIALS.items():
        series = []
        for name, row in named.items():
            if not name.startswith(spec["prefix"]):
                continue
            try:
                v = _arm_value(name, spec)
            except (ValueError, IndexError):
                continue
            series.append((v, row))
        if len(series) < 2:
            out[label] = dict(n=len(series), residual=None,
                              reason="fewer than 2 arms on this dial")
            continue
        series.sort()
        vals = np.array([s[0] for s in series])
        dkl = np.array([s[1]["native_kl"] for s in series])
        dln = np.array([s[1]["long_nll"] for s in series])
        # fit each side against the dial by least squares, so the estimate uses
        # every arm on the dial rather than one arbitrary pair
        A = np.vstack([vals, np.ones_like(vals)]).T
        sD = np.linalg.lstsq(A, dkl, rcond=None)[0][0]
        sL = np.linalg.lstsq(A, dln, rcond=None)[0][0]
        # Lagrangian L + lambda*D with lambda = -price; see the module docstring
        resid = float(sL - price * sD)
        out[label] = dict(
            n=len(series), param=spec["param"], anchor=float(spec["anchor"]),
            arms=[s[1]["name"] for s in series],
            dial_values=[float(v) for v in vals],
            d_native_kl_d_dial=float(sD), d_long_nll_d_dial=float(sL),
            residual=resid,
            lambda_hat=float(-price) if price is not None else None,
            residual_over_scale=float(abs(resid) / max(abs(sL), abs(price * sD),
                                                        1e-300)),
            reading=("objective falls faster than the constraint charges -- an "
                     "unclaimed improvement, incumbent NOT stationary on this "
                     "dial" if resid < 0 else
                     "objective rises faster than the constraint charges -- the "
                     "dial overspends past the incumbent" if resid > 0 else
                     "priced consistently with the frontier"),
        )
    if threshold is not None:
        for label in out:
            if out[label].get("residual") is not None:
                out[label]["kkt_consistent"] = bool(
                    abs(out[label]["residual"]) <= threshold)
    return out


def readout(rows, anchor="mrpro_n17", threshold=None, k=4):
    """The whole reading, ready for the receipt."""
    named = _by_name(rows)
    price = pareto_price(named, anchor=anchor, k=k)
    dials = dial_readout(named, price["price"], threshold) if price.get("price") \
        else {}
    return dict(
        anchor=anchor, pareto=price, dials=dials,
        threshold=threshold,
        n_arms_read=len(named),
        scope=("statements about the TWO DIALS SAMPLED at the anchor, not about "
               "the 64-slot design space; a zero residual here is stationarity "
               "along two named directions and nothing more"),
        objective=("long_nll is held-out language modelling, not a task score; a "
                   "residual is a statement about this objective on this corpus"),
        note=("residual = d(long_nll)/d(dial) - price * d(native_kl)/d(dial), "
              "the Lagrangian change with lambda = -price; zero means the dial is "
              "priced consistently with the frontier"),
    )
