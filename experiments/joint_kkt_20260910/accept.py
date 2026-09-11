"""sec.8 acceptance: the only thing standing between a model and a released table.

WHY THIS FILE EXISTS AT ALL.  The local model is a quasi-Newton model: real
gradients r_i and a positive-definite B_i that is NOT the measured Hessian
(`qcqp.py` states this, and every receipt has to carry it).  A model with real
gradients and an invented curvature will produce a confident step, and the step
will be wrong in exactly the regime that matters -- far from the measurement
point, which is where a long-range table wants to go.  So the step is not
released on the model's word:

    real  f_i(theta_k + d)  <=  q_i(d)      for every term i        (*)

`q_i(d) = r_i^T d + 0.5 d^T B_i d` is the model's prediction of the CHANGE, so
`f_i` on the left is the measured change `f_i(theta_k+d) - f_i(theta_k)` and
`q_i(0) = 0`.  Writing the test on raw values instead would silently compare two
quantities that differ by `f_i(theta_k)` and pass or fail for the wrong reason;
`check_acceptance` therefore works on deltas and reports the raw values beside
them so the convention is visible in the receipt.

THE GAIN IS NOT PRICED BY THE MODEL, AND THAT IS WHY (*) CANNOT BE SKIPPED.
The frozen Fisher F_N carries 64 frequency rows only -- the panel's arms all hold
the gain at MrPro's value, which is a design face (INTEGRATION R3) and a red line
in NEXT_DERIVATION ("F 不含 gain 自由度").  So the quadratic native model has no
gain coordinate and cannot price a gain move at all.  Two consequences:

  * the native constraint is MEASURED, never modelled, whenever `d_a != 0`.
    `check_acceptance` refuses a model-only native verdict in that case rather
    than reporting a number it cannot support.
  * the step's gain component is flagged `gain_unpriced` in the receipt, so a
    win that comes from the gain is not reported as a frequency-allocation win.
    That distinction is the whole of warning (a) and it belongs in a field, not
    in a paragraph.
  * `gain_unpriced` is a WARNING, NOT A REFUSAL.  It was in `reasons` -- and so
    in `accepted` -- in the first version, which refused every step that moved the
    gain and made the joint solve (sec.10C's main comparison) impossible.  The
    native test does not depend on the model, so a model with no gain coordinate
    cannot make a measured native verdict wrong.  See `check_acceptance`.

THE RHO BAND, AND WHY IT IS TWO-SIDED.  rho_trust is the measured long-range
decrease over the model's predicted decrease.  5.6pro fixes the band at
[0.5, 1.5].  A rho BELOW the band is the expected failure (the model promised
more than the real forward delivered) and is the reason the ladder shrinks the
step.  A rho ABOVE the band is not a bonus: it means the real forward improved
far more than the local model could account for, which says the quadratic is not
what is producing the gain -- the measurement is then evidence about something
other than this model, and accepting it would credit the wrong mechanism.  Both
ends escalate.  rho is undefined when the predicted decrease is ~0, and that is
reported as `rho=None` with a reason rather than as a large or small number.

THE LADDER.  sec.8's retry is not "try again": each failure mode has a different
remedy and the receipt has to say which was applied.

    infeasible step            -> shrink Delta (a smaller ball cannot make an
                                  inconsistent ordering floor consistent, but it
                                  distinguishes "step too big" from "floor wrong")
    model violated (*)         -> shrink Delta AND raise mu: the model is too
                                  optimistic at this radius, and more damping
                                  makes the next proposal less aggressive
    native budget exceeded     -> shrink Delta (the native check is the binding
                                  constraint; this is the expected failure)
    rho outside band           -> shrink Delta
    rho undefined              -> stop; the objective does not move at this
                                  radius, so nothing is being tested

The ladder is monotone: Delta only ever shrinks, mu only ever grows, so it cannot
cycle.  Exhausting it returns `accepted=False` with every attempt recorded --
never the last attempt relabelled as success.
"""
from __future__ import annotations

import numpy as np

from . import qcqp

# 5.6pro's pre-registered band.  Both ends escalate; see the module docstring.
RHO_BAND = (0.5, 1.5)
RHO_BAND_DEFAULT = RHO_BAND

# The native check is an output-KL in nats/token against the frozen native table.
# eps is the pre-registered native budget; it is an argument, never read off a
# curve (curvature_20260910/arms.py explains why it must be chosen in advance).
EPS_DEFAULT = 1e-3


# ---------------------------------------------------------------------------
# the predicate.  No model calls, so this is unit-testable on CPU and is the
# piece `selftest.py` exercises directly.
# ---------------------------------------------------------------------------
def model_values(entries, d):
    """q_i(d) for every entry: the model's predicted CHANGE, in entry order.

    Delegates to `qcqp.predicted` rather than repeating the formula: this value
    is the right-hand side of (*) and the solver optimises the same expression,
    so a copy here that drifted from the solver's would make the acceptance test
    check a model nobody solved.
    """
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    out = {}
    for e in entries:
        B = np.asarray(e["B"], dtype=np.float64)
        r = np.asarray(e["r"], dtype=np.float64).reshape(-1)
        out[e["name"]] = qcqp.predicted(B, r, d)[0]
    return out


def check_acceptance(entries, d, real_before, real_after, native_before,
                     native_after, eps=EPS_DEFAULT, rho_band=RHO_BAND_DEFAULT,
                     model_covers_gain=False, gain_index=None):
    """The whole sec.8 predicate, evaluated on measurements the caller made.

    `real_before` / `real_after` are dicts name -> measured value at theta_k and
    at theta_k + d.  `native_before` / `native_after` are the measured output-KL
    against the frozen native table at the same two points.

    THE NATIVE TEST IS ABSOLUTE, NOT A DELTA, AND THAT MATTERS ONCE THE LOOP
    ITERATES.  sec.4's constraint is  N(theta) <= N(theta_native) + eps  with
    N(theta_native) = 0 for an output-KL against the native table.  So the
    quantity compared against eps is `native_after` itself.  Reading it as a
    per-step delta is the same thing only on the first step: after that the
    reference point already carries a native drift, and a delta test would let the
    drift accumulate while every individual step looked affordable.  The delta is
    still reported (`native_change`) because it is what a step's receipt should
    show, but it is not what is tested.

    `model_covers_gain` is the caller's assertion that its B_i actually has a gain
    coordinate.  It defaults to False because the frozen F does not, and when it
    is False a nonzero gain component makes the native verdict model-inadmissible.
    """
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    q = model_values(entries, d)

    risk = [e for e in entries if e["kind"] == "risk"]
    native = [e for e in entries if e["kind"] != "risk"]
    if not risk:
        raise ValueError("no risk entries: there is no long-range claim to accept")
    missing = [e["name"] for e in risk if e["name"] not in real_after
               or e["name"] not in real_before]
    if missing:
        raise ValueError(f"no measurement for {missing} -- (*) cannot be checked "
                         "on a term that was not evaluated at both points")

    per_term = {}
    worst, worst_name = -np.inf, None
    for e in risk:
        name = e["name"]
        df = float(real_after[name]) - float(real_before[name])
        # (*).  violation > 0 means the real change was WORSE than the model
        # predicted -- the real loss fell by less than promised.  This is the
        # condition the damping exists to enforce: raising mu raises B, and
        # (*) holds once B dominates the real curvature, so the ladder's
        # "raise mu" is the remedy the algebra actually prescribes.
        viol = df - q[name]
        per_term[name] = dict(kind="risk", model=q[name], real=df,
                              raw_before=float(real_before[name]),
                              raw_after=float(real_after[name]),
                              violation=float(viol), ok=bool(viol <= 0.0))
        if viol > worst:
            worst, worst_name = float(viol), name

    # Native terms are deliberately NOT in the (*) max.  Their violation would be
    # measured against `limit` on the absolute scale while the risk terms' is a
    # delta -- two different quantities in one max, which lets a native overrun
    # masquerade as a (*) failure and (worse) masks the gain flag below.  Native
    # has its own check, on the MEASURED output-KL, and its model value is
    # recorded for the receipt only.
    for e in native:
        per_term[e["name"]] = dict(kind=e["kind"], model=q[e["name"]],
                                  real=float(native_after),
                                  real_change=float(native_after) - float(native_before),
                                  model_respects_limit=bool(q[e["name"]] <= float(e["limit"])),
                                  limit=float(e["limit"]), ok=True)

    pred_long = -min(q[e["name"]] for e in risk)
    real_long = -min(float(real_after[e["name"]]) - float(real_before[e["name"]])
                     for e in risk)

    # rho: the long-range side only, and only when the model actually predicted a
    # decrease.  A predicted decrease of ~0 makes rho a ratio of one small number
    # to another; that is reported as undefined, not as a large gain.
    #
    # `rho_state` IS A CODE, NOT A SENTENCE.  It used to carry the explanation
    # inline ("undefined: model predicts an INCREASE"), which made
    # `escalation_for`'s `rho_state == "undefined"` test -- the one rung of the
    # ladder that STOPS instead of shrinking -- impossible to reach, so an
    # undefined rho silently fell through to a shrink it could never satisfy.
    # The reason now lives in `rho_reason` and the code stays comparable.
    if not np.isfinite(pred_long) or pred_long <= 1e-12:
        # Two distinct cases with the same consequence, named separately so the
        # receipt distinguishes "this step does not move the objective" from
        # "this step is predicted to make it worse".
        rho = None
        rho_state = "undefined"
        rho_reason = ("model predicts an INCREASE" if pred_long < -1e-12 else
                      "model predicted no long-range decrease")
    else:
        rho = real_long / pred_long
        rho_state = "in_band" if rho_band[0] <= rho <= rho_band[1] else (
            "below_band" if rho < rho_band[0] else "above_band")
        rho_reason = None

    native_absolute = float(native_after)
    native_change = native_absolute - float(native_before)
    gain_moved = bool(gain_index is not None and d[gain_index] != 0.0)
    native_model_admissible = bool(model_covers_gain or not gain_moved)

    # `native_model_admissible` IS NOT AN ACCEPTANCE CRITERION.  It was one in the
    # first version, and that made the joint solve -- the plan's MAIN comparison
    # (sec.10C) -- impossible: every step that moved the gain was refused, because
    # the frozen F has no gain coordinate.  But the native constraint is MEASURED
    # here, not modelled (`native_after` is an argument; nothing in this function
    # consults `q` for the native verdict), so the model's inability to price a
    # gain move cannot make the measured verdict wrong.  What it makes inadmissible
    # is any claim that the MODEL predicted the native cost -- which is a statement
    # about the receipt, not about the step.
    parts = dict(
        model_satisfied=bool(worst <= 0.0),
        native_satisfied=bool(native_absolute <= eps),
        rho_ok=bool(rho is not None and rho_band[0] <= rho <= rho_band[1]),
    )
    accepted = all(parts.values())
    reasons = [k for k, v in parts.items() if not v]
    # a warning, deliberately kept OUT of `reasons`: `reasons` is what a refusal
    # lists, and a gain move is not a refusal -- it is a note about what the win
    # can be attributed to (INTEGRATION R3, warning (a))
    warnings = ["gain_unpriced"] if (gain_moved and not model_covers_gain) else []
    return dict(accepted=accepted, parts=parts, reasons=reasons,
                warnings=warnings,
                per_term=per_term, worst_violation=worst,
                worst_violation_of=worst_name, model=q,
                pred_long_decrease=float(pred_long), real_long_decrease=float(real_long),
                rho=rho, rho_state=rho_state, rho_reason=rho_reason,
                rho_band=[float(x) for x in rho_band],
                real_native_kl=native_absolute, native_change=native_change,
                native_before=float(native_before),
                native_after=float(native_after), eps=float(eps),
                gain_component=float(d[gain_index]) if gain_index is not None else None,
                gain_unpriced=bool(gain_moved and not model_covers_gain),
                # informational: whether the MODEL could price the native cost of
                # this step. False does not block the step (see `parts`), it says
                # the receipt must not attribute the native verdict to the model
                native_model_admissible=bool(native_model_admissible),
                d_norm=float(np.linalg.norm(d)),
                note="f_i measured on the same batch, full prefill, same "
                     "deployment path as the baseline")


def escalation_for(verdict):
    """Which ladder rung this failure calls for.  Named so a receipt shows the
    reasoning instead of just a smaller Delta."""
    if verdict["accepted"]:
        return None
    r = verdict["reasons"]
    if "rho_ok" in r and verdict["rho_state"] == "undefined":
        return dict(action="stop",
                    why=f"rho is undefined ({verdict.get('rho_reason')}): the "
                        "objective does not move at this radius, so nothing is "
                        "being tested and shrinking Delta would only make the "
                        "step smaller while the ratio stays undefined")
    if not verdict["parts"]["model_satisfied"]:
        return dict(action="shrink_and_damp", why="(*): the real forward exceeded the "
                                                  "model's own prediction at this radius")
    if not verdict["parts"]["native_satisfied"]:
        return dict(action="shrink", why="the native output-KL budget was exceeded -- "
                                         "this is the expected binding failure")
    if not verdict["parts"]["rho_ok"]:
        return dict(action="shrink", why=f"rho outside {verdict['rho_band']}: "
                                         f"{verdict['rho_state']}")
    return dict(action="shrink", why=f"unclassified failure {r}")


# ---------------------------------------------------------------------------
# the ladder.  Every model call is the caller's, injected, so this file stays
# testable without a checkpoint and the caller cannot accidentally let the
# ladder re-measure at a stale point.
# ---------------------------------------------------------------------------
def run_ladder(theta_k, propose, measure_terms, measure_native, entries,
               delta0, eps=EPS_DEFAULT, rho_band=RHO_BAND_DEFAULT,
               shrink=0.5, mu0=0.0, mu_growth=10.0, max_attempts=6,
               model_covers_gain=False, gain_index=None, min_delta=1e-8):
    """Propose, measure, decide, escalate; return the accepted step or a refusal.

    `propose(delta, mu)` returns (d, solve_receipt) -- it re-solves the QCQP at
    the given trust radius and damping, which is the expensive step, so it is
    called only when the ladder actually escalates.

    THE MODEL (*) IS CHECKED AGAINST IS THE ONE THAT WAS SOLVED, AND THAT IS NOT
    ALWAYS THE ONE THAT WAS PASSED IN.  A `propose` that damps must hand back the
    damped model in `solve_receipt["entries"]`; this function then prices (*) and
    rho against THAT model.  The alternative -- price against the undamped
    entries every attempt -- was tried and is provably unpassable for the exact
    failure damping is the remedy for: (*) fails when the real curvature exceeds
    the model's, and no radius changes that, because the violation is O(d^2) with
    the same sign at every scale.  Damping is what makes B dominate; if the check
    keeps looking at the undamped B, the ladder can only exhaust itself.

    The two-sided rho band is what keeps this honest.  With a heavily damped model
    (*) becomes easy to satisfy, so the acceptance test alone would get weaker as
    mu grows -- but the model's predicted decrease collapses at the same time, and
    a rho ABOVE the band is refused.  Damping therefore cannot buy a pass; it can
    only move the step into the window where the model is both conservative and
    still accounting for the gain.

    `measure_terms(theta)` returns name -> real value; `measure_native(theta)`
    returns the real output-KL against native.  BOTH are called at theta_k and at
    theta_k + d on every attempt: an acceptance check that reuses a cached value
    from the previous attempt would be checking a point that is no longer the
    candidate.
    """
    theta_k = np.asarray(theta_k, dtype=np.float64).reshape(-1)
    real_before = measure_terms(theta_k)
    native_before = measure_native(theta_k)
    # THE PRECONDITION IS FEASIBILITY, NOT PURITY.  This used to demand
    # native_before ~ 0 -- "the reference point is not the native table" -- which
    # is true only for the first step.  The loop's second step starts from a point
    # that already carries native drift, and the constraint it must respect is the
    # ABSOLUTE  N(theta) <= eps  (check_acceptance says why).  What has to be
    # refused here is starting the ladder from an infeasible point, because then
    # no step could be accepted and the refusal would be attributed to the step
    # instead of to the start.
    if native_before > eps * (1.0 + 1e-9):
        raise ValueError(
            f"native_before = {native_before:.6e} already exceeds the budget "
            f"{eps:.6e}: the start point is infeasible, so a refusal here would "
            "be about the start point and not about the step. Phase-I is what "
            "recovers feasibility, and it is not allowed to widen eps.")
    delta, mu = float(delta0), float(mu0)
    attempts = []
    for i in range(int(max_attempts)):
        d, solve = propose(delta, mu)
        d = np.asarray(d, dtype=np.float64).reshape(-1)
        entries_eff = solve.get("entries", entries)
        real_after = measure_terms(theta_k + d)
        native_after = measure_native(theta_k + d)
        v = check_acceptance(entries_eff, d, real_before, real_after, native_before,
                             native_after, eps=eps, rho_band=rho_band,
                             model_covers_gain=model_covers_gain,
                             gain_index=gain_index)
        v["attempt"] = i
        v["delta"] = delta
        v["mu"] = mu
        v["solve_feasible"] = bool(solve.get("feasible", True))
        v["model_damped"] = bool(entries_eff is not entries)
        if not v["solve_feasible"] and v["accepted"]:
            # A step that satisfies (*) numerically but was repaired onto the
            # feasible set is still a step whose direction was decided by the
            # repair, so the solver's own claim is recorded rather than dropped.
            v["note"] += "; solver reported an infeasible proposal that repair fixed"
        esc = escalation_for(v)
        v["escalation"] = esc
        attempts.append(v)
        if v["accepted"]:
            return dict(accepted=True, step=d, verdict=v, attempts=attempts,
                        n_attempts=len(attempts), theta_k=theta_k,
                        theta_new=theta_k + d)
        if esc is None or esc["action"] == "stop":
            break
        if esc["action"] == "shrink_and_damp":
            mu = mu * mu_growth if mu > 0 else mu_growth
        delta *= shrink
        if delta < min_delta:
            attempts[-1]["escalation"]["why"] += f"; Delta fell below {min_delta:g}, stopping"
            break
    return dict(accepted=False, step=None, verdict=attempts[-1] if attempts else None,
                attempts=attempts, n_attempts=len(attempts), theta_k=theta_k,
                theta_new=None,
                note="ladder exhausted: the last attempt is NOT accepted. A refused "
                     "step is a result -- it says the model could not be trusted at "
                     "any radius tried -- and must be reported as a refusal.")


def summarize_ladder(result):
    """One line per attempt, for a receipt and for the driver's log."""
    return [dict(attempt=a["attempt"], delta=a["delta"], mu=a["mu"],
                 accepted=a["accepted"], reasons=a["reasons"],
                 rho=a["rho"], rho_state=a["rho_state"],
                 rho_reason=a.get("rho_reason"),
                 real_native_kl=a["real_native_kl"],
                 worst_violation=a["worst_violation"],
                 worst_violation_of=a["worst_violation_of"],
                 action=(a["escalation"] or {}).get("action"))
            for a in result["attempts"]]
