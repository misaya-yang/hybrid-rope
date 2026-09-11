"""sec.8/sec.9: the sequential convex driver, its receipts, and Phase-I.

WHAT THIS FILE IS FOR.  Each module so far computes one thing correctly:
`design` fixes the coordinates, `qcqp` solves the local subproblem, `accept`
decides whether a step may be taken, `risk` says what the panel can and cannot
constrain.  This is where they are driven, and where the plan's two rules that
are not about arithmetic become mechanisms:

  * sec.9: "保持真实目标不变，反复重新计算loss/梯度并接受有限改动；不固定在
    MrPro处的一次Taylor展开."  Every step re-measures.  Nothing is carried over
    from the previous linearisation except the design point itself.
  * sec.8: "原生不可行的初始化必须做Phase-I可行性恢复。不要把限制改宽来隐藏
    不满足epsilon."  An infeasible start is repaired or reported, never excused.

THE OBJECTIVE INTERFACE IS INJECTED, AND THAT IS NOT A CONVENIENCE.  The driver
never imports torch.  It talks to an `Objective` with three methods, which lets
`selftest.py` exercise the entire loop -- the ladder, the receipt, the Pareto
continuation -- on a synthetic objective with a known answer, and lets a card
session supply the real one without the loop knowing what a checkpoint is.  A
driver that could only be tested on a GPU would be a driver that is first tested
on a GPU.

WHAT A STEP'S RECEIPT MUST CARRY, AND WHY EACH FIELD IS THERE.  sec.10B asks for
the native budget residual, the long-range change, the spectral/gain KKT residual
and a gap-transfer table.  Two more are added because omitting them would let a
result be read as something it is not:

  * `null_fraction` -- the share of the step lying in the directions the 30-row
    panel cannot price (`risk.span_guard` computes the rank).  A step that is
    mostly null-space is not evidence of a long-range gain even when the
    objective moved, and at n=30 < d=65 that is a live possibility on every step.
    The span it is measured against is a SNAPSHOT taken at `guard_iteration`;
    see `run` for when that stops being valid.
  * `gain_unpriced` -- whether the step moved the gain, which the frozen F has no
    coordinate for.  A win that comes from the gain is not a
    frequency-allocation win, and the field is what keeps the two apart.

WHAT THIS FILE DOES NOT DO.  It does not measure a Hessian: `B_i` comes from
`isotropic_curvature` unless the caller injects something better, and every
receipt says so.  It does not prove global optimality: the residual diagnostic
below is a first-order stationarity check on the panel's own objective, and
sec.9 is explicit that "残差近零只支持一阶驻点，不能证明全局最优".  It does not
touch a checkpoint, generate anything, or decide whether a table is good.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from . import accept as A
from . import design as D
from . import qcqp
from . import risk as RK


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Every number the loop uses, in one place, so a receipt can carry them.

    `eps` is the native budget.  It is PRE-REGISTERED (an argument, never read
    off a curve): curvature_20260910/arms.py explains why, and the reason holds
    here unchanged -- an eps chosen after seeing the frontier makes the frontier
    meaningless.
    """
    eps: float = 1e-3
    delta0: float = 0.05
    max_steps: int = 8
    max_attempts: int = 6
    shrink: float = 0.5
    mu_growth: float = 10.0
    delta_floor: float = 1e-6
    pin_ends: bool = True
    fix_span: bool = False
    rho_band: tuple = A.RHO_BAND
    curvature_scale: float = 0.5
    lambda_guard: float = 1e-6

    def receipt(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# the model of the objective, which is NOT the objective
# ---------------------------------------------------------------------------
def isotropic_curvature(r, delta, scale=0.5):
    """B = scale * ||r|| / delta * I, with scale < 1 DELIBERATELY.

    THE HONEST DEFAULT, AND IT IS NOT A HESSIAN.  sec.8 permits, for a first
    implementation, "真实梯度和正定上界/阻尼准Newton模型B_i，明确B_i不是已测
    真实Hessian".  This is that model: positive definite by construction, carrying
    no information about the real curvature beyond the gradient's own magnitude.
    Every receipt written by `run` carries the sentence.

    WHY THE SCALE IS BELOW ONE, IN ARITHMETIC.  Take a step of length delta along
    -r, and write B = beta*||r||/delta.  Then

        predicted decrease = ||r||*delta*(1 - beta/2)
        real decrease      = ||r||*delta*(1 - H*delta/(2||r||))     (H = true curve)
        rho                = (1 - H*delta/(2||r||)) / (1 - beta/2)

    For a nearly linear objective (H*delta << ||r||) this is rho = 1/(1 - beta/2).
    beta = 1 -- the value that puts the model's own minimiser exactly ON the trust
    boundary, which is the aggressive-looking choice -- therefore gives rho = 2 for
    EVERY such problem, outside 5.6pro's [0.5, 1.5] band.  A rho above the band is
    refused, and raising mu makes the model MORE conservative and pushes rho
    further up, so the ladder cannot recover: the failure is unreachable by any of
    its remedies.  Measured, not argued -- it is what `selftest.py` caught.
    Keeping beta <= 2/3 keeps rho in band with slack; the default 0.5 gives 1.33.

    The asymmetry is what makes this the safe direction to err in.  A model that is
    too WEAK fails (*) and the ladder fixes it by raising mu -- a remedy that
    works.  A model that is too STRONG fails rho from above and no remedy exists.
    So the default sits on the weak side, and the trust region, not the curvature,
    is what bounds a step.
    """
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(r))
    k = max(nrm, 1e-12) / max(float(delta), 1e-12)
    return float(scale) * k * np.eye(r.size)


def build_entries(risk_grads, native_grads, risk_values, native_values,
                  delta, eps, curvature=None, curvature_scale=0.5,
                  native_name=None):
    """The QCQP's term list, assembled from measured gradients.

    Risk terms enter with their GROUP MEAN gradient (the objective is a group
    mean, never a per-row fit -- `risk.guard` refuses the alternative) and a `B_i`
    from `curvature` or the isotropic default.

    THE NATIVE LIMIT IS ABSOLUTE, WHICH IS NOT WHAT IT SAYS HERE FIRST.  sec.4's
    constraint is  N(theta) <= N(theta_native) + eps = eps,  with the output-KL
    measured against the native table so the table itself reads zero.  The model
    predicts a CHANGE, so the constraint on that change is

        q_i(d) <= eps - N(theta_k)

    and that is the limit.  The first version of this function wrote
    `N(theta_k) + eps`, which coincides on the first step (N = 0) and is wrong by
    `2*N(theta_k)` afterwards -- a step that looks affordable while the drift
    accumulates.  `accept.check_acceptance` tests the absolute value, and the two
    have to agree or the subproblem is solving a different problem than the one
    being enforced.

    `native_name` selects the one native term the acceptance check measures.  The
    plan keeps more than one native metric ("两者分别定义和报告"), but the sec.8
    predicate compares a single measured output-KL, so the others are priced in
    the subproblem and reported in the receipt while exactly one is decisive.  A
    caller that does not say which is decisive gets an error rather than an
    arbitrary choice.
    """
    cur = curvature or (lambda r: isotropic_curvature(r, delta, curvature_scale))
    entries = []
    for name, g in risk_grads.items():
        entries.append(dict(name=name, kind="risk", r=np.asarray(g, np.float64),
                            B=np.asarray(cur(g), np.float64), limit=None))
    if not native_grads:
        raise ValueError("no native term: the constraint side of the problem is "
                         "what keeps the long-range gain from being free")
    if native_name is None and len(native_grads) > 1:
        raise ValueError(f"{len(native_grads)} native terms and no `native_name`: "
                         "the sec.8 predicate measures ONE output-KL, so which one "
                         "is decisive has to be stated, not defaulted")
    native_name = native_name or next(iter(native_grads))
    for name, g in native_grads.items():
        entries.append(dict(name=name, kind="native", r=np.asarray(g, np.float64),
                            B=np.asarray(cur(g), np.float64),
                            limit=float(eps) - float(native_values[name]),
                            measured=bool(name == native_name)))
    return entries


# ---------------------------------------------------------------------------
# sec.8 -- one accepted step
# ---------------------------------------------------------------------------
def one_step(obj, y, entries, cfg, native_name=None, rows_guard=None,
             verbose=False):
    """Propose, check, escalate, accept -- the whole of sec.8 for one step.

    `obj` supplies the real values.  They are re-measured at BOTH points on every
    attempt through `accept.run_ladder`; nothing here caches a value across an
    attempt, because a cached "after" would be the value at a point that is no
    longer the candidate.
    """
    # STEP form: solve_epigraph wants A d = b, and feasible_set returns the
    # point form A y = b.  design.step_rows does the one subtraction.
    A_mat, b_vec, G, h = D.step_rows(y, pin_ends=cfg.pin_ends,
                                     fix_span=cfg.fix_span)
    n = D.N_DESIGN

    def propose(delta, mu):
        # the damping is added to each group's own B, which is the same object as
        # adding mu*I to the assembled Q for a single shared step; the entries
        # returned in the receipt are the damped ones, because (*) must be
        # checked against the model that was actually solved (accept.py)
        damped = [dict(e, B=e["B"] + mu * np.eye(n)) for e in entries]
        out = qcqp.solve_epigraph(np.zeros(n), damped, A_mat, b_vec, G, h,
                                  np.eye(n), delta)
        return out["d"], dict(out, entries=damped)

    risk = [e for e in entries if e["kind"] == "risk"]
    native_measured = [e for e in entries
                       if e["kind"] != "risk" and e.get("measured")]
    if len(native_measured) != 1:
        raise ValueError(f"{len(native_measured)} measured native terms; the sec.8 "
                         "predicate compares exactly one output-KL")
    nname = native_measured[0]["name"]

    def measure_terms(theta):
        return obj.values(theta, names=[e["name"] for e in risk])

    def measure_native(theta):
        return float(obj.values(theta, names=[nname])[nname])

    res = A.run_ladder(y, propose, measure_terms, measure_native, entries,
                       delta0=cfg.delta0, eps=cfg.eps, rho_band=cfg.rho_band,
                       shrink=cfg.shrink, mu_growth=cfg.mu_growth,
                       max_attempts=cfg.max_attempts,
                       model_covers_gain=False, gain_index=D.A_INDEX)

    # `run_ladder` only carries a `note` on the refusal branch -- an accepted
    # ladder has nothing to explain.  Written as a default rather than an index
    # because the accepted branch is the common one and the KeyError there would
    # look like a missing receipt rather than a missing key.
    rec = dict(accepted=bool(res["accepted"]), n_attempts=res["n_attempts"],
               attempts=A.summarize_ladder(res),
               note=res.get("note", "accepted"),
               delta_final=(res["attempts"][-1]["delta"] if res["attempts"] else None),
               B_is_not_a_hessian=True)
    # the SAME top-level keys on both branches: a refusal that is missing the
    # fields an acceptance carries is a refusal nobody can aggregate over, and a
    # caller reading `receipt["rho"]` should not have to know which branch ran
    rec.update(_verdict_fields(res["attempts"][-1]) if res["attempts"] else {})
    if not res["accepted"]:
        rec["verdict"] = rec.get("rho")
        return dict(accepted=False, y=y, step=None, receipt=rec, ladder=res)

    d = res["step"]
    y_new = y + d
    # the same rows the subproblem was solved against, applied to the STEP -- the
    # acceptance ladder checks the model and the native budget, not the spectral
    # constraints, so the feasible set is verified here or not at all
    v_new = D.violations(y_new, y, A_mat, b_vec, G, h)
    rec["feasible_set_respected"] = bool(v_new["ineq_ok"] and v_new["call_ok"])
    rec["violations_after"] = v_new
    rec["design_change"] = D.describe(y_new)
    rec["design_before"] = D.describe(y)

    # the null-space share of the step: a 30-row panel sees at most 30 directions
    # of a 65-dimensional design, so a step can be large and still unpriced
    if rows_guard is not None:
        rec["null_fraction"] = RK.null_fraction(d, rows_guard)
        rec["panel_rank"] = int(rows_guard["rank"])
        rec["panel_null_dim"] = int(rows_guard["null_dim"])

    # the gain is a design variable the frozen F cannot price; a win that comes
    # from it is not a frequency-allocation win (INTEGRATION R3, warning (a))
    rec["gain_component"] = float(d[D.A_INDEX])
    rec["gain_unpriced"] = bool(d[D.A_INDEX] != 0.0)
    return dict(accepted=True, y=y_new, step=d, receipt=rec, ladder=res)


def _verdict_fields(v):
    return dict(rho=v["rho"], rho_state=v["rho_state"],
                rho_reason=v.get("rho_reason"),
                real_native_kl=v["real_native_kl"],
                worst_violation=v["worst_violation"],
                worst_violation_of=v["worst_violation_of"],
                model_satisfied=v["parts"]["model_satisfied"],
                native_satisfied=v["parts"]["native_satisfied"],
                pred_long_decrease=v["pred_long_decrease"],
                real_long_decrease=v["real_long_decrease"])


# ---------------------------------------------------------------------------
# sec.8 -- Phase-I
# ---------------------------------------------------------------------------
def phase_one(obj, y, cfg, native_name, max_steps=6, tol=1e-12,
              backtrack=(1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)):
    """Recover native feasibility WITHOUT touching eps.

    The temptation is to widen `eps` until the start point is admissible.  sec.8
    forbids exactly that ("不要把限制改宽来隐藏不满足epsilon"), because the budget
    is the statement of how much native ability the run is allowed to spend, and
    widening it turns a constraint into a description of whatever the solver
    happened to do.

    So the recovery minimises the NATIVE term itself, subject to the same spectral
    constraints and trust region, and returns the point it reached along with the
    residual.  If the residual does not come under the budget, the honest outcome
    is to stop and say so: the requested eps is not reachable from this start
    point, which is a result about the start point.

    WHY THIS BACKTRACKS, AND WHY THAT IS NOT BOLTING ON A LINE SEARCH.  The
    default curvature model is B = ||r|| / Delta, which puts the model's own
    minimiser exactly ON the trust boundary -- the most aggressive radius the
    region allows.  When the real curvature is larger than B, that step lands PAST
    the minimum, and without a check the iteration oscillates between two points
    forever (measured: a 2-cycle 2e-4 -> 4.5e-4 -> 2e-4 at the default settings).
    Every other step in this package is verified against the real function before
    it is taken; Phase-I runs outside the acceptance ladder, so it has to verify
    its own.  The check is the same one: did the real native term actually fall.
    """
    y0 = np.asarray(y, np.float64).copy()
    hist = []
    y = y0.copy()
    stalled = False
    for it in range(int(max_steps)):
        val = float(obj.values(y, names=[native_name])[native_name])
        hist.append(dict(iteration=it, native=val))
        if val <= cfg.eps:
            break
        g = obj.grads(y, names=[native_name])
        grad = g["grads"][0]
        A_mat, b_vec, G, h = D.step_rows(y, pin_ends=cfg.pin_ends,
                                         fix_span=cfg.fix_span)
        accepted = False
        for frac in backtrack:
            delta = cfg.delta0 * float(frac)
            if delta < cfg.delta_floor:
                break
            B = isotropic_curvature(grad, delta, cfg.curvature_scale)
            step = qcqp.solve_epigraph(
                np.zeros(D.N_DESIGN),
                [dict(name=native_name, kind="risk", r=grad, B=B, limit=None)],
                A_mat, b_vec, G, h, np.eye(D.N_DESIGN), delta)
            if not step["feasible"]:
                hist[-1].setdefault("solve_infeasible", []).append(
                    [float(frac), float(step["max_quad_violation"])])
                continue
            cand = _clip_gain(y + step["d"])
            cv = float(obj.values(cand, names=[native_name])[native_name])
            if cv < val - tol:
                y, val, accepted = cand, cv, True
                hist[-1]["step"] = dict(frac=float(frac), delta=float(delta),
                                        native_after=cv)
                break
        if not accepted:
            stalled = True
            hist[-1]["stalled"] = True
            break
    val = float(obj.values(y, names=[native_name])[native_name])
    return dict(y=y, started_at=y0, feasible=bool(val <= cfg.eps),
                native_final=val, native_start=hist[0]["native"] if hist else val,
                stalled=stalled, history=hist,
                note=("eps was NOT widened; if this is infeasible the requested "
                      "budget is unreachable from this start point"))


def _clip_gain(y):
    """Put the gain back inside its box after a Phase-I step.

    Phase-I's subproblem carries the gain box like any other solve, but it runs
    outside the acceptance ladder, so nothing downstream would catch a box
    violation.  Cheap, and the alternative is a Phase-I point that the main loop
    then reports as infeasible for a reason that has nothing to do with eps.
    """
    y = np.asarray(y, np.float64).copy()
    lo, hi = D.a_box()
    y[D.A_INDEX] = float(np.clip(y[D.A_INDEX], lo, hi))
    return y


# ---------------------------------------------------------------------------
# sec.5/sec.9 -- the diagnostics a receipt must carry
# ---------------------------------------------------------------------------
def stationarity_diagnostic(risk_grads, native_grads, active=None):
    """The sec.5 residual: how far the weighted long-range gradient is from being
    balanceable by the native constraint.

    With `e` the active long-range gradient and `n` the native gradient,

        lambda_hat = max(0, -n^T e / n^T n)
        residual   = e + lambda_hat * n

    and the READING IS ASYMMETRIC.  A nonzero residual names a direction that the
    current objective would still move along at no native price -- a real finding.
    A near-zero residual supports FIRST-ORDER stationarity on this panel and
    nothing more: sec.5 is explicit that "残差近零只支持一阶驻点，不能证明全局最优",
    and this panel has 65 design dimensions against at most 6 group gradients, so
    the residual is being computed in a space far smaller than the design.

    `active` names the risk group that is actually binding; when it is None the
    group with the largest gradient norm is used, which is a different thing and
    is reported as such.
    """
    if not risk_grads:
        raise ValueError("no risk gradients")
    if active is None:
        active = max(risk_grads, key=lambda k: float(np.linalg.norm(risk_grads[k])))
        how = "largest-gradient group (not the measured active set)"
    else:
        how = f"declared active group {active!r}"
    e = np.asarray(risk_grads[active], np.float64).reshape(-1)
    if not native_grads:
        return dict(active=active, selected_by=how, lambda_hat=None,
                    residual_norm=float(np.linalg.norm(e)), residual=None,
                    cos_en=None, note="no native term to balance against")
    n = np.asarray(next(iter(native_grads.values())), np.float64).reshape(-1)
    nn = float(n @ n)
    if nn <= 0:
        return dict(active=active, selected_by=how, lambda_hat=None,
                    residual_norm=float(np.linalg.norm(e)), residual=None,
                    cos_en=None, note="native gradient is zero")
    lam = max(0.0, -float(n @ e) / nn)
    resid = e + lam * n
    return dict(active=active, selected_by=how, lambda_hat=float(lam),
                residual_norm=float(np.linalg.norm(resid)),
                residual=[float(x) for x in resid],
                relative_residual=float(np.linalg.norm(resid)
                                        / max(np.linalg.norm(e), 1e-300)),
                cos_en=float((n @ e) / (np.linalg.norm(n) * np.linalg.norm(e)
                                        + 1e-300)),
                note="nonzero residual = a direction still movable at no native "
                     "price; zero residual = first-order stationarity on THIS "
                     "panel only, not global optimality")


def gap_transfer_table(v, top=8, k=D.K):
    """The most valuable gap transfers at this point: (i, k, price_k - price_i).

    sec.5: "资源由高价格gap移向低价格gap；自由gap在最优点价格相同."  The table is
    the receipt's version of that sentence -- it names which pair of gaps the
    objective would still pay to move budget between, and by how much per unit.
    The prices come from a gradient handed in, so this is only as meaningful as
    that gradient's objective (see design.gap_prices).
    """
    v = np.asarray(v, np.float64).reshape(-1)
    prices = D.gap_prices(v, k=k)
    pairs = []
    for i in range(1, k - 1):
        for j in range(i + 1, k - 1):
            d = float(prices[j] - prices[i])
            pairs.append((abs(d), i, j, d))
    pairs.sort(reverse=True)
    return [dict(gap_from=i, gap_to=j, price_difference=d)
            for _a, i, j, d in pairs[:int(top)]]


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def run(obj, y0, cfg=None, native_name=None, row_grads=None, verbose=True,
        log_path=None):
    """The sequential convex loop.

    `obj` implements:
        values(y, names=None)  -> {name: float}          forward-only
        grads(y, names=None)   -> {names, grads[n,65], values}
    `row_grads` (optional) is the (n_rows, 65) per-row gradient matrix used for
    the span guard.  It is measured ONCE, at the start, and reused.

    THAT IS A SNAPSHOT, AND THE RECEIPT SAYS SO.  The guard answers "which
    directions can this panel price", and the per-row gradients do move with the
    design point, so the rank is a property of the graph as it stood at
    `guard_iteration` (0).  Under a FIXED attention graph that is harmless --
    every row still exists at every step and the span moves slowly.  Under a
    graph that can CHANGE it is not: a top-k or routed model can drop a row
    entirely, and then `null_fraction` would be reporting the null space of an
    objective that no longer exists.  `guard_iteration` is what makes that
    visible in the receipt; a routed caller must refresh `row_grads` each step
    (one extra backward) rather than reuse a stale span.
    """
    cfg = cfg or Config()
    y0 = np.asarray(y0, np.float64).reshape(-1)
    started = time.time()
    guard = RK.span_guard(row_grads) if row_grads is not None else None
    risk_names_all = [n for n in obj.names() if not n.startswith("native")]
    risk_at_start = obj.values(y0, names=risk_names_all)

    steps, refusals = [], []
    y = y0.copy()

    # Phase-I first, and its result is recorded whether or not it succeeded
    native_names = [n for n in obj.names() if n.startswith("native")]
    if not native_names:
        raise ValueError("the objective exposes no native term (names beginning "
                         "with 'native'); the constraint side is required")
    nname = native_name or native_names[0]
    pi = phase_one(obj, y, cfg, nname)
    y = pi["y"]
    if not pi["feasible"]:
        # the requested budget is not reachable from this start point.  That is a
        # result about eps and the start point, and returning a table anyway would
        # be reporting an infeasible point as a solution.
        return dict(status="phase_one_failed", y=y, y0=y0, phase_one=pi,
                    steps=[], config=cfg.receipt(),
                    note="Phase-I could not reach the native budget from this "
                         "start point; eps was not widened and no step was taken")
    if verbose:
        print(f"Phase-I: native {pi['native_start']:.5f} -> {pi['native_final']:.5f} "
              f"(budget {cfg.eps:g})")

    for it in range(int(cfg.max_steps)):
        g = obj.grads(y)
        risk_g = {n: g["grads"][i] for i, n in enumerate(g["names"])
                  if not n.startswith("native")}
        nat_g = {n: g["grads"][i] for i, n in enumerate(g["names"])
                 if n.startswith("native")}
        risk_v = {n: g["values"][i] for i, n in enumerate(g["names"])
                  if not n.startswith("native")}
        nat_v = {n: g["values"][i] for i, n in enumerate(g["names"])
                 if n.startswith("native")}

        entries = build_entries(risk_g, nat_g, risk_v, nat_v, cfg.delta0, cfg.eps,
                                curvature_scale=cfg.curvature_scale,
                                native_name=nname)
        out = one_step(obj, y, entries, cfg, native_name=nname,
                       rows_guard=guard, verbose=verbose)
        diag = stationarity_diagnostic(risk_g, nat_g)
        out["receipt"]["stationarity"] = diag
        out["receipt"]["warnings"] = list(out["ladder"]["attempts"][-1].get("warnings", [])) \
            if out["ladder"]["attempts"] else []
        out["receipt"]["gap_transfers"] = gap_transfer_table(
            np.mean([risk_g[n] for n in risk_g], axis=0))
        out["receipt"]["iteration"] = it
        # the budget residual AFTER the step: negative means the step stayed
        # inside eps, positive means it overran -- and an overrun cannot be
        # accepted, so a positive value here can only come from a bug
        out["receipt"]["native_budget_residual"] = float(
            out["receipt"]["real_native_kl"] - cfg.eps)
        if verbose:
            tag = "ACCEPT" if out["accepted"] else "REFUSE"
            print(f"step {it}: {tag} native_kl={out['receipt'].get('real_native_kl', float('nan')):.3e} "
                  f"rho={out['receipt'].get('rho')} "
                  f"null={out['receipt'].get('null_fraction')}")
        if out["accepted"]:
            y = out["y"]
            steps.append(out["receipt"])
        else:
            refusals.append(out["receipt"])
            break

    final = obj.grads(y)
    risk_names = [n for n in final["names"] if not n.startswith("native")]
    nat_names = [n for n in final["names"] if n.startswith("native")]
    result = dict(
        status="ok" if steps else "no_step_accepted",
        y=y, y0=y0, steps=steps, refusals=refusals,
        n_steps=len(steps), n_refused=len(refusals),
        risk_final={n: final["values"][final["names"].index(n)]
                    for n in risk_names},
        risk_initial=risk_at_start, native_name=nname,
        native_final={n: final["values"][final["names"].index(n)]
                      for n in nat_names},
        design_final=D.describe(y), design_initial=D.describe(y0),
        phase_one=dict(native_start=pi["native_start"],
                       native_final=pi["native_final"], iterations=len(pi["history"])),
        guard=guard, guard_iteration=(0 if guard is not None else None),
        guard_is_a_snapshot=guard is not None,
        config=cfg.receipt(),
        B_is_not_a_hessian=True,
        wall_seconds=time.time() - started,
        no_step_note=("a run that accepted no step is a result -- it says the "
                      "model could not be trusted at any radius tried, or the "
                      "objective does not move -- and must be reported as one"),
    )
    if log_path:
        with open(log_path, "w") as f:
            json.dump(result, f, indent=1, default=str)
    return result


def pareto(obj, y0, epsilons, cfg=None, native_name=None, **kw):
    """sec.9 -- the frontier, from a sequence of separately-solved budgets.

    Each eps is solved from the SAME start point, not warm-started from the
    previous solution: warm starting would make the later points depend on the
    earlier path, and the plan's `dE*/deps = -lambda` reading is a statement about
    a KKT branch, not about a trajectory.

    The sensitivity is reported alongside the achieved pairs.  It is a CHECK, not
    a derivation: the plan's identity is exact on a smooth branch with one active
    native constraint, and the numbers here will not satisfy it exactly because
    the steps are finite and the accepted points are not exactly KKT.
    """
    rows = []
    base = (cfg or Config()).receipt()
    for eps in epsilons:
        cf = Config(**{**base, "eps": float(eps)})
        r = run(obj, y0, cfg=cf, native_name=native_name, **kw)
        last = r["steps"][-1] if r["steps"] else None
        lam = (last or {}).get("stationarity", {}).get("lambda_hat")
        # CUMULATIVE long-range change from the start, not the last step's own
        # decrease: a per-step figure is not comparable across budgets (a longer
        # run has smaller last steps), and the frontier is a statement about
        # where the run ended, not about where it stopped moving.
        total = {n: (r["risk_initial"][n] - v)
                 for n, v in r["risk_final"].items()} if r.get("risk_final") else {}
        rows.append(dict(eps=float(eps), status=r["status"],
                         n_steps=r["n_steps"],
                         native_kl=(r.get("native_final") or {}).get(
                             r.get("native_name") or native_name),
                         long_total_decrease=total,
                         long_decrease=sum(total.values()) / len(total) if total else None,
                         lambda_hat=lam))
    # THE SLOPE IS WRITTEN IN THE PLAN'S OWN SIGN CONVENTION, so the two columns
    # can be read against each other.  sec.9 states  dE*/deps = -lambda  for a
    # minimisation, i.e. the LOSS falls as the budget grows and the multiplier is
    # positive.  The rows above carry the DECREASE (a positive-is-better quantity),
    # so its slope is the negative of the plan's; flipping it here is what stops a
    # correct result from looking like a sign error.
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if a["native_kl"] is None or b["native_kl"] is None:
            continue
        d_eps = b["eps"] - a["eps"]
        if d_eps == 0:
            continue
        d_loss = -((b["long_decrease"] or 0.0) - (a["long_decrease"] or 0.0))
        b["slope_dloss_deps"] = d_loss / d_eps
        b["predicted_dloss_deps"] = -(b["lambda_hat"] or float("nan"))
        lam = b["lambda_hat"]
        b["slope_agrees_with_lambda"] = bool(
            lam not in (None, 0.0) and math.isfinite(d_loss / d_eps)
            and 0.25 <= (d_loss / d_eps) / (-lam) <= 4.0)
    return dict(frontier=rows, config=(cfg or Config()).receipt(),
                note="each eps solved from the same start point. The slope columns "
                     "CHECK the plan's dE*/deps = -lambda; they do not confirm it -- "
                     "the steps are finite and the accepted points are not exactly "
                     "KKT, so agreement is order-of-magnitude at best")
