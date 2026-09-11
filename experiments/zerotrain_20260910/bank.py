"""The pre-declared arm bank: what will be tried, and what each arm would show.

THE BANK IS DECLARED BEFORE ANY SCORE IS SEEN, and that is the whole point of
having it in a file rather than in someone's head.  A screen that can add an arm
after seeing a leaderboard is a scan, however it is described; a screen whose
arms and predictions are written down first is an experiment that may fail.  The
project has a rule against hidden multi-candidate scans, and the way to honour it
is not to run fewer arms but to say which ones and why, in advance.

THE CENTRAL ARM IS A ONE-PARAMETER FAMILY ANCHORED ON MrRoPE.  `tables.m_incr_power(r)`
sets the transition increments to eps_q ∝ q^r, and

    r = 1 reproduces MrRoPE EXACTLY (bit-for-bit, checked in selftest.py)

so the family is not a new curve that happens to be compared against MrRoPE: one
of its points IS MrRoPE.  The question "is the exponent optimal" then has a
pre-declared answer to compare against, and the two outcomes are both results:

  * the score improves for r > 1  ->  MrRoPE is NOT optimal.  No training, no
    tuning, and the direction (more front-protection, more back-concentration)
    is the one the current theory predicts should help.  That is a zero-training
    win and it is derivable, not searched.
  * the score is stationary at r = 1  ->  MrRoPE is a local optimum of the
    frozen-model long-range objective at this span, which is the EXPLAINS
    reading the project's own KKT framing is built to test.
  * the score PEAKS inside 0 < r < 1  ->  MrRoPE over-spends on front protection,
    and YaRN's flatter ramp was closer on this axis.  Also a result, and one that
    would need the two mechanisms separated.

RAISING r MOVES TWO THINGS AT ONCE, AND THE RECEIPT HAS TO SAY SO.  It reduces
the first transition slot's perturbation AND concentrates the compression later
AND lowers sum m.  Inside a fixed band with fixed endpoints those are not
separately controllable (`tables.band_report` gives the arithmetic why), so this
family measures the JOINT effect.  R4 makes sum m a free decision variable rather
than a conserved quantity, and the conserved span is held at exactly 1 for every
r, so the coupling is a limitation on interpretation and not a confound in the
design.  The screen reports sum m for every arm so a reader can see the coupling
rather than being told it is absent.
"""
from __future__ import annotations

import numpy as np

from ..curvature_20260910 import tables as T

K = T.K

# The r values that will be run.  Fixed here, not chosen per run.
R_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def _split(s, first="mrpro", second="yarn"):
    """MrPro/YaRN splice at slot s: `first` below s, `second` at and above.

    The corrected form of the K6/F9 counterfactual.  `max`/`min` on the two
    frequency vectors -- what the plan literally specifies -- differs from MrPro
    on only TWO slots (38, 39), because the two m-profiles cross at 37/38 and
    taking the larger frequency per slot therefore selects MrPro almost
    everywhere.  A split at s is what "A's front + B's back" actually means, and
    every s in the band keeps the table strictly monotone and the endpoints fixed
    (`analysis/route_read_20260910/K6_PREFLIGHT_20260910.md`).
    """
    src = {"mrpro": T.m_mrpro(17), "yarn": T.m_yarn(k=K)}
    idx = np.arange(K)
    return np.where(idx < s, src[first], src[second])


def _m(name):
    """Build the m-vector for a named arm."""
    if name in T.CONSTRUCTIONS:
        return np.asarray(T.CONSTRUCTIONS[name](), dtype=np.float64)
    if name.startswith("splitA_s"):
        return _split(int(name.split("_s")[1]), "mrpro", "yarn")
    if name.startswith("splitB_s"):
        return _split(int(name.split("_s")[1]), "yarn", "mrpro")
    raise KeyError(f"unknown arm {name!r}")


# ---------------------------------------------------------------------------
# The bank.  `prediction` is what the arm is FOR, written before measurement.
# ---------------------------------------------------------------------------

def _companding_arms():
    from .phase1_screen import companding_arms
    return companding_arms()


def _leak_arms():
    from .phase1_screen import leak_arms
    return leak_arms()


def arms(gain=None, gain_mrpro=True):
    """The arm list, in the order they will be run.

    The gain is held at one value for the whole screen.  Which value is a
    deliberate choice and it is not "whoever wins": the panel's frozen-F work has
    no gain coordinate ("F 不含 gain 自由度", NEXT_DERIVATION), so the gain is a
    DESIGN FACE (INTEGRATION R3) and a screen that let it float would be
    comparing tables under different faces.  MrRoPE's own `1 + 0.1 ln S` is the
    default because it is the incumbent's face; `gain` overrides it for the
    separate gain arm, which is scored as its own row and flagged.
    """
    g = float(T.GAIN_YARN if gain is None else gain)
    out = [
        dict(name="native", m=T.m_native(), gain=g, group="reference",
             prediction="costs nothing, buys nothing: the origin of the plane"),
        dict(name="yarn_lin", m=T.m_yarn(k=K), gain=g, group="incumbent",
             prediction="the published baseline; its frozen-Qwen score column is "
                        "EMPTY in the archive (R1 G5), so this row is new "
                        "information whichever way it lands"),
        dict(name="mrpro_n17", m=T.m_mrpro(17), gain=g, group="incumbent",
             prediction="the incumbent. Every other arm is read against this one"),
        dict(name="mrpro_n16", m=T.m_mrpro(16), gain=g, group="incumbent",
             prediction="MrRoPE's own family, one member narrower -- isolates "
                        "band WIDTH from band SHAPE without a new construction"),
    ]
    for r in R_VALUES:
        out.append(dict(
            name=f"incr_r{r:g}".replace(".", "p"), m=T.m_incr_power(r), gain=g,
            group="shape_family", r=float(r),
            prediction=("the anchor: must tie mrpro_n17 EXACTLY, so a mismatch "
                        "here is a harness bug and not a finding" if r == 1.0 else
                        f"back-loading r={r:g}; if the family improves past r=1 "
                        "MrRoPE is not optimal at this span")))
    # ---------------------------------------------------- mechanism separation
    # The r family moves BOTH mechanisms at once, so a score change along it
    # cannot be assigned to either.  These arms move one dial at a time, which is
    # what the theory question needs: WHICH structural feature of the YaRN family
    # carries its advantage.  Anchored at the incumbent again -- (a, r) =
    # (2/306, 1) is MrRoPE to within one ULP.
    a_mr = 2.0 / (17 * 18)
    for a in (0.001, 0.03, 0.10):
        out.append(dict(name=f"front_a{a:g}".replace(".", "p"),
                        m=T.m_incr_split(a, 1.0), gain=g, group="mechanism",
                        axis="front", a=float(a),
                        prediction=f"front-edge perturbation "
                                   f"{1 - 4.0 ** (-a):.2e} against MrRoPE's "
                                   f"{1 - 4.0 ** (-a_mr):.2e}, tail shape held at "
                                   "r=1. If the score is flat here, the front "
                                   "protection is NOT what carries the family"))
    for r in (0.0, 2.0, 4.0):
        out.append(dict(name=f"back_r{r:g}".replace(".", "p"),
                        m=T.m_incr_split(a_mr, r), gain=g, group="mechanism",
                        axis="back", r=float(r),
                        prediction=f"tail exponent r={r:g} against MrRoPE's 1, "
                                   "front perturbation held at MrRoPE's value. "
                                   "If the score is flat here, the back "
                                   "concentration is NOT what carries it"))
    for (a, r) in ((0.03, 2.0), (0.001, 2.0)):
        out.append(dict(name=f"both_a{a:g}_r{r:g}".replace(".", "p"),
                        m=T.m_incr_split(a, r), gain=g, group="mechanism",
                        axis="both", a=float(a), r=float(r),
                        prediction="both dials together; the interaction term "
                                   "that neither single-axis arm can see"))
    for s in (24, 28, 32, 36, 40):
        out.append(dict(name=f"splitA_s{s}", m=_split(s, "mrpro", "yarn"), gain=g,
                        group="splice", s=int(s),
                        prediction=f"MrPro front + YaRN back at s={s}; s=24 is "
                                   "all-YaRN and s=40 all-MrPro, so the ends are "
                                   "the two incumbents and the interior is the "
                                   "actual test"))
        out.append(dict(name=f"splitB_s{s}", m=_split(s, "yarn", "mrpro"), gain=g,
                        group="splice", s=int(s),
                        prediction=f"YaRN front + MrPro back at s={s}"))
    # ---------------------------------------------- support, not shape (new)
    # Declared AFTER round 1 and BEFORE these arms are scored, which is the only
    # property that makes a prediction worth recording.  Round 1 held the support
    # fixed and swept the shape; these hold the shape fixed and sweep the support.
    for name, m in _companding_arms():
        out.append(dict(name=name, m=m, gain=g, group="companding",
                        prediction="a GLOBAL companding: it touches every slot, "
                                   "including the ones the three-band family holds "
                                   "at m = 0. If the held plateau is load-bearing, "
                                   "this group pays for it in-window"))
    for name, m in _leak_arms():
        out.append(dict(name=name, m=m, gain=g, group="leak",
                        prediction="beta_b1 with a fraction of the compression "
                                   "moved below the band, sum(eps) held at 1. "
                                   "leak_a0 is beta_b1 exactly -- a mismatch there "
                                   "is a construction bug, not a finding. The "
                                   "first a at which in-window cost appears is "
                                   "the price of the held plateau"))
    return out


def describe(arm):
    m = np.asarray(arm["m"], dtype=np.float64)
    rec = T.band_report(m)
    rec.update(name=arm["name"], group=arm["group"], gain=float(arm["gain"]),
               r=arm.get("r"), s=arm.get("s"),
               prediction=arm["prediction"])
    return rec


def build(arm, cfg=T.QWEN25_3B):
    """-> {name, m, gain, theta, values_float32} ready for `install_table`."""
    m = np.asarray(arm["m"], dtype=np.float64)
    if m.shape != (K,):
        raise ValueError(f"{arm['name']}: expected {K} m entries, got {m.shape}")
    # THE GUARD IS ON nu, NOT ON m.  Requiring m non-decreasing is the natural
    # check and it is WRONG: the plan forbids it in as many words ("不要强制累计
    # 缩放 m 单调, 也不要默认 0 <= m <= 1; 这些会排除 EVQ 中的加速或非单调搬运").
    # nu = omega * S^-m, and ln(omega) falls by ln(theta)/K = 0.2159 per slot
    # while m is free, so m may fall by up to 0.2159/ln S = 1.557 in one slot and
    # nu still descends.  A splice that hands the back half to a less-compressed
    # profile does exactly that.  What must never happen is two slots at the same
    # frequency -- that is one slot, and the design silently changes meaning.
    nu_probe = T.m_to_inv_freq(m, cfg["theta"])
    if not (np.diff(nu_probe) < 0).all():
        raise ValueError(
            f"{arm['name']}: nu is not strictly descending, so two slots would "
            "collide and the design would silently reassign what each dimension "
            f"carries (min gap {float(np.diff(nu_probe).min()):.3e})")
    return dict(name=arm["name"], m=m, gain=float(arm["gain"]),
                theta=float(cfg["theta"]),
                values_float32=T.m_to_inv_freq(m, cfg["theta"]))
