#!/usr/bin/env python3
"""Phase 1 round 1: score the declared arm bank on the frozen Qwen3B NLL set.

Self-contained on purpose.  The server's checkout is a partial copy
(`experiments/` + `analysis/` only, no repo root, no .git), so importing a
package would depend on how that copy happens to be laid out.  The table
constructions below are pure NumPy and are copied VERBATIM from
`experiments/curvature_20260910/tables.py`, with the anchor check that proves it
(`incr_r1 == MrPro` bit-for-bit) run before anything touches the GPU.

WHAT IT MEASURES.  Tail-512 next-token NLL at prefix lengths 8192/16384/32768,
on the 16 archived FineWeb-Edu documents -- the project's own established
instrument (`experiments/nongeometric_screen/long_eval.py` uses the same
form).  The archived MrPro numbers for exactly these cells already exist, so
MrRoPE is NOT re-run: its row is read from `run_nll_01/MrPro.jsonl` and written
beside the new arms.  That is the GPU saving the campaign asks for, and it is
also why this file must not silently change the corpus or the tail length --
the comparison is only valid against the archived cells it is paired with.

WHAT IT IS NOT.  All three lengths are inside the 32768 native window, so this
is the IN-WINDOW instrument: it answers "did the table cost anything at the
distances the checkpoint was trained for".  The extrapolation question lives in
the 131072 RULER rows and is a separate, more expensive run.  A receipt that
reports these numbers as long-context capability would be misreading them.

BATCHING.  The archived harness evaluates one document per forward, which leaves
a 32 GB card mostly idle.  Documents of the same length are batched here, and
the batch size is tuned by measurement rather than assumed: the first length is
run at a probe batch, memory is read, and the batch is raised to fill the card.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

K = 64
LN_S = math.log(4.0)
QWEN25_3B = dict(theta=1_000_000.0, head_dim=128, window=32768, scale=4)
GAIN_YARN = 1.0 + 0.1 * LN_S


# ---------------------------------------------------------------------------
# table constructions, verbatim from experiments/curvature_20260910/tables.py
# ---------------------------------------------------------------------------
def native_inv_freq(theta, k=K):
    return theta ** (-np.arange(k, dtype=np.float64) / k)


def m_to_inv_freq(m, theta, k=K):
    return native_inv_freq(theta, k) * np.power(4.0, -np.asarray(m, dtype=np.float64))


def yarn_bands(alpha=1.0, beta=32.0, cfg=QWEN25_3B):
    dim, theta, W = cfg["head_dim"], cfg["theta"], cfg["window"]
    lo = dim * math.log(W / (beta * 2.0 * math.pi)) / (2.0 * math.log(theta))
    hi = dim * math.log(W / (alpha * 2.0 * math.pi)) / (2.0 * math.log(theta))
    return lo, hi, int(math.floor(lo)), int(math.ceil(hi))


def m_yarn(scale=4.0, alpha=1.0, beta=32.0, cfg=QWEN25_3B, k=K):
    _, _, lo, hi = yarn_bands(alpha, beta, cfg)
    ramp = np.clip((np.arange(k) - lo) / (hi - lo), 0.0, 1.0)
    ratio = 1.0 - (1.0 - 1.0 / scale) * ramp
    return -np.log(ratio) / math.log(scale)


def m_mrpro(n=17, low=23, k=K):
    m = np.zeros(k)
    for j in range(k):
        q = min(max(j - low, 0), n)
        m[j] = q * (q + 1) / (n * (n + 1))
    m[low + n:] = 1.0
    return m


def m_incr_power(r, n=17, low=23, k=K):
    q = np.arange(1, int(n) + 1, dtype=np.float64)
    w = np.power(q, float(r))
    mq = np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    out = np.zeros(k)
    out[low:low + int(n) + 1] = mq
    out[low + int(n) + 1:] = 1.0
    return out


def m_incr_split(a, r, n=17, low=23, k=K):
    a = float(a)
    if not (0.0 < a < 1.0):
        raise ValueError(f"a = {a} is not a share in (0, 1)")
    q = np.arange(2, int(n) + 1, dtype=np.float64)
    w = np.power(q, float(r))
    rest = (1.0 - a) * w / w.sum()
    eps = np.concatenate([[a], rest])
    mq = np.concatenate([[0.0], np.cumsum(eps)])
    out = np.zeros(k)
    out[low:low + int(n) + 1] = mq
    out[low + int(n) + 1:] = 1.0
    return out


def m_incr_beta(b, n=17, low=23, k=K):
    """eps_k ~ k (n+1-k)^b.  b=0 IS MrRoPE and b=1 IS the deployed BM table --
    both reproduced to float32 precision; they are the two ends of the only
    family this project has whose endpoints are real, measured, and opposite."""
    kk = np.arange(1, int(n) + 1, dtype=np.float64)
    w = kk * np.power(int(n) + 1 - kk, float(b))
    eps = w / w.sum()
    mq = np.concatenate([[0.0], np.cumsum(eps)])
    out = np.zeros(k)
    out[low:low + int(n) + 1] = mq
    out[low + int(n) + 1:] = 1.0
    return out


def band_from_turns(alpha, beta, theta, window, head_dim, k=K):
    """Band edges in SLOTS from the turn-count window they correspond to.

    YaRN's find_correction_dim is normally read as a formula with two tuned
    constants.  Substituting omega_j = theta^(-2j/head_dim) into
    W*omega/(2*pi) = turns gives

        j = head_dim * ln(W / (turns * 2 pi)) / (2 ln theta)

    which is YaRN's expression with turns = alpha at the slow edge and beta at
    the fast edge.  So alpha and beta ARE turn counts, and the rule is
    model-independent because turns are dimensionless: the same (alpha, beta)
    lands on different slots for different (theta, window, head_dim).
    """
    lo = head_dim * math.log(window / (float(beta) * 2.0 * math.pi)) / (2.0 * math.log(theta))
    hi = head_dim * math.log(window / (float(alpha) * 2.0 * math.pi)) / (2.0 * math.log(theta))
    return int(math.floor(lo)), int(math.ceil(hi))


def m_turns(alpha, beta, theta, window, head_dim, ramp="mrpro", k=K):
    """A band placed by its turn window, filled with a named ramp."""
    lo, hi = band_from_turns(alpha, beta, theta, window, head_dim, k=k)
    lo, hi = max(0, lo), min(k - 1, hi)
    n = hi - lo
    if n < 1:
        raise ValueError(f"empty turn window [{alpha},{beta}] on this config")
    q = np.arange(1, n + 1, dtype=np.float64)
    if ramp == "mrpro":
        eps = q / q.sum()
    elif ramp == "beta1":
        w = q * (n + 1 - q)
        eps = w / w.sum()
    elif ramp == "linear":
        eps = np.full(n, 1.0 / n)
    else:
        raise KeyError(f"unknown ramp {ramp!r}")
    m = np.zeros(k)
    m[lo + 1:hi + 1] = np.cumsum(eps)
    m[hi + 1:] = 1.0
    return m


def m_split(s, first="mrpro", second="yarn"):
    src = {"mrpro": m_mrpro(17), "yarn": m_yarn(k=K)}
    idx = np.arange(K)
    return np.where(idx < s, src[first], src[second])



# ---------------------------------------------------------------------------
# companding and leak constructions, verbatim from
# experiments/curvature_20260910/tables.py (the anchor check above proves the
# copies agree; `selftest.t_copies_agree` extends that proof to these).
# ---------------------------------------------------------------------------
def uniform_phi(k=K):
    return (np.arange(k, dtype=np.float64) + 0.5) / k


def evq_phi(tau, k=K):
    u = uniform_phi(k)
    if abs(tau) < 1e-8:
        return u
    return 1.0 - np.arcsinh((1.0 - u) * math.sinh(tau)) / tau


def m_evq_deployed(tau, k=K):
    return evq_phi(tau, k)


def m_evq_shift(tau, cfg=QWEN25_3B, k=K):
    return (evq_phi(tau, k) - uniform_phi(k)) * math.log(cfg["theta"]) / LN_S


def m_power_shift(p, k=K):
    if p <= 0:
        return np.zeros(k)
    a = np.power(uniform_phi(k), 1.0 / (p + 1.0))
    return (a - a[0]) / (a[-1] - a[0])


def m_order(perm, n=17, low=23, k=K, base="mrpro"):
    """MrRoPE's increment MULTISET in a permuted ORDER -- the ordering experiment.

    WHY THIS IS NOT ANOTHER CURVE.  Every arm the project has swept changes the
    multiset of increments: it moves mass between slots.  This changes only the
    ORDER in which the same mass is spent, holding the band, the endpoints, the
    total span AND the multiset fixed.  So a difference between two of these arms
    cannot be attributed to how much compression there is or where the band sits;
    it is attributable to sequence alone.  That is a causal question rather than a
    search, and it is the one the Pro analysis calls the highest-ROI experiment
    for "why is MrRoPE strong" -- because MrRoPE IS the progressive member.

    THE FOUR OUTCOMES ARE ALL INFORMATIVE, which is why it is worth running:
      progressive wins              -> ordering is the causal variable
      progressive wins only on exact match, margins flat
                                    -> the gain is threshold amplification across
                                       an argmax, not a capability improvement
      some random order ties or wins -> arithmetic progression is a LUCKY POINT,
                                       not a law, and the whole "MrRoPE is special"
                                       reading collapses to the multiset
      a different order wins        -> a concrete table to build, and the first
                                       non-archival winner this project would have

    LEGALITY IS AUTOMATIC.  eps > 0 for every entry, so m = cumsum(eps) is
    monotone by construction, and the frequencies stay strictly ordered because
    the native log-gap (ln theta / K = 0.216 nats here) dwarfs the largest
    possible compression step.  No permutation can produce a broken table, so the
    arms need no individual vetting -- unlike every other family in this file.

    `perm` = identity is MrRoPE to FLOATING POINT (2.8e-17, three of 64 entries
    differ by 1 ULP) and NOT bitwise -- the arithmetic path differs, exactly as it
    does for `m_incr_beta(0)`. The caller asserts the float bound, not equality;
    claiming bitwise here would be the L9 error this file has already made twice.

    ONE PROPERTY WORTH KNOWING BEFORE READING THE RESULTS.  Holding the multiset
    fixed does NOT hold sum(m) fixed: the progressive order spends its large
    increments last and gives sum(m) = 29.33, while the reversed order spends them
    first and gives 34.67.  So the six arms span a real range in "how much
    compression is applied on average", and if the scores track sum(m) rather than
    the order, then sum(m) is the operative variable and ordering is not.  That
    check is built into the design and is reported by the reader.
    """
    if sorted(int(x) for x in perm) != list(range(n)):
        raise ValueError(f"not a permutation of range({n}): {perm}")
    if base == "mrpro":
        eps = 2.0 * np.arange(1, n + 1, dtype=np.float64) / (n * (n + 1))
    elif base == "bm":
        qq = np.arange(1, n + 1, dtype=np.float64)
        w = qq * (n + 1 - qq)
        eps = w / w.sum()
    else:
        raise ValueError(f"unknown base {base!r}")
    eps = eps[np.asarray(perm, dtype=int)]
    m = np.zeros(k, dtype=np.float64)
    m[low + 1: low + n + 1] = np.cumsum(eps)
    m[low + n + 1:] = 1.0
    return m


def order_perms(n=17, seed=20260910):
    """The six orderings, with the random ones' seed DECLARED, not drawn.

    A random permutation drawn at run time makes the arm set irreproducible, and
    an irreproducible arm set cannot be pre-registered.  The seed is fixed here so
    that "the four random permutations" names four specific tables.
    """
    rng = np.random.default_rng(int(seed))
    perms = {"progressive": np.arange(n), "reversed": np.arange(n)[::-1]}
    for i in range(4):
        perms[f"rand{i + 1}"] = rng.permutation(n)
    return perms


def m_C42(lo=14, n=18, k=K):
    """The Pro analysis's C42 control: move the increment CENTROID, keep everything else.

    S and the increment centroid are the same coordinate.  Writing eps_k = m_k -
    m_{k-1} and using m_0 = 0, m_63 = 1,

        S = sum_j m_j = sum_k (64-k) eps_k = 64 - mu_eps,   mu_eps = sum_k k eps_k,

    so sum(m) IS (64 minus) the centroid of the added log-frequency spacing.  The
    three OLMo winners sit at mu_eps = 22 while the deployed BM is at 23.5 and
    MrRoPE at 26.33 -- so "raise the budget" and "move the centroid earlier" are
    the same statement, and no existing arm separates them from the band or the
    platform, which move at the same time.

    C42 does.  It is the minimal weighted perturbation of the deployed BM that
    moves the LOCAL centroid from 9.5 to 8 (global 23.5 -> 22, S 40.5 -> 42):

        eps_k = p_k [ 1 - (10/119)(k - 19/2) ],   p_k = 6k(19-k)/(18*19*20)

    with the band [14,32] and BOTH plateaus untouched -- the same 15 m=0 slots and
    the same 32 m=1 slots as the deployed table.  So BM -> C42 isolates "centroid
    moved" with band, platform, endpoints and gain all held.
    """
    kk = np.arange(1, int(n) + 1, dtype=np.float64)
    p = 6.0 * kk * (int(n) + 1 - kk) / (n * (n + 1) * (n + 2))
    eps = p * (1.0 - (10.0 / 119.0) * (kk - 19.0 / 2.0))
    if eps.min() <= 0:
        raise ValueError("C42 increment went non-positive")
    m = np.zeros(k, dtype=np.float64)
    m[lo + 1: lo + int(n) + 1] = np.cumsum(eps)
    m[lo + int(n) + 1:] = 1.0
    return m


def m_C42V24(lo=14, n=18, k=K):
    """C42 with the increment VARIANCE also matched to a1_b64 (24 instead of 15.6).

    Same centroid, same band, same plateaus, same S -- only the second moment of
    the increment distribution moves.  The pair (C42, C42-V24) therefore isolates
    "higher-order allocation at fixed centroid", which is the remaining freedom
    the campaign has not been able to test because every other family changes the
    centroid at the same time as everything else.

        eps_k = p_k [ 1 - (10/119)(k-19/2) + (35/1496)((k-19/2)^2 - 357/20) ]

    Still the analytic solution of the same weighted-minimal-perturbation problem,
    now with mass, first and second moments constrained.  All increments positive.
    """
    kk = np.arange(1, int(n) + 1, dtype=np.float64)
    p = 6.0 * kk * (int(n) + 1 - kk) / (n * (n + 1) * (n + 2))
    eps = p * (1.0 - (10.0 / 119.0) * (kk - 19.0 / 2.0)
               + (35.0 / 1496.0) * ((kk - 19.0 / 2.0) ** 2 - 357.0 / 20.0))
    if eps.min() <= 0:
        raise ValueError("C42-V24 increment went non-positive")
    m = np.zeros(k, dtype=np.float64)
    m[lo + 1: lo + int(n) + 1] = np.cumsum(eps)
    m[lo + int(n) + 1:] = 1.0
    return m


def m_taper(delta, hi=32, lo=14, k=K):
    """The plateau TAPERED linearly: m[j] = max(0, 1 - delta*(j-hi)) for j >= lo.

    THE PURE-BUDGET DIAL.  plan-design measured that this knob moves sum(m) by
    -16 units while leaving the m>=0.5 crossing, the untouched-slot count and
    the weighted budget Wc = sum_j omega_j m_j essentially fixed -- so it is the
    one construction that changes the budget WITHOUT changing anything else the
    surviving candidates measure.  That is what makes the 2x2 possible.

    Reading it as a table: instead of holding the slow slots at exactly m = 1
    (the deployed plateau), it lets them relax back toward native as j grows.
    delta = 0 IS the deployed plateau exactly.
    """
    if delta < 0:
        raise ValueError("delta < 0 would make the plateau rise")
    n = hi - lo
    q = np.arange(1, n + 1, dtype=np.float64)
    w = q * (n + 1 - q)                      # the deployed BM ramp, eps ~ k(n+1-k)
    eps = w / w.sum()
    j = np.arange(k, dtype=np.float64)
    m = np.zeros(k, dtype=np.float64)
    m[lo + 1: hi + 1] = np.cumsum(eps)       # ramp: m=0 at lo, m=1 at hi
    # the plateau, tapered from hi onward
    m[hi:] = np.maximum(0.0, 1.0 - delta * (j[hi:] - hi))
    return m


def m_step(hi, lo=14, k=K):
    """The EXTREME-concentration table: m = 0 up to `hi-1`, m = 1 from `hi`.

    WHY THIS FAMILY EXISTS.  Two stories fit the nine OLMo measurements and they
    make OPPOSITE predictions that no existing arm separates:

      S story      the score tracks the budget S = sum(m).  Then compress as much
                   as possible anywhere.
      concentration story   at matched S, the arms that HOLD MORE slots and cram
                   the compression into fewer of them win.  Measured: at S = 38.47
                   vs 38.50 with the SAME platform count, holding 19 slots beats
                   holding 15 by 12 points.

    Under the S story a step at hi=22 (S = 42, exactly the winners' budget,
    22 held, NO ramp at all) should score like the winners.  Under the
    concentration story it should score BETTER, because it is the most
    concentrated table with that budget -- and it is the flat-Fisher optimum,
    which the analytic derivation reached independently (the minimiser of
    eps^T K_flat eps subject to sum(eps) = 1 is the extreme step e_n).

    Walking `hi` walks BOTH axes in opposite directions and nothing else changes:

        hi = 22   S = 42   held 22   plateau 42   (the winners' budget)
        hi = 25   S = 39   held 25   plateau 39
        hi = 28   S = 36   held 28   plateau 36
        hi = 32   S = 32   held 32   plateau 32

    So the four points are a clean discriminator: monotone in S if the budget is
    the cause, monotone the OTHER way if holding is.

    LEGALITY is automatic -- one increment of size 1 cannot disorder the
    frequencies, since it lowers nu by a factor 4 at one slot and the native
    log-gap is far larger than the compression it has to absorb.
    """
    if not 0 < hi < k:
        raise ValueError("hi out of range")
    m = np.zeros(k, dtype=np.float64)
    m[hi:] = 1.0
    return m


def m_mixC(C, n=17, low=23, k=K):
    """The exact response to a uniform-plus-end-spike forcing: eps_k ∝ k(C-k).

    WHY THIS FAMILY AND NOT THE b-FAMILY.  The two deployed tables pin the
    forcing at its two ends: MrRoPE is the Dirichlet Green response to a pure
    spike at the last increment, BM to a constant.  A forcing that is any
    MIXTURE of those two, g = c*1 + d*e_n, therefore has the closed-form response
    eps_k ∝ k(C-k) with C = n+1 + 2d/(c(n+1)), and it interpolates the two
    deployed tables EXACTLY:

        C = n+1 (= 18)   eps ∝ k(n+1-k)   = BM
        C -> infinity    eps ∝ k          = MrRoPE

    The `k(n+1-k)^b` family that is currently being swept shares both endpoints
    but takes a DIFFERENT path between them.  Measured: the best b-arm for an
    interior C is off by up to 0.014 in m, against a max per-slot separation of
    0.27 between the endpoints.  So the b sweep is a bounded approximation
    (<= ~5% of the axis), not a wrong experiment -- but if an interior optimum
    turns up, THIS is the family to refine it in, because it is the one the
    measured endpoints actually imply.

    C must exceed n or the last increment goes negative.
    """
    if C <= n:
        raise ValueError(f"C = {C} does not exceed n = {n}; the tail would "
                         "run backwards")
    kk = np.arange(1, int(n) + 1, dtype=np.float64)
    w = kk * (float(C) - kk)
    eps = w / w.sum()
    m = np.zeros(k, dtype=np.float64)
    m[low + 1: low + n + 1] = np.cumsum(eps)
    m[low + n + 1:] = 1.0
    return m


def m_leak(a, base="bm", n=17, low=23, k=K):
    """beta_b1 with a share of the compression moved into the held plateau.

    Written in increments so the budget is exactly conserved: `low` slots below
    the band each get `a`, and the band's own increments are scaled by
    (1 - a*low).  Total is 1 for every a, so the only difference from beta_b1 is
    the SUPPORT of the compression.  a = 0 is beta_b1 exactly.
    """
    if not 0.0 <= a * low < 1.0:
        raise ValueError(f"leak {a} x {low} slots leaves no budget for the band")
    if base == "bm":
        m0 = np.asarray(m_incr_beta(1.0, n=n, low=low, k=k), dtype=np.float64)
    elif base == "mrpro":
        m0 = np.asarray(m_mrpro(n=n, low=low, k=k), dtype=np.float64)
    else:
        raise ValueError(f"unknown base {base!r}")
    if a == 0.0:
        # EXACT identity, not a diff(cumsum(...)) round trip.  `np.diff` of a
        # cumulative sum is not bit-identical to the increments that produced it,
        # and this arm exists ONLY to be a construction check -- a near-copy that
        # differs in the last bits would make the check vacuous while looking
        # like it passed.  Returning the base makes it exact by construction.
        return m0.copy()
    if m0[low] != 0.0:
        raise ValueError(f"base {base!r} does not start the band at m = 0")
    eps = np.diff(m0[low: low + n + 1])
    m = np.zeros(k, dtype=np.float64)
    m[1: low + 1] = a * np.arange(1, low + 1, dtype=np.float64)
    m[low + 1: low + n + 1] = a * low + np.cumsum(eps) * (1.0 - a * low)
    m[low + n + 1:] = 1.0
    return m


A_MR = 2.0 / 306.0
R_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
B_VALUES = (0.0, 0.25, 0.5, 1.0, 2.0)


def arms():
    """The declared bank.  Predictions live in the local `bank.py`; this list and
    that one must agree arm-for-arm, and `selftest.py` is what checks it."""
    out = [
        ("native", np.zeros(K)),
        # THE TRUE ARCHITECTURE NATIVE, gain 1.0.  Every other arm in this bank
        # is installed at YaRN's gain 1.138629436111989, because the gain is a
        # design FACE (INTEGRATION R3) and holding it fixed is what isolates the
        # ALLOCATION.  But that means the arm called `native` is "native
        # frequencies at YaRN's gain", not the checkpoint's own configuration --
        # and reading it as the fidelity floor is wrong by the whole gain effect.
        # Measured against the archive: at 32K the archived Native (gain 1.0) is
        # 2.03850 while native-frequencies-at-YaRN-gain is 2.08866, so the gain
        # accounts for ~0.050 of MrPro's 0.0505 in-window cost and the frequency
        # table for ~0.0007.  This arm measures that difference directly instead
        # of inheriting it from a differently-configured archive row.
        ("native_gain1", np.zeros(K)),
        ("yarn_lin", m_yarn(k=K)),
        ("mrpro_n17", m_mrpro(17)),
        ("mrpro_n16", m_mrpro(16)),
    ]
    for b in B_VALUES:
        out.append((f"beta_b{b:g}".replace(".", "p"), m_incr_beta(b)))
    for r in R_VALUES:
        out.append((f"incr_r{r:g}".replace(".", "p"), m_incr_power(r)))
    a_mr = A_MR
    for a in (0.001, 0.03, 0.10):
        out.append((f"front_a{a:g}".replace(".", "p"), m_incr_split(a, 1.0)))
    for r in (0.0, 2.0, 4.0):
        out.append((f"back_r{r:g}".replace(".", "p"), m_incr_split(a_mr, r)))
    for (a, r) in ((0.03, 2.0), (0.001, 2.0)):
        out.append((f"both_a{a:g}_r{r:g}".replace(".", "p"), m_incr_split(a, r)))
    for s in (24, 28, 32, 36, 40):
        out.append((f"splitA_s{s}", m_split(s, "mrpro", "yarn")))
        out.append((f"splitB_s{s}", m_split(s, "yarn", "mrpro")))
    out.extend(companding_arms())
    out.extend(leak_arms())
    out.extend(mix_arms())
    out.extend(order_arms())
    out.extend(b_extra_arms())
    return out


# ---------------------------------------------------------------------------
# THE TWO GROUPS ROUND 1 DID NOT RUN, and they are the ones the stated core
# question needs.  Round 1's 29 arms all sat inside the three-band family: every
# one of them held slots 0..23 at m = 0 and every one of them reached m = 1 by
# slot 40.  So the screen measured the SHAPE of the transition and nothing about
# its SUPPORT -- while the project's own method (EVQ) is a global companding that
# necessarily moves the held slots, and the method it is trying to explain (YaRN,
# MrRoPE, BM) all refuse to.  "Is the held plateau justified?" was never asked.
# ---------------------------------------------------------------------------
def companding_arms():
    """EVQ and the closed-form companding family, at the family's own settings."""
    # t = 1.414 is the value the project's earlier phases converged on
    # (EVQ tau* ~ 1.14-1.414 across settings), so it is the arm that represents
    # the method as actually deployed rather than a sweep point.
    out = [("evq_deploy_t1", m_evq_deployed(1.0)),
           ("evq_deploy_t2", m_evq_deployed(2.0)),
           ("evq_deploy_t4", m_evq_deployed(4.0)),
           ("evq_deploy_t1p414", m_evq_deployed(1.414)),
           ("evq_shift_t1", m_evq_shift(1.0)),
           ("evq_shift_t2", m_evq_shift(2.0)),
           ("power_p2", m_power_shift(2)),
           ("power_p4", m_power_shift(4))]
    return [(n, np.asarray(m, dtype=np.float64)) for n, m in out]


def order_arms(base="mrpro"):
    """The six orderings. `ord_progressive` must tie mrpro_n17 bitwise."""
    return [(f"ord_{name}", m_order(perm, base=base))
            for name, perm in order_perms().items()]


def b_extra_arms():
    """The b ramp continued past 2, plus (a,b) cells that BREAK a real confound.

    ON OLMo THE b SWEEP IS MONOTONE THROUGH AND PAST BM: 7.09 / 14.47 / 23.20 /
    41.67(b=1) / 50.01(b=2) per cent, rising as the increment peak moves forward
    (peak at k = (n+1)/(b+1): 18, 14.4, 12, 9, 6).  Two things move together along
    that ramp and the family cannot separate them:

        b        peak position      sum(m)
        0        18 (back)          37.67
        0.5      12                 39.21
        1         9 (middle)        40.50
        2         6                 42.38

    Both are monotone, so "front-loading helps" and "more total compression
    helps" fit the same five points.  The (a, b) cells do not: raising `a` moves
    the peak forward at a DIFFERENT rate in sum(m) than raising `b` does, so a
    pair that matches one axis and differs on the other separates them.  This is
    the family the source module has always had (`shape_a` has been in
    `tables.m_incr_beta` since it was written); it was simply never swept.
    """
    out = [(f"beta_b{b:g}".replace(".", "p"), m_incr_beta(b))
           for b in (3.0, 4.0, 6.0, 8.0)]
    for (a, b) in ((2.0, 1.0), (2.0, 2.0), (4.0, 1.0), (0.5, 4.0)):
        kk = np.arange(1, 18, dtype=np.float64)
        w = np.power(kk, a) * np.power(18 - kk, b)
        eps = w / w.sum()
        m = np.zeros(K)
        m[24:41] = np.cumsum(eps)
        m[41:] = 1.0
        out.append((f"ab_a{a:g}_b{b:g}".replace(".", "p"), m))
    return out


def mix_arms():
    """The mixture family: the interpolation the two deployed endpoints imply.

    C = n+1 is beta_b1 (BM) exactly, so `mixC_18` is another free construction
    check -- if it does not tie beta_b1 then this family is built wrong.  The
    b-arms currently being swept take a different path between the same two
    endpoints; these take the one derived from the forcing.
    """
    return [(f"mixC_{C}", m_mixC(C)) for C in (18, 20, 26, 40, 80)]


def leak_arms(base="bm"):
    """The held-plateau probe: same table, compression leaked below the band.

    `m_leak(a)` moves a share of the compression into the held slots while
    holding sum(eps) at exactly 1, so the ONLY difference from `beta_b1` is the
    support.  This is the controlled version of the question the companding arms
    ask crudely: at what leak does the in-window cost actually appear?
    """
    # a = 0.0 IS IN THE LIST ON PURPOSE.  It is beta_b1 exactly, so it is a free
    # construction check: if it does not tie beta_b1 then the leak family is
    # built wrong and the other three arms are uninterpretable.  The first
    # version of this list started at 0.002 and the docstring claimed an a = 0
    # arm existed -- a check that is described but not run is worse than no
    # check, because it reads like one.
    return [(f"leak_a{a:g}".replace(".", "p"), m_leak(a, base=base))
            for a in (0.0, 0.002, 0.005, 0.01, 0.02)]


def anchor_check():
    """Both ends of the beta family, checked before the GPU is touched.

    b = 1 must reproduce the DEPLOYED table's increments EXACTLY.  The archived
    counter's increments are eps_k = 6k(18-k)/(17*18*19) = k(18-k)/969, a rational
    number, so the comparison is against that formula and not against the
    decimals someone typed off a printout -- an earlier version compared against
    five-decimal literals and reported a 2.2e-4 "error" that was the literals.

    b = 0 equals MrRoPE to floating-point precision rather than bit-for-bit:
    MrRoPE's own code evaluates q(q+1)/(n(n+1)) directly while the family takes
    a normalised cumsum, and the two orders differ in the last bits.  The
    BIT-EXACT anchor is `incr_r1`, which is `m_incr_power(1.0)` and does agree
    with `m_mrpro(17)` exactly -- that pair is in the bank as two arms, and the
    driver refuses to run if they ever score differently.
    """
    a, b0 = m_incr_power(1.0), m_mrpro(17)
    beta0, beta1 = m_incr_beta(0.0), m_incr_beta(1.0)
    # the deployed Qwen BM increments are eps_k = 6k(18-k)/(17*18*19) = k(18-k)/969
    # for k = 1..17 -- seventeen of them, because the band is seventeen gaps wide
    # even though its endpoints span eighteen slots.  (OLMo's is n = 18 and the
    # same formula with 19 in place of 18; the general form is
    # 6k(n+1-k) / (n(n+1)(n+2)).)
    k = np.arange(1, 18, dtype=np.float64)
    exact_qwen_bm = k * (18 - k) / 969.0
    d = np.diff(beta1[23:41])
    # THE THREE LEGACY KEYS ARE NOT DECORATION.  `bit_exact`, `max_abs_diff` and
    # `sum_m` are what the callers in kkt_residual.py and phase1_ruler.py index,
    # and `bit_exact` is what this file's own main() indexes.  The detailed keys
    # below were added without updating those callers, so a chain that had been
    # running fine for thirty minutes died in one second at the next stage with
    # `KeyError: 'bit_exact'` -- the GPU then sat idle until someone read the log.
    # Callers and callee must agree on the contract; both are asserted in
    # selftest.py now, which is the only thing that would have caught it.
    bit_exact = bool(np.array_equal(a, b0))
    max_abs_diff = float(np.abs(a - b0).max())
    return dict(
        # the contract the callers use
        bit_exact=bit_exact,
        max_abs_diff=max_abs_diff,
        sum_m=float(a.sum()),
        # the detail, for the receipt
        incr_r1_bit_exact_vs_mrpro=bit_exact,
        beta_b0_max_abs_diff_vs_mrpro=float(np.abs(beta0 - b0).max()),
        beta_b0_equals_mrpro_to_float32=bool(
            np.array_equal(beta0.astype(np.float32), b0.astype(np.float32))),
        beta_b1_max_abs_err_vs_deployed=float(np.abs(d - exact_qwen_bm).max()),
        sum_m_b0=float(b0.sum()), sum_m_b1=float(beta1.sum()),
    )


def anchor_contract_keys():
    """The keys every caller may index, so a test can pin them.

    A dict return is an interface; nothing else in this repository pinned this
    one, which is how three files came to disagree about it.
    """
    return ("bit_exact", "max_abs_diff", "sum_m")


GAIN1_ARM = "native_gain1"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--history", default="/root/autodl-tmp/bm_transfer_20260908")
    ap.add_argument("--batch", type=int, default=0,
                    help="documents per forward at the LONGEST length; shorter "
                         "lengths scale up to the same token budget (0 = auto)")
    ap.add_argument("--batch-tokens", type=int, default=262144,
                    help="token budget per forward, from the measured 32K probe "
                         "(batch 8 x 32768 = 262144 at 27.8 GB peak)")
    ap.add_argument("--lengths", default="8192,16384,32768")
    ap.add_argument("--only", default=None, help="comma-separated arm names")
    ap.add_argument("--probe", action="store_true",
                    help="run one length at one batch and exit, for tuning")
    args = ap.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    lengths = [int(x) for x in args.lengths.split(",")]

    ac = anchor_check()
    print(json.dumps({"anchor": ac}), flush=True)
    if not ac["bit_exact"]:
        print("REFUSING: the bank's r=1 anchor is not bit-exact against MrRoPE",
              file=sys.stderr)
        return 2

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiments.nongeometric_screen.worker import Worker

    import torch
    w = Worker(root, args.history)
    theta = QWEN25_3B["theta"]

    def tbl(name, m):
        # the one arm that is not at YaRN's gain; see the comment in arms()
        gain = 1.0 if name == GAIN1_ARM else GAIN_YARN
        return dict(values_float32=m_to_inv_freq(m, theta).astype(np.float32),
                    gain=gain)

    # ---- the corpus and the archived MrPro cells -------------------------
    docs = w.nll_manifest["docs"]
    prepared = w.nll_inputs
    by_len = {L: [] for L in lengths}
    for L in lengths:
        for d in docs:
            by_len[L].append((d["file"], d["sha256"]))
    base = {(r["doc"], r["length"]): r for r in
            [json.loads(x) for x in open(Path(args.history) / "run_nll_01" / "MrPro.jsonl")]}
    missing = [(f, L) for L in lengths for f, _ in by_len[L] if (f, L) not in base]
    if missing:
        print(f"REFUSING: {len(missing)} archived MrRoPE NLL cells are missing "
              f"(e.g. {missing[:3]}); without them there is nothing to compare to",
              file=sys.stderr)
        return 2
    print(json.dumps({"corpus": dict(n_docs=len(docs), lengths=lengths,
                                     tail=w.nll_manifest["tail_tokens"],
                                     archived_mrpro_cells=len(base))}), flush=True)

    order = arms()
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        order = [(n, m) for n, m in order if n in want]
    tail = int(w.nll_manifest["tail_tokens"])

    def run_length(L, batch, tag):
        """One forward per batch of documents; returns per-doc tail NLL."""
        files = by_len[L]
        data = {f: np.load(prepared / f) for f, _ in files}
        tgt = {f: torch.tensor(data[f][L - tail + 1:L + 1].astype(np.int64),
                               device="cuda") for f, _ in files}
        res = dict()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        for i in range(0, len(files), batch):
            grp = files[i:i + batch]
            ids = torch.stack([torch.tensor(data[f][:L].astype(np.int64))
                               for f, _ in grp]).cuda()
            with torch.inference_mode():
                lg = w.model(ids, use_cache=False, logits_to_keep=tail).logits
                for j, (f, _) in enumerate(grp):
                    nll = torch.nn.functional.cross_entropy(
                        lg[j].float(), tgt[f], reduction="none")
                    res[f] = dict(nll=float(nll.mean()), token_nll=nll.tolist())
            del ids, lg
        secs = time.monotonic() - t0
        peak = torch.cuda.max_memory_allocated()
        print(json.dumps({"phase": tag, "length": L, "batch": batch,
                          "docs": len(files), "seconds": secs,
                          "peak_GB": peak / 1e9}), flush=True)
        return res, secs, peak

    def batch_for(L):
        """Fill the card at EVERY length, not just the longest.

        The probe at 32768 with batch 8 peaked at 27.8 GB of a 32 GB card, so the
        activations scale with batch x length and a fixed batch leaves the short
        lengths idle.  The budget below is that measured point restated as
        tokens; a length that does not fit falls back by halving rather than
        crashing the run, because a half-full card still produces the number.
        """
        if args.batch:
            return int(args.batch)
        return max(1, int(args.batch_tokens) // int(L))

    # ---- probe: measure, then set the batch to fill the card -------------
    if args.probe:
        run_length(lengths[-1], args.batch or 8, "PROBE")
        return 0

    # ---- RESUME.  This runner was the only one of the three that could not
    # resume: `phase1_ruler.py` and `olmo_beta.py` both rebuild their done-set
    # from the jsonl they append to, and this one re-ran every arm from scratch
    # and appended a SECOND copy of each.  That is not a cosmetic difference --
    # it made the in-window screen the one instrument that could not be run in
    # parallel with the card's other work, and it silently doubles a receipt if
    # anyone restarts it.  The key is the arm name, which is unique per table.
    rows = []
    done = set()
    raw = root / "rows.jsonl"
    if raw.exists():
        for line in raw.open():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["name"])
            except (KeyError, ValueError):
                continue
        print(json.dumps({"resume": True, "already_done": sorted(done)}), flush=True)

    manifest = dict(started=time.strftime("%Y-%m-%dT%H:%M:%S"), anchor=ac,
                    batch=args.batch, lengths=lengths, tail=tail,
                    n_arms=len(order) - len(done), arms_requested=len(order), mrpro_rerun=False,
                    scope="Tail 512 next-token NLL, prefix lengths inside the "
                          "32768 native window; in-window instrument, not "
                          "long-context capability",
                    weights_updated=False)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))

    for name, m in order:
        if name in done:
            print(json.dumps({"skip": name, "reason": "already in rows.jsonl"}),
                  flush=True)
            continue
        w.apply({"table": tbl(name, m)})
        rec = dict(name=name, sum_m=float(np.asarray(m).sum()), per_length={})
        for L in lengths:
            b = batch_for(L)
            while True:
                try:
                    res, secs, peak = run_length(L, b, f"NLL_{name}")
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if b == 1:
                        raise
                    b = max(1, b // 2)
                    print(json.dumps({"oom": True, "length": L,
                                      "retry_batch": b}), flush=True)
            per = {}
            for f, v in res.items():
                b = base[(f, L)]
                per[f] = dict(nll=v["nll"], mrpro=b["nll"],
                              delta=v["nll"] - b["nll"])
            rec["per_length"][str(L)] = dict(
                mean_nll=float(np.mean([v["nll"] for v in res.values()])),
                mrpro_mean=float(np.mean([base[(f, L)]["nll"] for f, _ in by_len[L]])),
                mean_delta=float(np.mean([v["nll"] - base[(f, L)]["nll"]
                                          for f, v in res.items()])),
                per_doc=per, seconds=secs, peak_bytes=peak)
        rec["mean_delta_all"] = float(np.mean(
            [rec["per_length"][str(L)]["mean_delta"] for L in lengths]))
        rows.append(rec)
        with (root / "rows.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({"arm": name, "sum_m": rec["sum_m"],
                          "mean_delta_all": rec["mean_delta_all"],
                          "by_len": {L: rec["per_length"][str(L)]["mean_delta"]
                                     for L in lengths}}), flush=True)

    manifest["status"] = "COMPLETE"
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["leaderboard"] = sorted(
        [dict(name=r["name"], sum_m=r["sum_m"], mean_delta_all=r["mean_delta_all"],
              **{f"d{L}": r["per_length"][str(L)]["mean_delta"] for L in lengths})
         for r in rows], key=lambda r: r["mean_delta_all"])
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"status": "COMPLETE", "n_arms": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
