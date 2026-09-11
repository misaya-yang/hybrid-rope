"""Table algebra for the frequency-allocation problem.  Pure numpy, no torch.

Every RoPE table in this programme is a point in one coordinate system.  With
omega_j = theta^{-j/K} the trained (native) frequency of slot j, define

    m_j := ln(omega_j / nu_j) / ln S          (so nu_j = omega_j * S^{-m_j})

m_j = 0 means "this slot still runs at its trained frequency"; m_j = 1 means
"this slot runs S times slower", i.e. exactly one full compression.  The
native grid is m = 0; pure interpolation is m = 1; YaRN and MrRoPE live in
between with an active set at both ends.  Expressing everything in m is what
makes YaRN, MrRoPE, EVQ and a solved table comparable at all -- they were
specified in three different spaces (dim index, radix exponent, quantile).

The construction here is the reference for the rest of the package and is
deliberately independent of the training code in scripts/lib/rope.
"""
from __future__ import annotations

import math

import numpy as np

K = 64
LN_S = math.log(4.0)

# Qwen2.5-3B-Instruct as pinned at revision aa8e7253...
QWEN25_3B = dict(theta=1_000_000.0, head_dim=128, window=32768, scale=4)
# Llama-3-8B, kept for the cross-model transfer arms.
LLAMA3_8B = dict(theta=500_000.0, head_dim=128, window=8192, scale=4)
# OLMo-2-1B, the from-scratch / fine-tune arm.
OLMO2_1B = dict(theta=500_000.0, head_dim=128, window=4096, scale=4)

# YaRN's mscale with mscale=1: 0.1*ln(S)+1.  The historical frozen-weight
# experiments used this value as `official_gain`; it is YaRN's own factor, not
# an independent choice.
GAIN_YARN = 1.0 + 0.1 * LN_S           # 1.138629436111989
GAIN_1 = 1.0
GAIN745 = 1.0 + 0.074 * LN_S           # 1.102585782722872


def native_inv_freq(theta, k=K):
    return theta ** (-np.arange(k, dtype=np.float64) / k)


def m_to_inv_freq(m, theta, k=K):
    return native_inv_freq(theta, k) * np.power(4.0, -np.asarray(m, dtype=np.float64))


def inv_freq_to_m(nu, theta, k=K):
    """Inverse of m_to_inv_freq.  Negative m means the slot was sped up."""
    return -np.log(np.asarray(nu, dtype=np.float64) / native_inv_freq(theta, k)) / LN_S


def uniform_phi(k=K):
    """The native grid's own quantile coordinate, u_j = (j+1/2)/K."""
    return (np.arange(k, dtype=np.float64) + 0.5) / k


# ---------------------------------------------------------------------------
# Named constructions
# ---------------------------------------------------------------------------
def m_native(k=K):
    return np.zeros(k)


def m_interp(k=K):
    return np.ones(k)


def m_mrpro(n=17, low=23, k=K):
    """MrRoPE's radial family: m_j = q(q+1)/(n(n+1)), q = clip(j-low, 0, n).

    Increments arithmetic in q and the endpoint m(n) = 1 together force the
    quadratic -- there is no free parameter left once n and the support are
    fixed.  n = 17 is the member whose transition width equals YaRN's
    dim*ln(beta/alpha)/(2 ln theta); n = 16 and n = 15 are the two neighbouring
    members of the same family, constructed but never evaluated.
    """
    q = np.clip(np.arange(k) - low, 0, n).astype(np.float64)
    m = q * (q + 1.0) / (n * (n + 1.0))
    m[np.arange(k) >= low + n + 1] = 1.0
    return m


def yarn_bands(alpha=1.0, beta=32.0, cfg=QWEN25_3B):
    """YaRN's raw (float) band edges from find_correction_dim, and the integer
    band the reference implementation actually ramps over.

    `find_correction_dim` returns 23.60 and 39.66 here, but every released
    implementation then takes floor(low) and ceil(high) before building the
    ramp, so the deployed transition is slot-space linear on [23, 40].  Both are
    returned because the difference is not cosmetic: ramping over the raw floats
    moves slot 39 by 0.034 in m, which is a third of a native log-gap.
    """
    dim, theta, W = cfg["head_dim"], cfg["theta"], cfg["window"]
    lo = dim * math.log(W / (beta * 2.0 * math.pi)) / (2.0 * math.log(theta))
    hi = dim * math.log(W / (alpha * 2.0 * math.pi)) / (2.0 * math.log(theta))
    return lo, hi, int(math.floor(lo)), int(math.ceil(hi))


def m_yarn(scale=4.0, alpha=1.0, beta=32.0, cfg=QWEN25_3B, k=K):
    """YaRN's NTK-by-parts ramp, converted to the m-coordinate.

    YaRN ramps in dim index: the extrapolation factor goes 1 at high frequency
    down to 0 at low frequency, and nu_j/omega_j = r + (1-r)/scale.  Bands are
    floor/ceil of find_correction_dim, matching the deployed table.

    Checked against analysis/unify_20260910/tables/ground_truth_tables.json:
    against the DEPLOYED YaRN_linear_official nu_j this now agrees to 8.4e-8
    relative (float32 rounding); ramping over the raw float edges instead
    disagrees by 0.0344 in m at slot 39.
    """
    _, _, lo, hi = yarn_bands(alpha, beta, cfg)
    ramp = np.clip((np.arange(k) - lo) / (hi - lo), 0.0, 1.0)  # 0 fast .. 1 slow
    ratio = 1.0 - (1.0 - 1.0 / scale) * ramp                   # 1 fast .. 1/S slow
    return -np.log(ratio) / math.log(scale)


def evq_phi(tau, k=K):
    """Canonical EVQ companding curve: scripts/lib/rope/schedules.py:162.

    phi = 1 - asinh((1-u) sinh tau) / tau,  u = (k+0.5)/K.  phi = 0 at the fast
    end, phi = 1 at the slow end; the induced density on phi is
    tau cosh[tau(1-phi)] / sinh tau, i.e. the slow clocks are spread and the
    fast ones are packed.
    """
    u = uniform_phi(k)
    if abs(tau) < 1e-8:
        return u
    return 1.0 - np.arcsinh((1.0 - u) * math.sinh(tau)) / tau


def m_evq_shift(tau, cfg=QWEN25_3B, k=K):
    """EVQ read as a shift applied on top of the native grid.

    nu_j = theta^{-phi_j}, so relative to omega_j = theta^{-u_j} the shift is
    m_j = (phi_j - u_j) ln(theta) / ln S.  Negative where EVQ speeds a slot up,
    positive where it slows one down.  This is the ONLY reading of EVQ that is
    commensurate with YaRN/MrRoPE, and it makes the difference visible: EVQ
    redistributes inside the native span (sum m is not the budget), whereas
    YaRN/MrRoPE extend the span by exactly ln S.
    """
    return (evq_phi(tau, k) - uniform_phi(k)) * math.log(cfg["theta"]) / LN_S


def m_evq_deployed(tau, cfg=QWEN25_3B, k=K):
    """EVQ deployed the way YaRN/MrRoPE are: companding composed with S-scaling.

    nu_j = omega_j * S^{-phi_j}.  Endpoints are then m_0 = 0 and m_63 = 1, exactly
    like the geometric families, so any performance difference is attributable
    to the SHAPE of the interior profile and not to the budget or the endpoints.
    This is the arm that makes the EVQ-vs-YaRN comparison a controlled one.
    """
    return evq_phi(tau, k)


def m_power_shift(p, k=K):
    """The rho_p family from the closed-form companding analysis: rho ~ a^{1/(p+1)}.

    p = 2 reproduces the cube root (the scratch-training optimum); p -> inf
    gives sqrt(a) and the arithmetic grid.  Endpoints are anchored on [0, 1] so
    the family shares MrRoPE's budget and endpoints and differs only in the
    SHAPE of the interior profile.  Included so the companding family can be
    compared against the geometric ones inside the same frozen-weight harness.
    """
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
    """The three-band table with compression LEAKED into the held plateau.

    THE ONE MEASUREMENT THE EVQ QUESTION REDUCES TO.  Every table in the
    three-band family holds slots 0..low at exactly m = 0 -- YaRN, MrRoPE and the
    deployed BM all do -- while EVQ is a GLOBAL companding that spreads the
    compression over the whole spectrum, so it necessarily moves those slots.
    The project's stated reason for the plateau is that moving them is expensive
    in-window; the reason has never been measured, because no arm in any bank has
    ever put m > 0 there.

    This construction does exactly that and nothing else.  Written in the
    increment coordinate, where the budget is exactly "the increments sum to 1":

        eps_j = a                  for the `low` slots below the band
        eps_j = (1 - a*low) * eps^base_j    on the band, rescaled

    so the total is a*low + (1 - a*low) = 1 for every a, and the table still runs
    from m = 0 at slot 0 to m = 1 at slot low+n.  Only the SUPPORT of the
    compression moved.

      a = 0     reproduces `base` exactly (asserted, not assumed)
      a small   a plateau of a few percent: does in-window NLL even notice?
      a large   approaching the global spread the companding families use

    WHY THE BUDGET IS HELD FIXED.  If the leaked table also got more total
    compression it would win or lose for two reasons at once, and the project has
    already paid for that mistake often enough to have a rule about it
    (ds_workspace/LESSONS L4).  Holding sum(eps) at 1 makes the ONLY difference
    the support.

    AN EARLIER VERSION OF THIS FUNCTION DID NOT HOLD THE BUDGET.  It added `a` to
    every held slot and rescaled the band by (1 - a*low), which raises the whole
    band by a*low as well and grows sum(m) by (n+1)*a*low -- and it failed its own
    a = 0 identity check.  The caller asserts the identity, which is why the bug
    was caught on the first run rather than after the GPU had scored 4 arms.
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
    if not (m0[low] == 0.0):
        raise ValueError(f"base {base!r} does not start the band at m = 0")
    eps = np.diff(m0[low: low + n + 1])                  # n increments
    m = np.zeros(k, dtype=np.float64)
    m[1: low + 1] = a * np.arange(1, low + 1, dtype=np.float64)
    m[low + 1: low + n + 1] = a * low + np.cumsum(eps) * (1.0 - a * low)
    m[low + n + 1:] = 1.0
    return m



def m_global_compand(tau, k=K):
    """`m_evq_deployed` under a name that says what it structurally is.

    Kept as an alias so a reader comparing the three-band arms against the
    companding ones does not have to remember that `evq_deployed` means "a
    companding applied to the whole spectrum with endpoints 0 and 1".
    """
    return m_evq_deployed(tau, k=k)


def m_incr_power(r, n=17, low=23, k=K):
    """Increments eps_q proportional to q^r over the transition band.

    eps_q = q^r / sum_i i^r,  q = 1..n;  m_q = sum_{i<=q} eps_i.

    r = 1 IS MrRoPE EXACTLY.  sum_i i = n(n+1)/2, so m_q = q(q+1)/(n(n+1)) and
    `m_incr_power(1)` reproduces `m_mrpro(17)` to the last bit -- checked in
    `m_incr_power_is_anchored` below.  That is what makes the family usable as a
    one-parameter test of MrRoPE rather than another curve: one point of it is a
    known, deployed table, so the question "is the exponent optimal" has a
    pre-declared answer to compare against, and r = 1 is not a special case that
    happens to fall out of the parameterisation.

    WHAT THE PARAMETER DOES.  Raising r makes the increments more back-loaded,
    which moves BOTH of the things the current round of theory attributes
    MrRoPE's advantage to, in the same direction:

        r      m_1       1 - nu_1/omega_1     sum m
        0      0.0588    7.8e-02              32.000
        1      0.00654   9.0e-03              29.333   <- MrRoPE
        2      5.60e-04  7.8e-04              27.886
        4      3.05e-06  4.2e-06              26.437

    i.e. the first transition slot is perturbed less AND the compression is
    concentrated later.  If MrRoPE is optimal these two are already balanced and
    the score is stationary in r; if they are not, the optimum sits away from 1
    and the sign says which mechanism is under- or over-spent.

    ONE CONFOUND, NAMED BECAUSE IT IS THE ONE THAT BIT BEFORE.  Raising r lowers
    sum m (29.333 -> 26.437), and a table with a smaller sum m spends less total
    compression.  R4 says sum m is a free decision variable rather than a
    conserved quantity, so that is not an error -- but a score difference along r
    cannot be attributed to SHAPE until sum m is matched.  `match_sum_m` below
    builds that control; the screen reports both, and the pair is the experiment.
    """
    if r < 0:
        raise ValueError("r < 0 would put the largest increment first")
    q = np.arange(1, int(n) + 1, dtype=np.float64)
    w = np.power(q, float(r))
    mq = np.concatenate([[0.0], np.cumsum(w) / w.sum()])    # m_0..m_n, m_n = 1
    out = np.zeros(k)
    out[low:low + int(n) + 1] = mq
    out[low + int(n) + 1:] = 1.0
    return out


def band_report(m, low=23, n=17, k=K):
    """sum m, the band's span, and the endpoint values -- for the receipt.

    WHY THIS EXISTS RATHER THAN A sum-m-MATCHED CONTROL.  The obvious control for
    a shape family is "same sum m, different shape", and inside a fixed band with
    fixed endpoints THAT TABLE DOES NOT EXIST.  sum m = (k - n - 1) + sum_q m_q
    with m_0 = 0 and m_n = 1, so scaling the ramp to hit a target sum m drives
    m_n past 1 and the table stops being monotone; the rescale that keeps it
    valid is not available.  Varying the shape exponent at fixed band width
    therefore varies sum m with it, and the two are not separately identifiable
    from this family.

    That is a statement about the design, not a defect to paper over, and R4
    makes it less awkward than it sounds: sum m is a FREE DECISION VARIABLE, not
    a conserved quantity, while the conserved one is the band's total span
    (m_n - m_0 = 1, i.e. the compression budget in log-frequency).  The family
    holds the SPAN and the endpoints fixed by construction and lets sum m move,
    which is the trade the plan actually prices.  The receipt carries both
    numbers so a reader can see the coupling instead of being told it is absent.
    """
    m = np.asarray(m, dtype=np.float64)
    return dict(sum_m=float(m.sum()),
                band_span=float(m[low + n] - m[low]),
                m_first=float(m[low]), m_last=float(m[low + n]),
                tail_value=float(m[low + n + 1]) if low + n + 1 < k else None,
                monotone=bool((np.diff(m) >= -1e-15).all()),
                reachable=bool(np.isfinite(m).all()))


def m_incr_split(a, r, n=17, low=23, k=K):
    """eps_1 = a, and eps_q ∝ q^r for q >= 2 -- the TWO mechanisms, separated.

    WHY A SECOND FAMILY EXISTS AT ALL.  `m_incr_power(r)` moves both of the
    structural differences theory attributes to MrRoPE at once: raising r makes
    the first transition slot's perturbation smaller AND concentrates the
    compression later.  A score change along r therefore cannot be assigned to
    either mechanism, and the question being asked -- which feature of the YaRN
    family carries its advantage -- is exactly the question that needs them
    apart.  Here they are separate dials:

        a   the FIRST increment's share of the budget.  It sets the front-edge
            perturbation 1 - nu_1/omega_1 = 1 - S^{-a}, so a is "how much of the
            native operator at the boundary do we disturb".
        r   the tail exponent, i.e. how steeply the remaining budget is spent.
            It sets the shape of eta_q = m_q over the back of the band.

    ANCHORED AGAIN, TO WITHIN ONE ULP RATHER THAN BIT-FOR-BIT.  a = 2/(n(n+1))
    with r = 1 has MrRoPE's increments in exact arithmetic: MrRoPE's
    eps_q = 2q/(n(n+1)), and the normalised form here gives
    (1 - a) q / sum_{i>=2} i = (304/306) q / 152 = 2q/306, the same expression.
    In float64 the extra renormalisation costs one rounding: three of the 64
    entries differ by 2.8e-17 and one frequency differs in its last bit.
    `m_incr_power(1)` IS bit-exact against `m_mrpro(17)` because it needs no
    renormalisation; this one is not, and saying so is cheaper than a special
    case that would make the two families disagree at every other point.  The
    selftest asserts the 1e-15 agreement and RECORDS the ULP gap rather than
    asserting equality it does not have.

    THE SPAN IS HELD AT 1 FOR EVERY (a, r) because the increments are normalised,
    so any difference between two points of this plane is shape, not budget.
    sum m does move (it is 23 + sum_q m_q) and is reported; R4 makes it a free
    decision variable rather than a conserved one.
    """
    a = float(a)
    if not (0.0 < a < 1.0):
        raise ValueError(f"a = {a} is not a share of the budget in (0, 1)")
    q = np.arange(2, int(n) + 1, dtype=np.float64)
    w = np.power(q, float(r))
    rest = (1.0 - a) * w / w.sum()
    eps = np.concatenate([[a], rest])
    mq = np.concatenate([[0.0], np.cumsum(eps)])
    out = np.zeros(k)
    out[low:low + int(n) + 1] = mq
    out[low + int(n) + 1:] = 1.0
    return out


def band_from_turns(alpha, beta, theta, window, head_dim, k=K):
    """The band edges in SLOTS, from the TURN-COUNT window they correspond to.

    YaRN's `find_correction_dim` is usually read as a formula with two tuned
    constants.  It is not: substituting omega_j = theta^(-2j/head_dim) into
    W*omega/(2*pi) = turns gives

        j = head_dim * ln(W / (turns * 2 pi)) / (2 ln theta)

    which is exactly YaRN's expression with turns = alpha for the slow edge and
    turns = beta for the fast edge.  So the band is the set of slots whose
    IN-WINDOW TURN COUNT lies in [alpha, beta], and alpha = 1, beta = 32 -- the
    numbers YaRN tunes by hand -- are turn counts, not slot indices and not
    fractions.

    THIS IS THE MODEL-INDEPENDENT FORM OF THE RULE, and that is the point.  A
    turn count is dimensionless, so the same (alpha, beta) applies to every
    checkpoint, and each one gets a different band because its theta and its
    window differ.  Checked: Qwen2.5-3B (theta 1e6, W 32768) gives [23, 40] and
    OLMo-2-0425-1B (theta 5e5, W 4096) gives [14, 32], which are the bands the
    two deployed tables actually use.

    WHY IT IS AN APPROXIMATION AND NOT AN OPTIMUM.  The marginal cost of
    compressing a slot is small where the slot is unresolved in-window (few
    turns) and large where it is well resolved (many turns); the KKT condition
    for  min L_long s.t. D_N <= eps  puts the boundary where that cost crosses
    the long-range benefit.  A fixed turn window puts it at a fixed place
    instead, so the optimal (alpha, beta) is model-dependent and [1, 32] is a
    constant approximation to a curve.  Whether it is a good approximation is an
    experiment: sweep (alpha, beta) with the SAME values on different models.
    """
    import math as _m
    lo = head_dim * _m.log(window / (float(beta) * 2.0 * _m.pi)) / (2.0 * _m.log(theta))
    hi = head_dim * _m.log(window / (float(alpha) * 2.0 * _m.pi)) / (2.0 * _m.log(theta))
    return int(_m.floor(lo)), int(_m.ceil(hi))


def m_turns(alpha=1.0, beta=32.0, theta=1e6, window=32768, head_dim=128, k=K,
            ramp="mrpro"):
    """A band placed by its turn window, filled with a named ramp.

    `ramp` selects the interior shape: 'mrpro' is eps_k ~ k (the incumbent),
    'beta1' is eps_k ~ k(n+1-k) (the deployed BM table).  The band is derived
    from (alpha, beta, theta, window, head_dim) rather than passed as slots, so
    the SAME call is the same experiment on any checkpoint.
    """
    lo, hi = band_from_turns(alpha, beta, theta, window, head_dim, k=k)
    lo = max(0, lo)
    hi = min(k - 1, hi)
    n = hi - lo
    if n < 1:
        raise ValueError(f"turn window [{alpha}, {beta}] is empty on this config "
                         f"(slots {lo}..{hi})")
    m = np.zeros(k)
    if ramp == "mrpro":
        q = np.arange(1, n + 1, dtype=np.float64)
        eps = q / q.sum()
    elif ramp == "beta1":
        q = np.arange(1, n + 1, dtype=np.float64)
        w = q * (n + 1 - q)
        eps = w / w.sum()
    elif ramp == "linear":
        eps = np.full(n, 1.0 / n)
    else:
        raise KeyError(f"unknown ramp {ramp!r}")
    m[lo] = 0.0
    m[lo + 1:hi + 1] = np.cumsum(eps)
    m[hi + 1:] = 1.0
    return m


def m_incr_beta(b, n=17, low=23, shape_a=1.0, k=K):
    """eps_k proportional to k^a (n+1-k)^b -- THE family whose two ends are the
    two tables the project has actually measured.

    RECOVERED BY INVERTING THE DEPLOYED TABLES, not fitted to a curve.  Reading
    the increments out of the archived counters:

        MrRoPE   eps_k = 2k / (n(n+1))                 -> k^1 (n+1-k)^0   (b = 0)
        MrProBM  eps_k = 6k(n+1-k) / (n(n+1)(n+2))     -> k^1 (n+1-k)^1   (b = 1)

    and both reproduce their deployed tables exactly -- MrRoPE bit-for-bit, and
    BM to the last float32 (checked in `selftest.py`).  So `b` is not a shape
    dial someone invented: it is the single parameter that separates the
    incumbent from the one table that beats it.

    WHY THIS IS THE RIGHT FAMILY TO SWEEP.  MrPro and BM share the band (each
    derived per configuration), the gain, and the factor S; they differ in this
    one exponent.  And they behave differently on the two models: on
    OLMo-2-0425-1B BM reaches 16K RULER 51.32% against MrPro's 2.78% with NLL
    2.86206 against 3.68798, while on Qwen2.5-3B the same pair ties in-window
    (2.08789 vs 2.08899) and BM loses at 128K (70.83% vs 78.13%).  A single
    parameter that separates a large win from a large loss, on tables that both
    exist, is the cleanest thing to sweep that this project has had.

    WHAT b DOES.  b > 0 makes the increments SYMMETRIC about the band's middle:
    the compression starts gently, peaks centrally, and eases off again.  MrRoPE
    (b = 0) is monotone, so it compresses hardest at the SLOW edge of the band.
    Those are different claims about where the transition should be steep, and
    the two models disagree about which is better.

    The span is held at 1 for every b (the increments are normalised), so a
    difference along b is shape and not budget; sum m does move and is reported.
    """
    if b < 0:
        raise ValueError("b < 0 would put the largest increment at the band edge")
    kk = np.arange(1, int(n) + 1, dtype=np.float64)
    w = np.power(kk, float(shape_a)) * np.power(int(n) + 1 - kk, float(b))
    eps = w / w.sum()
    mq = np.concatenate([[0.0], np.cumsum(eps)])
    out = np.zeros(k)
    out[low:low + int(n) + 1] = mq
    out[low + int(n) + 1:] = 1.0
    return out


def m_smoothstep(lo=None, hi=None, k=K, cfg=QWEN25_3B):
    """The smoothstep ramp over a derived band: m(t) = 3t^2 - 2t^3.

    THIS IS THE TABLE THE PROJECT'S ONE SURVIVING POSITIVE RESULT USES.  On
    OLMo-2-0425-1B-Instruct it scores 16K RULER 51.32% against MrRoPE's 2.78%,
    44 wins / 0 losses / 28 ties on 72 rows, and NLL 16K 2.86206 against 3.68798;
    on Qwen2.5-3B it ties MrRoPE in-window (32K NLL 2.08789 vs 2.08899) and
    loses at 128K (70.83% vs 78.13%).  So it is the incumbent to beat on both
    models, and its defining feature is a fact worth keeping separate from the
    band: both tables use the SAME band, derived per configuration, and differ
    only in the ramp's SHAPE -- symmetric here, monotone-increasing for MrRoPE.

    Recovered, not guessed: the deployed Qwen counter's increments correlate
    0.9999999997 with d(3t^2-2t^3)/dt over its band, checked in `selftest.py`.

    `lo`/`hi` default to YaRN's own derived band for the config, which is what
    the deployed table uses on both models -- OLMo's is [14, 32] at
    (head_dim 128, theta 5e5, W 4096) and Qwen's is [23, 40] at
    (128, 1e6, 32768).  Passing them explicitly is how a band-placement arm is
    built; the sweep over that is a different experiment from the ramp sweep.
    """
    if lo is None or hi is None:
        _, _, lo_d, hi_d = yarn_bands(cfg=cfg)
        lo = lo_d if lo is None else lo
        hi = hi_d if hi is None else hi
    lo, hi = int(lo), int(hi)
    if not (0 <= lo < hi <= k):
        raise ValueError(f"band [{lo}, {hi}] is not inside [0, {k}]")
    m = np.zeros(k)
    n = hi - lo
    q = np.arange(n + 1, dtype=np.float64) / n
    m[lo:hi + 1] = 3.0 * q ** 2 - 2.0 * q ** 3
    # THE TAIL IS PART OF THE TABLE.  Leaving m = 0 below the band is the
    # natural thing to write and it is wrong: the deployed table's compressed
    # half carries m = 1, and a version without it differs by a factor of S in
    # nu on those slots -- which is exactly the factor of 4 that the
    # reproduction check caught.  A band-only table is a different object, not a
    # near-miss, and it would have been scored as if it were the incumbent.
    m[hi + 1:] = 1.0
    return m


def m_flat_gap_donor(source, theta, k=K):
    """Move one uniform native log-gap's worth of budget, as the harness does.

    Kept so tables produced here can be matched against the archived
    HighGapToLong / HighGapToMid arms in ground_truth_tables.json.
    """
    gap = math.log(theta) / k
    return np.asarray(source, dtype=np.float64) + gap / LN_S


CONSTRUCTIONS = {
    "native": m_native,
    "interp": m_interp,
    "mrpro_n17": lambda: m_mrpro(17),
    "mrpro_n16": lambda: m_mrpro(16),
    "mrpro_n15": lambda: m_mrpro(15),
    "mrpro_n13": lambda: m_mrpro(13),
    "yarn_lin": m_yarn,
    "evq_shift_t1": lambda: m_evq_shift(1.0),
    "evq_shift_t2": lambda: m_evq_shift(2.0),
    "evq_deploy_t1": lambda: m_evq_deployed(1.0),
    "evq_deploy_t2": lambda: m_evq_deployed(2.0),
    "evq_deploy_t4": lambda: m_evq_deployed(4.0),
    "beta_b0": lambda: m_incr_beta(0.0),
    "beta_b0p25": lambda: m_incr_beta(0.25),
    "beta_b0p5": lambda: m_incr_beta(0.5),
    "beta_b1": lambda: m_incr_beta(1.0),
    "beta_b2": lambda: m_incr_beta(2.0),
    "smoothstep_qwen": lambda: m_smoothstep(),
    "smoothstep_olmo": lambda: m_smoothstep(15, 32),
    "power_p2": lambda: m_power_shift(2),
    "power_p4": lambda: m_power_shift(4),
    "incr_r0": lambda: m_incr_power(0.0),
    "incr_r0p5": lambda: m_incr_power(0.5),
    "incr_r1": lambda: m_incr_power(1.0),
    "incr_r1p5": lambda: m_incr_power(1.5),
    "incr_r2": lambda: m_incr_power(2.0),
    "incr_r3": lambda: m_incr_power(3.0),
    "incr_r4": lambda: m_incr_power(4.0),
    "split_a_p1": lambda: m_incr_split(2.0 / 306.0, 1.0),
    "split_a_p01": lambda: m_incr_split(0.001, 1.0),
    "split_a_p03": lambda: m_incr_split(0.03, 1.0),
    "split_a_p10": lambda: m_incr_split(0.10, 1.0),
    "split_ar_p03_r2": lambda: m_incr_split(0.03, 2.0),
    "split_ar_p001_r2": lambda: m_incr_split(0.001, 2.0),
    "split_ar_p03_r0": lambda: m_incr_split(0.03, 0.0),
}


def build(name, gain=GAIN_YARN, cfg=QWEN25_3B):
    """-> dict(values_float32[64], gain, m[64], name, theta) ready for install."""
    if name in CONSTRUCTIONS:
        m = np.asarray(CONSTRUCTIONS[name](), dtype=np.float64)
    else:
        raise KeyError(f"unknown table {name!r}; have {sorted(CONSTRUCTIONS)}")
    if m.shape != (K,):
        raise ValueError(f"{name}: expected {K} entries, got {m.shape}")
    return dict(name=name, m=m, gain=float(gain), theta=float(cfg["theta"]),
                values_float32=m_to_inv_freq(m, cfg["theta"]))


def from_eps(d_eps, base_name="mrpro_n17", gain=GAIN_YARN, cfg=QWEN25_3B, k=K):
    """Apply a solver step to a base table.

    The solver works in eps_j = ln(omega_j / nu_j), so a step d gives
    nu_j -> nu_j * exp(-d_j).  Scaling the step by alpha is exactly the trust
    sweep forward_check.py runs, and alpha = 0 recovers the base table bitwise.
    """
    base = build(base_name, gain=gain, cfg=cfg)
    d = np.asarray(d_eps, dtype=np.float64)
    if d.shape != (k,):
        raise ValueError(f"expected {k} step entries, got {d.shape}")
    if not np.isfinite(d).all():
        raise ValueError("non-finite step")
    nu = base["values_float32"].astype(np.float64) * np.exp(-d)
    if not (nu > 0).all() or not np.isfinite(nu).all():
        raise ValueError("step leaves the positive-frequency cone")
    return dict(name=f"{base_name}+step", m=base["m"] + d / LN_S, gain=float(gain),
                theta=float(cfg["theta"]), values_float32=nu.astype(np.float32))


def verify(gt_path, tol=1e-6):
    """Pre-flight gate: does this algebra reproduce the DEPLOYED tables?

    Reads the ground-truth ledger and compares three tables that are known
    exactly -- Native (the checkpoint's own grid), MrPro (the table actually
    installed on the GPU), YaRN_linear_official (ditto).  A mismatch here means
    the m-coordinate or a construction is wrong and nothing downstream is worth
    running; it costs no GPU time and needs only numpy, so it is the first thing
    driver.sh executes.

    MrPro is the load-bearing one: it is the base table for the whole package,
    and its 29.3333 total compression is what the step is measured against.
    """
    import json

    gt = json.load(open(gt_path))["methods"]
    out = {}

    def cmp(label, nu_mine, nu_gt):
        a = np.asarray(nu_mine, dtype=np.float64)
        b = np.asarray(nu_gt, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError(f"{label}: shape {a.shape} vs ground truth {b.shape}")
        out[label] = float(np.abs(a / b - 1.0).max())

    cmp("native", native_inv_freq(QWEN25_3B["theta"]), gt["Native"]["nu_j"])
    cmp("mrpro_n17", build("mrpro_n17")["values_float32"], gt["MrPro"]["nu_j"])
    cmp("yarn_lin", build("yarn_lin")["values_float32"], gt["YaRN_linear_official"]["nu_j"])

    bad = {k: v for k, v in out.items() if v > tol}
    if bad:
        raise ValueError(f"reconstruction does not match the deployed tables: {bad} "
                         f"(tolerance {tol:g}); the m-coordinate or a construction is wrong")
    return out


def describe(nu_or_m, as_m=True, theta=QWEN25_3B["theta"]):
    """Compact human summary used in receipts and logs."""
    m = np.asarray(nu_or_m, dtype=np.float64) if as_m else inv_freq_to_m(nu_or_m, theta)
    d = np.diff(m)
    return dict(m_sum=float(m.sum()), m_min=float(m.min()), m_max=float(m.max()),
                m_at_0=float(m[0]), m_at_63=float(m[-1]),
                gap_min=float(d.min()), gap_max=float(d.max()),
                on_high=float((m[:24] == 0).mean()), at_full=float((m[24:] == 1).mean()),
                monotone=bool(np.all(d >= -1e-12)))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        gt = sys.argv[2] if len(sys.argv) > 2 else \
            "analysis/unify_20260910/tables/ground_truth_tables.json"
        print(json.dumps(verify(gt), indent=1))
        raise SystemExit(0)

    hdr = (f"{'table':16s} {'sum m':>8s} {'m_min':>7s} {'m_max':>7s} {'m0':>6s} {'m63':>6s} "
           f"{'min gap':>9s} {'max gap':>9s}")
    print(hdr)
    for name in CONSTRUCTIONS:
        d = describe(build(name)["m"])
        print(f"{name:16s} {d['m_sum']:8.4f} {d['m_min']:7.3f} {d['m_max']:7.3f} "
              f"{d['m_at_0']:6.3f} {d['m_at_63']:6.3f} "
              f"{d['gap_min']:9.5f} {d['gap_max']:9.5f}")
