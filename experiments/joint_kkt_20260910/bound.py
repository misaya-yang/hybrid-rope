"""The sec.3 complete-output error bound, and the sec.4 certificate it licenses.

WHY A BOUND AND NOT A SCORE.  The quantity this package can actually measure on a
long input is a teacher-forced logit margin; the quantity the paper is about is a
task score under greedy generation.  sec.3 is the bridge, and it is a one-sided
one:

    b_e(theta) = max_{t, v != y_t} [ z_{etv}(theta) - z_{et, y_t}(theta) ]

    1{greedy output != reference}  <=  softplus(b_e) / log 2                (*)

The argument is an induction on the reference prefix, and it is why the bound is
valid rather than fitted: before the first divergence the model's greedy prefix IS
the reference prefix, so the first disagreement is decided by exactly the logits
(*) maximises over.  The plan is careful that this is a per-instance bound and
NOT the claim "a lower bound makes every sample score higher" -- reference
diversity makes it conservative, and it is not a margin normalised between two
answers, nor a single head's margin.

THE SMOOTHING, AND WHY eta IS NOT A TUNING KNOB.  A max is not differentiable at
its own maximiser, so the objective needs a smooth surrogate that stays an upper
bound.  With N_e = T_e (V - 1) terms under the max,

    b_tilde = tau_e * logsumexp((wrong - right) / tau_e),   tau_e = eta / log(N_e)

satisfies  b <= b_tilde <= b + eta  exactly, because logsumexp exceeds the max by
at most tau*log(N).  So eta is a declared numerical smoothing allowance, and a
receipt that reports b_tilde must say eta.  Choosing eta after seeing a result
would silently convert the certificate into a fit; sec.3 fixes the default at
0.02 logits and `selftest.py` checks the sandwich numerically.

THE BOUND IS ALLOWED TO BE VACUOUS, AND THAT IS A REAL RISK, NOT A FOOTNOTE.
softplus(b)/log 2 exceeds 1 for b > 0, so on a row where some wrong token
outranks the reference token the bound says nothing beyond the trivial
"<= 1".  For the bound to be informative at the 0.1 level it needs b < -2.63
nats on EVERY answer position.  Whether a frozen instruct model clears that on a
7-digit needle is an empirical question, and it is the FIRST thing worth
measuring because if it does not, the sec.3 objective carries no signal and the
solve must fall back to the sec.4 native-NLL formulation.  `bound_summary`
reports the vacuum rate for exactly this purpose.
"""
from __future__ import annotations

import math

import numpy as np
import torch

LOG2 = math.log(2.0)
# sec.3's declared smoothing allowance, in logits.  Not a free parameter.
ETA_DEFAULT = 0.02


def n_terms(t_e, vocab):
    """N_e = T_e (V - 1): the number of (position, wrong token) pairs under the
    max.  Used only through log(N_e), so the -1 is kept for exactness rather than
    absorbed into the vocab size."""
    return max(int(t_e) * (int(vocab) - 1), 1)


def tau_for(t_e, vocab, eta=ETA_DEFAULT):
    """tau_e = eta / log(N_e).  log(N_e) = 0 only when N_e = 1, i.e. one position
    and one wrong token, where the plan says take the max directly; that case is
    signalled with tau = inf by the caller, so this returns the finite fallback."""
    n = n_terms(t_e, vocab)
    if n <= 1:
        return None
    return float(eta) / math.log(n)


def raw_bound(logits, answer_ids, mask=None):
    """b_e from a (T_e, V) logits block, unsmoothed.

    `logits[i]` must be the distribution that PREDICTS `answer_ids[i]`, i.e. the
    caller prefills prompt + answer and keeps len(answer) positions.  `mask`
    optionally removes positions (e.g. padding) from the max.

    Returns a dict with b, the position that attains it, and the runner-up gap.
    """
    z = logits if torch.is_tensor(logits) else torch.as_tensor(logits)
    if z.dim() != 2:
        raise ValueError(f"expected (T_e, V) logits, got {tuple(z.shape)}")
    y = torch.as_tensor(answer_ids, dtype=torch.long, device=z.device)
    if y.shape[0] != z.shape[0]:
        raise ValueError(f"{z.shape[0]} logit rows for {y.shape[0]} answer tokens")
    t_e, vocab = z.shape
    right = z.gather(1, y[:, None])                       # (T_e, 1)
    wrong = z.clone()
    wrong.scatter_(1, y[:, None], float("-inf"))          # exclude v = y_t exactly
    diff = wrong - right                                  # (T_e, V)
    if mask is not None:
        m = torch.as_tensor(mask, dtype=torch.bool, device=z.device)
        diff = diff[m]
    flat = diff.reshape(-1)
    b, idx = torch.max(flat, dim=0)
    pos = int(idx) // vocab
    top2 = torch.topk(flat, k=min(2, flat.numel()))
    return dict(b=float(b.detach()), pos=pos,
                top2=[float(v.detach()) for v in top2.values],
                t_e=int(t_e), vocab=int(vocab), n_terms=n_terms(t_e, vocab))


def smoothed_bound(logits, answer_ids, eta=ETA_DEFAULT, mask=None):
    """b_tilde, differentiable in `logits`, with the sandwich guaranteed.

    b <= b_tilde <= b + eta.  Returned alongside the raw b so the receipt can
    show both and the gap can be checked against eta rather than trusted.
    """
    z = logits if torch.is_tensor(logits) else torch.as_tensor(logits)
    y = torch.as_tensor(answer_ids, dtype=torch.long, device=z.device)
    t_e, vocab = z.shape
    tau = tau_for(t_e, vocab, eta)
    raw = raw_bound(z, y, mask=mask)
    if tau is None:
        return dict(b_tilde=raw["b"], b=raw["b"], tau=None, eta=float(eta),
                    gap=0.0, sandwich_ok=True, **{k: raw[k] for k in
                                                  ("pos", "t_e", "vocab", "n_terms")})
    right = z.gather(1, y[:, None])
    wrong = z.clone()
    wrong.scatter_(1, y[:, None], float("-inf"))
    diff = (wrong - right) / tau
    if mask is not None:
        m = torch.as_tensor(mask, dtype=torch.bool, device=z.device)
        diff = diff[m]
    flat = diff.reshape(-1)
    m = flat.max()
    b_tilde = tau * (m + torch.log(torch.exp(flat - m).sum()))
    return dict(b_tilde=b_tilde, b=raw["b"], tau=float(tau), eta=float(eta),
                gap=float(b_tilde) - raw["b"],
                sandwich_ok=bool(float(b_tilde) >= raw["b"] - 1e-9
                                 and float(b_tilde) <= raw["b"] + float(eta) + 1e-6),
                pos=raw["pos"], t_e=raw["t_e"], vocab=raw["vocab"],
                n_terms=raw["n_terms"])


def bound_loss(logits, answer_ids, eta=ETA_DEFAULT, mask=None):
    """softplus(b_tilde)/log 2 -- the per-instance upper bound on
    1{greedy != reference}, as a differentiable tensor."""
    s = smoothed_bound(logits, answer_ids, eta=eta, mask=mask)
    bt = s["b_tilde"]
    bt = bt if torch.is_tensor(bt) else torch.tensor(float(bt), requires_grad=logits.requires_grad
                                                     if torch.is_tensor(logits) else False)
    return torch.nn.functional.softplus(bt) / LOG2, s


def bound_is_informative(b, target=0.1):
    """Is softplus(b)/log2 below `target`?  b < log(exp(target*log2) - 1)."""
    thr = math.log(math.expm1(target * LOG2))
    return bool(b < thr), thr


# ---------------------------------------------------------------------------
# sec.4 -- the native-retention certificate
# ---------------------------------------------------------------------------
def gamma_from_native(b_native):
    """gamma_e = -b_e(theta_native), defined only where the baseline is correct.

    The plan requires that rows with b_e(native) >= 0 are NOT included with a
    floored denominator: a baseline that does not clear its own reference has no
    margin to certify, and dividing by a floor would produce a certificate that
    looks the same as a real one.  Those rows are separated here.
    """
    b = np.asarray(b_native, dtype=np.float64)
    ok = b < 0
    return dict(gamma=np.where(ok, -b, np.nan), certified=ok,
                n_certified=int(ok.sum()), n_total=int(b.size),
                ties=[int(j) for j in np.flatnonzero(np.abs(b) <= 1e-12)])


def d_keep(b_theta, b_native, weights=None):
    """D_keep(theta) = mean over certified rows of [1 + b_e(theta)/gamma_e]_+.

    Valid on the same data and the same deterministic execution map as the
    baseline, where it bounds the drop in native accuracy; D_keep(native) = 0 by
    construction.  Non-smooth positive part -- the epigraph slack in qcqp/loop is
    what keeps it out of a C2 Hessian, per sec.8.
    """
    g = gamma_from_native(b_native)
    ok = g["certified"]
    if not ok.any():
        return dict(d_keep=float("nan"), n_certified=0, undefined=True)
    b = np.asarray(b_theta, dtype=np.float64)[ok]
    gam = g["gamma"][ok]
    val = np.maximum(0.0, 1.0 + b / gam)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)[ok]
        dk = float((val * w).sum() / w.sum())
    else:
        dk = float(val.mean())
    return dict(d_keep=dk, n_certified=int(ok.sum()),
                n_undefined=int((~ok).sum()), undefined=False,
                per_row=[float(v) for v in val],
                note="bounds the native accuracy drop on THIS sample; not a "
                     "population guarantee")


def bound_summary(bs, etas=None):
    """The pre-registered gate on whether the sec.3 objective carries signal.

    Reports the vacuum rate (b >= 0, where the bound says nothing beyond <= 1) and
    the informative rate, plus the threshold b must clear for a 0.1 bound.  This
    is cheap to compute and it decides whether the long-range side of the solve
    can use sec.3 at all, so it is the first measurement on any new corpus.
    """
    b = np.asarray(bs, dtype=np.float64)
    _, thr = bound_is_informative(0.0)
    vac = b >= 0.0
    return dict(n=int(b.size), median=float(np.median(b)), min=float(b.min()),
                max=float(b.max()),
                vacuum_rate=float(vac.mean()),
                informative_0p1=float((b < thr).mean()),
                threshold_for_0p1=float(thr),
                bound_median=float(np.mean(np.log1p(np.exp(np.clip(b, -700, 700)))) / LOG2),
                note="b >= 0 means the max wrong token outranks the reference; "
                     "the bound is then vacuous at 1.0")
