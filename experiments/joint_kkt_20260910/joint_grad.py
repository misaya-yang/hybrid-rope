"""Real model derivatives for the 65 shared design variables, in one backward.

FINAL_PLAN.md sec.8 is a requirement, not a suggestion:

    "每个完整loss backward可一次给出所有65个共享设计变量的梯度。权重
     requires_grad=False，但模型前向不能no_grad；频率缓存构造也不能detach。"

Three separate constraints, and each one is easy to violate in a way that still
produces a number:

  * ONE backward for all 65.  `curvature_20260910/model.py` does this for the 64
    frequencies (`grad_wrt_freq`) but the gain was a Python float, so 65 was not
    reachable.  Differencing 65 directions one at a time would cost 65 forwards.
  * Weights frozen, forward NOT in `no_grad`.  `requires_grad_(False)` on the
    parameters is what keeps the backward from computing 3B weight gradients; it
    is not a licence to run the forward under `torch.no_grad()`, which would kill
    the graph before the frequencies ever see it.
  * The frequency cache must not be detached.  `cos`/`sin` are built from
    `inv_freq @ position_ids`; a `.detach()` anywhere on that path -- including on
    the intermediate `freqs` -- leaves the design variables as dead leaves and
    autograd returns zero, silently.  The patch below is written to match the
    transformers 5.15 arithmetic exactly apart from being differentiable, so the
    two can be cross-checked rather than trusted.

THE CHAIN RULE, AND THE ONE DIRECTION THIS COORDINATE CANNOT REACH.  The design
variable is not (nu, g) but the plan's (x, a):  nu_j = exp(-x_j) and
g = exp(a/2).  Autograd returns dL/dnu and dL/dg, so

    dL/dx_j = -nu_j * dL/dnu_j        (because d nu_j/dx_j = -nu_j)
    dL/da   = (g/2) * dL/dg           (because dg/da = g/2)

This is the step where a sign error is invisible -- a wrong sign here still
yields a plausible-looking descent direction, and the acceptance check in
accept.py would reject the resulting step without saying why.  `selftest.py`
checks the two factors against finite differences of the real forward on a tiny
model, which is the only place the whole chain is verified end to end.

Two corrections to what this docstring used to claim, both of them the kind of
statement that is only wrong at an endpoint:

  * the frequency factor is `-nu`, which is NEGATIVE for every finite x; only
    the gain factor `g/2` is positive.  The magnitudes are what matter for the
    "no direction is annihilated" reading, but a reader checking signs against
    this sentence would find it contradicted by the code two lines down.
  * `nu > 0` for every finite x, so the factor never vanishes at a REACHABLE
    point -- but that is not the same as the coordinate being able to express
    every deployment choice.  nu -> 0 only as x -> inf (exp(-745) is already at
    the float64 denormal floor), so **NoPE -- a slot with no rotation at all --
    is outside this parameterization**, and the frequency gradient tends to zero
    exactly as a slot approaches it.  Anything that wants to decide "how much
    resource does not encode explicit position" (which the current round of
    theory puts on equal footing with the spacing question) cannot use x.

    The project already derived and implemented the coordinate that can:
    additive, relative to native,  nu = nu_ref + omega_native * delta
    -- `docs/research/ROPE_RESEARCH_FAILURE_REVIEW_20260907.md:180`, which gives
    the reason verbatim ("nu_ref = 0 时, nu_ref exp(alpha) 对 alpha 的导数恒为 0,
    无法重新引入旋转; 实际 d y / d nu 却可以非零"), implemented at
    `scripts/analysis/shared_frequency_response.py:17` and exercised with a
    zero-frequency slot in its CPU check.  This package never adopted it.  It has
    not bitten because the operating point sits far from that end (x in roughly
    [0.1, 15] against underflow at x >~ 745), and it will bite the moment a
    no-rotation arm enters the design.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import design as D

K = D.K
A_INDEX = D.A_INDEX


# ---------------------------------------------------------------------------
# Grad-capable objectives.
#
# These exist because `FrozenRoPE.nll`, `.nll_per_token` and `.log_probs` wrap
# their forwards in `torch.no_grad()`.  That is correct for the probe path -- it
# is what makes a 128K sweep affordable -- and fatal here: a loss taken through
# them is a constant with no graph, autograd returns None for the design leaves,
# and the failure surfaces as a zero gradient rather than an exception.  Every
# objective used for a derivative must come from this file.
# ---------------------------------------------------------------------------
def nll_tensor(frozen, ids, keep, want_grad=True):
    """Mean next-token NLL over the last `keep` targets, ON the graph."""
    lg = frozen.logits(ids, keep + 1, want_grad=want_grad)
    return torch.nn.functional.cross_entropy(lg[:-1], ids[0, -keep:], reduction="mean")


def nll_per_token_tensor(frozen, ids, keep, want_grad=True):
    lg = frozen.logits(ids, keep + 1, want_grad=want_grad)
    return torch.nn.functional.cross_entropy(lg[:-1], ids[0, -keep:], reduction="none")


def logp_tensor(frozen, ids, keep, want_grad=True):
    """Log-softmax in float32, ON the graph -- the primitive the sec.3 bound and
    the output-KL native metric are both assembled from."""
    return torch.log_softmax(frozen.logits(ids, keep, want_grad=want_grad), dim=-1)


def nll_value(frozen, ids, keep):
    """Forward-only twin, for the acceptance check and every non-derivative use."""
    with torch.no_grad():
        lg = frozen.logits(ids, keep + 1, want_grad=False)
    return torch.nn.functional.cross_entropy(lg[:-1], ids[0, -keep:], reduction="mean")


class JointDesign:
    """A `FrozenRoPE` whose 64 frequencies AND gain are differentiable leaves.

    Usage:

        jd = JointDesign(frozen)              # wraps, does not copy weights
        y = D.pack(nu, gain)
        out = jd.value_and_grad([(1.0, native_nll), (1.0, long_risk)])
        # out["grad"] is 65 numbers: dL/dx[64] and dL/da

    The wrapper owns the install/restore protocol because leaving a tensor
    `attention_scaling` behind would corrupt the forward-only probe path in
    `curvature_20260910`, which expects a float there.  `restore()` puts the
    module back the way it was found.
    """

    def __init__(self, frozen, checkpointing=False):
        self.f = frozen
        self.f.enable_grad_path()            # grad-capable twin of the 5.15 forward
        self._inv = None
        self._gain = None
        self._y = None
        self._checkpointing = False
        if checkpointing:
            self.enable_checkpointing()

    # -- checkpointing ------------------------------------------------------
    def enable_checkpointing(self):
        """Needed for the long side: a 131072-token backward stores activations
        for 36 layers, which does not fit.  `enable_input_require_grads` is the
        usual companion -- checkpointed segments are not differentiable unless
        their input carries grad, and our input is token ids, which never do."""
        m = self.f.model
        m.gradient_checkpointing_enable()
        if hasattr(m, "enable_input_require_grads"):
            m.enable_input_require_grads()
        self._checkpointing = True

    def disable_checkpointing(self):
        m = self.f.model
        if hasattr(m, "gradient_checkpointing_disable"):
            m.gradient_checkpointing_disable()
        self._checkpointing = False

    # -- install / restore --------------------------------------------------
    def install(self, y):
        """Set the design point.  Leaves are float32 on the model's device.

        The values are the model's own dtype-independent quantities: the patched
        forward casts `inv_freq` to float before the matmul (`inv.float()`), so
        keeping the leaves float32 costs nothing at long length and is what makes
        the 65 gradients comparable in magnitude -- in bf16 the smallest of them
        sit under the format's own resolution and would read as exact zeros.
        """
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (D.N_DESIGN,):
            raise ValueError(f"expected {D.N_DESIGN} design entries, got {y.shape}")
        if not np.isfinite(y).all():
            raise ValueError("non-finite design point")
        nu, g = D.unpack(y)
        dev = self.f.device
        self._inv = torch.tensor(nu, dtype=torch.float32, device=dev, requires_grad=True)
        self._gain = torch.tensor(g, dtype=torch.float32, device=dev, requires_grad=True)
        self.f.rotary.inv_freq = self._inv
        self.f.rotary.original_inv_freq = self._inv.detach().clone()
        self.f.rotary.attention_scaling = self._gain
        self._y = y
        return self

    def restore(self):
        """Hand the module back in the state the probe path expects: float gain,
        plain tensor frequencies, no graph."""
        nu = self.f.native_inv_freq if self._inv is None else \
            self._inv.detach().float().cpu().numpy()
        self.f.rotary.inv_freq = torch.tensor(nu, dtype=torch.float32,
                                              device=self.f.device)
        self.f.rotary.original_inv_freq = self.f.rotary.inv_freq.clone()
        self.f.rotary.attention_scaling = 1.0
        self._inv = self._gain = None
        return self

    @property
    def y(self):
        return None if self._y is None else self._y.copy()

    # -- the one backward ---------------------------------------------------
    def value_and_grad(self, terms, retain_graph=False):
        """terms: iterable of (weight, fn) with fn() -> scalar tensor.

        All forwards happen inside one graph and one backward returns all 65
        design gradients.  `fn` is called here rather than passed in
        pre-computed so that a term cannot accidentally be built from a forward
        taken outside the graph -- which is the failure mode that yields zeros.

        Returns dict(value, grad[65], raw_grad, per_term, y, factors).
        """
        if self._inv is None:
            raise RuntimeError("install() first")
        terms = list(terms)
        if not terms:
            raise ValueError("no terms")
        total = None
        per_term = []
        for w, fn in terms:
            t = fn()
            if not torch.is_tensor(t) or not t.requires_grad:
                raise RuntimeError(
                    "a term is not connected to the graph -- the forward was "
                    "taken under no_grad, or the frequency path was detached")
            per_term.append((float(w), float(t.detach())))
            total = t * float(w) if total is None else total + t * float(w)
        g_inv, g_gain = torch.autograd.grad(total, [self._inv, self._gain],
                                            retain_graph=retain_graph,
                                            allow_unused=False)
        nu, g = D.unpack(self._y)
        # chain rule to the plan's coordinates, sec.2
        grad = np.empty(D.N_DESIGN)
        grad[:K] = -nu * g_inv.detach().double().cpu().numpy()
        grad[A_INDEX] = 0.5 * g * float(g_gain.detach().double().cpu())
        return dict(value=float(total.detach()), grad=grad,
                    raw_grad=dict(nu=g_inv.detach().double().cpu().numpy(),
                                  gain=float(g_gain.detach().double().cpu())),
                    per_term=per_term, y=self._y.copy(),
                    factors=dict(minus_nu=(-nu).copy(), half_g=0.5 * g),
                    dtype=str(self.f.dtype), checkpointing=self._checkpointing)

    def per_group_value_and_grad(self, group_fn):
        """Per-group values AND per-group 65-dim gradients, from ONE forward.

        `value_and_grad` returns the gradient of the weighted SUM, which is what
        a step needs and is not what the quadratic model needs: sec.8 assembles
        its model from one (B_i, grad_i) pair per risk group, and a summed
        gradient cannot be split back apart.

        THE CONTRACT IS THE SHARED FORWARD, AND IT IS THE CALLER'S TO KEEP.
        `group_fn()` returns an ordered mapping name -> scalar tensor, and every
        entry must be a reduction of ONE batched forward.  That matters more than
        it looks: the panel is 6 groups of 5 rows, so G separate forwards is 6
        prefills per gradient evaluation and 18 per accepted step once the
        acceptance check's two extra evaluations are counted.  At the recorded
        ~33.9 s per 128K row that is hours per step, against minutes for one
        batched forward.  This method cannot enforce the contract -- two
        reductions of two forward passes look exactly like two reductions of one
        -- so the driver states it and `loop.py` records `n_forwards` as a
        claimed, not a measured, number.

        The gradients are G backwards over that one graph via `autograd.grad` per
        group with retain_graph=True; the chain rule to the plan's (x, a)
        coordinates is applied here, in the same two lines as in
        `value_and_grad`, so the entry points cannot drift apart.
        """
        if self._inv is None:
            raise RuntimeError("install() first")
        groups = group_fn()
        if not groups:
            raise ValueError("group_fn returned nothing")
        names = list(groups.keys())
        tensors = [groups[k] for k in names]
        for k, t in zip(names, tensors):
            if not torch.is_tensor(t) or not t.requires_grad:
                raise RuntimeError(
                    f"group {k!r} is not connected to the graph -- its forward was "
                    "taken under no_grad, or the frequency path was detached")
        nu, g = D.unpack(self._y)
        grads = np.empty((len(tensors), D.N_DESIGN))
        for i, t in enumerate(tensors):
            gi, gg = torch.autograd.grad(
                t, [self._inv, self._gain],
                retain_graph=bool(i < len(tensors) - 1), allow_unused=False)
            grads[i, :K] = -nu * gi.detach().double().cpu().numpy()
            grads[i, A_INDEX] = 0.5 * g * float(gg.detach().double().cpu())
        return dict(names=names, grads=grads,
                    values=[float(t.detach()) for t in tensors],
                    n_groups=len(names), factors=dict(minus_nu=(-nu).copy(),
                                                      half_g=0.5 * g))

    # -- forward-only values, for the acceptance check ----------------------
    @torch.no_grad()
    def values(self, terms):
        """The same terms, forward only.  Used by sec.8's acceptance check, which
        compares the REAL values after the step against the model's prediction,
        and by the loop to evaluate a candidate without building a graph."""
        out = []
        for w, fn in terms:
            out.append((float(w), float(fn())))
        return out

    def make_installer(self):
        """A closure the term functions call before their forward.

        Terms are written as `lambda: nll_at(y)` rather than capturing a table, so
        that a term can be evaluated at ANY design point inside one graph -- the
        acceptance check needs exactly that, and a term that captured its table
        would silently re-measure the old point.
        """
        def at(y):
            self.install(y)
            return self
        return at


# ---------------------------------------------------------------------------
def finite_difference_check(frozen, ids, keep, y0, step=1e-3, slots=(0, 31, 63),
                            gain=True, rel_tol=5e-2):
    """Verify the 65-dim gradient against the real forward, on CPU, on a tiny
    input.  This is the only check that exercises the whole chain -- patch,
    autograd, and the two chain-rule factors -- rather than a piece of it, so it
    is the gate that must pass before any card is touched.

    Returns per-coordinate (analytic, numeric, relative error).  A sign error in
    the chain rule shows up here as a large relative error with the right
    magnitude, which is exactly the signature a magnitude-only check would miss.
    """
    jd = JointDesign(frozen)
    jd.install(y0)
    res = jd.value_and_grad([(1.0, lambda: nll_tensor(frozen, ids, keep))])
    ana = res["grad"]

    num = {}
    for j in list(slots) + ([A_INDEX] if gain else []):
        row = {}
        for sign in (+1.0, -1.0):
            y = y0.copy()
            y[j] += sign * step
            with torch.no_grad():
                jd.install(y)
                row[sign] = float(nll_value(frozen, ids, keep))
        num[j] = (row[+1.0] - row[-1.0]) / (2.0 * step)

    out = {}
    for j, v in num.items():
        a = float(ana[j])
        rel = abs(a - v) / max(abs(v), 1e-12)
        out[int(j)] = dict(analytic=a, numeric=v, rel=rel,
                           ok=bool(rel < rel_tol))
    jd.restore()
    return dict(per_coord=out, ok=bool(all(d["ok"] for d in out.values())),
                step=step, rel_tol=rel_tol,
                note="central difference on the real forward; step is in x-units")
